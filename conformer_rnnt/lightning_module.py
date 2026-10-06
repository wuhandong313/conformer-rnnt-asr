"""LibriSpeech dataset with phoneme transcription support.

Provides the LIBRISPEECH dataset class (reading phoneme .trans.txt files),
a CustomDataset wrapper for token-count-based batching, and the LibriSpeech
LightningModule with data pipelines, training loop, and beam-search decoding.
"""

import os
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar, _load_waveform
from torchaudio.models import RNNTBeamSearch

from common import (
    Batch,
    batch_by_token_count,
    FunctionalModule,
    GlobalStatsNormalization,
    piecewise_linear_log,
    post_process_hypos,
    WarmupLR,
)
from conformer_rnnt.model import conformer_rnnt_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"
SAMPLE_RATE = 16000
_DATA_SUBSETS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]
_CHECKSUMS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz": (
        "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
    ),
    "http://www.openslr.org/resources/12/dev-other.tar.gz": (
        "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365"
    ),
    "http://www.openslr.org/resources/12/test-clean.tar.gz": (
        "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23"
    ),
    "http://www.openslr.org/resources/12/test-other.tar.gz": (
        "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29"
    ),
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz": (
        "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2"
    ),
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz": (
        "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf"
    ),
    "http://www.openslr.org/resources/12/train-other-500.tar.gz": (
        "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2"
    ),
}


# ---------------------------------------------------------------------------
# Helper: download LibriSpeech
# ---------------------------------------------------------------------------


def _download_librispeech(root: str, url: str) -> None:
    base_url = "http://www.openslr.org/resources/12/"
    ext_archive = ".tar.gz"
    filename = url + ext_archive
    archive = os.path.join(root, filename)
    download_url = os.path.join(base_url, filename)
    if not os.path.isfile(archive):
        checksum = _CHECKSUMS.get(download_url, None)
        download_url_to_file(download_url, archive, hash_prefix=checksum)
    _extract_tar(archive)


def _get_librispeech_metadata(
    fileid: str,
    root: str,
    folder: str,
    ext_audio: str,
    ext_txt: str,
    transcript_suffix: str = "-ph",
) -> Tuple[str, int, str, int, int, int]:
    speaker_id, chapter_id, utterance_id = fileid.split("-")
    fileid_audio = f"{speaker_id}-{chapter_id}-{utterance_id}"
    filepath = os.path.join(folder, speaker_id, chapter_id, f"{fileid_audio}{ext_audio}")

    file_text = f"{speaker_id}-{chapter_id}{transcript_suffix}{ext_txt}"
    file_text = os.path.join(root, folder, speaker_id, chapter_id, file_text)
    with open(file_text) as ft:
        for line in ft:
            fileid_text, transcript = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            raise FileNotFoundError(f"Translation not found for {fileid_audio}")

    return (
        filepath,
        SAMPLE_RATE,
        transcript,
        int(speaker_id),
        int(chapter_id),
        int(utterance_id),
    )


# ---------------------------------------------------------------------------
# LibriSpeech dataset (phoneme transcription variant)
# ---------------------------------------------------------------------------


class LIBRISPEECH(Dataset):
    """LibriSpeech dataset with phoneme-level transcriptions.

    Expects ``.flac`` audio files and ``.trans.txt`` phoneme transcription files
    in the standard LibriSpeech directory layout.

    Args:
        root: Path to the dataset root directory.
        url: Dataset subset name, e.g. ``"train-clean-100"``.
        folder_in_archive: Top-level directory name inside the archive.
        download: If ``True``, download the dataset when not found locally.
    """

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        transcript_suffix: str = "-ph",
    ) -> None:
        self._url = url
        self._transcript_suffix = transcript_suffix
        if url not in _DATA_SUBSETS:
            raise ValueError(
                f"Invalid url '{url}'; please provide one of {_DATA_SUBSETS}."
            )

        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)
        self._path = os.path.join(root, folder_in_archive, url)

        if not os.path.isdir(self._path):
            if download:
                _download_librispeech(root, url)
            else:
                raise RuntimeError(
                    f"Dataset not found at {self._path}. "
                    f"Set `download=True` to download automatically."
                )

        self._walker = sorted(
            str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio)
        )

    def get_metadata(self, n: int) -> Tuple[str, int, str, int, int, int]:
        """Get metadata for the n-th sample (filepath instead of waveform)."""
        fileid = self._walker[n]
        return _get_librispeech_metadata(
            fileid, self._archive, self._url, self._ext_audio, self._ext_txt,
            transcript_suffix=self._transcript_suffix,
        )

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """Load the n-th sample: (waveform, sample_rate, transcript, speaker, chapter, utterance)."""
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self._walker)


# ---------------------------------------------------------------------------
# CustomDataset: sort by length, batch by token count
# ---------------------------------------------------------------------------


class CustomDataset(torch.utils.data.Dataset):
    """Sort LibriSpeech samples by target length and batch to max token count.

    Args:
        base_dataset: A :class:`LIBRISPEECH` instance.
        max_token_limit: Maximum total target tokens per batch.
    """

    def __init__(self, base_dataset: LIBRISPEECH, max_token_limit: int):
        super().__init__()
        self.base_dataset = base_dataset

        fileid_to_target_length = {}
        idx_target_lengths = [
            (idx, self._target_length(fileid, fileid_to_target_length))
            for idx, fileid in enumerate(self.base_dataset._walker)
        ]

        assert len(idx_target_lengths) > 0

        idx_target_lengths = sorted(
            idx_target_lengths, key=lambda x: x[1], reverse=True
        )

        assert max_token_limit >= idx_target_lengths[0][1]

        self.batches = batch_by_token_count(idx_target_lengths, max_token_limit)

    def _target_length(self, fileid: str, fileid_to_target_length: dict) -> int:
        if fileid not in fileid_to_target_length:
            speaker_id, chapter_id, _ = fileid.split("-")
            file_text = (
                speaker_id
                + "-"
                + chapter_id
                + self.base_dataset._transcript_suffix
                + self.base_dataset._ext_txt
            )
            file_text = os.path.join(
                self.base_dataset._path, speaker_id, chapter_id, file_text
            )
            with open(file_text) as ft:
                for line in ft:
                    fileid_text, transcript = line.strip().split(" ", 1)
                    fileid_to_target_length[fileid_text] = len(transcript)
        return fileid_to_target_length[fileid]

    def __getitem__(self, idx: int):
        return [self.base_dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self) -> int:
        return len(self.batches)


# ---------------------------------------------------------------------------
# PyTorch Lightning module
# ---------------------------------------------------------------------------


class LibriSpeechRNNTModule(LightningModule):
    """PyTorch Lightning module for Conformer-RNNT training on LibriSpeech.

    All model, training, and data parameters are configurable.
    Prefer :meth:`from_config` to build from a YAML config dictionary.

    Args:
        librispeech_path: Path to the LibriSpeech dataset root.
        sp_model_path: Path to the SentencePiece model file.
        global_stats_path: Path to the global feature statistics JSON.
        num_symbols: SentencePiece vocabulary size (blank auto +1).
        input_dim: Mel filterbank dimension.
        encoding_dim: Output encoding dimension.
        right_context_length: Right-context frames.
        time_reduction_input_dim: Dim after input projection.
        time_reduction_stride: Frame merging factor.
        num_heads: Attention heads per Conformer layer.
        ffn_dim: Feed-forward hidden dimension.
        num_layers_block1: Conformer layers before compression.
        num_layers_block2: Conformer layers after compression.
        depthwise_conv_kernel_size: Depthwise conv kernel size.
        dropout: Dropout probability.
        compression_filter: Dynamic compression mode.
        hanning_window_size: Hanning window size.
        weight_predictor_mode: Weight predictor input mode.
        symbol_embedding_dim: Token embedding dimension.
        num_lstm_layers: Predictor LSTM layers.
        lstm_hidden_dim: LSTM hidden dimension.
        lstm_layer_norm: Whether to use layer-norm LSTM.
        lstm_layer_norm_epsilon: Epsilon for LSTM LayerNorm.
        lstm_dropout: LSTM dropout.
        joiner_activation: Joiner activation (relu|tanh).
        learning_rate: Peak learning rate.
        adam_betas: Adam beta coefficients.
        adam_eps: Adam epsilon.
        warmup_steps: Linear warmup steps.
        rnnt_loss_reduction: RNNT loss reduction mode.
        rnnt_loss_clamp: RNNT loss logit clamp.
        spec_aug_freq_mask_param: Frequency masking parameter.
        spec_aug_time_mask_param: Time masking parameter.
        spec_aug_time_mask_prob: Time masking probability.
        max_token_limit: Max target tokens per batch.
        num_workers: DataLoader worker count.
        train_subset: LibriSpeech training subset name, or a list of names for
            merged training (e.g. all three train subsets = 960 hours).
        val_subset: LibriSpeech validation subset name.
        test_subset: LibriSpeech test subset name.
    """

    def __init__(
        self,
        *,
        # Paths
        librispeech_path: str,
        sp_model_path: str,
        global_stats_path: str,
        # Model — encoder
        num_symbols: int = 74,
        input_dim: int = 80,
        encoding_dim: int = 512,
        right_context_length: int = 4,
        time_reduction_input_dim: int = 128,
        time_reduction_stride: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        num_layers_block1: int = 2,
        num_layers_block2: int = 8,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        # Model — compression
        compression_filter: str = "hanning",
        hanning_window_size: int = 5,
        weight_predictor_mode: str = "diff5",
        # Model — predictor
        symbol_embedding_dim: int = 128,
        num_lstm_layers: int = 2,
        lstm_hidden_dim: int = 128,
        lstm_layer_norm: bool = True,
        lstm_layer_norm_epsilon: float = 1e-3,
        lstm_dropout: float = 0.3,
        # Model — joiner
        joiner_activation: str = "relu",
        # Training
        learning_rate: float = 5e-4,
        adam_betas: tuple = (0.9, 0.999),
        adam_eps: float = 1e-8,
        warmup_steps: int = 10000,
        rnnt_loss_reduction: str = "sum",
        rnnt_loss_clamp: float = 1.0,
        # Data — augmentation
        spec_aug_freq_mask_param: int = 27,
        spec_aug_time_mask_param: int = 100,
        spec_aug_time_mask_prob: float = 0.2,
        # Data — spectrogram
        sample_rate: int = 16000,
        n_fft: int = 400,
        n_mels: int = 80,
        hop_length: int = 80,
        # Data — loading
        max_token_limit: int = 1000,
        num_workers: int = 32,
        prefetch_factor: int = 2,
        train_subset: Union[str, List[str]] = "train-clean-100",
        val_subset: str = "dev-clean",
        test_subset: str = "test-clean",
        # Data — transcript protocol
        # transcript_suffix: per-chapter transcript filename suffix ("-ph" for
        # phoneme transcripts, "" for official word transcripts).
        # strip_spaces: remove spaces before SPM encoding (phoneme protocol);
        # must be False for word-level SPM models (pieces carry ▁ space marks).
        transcript_suffix: str = "-ph",
        strip_spaces: bool = True,
        # Decoding
        beam_width: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Build model ----
        self.model = conformer_rnnt_model(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            num_symbols=num_symbols + 1,
            right_context_length=right_context_length,
            time_reduction_input_dim=time_reduction_input_dim,
            time_reduction_stride=time_reduction_stride,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers_block1=num_layers_block1,
            num_layers_block2=num_layers_block2,
            dropout=dropout,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            compression_filter=compression_filter,
            hanning_window_size=hanning_window_size,
            weight_predictor_mode=weight_predictor_mode,
            symbol_embedding_dim=symbol_embedding_dim,
            num_lstm_layers=num_lstm_layers,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layer_norm=lstm_layer_norm,
            lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
            lstm_dropout=lstm_dropout,
        )

        # ---- Mel spectrogram params (not a submodule — stays on CPU) ----
        self._mel_kwargs = dict(
            sample_rate=sample_rate, n_fft=n_fft,
            n_mels=n_mels, hop_length=hop_length,
        )
        self._mel_transform = None  # lazy init on CPU

        # ---- Tokenizer (needed before loss — blank_idx derives from it) ----
        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.blank_idx = self.sp_model.get_piece_size()

        # ---- Loss ----
        # blank_idx (= SP vocab size) is the blank label appended past the
        # real symbol ids (0..blank_idx-1). torchaudio RNNTLoss defaults
        # blank=-1 which is wrong here — pass it explicitly.
        self.loss = torchaudio.transforms.RNNTLoss(
            blank=self.blank_idx,
            reduction=rnnt_loss_reduction,
            clamp=rnnt_loss_clamp,
        )

        # ---- Optimizer & scheduler ----
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=adam_betas,
            eps=adam_eps,
        )
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, warmup_steps)

        # ---- Data pipelines ----
        right_pad = right_context_length  # pad amount in time dim
        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            torchaudio.transforms.FrequencyMasking(spec_aug_freq_mask_param),
            torchaudio.transforms.FrequencyMasking(spec_aug_freq_mask_param),
            torchaudio.transforms.TimeMasking(
                spec_aug_time_mask_param, p=spec_aug_time_mask_prob
            ),
            torchaudio.transforms.TimeMasking(
                spec_aug_time_mask_param, p=spec_aug_time_mask_prob
            ),
            FunctionalModule(
                partial(torch.nn.functional.pad, pad=(0, right_pad))
            ),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            FunctionalModule(
                partial(torch.nn.functional.pad, pad=(0, right_pad))
            ),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )

        # ---- Data config ----
        self.librispeech_path = librispeech_path
        self.max_token_limit = max_token_limit
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        # train_subset accepts a single subset name or a list (merged training,
        # e.g. ["train-clean-100", "train-clean-360", "train-other-500"] = 960h)
        self.train_subsets = (
            [train_subset] if isinstance(train_subset, str) else list(train_subset)
        )
        self.val_subset = val_subset
        self.test_subset = test_subset
        self.transcript_suffix = transcript_suffix
        self.strip_spaces = strip_spaces

        # ---- Decoding ----
        # beam_width=0 → greedy (RNNTBeamSearch width=1); used as the
        # lossless oracle baseline for future speculative decoding.
        self.beam_width = beam_width

    # ------------------------------------------------------------------
    # Factory: build from a config dictionary (e.g. loaded from YAML)
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "LibriSpeechRNNTModule":
        """Build a LibriSpeechRNNTModule from a configuration dictionary.

        The config dict should follow the structure of ``configs/default.yaml``.
        """
        paths = config.get("paths", {})
        model_cfg = config.get("model", {})
        comp_cfg = model_cfg.get("compression", {})
        pred_cfg = model_cfg.get("predictor", {})
        joiner_cfg = model_cfg.get("joiner", {})
        training_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        aug_cfg = data_cfg.get("spec_augment", {})
        decoding_cfg = config.get("decoding", {})

        return cls(
            # Paths
            librispeech_path=paths.get("dataset_path", "./librispeech/Librispeech_phoneme_transcription"),
            sp_model_path=paths.get("sp_model_path", "spm_bpe_74.model"),
            global_stats_path=paths.get("global_stats_path", "global_stats.json"),
            # Encoder
            num_symbols=model_cfg.get("num_symbols", 74),
            input_dim=model_cfg.get("input_dim", 80),
            encoding_dim=model_cfg.get("encoding_dim", 512),
            right_context_length=model_cfg.get("right_context_length", 4),
            time_reduction_input_dim=model_cfg.get("time_reduction_input_dim", 128),
            time_reduction_stride=model_cfg.get("time_reduction_stride", 4),
            num_heads=model_cfg.get("num_heads", 8),
            ffn_dim=model_cfg.get("ffn_dim", 1024),
            num_layers_block1=model_cfg.get("num_layers_block1", 2),
            num_layers_block2=model_cfg.get("num_layers_block2", 8),
            depthwise_conv_kernel_size=model_cfg.get("depthwise_conv_kernel_size", 31),
            dropout=model_cfg.get("dropout", 0.1),
            # Compression
            compression_filter=comp_cfg.get("filter", "hanning"),
            hanning_window_size=comp_cfg.get("hanning_window_size", 5),
            weight_predictor_mode=comp_cfg.get("weight_predictor_mode", "diff5"),
            # Predictor
            symbol_embedding_dim=pred_cfg.get("symbol_embedding_dim", 128),
            num_lstm_layers=pred_cfg.get("num_lstm_layers", 2),
            lstm_hidden_dim=pred_cfg.get("lstm_hidden_dim", 128),
            lstm_layer_norm=pred_cfg.get("lstm_layer_norm", True),
            lstm_layer_norm_epsilon=pred_cfg.get("lstm_layer_norm_epsilon", 1e-3),
            lstm_dropout=pred_cfg.get("lstm_dropout", 0.3),
            # Joiner
            joiner_activation=joiner_cfg.get("activation", "relu"),
            # Training
            learning_rate=training_cfg.get("learning_rate", 5e-4),
            adam_betas=tuple(training_cfg.get("adam_betas", [0.9, 0.999])),
            adam_eps=training_cfg.get("adam_eps", 1e-8),
            warmup_steps=training_cfg.get("warmup_steps", 10000),
            rnnt_loss_reduction=training_cfg.get("rnnt_loss_reduction", "sum"),
            rnnt_loss_clamp=training_cfg.get("rnnt_loss_clamp", 1.0),
            # Data — augmentation
            spec_aug_freq_mask_param=aug_cfg.get("freq_mask_param", 27),
            spec_aug_time_mask_param=aug_cfg.get("time_mask_param", 100),
            spec_aug_time_mask_prob=aug_cfg.get("time_mask_prob", 0.2),
            # Data — spectrogram
            sample_rate=data_cfg.get("sample_rate", 16000),
            n_fft=data_cfg.get("n_fft", 400),
            n_mels=data_cfg.get("n_mels", 80),
            hop_length=data_cfg.get("hop_length", 80),
            # Data — loading
            max_token_limit=data_cfg.get("max_token_limit", 1000),
            num_workers=data_cfg.get("num_workers", 32),
            prefetch_factor=data_cfg.get("prefetch_factor", 2),
            train_subset=data_cfg.get("train_subset", "train-clean-100"),
            val_subset=data_cfg.get("val_subset", "dev-clean"),
            test_subset=data_cfg.get("test_subset", "test-clean"),
            # Transcript protocol
            transcript_suffix=data_cfg.get("transcript_suffix", "-ph"),
            strip_spaces=data_cfg.get("strip_spaces", True),
            # Decoding
            beam_width=decoding_cfg.get("beam_width", 20),
        )

    # ------------------------------------------------------------------
    # Mel spectrogram (lazy init on CPU — not registered as submodule)
    # ------------------------------------------------------------------
    def _get_mel_transform(self):
        if self._mel_transform is None:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(**self._mel_kwargs)
        return self._mel_transform

    # ------------------------------------------------------------------
    # Label extraction
    # ------------------------------------------------------------------
    def _extract_labels(self, samples: List):
        targets = [
            self.sp_model.encode(
                sample[2].lower().replace(" ", "") if self.strip_spaces
                else sample[2].lower()
            )
            for sample in samples
        ]
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = (
            torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(elem) for elem in targets],
                batch_first=True,
                padding_value=1.0,
            )
            .to(dtype=torch.int32)
        )
        return targets, lengths

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _train_extract_features(self, samples: List):
        mel = self._get_mel_transform()
        mel_features = [
            mel(sample[0].squeeze()).transpose(1, 0)
            for sample in samples
        ]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        lengths = torch.tensor(
            [elem.shape[0] for elem in mel_features], dtype=torch.int32
        )
        features = self.train_data_pipeline(features)
        return features, lengths

    def _valid_extract_features(self, samples: List):
        mel = self._get_mel_transform()
        mel_features = [
            mel(sample[0].squeeze()).transpose(1, 0)
            for sample in samples
        ]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        lengths = torch.tensor(
            [elem.shape[0] for elem in mel_features], dtype=torch.int32
        )
        features = self.valid_data_pipeline(features)
        return features, lengths

    # ------------------------------------------------------------------
    # Collate functions
    # ------------------------------------------------------------------
    def _train_collate_fn(self, samples: List):
        features, feature_lengths = self._train_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _valid_collate_fn(self, samples: List):
        features, feature_lengths = self._valid_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _test_collate_fn(self, samples: List):
        return self._valid_collate_fn(samples), [sample[2] for sample in samples]

    # ------------------------------------------------------------------
    # Training / validation / test step
    # ------------------------------------------------------------------
    def _step(self, batch: Batch, batch_idx: int, step_type: str):
        if batch is None:
            return None

        prepended_targets = batch.targets.new_empty(
            [batch.targets.size(0), batch.targets.size(1) + 1]
        )
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1

        output, src_lengths, _, _ = self.model(
            batch.features,
            batch.feature_lengths,
            prepended_targets,
            prepended_target_lengths,
        )

        # Clamp source lengths to output time dimension
        max_time = output.size(1)
        src_lengths = torch.clamp(src_lengths, max=max_time).to(torch.int32)
        target_lengths = batch.target_lengths.to(torch.int32)

        loss = self.loss(output, batch.targets, src_lengths, target_lengths)
        self.log(f"Losses/{step_type}_loss", loss, on_step=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        return (
            [self.optimizer],
            [{"scheduler": self.warmup_lr_scheduler, "interval": "step"}],
        )

    def forward(self, batch: Batch):
        # beam_width=0 → greedy: RNNTBeamSearch with width=1 is the greedy
        # path and serves as the lossless oracle for speculative decoding.
        # beam_width>0 → standard beam search at the given width.
        width = 1 if self.beam_width == 0 else self.beam_width
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        hypotheses = decoder(
            batch.features.to(self.device),
            batch.feature_lengths.to(self.device),
            width,
        )
        return post_process_hypos(hypotheses, self.sp_model)[0][0]

    def training_step(self, batch: Batch, batch_idx: int):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch: Batch, batch_idx: int):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch_tuple, batch_idx: int):
        return self._step(batch_tuple[0], batch_idx, "test")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        # One CustomDataset per subset (batches never mix subsets), then concat
        dataset = torch.utils.data.ConcatDataset(
            [
                CustomDataset(
                    LIBRISPEECH(
                        self.librispeech_path,
                        url=subset,
                        transcript_suffix=self.transcript_suffix,
                    ),
                    max_token_limit=self.max_token_limit,
                )
                for subset in self.train_subsets
            ]
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._train_collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        dataset = torch.utils.data.ConcatDataset(
            [
                CustomDataset(
                    LIBRISPEECH(
                        self.librispeech_path,
                        url=self.val_subset,
                        transcript_suffix=self.transcript_suffix,
                    ),
                    max_token_limit=self.max_token_limit,
                ),
            ]
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._valid_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        dataset = LIBRISPEECH(
            self.librispeech_path,
            url=self.test_subset,
            transcript_suffix=self.transcript_suffix,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=self._test_collate_fn,
        )
