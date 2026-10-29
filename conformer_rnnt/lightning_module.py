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
from conformer_rnnt.model import conformer_rnnt_model, sequence_options_from_config
from conformer_rnnt.compression.stages import placement_options_from_config
from ipa_tokenizer import DEFAULT_VOCAB, FixedIPATokenizer, encode_transcript
from phone69_tokenizer import DEFAULT_PHONE69_VOCAB, FixedPhone69Tokenizer
from ipa_labels import ExternalIPALabels
from ipa_features import DEFAULT_FEATURES, IPAFeatureTable


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

    Reads ``.flac`` audio in the standard LibriSpeech layout. Targets can use
    legacy chapter ``.trans.txt`` files or external whole-subset Epitran files.

    Args:
        root: Path to the dataset root directory.
        url: Dataset subset name, e.g. ``"train-clean-100"``.
        folder_in_archive: Top-level directory name inside the archive.
        download: If ``True``, download the dataset when not found locally.
        ipa_labels_path: Separate label root containing ``train_clean_100/ipa``.
        ipa_label_filename: Fixed inventory variant; currently ``"ipa"``.
        ipa_tokenizer: Shared fixed tokenizer used to validate external labels.
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
        ipa_labels_path: Optional[str] = None,
        ipa_label_filename: str = "ipa",
        ipa_tokenizer=None,
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

        self._ipa_labels = (
            ExternalIPALabels(ipa_labels_path, url, ipa_label_filename, ipa_tokenizer)
            if ipa_labels_path is not None else None
        )
        self._walker = sorted(
            str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio)
        )
        if self._ipa_labels is not None:
            self._ipa_labels.validate_audio_ids(self._walker)

    def get_metadata(self, n: int) -> Tuple[str, int, str, int, int, int]:
        """Get metadata for the n-th sample (filepath instead of waveform)."""
        fileid = self._walker[n]
        if self._ipa_labels is not None:
            speaker, chapter, utterance = fileid.split("-")
            return (
                os.path.join(self._url, speaker, chapter, fileid + self._ext_audio),
                SAMPLE_RATE, self._ipa_labels.transcripts[fileid],
                int(speaker), int(chapter), int(utterance),
            )
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

    Optionally applies an RNNT-kernel overflow guard: the torchaudio RNNT
    CUDA kernel indexes the joint output tensor with int32, so any batch
    with ``B × T × (U+1) × V >= 2^31`` elements crashes with an illegal
    memory access. When ``sp_model`` and ``rnnt_vocab`` are given, batches
    are greedily split so every batch stays under ``numel_limit``; a single
    sentence that alone exceeds the limit is dropped and reported.

    Args:
        base_dataset: A :class:`LIBRISPEECH` instance.
        max_token_limit: Maximum target phones (IPA) or legacy chars per batch.
        sp_model: Fixed phone tokenizer or word SentencePiece model.
        rnnt_vocab: RNNT output vocab size V incl. blank (guard needs V > 0).
        strip_spaces: Space-stripping applied before legacy encoding — must
            match the training label extraction exactly.
        frames_per_char: Conservative encoder-output-frames-per-char ratio
            used to bound T (measured average ≈ 1.75 at 25 fps output).
        numel_limit: Batch element budget B×T×(U+1)×V (kernel limit 2^31).
    """

    def __init__(
        self,
        base_dataset: LIBRISPEECH,
        max_token_limit: int,
        sp_model=None,
        rnnt_vocab: int = 0,
        strip_spaces: bool = True,
        frames_per_char: float = 3.0,
        numel_limit: float = 1.9e9,
    ):
        super().__init__()
        self.base_dataset = base_dataset

        fileid_to_transcript = {}
        idx_target_lengths = [
            (idx, len(encode_transcript(sp_model, self._transcript(fileid, fileid_to_transcript), strip_spaces))
             if getattr(sp_model, "is_phone_tokenizer", False)
             else len(self._transcript(fileid, fileid_to_transcript)))
            for idx, fileid in enumerate(self.base_dataset._walker)
        ]

        assert len(idx_target_lengths) > 0

        idx_target_lengths = sorted(
            idx_target_lengths, key=lambda x: x[1], reverse=True
        )

        assert max_token_limit >= idx_target_lengths[0][1]

        self.batches = batch_by_token_count(idx_target_lengths, max_token_limit)

        if sp_model is not None and rnnt_vocab > 0:
            self.batches = self._apply_numel_guard(
                idx_target_lengths, fileid_to_transcript, sp_model,
                rnnt_vocab, strip_spaces, frames_per_char, numel_limit,
            )

    def _transcript(self, fileid: str, fileid_to_transcript: dict) -> str:
        if self.base_dataset._ipa_labels is not None:
            text = self.base_dataset._ipa_labels.transcripts[fileid]
            fileid_to_transcript[fileid] = text
            return text
        if fileid not in fileid_to_transcript:
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
                    fileid_to_transcript[fileid_text] = transcript
        return fileid_to_transcript[fileid]

    def _apply_numel_guard(
        self, idx_target_lengths, fileid_to_transcript, sp_model,
        rnnt_vocab, strip_spaces, frames_per_char, numel_limit,
    ):
        """Split batches so B×T_est×(U+1)×V stays below the int32 limit.

        U (real token count) is computed with the exact same encoding path
        as ``_extract_labels``; T is bounded conservatively via
        ``chars × frames_per_char``.
        """
        chars_by_idx = {
            idx: len(fileid_to_transcript[fileid])
            for idx, fileid in enumerate(self.base_dataset._walker)
        }
        token_len_by_idx = {}
        for idx, fileid in enumerate(self.base_dataset._walker):
            token_len_by_idx[idx] = len(encode_transcript(
                sp_model, fileid_to_transcript[fileid], strip_spaces
            ))

        kept, dropped = [], []
        worst = 0.0

        def flush(cur, cur_t, cur_u):
            nonlocal worst
            if not cur:
                return
            if len(cur) == 1 and cur_t * cur_u * rnnt_vocab >= numel_limit:
                fid = self.base_dataset._walker[cur[0]]
                dropped.append((fid, int(cur_t), cur_u - 1))
                return
            worst = max(worst, len(cur) * cur_t * cur_u * rnnt_vocab)
            kept.append(cur)

        for batch in self.batches:
            cur, cur_t, cur_u = [], 0.0, 0
            for i in batch:
                t = chars_by_idx[i] * frames_per_char
                u = token_len_by_idx[i] + 1  # +1: blank slot on the U axis
                if cur and (
                    (len(cur) + 1) * max(cur_t, t) * max(cur_u, u) * rnnt_vocab
                    >= numel_limit
                ):
                    flush(cur, cur_t, cur_u)
                    cur, cur_t, cur_u = [], 0.0, 0
                cur.append(i)
                cur_t, cur_u = max(cur_t, t), max(cur_u, u)
            flush(cur, cur_t, cur_u)

        msg = (
            f"[CustomDataset:{self.base_dataset._url}] RNNT numel guard "
            f"(V={rnnt_vocab}, limit={numel_limit:.3g}): "
            f"batches {len(self.batches)} -> {len(kept)}"
        )
        if dropped:
            msg += f", dropped {len(dropped)} oversized sentence(s): {dropped}"
        msg += f", worst batch est. {worst:.3g} elements"
        print(msg)
        return kept

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
        num_symbols: Tokenizer vocabulary size (blank auto +1).
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
        val_subset: LibriSpeech validation subset name, or a list of names.
        test_subset: LibriSpeech test subset name, or a list of names.
        rnnt_numel_limit: Max B×T×(U+1)×V elements per train/val batch (guard
            against the RNNT CUDA kernel's int32 indexing limit of 2^31).
        rnnt_frames_per_char: Conservative output-frames-per-char estimate
            used by the overflow guard (measured average ≈ 1.75).
    """

    def __init__(
        self,
        *,
        # Paths
        librispeech_path: str,
        global_stats_path: str,
        sp_model_path: Optional[str] = None,
        tokenizer_type: str = "phone69",
        ipa_vocab_path: Optional[str] = None,
        phone69_vocab_path: Optional[str] = None,
        ipa_features_path: Optional[str] = None,
        ipa_labels_path: Optional[str] = None,
        ipa_label_filename: str = "ipa",
        # Model — encoder
        num_symbols: Optional[int] = None,
        input_dim: int = 80,
        encoding_dim: int = 512,
        right_context_length: int = 4,
        time_reduction_input_dim: int = 128,
        time_reduction_stride: int = 4,
        compression_right_context_length: Optional[int] = None,
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
        compression_detector: Optional[dict] = None,
        compression_mixer: Optional[dict] = None,
        compression_keep_rate: Optional[float] = None,
        sequence_mode: str = "legacy",
        encoder_num_layers: Optional[int] = None,
        compression_stages: Optional[List[dict]] = None,
        compression_execution_backend: str = "reference",
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
        # "batch" = one batched rnnt_loss call; "sample" = per-sample calls;
        # "group" = samples packed into sub-batches under a numel cap (see
        # _pack_groups). All three are identical under reduction="sum", but
        # the kernel's DP has a ~35ms sequential wavefront latency per call
        # that batching parallelizes across samples — per-sample is B× that
        # latency. "group" sidesteps the torchaudio 2.6 CUDA kernel
        # illegal-memory-access (which only ever fired on large multi-sample
        # calls) while keeping most of the batched-call speed.
        rnnt_loss_chunk: str = "batch",
        rnnt_group_numel_limit: float = 2.5e8,
        rnnt_group_max_b: int = 8,
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
        val_subset: Union[str, List[str]] = "dev-clean",
        test_subset: Union[str, List[str]] = "test-clean",
        # Data — transcript protocol
        # transcript_suffix: per-chapter transcript filename suffix ("-ph" for
        # ARPABET transcripts, "-ipa" for IPA, "" for official word transcripts).
        # Keep phone boundaries and word spaces intact.
        transcript_suffix: str = "-ph",
        strip_spaces: bool = False,
        # Data — RNNT kernel overflow guard (word-BPE V=4001 can exceed the
        # int32 indexing limit of the torchaudio RNNT CUDA kernel)
        rnnt_numel_limit: float = 1.9e9,
        rnnt_frames_per_char: float = 3.0,
        # Decoding
        beam_width: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Fixed phone inventories do not import or load SentencePiece.
        self.tokenizer_type = tokenizer_type
        self.is_phone_tokenizer = tokenizer_type in {"ipa", "phone69"}
        if self.is_phone_tokenizer and strip_spaces:
            raise ValueError(f"{tokenizer_type} requires data.strip_spaces: false to preserve phone boundaries")
        if tokenizer_type == "phone69":
            if ipa_labels_path is not None:
                raise ValueError("External Epitran labels require tokenizer.type: ipa")
            self.tokenizer = FixedPhone69Tokenizer(phone69_vocab_path or DEFAULT_PHONE69_VOCAB)
        elif tokenizer_type == "ipa":
            self.tokenizer = FixedIPATokenizer(ipa_vocab_path or DEFAULT_VOCAB)
            self.ipa_features = IPAFeatureTable(self.tokenizer, ipa_features_path or DEFAULT_FEATURES)
            self.register_buffer(
                "ipa_articulatory_table",
                torch.tensor([*self.ipa_features.vectors, (0,) * 24], dtype=torch.float32),
                persistent=False,
            )
        elif tokenizer_type == "sentencepiece":
            if ipa_labels_path is not None:
                raise ValueError("External Epitran labels require tokenizer.type: ipa")
            if transcript_suffix == "-ph":
                raise ValueError("Phone labels require a fixed tokenizer (phone69 or ipa), not SentencePiece")
            import sentencepiece as spm
            if not sp_model_path:
                raise ValueError("SentencePiece mode requires paths.sp_model_path")
            self.tokenizer = spm.SentencePieceProcessor(model_file=sp_model_path)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")
        self.sp_model = self.tokenizer  # Legacy batch/decoding adapter name.
        self.blank_idx = self.tokenizer.get_piece_size()
        if num_symbols is not None and num_symbols != self.blank_idx:
            raise ValueError(
                f"model.num_symbols={num_symbols} does not match tokenizer size {self.blank_idx} (excluding blank)"
            )
        num_symbols = self.blank_idx
        self.padding_idx = self.blank_idx if self.is_phone_tokenizer else 1

        # ---- Build model ----
        self.model = conformer_rnnt_model(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            num_symbols=num_symbols + 1,
            right_context_length=right_context_length,
            time_reduction_input_dim=time_reduction_input_dim,
            time_reduction_stride=time_reduction_stride,
            compression_right_context_length=compression_right_context_length,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers_block1=num_layers_block1,
            num_layers_block2=num_layers_block2,
            dropout=dropout,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            compression_filter=compression_filter,
            hanning_window_size=hanning_window_size,
            weight_predictor_mode=weight_predictor_mode,
            compression_detector=compression_detector,
            compression_mixer=compression_mixer,
            compression_keep_rate=compression_keep_rate,
            sequence_mode=sequence_mode,
            encoder_num_layers=encoder_num_layers,
            compression_stages=compression_stages,
            compression_execution_backend=compression_execution_backend,
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

        # ---- Loss ----
        # blank_idx (= tokenizer size) is the blank label appended past the
        # real symbol ids (0..blank_idx-1). torchaudio RNNTLoss defaults
        # blank=-1 which is wrong here — pass it explicitly.
        self.loss = torchaudio.transforms.RNNTLoss(
            blank=self.blank_idx,
            reduction=rnnt_loss_reduction,
            clamp=rnnt_loss_clamp,
        )
        self.rnnt_loss_chunk = rnnt_loss_chunk
        self.rnnt_group_numel_limit = rnnt_group_numel_limit
        self.rnnt_group_max_b = rnnt_group_max_b

        # ---- Optimizer & scheduler ----
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=adam_betas,
            eps=adam_eps,
        )
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, warmup_steps)

        # ---- Data pipelines ----
        self.sequence_mode = sequence_mode
        self.offline_native_definition = None
        if sequence_mode == "offline":
            self.offline_native_definition = {
                "version": 1, "sequence_mode": sequence_mode, "tail_alignment": "zero_pad_ceil",
                "stride": time_reduction_stride, "input_projection": time_reduction_input_dim,
                "encoding_dim": encoding_dim, "num_heads": num_heads, "ffn_dim": ffn_dim,
                "layers": [num_layers_block1, num_layers_block2],
                "detector": compression_detector, "mixer": compression_mixer,
                "filter": compression_filter, "window_size": hanning_window_size,
                "weight_predictor_mode": weight_predictor_mode,
            }
            if compression_stages is not None:
                self.offline_native_definition = {
                    "version": 2, "sequence_mode": sequence_mode,
                    "tail_alignment": "zero_pad_ceil",
                    "stride": time_reduction_stride,
                    "input_dim": input_dim, "input_projection": time_reduction_input_dim,
                    "encoding_dim": encoding_dim, "num_heads": num_heads, "ffn_dim": ffn_dim,
                    "depthwise_conv_kernel_size": depthwise_conv_kernel_size,
                    "dropout": dropout,
                    **self.model.transcriber.placement_definition,
                }
        right_pad = right_context_length  # legacy only; offline pipeline omits this operation
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
            *([FunctionalModule(
                partial(torch.nn.functional.pad, pad=(0, right_pad))
            )] if sequence_mode != "offline" else []),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            *([FunctionalModule(
                partial(torch.nn.functional.pad, pad=(0, right_pad))
            )] if sequence_mode != "offline" else []),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )

        # ---- Data config ----
        self.librispeech_path = librispeech_path
        self.ipa_labels_path = ipa_labels_path
        self.ipa_label_filename = ipa_label_filename
        self.max_token_limit = max_token_limit
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        # train_subset/val_subset/test_subset accept a single subset name or a
        # list of names for merged loading (e.g. all three train subsets = 960h;
        # ["test-clean", "test-other"] = mixed eval). Batches never mix subsets.
        self.train_subsets = self._as_list(train_subset)
        self.val_subsets = self._as_list(val_subset)
        self.test_subsets = self._as_list(test_subset)
        self.transcript_suffix = transcript_suffix
        self.strip_spaces = strip_spaces
        self.rnnt_numel_limit = rnnt_numel_limit
        self.rnnt_frames_per_char = rnnt_frames_per_char

        # ---- Decoding ----
        # beam_width=0 → greedy (RNNTBeamSearch width=1); used as the
        # lossless oracle baseline for future speculative decoding.
        self.beam_width = beam_width

    @staticmethod
    def _as_list(subset: Union[str, List[str]]) -> List[str]:
        return [subset] if isinstance(subset, str) else list(subset)

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
            sp_model_path=paths.get("sp_model_path"),
            global_stats_path=paths.get("global_stats_path", "assets/statistics/global_stats.json"),
            # Encoder
            num_symbols=model_cfg.get("num_symbols"),
            tokenizer_type=config.get("tokenizer", {}).get(
                "type", "sentencepiece" if data_cfg.get("transcript_suffix", "-ph") == "" else "phone69"
            ),
            ipa_vocab_path=config.get("tokenizer", {}).get("vocab_path"),
            phone69_vocab_path=config.get("tokenizer", {}).get("vocab_path"),
            ipa_features_path=config.get("tokenizer", {}).get("features_path"),
            ipa_labels_path=paths.get("ipa_labels_path"),
            ipa_label_filename=data_cfg.get("ipa_label_filename", "ipa"),
            input_dim=model_cfg.get("input_dim", 80),
            encoding_dim=model_cfg.get("encoding_dim", 512),
            **sequence_options_from_config(model_cfg),
            **placement_options_from_config(model_cfg),
            time_reduction_input_dim=model_cfg.get("time_reduction_input_dim", 128),
            time_reduction_stride=model_cfg.get("time_reduction_stride", 4),
            num_heads=model_cfg.get("num_heads", 8),
            ffn_dim=model_cfg.get("ffn_dim", 1024),
            num_layers_block1=model_cfg.get("num_layers_block1", 2),
            num_layers_block2=model_cfg.get("num_layers_block2", 8),
            depthwise_conv_kernel_size=model_cfg.get("depthwise_conv_kernel_size", 31),
            dropout=model_cfg.get("dropout", 0.1),
            # Compression
            compression_filter=comp_cfg.get("filter", "none" if "stages" in comp_cfg else "hanning"),
            hanning_window_size=comp_cfg.get("hanning_window_size", 5),
            weight_predictor_mode=comp_cfg.get("weight_predictor_mode", "five_frames" if "stages" in comp_cfg else "diff5"),
            compression_detector=comp_cfg.get("detector"),
            compression_mixer=comp_cfg.get("mixer"),
            compression_keep_rate=comp_cfg.get("keep_rate"),
            compression_execution_backend=comp_cfg.get("execution_backend", "reference"),
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
            rnnt_loss_chunk=training_cfg.get("rnnt_loss_chunk", "batch"),
            rnnt_group_numel_limit=float(
                training_cfg.get("rnnt_group_numel_limit", 2.5e8)
            ),
            rnnt_group_max_b=int(training_cfg.get("rnnt_group_max_b", 8)),
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
            strip_spaces=data_cfg.get("strip_spaces", False),
            # RNNT kernel overflow guard (float(): PyYAML may hand back a str)
            rnnt_numel_limit=float(data_cfg.get("rnnt_numel_limit", 1.9e9)),
            rnnt_frames_per_char=float(data_cfg.get("rnnt_frames_per_char", 3.0)),
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
            encode_transcript(self.tokenizer, sample[2], self.strip_spaces)
            for sample in samples
        ]
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = (
            torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(elem) for elem in targets],
                batch_first=True,
                padding_value=self.padding_idx,
            )
            .to(dtype=torch.int32)
        )
        return targets, lengths

    def _extract_articulatory_labels(self, samples: List):
        """Optional fixed targets [B, U, 24] with padding/blank masked out."""
        if self.tokenizer_type != "ipa":
            raise ValueError("Articulatory targets require the fixed IPA tokenizer")
        targets, lengths = self._extract_labels(samples)
        targets = targets.to(self.ipa_articulatory_table.device)
        vectors = self.ipa_articulatory_table[targets.long()]
        mask = torch.arange(targets.size(1), device=targets.device)[None, :] < lengths.to(targets.device)[:, None]
        return vectors, mask

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
        if self.sequence_mode == "offline":
            features = self._mask_feature_padding(features, lengths)
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
        if self.sequence_mode == "offline":
            features = self._mask_feature_padding(features, lengths)
        return features, lengths

    @staticmethod
    def _mask_feature_padding(features, lengths):
        valid = torch.arange(features.size(1), device=features.device)[None, :] < lengths.to(features.device)[:, None]
        return features.masked_fill(~valid[:, :, None], 0.0)

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

        loss = self._rnnt_loss(output, batch.targets, src_lengths, target_lengths)
        # reduction="sum": `loss` is a whole-batch sum. Batching is dynamic
        # (batch_size=None, and the collate can drop over-limit batches), so a
        # plain mean-over-batches swings tens of percent between validations
        # from composition alone. Divide by the utterance count and pass
        # batch_size so the epoch aggregate is sum(loss) / sum(utterances).
        bsz = batch.targets.size(0)
        self.log(
            f"Losses/{step_type}_loss",
            loss / bsz,
            on_step=True,
            on_epoch=True,
            batch_size=bsz,
        )
        return loss

    def _rnnt_loss(self, output, targets, src_lengths, target_lengths):
        """RNNT loss — batched, per-sample, or grouped calls (``rnnt_loss_chunk``).

        All modes are exactly equivalent under ``reduction="sum"``: the DP
        never enters padded frames, so cropping changes nothing.

        NOTE: the torchaudio rnnt_loss op (v2.6.0, compute.cu) requires
        ``logits.size(1) == max(logit_lengths)``, ``logits.size(2) ==
        max(target_lengths) + 1`` and ``targets.size(1) == max(target_lengths)``
        *strictly*, and requires a contiguous logits tensor (compute.cu:39).
        Every non-batch call therefore crops to the max lengths *within its
        group* — the op handles per-sample lengths internally.

        Perf: crop bounds come from a single ``.tolist()`` — indexing CUDA
        scalars with ``int()`` per sample would drain the GPU queue once per
        sample and serialize the whole step on host<->device syncs.
        """
        mode = self.rnnt_loss_chunk
        if mode == "batch":
            return self.loss(output, targets, src_lengths, target_lengths)

        t_list = src_lengths.tolist()
        u_list = target_lengths.tolist()

        if mode == "sample":
            ranges = [(i, i + 1) for i in range(output.size(0))]
        elif mode == "group":
            order = sorted(range(len(t_list)), key=lambda i: (t_list[i], u_list[i]))
            ranges = self._pack_groups([t_list[i] for i in order],
                                       [u_list[i] for i in order],
                                       output.size(3))
            # Reorder so each group is a contiguous slice (index_select copy
            # is order-invariant for a summed loss; autograd scatters back).
            order_t = torch.tensor(order, device=output.device, dtype=torch.long)
            output = output.index_select(0, order_t)
            targets = targets.index_select(0, order_t)
            src_lengths = src_lengths.index_select(0, order_t)
            target_lengths = target_lengths.index_select(0, order_t)
            t_list = src_lengths.tolist()
            u_list = target_lengths.tolist()
        else:
            raise ValueError(f"unknown rnnt_loss_chunk mode: {mode!r}")

        losses = [
            self.loss(
                output[lo:hi, : max(t_list[lo:hi]),
                       : max(u_list[lo:hi]) + 1].contiguous(),
                targets[lo:hi, : max(u_list[lo:hi])].contiguous(),
                src_lengths[lo:hi],
                target_lengths[lo:hi],
            )
            for lo, hi in ranges
        ]
        return torch.stack(losses).sum()

    def _pack_groups(self, t_list, u_list, vocab):
        """Greedy-pack T-sorted samples into contiguous [lo, hi) ranges with
        group_size x max_T x (max_U+1) x V under ``rnnt_group_numel_limit``
        (and at most ``rnnt_group_max_b`` samples per group). The limit keeps
        every multi-sample op call far below the numel regime where the
        torchaudio 2.6 kernel has shown illegal memory access. A sample that
        alone exceeds the limit stays in its own single-sample group."""
        n = len(t_list)
        ranges = []
        start = 0
        for end in range(1, n + 1):
            t_g = max(t_list[start:end])
            u_g = max(u_list[start:end]) + 1
            over = (
                (end - start) * t_g * u_g * vocab > self.rnnt_group_numel_limit
                or (end - start) > self.rnnt_group_max_b
            )
            if over and end - start > 1:
                # sample end-1 pushed the window over — close the group
                # before it and let it start the next one
                ranges.append((start, end - 1))
                start = end - 1
        ranges.append((start, n))
        return ranges

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint):
        if self.sequence_mode == "offline":
            checkpoint["offline_native_definition"] = self.offline_native_definition
        if self.tokenizer_type == "phone69":
            checkpoint["phone69_tokenizer"] = {
                "fingerprint": self.tokenizer.fingerprint,
                "tokens": list(self.tokenizer.tokens),
            }
        if self.tokenizer_type == "ipa":
            checkpoint["ipa_tokenizer"] = {
                "fingerprint": self.tokenizer.fingerprint,
                "tokens": list(self.tokenizer.tokens),
                "features_fingerprint": self.ipa_features.fingerprint,
            }

    def validate_tokenizer_checkpoint(self, checkpoint):
        if self.tokenizer_type == "phone69":
            metadata = checkpoint.get("phone69_tokenizer") or {}
            if metadata.get("fingerprint") != self.tokenizer.fingerprint:
                raise ValueError(
                    "Checkpoint phone69 vocabulary is missing or differs from the configured inventory. "
                    "SPM checkpoints cannot be resumed as fixed phone69 checkpoints."
                )
        if self.tokenizer_type == "ipa":
            metadata = checkpoint.get("ipa_tokenizer") or {}
            if metadata.get("fingerprint") != self.tokenizer.fingerprint:
                raise ValueError(
                    "Checkpoint IPA vocabulary is missing or differs from the configured inventory. "
                    "Word/SPM checkpoints cannot be resumed as IPA checkpoints."
                )
            if metadata.get("features_fingerprint") != self.ipa_features.fingerprint:
                raise ValueError("Checkpoint PanPhon feature table is missing or differs from the configuration")

    def on_load_checkpoint(self, checkpoint):
        self.validate_tokenizer_checkpoint(checkpoint)
        if self.sequence_mode == "offline" and checkpoint.get("offline_native_definition") != self.offline_native_definition:
            raise ValueError("Checkpoint does not match the offline natural-compression definition")

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
                        ipa_labels_path=self.ipa_labels_path,
                        ipa_label_filename=self.ipa_label_filename,
                        ipa_tokenizer=self.tokenizer,
                    ),
                    max_token_limit=self.max_token_limit,
                    sp_model=self.sp_model,
                    rnnt_vocab=self.blank_idx + 1,
                    strip_spaces=self.strip_spaces,
                    frames_per_char=self.rnnt_frames_per_char,
                    numel_limit=self.rnnt_numel_limit,
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
                        url=subset,
                        transcript_suffix=self.transcript_suffix,
                        ipa_labels_path=self.ipa_labels_path,
                        ipa_label_filename=self.ipa_label_filename,
                        ipa_tokenizer=self.tokenizer,
                    ),
                    max_token_limit=self.max_token_limit,
                    sp_model=self.sp_model,
                    rnnt_vocab=self.blank_idx + 1,
                    strip_spaces=self.strip_spaces,
                    frames_per_char=self.rnnt_frames_per_char,
                    numel_limit=self.rnnt_numel_limit,
                )
                for subset in self.val_subsets
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
        dataset = torch.utils.data.ConcatDataset(
            [
                LIBRISPEECH(
                    self.librispeech_path,
                    url=subset,
                    transcript_suffix=self.transcript_suffix,
                    ipa_labels_path=self.ipa_labels_path,
                    ipa_label_filename=self.ipa_label_filename,
                    ipa_tokenizer=self.tokenizer,
                )
                for subset in self.test_subsets
            ]
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=self._test_collate_fn,
        )
