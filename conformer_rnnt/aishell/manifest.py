"""WAV manifests, real-length batch plans and reproducible paired sampling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import wave

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .protocol import FRONTEND, feature_frames, file_sha, fingerprint, read_jsonl, relative_file, validate_bundle


def pcm_audio(path, expected_rate=16000):
    """Read the official mono PCM16 format without resampling or codec downloads."""
    with wave.open(str(path), "rb") as reader:
        if (reader.getframerate() != expected_rate or reader.getnchannels() != 1
                or reader.getsampwidth() != 2 or reader.getcomptype() != "NONE"):
            raise ValueError(f"Expected mono {expected_rate}Hz uncompressed PCM16: {path}")
        count = reader.getnframes()
        raw = reader.readframes(count)
        if len(raw) != count * 2:
            raise ValueError(f"Truncated audio: {path}")
    waveform = torch.from_numpy(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)[None]
    return waveform, hashlib.sha256(raw).hexdigest()


def target_units(row):
    return row["syllables"] if "syllables" in row else row["phones"]


def row_audio(root, row):
    rate = row.get("original_sample_rate", 16000)
    waveform, digest = pcm_audio(relative_file(root, row["audio_path"]), expected_rate=rate)
    if digest != row["audio_pcm_sha256"] or waveform.shape[1] != row.get("original_num_samples", row["num_samples"]):
        raise ValueError(f"Audio differs from audited manifest: {row['utterance_id']}")
    if rate != 16000:
        import torchaudio
        from conformer_rnnt.mandarin.preparation import RESAMPLING
        if row.get("resampling") != RESAMPLING:
            raise ValueError("Unapproved resampling recipe")
        waveform = torchaudio.functional.resample(waveform, rate, 16000, **RESAMPLING)
    if waveform.shape[1] != row["num_samples"]:
        raise ValueError("Resampled length differs from audited frame definition")
    if row.get("canonical_audio_sha256") and hashlib.sha256(waveform.numpy().tobytes()).hexdigest() != row["canonical_audio_sha256"]:
        raise ValueError("Canonical audio differs; verify frontend/library version")
    return waveform, digest


class ManifestDataset(Dataset):
    def __init__(self, root, bundle_root, split, tokenizer, *, freeze=None):
        self.root = Path(root)
        self.bundle_root = Path(bundle_root)
        self.freeze = freeze if freeze is not None else validate_bundle(bundle_root)
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown manifest split: {split}")
        self.split = split
        self.rows = list(read_jsonl(self.bundle_root / (split + ".jsonl")))
        if len(self.rows) != self.freeze["counts"][split]:
            raise ValueError("Manifest sample count differs from frozen split")
        self.by_id = {}
        for i, row in enumerate(self.rows):
            uid = row["utterance_id"]
            if uid in self.by_id or row["split"] != split or row["sample_rate"] != 16000:
                raise ValueError(f"Duplicate ID or inconsistent manifest row: {uid}")
            self.by_id[uid] = i
            relative_file(root, row["audio_path"])
            if row["num_frames"] != feature_frames(row["num_samples"]):
                raise ValueError(f"Incorrect feature-frame count: {uid}")
            units = target_units(row)
            if not units or tokenizer.tokenize(" ".join(units)) != units:
                raise ValueError(f"Missing or invalid raw phone labels: {uid}")

    def __len__(self):
        return len(self.rows)

    def transcript(self, index):
        return " ".join(target_units(self.rows[index]))

    def get_metadata(self, index):
        row = self.rows[index]
        return row["audio_path"], 16000, self.transcript(index), row["speaker_id"], self.split, row["utterance_id"]

    def __getitem__(self, index):
        row = self.rows[index]
        waveform, digest = row_audio(self.root, row)
        if waveform.shape[1] != row["num_samples"] or digest != row["audio_pcm_sha256"]:
            raise ValueError(f"Audio differs from audited manifest: {row['utterance_id']}")
        return (waveform,) + self.get_metadata(index)[1:]


def make_batches(rows, token_budget, vocab=218, numel_limit=1900000000):
    if type(token_budget) is not int or token_budget <= 0:
        raise ValueError("Positive integer token budget required")
    order = sorted(rows, key=lambda r: (-len(target_units(r)), r["utterance_id"]))
    batches, current, token_count, max_t, max_u = [], [], 0, 0, 0
    for row in order:
        u, t = len(target_units(row)), row["num_frames"]
        if not u or u > token_budget or t * (u + 1) * vocab >= numel_limit:
            raise ValueError(f"Single sample cannot safely fit; never dropped: {row['utterance_id']}")
        if current and (token_count + u > token_budget
                        or (len(current) + 1) * max(max_t, t) * (max(max_u, u) + 1) * vocab >= numel_limit):
            batches.append(current)
            current, token_count, max_t, max_u = [], 0, 0, 0
        current.append(row["utterance_id"])
        token_count += u
        max_t, max_u = max(max_t, t), max(max_u, u)
    if current:
        batches.append(current)
    validate_batches(rows, batches, token_budget, vocab, numel_limit)
    return batches


def validate_batches(rows, batches, token_budget, vocab=218, numel_limit=1900000000):
    by_id = {r["utterance_id"]: r for r in rows}
    ids = [uid for batch in batches for uid in batch]
    if len(by_id) != len(rows) or len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError("Batch plan must cover every sample exactly once")
    for batch in batches:
        if not batch:
            raise ValueError("Empty batch")
        selected = [by_id[uid] for uid in batch]
        t, u = max(r["num_frames"] for r in selected), max(len(target_units(r)) for r in selected)
        if sum(len(target_units(r)) for r in selected) > token_budget or len(batch) * t * (u + 1) * vocab >= numel_limit:
            raise ValueError("Batch exceeds its real-length token/joint guard")


def create_batch_plan(bundle_root, budget, *, seed=42):
    freeze = validate_bundle(bundle_root)
    vocab = freeze.get("vocab_including_blank", 218)
    plan = {"version": 1, "bundle_fingerprint": freeze["fingerprint"], "seed": seed,
            "token_budget": budget, "vocab_including_blank": vocab, "numel_limit": 1900000000,
            "encoder_time_upper_bound": "full_mel_frames_no_assumed_dynamic_ratio",
            "batches": {split: make_batches(list(read_jsonl(Path(bundle_root) / (split + ".jsonl"))), budget, vocab=vocab)
                        for split in ("train", "dev")}}
    plan["fingerprint"] = fingerprint(plan)
    return plan


class ManifestBatches(Dataset):
    def __init__(self, dataset, plan):
        if (plan.get("fingerprint") != fingerprint({k: v for k, v in plan.items() if k != "fingerprint"})
                or plan["bundle_fingerprint"] != dataset.freeze["fingerprint"]
                or plan["vocab_including_blank"] != dataset.freeze.get("vocab_including_blank", 218) or plan["numel_limit"] != 1900000000):
            raise ValueError("Batch plan fingerprint or frozen data definition differs")
        self.dataset = dataset
        self.batches = plan["batches"][dataset.split]
        validate_batches(dataset.rows, self.batches, plan["token_budget"], vocab=plan["vocab_including_blank"])
        self.plan_fingerprint = plan["fingerprint"]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, item):
        epoch, index = item if isinstance(item, tuple) else (0, item)
        context = {"epoch": epoch, "batch_id": index, "plan": self.plan_fingerprint}
        return [self.dataset[self.dataset.by_id[uid]] + (context,) for uid in self.batches[index]]


class EpochBatchSampler(Sampler):
    def __init__(self, length, *, seed=42, shuffle=True, epoch_getter=None):
        self.length, self.seed, self.shuffle, self.epoch = length, seed, shuffle, 0
        self.epoch_getter = epoch_getter

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.length

    def __iter__(self):
        indices = list(range(self.length))
        epoch = int(self.epoch_getter()) if self.epoch_getter else self.epoch
        if self.shuffle:
            random.Random(self.seed + epoch).shuffle(indices)
        return iter((epoch, index) for index in indices)


def augmentation_seed(seed, context):
    return int(fingerprint({"seed": seed, **context})[:15], 16)
