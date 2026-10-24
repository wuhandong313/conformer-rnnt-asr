"""Pair validation, resource fingerprints and code-only readiness checks."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .protocol import FRONTEND, file_sha, fingerprint, validate_bundle

BUDGETS = (6000, 4500, 3000, 2000, 1000)
SCHEMES = ("stride4_none", "stride1_cosd_learned")


def profile(config):
    return "mandarin_syllable_mix" if config["tokenizer"]["type"] == "mandarin_syllable" else "aishell1_lexicon"


def read_configs(paths):
    configs = [yaml.safe_load(Path(p).read_text(encoding="utf-8")) for p in paths]
    validate_pair(configs)
    return configs


def validate_pair(configs):
    if len(configs) != 2:
        raise ValueError("Exactly two ordered Chinese configs are required")
    common = []
    for config, scheme in zip(configs, SCHEMES):
        if (set(config) != {"paths", "tokenizer", "model", "training", "data", "trainer", "decoding", "debug", "seed", "execution"}
                or set(config["paths"]) != {"dataset_path", "manifest_root", "global_stats_path", "batch_plan_path", "exp_dir", "from_checkpoint"}
                or set(config["tokenizer"]) != {"type", "vocab_path"}
                or set(config["execution"]) != {"allow_dataset_download", "allow_training", "reason"}
                or type(config["execution"]["allow_training"]) is not bool or config["debug"] is not False):
            raise ValueError("Unknown or unapproved recipe controls are not silently accepted")
        m, t, d = config["model"], config["training"], config["data"]
        syllables = config["tokenizer"]["type"] == "mandarin_syllable"
        symbols = 217
        if syllables:
            symbols = m["num_symbols"]
            if symbols is not None:
                from mandarin_syllable_tokenizer import MandarinSyllableTokenizer
                if symbols != MandarinSyllableTokenizer(config["tokenizer"]["vocab_path"]).get_piece_size():
                    raise ValueError("Resolved symbol count differs from the frozen syllable inventory")
        stride = 4 if scheme == "stride4_none" else 1
        expected_compression = ({"filter": "none"} if stride == 4 else
                                {"filter": "none", "detector": {"type": "cosine_diff", "preprocess": "none"},
                                 "mixer": {"type": "learned"}, "hanning_window_size": 5,
                                 "weight_predictor_mode": "five_frames"})
        expected = {"sequence_mode": "offline", "input_dim": 80, "encoding_dim": 256, "num_symbols": symbols,
                    "time_reduction_stride": stride, "time_reduction_input_dim": 128 // stride,
                    "num_heads": 8, "ffn_dim": 512, "num_layers_block1": 2, "num_layers_block2": 8,
                    "depthwise_conv_kernel_size": 31, "dropout": 0.1,
                    "predictor": {"symbol_embedding_dim": 128, "num_lstm_layers": 2, "lstm_hidden_dim": 128,
                                  "lstm_layer_norm": True, "lstm_layer_norm_epsilon": .001, "lstm_dropout": .3},
                    "joiner": {"activation": "relu"}, "compression": expected_compression}
        if m != expected:
            raise ValueError(f"Unapproved model/compression definition: {scheme}")
        fixed_training = {"learning_rate": .0005, "adam_betas": [.9, .999], "adam_eps": 1.e-8,
                          "warmup_steps": 10000, "rnnt_loss_reduction": "sum", "rnnt_loss_clamp": 1.,
                          "rnnt_loss_chunk": "batch", "gradient_clip_val": 10., "epochs": 50}
        fixed_trainer = {"accelerator": "cuda", "strategy": "auto", "devices": 1, "num_nodes": 1,
                         "log_every_n_steps": 10, "val_check_interval": 200, "num_sanity_val_steps": 0,
                         "fast_dev_run": False, "matmul_precision": "high", "accumulate_grad_batches": 1,
                         "precision": "32-true", "checkpoint": {"immutable": True, "save_weights_only": False,
                         "train_checkpoint_interval": 500, "save_epoch_boundaries": True, "save_initial": True}}
        fixed_data = {"sample_rate": 16000, "n_fft": 400, "n_mels": 80, "hop_length": 160,
                      "spec_augment": {"freq_mask_param": 27, "time_mask_param": 100, "time_mask_prob": .2},
                      "strip_spaces": False, "rnnt_numel_limit": 1900000000,
                      "train_subset": "train", "val_subset": "dev", "test_subset": "test", "dataset_type": "manifest"}
        if (t != fixed_training or config["trainer"] != fixed_trainer
                or any(d.get(k) != v for k, v in fixed_data.items())
                or set(d) != set(fixed_data) | {"max_token_limit", "num_workers", "prefetch_factor"}
                or d["max_token_limit"] not in BUDGETS or config["decoding"] != {"beam_width": 20}):
            raise ValueError("Unapproved frontend, predictor, augmentation, optimizer or trainer controls")
        def keys(value):
            return {k for k, v in value.items()} | set().union(*(keys(v) for v in value.values() if isinstance(v, dict)))
        if {"keep_rate", "right_context_length", "left_context_length", "stages", "articulatory"} & keys(m):
            raise ValueError("Obsolete context/rate or extra model controls are forbidden")
        if (config["tokenizer"]["type"] not in {"aishell_lexicon", "mandarin_syllable"} or d["dataset_type"] != "manifest"
                or d["strip_spaces"] is not False or config["seed"] != 42
                or t["epochs"] != 50 or t["learning_rate"] != 0.0005 or t["warmup_steps"] != 10000
                or t.get("early_stopping") or t["rnnt_loss_reduction"] != "sum"
                or config["decoding"]["beam_width"] != 20
                or config["paths"].get("from_checkpoint") is not None
                or config["execution"].get("allow_dataset_download") is not False
                or Path(config["paths"]["exp_dir"]).name != f"exp_{profile(config)}_{scheme}_s42_e50"):
            raise ValueError("Pair violates the approved data/target/training protocol")
        reduced = json.loads(json.dumps(config))
        for key in ("time_reduction_stride", "time_reduction_input_dim", "compression"):
            reduced["model"].pop(key)
        reduced["paths"].pop("exp_dir")
        common.append(reduced)
    if common[0] != common[1]:
        raise ValueError("The pair must share labels, frontend, budget, batches, augmentation and training settings")


def source_fingerprints(repo):
    repo = Path(repo)
    paths = list((repo / "conformer_rnnt").rglob("*.py"))
    paths += list((repo / "scripts/server").glob("*aishell1*.py"))
    paths += list((repo / "scripts/evaluation").glob("*aishell1*.py"))
    paths += list((repo / "scripts/data").glob("*aishell1*.py"))
    for directory in ("scripts/server", "scripts/data", "scripts/evaluation"):
        paths += list((repo / directory).glob("*mandarin*.py"))
    paths += list(repo.glob("*.py"))
    paths += list((repo / "tests").glob("test_*.py"))
    paths += [repo / p for p in ("requirements-aishell-prep.txt", "configs/aishell_lexicon_vocab.json")]
    return {str(p.relative_to(repo)): file_sha(p) for p in sorted(set(paths))}


def readiness(configs):
    validate_pair(configs)
    result = {"status": "code_ready_data_not_checked", "dataset_download_allowed": False,
              "training_allowed": all(c["execution"].get("allow_training") is True for c in configs), "missing": []}
    checks = {key: configs[0]["paths"][key] for key in ("dataset_path", "manifest_root", "batch_plan_path", "global_stats_path")}
    checks["vocabulary"] = configs[0]["tokenizer"]["vocab_path"]
    for key, value in checks.items():
        path = Path(value)
        if not path.exists():
            result["missing"].append({"key": key, "path": str(path)})
    if result["missing"]:
        result["status"] = "blocked_missing_local_data_or_frozen_artifacts"
    else:
        try:
            freeze = validate_bundle(configs[0]["paths"]["manifest_root"])
            result.update(status="frozen_data_ready_preflight_required", bundle_fingerprint=freeze["fingerprint"])
        except (ValueError, OSError, KeyError) as error:
            result.update(status="blocked_invalid_frozen_artifacts", error=str(error))
    return result
