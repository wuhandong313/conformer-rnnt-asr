#!/usr/bin/env python3
"""Generate experiment configs for the A4 equal-rate baselines and B-matrix.

Usage:
    python scripts/gen_configs.py --rate 0.30 --out-dir configs/exp
    python scripts/gen_configs.py --rate 0.30 --smoke        # 2-epoch dry run

Reads ``configs/server.yaml`` as the template. Every generated config differs
only in ``model.compression.{detector,mixer,keep_rate}`` and ``paths.exp_dir``
(per-run isolation — training-plan §2.1). Ckpt filenames automatically carry
the detector/mixer/rate tag (train.py).

Config inventory
----------------
A4 (10, equal-rate, mixer=hard unless stated):
    a4-uniform a4-random a4-topk-diff a4-energy a4-max-pool a4-avg-pool*
    a4-mmdp a4-ple a4-cosine-diff a4-savgol-cal
    (*avg-pool pairs with the seg_avg mixer; our main config runs the legacy
    dynamic savgol path unchanged — see configs/server.yaml.)
B-matrix (4×3 = 12 + PLE historical 3):
    b-{raw,savgol,cosine_diff,cosine_diff_savgol} × {hard,seg-avg,learned}
    (fir dropped from the narrative 2026-10-05 — redundant smoothing variant
    of raw; ple kept for the collapse lesson. cos+savgol completes the
    signal × smoothing 2×2 grid.)
"""

import argparse
import copy
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "server.yaml"
BASE_EXP_ROOT = "/test21/conformer-rnnt-asr"

# name -> (detector type, mixer type, extra params)
A4_BASELINES = {
    "a4-uniform": ("uniform", "hard", {}),
    "a4-random": ("random", "hard", {}),
    "a4-topk-diff": ("topk_diff", "hard", {}),
    "a4-energy": ("energy", "hard", {}),
    "a4-max-pool": ("max_pool", "hard", {}),
    "a4-avg-pool": ("avg_pool", "seg_avg", {}),
    "a4-mmdp": ("mmdp", "hard", {}),
    "a4-ple": ("ple", "hard", {"threshold": 6.0}),
    "a4-cosine-diff": ("cosine_diff", "hard", {}),
    "a4-savgol-cal": ("savgol", "hard", {}),
    # P12 oracle ceiling — global optimal segmentation, inference-only on the
    # none anchor (manifest passes the anchor ckpt; never trained).
    "oracle-dp": ("dp_seg", "seg_avg", {}),
}

B_DETECTORS = ["raw", "savgol", "ple", "cosine_diff", "cosine_diff_savgol"]
B_MIXERS = ["hard", "seg_avg", "learned"]


def make_config(template: dict, name: str, detector: dict, mixer: dict,
                keep_rate, exp_dir: str) -> dict:
    cfg = copy.deepcopy(template)
    comp = cfg.setdefault("model", {}).setdefault("compression", {})
    comp["detector"] = detector
    comp["mixer"] = mixer
    comp["keep_rate"] = keep_rate
    cfg.setdefault("paths", {})["exp_dir"] = exp_dir
    # Standing decision (2026-10-01): all training runs use 50 epochs
    # (template had 120; val-best lands well before that anyway).
    cfg.setdefault("training", {})["epochs"] = 50
    return cfg


def write(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=0.30,
                        help="Equal-rate keep_rate for all generated configs "
                             "(set from A0-3 measured savgol rate).")
    parser.add_argument("--out-dir", default=str(REPO / "configs" / "exp"))
    parser.add_argument("--exp-root", default=BASE_EXP_ROOT)
    parser.add_argument("--smoke", action="store_true",
                        help="Emit one 2-epoch dry-run config instead.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of names to generate.")
    args = parser.parse_args()

    with open(TEMPLATE) as f:
        template = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    only = set(args.only.split(",")) if args.only else None
    made = []

    if args.smoke:
        cfg = copy.deepcopy(template)
        cfg.setdefault("paths", {})["exp_dir"] = f"{args.exp_root}/exp_smoke"
        cfg.setdefault("training", {})["epochs"] = 2
        # Exercise the new-style detector/mixer path (not the legacy filter).
        comp = cfg.setdefault("model", {}).setdefault("compression", {})
        comp["detector"] = {"type": "savgol"}
        comp["mixer"] = {"type": "hard"}
        comp["keep_rate"] = args.rate
        # NOTE: *_checkpoint_interval are every_n_train_steps (train.py), so
        # the template's multi-thousand-step intervals would never fire in a
        # 2-epoch run. Force small step intervals so ckpts exist for eval.
        cfg.setdefault("trainer", {})["checkpoint"] = {
            "val_checkpoint_interval": 500,
            "train_checkpoint_interval": 500,
            "save_top_k": 1,
            "save_weights_only": True,
        }
        write(cfg, out_dir / "smoke-detector.yaml")
        return 0

    for name, (det, mix, params) in A4_BASELINES.items():
        if only and name not in only:
            continue
        detector = {"type": det, "params": params}
        mixer = {"type": mix}
        if det == "ple":
            mixer["mode"] = "diff5"
        cfg = make_config(template, name, detector, mixer, args.rate,
                          f"{args.exp_root}/exp_a4/{name}")
        write(cfg, out_dir / f"{name}.yaml")
        made.append(name)

    for det in B_DETECTORS:
        for mix in B_MIXERS:
            name = f"b-{det}-{mix}"
            if only and name not in only:
                continue
            # Plan §Stage1: A4 AND B are all locked to the same measured
            # keep_rate=r for a controlled detector×mixer comparison.
            cfg = make_config(template, name, {"type": det}, {"type": mix},
                              args.rate, f"{args.exp_root}/exp_b/{name}")
            write(cfg, out_dir / f"{name}.yaml")
            made.append(name)

    print(f"\n{len(made)} configs in {out_dir} (rate={args.rate})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
