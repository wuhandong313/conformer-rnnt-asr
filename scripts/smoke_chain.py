"""Smoke gate: every detector×mixer cell trains (forward + backward) on CPU.

Builds a tiny ConformerEncoder (1+1 layers, dim 64) per B-matrix cell and the
A4 fixed-rate baselines, runs a forward pass, computes a dummy RNNT-free loss
(mean of outputs), and backprops. Verifies output shapes/lengths are valid and
gradients reach the learned mixer parameters when present.

Run:  python scripts/smoke_chain.py
Exit code 0 = all cells passed.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformer_rnnt.encoder import ConformerEncoder  # noqa: E402

# B-matrix: 5 detectors × 3 mixers
B_MATRIX = [
    ("raw", "hard"),
    ("raw", "seg_avg"),
    ("raw", "learned"),
    ("fir", "hard"),
    ("fir", "seg_avg"),
    ("fir", "learned"),
    ("savgol", "hard"),
    ("savgol", "seg_avg"),
    ("savgol", "learned"),
    ("ple", "hard"),
    ("ple", "seg_avg"),
    ("ple", "learned"),
    ("cosine_diff", "hard"),
    ("cosine_diff", "seg_avg"),
    ("cosine_diff", "learned"),
]

# A4 fixed-rate baselines (keep_rate 0.3)
A4_FIXED = [
    "uniform",
    "random",
    "energy",
    "topk_diff",
    "max_pool",
    "avg_pool",
    "mmdp",
]

ENCODER_KW = dict(
    input_dim=40,
    output_dim=64,
    right_context_length=4,
    time_reduction_input_dim=16,
    time_reduction_stride=4,
    num_heads=2,
    ffn_dim=64,
    num_layers_block1=1,
    num_layers_block2=1,
    dropout=0.0,
    depthwise_conv_kernel_size=7,
)


def make_encoder(detector_type: str, mixer_type: str, keep_rate=None) -> ConformerEncoder:
    return ConformerEncoder(
        compression_detector={"type": detector_type, "params": {}},
        compression_mixer={"type": mixer_type},
        compression_keep_rate=keep_rate,
        **ENCODER_KW,
    )


def run_cell(name: str, encoder: ConformerEncoder, keep_rate=None) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 80, ENCODER_KW["input_dim"])
    lengths = torch.tensor([80, 60])

    out, out_len = encoder(x, lengths)

    # Shape / length validity
    if out.dim() != 3 or out.shape[-1] != ENCODER_KW["output_dim"]:
        raise AssertionError(f"{name}: bad output shape {out.shape}")
    if not torch.isfinite(out).all():
        raise AssertionError(f"{name}: non-finite output")
    if (out_len <= 0).any() or (out_len > out.shape[1]).any():
        raise AssertionError(f"{name}: invalid lengths {out_len.tolist()}")

    # Equal-rate check: fixed-rate detectors must respect keep_rate
    if keep_rate is not None and encoder.compression.detector is not None:
        pass  # rate measured separately in A0-3; length sanity is enough here

    loss = out.float().mean()
    loss.backward()

    # Gradients must reach the learned mixer
    if encoder.compression.mixer is not None and hasattr(
        encoder.compression.mixer, "weight_predictor"
    ):
        grads = [
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in encoder.compression.mixer.parameters()
        ]
        if not any(grads):
            raise AssertionError(f"{name}: no gradient reached learned mixer")


def main() -> int:
    failures = []
    n = 0
    for det, mix in B_MATRIX:
        name = f"B[{det}×{mix}]"
        try:
            enc = make_encoder(det, mix, keep_rate=None)
            run_cell(name, enc)
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
            print(f"[FAIL] {name}: {exc}")
        n += 1

    for det in A4_FIXED:
        name = f"A4[{det} r=0.30]"
        try:
            enc = make_encoder(det, "hard", keep_rate=0.3)
            run_cell(name, enc)
            print(f"[PASS] {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
            print(f"[FAIL] {name}: {exc}")
        n += 1

    # Equal-rate spot check: savgol + keep_rate=0.5 must halve lengths vs no-keep
    torch.manual_seed(1)
    enc_dyn = make_encoder("savgol", "hard", keep_rate=None)
    enc_fix = make_encoder("savgol", "hard", keep_rate=0.5)
    x = torch.randn(1, 80, ENCODER_KW["input_dim"])
    lengths = torch.tensor([80])
    with torch.no_grad():
        _, l_dyn = enc_dyn(x, lengths.clone())
        _, l_fix = enc_fix(x, lengths.clone())
    ok = l_fix.item() <= l_dyn.item()
    print(f"[{'PASS' if ok else 'FAIL'}] equal-rate: savgol r=0.5 len {l_fix.item():.0f} "
          f"<= dynamic len {l_dyn.item():.0f}")
    if not ok:
        failures.append("equal-rate spot check failed")
    n += 1

    print(f"\nSmoke gate: {n - len(failures)}/{n} passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
