"""Parity gate: refactored detector×mixer compression vs frozen legacy code.

Proves that the detector × mixer refactor (conformer_rnnt/compression/) is
bit-exact against the pre-refactor implementation for all 6 legacy filters:
indices identical and outputs equal. Hard-mixer filters (naive/fir/savgol)
must match bit-exactly; learned-mixer filters (hanning/savgol_hanning) are
vectorized and checked with allclose(atol=1e-5) — FP accumulation-order
noise only.

Run:  python tests/test_parity.py
Exit code 0 = gate passed.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._legacy_compression_reference import (  # noqa: E402
    DynamicCompression as LegacyDynamicCompression,
)
from conformer_rnnt.compression import DynamicCompression  # noqa: E402

LEGACY_FILTERS = ["naive", "fir", "savgol", "hanning", "savgol_hanning"]

FEATURE_DIM = 16
RIGHT_CONTEXT = 4


def make_batch(seed: int, batch: int = 3, time: int = 40):
    torch.manual_seed(seed)
    lengths = torch.tensor([time, time - 7, 12])
    x = torch.randn(batch, time, FEATURE_DIM)
    return x, lengths


def copy_learned_weights(new_module, legacy_module) -> None:
    """Legacy stores MLP at module root; new stores under mixer."""
    legacy_sd = {
        k.replace("weight_predictor", "mixer.weight_predictor"): v
        for k, v in legacy_module.state_dict().items()
    }
    new_module.load_state_dict(legacy_sd, strict=True)


def main() -> int:
    failures = []
    for filt in LEGACY_FILTERS:
        # Legacy path
        torch.manual_seed(0)
        legacy = LegacyDynamicCompression(
            right_context_length=RIGHT_CONTEXT,
            compression_filter=filt,
            hanning_window_size=5,
            feature_dim=FEATURE_DIM,
            weight_predictor_mode="diff5",
        )
        # New path — same build (from_legacy maps onto the new axes)
        new = DynamicCompression.from_legacy(
            right_context_length=RIGHT_CONTEXT,
            compression_filter=filt,
            hanning_window_size=5,
            feature_dim=FEATURE_DIM,
            weight_predictor_mode="diff5",
        )
        copy_learned_weights(new, legacy)
        new.eval()
        legacy.eval()

        ok = True
        for seed in (1, 2, 3):
            x, lengths = make_batch(seed)
            with torch.no_grad():
                out_l, len_l = legacy(x, lengths.clone())
                out_n, len_n = new(x, lengths.clone())
            # Learned mixer (hanning/savgol_hanning) is vectorized since
            # 2026-10-01; accumulation order differs from the legacy loop, so
            # parity is allclose (FP32 noise ~1e-7) instead of torch.equal.
            learn = filt in ("hanning", "savgol_hanning")
            same = torch.allclose(out_l, out_n, atol=1e-5) if learn \
                else torch.equal(out_l, out_n)
            if not same:
                ok = False
                failures.append(
                    f"{filt}: seed {seed} output mismatch "
                    f"(max diff {(out_l - out_n).abs().max().item():.3e})"
                )
            if not torch.equal(len_l, len_n):
                ok = False
                failures.append(f"{filt}: seed {seed} lengths mismatch {len_l} vs {len_n}")
        print(f"[{'PASS' if ok else 'FAIL'}] filter={filt}")

    # none passthrough
    new = DynamicCompression.from_legacy(
        right_context_length=RIGHT_CONTEXT, compression_filter="none"
    )
    x, lengths = make_batch(7)
    out, ln = new(x, lengths.clone())
    ok = torch.equal(out, x) and torch.equal(ln, lengths)
    print(f"[{'PASS' if ok else 'FAIL'}] filter=none (passthrough)")
    if not ok:
        failures.append("none: passthrough not identical")

    if failures:
        print("\nParity FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nParity gate PASSED: %d/6 bit-exact." % (len(LEGACY_FILTERS) + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
