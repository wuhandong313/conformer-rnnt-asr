"""dbg3 v2 — soak-test rnnt_loss_chunk="group" WITHOUT stopping the running
training (which holds ~49G of the 80G GPU).

Memory discipline: every batch is trimmed to a leading prefix whose summed
per-sample numel (T_est x (U+1) x V) stays under GROUP_CAP — exactly the
op-call regime group mode produces, so the soak exercises the real packing
path within the ~30G free VRAM.

phase0: group loss/grad vs per-sample reference (proven safe) on 3 batches.
phase2: 2000 real m._step fwd+bwd in group mode.
"""

# Allow direct execution from the repository root.
import sys as _sys
from pathlib import Path as _Path
_repo_root = str(_Path(__file__).resolve().parents[2])
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

import os
import time

import torch
from pytorch_lightning import seed_everything

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
import train as T

GROUP_CAP = 2.0e8  # per-batch trimmed budget (< group mode's 2.5e8 call cap)
V = 4001

cfg = T.load_config("configs/server_lstm_word.yaml")
seed_everything(cfg.get("seed", 42), workers=True)
m = LibriSpeechRNNTModule.from_config(cfg).to("cuda")
print(f"[setup] rnnt_loss_chunk = {m.rnnt_loss_chunk!r}", flush=True)
assert m.rnnt_loss_chunk == "group", "config did not enable group chunking!"


def trim(batch):
    """Keep the leading prefix of samples under GROUP_CAP summed numel."""
    fl = batch.feature_lengths.tolist()  # input frames
    ul = batch.target_lengths.tolist()
    keep, total = 0, 0.0
    for f, u in zip(fl, ul):
        t_est = f // 4 + 2  # frames/4 + right-context, +margin
        n = t_est * (u + 1) * V
        if keep > 0 and total + n > GROUP_CAP:
            break
        if keep == 0 and n * 2 > GROUP_CAP:  # single sample twice over cap
            return None
        total += n
        keep += 1
    if keep == 0:
        return None
    # restore the production invariant padded T dim == max(feature_lengths) + 4
    # (encoder right_context_length=4: input is (B, T + rc, D), lengths exclude rc)
    max_fl = max(batch.feature_lengths[:keep].tolist()) + 4
    if batch.features.size(2) == 80:
        feats = batch.features[:keep, :max_fl]
    else:
        feats = batch.features[:keep, :, :max_fl]
    return batch._replace(
        features=feats,
        feature_lengths=batch.feature_lengths[:keep],
        targets=batch.targets[:keep],
        target_lengths=batch.target_lengths[:keep],
    )


def to_cuda(batch):
    return batch._replace(
        features=batch.features.cuda(),
        feature_lengths=batch.feature_lengths.cuda(),
        targets=batch.targets.cuda(),
        target_lengths=batch.target_lengths.cuda(),
    )


loader = m.train_dataloader()
it = iter(loader)

# ---------------------------------------------------------------- phase 0
m.eval()
done0 = 0
while done0 < 3:
    raw = next(it)
    if raw is None:
        continue
    batch = trim(raw)
    if batch is None:
        continue
    batch = to_cuda(batch)
    with torch.no_grad():
        prepended = batch.targets.new_empty(
            [batch.targets.size(0), batch.targets.size(1) + 1]
        )
        prepended[:, 1:] = batch.targets
        prepended[:, 0] = m.blank_idx
        output, src_lengths, _, _ = m.model(
            batch.features,
            batch.feature_lengths,
            prepended,
            batch.target_lengths + 1,
        )
    max_time = output.size(1)
    sl = torch.clamp(src_lengths, max=max_time).to(torch.int32)
    tl = batch.target_lengths.to(torch.int32)
    tg = batch.targets

    # reference: per-sample calls (proven bit-exact vs batched, crash-free)
    out_a = output.detach().clone().requires_grad_(True)
    sl_l, tl_l = sl.tolist(), tl.tolist()
    ref = torch.stack(
        [
            m.loss(
                out_a[i : i + 1, : sl_l[i], : tl_l[i] + 1].contiguous(),
                tg[i : i + 1, : tl_l[i]].contiguous(),
                sl[i : i + 1],
                tl[i : i + 1],
            )
            for i in range(out_a.size(0))
        ]
    ).sum()
    ref.backward()

    out_b = output.detach().clone().requires_grad_(True)
    got = m._rnnt_loss(out_b, tg, sl, tl)
    got.backward()

    rel = abs(ref.item() - got.item()) / max(1.0, abs(ref.item()))
    dg = (out_a.grad - out_b.grad).abs().max().item()
    print(f"[phase0.{done0}] B={output.size(0)} ref={ref.item():.2f} "
          f"group={got.item():.2f} rel={rel:.2e} dgrad={dg:.2e}", flush=True)
    assert rel < 1e-3 and dg < 1e-2, f"mismatch at batch {done0}"
    done0 += 1
print("[phase0] EQUIVALENT", flush=True)
m.train()

# ---------------------------------------------------------------- phase 2
stats = {"calls": 0, "groups": 0, "sizes": [], "skipped": 0}
_orig_pack = m._pack_groups


def _counted(ts, us, vocab):
    r = _orig_pack(ts, us, vocab)
    stats["calls"] += 1
    stats["groups"] += len(r)
    stats["sizes"].extend(hi - lo for lo, hi in r)
    return r


m._pack_groups = _counted

N_STEPS = int(os.environ.get("DBG3_STEPS", "2000"))
t0 = time.time()
done = 0
for raw in it:
    if done >= N_STEPS:
        break
    if raw is None:
        continue
    batch = trim(raw)
    if batch is None:
        stats["skipped"] += 1
        continue
    loss = m._step(to_cuda(batch), done, "train")
    m.zero_grad(set_to_none=True)
    loss.backward()
    done += 1
    if done == 1 or done % 200 == 0:
        mem = torch.cuda.max_memory_allocated() / 2**30
        rate = done / (time.time() - t0)
        print(f"[step {done:4d}] loss={loss.item():9.2f} peak={mem:.1f}G "
              f"{rate:.2f} it/s elapsed={time.time() - t0:.0f}s", flush=True)

n_calls = max(1, stats["calls"])
print(f"[group stats] batches={stats['calls']} skipped={stats['skipped']} "
      f"avg_groups/batch={stats['groups'] / n_calls:.1f} "
      f"avg_group_size={sum(stats['sizes']) / max(1, len(stats['sizes'])):.1f}",
      flush=True)
print(f"[dbg3] ALL OK — {done} group-mode steps passed, "
      f"{time.time() - t0:.0f}s ({done / (time.time() - t0):.2f} it/s, "
      f"contended with the running training)", flush=True)
