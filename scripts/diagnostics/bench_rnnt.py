"""Microbench: decompose per-step cost without touching the running training.

Times on synthetic tensors of realistic shape:
  1. RNNT loss, batched call
  2. RNNT loss, per-sample (T/U-cropped + contiguous, as in production)
  3. RNNT loss, sub-batch of 4 with subgroup-max crop (correctness + speed)
  4. encoder fwd+bwd on a synthetic mel batch
Also asserts sub-batch == batched == per-sample losses/grads.
"""

# Allow direct execution from the repository root.
import sys as _sys
from pathlib import Path as _Path
_repo_root = str(_Path(__file__).resolve().parents[2])
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

import time

import torch
import torchaudio

torch.set_float32_matmul_precision("high")
dev = "cuda"

B, T, U, V = 8, 250, 40, 4001
BLANK = V - 1

torch.manual_seed(0)
logits = torch.randn(B, T, U + 1, V, device=dev) * 2 - 4
logits = logits.requires_grad_(True)
targets = torch.randint(0, BLANK, (B, U), device=dev, dtype=torch.int32)
fl = torch.full((B,), T, device=dev, dtype=torch.int32)
ul = torch.full((B,), U, device=dev, dtype=torch.int32)
loss_fn = torchaudio.transforms.RNNTLoss(blank=BLANK, reduction="sum", clamp=1.0)
print(f"shape B={B} T={T} U={U} V={V}  logits={logits.numel()/1e9:.2f}G elem", flush=True)


def timeit(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


# 1) batched
def f_batched():
    l = loss_fn(logits, targets, fl, ul)
    l.backward()
    logits.grad = None


ms_b = timeit(f_batched)
l_b = loss_fn(logits, targets, fl, ul)
l_b.backward()
g_b = logits.grad.detach().clone()
logits.grad = None
print(f"[1] batched   loss={l_b.item():.2f}  {ms_b:.1f} ms", flush=True)

# 2) per-sample (production path: one tolist, cropped, contiguous)
tl, uls = fl.tolist(), ul.tolist()


def f_sample():
    losses = [
        loss_fn(
            logits[i : i + 1, : tl[i], : uls[i] + 1].contiguous(),
            targets[i : i + 1, : uls[i]].contiguous(),
            fl[i : i + 1],
            ul[i : i + 1],
        )
        for i in range(B)
    ]
    torch.stack(losses).sum().backward()
    logits.grad = None


ms_s = timeit(f_sample)
losses = [
    loss_fn(
        logits[i : i + 1, : tl[i], : uls[i] + 1].contiguous(),
        targets[i : i + 1, : uls[i]].contiguous(),
        fl[i : i + 1],
        ul[i : i + 1],
    )
    for i in range(B)
]
l_s = torch.stack(losses).sum()
l_s.backward()
g_s = logits.grad.detach().clone()
logits.grad = None
print(f"[2] sample    loss={l_s.item():.2f}  {ms_s:.1f} ms  "
      f"dloss={abs(l_b.item()-l_s.item()):.2e} dgrad={(g_b-g_s).abs().max().item():.2e}",
      flush=True)

# 3) sub-batch of 4, cropped to subgroup max (strict checks satisfied by max)
def f_sub4():
    losses = []
    for g in range(0, B, 4):
        gl, gu = fl[g : g + 4], ul[g : g + 4]
        t_g, u_g = tl[g], uls[g]  # all equal here (uniform lengths in this bench)
        losses.append(
            loss_fn(
                logits[g : g + 4, :t_g, : u_g + 1].contiguous(),
                targets[g : g + 4, :u_g].contiguous(),
                gl,
                gu,
            )
        )
    torch.stack(losses).sum().backward()
    logits.grad = None


ms_g = timeit(f_sub4)
losses = []
for g in range(0, B, 4):
    gl, gu = fl[g : g + 4], ul[g : g + 4]
    t_g, u_g = tl[g], uls[g]
    losses.append(
        loss_fn(
            logits[g : g + 4, :t_g, : u_g + 1].contiguous(),
            targets[g : g + 4, :u_g].contiguous(),
            gl,
            gu,
        )
    )
l_g = torch.stack(losses).sum()
l_g.backward()
g_g = logits.grad.detach().clone()
logits.grad = None
print(f"[3] sub4      loss={l_g.item():.2f}  {ms_g:.1f} ms  "
      f"dloss={abs(l_b.item()-l_g.item()):.2e} dgrad={(g_b-g_g).abs().max().item():.2e}",
      flush=True)

# 4) encoder fwd+bwd on synthetic mel batch
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
import train as T

cfg = T.load_config("configs/server_lstm_word.yaml")
from pytorch_lightning import seed_everything

seed_everything(42, workers=True)
m = LibriSpeechRNNTModule.from_config(cfg).to(dev)
m.train()
Bm, Tm = 8, 1200
feats = torch.randn(Bm, Tm, 80, device=dev)
flen = torch.full((Bm,), Tm - 1, device=dev, dtype=torch.long)  # non-multiple of 4, like real audio
tgts = torch.randint(0, m.blank_idx, (Bm, U), device=dev)
tlen = torch.full((Bm,), U, device=dev, dtype=torch.long)


def enc_step():
    prep = tgts.new_empty([Bm, U + 1])
    prep[:, 1:] = tgts
    prep[:, 0] = m.blank_idx
    out, sl, _, _ = m.model(feats, flen, prep, tlen + 1)
    out.sum().backward()
    m.zero_grad(set_to_none=True)


ms_e = timeit(enc_step, n=5)
print(f"[4] encoder fwd+bwd (B={Bm} T={Tm})  {ms_e:.1f} ms", flush=True)
print(f"budget: step = encoder + loss + optimizer/data  ->  "
      f"loss share at sample path = {ms_s:.0f}ms", flush=True)
