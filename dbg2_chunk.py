"""dbg2 v2 — verify per-sample cropped RNNT loss, then step 170 train batches.

phase0: batched loss vs per-sample T/U-cropped loss on one real batch
        (frozen logits in eval mode): loss + dLoss/dlogits must match.
phase1/2: real m._step forward+backward through 170 batches (covers the
        ~step-98 crash point of the batched kernel).
Run with CUDA_LAUNCH_BLOCKING=1 so any failure localizes exactly.
"""
import time

import torch
from pytorch_lightning import seed_everything

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
import train as T

cfg = T.load_config("configs/server_lstm_word.yaml")
seed_everything(cfg.get("seed", 42), workers=True)
m = LibriSpeechRNNTModule.from_config(cfg).to("cuda")
print(f"[setup] rnnt_loss_chunk = {m.rnnt_loss_chunk!r}", flush=True)
assert m.rnnt_loss_chunk == "sample", "config did not enable sample chunking!"

loader = m.train_dataloader()
it = iter(loader)

# ---------------------------------------------------------------- phase 0
batch = next(it)
assert batch is not None
m.eval()  # freeze dropout for a deterministic forward
with torch.no_grad():
    prepended = batch.targets.new_empty(
        [batch.targets.size(0), batch.targets.size(1) + 1]
    )
    prepended[:, 1:] = batch.targets
    prepended[:, 0] = m.blank_idx
    output, src_lengths, _, _ = m.model(
        batch.features.cuda(),
        batch.feature_lengths.cuda(),
        prepended.cuda(),
        (batch.target_lengths + 1).cuda(),
    )
max_time = output.size(1)
src_lengths = torch.clamp(src_lengths, max=max_time).to(torch.int32)
target_lengths = batch.target_lengths.cuda().to(torch.int32)
targets = batch.targets.cuda()
print(f"[phase0] batch B={output.size(0)} T={output.size(1)} "
      f"U_max={int(target_lengths.max())} V={output.size(3)}", flush=True)

out_a = output.detach().clone().requires_grad_(True)
loss_a = m.loss(out_a, targets, src_lengths, target_lengths)
loss_a.backward()

out_b = output.detach().clone().requires_grad_(True)
losses = []
for i in range(out_b.size(0)):
    t_i = int(src_lengths[i])
    u_i = int(target_lengths[i])
    losses.append(
        m.loss(
            out_b[i : i + 1, :t_i, : u_i + 1].contiguous(),
            targets[i : i + 1, :u_i].contiguous(),
            src_lengths[i : i + 1],
            target_lengths[i : i + 1],
        )
    )
loss_b = torch.stack(losses).sum()
loss_b.backward()

d_loss = abs(loss_a.item() - loss_b.item())
rel = d_loss / max(1.0, abs(loss_a.item()))
d_grad = (out_a.grad - out_b.grad).abs().max().item()
print(f"[phase0] batched={loss_a.item():.4f} sample={loss_b.item():.4f} "
      f"|dloss|={d_loss:.3e} rel={rel:.3e} |dgrad|max={d_grad:.3e}", flush=True)
assert rel < 1e-3, f"loss mismatch: rel={rel:.3e}"
assert d_grad < 1e-2, f"grad mismatch: {d_grad:.3e}"
print("[phase0] EQUIVALENT", flush=True)
m.train()

# ------------------------------------------------------------ phase 1 + 2
N_STEPS = 170
t0 = time.time()
done = 0
for step, batch in enumerate(it):
    if done >= N_STEPS:
        break
    if batch is None:
        continue
    batch = batch._replace(
        features=batch.features.cuda(),
        feature_lengths=batch.feature_lengths.cuda(),
        targets=batch.targets.cuda(),
        target_lengths=batch.target_lengths.cuda(),
    )
    loss = m._step(batch, done, "train")
    m.zero_grad(set_to_none=True)
    loss.backward()
    done += 1
    if done == 1 or done % 20 == 0:
        mem = torch.cuda.max_memory_allocated() / 2**30
        rate = done / (time.time() - t0)
        print(f"[step {done:3d}] loss={loss.item():9.2f} peak={mem:.1f}G "
              f"{rate:.1f} it/s elapsed={time.time() - t0:.0f}s", flush=True)

print(f"[dbg2] ALL OK — {done} _step fwd+bwd passed, "
      f"{time.time() - t0:.0f}s total", flush=True)
