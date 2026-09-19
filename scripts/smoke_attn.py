"""PredictorAttn (v2) 冒烟测试。

覆盖 v2 修复的关键不变量：
  1. 全序列前向 (forward) 与逐 token 增量 (predict + KV cache) 数值等价
     —— 含 left_context 截断区间 (U > left_context)。
  2. 绝对位置 P 正确随 state 累积（KV 被截断时 P 仍 = 已消费 token 数）。
  3. 位置编码可训练（梯度到达 pos embedding）。
  4. torch.jit.script(predict) 可编译（RNNTBeamSearch 契约）。

用法: python scripts/smoke_attn.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformer_rnnt.predictor_attn import PredictorAttn  # noqa: E402

V = 74
torch.manual_seed(0)


def build(left_context: int) -> PredictorAttn:
    torch.manual_seed(42)
    return PredictorAttn(
        num_symbols=V, d_model=64, num_layers=2, num_heads=4,
        ffn_dim=128, left_context=left_context, dropout=0.0, max_len=64,
    ).eval()


def incremental(model: PredictorAttn, tokens: torch.Tensor):
    state = None
    outs = []
    for u in range(tokens.size(1)):
        out, _, state = model.predict(tokens[:, u : u + 1], None, state)
        outs.append(out)
    return torch.cat(outs, dim=1), state


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail and not cond else ''}")
    return cond


ok = True

# ── 1/2. 等价性: 正常区间 + 截断区间 ──
for lc, U in [(16, 10), (8, 20)]:
    m = build(lc)
    tokens = torch.randint(0, V, (2, U))
    with torch.no_grad():
        full, _, _ = m(tokens, None, None)
        inc, state = incremental(m, tokens)
    diff = (full - inc).abs().max().item()
    ok &= check(f"等价性 full==incremental (left_context={lc}, U={U})", diff < 1e-5, f"max|Δ|={diff:.2e}")
    # P 追踪: 即使 KV 被截断，绝对位置仍 = U
    p_val = int(state[0][2].item())
    ok &= check(f"绝对位置 P=={U} (left_context={lc})", p_val == U, f"P={p_val}")
    kv_len = state[0][0].size(2)
    ok &= check(f"KV 截断到 ≤ left_context (kv_len={kv_len} ≤ {lc})", kv_len <= lc)

# ── 3. 位置编码可训练 ──
m = build(16)
tokens = torch.randint(0, V, (2, 12))
out, _, _ = m(tokens, None, None)
out.sum().backward()
g = m.pos.weight.grad
ok &= check("梯度到达位置编码 (pos.weight.grad 非空且非零)",
            g is not None and g.abs().sum().item() > 0)

# ── 4. JIT 可脚本化 + 脚本化 predict 与 eager predict 数值一致 ──
# 注意：TorchScript 下 lengths 必须是真实 Tensor（None 会报 undefined Tensor）；
# torchaudio RNNTBeamSearch 实际调用时总是传张量，故契约无碍。
try:
    scripted = torch.jit.script(m)
    lengths = torch.zeros(1, dtype=torch.int64)
    tokens = torch.randint(0, V, (1, 6))
    with torch.no_grad():
        state_s = None
        outs_s = []
        for u in range(6):
            out_s, _, state_s = scripted.predict(tokens[:, u : u + 1], lengths, state_s)
            outs_s.append(out_s)
        state_e = None
        outs_e = []
        for u in range(6):
            out_e, _, state_e = m.predict(tokens[:, u : u + 1], lengths, state_e)
            outs_e.append(out_e)
        diff = (torch.cat(outs_s) - torch.cat(outs_e)).abs().max().item()
    ok &= check("torch.jit.script(predict) 可编译且与 eager 一致", diff < 1e-5, f"max|Δ|={diff:.2e}")
except Exception as e:  # noqa: BLE001
    ok &= check("torch.jit.script(predict) 可编译且与 eager 一致", False, repr(e))

print("\nSMOKE_ATTN:", "ALL PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
