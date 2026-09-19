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
    # P 追踪: 即使 KV 是循环定长, 绝对位置仍 = U
    p_val = state[0][2].max().item()
    ok &= check(f"绝对位置 P=={U} (left_context={lc})", p_val == U, f"P={p_val}")
    kv_len = state[0][0].size(2)
    ok &= check(f"KV cache 定长 == left_context (kv_len={kv_len})", kv_len == lc)

# ── 2b. beam-search 批处理契约: 不同 hypo 消费不同 token 数后可 cat/index_select ──
# 模拟 torchaudio RNNTBeamSearch: 同帧内 a-hypo 续扩, hypo0 消费 6 token、
# hypo1 消费 4 token, 各自 state cat 成 batch → predict → index_select 切片,
# 且切片后单独 predict 的输出与 batch 内该行输出一致。
m = build(16)
with torch.no_grad():
    lengths = torch.ones(1, dtype=torch.long)
    # hypo0: 6 步
    s_a = None
    for _ in range(6):
        _, _, s_a = m.predict(torch.randint(0, V, (1, 1)), lengths, s_a)
    # hypo1: 4 步
    s_b = None
    for _ in range(4):
        _, _, s_b = m.predict(torch.randint(0, V, (1, 1)), lengths, s_b)
    nxt = torch.randint(0, V, (2, 1))
    # cat 成 batch (torchaudio _batch_state 语义)
    batched = [[torch.cat([s_a[li][j], s_b[li][j]], dim=0) for j in range(3)]
               for li in range(len(s_a))]
    out_b, _, s_out = m.predict(nxt, torch.ones(2, dtype=torch.long), batched)
    ok &= check("混合 token 数 hypo 批处理: P 逐行 [7,5]",
                s_out[0][2].tolist() == [7, 5], f"P={s_out[0][2].tolist()}")
    ok &= check("混合 token 数 hypo 批处理: cache 定长 (B=2, L=16)",
                s_out[0][0].size(0) == 2 and s_out[0][0].size(2) == 16)
    # 逐行一致性: 单独喂各 hypo 的 state 应得到与 batch 中该行相同的输出
    out_a, _, _ = m.predict(nxt[0:1], lengths, s_a)
    out_c, _, _ = m.predict(nxt[1:2], lengths, s_b)
    d0 = (out_b[0:1] - out_a).abs().max().item()
    d1 = (out_b[1:2] - out_c).abs().max().item()
    ok &= check("批处理输出与逐 hypo 输出一致", max(d0, d1) < 1e-5, f"Δ={max(d0, d1):.2e}")
    # _slice_state 语义: index_select 切出第 1 行后继续 predict
    sliced = [[t.index_select(0, torch.tensor([1])) for t in layer] for layer in s_out]
    out_s, _, s_s = m.predict(torch.randint(0, V, (1, 1)), lengths, sliced)
    ok &= check("index_select 切片后 predict 正常且 P=6",
                s_s[0][2].tolist() == [6] and out_s.shape == (1, 1, 512))

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

# ── 5. 超 max_len 训练路径显式报错（不允许静默 clamp） ──
m = build(16)  # max_len=64
tokens = torch.randint(0, V, (1, 100))
try:
    with torch.no_grad():
        m(tokens, None, None)
    ok &= check("forward 超 max_len 显式报错", False, "未触发 _assert")
except (AssertionError, RuntimeError):
    ok &= check("forward 超 max_len 显式报错", True)

print("\nSMOKE_ATTN:", "ALL PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
