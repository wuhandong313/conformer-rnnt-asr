#!/usr/bin/env python3
"""精确定位 PredictorAttn.predict() 与 forward() 的发散点。

三个对比 (全部 eval 模式, 无 dropout 干扰):
  A. predict() 逐步 (state 走 predict 返回的 state)  vs  forward() 全序列
  B. 手动逐步 (_embed + forward_inc + wrapper, 自管 state)  vs  forward() 全序列
  C. A vs B  (predict 内部 vs 手动, 同一 state 链)
若 B==full 而 A!=full, bug 在 predict() 本身; 若 B!=full, bug 在层/嵌入。
"""

import sys

import torch

sys.path.insert(0, "/test21/conformer-rnnt-asr-attn")

from conformer_rnnt.predictor_attn import PredictorAttn


def run_case(name: str, pa: PredictorAttn, tokens: torch.Tensor) -> None:
    pa.eval()
    B, U = tokens.shape
    device = tokens.device
    lengths = torch.full((B,), U, dtype=torch.long, device=device)

    with torch.no_grad():
        out_full, _, _ = pa(tokens, lengths)  # (B,U,out_dim)

        # A: predict 逐步
        outs_a = []
        state = None
        for u in range(U):
            o, _, state = pa.predict(tokens[:, u : u + 1], lengths, state)
            outs_a.append(o[:, 0, :])
        A = torch.stack(outs_a, dim=1)  # (B,U,out_dim)

        # B: 手动逐步 (_embed + forward_inc + wrapper)
        outs_b = []
        caches = [
            [torch.zeros(B, pa.num_heads, 0, pa.head_dim, device=device),
             torch.zeros(B, pa.num_heads, 0, pa.head_dim, device=device)]
            for _ in pa.layers
        ]
        pos = 0
        for u in range(U):
            x = pa._embed(tokens[:, u : u + 1], pos)
            new_caches = []
            for li, layer in enumerate(pa.layers):
                x, nk, nv = layer.forward_inc(x, caches[li][0], caches[li][1])
                new_caches.append([nk, nv])
            caches = new_caches
            pos += 1
            x = pa.dropout(pa.final_norm(x))
            o = pa.output_layer_norm(pa.linear(x))
            outs_b.append(o[:, 0, :])
        Bm = torch.stack(outs_b, dim=1)

    dA = (A - out_full).abs().max().item()
    dB = (Bm - out_full).abs().max().item()
    dAB = (A - Bm).abs().max().item()
    print(f"[{name}] U={U}  A(predict)vs full: {dA:.6f}   "
          f"B(manual)vs full: {dB:.6f}   A vs B: {dAB:.6f}")
    # 找 A 首个发散位置
    if dA > 1e-4:
        posdiff = (A - out_full).abs().amax(dim=(0, 2))
        first = int((posdiff > 1e-4).nonzero()[0])
        print(f"    A 首个发散位置 u={first}, 逐位: "
              + " ".join(f"{v:.2e}" for v in posdiff.tolist()[:12]))


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    B, U = 2, 32
    tokens = torch.randint(0, 74, (B, U), device=device)
    tokens[:, 0] = 74  # blank 开头, 与解码约定一致

    # 小窗 (truncation 生效)
    pa_small = PredictorAttn(num_symbols=74, symbol_embedding_dim=16,
                             d_model=32, num_layers=2, num_heads=4,
                             ffn_dim=64, left_context=8, dropout=0.0)
    # 大窗 (left_context=1024 > U, 不截断 — 与训练/解码实际配置一致)
    pa_big = PredictorAttn(num_symbols=74, symbol_embedding_dim=16,
                           d_model=32, num_layers=2, num_heads=4,
                           ffn_dim=64, left_context=1024, dropout=0.0)

    run_case("small-win lc=8", pa_small, tokens)
    run_case("big-win   lc=1024", pa_big, tokens)

    # 长序列 (> left_context) 的截断正确性
    tokens_long = torch.randint(0, 74, (1, 2000), device=device)
    tokens_long[:, 0] = 74
    pa_lc = PredictorAttn(num_symbols=74, symbol_embedding_dim=16,
                          d_model=32, num_layers=2, num_heads=4,
                          ffn_dim=64, left_context=64, dropout=0.0)
    run_case("long-seq lc=64 U=2000", pa_lc, tokens_long)


if __name__ == "__main__":
    main()
