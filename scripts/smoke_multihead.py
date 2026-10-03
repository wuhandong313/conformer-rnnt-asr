#!/usr/bin/env python3
"""Smoke gate: attn predictor trunk × Medusa 多头架构全链路 (CPU)。

覆盖 (attn-multihead 分支重构的回归门):
  1. zero-init 等价: ResBlock 零初始化 ⇒ 初始 head k logits == head 0 logits;
  2. shifted_ce_loss 对齐钉死: head k 在列 l 的 label 必须是 targets[l+k]
     (one-hot 构造; 对齐正确 ⇒ loss≈0, 偏移一位 ⇒ loss 显著大于 0) —
     该对齐是 spec_decode draft_before_emit 草拟时机的基础;
  3. aux 损失反传: forward_train_aux + shifted CE + KL, 梯度到达
     medusa res_blocks 与 attn predictor trunk;
  4. spec-decode 无损 + 对齐验收: 用「oracle 喂样」头 (draft 恒等于真实
     未来 token) 强制走满接受路径, 断言输出 == 贪心 oracle 且草拟轮数
     R < emit 数 M (R≈ceil(M/3) 说明每轮 2 个 draft 全被接受 — 若草拟
     时机偏移一位, d_1 恒失配, R≈M);
  5. config factory: predictor.type=attn + medusa.num_heads=2 正确组装;
     num_heads=0 时 medusa_heads_from_config 返回 None。

Run:  python scripts/smoke_multihead.py
Exit code 0 = all passed.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conformer_rnnt.medusa import (  # noqa: E402
    MedusaHeads,
    kl_to_base_loss,
    medusa_heads_from_config,
    shifted_ce_loss,
)
from conformer_rnnt.model import conformer_rnnt_model, conformer_rnnt_model_from_config  # noqa: E402
from conformer_rnnt.spec_decode import greedy_decode, speculative_greedy_decode  # noqa: E402

BLANK = 74  # 与仓库约定一致: num_symbols=74 真实符号, blank=74 (挂末位)
# tiny 模型: 直接工厂调用 num_symbols=13 ⇒ joiner/medusa V=13, predictor
# embedding 14 行; blank 取 12, targets 仅用 0..11
BLANK_TINY = 12


# ═══════════════════════════════════════════════════════════════════════════
# 1. zero-init 等价
# ═══════════════════════════════════════════════════════════════════════════


def test_zero_init_equivalence() -> None:
    torch.manual_seed(0)
    heads = MedusaHeads(input_dim=64, num_vocab=12, num_heads=2)
    g = torch.randn(2, 7, 64)
    base, future = heads(g)
    assert base.shape == (2, 7, 12)
    assert len(future) == 2 and future[0].shape == (2, 7, 12)
    for k_logits in future:
        assert torch.equal(base, k_logits), (
            "零初始化 ResBlock 应为恒等 ⇒ head k logits == head 0 logits"
        )
    print("OK  zero-init equivalence")


# ═══════════════════════════════════════════════════════════════════════════
# 2. shifted_ce_loss 对齐钉死
# ═══════════════════════════════════════════════════════════════════════════


def test_shifted_ce_alignment() -> None:
    torch.manual_seed(1)
    B, U, V, L = 2, 6, 10, 7  # L = U+1 (prepended trunk 序列)
    margin = 100.0
    targets = torch.randint(0, V - 1, (B, U))  # 0..V-2, 不含 blank 语义

    def make_logits(offset: int) -> list:
        # head k (1-based) 在列 l 的正确 label = targets[l+k]; offset 模拟错位
        logits_future = []
        for k in (1, 2):
            lg = torch.full((B, L, V), -margin)
            for l in range(U - k):
                label = int(targets[0, min(l + k + offset, U - 1)])
                lg[0, l, label] = margin
                label1 = int(targets[1, min(l + k + offset, U - 1)])
                lg[1, l, label1] = margin
            logits_future.append(lg)
        return logits_future

    good = shifted_ce_loss(make_logits(0), targets, weights=[1.0, 1.0])
    bad = shifted_ce_loss(make_logits(1), targets, weights=[1.0, 1.0])
    assert good.item() < 1e-3, f"对齐正确时 loss 应≈0, got {good.item():.4f}"
    assert bad.item() > 1.0, f"偏移一位时 loss 应显著, got {bad.item():.4f}"
    print(f"OK  shifted_ce alignment (good={good.item():.2e}, bad={bad.item():.2f})")


# ═══════════════════════════════════════════════════════════════════════════
# 3. aux 损失反传
# ═══════════════════════════════════════════════════════════════════════════


def _tiny_model(medusa_num_heads: int = 2):
    torch.manual_seed(2)
    # 直接工厂调用不做 +1: num_symbols=13 ⇒ joiner V=13, predictor
    # embedding 14 行, medusa num_vocab=13 — blank id=12 三处一致
    return conformer_rnnt_model(
        input_dim=40,
        encoding_dim=64,
        num_symbols=13,
        right_context_length=4,
        time_reduction_input_dim=16,
        time_reduction_stride=4,
        num_heads=2,
        ffn_dim=64,
        num_layers_block1=1,
        num_layers_block2=1,
        dropout=0.0,
        depthwise_conv_kernel_size=7,
        compression_filter="none",
        symbol_embedding_dim=32,
        predictor_type="attn",
        attn_d_model=64,
        attn_num_layers=2,
        attn_num_heads=4,
        attn_ffn_dim=64,
        attn_left_context=128,
        attn_dropout=0.0,  # 解码确定性必需 (默认 0.1 会使逐次解码不一致)
        attn_max_len=512,
        joiner_activation="relu",
        medusa_num_heads=medusa_num_heads,
    )
    model.blank_idx = BLANK_TINY  # 供测试参考
    return model


def test_aux_loss_backward() -> None:
    model = _tiny_model()
    assert model.medusa_heads is not None
    # encoder 输入约定: (B, T+rc, D), lengths 不含 rc — 沿用 smoke_chain 的
    # (2, 80, 40) + [78, 62], 避开 x_len 为 stride 整数倍的 mask 边界 bug
    B, U = 2, 8
    features = torch.randn(B, 80, 40)
    feature_lengths = torch.tensor([78, 62])
    targets = torch.randint(0, BLANK_TINY, (B, U))
    # 复刻 lightning _step: trunk 吃 prepended targets ([blank, y_0..y_{U-1}])
    prepended = targets.new_empty(B, U + 1)
    prepended[:, 0] = BLANK_TINY
    prepended[:, 1:] = targets

    _, src_len, tgt_len, _, logits_base, logits_future = model.forward_train_aux(
        features, feature_lengths, prepended, torch.tensor([U + 1, U - 2 + 1])
    )
    assert logits_base is not None and len(logits_future) == 2
    # trunk 序列长 = U+1 (blank 前置)
    assert logits_base.shape[1] == U + 1, (
        f"forward_train_aux 应喂 prepended targets (L=U+1), got {logits_base.shape}"
    )

    ce = shifted_ce_loss(logits_future, targets, weights=[0.8, 0.64])
    kl = kl_to_base_loss(logits_future, logits_base, beta=0.05)
    loss = ce + kl
    assert torch.isfinite(loss), "aux loss 必须有限"
    loss.backward()

    rb0 = model.medusa_heads.res_blocks[0].linear2.weight.grad
    assert rb0 is not None and torch.isfinite(rb0).all(), "梯度未到达 medusa res_block"
    assert torch.count_nonzero(rb0) > 0, "res_block 梯度不应全零"
    trunk_grads = [
        p.grad for p in model.predictor.layers[0].parameters() if p.grad is not None
    ]
    assert trunk_grads, "梯度未到达 attn predictor trunk"
    print(f"OK  aux loss backward (ce={ce.item():.4f}, kl={kl.item():.4f})")


# ═══════════════════════════════════════════════════════════════════════════
# 4. spec-decode 无损 + oracle 喂样对齐验收
# ═══════════════════════════════════════════════════════════════════════════


def _bias_blank(model, value: float) -> None:
    """blank logit 加常量偏置。随机权重下贪心会陷入已知病理循环
    (from-scratch 模型 repeating, 见 eval_display MAX_EMITS_PER_FRAME 注),
    blank 主导后解码可自然终止 (t 走完所有帧), 同时保留少量 emit。"""
    with torch.no_grad():
        model.joiner.linear.bias[BLANK_TINY] += value


def _decode_setup(seed: int, blank_bias: float = 3.0):
    """构建 tiny 模型 + 特征, 返回 (model, features, feature_lengths)。"""
    torch.manual_seed(seed)
    model = _tiny_model()
    model.eval()  # 关 dropout, 保证解码逐次一致
    _bias_blank(model, blank_bias)
    features = torch.randn(1, 80, 40)
    feature_lengths = torch.tensor([78])
    return model, features, feature_lengths


def test_spec_decode_lossless_random() -> None:
    n_checked = 0
    for seed in range(3, 16):
        model, features, feature_lengths = _decode_setup(seed, blank_bias=1.0)
        oracle = greedy_decode(
            model, features, feature_lengths, BLANK_TINY, max_steps=20000
        )
        # 跳过: 无 emit (纯 blank) 或病理循环 (from-scratch 已知 repeating)
        if len(oracle) < 1 or len(oracle) > 2000:
            continue
        spec = speculative_greedy_decode(
            model, features, feature_lengths, BLANK_TINY,
            max_draft=2, per_token_blank=4, max_steps=20000,
        )
        assert oracle == spec, (
            f"seed={seed}: spec 输出必须逐 token 等于 oracle; "
            f"len oracle={len(oracle)} spec={len(spec)}"
        )
        n_checked += 1
    assert n_checked >= 2, f"有效对照 seed 不足 ({n_checked})"
    print(f"OK  spec-decode lossless (blank-biased random weights, {n_checked} seeds)")


def test_spec_decode_oracle_fed_alignment() -> None:
    """用 oracle 喂样头强制满接受: 验证草拟时机与 shifted_ce 对齐一致。

    包装 predict: 计数 predictor 已消费 token 数 c (blank 1 次 + 已消费
    label)。draft_before_emit 语义下, emit pred0 后、消费 pred0 前, 消费
    计数 = c, 此时已 emit s=c 个 label (含 pred0), 头 k 应给出 oracle[c+k-1]
    → d_j = oracle[c+j]。若实现偏移一位 (post-emit 草拟), d_1 = oracle[c]
    vs 实际下一 emit oracle[c-1] 恒失配 ⇒ 每轮只接受 0 个 draft ⇒ 轮数
    R≈M; 正确实现每轮接受 2 个 ⇒ R≈ceil(M/3) < M。
    """
    model, features, feature_lengths = _decode_setup(3, blank_bias=1.0)
    oracle = greedy_decode(
        model, features, feature_lengths, BLANK_TINY, max_steps=20000
    )
    M = len(oracle)
    assert 4 <= M <= 2000, f"oracle 序列长度异常 ({M})，换 seed 重试"

    # 包装 predict 计数 + predict_heads 喂 oracle 未来 token
    state = {"consumed": 0, "rounds": 0}
    orig_predict = model.predict

    def counting_predict(targets, target_lengths, predictor_state=None):
        state["consumed"] += targets.size(1)
        return orig_predict(targets, target_lengths, predictor_state)

    model.predict = counting_predict

    def oracle_fed_heads(g):
        state["rounds"] += 1
        c = state["consumed"]  # blank(1) + 已消费 label 数; 已 emit s=c 个 (含 pred0 未消费)
        V = 13
        drafts = []
        for j in (1, 2):  # d_j = oracle[c+j-1]
            idx = c + j - 1
            lg = torch.full((1, 1, V), -100.0)
            if idx < M:
                lg[0, 0, oracle[idx]] = 100.0
            else:
                lg[0, 0, BLANK_TINY] = 100.0  # 弃权
            drafts.append(lg)
        return None, drafts

    model.predict_heads = oracle_fed_heads

    spec = speculative_greedy_decode(
        model, features, feature_lengths, BLANK_TINY,
        max_draft=2, per_token_blank=4, max_steps=20000,
    )
    assert spec == oracle, "oracle 喂样下输出必须等于 oracle"
    R = state["rounds"]
    assert R < M, (
        f"草拟轮数 R={R} 未小于 emit 数 M={M} — 草拟时机偏移一位 "
        f"(post-emit), d_1 恒失配, 接受路径未被走通"
    )
    print(f"OK  spec-decode oracle-fed alignment (M={M}, rounds R={R}, "
          f"R≈ceil(M/3)={-(-M // 3)})")


# ═══════════════════════════════════════════════════════════════════════════
# 5. config factory
# ═══════════════════════════════════════════════════════════════════════════


def test_config_factory() -> None:
    cfg = {
        "model": {
            "input_dim": 40,
            "encoding_dim": 64,
            "num_symbols": 12,
            "predictor": {
                "type": "attn",
                "symbol_embedding_dim": 32,
                "attn": {
                    "d_model": 64,
                    "num_layers": 2,
                    "num_heads": 4,
                    "ffn_dim": 64,
                    "left_context": 128,
                    "max_len": 512,
                },
            },
            "medusa": {"num_heads": 2},
        }
    }
    model = conformer_rnnt_model_from_config(cfg)
    assert model.medusa_heads is not None, "medusa.num_heads=2 应组装 MedusaHeads"
    assert model.medusa_heads.num_heads == 2
    assert len(model.predictor.layers) == 2  # attn trunk 生效
    assert type(model.predictor).__name__ == "PredictorAttn"

    cfg0 = {"model": {"medusa": {"num_heads": 0}}}
    assert medusa_heads_from_config(cfg0) is None, "num_heads=0 应返回 None (关闭)"
    print("OK  config factory (attn + medusa=2)")


def main() -> int:
    test_zero_init_equivalence()
    test_shifted_ce_alignment()
    test_aux_loss_backward()
    test_spec_decode_lossless_random()
    test_spec_decode_oracle_fed_alignment()
    test_config_factory()
    print("ALL SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
