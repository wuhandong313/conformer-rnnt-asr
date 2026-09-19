"""受限因果自注意力 Predictor trunk (Family A — MSA-Transducer 风格)。

替代原 LSTM `Predictor`，作为 Medusa 多头的共享 backbone：
    Embedding + 位置编码 → LayerNorm → N 层 pre-norm 因果 Transformer(受限左上下文)
    → Dropout → final LayerNorm → Linear → LayerNorm
output_dim=512 契约不变，与 Joiner 加性融合兼容。

v2 修复（针对首轮训练失败的诊断，见探针实验）：
  1. 补 learned positional embedding——无位置编码的因果 attention 无法区分
     「最近一次出现的同 id token」，重复音素发射能力显著弱于 LSTM。
  2. post-norm → pre-norm + 出口 final_norm——post-norm 从零训练收敛慢且不稳。
  3. 绝对位置随 state 走：每层 state = [K, V, P]，P (batch,) = 已消费 token
     数。KV cache 定长 left_context（循环写槽位 p % left_context），
     保证 torchaudio beam-search 批处理 (cat/index_select) 对「同帧内不同
     hypo 消费不同 token 数」的场景仍成立，且 P 逐行携带绝对位置。

受限因果：因果 mask + 可配左上下文窗 `left_context`。非流式评测用大窗(≈全因果)；
流式收紧。对应 Luo 2021 MSA-Transducer (2102.11594)。

state 类型仍是 `List[List[Tensor]]`（对 beam-search 不透明，已核实 torchaudio 2.6
`RNNTBeamSearch` 不查内部结构）⇒ 换 trunk 不破坏 beam-search 契约。
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# 单层因果 Transformer
# ═══════════════════════════════════════════════════════════════════════════


class _CausalAttnLayer(torch.nn.Module):
    """受限因果自注意力 + FFN，带定长循环 KV cache 支持逐 token 增量解码。

    训练/批量前向：full sequence，对每个 query 位置 q 只 attend [q-left_context, q]。
    增量解码 (beam-search predict)：cache 恒为 left_context 长，新 token 写入
    槽位 p % left_context，p 为该行已消费 token 数 (绝对位置)。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        left_context: int,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.left_context = left_context  # 每个位置最多回看多少历史(含自身)

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, d_model),
            torch.nn.Dropout(dropout),
        )

    # ------------------------------------------------------------------
    def _mha_full(self, x: Tensor) -> Tensor:
        """完整序列前向，受限因果 mask。x: (B, U, D)。"""
        B, U, _ = x.shape
        q = self.q_proj(x).view(B, U, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, U, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, U, self.num_heads, self.head_dim).transpose(1, 2)

        # 受限因果：query i attend keys [max(0, i-left_context+1) .. i]
        idx = torch.arange(U, device=x.device)
        lo = torch.clamp(idx - self.left_context + 1, min=0)
        attn_mask = (idx[None, :] >= lo[:, None]) & (idx[None, :] <= idx[:, None])  # (U,U)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1,1,U,U)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, U, self.d_model)
        return self.out_proj(out)

    def _mha_increment(
        self, x_new: Tensor, k_cache: Tensor, v_cache: Tensor, p: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """单步增量 (定长循环 KV cache)。x_new: (B,1,D), p: (B,) 已消费 token 数
        (新 token 的绝对位置)。返回 (out, new_k_cache, new_v_cache)。

        cache 恒为 left_context 长, 新 token 写入槽位 p % left_context;
        槽位 s 上 token 的"年龄" age = (p - s) % left_context, 仅当 age ≤ p
        (对应绝对位置 ≥ 0, 未被首次写入) 时参与 attention — 等价于
        [p-left_context+1, p] 的受限因果窗, 与 _mha_full 逐位一致。
        定长 cache 保证不同 hypo 消费不同 token 数时 _batch_state 仍可 cat。
        """
        B, L = x_new.size(0), k_cache.size(2)
        q = self.q_proj(x_new).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x_new).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x_new).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        slot = (p % L).view(B, 1, 1, 1).expand(B, self.num_heads, 1, self.head_dim)
        new_k = k_cache.scatter(2, slot, k_new)
        new_v = v_cache.scatter(2, slot, v_new)

        s = torch.arange(L, device=x_new.device)
        delta = (p.view(B, 1) - s.view(1, L)) % L  # (B, L) 槽位 token 年龄
        valid = (delta <= p.view(B, 1))  # (B, L) bool
        out = torch.nn.functional.scaled_dot_product_attention(
            q, new_k, new_v, attn_mask=valid.view(B, 1, 1, L), dropout_p=0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        return self.out_proj(out), new_k, new_v

    # ------------------------------------------------------------------
    def forward_full(self, x: Tensor) -> Tensor:
        """完整序列残差前向（pre-norm）。x: (B, U, D) → (B, U, D)。"""
        x = x + self.dropout(self._mha_full(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_inc(
        self, x_new: Tensor, k_cache: Tensor, v_cache: Tensor, p: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """单步增量残差前向（pre-norm）。x_new: (B,1,D), p: (B,) → (out, new_k, new_v)。"""
        attn_out, new_k, new_v = self._mha_increment(self.norm1(x_new), k_cache, v_cache, p)
        x = x_new + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, new_k, new_v


# ═══════════════════════════════════════════════════════════════════════════
# 受限因果自注意力 Predictor
# ═══════════════════════════════════════════════════════════════════════════


class PredictorAttn(torch.nn.Module):
    r"""受限因果自注意力 predictor trunk。

    与原 `Predictor` 接口/语义对齐：
        forward(input:(B,U) int, lengths:(B,), state:Optional[List[List[Tensor]]])
            -> (out:(B,U,output_dim), lengths, state_out:List[List[Tensor]])

    Args:
        num_symbols: 词表大小(含 blank)，即 embedding 行数 = num_symbols+1 之外的真实词表。
            这里 num_symbols 沿用仓库约定 = 真实符号数，embedding 行数 = num_symbols+1。
        output_dim: 输出维度(=512)，与 Joiner 契约。
        symbol_embedding_dim: token embedding 维度(同时作为 d_model 的输入投影维度)。
        d_model: Transformer 内部维度。
        num_layers: Transformer 层数。
        num_heads: 注意力头数。
        ffn_dim: FFN 隐藏维。
        left_context: 受限左上下文窗(含自身)。设大则近似全因果。
        dropout: dropout。
        max_len: 位置编码表长度(≥ 训练/解码最长 token 序列)。
    """

    def __init__(
        self,
        *,
        num_symbols: int,
        output_dim: int = 512,
        symbol_embedding_dim: int = 128,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        left_context: int = 1024,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        # embedding 行数 = num_symbols+1 (与原 Predictor 一致，blank 挂末位)
        self.embedding = torch.nn.Embedding(num_symbols + 1, symbol_embedding_dim)
        self.input_proj = torch.nn.Linear(symbol_embedding_dim, d_model)
        self.pos = torch.nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.input_layer_norm = torch.nn.LayerNorm(d_model)
        self.layers = torch.nn.ModuleList(
            [
                _CausalAttnLayer(d_model, num_heads, ffn_dim, dropout, left_context)
                for _ in range(num_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.final_norm = torch.nn.LayerNorm(d_model)  # pre-norm 栈的出口归一化
        self.linear = torch.nn.Linear(d_model, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.left_context = left_context

    # ------------------------------------------------------------------
    def _embed(self, tokens: Tensor, positions: Tensor) -> Tensor:
        """token ids (B,U) + 每行起始绝对位置 positions (B,) → (B,U,d_model)。"""
        U = tokens.size(1)
        pos = positions.view(-1, 1) + torch.arange(U, device=tokens.device)
        pos = torch.clamp(pos, max=self.max_len - 1)
        h = self.input_proj(self.embedding(tokens)) + self.pos(pos)
        return self.input_layer_norm(h)

    # ------------------------------------------------------------------
    def _empty_cache(self, batch: int, device: torch.device, dtype: torch.dtype) -> List[List[Tensor]]:
        """空 cache：每层 [K, V, P]，全部定长 — K/V 恒为 left_context 长
        (循环写槽位 p % left_context)，P (batch,) = 已消费 token 数。

        定长 cache 是 torchaudio beam-search 批处理 (_batch_state 沿 batch
        维 cat) 的前提: 不同 hypo 在同一帧内消费的 token 数可以不同
        (A-hypo 同帧续扩), 变长 KV cache 无法 cat。
        """
        L = self.left_context
        return [
            [
                torch.zeros(batch, self.num_heads, L, self.head_dim, device=device, dtype=dtype),
                torch.zeros(batch, self.num_heads, L, self.head_dim, device=device, dtype=dtype),
                torch.zeros(batch, dtype=torch.long, device=device),
            ]
            for _ in self.layers
        ]

    # ------------------------------------------------------------------
    def forward(
        self,
        input: Tensor,
        lengths: Tensor,
        state: Optional[List[List[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, List[List[Tensor]]]:
        """训练/批量前向：完整序列并行算，受限因果 mask。

        Args:
            input: (B, U) int token ids。U 必须 ≤ max_len（_assert 显式拦截）。
            lengths: (B,) 有效长度(原样返回，beam-search 逐 token 不用它做 mask)。
            state: None → 全序列路径(训练)；非 None 且 input 为单步 (B,1)
                → 委托 predict() 走增量路径(torchaudio RNNTBeamSearch 的
                decode 契约: model.predict → predictor.forward 带状态单步)。
                若 multi-token 且带 state，退回全序列路径并忽略 state
                (当前无此调用点)。

        注意（隐式契约）：本层不做 padding mask。其安全性依赖两点——
        padding 只出现在序列尾部，且因果 mask 保证 query 看不到右侧 key，
        故 padded 位置的输出是垃圾但不会被非 padded query 读到；loss 侧
        必须按 lengths 屏蔽 padded 位置。若改变这两点任一，需补 padding mask。
        """
        torch._assert(
            input.size(1) <= self.max_len,
            "PredictorAttn: 序列长度超过 max_len 位置编码表；请调大 attn.max_len",
        )
        if state is not None and input.size(1) == 1:
            return self.predict(input, lengths, state)
        B = input.size(0)
        positions = torch.zeros(B, dtype=torch.long, device=input.device)
        x = self._embed(input, positions)  # (B, U, d_model)
        for layer in self.layers:
            x = layer.forward_full(x)
        x = self.dropout(self.final_norm(x))
        out = self.output_layer_norm(self.linear(x))  # (B, U, output_dim)
        return out, lengths, self._empty_cache(B, input.device, x.dtype)

    # ------------------------------------------------------------------
    @torch.jit.export
    def predict(
        self,
        input: Tensor,
        lengths: Tensor,
        state: Optional[List[List[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, List[List[Tensor]]]:
        """增量逐 token 前向(被 RNNTBeamSearch 调用)。

        input: (B, 1) 单 token。state: 每层 [K_cache, V_cache, P] 或 None；
        K/V 定长 left_context (循环写槽位 P % left_context)；P (batch,)
        = 各行已消费 token 数(绝对位置) — beam search 中不同 hypo 的 P
        可以不同，逐行取位。
        返回 (out:(B,1,output_dim), lengths, new_state)。
        """
        if state is None:
            state = self._empty_cache(
                input.size(0), input.device, self.pos.weight.dtype
            )
        p = state[0][2]  # (B,) long, 逐行绝对位置

        x = self._embed(input, p)  # (B,1,d_model)
        new_state: List[List[Tensor]] = []
        p_new = p + input.size(1)
        for layer_idx, layer in enumerate(self.layers):
            st = state[layer_idx]
            x, new_k, new_v = layer.forward_inc(x, st[0], st[1], p)
            new_state.append([new_k, new_v, p_new])
        x = self.dropout(self.final_norm(x))
        out = self.output_layer_norm(self.linear(x))  # (B,1,output_dim)
        return out, lengths, new_state


def conformer_rnnt_predictor_attn_from_config(config: dict) -> "PredictorAttn":
    """从 config['model']['predictor']['attn'] 段构建(缺省回退默认)。"""
    m = config.get("model", config)
    p = m.get("predictor", {})
    a = p.get("attn", {})
    return PredictorAttn(
        num_symbols=m.get("num_symbols", 74),
        output_dim=m.get("encoding_dim", 512),
        symbol_embedding_dim=p.get("symbol_embedding_dim", 128),
        d_model=a.get("d_model", 256),
        num_layers=a.get("num_layers", 4),
        num_heads=a.get("num_heads", 4),
        ffn_dim=a.get("ffn_dim", 1024),
        left_context=a.get("left_context", 1024),
        dropout=a.get("dropout", 0.1),
        max_len=a.get("max_len", 4096),
    )