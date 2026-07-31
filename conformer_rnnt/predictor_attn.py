"""受限因果自注意力 Predictor trunk (Family A — MSA-Transducer 风格)。

替代原 LSTM `Predictor`，作为 Medusa 多头的共享 backbone：
    Embedding → LayerNorm → N 层因果 Transformer(受限左上下文) → Dropout → Linear → LayerNorm
output_dim=512 契约不变，与 Joiner 加性融合兼容。

受限因果：因果 mask + 可配左上下文窗 `left_context`。非流式评测用大窗(≈全因果)；
流式收紧。对应 Luo 2021 MSA-Transducer (2102.11594)。

state=KV cache：每层 `[K_cache, V_cache]`，类型 `List[List[Tensor]]`——与原 LSTM
`[h,c]` 同型。`torchaudio.models.RNNTBeamSearch` 把 state 当不透明对象穿线、
不查内部结构 (已核实 torchaudio 2.6 源码) ⇒ 换 trunk 不破坏 beam-search 契约。
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# 单层因果 Transformer
# ═══════════════════════════════════════════════════════════════════════════


class _CausalAttnLayer(torch.nn.Module):
    """受限因果自注意力 + FFN，带 KV cache 支持逐 token 增量解码。

    训练/批量前向：full sequence，对每个 query 位置 q 只 attend [q-left_context, q]。
    增量解码 (beam-search predict)：传入 prior KV cache，仅算新 token 一步，
        append 进 cache 返回。left_context 限制 cache 长度(超出的左截断)。
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
        self, x_new: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """单步增量。x_new: (B, 1, D)。返回 (out, new_k_cache, new_v_cache)。

        prior cache 长度若已达 left_context-1，左截断 1 帧保持窗内。
        """
        B = x_new.size(0)
        q = self.q_proj(x_new).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(x_new).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x_new).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # append + 受限窗截断
        new_k = torch.cat([k_cache, k_new], dim=2)
        new_v = torch.cat([v_cache, v_new], dim=2)
        if new_k.size(2) > self.left_context:
            new_k = new_k[:, :, -self.left_context:, :]
            new_v = new_v[:, :, -self.left_context:, :]

        # q (位置 p) attend 全部 cache（cache 已被窗截断到 ≤ left_context），
        # 即 attend [p-left_context+1 .. p]，等价受限因果。
        out = torch.nn.functional.scaled_dot_product_attention(
            q, new_k, new_v, dropout_p=0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        return self.out_proj(out), new_k, new_v

    # ------------------------------------------------------------------
    def forward_full(self, x: Tensor) -> Tensor:
        """完整序列残差前向。x: (B, U, D) → (B, U, D)。"""
        x = x + self.dropout(self._mha_full(x))
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

    def forward_inc(
        self, x_new: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """单步增量残差前向。x_new: (B,1,D) → (out, new_k, new_v)。"""
        attn_out, new_k, new_v = self._mha_increment(x_new, k_cache, v_cache)
        x = x_new + self.dropout(attn_out)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
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
    ) -> None:
        super().__init__()
        # embedding 行数 = num_symbols+1 (与原 Predictor 一致，blank 挂末位)
        self.embedding = torch.nn.Embedding(num_symbols + 1, symbol_embedding_dim)
        self.input_proj = torch.nn.Linear(symbol_embedding_dim, d_model)
        self.input_layer_norm = torch.nn.LayerNorm(d_model)
        self.layers = torch.nn.ModuleList(
            [
                _CausalAttnLayer(d_model, num_heads, ffn_dim, dropout, left_context)
                for _ in range(num_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(d_model, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.left_context = left_context

    # ------------------------------------------------------------------
    def _empty_cache(self, batch: int, device: torch.device, dtype: torch.dtype) -> List[List[Tensor]]:
        """空 KV cache：每层 [K(0 dim), V(0 dim)]。"""
        zero = torch.zeros(batch, self.num_heads, 0, self.head_dim).to(device=device, dtype=dtype)
        return [[zero, zero] for _ in self.layers]

    # ------------------------------------------------------------------
    def forward(
        self,
        input: Tensor,
        lengths: Tensor,
        state: Optional[List[List[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, List[List[Tensor]]]:
        """训练/批量前向：完整序列并行算，受限因果 mask。

        Args:
            input: (B, U) int token ids。
            lengths: (B,) 有效长度(原样返回，beam-search 逐 token 不用它做 mask)。
            state: 忽略(批量前向不接 cache)；返回空 cache 供后续增量起点。
        """
        emb = self.embedding(input)  # (B, U, emb_dim)
        x = self.input_layer_norm(self.input_proj(emb))  # (B, U, d_model)
        for layer in self.layers:
            x = layer.forward_full(x)
        x = self.dropout(x)
        out = self.output_layer_norm(self.linear(x))  # (B, U, output_dim)
        return out, lengths, self._empty_cache(input.size(0), input.device, x.dtype)

    # ------------------------------------------------------------------
    @torch.jit.export
    def predict(
        self,
        input: Tensor,
        lengths: Tensor,
        state: Optional[List[List[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, List[List[Tensor]]]:
        """增量逐 token 前向(被 RNNTBeamSearch 调用)。

        input: (B, 1) 单 token。state: 每层 [K_cache, V_cache] 或 None。
        返回 (out:(B,1,output_dim), lengths, new_state)。
        new_state 把新 token 的 K/V append 进各层 cache，并按 left_context 截断。
        """
        if state is None:
            state = self._empty_cache(input.size(0), input.device, torch.float32)

        emb = self.embedding(input)  # (B,1,emb)
        x = self.input_layer_norm(self.input_proj(emb))  # (B,1,d_model)
        new_state: List[List[Tensor]] = []
        layer_idx = 0
        for layer in self.layers:
            st = state[layer_idx]
            k_cache = st[0]
            v_cache = st[1]
            x, new_k, new_v = layer.forward_inc(x, k_cache, v_cache)
            new_state.append([new_k, new_v])
            layer_idx += 1
        x = self.dropout(x)
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
    )