"""Medusa 式多头未来标签预测器。

head 0 继续使用原始 Joiner 的 ``joint(t, u)``，作为无损 verifier 的唯一
打分锚点。未来 head 在同一个 joint 上叠加一个零初始化的 context delta，
初始化时与 base 逐位一致，训练后才有条件感知。

iter-10 (删窗 + draft 侧因果序列模型):
  - 头不再拥有自己的声学窗。``_anchor_window`` / ``context_acoustic`` /
    ``window_prebuilt`` / ``past_frames`` / ``lookahead_frames`` 全部删除。
    解码到哪一帧就条件于哪一帧的编码特征 ``enc[t]``（经零初始化
    ``acoustic_frame_proj`` 加性注入）；序列上下文感知是 encoder 的
    chunk 注意力 (left/segment/right) 的职责，不在头侧重复。
  - 序列建模增强转向 draft 侧可训练模块 ``seq_model``
    （``draft_seq_model`` 配置，type=transformer|lstm）：
      * transformer: 因果 self-attn（默认 3 层 × 8 头，d=512），
        训练吃 [BOS, y_1..y_U] 全序列（teacher forcing），推理带 KV cache
        逐步滚动（快照回滚 = 截断 KV 长度）。
      * lstm: base 之上的可训练加深 LSTM（默认 4 层 × 256，LayerNorm），
        输出投影到 512；隐状态 tuple 即 cache，回滚 = 恢复快照。
    两者输出经零初始化 ``draft_state_proj`` 加性注入 fused。
  - 对齐律：head-k（1-based）在网格列 l 条件于「消费完 y_{l+k-1} 的
    状态」并预测 y_{l+k}。三条通道同构：LSTM rolled 状态
    ``pad(tgt_enc,K)[:, k-1:k-1+L]``、序列模型隐藏态
    ``pad(h_seq,K)[:, k-1:k-1+L]``、推理 chain_hiddens[k]。
  - ``draft_lm_proj``（零初始化）把 h_seq 映成 LM logits（共享冻结
    w_vocab），支持 ``lm_aux_loss`` 纯 token 侧 next-token CE。

iter-9: sim 通道整体移除 (chain_sim 扫描 / sim_bias 直注 /
verifier_distillation / delay_decompose)。监督回归 shifted CE (+ 可选
KL / base_next_token CE)。
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from conformer_rnnt.predictor import CustomLSTM


# ═══════════════════════════════════════════════════════════════════════════
# ResBlock (零初始化残差)
# ═══════════════════════════════════════════════════════════════════════════


class MedusaResBlock(torch.nn.Module):
    """Linear→SiLU→Linear(零初始化)→+residual。零初始化 ⇒ 初始 ≈ 恒等。"""

    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.act = torch.nn.SiLU()
        self.linear2 = torch.nn.Linear(dim, dim)
        torch.nn.init.zeros_(self.linear2.weight)
        torch.nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.linear2(self.act(self.linear1(x)))


# ═══════════════════════════════════════════════════════════════════════════
# Draft 侧因果序列模型 (transformer | lstm)，统一 cache 接口
# ═══════════════════════════════════════════════════════════════════════════


class _CausalSelfAttention(torch.nn.Module):
    """多头因果 self-attention，支持 KV cache 增量解码。"""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} 不能被 nhead {nhead} 整除")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, x: Tensor, cache: Optional[Tensor] = None,
        causal_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """x: (B, L, D) → (out (B, L, D), cache (k, v) 各 (B, H, P+L, hd))。"""
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, L, D) → (B, H, L, hd)
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        if cache is not None:
            pk, pv = cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(weights, v).transpose(1, 2).reshape(B, L, D)
        return self.out(out), new_cache


class _TransformerLayer(torch.nn.Module):
    """Pre-norm 因果 transformer 层（self-attn + FFN）。"""

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, nhead, dropout)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(ffn_dim, d_model),
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, x: Tensor, cache: Optional[Tuple[Tensor, Tensor]] = None,
        causal_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h, new_cache = self.attn(self.norm1(x), cache, causal_mask)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, new_cache


class DraftCausalTransformer(torch.nn.Module):
    """Draft 侧因果 transformer：TF 全序列 + KV cache 单步两路径。

    cache 是逐层 (k, v) 张量列表；快照 = 浅拷贝列表（cat 产生新张量，
    旧引用永远有效），回滚 = 恢复快照引用后重放已提交 token。
    """

    def __init__(
        self,
        num_vocab: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        emb_dim: int = 256,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(num_vocab, emb_dim)
        self.proj = (
            torch.nn.Identity() if emb_dim == d_model
            else torch.nn.Linear(emb_dim, d_model)
        )
        self.pos = torch.nn.Embedding(max_len, d_model)
        self.layers = torch.nn.ModuleList(
            [_TransformerLayer(d_model, nhead, ffn_dim, dropout)
             for _ in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(d_model)

    def _embed(self, tokens: Tensor, start_pos: int) -> Tensor:
        positions = torch.arange(
            start_pos, start_pos + tokens.size(1), device=tokens.device
        )
        return self.proj(self.embed(tokens)) + self.pos(positions).unsqueeze(0)

    def forward_full(self, tokens: Tensor) -> Tensor:
        """(B, P) token id → 因果隐藏态 (B, P, D)。"""
        L = tokens.size(1)
        x = self._embed(tokens, 0)
        mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        for layer in self.layers:
            x, _ = layer(x, None, mask)
        return self.norm(x)

    def init_cache(self, tokens: Tensor) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """消费前缀 token（如 BOS），返回 (末位隐藏 (B,1,D), cache)。"""
        x = self._embed(tokens, 0)
        L = tokens.size(1)
        mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        cache: List[Tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            x, layer_cache = layer(x, None, mask)
            cache.append(layer_cache)
        return self.norm(x)[:, -1:], cache

    def forward_step(
        self, token: Tensor, cache: List[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """单 token 增量步。token: (B, 1) long。"""
        pos = cache[0][0].size(2)
        x = self._embed(token, pos)
        new_cache: List[Tuple[Tensor, Tensor]] = []
        for layer, layer_cache in zip(self.layers, cache):
            x, layer_cache = layer(x, layer_cache, None)
            new_cache.append(layer_cache)
        return self.norm(x), new_cache

    @staticmethod
    def clone_cache(cache: List[Tuple[Tensor, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        return list(cache)

    @staticmethod
    def restore_cache(snapshot: List[Tuple[Tensor, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        return list(snapshot)

    @staticmethod
    def cache_length(cache: List[Tuple[Tensor, Tensor]]) -> int:
        return int(cache[0][0].size(2))


class DraftLSTM(torch.nn.Module):
    """Draft 侧可训练加深 LSTM（base predictor 不动，见 iter-10 计划）。

    cache = (逐层 [h, c] 状态, 已消费 token 数)；回滚 = 恢复深拷贝快照。
    """

    def __init__(
        self,
        num_vocab: int,
        output_dim: int = 512,
        emb_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        layer_norm: bool = True,
        layer_norm_epsilon: float = 1e-3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(num_vocab, emb_dim)
        self.input_norm = torch.nn.LayerNorm(emb_dim)
        self.layers = torch.nn.ModuleList(
            [
                CustomLSTM(
                    emb_dim if idx == 0 else hidden_dim,
                    hidden_dim,
                    layer_norm=layer_norm,
                    layer_norm_epsilon=layer_norm_epsilon,
                )
                for idx in range(num_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.out_proj = torch.nn.Linear(hidden_dim, output_dim)
        self.out_norm = torch.nn.LayerNorm(output_dim)

    def _run(self, tokens: Tensor, state: Optional[List[List[Tensor]]]):
        """tokens: (T, B) long；返回末层输出 (T, B, hidden) 与新状态。"""
        x = self.input_norm(self.embed(tokens))
        state_out: List[List[Tensor]] = []
        for layer_idx, lstm in enumerate(self.layers):
            x, layer_state = lstm(
                x, None if state is None else state[layer_idx]
            )
            x = self.dropout(x)
            state_out.append(layer_state)
        return x, state_out

    def _to_out(self, x_tb: Tensor) -> Tensor:
        return self.out_norm(self.out_proj(x_tb)).permute(1, 0, 2)

    def forward_full(self, tokens: Tensor) -> Tensor:
        """(B, P) token id → 隐藏态 (B, P, D)。"""
        x, _ = self._run(tokens.permute(1, 0), None)
        return self._to_out(x)

    def init_cache(
        self, tokens: Tensor,
    ) -> Tuple[Tensor, Tuple[List[List[Tensor]], int]]:
        x, state = self._run(tokens.permute(1, 0), None)
        return self._to_out(x)[:, -1:], (state, int(tokens.size(1)))

    def forward_step(
        self, token: Tensor, cache: Tuple[List[List[Tensor]], int],
    ) -> Tuple[Tensor, Tuple[List[List[Tensor]], int]]:
        state, length = cache
        x, new_state = self._run(token.permute(1, 0), state)
        return self._to_out(x), (new_state, length + 1)

    @staticmethod
    def clone_cache(cache: Tuple[List[List[Tensor]], int]):
        state, length = cache
        # state = [layer, ...]，每个 layer = [h, c]
        return (
            [[h.detach().clone(), c.detach().clone()] for h, c in state],
            length,
        )

    @staticmethod
    def restore_cache(snapshot: Tuple[List[List[Tensor]], int]):
        state, length = snapshot
        restored = [[h.clone(), c.clone()] for h, c in state]
        return (restored, length)

    @staticmethod
    def cache_length(cache: Tuple[List[List[Tensor]], int]) -> int:
        return int(cache[1])


def build_draft_seq_model(num_vocab: int, output_dim: int,
                          spec: Optional[Dict[str, Any]]):
    """根据配置构造 draft 侧序列模型（``draft_seq_model`` YAML 节）。

    spec: ``{type: transformer, num_layers: 3, d_model: 512, nhead: 8,
    ffn_dim: 1024, dropout: 0.1}`` 或 ``{type: lstm, num_layers: 4,
    hidden_dim: 256, dropout: 0.3}``。None → 不构建（旧消融行为）。
    """
    if spec is None:
        return None
    spec = dict(spec)
    kind = spec.get("type", "transformer")
    if kind == "transformer":
        return DraftCausalTransformer(
            num_vocab=num_vocab,
            d_model=int(spec.get("d_model", output_dim)),
            nhead=int(spec.get("nhead", 8)),
            num_layers=int(spec.get("num_layers", 3)),
            ffn_dim=int(spec.get("ffn_dim", 1024)),
            dropout=float(spec.get("dropout", 0.1)),
            emb_dim=int(spec.get("emb_dim", 256)),
        )
    if kind == "lstm":
        return DraftLSTM(
            num_vocab=num_vocab,
            output_dim=output_dim,
            emb_dim=int(spec.get("emb_dim", 256)),
            hidden_dim=int(spec.get("hidden_dim", 256)),
            num_layers=int(spec.get("num_layers", 4)),
            layer_norm=bool(spec.get("layer_norm", True)),
            layer_norm_epsilon=float(spec.get("layer_norm_epsilon", 1e-3)),
            dropout=float(spec.get("dropout", 0.3)),
        )
    raise ValueError(
        f"Unknown draft_seq_model.type {kind!r}; choose transformer or lstm"
    )


class MedusaHeads(torch.nn.Module):
    r"""K 个 Medusa 残差头 + 共享词表投影 + draft 侧因果序列模型。

    ``current_joint`` 保留旧的 joint-only 消融；``predictor_only`` 只给
    context 分支 predictor state；``stream_local`` 使用当前 encoder frame
    （``enc[t]``，无窗）和 predictor state。context 分支的最后一层零初始
    化，因此所有新 head 初始仍与 base head 完全一致。

    **对齐律 (iter-10)**: head-k（1-based）在网格列 l 条件于「消费完
    y_{l+k-1} 的状态」，预测 y_{l+k}。三条条件通道同构：
      1. LSTM rolled 状态 ``pad(tgt_enc,K)[:, k-1 : k-1+L]``；
      2. draft 序列模型隐藏态 ``pad(h_seq,K)[:, k-1 : k-1+L]``
         （h_seq[m] = 消费完 prep[m] 的隐藏态，prep=[BOS, y_1..y_U]）；
      3. 推理 ``chain_states[k]`` / ``chain_hiddens[k]``（emit y_u 后
         head-1 吃 u+1 个 token（含 BOS），head-k 吃 u+k 个）。

    Args:
        input_dim: 特征维度 (=512)，ResBlock 在此维度做残差。
        num_vocab: 词表大小 (含 blank) = num_symbols+1。
        num_heads: K，未来预测头数 (head 1..K)。
        w_vocab: 共享输出投影，必须传 joiner.linear。
        input_mode: ``current_joint`` | ``predictor_only`` | ``stream_local``。
        chain_states: 链式对齐（draft_seq_model 需要开启）。
        draft_seq_model: draft 侧序列模型配置（None = 不构建，旧消融行为）。
    """

    def __init__(
        self,
        input_dim: int,
        num_vocab: int,
        num_heads: int,
        w_vocab: torch.nn.Linear,
        input_mode: str = "stream_local",
        draft_after_emit: bool = False,
        chain_states: bool = False,
        draft_seq_model: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if w_vocab.in_features != input_dim or w_vocab.out_features != num_vocab:
            raise ValueError(
                f"w_vocab 形状不匹配: 传入 {w_vocab.in_features}→{w_vocab.out_features}, "
                f"期望 {input_dim}→{num_vocab}"
            )
        if input_mode not in {"current_joint", "predictor_only", "stream_local"}:
            raise ValueError(
                "Unknown Medusa input_mode {!r}; choose current_joint, "
                "predictor_only, or stream_local".format(input_mode)
            )
        if draft_seq_model is not None and not chain_states:
            raise ValueError(
                "draft_seq_model 需要 chain_states=True（对齐律依赖 rolled 状态）"
            )

        self.num_heads = num_heads
        self.input_mode = input_mode
        self.draft_after_emit = bool(draft_after_emit)
        self.chain_states = bool(chain_states)
        # 本仓库约定 blank = 词表最后一个 id (lightning: blank_idx =
        # sp_model.get_piece_size(), num_vocab = num_symbols+1)。
        self.blank_idx = num_vocab - 1
        self.res_blocks = torch.nn.ModuleList(
            [MedusaResBlock(input_dim) for _ in range(num_heads)]
        )

        self.context_norm: Optional[torch.nn.LayerNorm]
        self.context_mlp: Optional[torch.nn.Sequential]
        self.context_predictor: Optional[torch.nn.Linear]
        self.context_base_logits: Optional[torch.nn.Linear]
        if input_mode == "stream_local":
            self.acoustic_frame_proj = torch.nn.Linear(input_dim, input_dim)
            torch.nn.init.zeros_(self.acoustic_frame_proj.weight)
            torch.nn.init.zeros_(self.acoustic_frame_proj.bias)
            self.context_predictor = torch.nn.Linear(input_dim, input_dim)
            self.context_base_logits = torch.nn.Linear(num_vocab, input_dim)
            self.context_norm = torch.nn.LayerNorm(input_dim)
            # iter-6/7: 原 context_mlp / extra / per-head 深路径全保留
            # (ep07 warm-start 热载)。
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
            self.context_mlp_extra = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
            torch.nn.init.zeros_(self.context_mlp_extra[-1].weight)
            torch.nn.init.zeros_(self.context_mlp_extra[-1].bias)
            self.context_mlp_heads = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 2 * input_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(2 * input_dim, 2 * input_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(2 * input_dim, input_dim),
                )
                for _ in range(num_heads)
            ])
            for mlp_h in self.context_mlp_heads:
                torch.nn.init.zeros_(mlp_h[-1].weight)
                torch.nn.init.zeros_(mlp_h[-1].bias)
        elif input_mode == "predictor_only":
            self.acoustic_frame_proj = None
            self.context_predictor = torch.nn.Linear(input_dim, input_dim)
            self.context_base_logits = torch.nn.Linear(num_vocab, input_dim)
            self.context_norm = torch.nn.LayerNorm(input_dim)
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
        else:
            self.acoustic_frame_proj = None
            self.context_predictor = None
            self.context_base_logits = None
            self.context_norm = None
            self.context_mlp = None

        if self.context_base_logits is not None:
            # Zero init preserves an old head's behavior while allowing the
            # new signal to be learned.
            torch.nn.init.zeros_(self.context_base_logits.weight)
            torch.nn.init.zeros_(self.context_base_logits.bias)

        # Draft-conditioned chaining (iter-3/4, 保留热载)。
        self.chain_draft_embed: Optional[torch.nn.Embedding]
        self.chain_draft_proj: Optional[torch.nn.Linear]
        self.chain_state_proj: Optional[torch.nn.Linear]
        if self.chain_states and num_heads > 1:
            self.chain_draft_embed = torch.nn.Embedding(num_vocab, 128)
            self.chain_draft_proj = torch.nn.Linear(128, input_dim)
            torch.nn.init.zeros_(self.chain_draft_proj.weight)
            torch.nn.init.zeros_(self.chain_draft_proj.bias)
            self.chain_state_proj = torch.nn.Linear(input_dim, input_dim)
            torch.nn.init.zeros_(self.chain_state_proj.weight)
            torch.nn.init.zeros_(self.chain_state_proj.bias)
            self.chain_state_proj_heads = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, input_dim)
                for _ in range(num_heads)
            ])
            for proj in self.chain_state_proj_heads:
                torch.nn.init.zeros_(proj.weight)
                torch.nn.init.zeros_(proj.bias)
        else:
            self.chain_draft_embed = None
            self.chain_draft_proj = None
            self.chain_state_proj = None

        if self.context_mlp is not None:
            # Preserve the exact base-logit initialization. The context path
            # becomes active only after its final projection learns.
            torch.nn.init.zeros_(self.context_mlp[-1].weight)
            torch.nn.init.zeros_(self.context_mlp[-1].bias)

        # ---- iter-10: draft 侧因果序列模型 + 零初始化注入通道 ----
        self.seq_model = build_draft_seq_model(num_vocab, input_dim, draft_seq_model)
        self.draft_state_proj: Optional[torch.nn.Linear]
        self.draft_lm_proj: Optional[torch.nn.Linear]
        if self.seq_model is not None:
            self.draft_state_proj = torch.nn.Linear(input_dim, input_dim)
            self.draft_lm_proj = torch.nn.Linear(input_dim, input_dim)
            torch.nn.init.zeros_(self.draft_state_proj.weight)
            torch.nn.init.zeros_(self.draft_state_proj.bias)
            torch.nn.init.zeros_(self.draft_lm_proj.weight)
            torch.nn.init.zeros_(self.draft_lm_proj.bias)
        else:
            self.draft_state_proj = None
            self.draft_lm_proj = None

        # 与 joiner 共享输出投影：head 0 ≡ joiner 精确一致。
        self.w_vocab = w_vocab

    # ------------------------------------------------------------------
    def _context_delta(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor],
        target_encodings: Tensor,
        base_logits: Optional[Tensor],
    ) -> Tensor:
        """Non-chain 模式的 context delta，形状 ``(B,T,L,D)``。"""
        if self.input_mode == "current_joint" or self.context_mlp is None:
            return torch.zeros_like(joint)

        B, T, L, _ = joint.shape
        if target_encodings.size(0) != B:
            raise ValueError(
                "target_encodings must match joint batch size: "
                f"joint={tuple(joint.shape)}, targets={tuple(target_encodings.shape)}"
            )

        if self.chain_states:
            # Draft-conditioned chaining — handled in forward(), which builds
            # heads sequentially. This path is unreachable in production.
            raise RuntimeError(
                "chain mode builds heads in forward(); _context_delta "
                "should not be called with chain_states=True"
            )

        if target_encodings.size(1) != L:
            raise ValueError(
                "target_encodings must match joint on label axis: "
                f"joint={tuple(joint.shape)}, targets={tuple(target_encodings.shape)}"
            )

        predictor = self.context_predictor(target_encodings)
        if base_logits is None or self.context_base_logits is None:
            return torch.zeros_like(joint)
        if base_logits.shape[:3] != (B, T, L):
            raise ValueError(
                "base_logits must match joint on batch/time/label axes: "
                f"joint={tuple(joint.shape)}, base={tuple(base_logits.shape)}"
            )
        base_context = self.context_base_logits(base_logits.detach())
        if self.input_mode == "predictor_only":
            fused = predictor.unsqueeze(1).expand(-1, T, -1, -1) + base_context
        else:
            if source_encodings is None:
                return torch.zeros_like(joint)
            if source_encodings.shape != (B, T, joint.size(-1)):
                raise ValueError(
                    "source_encodings must be (B, T, D) matching joint: "
                    f"joint={tuple(joint.shape)}, source={tuple(source_encodings.shape)}"
                )
            acoustic = self.acoustic_frame_proj(source_encodings)
            fused = acoustic.unsqueeze(2) + predictor.unsqueeze(1) + base_context

        normalized = self.context_norm(fused)
        extra = getattr(self, "context_mlp_extra", None)
        delta = self.context_mlp(normalized) + (
            extra(normalized) if extra is not None else 0.0
        )
        return delta

    def _chain_shared_terms(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor],
        target_encodings: Tensor,
        base_logits: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Chain 模式共享项：predictor / 当前帧声学 / base logits detach。

        声学项 = ``acoustic_frame_proj(source_encodings)``：训练 (B,T,D)、
        推理 (B,1,D)，时间轴与 joint 的 T 严格对齐（无窗概念）。
        """
        B, T, L, D = joint.shape
        predictor = self.context_predictor(target_encodings)  # (B,L,D)
        acoustic_term: Optional[Tensor] = None
        if self.input_mode == "stream_local":
            if source_encodings is None:
                raise ValueError("stream_local chain mode requires source_encodings")
            if source_encodings.shape != (B, T, D):
                raise ValueError(
                    "source_encodings must be (B, T, D) matching joint time axis: "
                    f"joint={tuple(joint.shape)}, source={tuple(source_encodings.shape)}"
                )
            acoustic_term = self.acoustic_frame_proj(source_encodings)  # (B,T,D)
        base_term: Optional[Tensor] = None
        if base_logits is not None and self.context_base_logits is not None:
            base_term = self.context_base_logits(base_logits.detach())
        return predictor, acoustic_term, base_term

    def _chain_head_delta(
        self,
        shared: Tuple[Tensor, Optional[Tensor], Optional[Tensor]],
        joint: Tensor,
        k: int,
        draft_ids: Optional[Tensor] = None,
        prev_logits: Optional[Tensor] = None,
        rolled_state: Optional[Tensor] = None,
        rolled_hidden: Optional[Tensor] = None,
    ) -> Tensor:
        """Context delta for chain head ``k`` (0-based)。

        ``draft_ids`` (inference, (B,1,1) long) or ``prev_logits`` (training,
        (B,T,L) detached) supply the previous head's draft embedding. For
        ``k == 0`` neither is used.

        ``rolled_state`` (iter-4): head-k 的 rolled predictor 状态 — 训练
        ``(B,L,D)`` (tgt_enc 移位 k-1)，推理 ``(B,1,D)``。

        ``rolled_hidden`` (iter-10): draft 序列模型隐藏态 — 训练
        ``(B,L,D)`` (h_seq 移位 k-1)，推理 ``(B,1,D)`` (cache 末位)。
        """
        predictor, acoustic_term, base_term = shared
        fused = predictor.unsqueeze(1)  # (B,1,L,D)
        if base_term is not None:
            fused = fused + base_term
        if acoustic_term is not None:
            fused = fused + acoustic_term.unsqueeze(2)
        if rolled_state is not None and self.chain_state_proj is not None:
            state_term = self.chain_state_proj(rolled_state)
            if state_term.dim() == 3:
                state_term = state_term.unsqueeze(1)
            fused = fused + state_term
            proj_heads = getattr(self, "chain_state_proj_heads", None)
            if proj_heads is not None and k < len(proj_heads):
                head_term = proj_heads[k](rolled_state)
                if head_term.dim() == 3:
                    head_term = head_term.unsqueeze(1)
                fused = fused + head_term
        if rolled_hidden is not None and self.draft_state_proj is not None:
            hid_term = self.draft_state_proj(rolled_hidden)
            if hid_term.dim() == 3:
                hid_term = hid_term.unsqueeze(1)
            fused = fused + hid_term
        if k >= 1:
            if draft_ids is not None:
                ids = draft_ids
            elif prev_logits is not None:
                ids = prev_logits.argmax(dim=-1).detach()  # (B,T,L)
            else:
                ids = None
            if ids is not None:
                if ids.dim() == 4:
                    ids = ids.squeeze(-1)
                emb = self.chain_draft_proj(self.chain_draft_embed(ids))
                if emb.dim() == 3:
                    emb = emb.unsqueeze(1)  # (B,1,1,D) inference → broadcast
                fused = fused + emb
        normalized = self.context_norm(fused)
        extra = getattr(self, "context_mlp_extra", None)
        delta = self.context_mlp(normalized) + (
            extra(normalized) if extra is not None else 0.0
        )
        mlp_heads = getattr(self, "context_mlp_heads", None)
        if mlp_heads is not None and k < len(mlp_heads):
            delta = delta + mlp_heads[k](normalized)
        return delta

    def forward(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor] = None,
        target_encodings: Optional[Tensor] = None,
        chain_drafts: Optional[List[int]] = None,
        chain_states: Optional[List[Tensor]] = None,
        chain_hiddens: Optional[List[Tensor]] = None,
        chain_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor], Optional[Tensor]]:
        """计算所有 head 的 logits (训练: 全网格; 推理: 单点)。

        Args:
            joint: ``(B,T,L,D)`` joiner fused features.
            source_encodings: ``(B,T,D)`` encoder outputs — 训练全序列,
                推理恰为当前帧 ``(B,1,D)``（无窗）。
            target_encodings: ``(B,L,D)`` predictor outputs。
            chain_drafts: chain 模式推理 — 已确定的前序 draft id 列表
                (d_1..d_{k-1}); head-k 消费 chain_drafts[k-1]。
            chain_states: chain 模式推理 — K 个 (B,1,D) predictor 滚动状态;
                head-k 消费 chain_states[k]。
            chain_hiddens: chain 模式推理 (iter-10) — 逐 head 的 draft 序列
                模型隐藏态 (B,1,D); head-k 消费 chain_hiddens[k]
                (消费完 y_u + d_1..d_{k-1} 后)。
            chain_tokens: chain 模式训练 — ``[BOS, y_1..y_U]`` (B, P) long,
                draft 序列模型 teacher-forcing 输入。

        Returns:
            ``(logits_base, logits_future, logits_lm)``。head 0 是精确共享
            Joiner 投影；head-k 预测 ``y[u+k]``；logits_lm 仅训练且配置了
            draft_seq_model 时非 None（列 l 预测 prep[l+1]）。
        """
        logits_base = self.w_vocab(joint)
        if self.chain_states:
            K = self.num_heads
            shared = self._chain_shared_terms(
                joint, source_encodings, target_encodings, logits_base,
            )
            logits_future: List[Tensor] = []
            inference = chain_drafts is not None

            rolled: List[Optional[Tensor]] = [None] * K
            rolled_hidden: List[Optional[Tensor]] = [None] * K
            h_seq: Optional[Tensor] = None
            lm_logits: Optional[Tensor] = None

            if inference:
                for k in range(K):
                    if chain_states is not None and k < len(chain_states):
                        rolled[k] = chain_states[k]
                    if chain_hiddens is not None and k < len(chain_hiddens):
                        rolled_hidden[k] = chain_hiddens[k]
            else:
                L = joint.size(2)
                padded = F.pad(target_encodings, (0, 0, 0, K), value=0.0)
                for k in range(K):
                    rolled[k] = padded[:, k : k + L, :]
                if self.seq_model is not None:
                    if chain_tokens is None:
                        raise ValueError(
                            "chain 模式训练 + draft_seq_model 需要 chain_tokens "
                            "(BOS 前置的 target 序列)"
                        )
                    h_seq = self.seq_model.forward_full(chain_tokens)  # (B,P,D)
                    padded_h = F.pad(h_seq, (0, 0, 0, K), value=0.0)
                    for k in range(K):
                        rolled_hidden[k] = padded_h[:, k : k + L, :]
                    if self.draft_lm_proj is not None:
                        lm_logits = self.w_vocab(self.draft_lm_proj(h_seq))

            for k in range(K):
                draft_tok: Optional[Tensor] = None
                prev = None
                if k >= 1:
                    if inference:
                        if k - 1 < len(chain_drafts):
                            draft_tok = torch.full(
                                (joint.size(0), 1, 1), int(chain_drafts[k - 1]),
                                dtype=torch.long, device=joint.device,
                            )
                        # else: incremental inference — fallback to previous
                        # head argmax keeps the graph well-formed; unused.
                        else:
                            prev = logits_future[k - 1].detach()
                    else:
                        prev = logits_future[k - 1].detach()
                delta_k = self._chain_head_delta(
                    shared, joint, k,
                    draft_ids=draft_tok,
                    prev_logits=prev,
                    rolled_state=rolled[k],
                    rolled_hidden=rolled_hidden[k],
                )
                logits_k = self.w_vocab(self.res_blocks[k](joint + delta_k))
                logits_future.append(logits_k)
            return logits_base, logits_future, lm_logits

        medusa_input = joint + self._context_delta(
            joint, source_encodings, target_encodings, logits_base
        )
        logits_future = []
        for block in self.res_blocks:
            logits_future.append(self.w_vocab(block(medusa_input)))
        return logits_base, logits_future, None


# ═══════════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════════


def shifted_ce_loss(
    logits_future: List[Tensor],
    targets: Tensor,
    weights: List[float],
    target_lengths: Tensor,
    src_lengths: Tensor,
    blank_idx: Optional[int] = None,
    future_state_offset: int = 0,
) -> Tensor:
    r"""Σ λ_k · CE(head_k 在 (t,u) 的 logits, targets[u+k])。

    ``targets`` 是不含 blank 的真实 label 序列；``blank_idx`` 作为额外
    防线用于屏蔽可能混入的 blank 标签。每个 head 使用自己的
    ``u+k < target_lengths`` mask，所有全 padding 的 batch 不会产生 NaN。
    """
    if not logits_future:
        return targets.new_zeros(())
    future_state_offset = max(0, int(future_state_offset))
    B, T, _, V = logits_future[0].shape
    U = targets.size(1)
    total = targets.new_zeros((), dtype=logits_future[0].dtype)
    tl = target_lengths.to(targets.device)
    sl = src_lengths.to(targets.device)
    t_idx = torch.arange(T, device=targets.device).unsqueeze(0)
    valid_t = t_idx < sl.unsqueeze(1)

    for k_idx, logits_k in enumerate(logits_future):
        k = k_idx + 1
        positions = min(U - k, logits_k.size(2) - future_state_offset)
        if positions <= 0:
            continue
        pred = logits_k[
            :, :, future_state_offset : future_state_offset + positions, :
        ].reshape(-1, V)
        label = targets[:, k : k + positions]
        label = label.unsqueeze(1).expand(-1, T, -1)
        u_idx = torch.arange(positions, device=targets.device).unsqueeze(0)
        valid_u = (u_idx + k) < tl.unsqueeze(1)
        valid = valid_t.unsqueeze(2) & valid_u.unsqueeze(1)
        if blank_idx is not None:
            valid = valid & (label != blank_idx)
        flat_valid = valid.reshape(-1)
        if not flat_valid.any():
            continue
        ce = F.cross_entropy(
            pred[flat_valid], label.reshape(-1)[flat_valid].long(), reduction="mean"
        )
        total = total + weights[k_idx] * ce
    return total


def lm_aux_loss(
    logits_lm: Tensor,
    prepended_targets: Tensor,
    prepended_target_lengths: Tensor,
) -> Tensor:
    """纯 token 侧 next-token CE（LM 辅助损失，iter-10）。

    ``logits_lm``: (B, P, V) — 列 l 的隐藏态 = 消费完 prep[l] 后；
    目标 = prep[l+1]（y_{l+1}）。mask：``l+1 < prepended_target_lengths``
    （即 l+1 ≤ U_sample）。无 t 轴、无 blank（targets 为 raw label 序列）。
    """
    B, P, V = logits_lm.shape
    positions = P - 1
    if positions <= 0:
        return logits_lm.new_zeros(())
    pred = logits_lm[:, :-1, :].reshape(-1, V)
    label = prepended_targets[:, 1:]
    l_idx = torch.arange(positions, device=logits_lm.device).unsqueeze(0)
    valid = l_idx < (prepended_target_lengths.to(logits_lm.device) - 1).unsqueeze(1)
    flat_valid = valid.reshape(-1)
    if not flat_valid.any():
        return logits_lm.new_zeros(())
    return F.cross_entropy(
        pred[flat_valid], label.reshape(-1)[flat_valid].long(), reduction="mean"
    )


def kl_to_base_loss(
    logits_future: List[Tensor],
    logits_base: Tensor,
    beta: float,
    src_lengths: Tensor,
    target_lengths: Tensor,
    future_state_offset: int = 0,
    chain_align: bool = False,
) -> Tensor:
    r"""β · Σ KL(head_k(u) ‖ base(u+shift).detach()).

    Future head ``k`` is aligned with the base distribution at the predictor
    position that corresponds to the same future token. The mask uses
    ``u+k < target_lengths`` independently for every head, so padding and
    unavailable future labels do not dilute the auxiliary loss.

    With ``chain_align=True`` (chain mode), head ``k`` at L-position ``l`` is
    conditioned on the rolled state ``tgt_enc[l+k-1]``, matching the base
    logits at the same L-position (base@l+k-1), not ``l+k``.
    """
    if beta <= 0 or not logits_future:
        return logits_base.new_zeros(())
    future_state_offset = max(0, int(future_state_offset))
    _, T, L, _ = logits_base.shape
    dev = logits_base.device
    sl = src_lengths.to(dev)
    tl = target_lengths.to(dev)
    t_idx = torch.arange(T, device=dev).unsqueeze(0)
    valid_t = t_idx < sl.unsqueeze(1)
    total = logits_base.new_zeros(())

    for k_idx, logits_k in enumerate(logits_future):
        k = k_idx + 1
        positions = min(logits_k.size(2) - future_state_offset, L - k)
        if positions <= 0:
            continue
        u_idx = torch.arange(positions, device=dev).unsqueeze(0)
        valid_u = (u_idx + k) < tl.unsqueeze(1)
        mask = valid_t[:, : logits_k.size(1), None] & valid_u[:, None, :]
        if not bool(mask.any()):
            continue
        future = logits_k[
            :, :, future_state_offset : future_state_offset + positions, :
        ]
        # Chain mode: head-k@l is conditioned on tgt_enc[l+k-1] and predicts
        # targets[l+k-1]; align to the base distribution at l+k-1.
        base_shift = (k - 1) if chain_align else k
        base = logits_base[:, :, base_shift : base_shift + positions, :]
        log_p0 = F.log_softmax(base, dim=-1).detach()
        log_pk = F.log_softmax(future, dim=-1)
        kl = (log_pk.exp() * (log_pk - log_p0)).sum(dim=-1)
        total = total + (kl * mask).sum() / mask.sum()
    return beta * total


def base_next_token_ce_loss(
    logits_future: List[Tensor],
    logits_base: Tensor,
    weights: List[float],
    target_lengths: Tensor,
    src_lengths: Tensor,
    blank_idx: int,
    future_state_offset: int = 1,
    max_blank: int = 4,
    chain_align: bool = False,
) -> Tensor:
    r"""Distill the verifier's next emitted token into future heads.

    In post-emission mode, head ``k`` is queried at predictor state ``u+1``
    while the verifier reaches the corresponding token at state ``u+k``.
    The verifier may emit blank for a few acoustic frames before that token,
    so the target is the first non-blank argmax in the bounded frame window.
    This auxiliary objective matches the actual acceptance event and does not
    backpropagate through the base logits.
    """
    if not logits_future:
        return logits_base.new_zeros(())
    future_state_offset = max(0, int(future_state_offset))
    max_blank = max(0, int(max_blank))
    _, T, L, _ = logits_base.shape
    dev = logits_base.device
    sl = src_lengths.to(dev)
    tl = target_lengths.to(dev)
    valid_t = torch.arange(T, device=dev).unsqueeze(0) < sl.unsqueeze(1)
    base_ids = logits_base.argmax(dim=-1).detach()
    total = logits_base.new_zeros(())

    for k_idx, logits_k in enumerate(logits_future):
        k = k_idx + 1
        positions = min(logits_k.size(2) - future_state_offset, L - k)
        if positions <= 0:
            continue
        # State u+k is the verifier state used to emit the k-th future token.
        # Chain mode: head-k@l predicts targets[l+k-1] conditioned on the
        # rolled state tgt_enc[l+k-1]; the verifier's decision for that token
        # lives at base L-position l+k-1.
        base_shift = (k - 1) if chain_align else k
        candidates = base_ids[:, :, base_shift : base_shift + positions]
        chosen = torch.full_like(candidates, blank_idx)
        found = torch.zeros_like(candidates, dtype=torch.bool)
        for delta in range(max_blank + 1):
            if delta == 0:
                shifted = candidates
            else:
                shifted = F.pad(
                    candidates[:, delta:, :],
                    (0, 0, 0, delta),
                    value=blank_idx,
                )
            nonblank = shifted != blank_idx
            take = nonblank & ~found
            chosen = torch.where(take, shifted, chosen)
            found = found | nonblank

        u_idx = torch.arange(positions, device=dev).unsqueeze(0)
        valid_u = (u_idx + k) < tl.unsqueeze(1)
        valid = valid_t.unsqueeze(2) & valid_u.unsqueeze(1) & found
        flat_valid = valid.reshape(-1)
        if not flat_valid.any():
            continue
        pred = logits_k[
            :, :, future_state_offset : future_state_offset + positions, :
        ].reshape(-1, logits_k.size(-1))
        ce = F.cross_entropy(pred[flat_valid], chosen.reshape(-1)[flat_valid].long())
        total = total + weights[k_idx] * ce
    return total
