"""Medusa 式多头未来标签预测器。

head 0 继续使用原始 Joiner 的 ``joint(t, u)``，作为无损 verifier 的唯一
打分锚点。未来 head 则可以在同一个 joint 上叠加一个零初始化的、流式受限
context delta。这样新 head 在初始化时仍然等价于 base，但训练后能看到
``tgt[u]``、当前声学帧和有限的 right-context，而不是只从当前 joint 猜
``y[u+k]``。

``stream_local`` 的 acoustic window 只包含当前 encoder frame 和配置允许的
lookahead frame；窗口在序列末尾用零填充，不会泄露 right-context 之外的信息。
future head 另外接收当前 base logits 的 detached 投影，让它明确知道本轮
已经由 verifier 选出的 token 分布，而不会改变 base/verifier 的输出。
在 ``draft_after_emit`` 模式中，draft 从消费当前 token 后的 predictor state
开始，head k 对齐到 ``y[u+k]``，避免用 ``g_u`` 预测本应由 ``g_{u+1}``
条件化的 token。
所有 future head 仍共享 ``joiner.linear``，因此 verifier 的 base logits
和 head 0 使用完全相同的词表投影。

iter-8 (流式合法的预测能力增强, 1 帧 lookahead 预算):
  1. 声学窗重排为 ``[t-past .. t-1, t, t+lookahead]`` — 用已提交的过去帧
     替换非法未来帧, 结构宽度不变 (warm-start 保形)。
  2. 验证器模拟扫描特征 (chain_sim): head-k 读取真 joiner 在
     ``[t, t+lookahead]`` 窗口对其 rolled state 的逐步 argmax — 窗口内的
     接受事件从「学习猜」变成「确定性读出」。
  3. 时延分解 (delay_decompose): 每头一个时延分类器预测验证器对本头
     token 的 blank-walk 时延; 期望时延 embedding 以加性调制的方式
     注入 token 路径。
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


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


class MedusaHeads(torch.nn.Module):
    r"""K 个 Medusa 残差头 + 共享词表投影。

    ``current_joint`` 保留旧的 joint-only 消融；``predictor_only`` 只给
    context 分支 predictor state；``stream_local`` 使用当前 encoder
    frame、有限 lookahead 和当前 predictor state。context 分支的最后一层
    零初始化，因此所有新 head 初始仍与 base head 完全一致。

    **链式对齐 (iter-2)**: ``chain_states=True`` 时 head-k 的 context
    分支条件于 teacher-forcing 滚动状态 ``tgt_enc[u+k]`` (吃掉
    ``y_{u+1..u+k-1}`` 之后) 而非统一的 ``tgt_enc[u+1]``。这与推理时
    「draft 喂回 predictor 滚动」严格对应: draft 全对时滚动状态 ==
    tgt_enc[u+k],信息差归零,head-k 只需在正确前缀上做一步预测。

    Args:
        input_dim: 特征维度 (=512)，ResBlock 在此维度做残差。
        num_vocab: 词表大小 (含 blank) = num_symbols+1。
        num_heads: K，未来预测头数 (head 1..K)。
        w_vocab: 共享输出投影，必须传 joiner.linear。
        input_mode: ``current_joint`` | ``predictor_only`` | ``stream_local``。
        lookahead_frames: 允许读取的后续 encoder frame 数，不含当前帧。
        chain_states: 链式对齐 (head-k 条件于 tgt_enc[u+k])。
    """

    def __init__(
        self,
        input_dim: int,
        num_vocab: int,
        num_heads: int,
        w_vocab: torch.nn.Linear,
        input_mode: str = "stream_local",
        lookahead_frames: int = 1,
        draft_after_emit: bool = False,
        chain_states: bool = False,
        past_frames: int = 0,
        chain_sim: bool = False,
        delay_decompose: bool = False,
        rhythm_feature: bool = False,
        per_token_blank: int = 4,
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

        self.num_heads = num_heads
        self.input_mode = input_mode
        self.lookahead_frames = max(0, int(lookahead_frames))
        # iter-8: 声学窗重排为 [t-past .. t-1, t, t+lookahead]。
        # past_frames=0 时窗口退化为旧布局 [t .. t+lookahead] (向后兼容)。
        self.past_frames = max(0, int(past_frames))
        self.draft_after_emit = bool(draft_after_emit)
        self.chain_states = bool(chain_states)
        # 本仓库约定 blank = 词表最后一个 id (lightning: blank_idx =
        # sp_model.get_piece_size(), num_vocab = num_symbols+1)。
        self.blank_idx = num_vocab - 1
        # 时延分类桶数: 相对偏移 0..per_token_blank-1 + not-found。
        self.per_token_blank = max(1, int(per_token_blank))
        self.res_blocks = torch.nn.ModuleList(
            [MedusaResBlock(input_dim) for _ in range(num_heads)]
        )

        # Project the acoustic window and predictor state separately before
        # combining them. This keeps full-grid training memory at O(B*T*L*D)
        # instead of materializing O(B*T*L*window*D) concatenations.
        self.context_norm: Optional[torch.nn.LayerNorm]
        self.context_mlp: Optional[torch.nn.Sequential]
        self.context_acoustic: Optional[torch.nn.Linear]
        self.context_predictor: Optional[torch.nn.Linear]
        self.context_base_logits: Optional[torch.nn.Linear]
        if input_mode == "stream_local":
            # iter-8: 窗口 = past + 当前 + lookahead (重排后宽度含过去帧;
            # iter-7 布局 past=0 时退化为 1+lookahead, 形状不变)。
            window_dim = (1 + self.past_frames + self.lookahead_frames) * input_dim
            self.context_acoustic = torch.nn.Linear(window_dim, input_dim)
            self.context_predictor = torch.nn.Linear(input_dim, input_dim)
            self.context_base_logits = torch.nn.Linear(num_vocab, input_dim)
            self.context_norm = torch.nn.LayerNorm(input_dim)
            # iter-6: residual capacity expansion. context_mlp keeps the
            # exact iter-4 two-layer structure (warm-start loads it fully);
            # context_mlp_extra adds a parallel deeper path whose output
            # projection is zero-initialized — at step 0 the module is
            # bit-identical to iter-4.
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
            # iter-7: per-head deep paths. The shared iter-6 extra path is
            # kept (warm-start loads its keys); each head additionally gets
            # its own zero-init deep path so h2/h3 escape the shared-capacity
            # bottleneck without disturbing the warm-start equivalence.
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
            self.context_acoustic = None
            self.context_predictor = torch.nn.Linear(input_dim, input_dim)
            self.context_base_logits = torch.nn.Linear(num_vocab, input_dim)
            self.context_norm = torch.nn.LayerNorm(input_dim)
            self.context_mlp = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(2 * input_dim, input_dim),
            )
        else:
            self.context_acoustic = None
            self.context_predictor = None
            self.context_base_logits = None
            self.context_norm = None
            self.context_mlp = None

        if self.context_base_logits is not None:
            # This feature is added to checkpoints that may already contain a
            # trained context MLP. Zero initialization therefore preserves an
            # old head's behavior while allowing the new signal to be learned.
            torch.nn.init.zeros_(self.context_base_logits.weight)
            torch.nn.init.zeros_(self.context_base_logits.bias)

        # Draft-conditioned chaining (iter-3): head-k (k>=2) additionally
        # consumes the embedding of the previous head's draft token. Training
        # uses the previous head's argmax (detached) on the same grid, so the
        # train and inference conditions are exactly isomorphic — no LSTM
        # rolling and no emit-gap sequence dependence.
        self.chain_draft_embed: Optional[torch.nn.Embedding]
        self.chain_draft_proj: Optional[torch.nn.Linear]
        self.chain_state_proj: Optional[torch.nn.Linear]
        if self.chain_states and num_heads > 1:
            self.chain_draft_embed = torch.nn.Embedding(num_vocab, 128)
            self.chain_draft_proj = torch.nn.Linear(128, input_dim)
            torch.nn.init.zeros_(self.chain_draft_proj.weight)
            torch.nn.init.zeros_(self.chain_draft_proj.bias)
            # iter-4: rolled-language-state input. Training uses the
            # teacher-forcing tgt_enc[l+k-1]; inference uses the predictor
            # rolled on accepted drafts (bit-exact when drafts hit).
            self.chain_state_proj = torch.nn.Linear(input_dim, input_dim)
            torch.nn.init.zeros_(self.chain_state_proj.weight)
            torch.nn.init.zeros_(self.chain_state_proj.bias)
            # iter-7: per-head state projections (zero-init, additive on top
            # of the shared one).
            self.chain_state_proj_heads = torch.nn.ModuleList([
                torch.nn.Linear(input_dim, input_dim)
                for _ in range(num_heads)
            ])
            for proj in self.chain_state_proj_heads:
                torch.nn.init.zeros_(proj.weight)
                torch.nn.init.zeros_(proj.bias)
            # iter-8: 验证器模拟扫描特征。head-k 读到真 joiner 在
            # [t, t+lookahead] 窗口对其 rolled state 的逐步 argmax 结果
            # (token/位置/置信度) — 流式窗口内的接受事件成为确定性读出。
            # 训练从 base_logits 网格 gather (零 FLOPs); 推理由
            # model.predict_heads 用真 joiner 算好窗口 logits 传入。
            self.chain_sim_enabled = bool(chain_sim)
            if self.chain_sim_enabled:
                self.chain_sim_proj = torch.nn.Linear(128 + 2, input_dim)
                torch.nn.init.zeros_(self.chain_sim_proj.weight)
                torch.nn.init.zeros_(self.chain_sim_proj.bias)
                # iter-8b: logit 级 sim 直注 — 每 head 一个 (V,V) 零初始化
                # 偏查表, logits_k += sim_bias[k][sim_id]。隐路径
                # (chain_sim_proj → res_block → w_vocab) 学习「复制 sim」
                # 很慢 (epoch0-2 平台 0.50 vs 上限 0.92); 直查表让每条样本
                # 对 bias[S] 的梯度直达, 75 类复制问题 1 个 epoch 可收敛。
                self.sim_bias_heads = torch.nn.ModuleList([
                    torch.nn.Embedding(num_vocab, num_vocab)
                    for _ in range(num_heads)
                ])
                for bias in self.sim_bias_heads:
                    torch.nn.init.zeros_(bias.weight)
            # iter-8: 时延分解。每头一个分类器预测验证器对本头 token 的
            # blank-walk 时延 (相对偏移 0..pb-1 或 not-found); 期望时延
            # embedding 经零初始化投影调制 token hidden — 因 w_vocab 是
            # 线性层, 这等价于对时延后验做 token logits 边缘化。
            self.delay_enabled = bool(delay_decompose)
            if self.delay_enabled:
                self.delay_classes = self.per_token_blank + 1
                self.delay_heads = torch.nn.ModuleList([
                    torch.nn.Linear(input_dim, self.delay_classes)
                    for _ in range(num_heads)
                ])
                self.delay_emb = torch.nn.Embedding(self.delay_classes, input_dim)
                self.delay_proj = torch.nn.Linear(input_dim, input_dim)
                torch.nn.init.zeros_(self.delay_proj.weight)
                torch.nn.init.zeros_(self.delay_proj.bias)
            # 节律特征 (rhythm_feature) 在设计中被砍掉: 训练网格无法精确
            # 重构推理端「自上次发射以来的 blank 数」(路径依赖), 会破坏
            # 训练-推理同构。参数保留仅为配置兼容。
            self.rhythm_enabled = False
            # iter-5 (abandoned): frame-level attention over the acoustic
            # window. Training destabilized (PER 0.07 → 0.50, greedy output
            # corrupted) — the modules are kept for checkpoint compatibility
            # but disabled via win_attn_enabled=False.
            self.win_attn_enabled = False
            self.win_attn_key = torch.nn.Linear(input_dim, 128)
            self.win_attn_val = torch.nn.Linear(input_dim, input_dim)
            self.win_attn_query = torch.nn.Linear(input_dim, 128)
            self.win_attn_out = torch.nn.Linear(input_dim, input_dim)
            torch.nn.init.zeros_(self.win_attn_out.weight)
            torch.nn.init.zeros_(self.win_attn_out.bias)
        else:
            self.chain_draft_embed = None
            self.chain_draft_proj = None
            self.chain_state_proj = None

        if self.context_mlp is not None:
            # Preserve the exact base-logit initialization. The context path
            # becomes active only after its final projection learns.
            torch.nn.init.zeros_(self.context_mlp[-1].weight)
            torch.nn.init.zeros_(self.context_mlp[-1].bias)

        # iter-8 开关默认值 (chain 块内按配置覆盖); 非 chain 模式恒 False。
        if not (self.chain_states and num_heads > 1):
            self.chain_sim_enabled = False
            self.delay_enabled = False

        # 与 joiner 共享输出投影：head 0 ≡ joiner 精确一致。
        self.w_vocab = w_vocab

    # ------------------------------------------------------------------
    def _anchor_window(self, source_encodings: Tensor, T: int) -> Tensor:
        """iter-8 声学窗重排: [t-past .. t-1, t, t+1..t+lookahead]。

        旧布局 (past_frames=0) 为 [t .. t+lookahead], 行为不变。开启
        past_frames 后, 窗口宽度不变 (warm-start 保形), 但把非法的未来帧
        替换为已提交的过去帧 — 流式部署时过去帧免费可得。序列起始处过去
        帧零填充。返回形状 (B, T, W, D), W = 1 + past + lookahead。

        iter-9: 仅供训练路径使用 (T = 序列长度)。推理时 T == 1, 本函数会
        把全部未来列清零并在 past_frames > 0 时把 enc[t-past] 当作当前帧,
        因此推理改由调用方传入预建窗口 (见 ``window_prebuilt``)。
        """
        B = source_encodings.size(0)
        width = 1 + self.past_frames + self.lookahead_frames
        required = T + self.lookahead_frames
        source = source_encodings
        if source.size(1) < required:
            source = F.pad(source, (0, 0, 0, required - source.size(1)))
        cols: List[Tensor] = []
        # 过去帧: t-p .. t-1 (p = past..1), 越界 (t-p < 0) 用零帧。
        for p in range(self.past_frames, 0, -1):
            past = source[:, :T].roll(p, dims=1)
            if p > 0:
                past[:, :p] = 0.0
            cols.append(past)
        cols.append(source[:, :T])  # 当前帧
        # 未来帧: t+1 .. t+lookahead, 序列尾越界部分由上面的 pad 兜底。
        for f in range(1, self.lookahead_frames + 1):
            fut = torch.zeros_like(source[:, :T])
            take = T - f
            if take > 0:
                fut[:, :take] = source[:, f:T]
            cols.append(fut)
        return torch.stack(cols, dim=2)  # (B, T, W, D)

    # ------------------------------------------------------------------
    def _sim_scan_grid(
        self,
        base_ids: Tensor,
        base_conf: Tensor,
        valid_t: Tensor,
    ) -> List[Dict[str, Tensor]]:
        """iter-8: 在 base-argmax 网格上模拟验证器扫描 (训练路径)。

        验证器对 draft k 的接受判定 = 从上一发射帧起逐帧扫
        ``argmax(joiner(enc[t'], g_{l+k}))``, 取首个非 blank (blank 预算
        per_token_blank)。这里在 (B, T, L) 网格上对每个 head k 递归执行:
        head-0 从 t 扫, head-k 从 head-(k-1) 的解析发射帧起扫 — 与
        spec_decode 的验证循环逐步一致。

        Args:
            base_ids: (B, T, L) long — joiner argmax 网格 (detach)。
            base_conf: (B, T, L) float — 对应 softmax 置信度 (detach)。
            valid_t: (B, T) bool — 有效帧掩码。

        Returns:
            K 个 dict: sim_id (B,T,L) long, sim_conf (B,T,L),
            sim_delta (B,T,L) long 相对帧偏移 0..pb, sim_found (B,T,L) bool。
            找不到时 sim_id = blank, sim_delta = pb (最后一桶), found=False。
        """
        pb = self.per_token_blank
        B, T, L = base_ids.shape
        results: List[Dict[str, Tensor]] = []
        # prev_start[b, t, l]: 本头扫描起点帧 (绝对坐标)。head-0 从网格点
        # 自己的帧 t 起 (= 验证器在 emit 后从当前帧起扫); head-k 从上一头
        # 解析发射帧的下一帧起。与推理相对窗口的 prev_emit=0 语义一致。
        t_range = torch.arange(T, device=base_ids.device).view(1, T, 1)
        prev_start = t_range.expand(B, T, L).contiguous()
        for k in range(self.num_heads):
            # head-k 的验证器状态在网格 L 位置 l'+k (rolled 状态), 扫描读
            # 该状态的 joiner argmax — 左移 k 对齐, 越界补 blank。
            ids_k = F.pad(base_ids, (0, k), value=self.blank_idx)[:, :, k : k + L]
            conf_k = F.pad(base_conf, (0, k), value=0.0)[:, :, k : k + L]
            found = torch.zeros(B, T, L, dtype=torch.bool, device=base_ids.device)
            sim_id = torch.full((B, T, L), self.blank_idx, dtype=torch.long,
                                device=base_ids.device)
            sim_conf = torch.zeros(B, T, L, dtype=base_conf.dtype,
                                   device=base_conf.device)
            sim_delta = torch.full((B, T, L), pb, dtype=torch.long,
                                   device=base_ids.device)
            for d in range(pb + 1):
                t_idx = prev_start + d
                in_bound = t_idx < T
                safe_t = t_idx.clamp(max=T - 1)
                if T == 0:
                    break
                # gather(dim=1): out[b,t,l] = ids_k[b, safe_t[b,t,l], l]
                ids = torch.gather(ids_k, 1, safe_t)
                conf = torch.gather(conf_k, 1, safe_t)
                nonblank = (ids != self.blank_idx) & in_bound & valid_t.unsqueeze(-1)
                take = nonblank & ~found
                sim_id = torch.where(take, ids, sim_id)
                sim_conf = torch.where(take, conf, sim_conf)
                sim_delta = torch.where(
                    take, torch.full_like(sim_delta, d), sim_delta
                )
                found = found | (nonblank & in_bound)
            results.append({
                "sim_id": sim_id, "sim_conf": sim_conf,
                "sim_delta": sim_delta, "found": found,
                "start": prev_start.clone(),
            })
            # 下一头从本头解析发射帧同帧起扫 (验证器接受后 t 不前进, 同帧
            # 可连发 — spec_decode 验证循环语义); 本头未命中时起点不前进。
            emit_t = (prev_start + sim_delta).clamp(max=T - 1)
            prev_start = torch.where(found, emit_t, prev_start)
        return results

    def _sim_feature(self, scan: Dict[str, Tensor]) -> Tensor:
        """把单头扫描读出投成 delta 向量 (B,T,L,D), 全零初始化投影。"""
        conf_vec = scan["sim_conf"].unsqueeze(-1)
        pos_vec = (
            scan["sim_delta"].to(conf_vec.dtype).unsqueeze(-1)
            / max(self.per_token_blank, 1)
        )
        emb = self.chain_draft_embed(scan["sim_id"])  # (B,T,L,128)
        feats = torch.cat([emb, pos_vec, conf_vec], dim=-1)
        return self.chain_sim_proj(feats)

    # ------------------------------------------------------------------
    def _delay_mod(self, dl: Tensor, norm: Tensor) -> Tensor:
        """iter-8 时延调制项: delay_proj(norm + Σ_δ p(δ)·emb_δ)。

        ``dl``: (B,T,L,C) 本头时延分类 logits。与 token 路径线性边缘化
        近似等价的加性调制 — w_vocab 线性 ⇒ 期望 embedding ≈ 期望 logits。
        """
        probs = F.softmax(dl, dim=-1)  # (B,T,L,C)
        embs = self.delay_emb.weight.to(norm.dtype)  # (C,D)
        expected = torch.einsum("btlc,cd->btld", probs, embs)
        return self.delay_proj(self.context_norm(expected + norm))

    # ------------------------------------------------------------------
    def _context_delta(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor],
        target_encodings: Tensor,
        base_logits: Optional[Tensor],
        chain_target_encodings: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Build a bounded context delta with shape ``(B,T,L,D)``.

        With ``chain_states`` and a full-sequence ``target_encodings`` the
        head-k context uses the teacher-forcing rolled state at ``u+k``
        (training). At inference ``chain_target_encodings`` supplies the
        draft-rolled states directly (one tensor per head).
        """
        if self.input_mode == "current_joint" or self.context_mlp is None:
            return torch.zeros_like(joint)

        B, T, L, _ = joint.shape
        if target_encodings.size(0) != B:
            raise ValueError(
                "target_encodings must match joint batch size: "
                f"joint={tuple(joint.shape)}, targets={tuple(target_encodings.shape)}"
            )

        if self.chain_states:
            # Draft-conditioned chaining (iter-3) — handled in forward(),
            # which builds heads sequentially. This path is unreachable in
            # production; raise to catch stale callers.
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
        # The verifier distribution is an inference-available summary of the
        # current token decision. Detach it so auxiliary head training cannot
        # alter the base model through this shortcut.
        base_context = self.context_base_logits(base_logits.detach())
        if self.input_mode == "predictor_only":
            fused = predictor.unsqueeze(1).expand(-1, T, -1, -1) + base_context
        else:
            if source_encodings is None:
                return torch.zeros_like(joint)
            if source_encodings.size(0) != B:
                raise ValueError(
                    "source_encodings must match joint batch size: "
                    f"joint={tuple(joint.shape)}, source={tuple(source_encodings.shape)}"
                )
            width = 1 + self.lookahead_frames
            source = source_encodings
            required = T + self.lookahead_frames
            if source.size(1) < required:
                source = F.pad(source, (0, 0, 0, required - source.size(1)))
            windows = source.unfold(1, width, 1)
            # Tensor.unfold appends the window axis after D: (B,T,D,W).
            windows = windows.permute(0, 1, 3, 2).contiguous()[:, :T]
            acoustic = self.context_acoustic(windows.flatten(start_dim=2))
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
        window_prebuilt: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """Shared chain inputs.

        Args:
            window_prebuilt: inference — ``source_encodings`` is already the
                ``(B, W, D)`` window for the single scored frame, built by
                ``spec_decode``. Use it verbatim instead of re-anchoring.

        Returns (predictor, flat_acoustic, base_term, attn_frames) where
        ``attn_frames`` is ``(B, T, W, D)`` — the raw per-frame window tensor
        (iter-8: past-anchored layout) for the disabled iter-5 attention.
        """
        B, T, L, D = joint.shape
        predictor = self.context_predictor(target_encodings)  # (B,L,D)
        acoustic_term: Optional[Tensor] = None
        attn_frames: Optional[Tensor] = None
        if self.input_mode == "stream_local":
            if source_encodings is None:
                raise ValueError("stream_local chain mode requires source_encodings")
            if window_prebuilt:
                # iter-9 fix: at inference T == 1, so re-anchoring would zero
                # every lookahead column (take = T - f <= 0) and, with
                # past_frames > 0, pick enc[t-past] as "current". Take the
                # caller-built window as-is.
                if T != 1:
                    raise ValueError(
                        "window_prebuilt requires a single scored frame, got "
                        f"T={T}"
                    )
                windows = source_encodings.unsqueeze(1)  # (B,1,W,D)
            else:
                windows = self._anchor_window(source_encodings, T)  # (B,T,W,D)
            expected_w = 1 + self.past_frames + self.lookahead_frames
            if windows.size(2) != expected_w:
                raise ValueError(
                    f"acoustic window width {windows.size(2)} != expected "
                    f"{expected_w} (past={self.past_frames}, "
                    f"lookahead={self.lookahead_frames})"
                )
            attn_frames = windows
            acoustic_term = self.context_acoustic(windows.flatten(start_dim=2))
        base_term: Optional[Tensor] = None
        if base_logits is not None and self.context_base_logits is not None:
            base_term = self.context_base_logits(base_logits.detach())
        return predictor, acoustic_term, base_term, attn_frames

    def _chain_head_delta(
        self,
        shared: Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]],
        joint: Tensor,
        k: int,
        draft_ids: Optional[Tensor] = None,
        prev_logits: Optional[Tensor] = None,
        rolled_state: Optional[Tensor] = None,
        sim_feature: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Context delta for chain head ``k`` (0-based).

        ``draft_ids`` (inference, (B,1,1) long) or ``prev_logits`` (training,
        (B,T,L) detached) supply the previous head's draft embedding. For
        ``k == 0`` neither is used — the head is the plain non-chain head.

        ``rolled_state`` (iter-4) is the rolled language state for head k:
        ``(B, L, D)`` at training (tgt_enc shifted by k) or ``(B, 1, D)`` at
        inference (predictor advanced on accepted drafts). It carries the LM
        context the verifier will have when reaching this token. It also
        serves as the attention query (iter-5) over the acoustic window.

        iter-8: ``sim_feature`` (B,T,L,D) 训练 / (B,1,1,D) 推理, 为验证器
        模拟扫描读出的零初始化投影。时延分类在本头 ``normalized`` 上计算,
        期望 embedding 调制本头 token 路径。

        Returns:
            ``(delta, delay_logits)`` — delay_logits 为本头时延分类
            (B,T,L,C) 或 (B,1,1,C), delay 关闭时 None。
        """
        predictor, acoustic_term, base_term, attn_frames = shared
        fused = predictor.unsqueeze(1)  # (B,1,L,D)
        if base_term is not None:
            fused = fused + base_term
        if acoustic_term is not None:
            fused = fused + acoustic_term.unsqueeze(2)
        if rolled_state is not None and self.chain_state_proj is not None:
            state_term = self.chain_state_proj(rolled_state)
            if state_term.dim() == 3:
                state_term = state_term.unsqueeze(1)  # (B,1,L,D) or (B,1,1,D)
            fused = fused + state_term
            # iter-7: per-head additive state projection
            proj_heads = getattr(self, "chain_state_proj_heads", None)
            if proj_heads is not None and k < len(proj_heads):
                head_term = proj_heads[k](rolled_state)
                if head_term.dim() == 3:
                    head_term = head_term.unsqueeze(1)
                fused = fused + head_term
            # iter-5: frame attention — query from this head's rolled state,
            # keys/values over the (B,T,W,D) window frames. Output mixes
            # additively into the fused context.
            if attn_frames is not None and getattr(self, "win_attn_enabled", False):
                B, T, L2, D = fused.shape
                frames = attn_frames  # (B,T,W,D)
                W = frames.size(2)
                keys = self.win_attn_key(frames)          # (B,T,W,128)
                vals = self.win_attn_val(frames)          # (B,T,W,D)
                # query: (B, L, D) → (B, L, 128); broadcast over T
                q = self.win_attn_query(rolled_state)     # (B,L,128) or (B,1,128)
                scores = torch.einsum(
                    "bld,btwd->bltw", q, keys
                ) / (128 ** 0.5)                          # (B,L,T,W)
                weights = F.softmax(scores, dim=-1)
                attended = torch.einsum(
                    "bltw,btwd->bltd", weights, vals
                )                                         # (B,L,T,D)
                out = self.win_attn_out(attended)         # (B,L,T,D)
                fused = fused + out.permute(0, 2, 1, 3)   # → (B,T,L,D)
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
        # iter-7: per-head deep path (zero-init output; additive)
        mlp_heads = getattr(self, "context_mlp_heads", None)
        if mlp_heads is not None and k < len(mlp_heads):
            delta = delta + mlp_heads[k](normalized)
        # iter-8: verifier-lookahead 读出 (零初始化)
        if sim_feature is not None:
            delta = delta + sim_feature
        # iter-8: 时延分类 + 期望 embedding 调制 (分类器自然初始化,
        # 调制投影零初始化 ⇒ step-0 无扰动)。
        dl: Optional[Tensor] = None
        if self.delay_enabled:
            dl = self.delay_heads[k](normalized)
            delta = delta + self._delay_mod(dl, normalized)
        return delta, dl

    def forward(
        self,
        joint: Tensor,
        source_encodings: Optional[Tensor] = None,
        target_encodings: Optional[Tensor] = None,
            chain_drafts: Optional[List[int]] = None,
            chain_states: Optional[List[Tensor]] = None,
            sim_window_logits: Optional[List[Tensor]] = None,
            window_prebuilt: bool = False,
        ) -> Tuple[Tensor, List[Tensor]]:
        """计算所有 head 的 logits (训练: 全网格; 推理: 单点/小窗口)。

        Args:
            joint: ``(B,T,L,D)`` joiner fused features.
            source_encodings: ``(B,T,D)`` encoder outputs for stream-local
                context. At inference this may be a current+lookahead window.
            target_encodings: ``(B,L,D)`` predictor outputs.
            chain_drafts: chain 模式推理时使用 — 已确定的前序 draft id 列表
                (d_1..d_{k-1}); head-k 消费 chain_drafts[k-1]。
            chain_states: chain 模式推理时使用 (iter-4) — K 个 (B,1,D)
                predictor 滚动状态; head-k 消费 chain_states[k]
                (k=0 为 draft 点状态本身)。
            sim_window_logits: iter-8 推理 — 每 head 一个 (B, W+1, V) 真
                joiner 窗口 logits (model.predict_heads 计算); 启用
                chain_sim 时 head-k 在其上做验证器模拟扫描读出。
            window_prebuilt: iter-9 — 推理时 ``source_encodings`` 已是
                spec_decode 拼好的 (B, W, D) 窗口 (W = 1+past+lookahead),
                直接作为声学窗使用, 不再调用 ``_anchor_window`` 二次开窗。

        Returns:
            ``(logits_base, logits_future)`` where head 0 is the exact shared
            Joiner projection and each future head predicts ``y[u+k]``.
            训练时的辅助量 (时延 logits / sim 扫描) 暂存在
            ``self.last_train_aux`` (dict 或 None), 供损失函数取用。
        """
        logits_base = self.w_vocab(joint)
        self.last_train_aux: Optional[Dict[str, object]] = None
        if self.chain_states:
            # Draft-conditioned chain: heads are built sequentially — head-k
            # consumes the previous head's draft (training: actual head
            # logits' argmax over the grid, detached; inference: fixed ids
            # from chain_drafts). _chain_shared_terms is precomputed once.
            K = self.num_heads
            shared = self._chain_shared_terms(
                joint, source_encodings, target_encodings, logits_base,
                window_prebuilt=window_prebuilt,
            )
            logits_future: List[Tensor] = []
            inference = chain_drafts is not None
            # Rolled states: inference passes them via chain_states; training
            # shifts tgt_enc so head-k sees the state after consuming
            # y_l..y_{l+k-1} (teacher forcing; bit-exact with inference when
            # drafts are accepted).
            rolled: List[Optional[Tensor]] = [None] * K
            if inference:
                for k in range(K):
                    if chain_states is not None and k < len(chain_states):
                        rolled[k] = chain_states[k]
            else:
                L = joint.size(2)
                padded = F.pad(target_encodings, (0, 0, 0, K), value=0.0)
                for k in range(K):
                    rolled[k] = padded[:, k : k + L, :]

            # iter-8 训练: 从 base-argmax 网格模拟验证器扫描 (零 FLOPs)。
            sim_feats: List[Optional[Tensor]] = [None] * K
            scans: Optional[List[Dict[str, Tensor]]] = None
            if not inference and self.chain_sim_enabled and logits_base is not None:
                B, T = joint.size(0), joint.size(1)
                base_ids = logits_base.argmax(dim=-1).detach()
                base_conf = (
                    F.softmax(logits_base.detach(), dim=-1)
                    .max(dim=-1).values
                )
                scans = self._sim_scan_grid(
                    base_ids, base_conf,
                    torch.ones(B, T, dtype=torch.bool, device=joint.device),
                )
                for k in range(K):
                    sim_feats[k] = self._sim_feature(scans[k])

            # iter-8 推理: 每 head 在自己的窗口 logits 上顺序扫描。
            sim_scan_infer: List[Optional[Dict[str, Tensor]]] = [None] * K
            if inference and self.chain_sim_enabled and sim_window_logits:
                prev_emit = torch.zeros(
                    joint.size(0), dtype=torch.long, device=joint.device
                )
                for k in range(min(K, len(sim_window_logits))):
                    win = sim_window_logits[k].detach()  # (B, W+1, V)
                    W = win.size(1)
                    probs = F.softmax(win, dim=-1)
                    ids = win.argmax(dim=-1)            # (B, W+1)
                    conf = probs.max(dim=-1).values      # (B, W+1)
                    sim_id = torch.full((joint.size(0),), self.blank_idx,
                                        dtype=torch.long, device=joint.device)
                    sim_conf = torch.zeros(joint.size(0), device=joint.device)
                    sim_delta = torch.full((joint.size(0),),
                                           self.per_token_blank,
                                           dtype=torch.long, device=joint.device)
                    found = torch.zeros(joint.size(0), dtype=torch.bool,
                                        device=joint.device)
                    for d in range(self.per_token_blank + 1):
                        idx = prev_emit + d
                        ok = idx < W
                        safe = idx.clamp(max=W - 1)
                        id_b = ids.gather(1, safe.unsqueeze(1)).squeeze(1)
                        c_b = conf.gather(1, safe.unsqueeze(1)).squeeze(1)
                        take = ok & (id_b != self.blank_idx) & ~found
                        sim_id = torch.where(take, id_b, sim_id)
                        sim_conf = torch.where(take, c_b, sim_conf)
                        sim_delta = torch.where(
                            take, torch.full_like(sim_delta, d), sim_delta
                        )
                        found = found | take
                    sim_scan_infer[k] = {
                        "sim_id": sim_id, "sim_conf": sim_conf,
                        "sim_delta": sim_delta, "found": found,
                    }
                    emit = (prev_emit + sim_delta).clamp(max=W - 1)
                    prev_emit = torch.where(found, emit, prev_emit)
                # 暂存推理扫描读出 (供诊断/测试对比训练网格路径)
                self.last_infer_scan = sim_scan_infer

            aux_delay: List[Optional[Tensor]] = [None] * K
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
                        # else: incremental inference — this head's draft is
                        # not fixed yet; its logits are unused by the caller.
                        # Fall back to the training-style previous head argmax
                        # so the graph stays well-formed.
                        else:
                            prev = logits_future[k - 1].detach()
                    else:
                        prev = logits_future[k - 1].detach()
                sim_k: Optional[Tensor] = None
                if not inference:
                    sim_k = sim_feats[k]
                elif sim_scan_infer[k] is not None:
                    # 单点读出: (B,) → (B,1,·) → proj → (B,1,1,D)
                    sc = sim_scan_infer[k]
                    assert sc is not None and self.chain_sim_proj is not None
                    emb = self.chain_draft_embed(sc["sim_id"])      # (B,128)
                    pos = (sc["sim_delta"].to(emb.dtype)
                           / max(self.per_token_blank, 1)).unsqueeze(1)
                    cf = sc["sim_conf"].to(emb.dtype).unsqueeze(1)
                    sim_k = self.chain_sim_proj(
                        torch.cat([emb, pos, cf], dim=-1)
                    ).unsqueeze(1)                                   # (B,1,1,D)
                delta_k, dl_k = self._chain_head_delta(
                    shared, joint, k,
                    draft_ids=draft_tok,
                    prev_logits=prev,
                    rolled_state=rolled[k],
                    sim_feature=sim_k,
                )
                aux_delay[k] = dl_k
                logits_k = self.w_vocab(self.res_blocks[k](joint + delta_k))
                # iter-8b: logit 级 sim 直注 (零初始化偏查表)。只在 found
                # 位置应用 — 稠密训练网格上大多数位置 sim=blank (发射稀疏),
                # 若不遮挡, 偏置在 blank 行学出巨大的反 blank 信号, 真实
                # token 行的复制梯度被稀释 (实测 diag≈0)。found 位置 ==
                # 推理时 draft 实际发生并被接受的条件, 训练/推理同规则。
                if self.chain_sim_enabled:
                    if scans is not None:
                        bias = self.sim_bias_heads[k](scans[k]["sim_id"])
                        mask = scans[k]["found"].unsqueeze(-1).to(bias.dtype)
                        logits_k = logits_k + bias * mask
                    elif sim_scan_infer[k] is not None:
                        sc = sim_scan_infer[k]
                        bias = self.sim_bias_heads[k](sc["sim_id"])
                        m = sc["found"].to(bias.dtype).unsqueeze(-1)
                        logits_k = logits_k + (bias * m).unsqueeze(1).unsqueeze(1)
                logits_future.append(logits_k)
            if not inference:
                self.last_train_aux = {
                    "delay_logits": aux_delay,
                    "sim_scans": scans,
                }
            return logits_base, logits_future

        medusa_input = joint + self._context_delta(
            joint, source_encodings, target_encodings, logits_base
        )
        logits_future = []
        for block in self.res_blocks:
            logits_future.append(self.w_vocab(block(medusa_input)))
        return logits_base, logits_future


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


def kl_to_base_loss(
    logits_future: List[Tensor],
    logits_base: Tensor,
    beta: float,
    src_lengths: Tensor,
    target_lengths: Tensor,
    future_state_offset: int = 0,
    chain_align: bool = False,
) -> Tensor:
    r"""β · Σ KL(head_k(u) ‖ base(u+k).detach()).

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


def delay_ce_loss(
    aux_delay: List[Optional[Tensor]],
    sim_scans: Optional[List[Dict[str, Tensor]]],
    weights: List[float],
    target_lengths: Tensor,
    src_lengths: Tensor,
    future_state_offset: int = 1,
) -> Tensor:
    r"""iter-8: 时延分类 CE — 蒸馏验证器对本头 token 的 blank-walk 时延。

    目标 = ``sim_scans[k]["sim_delta"]`` (``_sim_scan_grid`` 的解析结果,
    相对偏移 0..per_token_blank, not-found 桶不产生损失)。时延标签只在
    训练网格上存在 (标签不是输入), 不违反流式约束。
    """
    if not aux_delay or sim_scans is None:
        return torch.zeros(())
    future_state_offset = max(0, int(future_state_offset))
    dev = target_lengths.device
    sl = src_lengths.to(dev)
    tl = target_lengths.to(dev)
    total = torch.zeros((), device=dev)
    for k_idx, dl in enumerate(aux_delay):
        if dl is None or k_idx >= len(sim_scans):
            continue
        scan = sim_scans[k_idx]
        k = k_idx + 1
        positions = min(dl.size(2) - future_state_offset, scan["found"].size(2) - 0)
        offset = future_state_offset
        if positions <= 0:
            continue
        T = dl.size(1)
        t_idx = torch.arange(T, device=dev).unsqueeze(0)
        valid_t = t_idx < sl.unsqueeze(1)
        u_idx = torch.arange(positions, device=dev).unsqueeze(0)
        valid_u = (u_idx + k) < tl.unsqueeze(1)
        label = scan["sim_delta"][:, :, offset : offset + positions]
        found = scan["found"][:, :, offset : offset + positions]
        valid = (
            valid_t.unsqueeze(2)
            & valid_u.unsqueeze(1)
            & found
            & (label < dl.size(-1) - 1)
        )
        flat_valid = valid.reshape(-1)
        if not bool(flat_valid.any()):
            continue
        pred = dl[:, :, offset : offset + positions, :].reshape(
            -1, dl.size(-1)
        )
        ce = F.cross_entropy(
            pred[flat_valid], label.reshape(-1)[flat_valid].long(),
            reduction="mean",
        )
        total = total + weights[k_idx] * ce
    return total


def verifier_distillation_loss(
    logits_future: List[Tensor],
    logits_base: Tensor,
    sim_scans: Optional[List[Dict[str, Tensor]]],
    weights: List[float],
    target_lengths: Tensor,
    src_lengths: Tensor,
    future_state_offset: int = 1,
) -> Tensor:
    r"""iter-8: 验证器软目标蒸馏。

    对每个 head-k, 在其解析发射帧 ``(start_k + δ_k)`` 与 rolled 状态
    ``l + k`` 处取 base 分布 ``softmax(base_logits[t_emit, l+k])`` 作为
    软目标 — 即「验证器真正发射本头 token 时看到的完整分布」。训练网格
    全知, 可取窗口外的帧: 这教会头从合法输入预测非法未来帧的 joint 输出,
    是流式约束下预测能力的来源。not-found 位置不产生损失。
    """
    if not logits_future or sim_scans is None:
        return logits_base.new_zeros(())
    future_state_offset = max(0, int(future_state_offset))
    offset = future_state_offset
    _, T, L, V = logits_base.shape
    dev = logits_base.device
    sl = src_lengths.to(dev)
    tl = target_lengths.to(dev)
    t_idx = torch.arange(T, device=dev).unsqueeze(0)
    valid_t = t_idx < sl.unsqueeze(1)
    log_base = F.log_softmax(logits_base.detach(), dim=-1)
    total = logits_base.new_zeros(())
    for k_idx, logits_k in enumerate(logits_future):
        if k_idx >= len(sim_scans):
            continue
        scan = sim_scans[k_idx]
        k = k_idx + 1
        positions = min(
            logits_k.size(2) - offset,
            scan["found"].size(2) - 0,
            L - k,
        )
        if positions <= 0:
            continue
        u_idx = torch.arange(positions, device=dev).unsqueeze(0)
        valid_u = (u_idx + k) < tl.unsqueeze(1)
        found = scan["found"][:, :, offset : offset + positions]
        valid = valid_t.unsqueeze(2) & valid_u.unsqueeze(1) & found
        flat_valid = valid.reshape(-1)
        if not bool(flat_valid.any()):
            continue
        # 解析发射帧: t_emit = start + δ (逐 (b, t, l) 网格点)。
        start = scan["start"][:, :, offset : offset + positions]
        delta = scan["sim_delta"][:, :, offset : offset + positions]
        t_emit = (start + delta).clamp(max=T - 1)
        l_state = (offset + u_idx.unsqueeze(0) + k_idx).clamp(max=L - 1)
        l_state = l_state.expand_as(t_emit)
        # 目标: base 在 (t_emit, l_state) 的 log 分布。先 L 轴 gather
        # (rolled 状态列), 再 T 轴 gather (解析发射帧)。
        sub = log_base.gather(
            2, l_state.unsqueeze(-1).expand(-1, -1, -1, V)
        )  # (B,T,P,V)
        sub = sub.gather(
            1, t_emit.unsqueeze(-1).expand(-1, -1, -1, V)
        )  # (B,T,P,V)
        pred = logits_k[:, :, offset : offset + positions, :]
        log_pk = F.log_softmax(pred, dim=-1)
        soft_ce = -(sub.exp() * log_pk).sum(dim=-1)  # (B,T,P)
        total = total + weights[k_idx] * (
            soft_ce.reshape(-1)[flat_valid].mean()
        )
    return total
