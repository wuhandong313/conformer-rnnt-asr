# 流式识别实现方案（streaming 分支）

## Context

现有系统是整句（utterance）识别：`ConformerEncoder` 全句双向注意力 + `DynamicCompression`
整句峰值检测 + 整句 greedy/spec 解码。目标是在 **`streaming` 分支**（自 `multihead` 切出，
已创建）上实现 **chunked 流式识别**，满足四条硬约束：

1. **Right context 限制**：每个 chunk 有 `left_ctx / chunk_size / right_ctx`（mel 帧数），
   `right_ctx` 是模型可见的唯一"未来"，不得超出（不得泄露）。
2. **Greedy 解码与原来一致**：streaming greedy 输出与 offline greedy（同一模型、同一
   checkpoint、同一音频）**逐 token 完全一致**（结构性保证，见一致性论证）。
3. **Medusa 只作显示用 lookahead**：draft head 输出仅作为 partial-hypothesis 的"预测
   lookahead"展示，不参与验证、不回写、不影响解码轨迹。
4. **独立维护**：不改 `conformer_rnnt/` 现有实验代码的任何行为；流式功能全部是新增
   文件，只读复用现有权重与组件。原 `multihead` 实验管线不受影响。

## 现状核查（方案依据的关键代码事实）

- **Encoder 全注意力**：`ConformerEncoder` = `input_linear → TimeReduction(stride 4) →
  conformer1(2层) → DynamicCompression → conformer2(8层) → output_linear + LN`，全部整句
  前向。流式必须分块。
- **TimeReduction stride=4**：mel 帧必须按 stride 对齐切块，否则落帧。`server.yaml`
  hop=160 → 100fps mel → 25fps reduced（每 reduced 帧 40ms）。
- **Right context 的语义**：数据管线在输入尾部 pad `right_context_length` mel 帧；
  encoder 内 `rc_frames = rc // stride`（=1）个 reduced 帧由压缩模块保留不压缩。已知索引
  混淆（reduced 域减 mel 域量）train/eval 一致，本方案不修（复用现有 checkpoint 就必须
  保持其行为）。
- **Greedy 解码严格因果**（`spec_decode.greedy_decode`）：每步只需 `enc[t]` + predictor
  状态（LSTM state + 已发 token）。blank→t+=1；emit 后 t 不变。这是"流式 = offline"
  可证明的核心依据。
- **Medusa draft 窗口**：`predict_heads` 锚定当前帧，`stream_local` 窗 = 1+lookahead
  （≤ rc/stride = 1）reduced 帧。preview 不需要 chunk 右界以外的任何未来。
- **Checkpoint 完全复用**：`val-naive-h2-frz-sl1-post-*.ckpt`（服务器）。snapshot 校验
  `compression.filter=naive`、`medusa.num_heads=2`。不重训、不新增权重键。

## 方案总览

```
audio stream ──> StreamingFeatureExtractor   逐 chunk 提 mel（chunk 边界 stride 对齐）
                     │  chunk 窗 = left_ctx + C + right_ctx（mel 帧）
                     ▼
               ChunkedEncoder               复用现有 encoder 权重，overlap 重算：
                     │                        每次前向 left+C+right 窗口，
                     │                        只提交 chunk 中段的 reduced 帧为"已定帧"
                     ▼
               StreamingGreedyDecoder        复用 greedy_decode 的状态推进 helper，
                     │                        committed 帧到达即推进，partial hypothesis 更新
                     ▼
               MedusaPreview（可选）          在当前 (t, g_last) 调 predict_heads，
                                              返回 lookahead tokens 仅作显示，绝不回写
```

### 核心设计决策：recompute-overlap 的 chunked encoder

流式 Conformer 两条常规路线：(a) 改造注意力为 chunked/limited attention（要改权重结构/
重训）；(b) 不改结构，每 chunk 用 `left+C+right` 窗口整段前向、只保留中段输出（overlap
重算）。**选 (b)**：零结构改动、checkpoint 直接复用，代价是编码计算量放大
`(L+C+R)/C` 倍。

- **左文窗（left_ctx）**：为 chunk 中段提供 conv 与 attention 的左感受野，使中段帧编码
  与 offline 一致（conv 精确；attention 为有限近似，误差随 L 衰减）。
- **右文窗（right_ctx）**：等于训练时 rc 的限制（4 mel = 1 reduced 帧），保证模型可见
  未来不超训练条件，也不泄露。
- **压缩模块跨 chunk**：`DynamicCompression` 的 diff/峰值检测是整句的。分块后每 chunk
  在自己窗口（含 left_ctx 历史）内运行压缩：窗口内 diff 值与 offline 相同（历史帧参与
  diff 计算），chunk 边界对齐 stride 且压缩分割点落在中段，边界效应被限制在丢弃的
  overlap 区。`filter: none` 时压缩为 identity，退化为纯 overlap 重算，**零边界效应**。

### 与 offline 的一致性论证（= 验收标准）

**Decoder 结构性无损**：greedy 每步的 argmax 只依赖 `enc[t]` + predictor 状态；两者在
streaming 与 offline 下逐位相同（encoder 近似误差除外）⇒ 输出逐 token 一致。不依赖
训练，可断言验证。

**Encoder 有限近似**：
- conv（depthwise kernel=31，因果左感受野 15 reduced 帧）：`left_ctx ≥ 15 reduced` 即
  精确。初值 24 reduced ✓。
- 自注意力（左感受野理论无限）：left_ctx 有限 ⇒ 残余近似误差，随 L 增大衰减。初值
  24 reduced（0.96s），用实验标定（验证节）。
- 压缩（naive 峰值检测）：跨 chunk 边界的峰值选择可能差异，控制为落在 overlap 丢弃区
  /中段边界附近，验收用 PER 差阈值。

## 文件布局（独立维护）

全部新增，现有文件 **0 修改**：

```
conformer_rnnt/streaming/
├── __init__.py        # 导出 StreamingRecognizer
├── plan.py            # ChunkPlan：left_ctx/chunk_size/right_ctx + stride 对齐校验
├── features.py        # StreamingFeatureExtractor：逐 chunk mel（hop 对齐、尾 chunk 零 pad）
├── encoder.py         # ChunkedEncoder：overlap 重算，缓存左文历史，提交中段已定帧
├── decoder.py         # StreamingGreedyDecoder：事件驱动，帧到达即推进
└── preview.py         # MedusaPreview：只读 lookahead（不回写任何状态）
streaming_eval.py      # CLI：流式评测 + --check-offline 一致性断言
streaming_tests.py     # 可独立运行的测试（无 pytest 依赖，仓库惯例）
configs/streaming.yaml # 新 config：server.yaml 的 model 部分 + 新 streaming 节
docs/streaming_design.md
```

依赖方向单向：`streaming/` → import 现有 `ConformerRNNT`（`transcribe`/`predict`/`join`/
`predict_heads` 接口）与 `spec_decode` 的 helper。现有代码不 import streaming。

## 组件设计

### 1. `ChunkPlan`（plan.py）
- 字段：`left_ctx, chunk_size, right_ctx`（mel 帧），`stride`（从 config 读）。
- 校验：三者均 stride 的倍数（或明确报错）；`right_ctx ≤ 训练 rc`（读 checkpoint snapshot
  对齐）。
- 产出 chunk 序列：`[s_i, s_i+L+C+R)`，`s_{i+1} = s_i + C`；尾 chunk 不足时零 pad。

### 2. `StreamingFeatureExtractor`（features.py）
- 维护 mel 帧缓冲；音频逐块到达即追加提帧（hop=160 样本/帧）。
- `ready_chunks()`：缓冲中满足下一 chunk 窗口的就绪 chunk 列表。
- flush：音频结束，零 pad 补尾 chunk，产出 `is_final` 标记。
- mel transform 复用 `_get_mel_transform()` 同款参数（流式侧自建实例，逐 chunk 无状态
  —— MellSpectrogram 逐帧因果，无未来依赖）。

### 3. `ChunkedEncoder`（encoder.py）
- 输入：chunk 窗（`L+C+R` mel 帧）→ 直接调 `model.transcriber(window, lengths)`（现有
  `ConformerEncoder.forward` 支持任意长度，无需改动）。
- 提交规则：输出的 reduced 帧中，只取对应 **中段 `[s_i+L, s_i+L+C)`** 的帧（按
  `stride` 折算的索引区间）为已定帧；overlap 区输出丢弃。
- 左文缓存：`s_{i+1}` 窗的左文 = 前一窗的尾部帧（encoder 输入侧直接从特征缓冲取，
  无需缓存 encoder 状态——overlap 重算本质是无状态的）。
- 输出：`push(chunk_mel) -> (committed_enc_frames, is_final)`，流式供给 decoder。

### 4. `StreamingGreedyDecoder`（decoder.py）
- 复用 `spec_decode` 的 `_init_decode/_make_tok/_join_argmax/_advance_predictor`（import
  私有 helper，不改原文件；如需可小包装）。
- 状态：`t`（当前帧索引）、`g_last/state`（predictor）、`out`（已发 token）。
- `push(enc_frames)`：新 committed 帧追加到 `enc` 缓冲，然后照 `greedy_decode` 的规则
  尽量推进（`t < len(enc)` 时循环：blank→t+=1；emit→append + predictor 吃 token）。
- **性质**：每步 argmax 只依赖 `enc[t]` + predictor 状态 ⇒ 与 offline 逐步相同。
- `partial()`：返回当前 `out`（已确定的 token）。

### 5. `MedusaPreview`（preview.py）
- `lookahead(decoder_state) -> List[int]`：在当前 `(t, g_last)` 调
  `model.predict_heads(enc[t : t+1+lookahead], g_last)`，取各 future head 的 argmax，
  返回 k 个 token id 供显示（"接下来可能说的音素"）。
- **只读铁律**：不修改 `t/g_last/state/out`；单元测试断言调用前后状态逐位不变。
- 窗口 ≤ chunk 的 right_ctx（1 reduced 帧）⇒ 无额外等待、无额外延迟。注意
  `draft_after_emit=true` 的训练语义在 preview 里**不适用**（preview 不发射，直接在当前
  状态查询，显示"如果继续，接下来会是什么"）。

### 6. `StreamingRecognizer`（__init__.py 组装）
```python
recognizer = StreamingRecognizer(model, config)
for pcm in audio_chunks:               # float32 mono, 任意大小
    result = recognizer.push(pcm)      # -> {partial: [...], lookahead: [...], is_final: bool}
result = recognizer.flush()            # 结束：补尾 chunk、收尾解码
```

### 7. `streaming_eval.py`
- CLI：`config` + `--checkpoint` + `--left-ctx/--chunk/--right-ctx`（覆盖 config）+
  `--max-samples` + `--check-offline`（对每样本断言 streaming greedy == offline
  greedy）+ `--no-preview`。
- 输出：PER / RTF / 一致率 / preview 统计（可选），格式对齐 `eval.py`。

## 参数选择（configs/streaming.yaml 初值）

```yaml
streaming:
  left_ctx: 96      # mel 帧 = 24 reduced (0.96s @hop160) ≥ conv 感受野 15
  chunk_size: 64    # mel 帧 = 16 reduced (0.64s) 的产出粒度
  right_ctx: 4      # mel 帧 = 1 reduced —— 等于训练 rc，不泄露
  medusa_preview: true
  preview_tokens: 2 # 显示的 lookahead token 数（= heads 数）
```

- 计算量：`(L+C+R)/C = (96+64+4)/64 ≈ 2.6x` 编码重算开销（换 0.64s 的 chunk 粒度）。
- 延迟：每 chunk 0.64s 音频到达后 ≤ 一次 encoder 前向 + 该 chunk 的解码推进。
- 参数是实验标定量：验证节的矩阵扫描定稿。

## 测试（streaming_tests.py，独立可跑）

单元（CPU、小随机模型或随机 mel 即可）：
1. `ChunkPlan` 对齐：随机参数，chunk 边界 stride 对齐、尾 chunk 处理正确。
2. `ChunkedEncoder` vs offline `transcriber`：`filter=none` 时 committed 帧数值
   **逐位一致**；`naive` 时边界差异只出现在 chunk 边界附近（且用 PER 验收兜底）。
3. `StreamingGreedyDecoder` vs `greedy_decode`：同一 enc（直接喂 offline enc）→
   输出逐 token 一致（结构性断言，必过）。
4. `MedusaPreview` 只读性：调用前后 decoder 状态逐位不变。

端到端（服务器、真 checkpoint）：
1. `--check-offline`：100 dev-clean 样本 streaming greedy vs offline greedy 逐 token
   一致率。`filter=none` 期望 100%；`naive` 以 PER 差 < 0.1（绝对）或一致率 ≥ 95% 为
   验收线，实测后定稿。
2. RTF 对比（streaming vs offline greedy，含重算开销）。
3. Preview 展示质量：lookahead token 与实际后续 token 的重合率（仅统计，不作验收）。
4. 参数矩阵：left ∈ {96, 192} mel × chunk ∈ {64, 160} mel，right=4 固定，取 PER 差
   最小者定稿。

## 移植到服务器

`git push origin streaming`（Gitee）→ 服务器 `git fetch && git checkout streaming` →
conda `whd` 环境跑 `streaming_eval.py`。数据/checkpoint 都在服务器本地，无需迁移。

## 工作量估计

- `conformer_rnnt/streaming/`（6 文件）：约 600 行
- `streaming_eval.py` + `streaming_tests.py`：约 300 行
- `configs/streaming.yaml` + 本文档：少量
- **现有代码修改：0 行**；分支 `streaming` 已从 `multihead` 创建