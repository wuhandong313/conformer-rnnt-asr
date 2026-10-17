# Phone69 460h：自然峰谷动态压缩与非流式四组重训计算逻辑

本报告先于新训练启动生成。2026-10-18 用户明确修正方法：动态压缩不指定保留比例，只依据峰谷选帧；任务为非流式，不划分、追加或单独保留右上下文。以下定义取代此前 `keep_rate=0.27` 的实验定义。旧指标、日志与配置留作历史审计，不作为本轮自然峰谷方法的实验结果。

## 1. 四组配置与公平性

新文件统一位于 `configs/server_phone69_460h_offline_native_<方案>_s42_e50.yaml`，实验目录为 `/test21/conformer-rnnt-asr/exp_phone69_460h_offline_native_<方案>_s42_e50`。四组均从随机初始化开始，不读取旧权重、优化器、调度器或训练步数。

| 串行顺序 | 方案 | 输入线性投影 | 拼帧 stride | Conformer 内部宽度 | 中间操作 | 第二块长度 |
|---|---|---:|---:|---:|---|---|
| 1 | stride4_none | 80→32 | 4 | 32×4=128 | 恒等映射 | ceil(T/4) |
| 2 | stride1_cosd_learned | 80→128 | 1 | 128×1=128 | 余弦距离峰谷 + 学习融合 | K₁(x) |
| 3 | stride2_cosd_learned | 80→64 | 2 | 64×2=128 | 余弦距离峰谷 + 学习融合 | K₂(x) |
| 4 | stride4_cosd_learned | 80→32 | 4 | 32×4=128 | 余弦距离峰谷 + 学习融合 | K₄(x) |

所有组：前块 2 层、后块 8 层，8 个注意力头，每头 16 维；FFN 隐层 512；深度卷积核 31；Conformer dropout 0.1；编码输出 128→256 并 LayerNorm。预测器为 128 维 embedding、2 层 128 隐层 LSTM，输出 256；joiner 为 256 维加和、ReLU，再投影至 70 类（69 phones + blank）。

公平的是 Conformer 主干宽度、层数、头数、FFN、预测器与输出维度。前端投影参数随 stride 不同；三组动态模块均额外包含同一个 325 参数的融合 MLP，所以不能声称四组总参数量完全相同。实际参数计数由预检生成 `dimension_audit.json`。

## 2. 特征与拼帧

音频 16 kHz，FFT 400，Mel 80，hop 160（10 ms），沿用现有 log 变换、全局统计归一化和 SpecAugment。T 为真实 Mel 帧数，批内补齐区域在特征流水线之后重置为零，不计为真实帧。

投影不带 bias。相邻 S 帧沿通道拼接：`[B,T,80] → [B,T,128/S] → [B,ceil(T/S),128]`。末尾不足 S 帧用零补齐，真实长度取 `ceil(T/S)`，避免把最后 1–3 个真实 Mel 帧丢弃。这是张量拼帧对齐，不是额外语音上下文。四组使用同一长度规则。

不追加 R 帧，不将长度加 R/S，不划分 main/context，不绕过压缩保留尾帧。新配置显式使用 `model.sequence_mode: offline`；该模式拒绝配置 `right_context_length` 或 `keep_rate`，避免误用旧默认值。旧配置兼容代码仅供历史实验读取，新四组走独立的完整序列压缩路径。

## 3. 自然峰谷检测：没有指定压缩率

第一块的真实输出为 `H=(h₀,…,hₙ₋₁)`，n=ceil(T/S)，每帧 128 维。计算：

`c[t] = 1 - cosine(h[t], h[t+1])`，t=0,…,n−2。

沿用仓库现有 `cosine_diff` 检测器的定位定义：`P=find_peaks(c)`，`V=find_peaks(-c)`；候选帧索引为 `I=sort(unique((P+1) ∪ V))`，仅取合法索引。`find_peaks` 不设置高度、显著性或间距阈值；平台峰的行为沿用 SciPy 实现。没有候选时保留索引 0；n≤1 直接传递。

所有候选均保留。没有 top-k、`round(n*r)`、等比例补帧、目标长度或可学习的预算参数。输出 K=|I| 由输入特征决定；`K/n` 是事后测量量，不是人为指定值。同一句话在不同 stride、模型参数或训练增强下可以产生不同 K。

峰谷检测本身不反向传播（离散索引），融合后的特征及融合 MLP 正常参与反向传播。该方法是离散的输入自适应选帧，不能声称选帧位置本身由可微目标端到端学习。

## 4. learned / diff5 融合的精确定义

选帧依据是上面的余弦距离。融合权重使用另一条局部信号：`d[t]=sum_f abs(h[t+1,f]-h[t,f])`。对每个候选 i，读取 `d[i−2:i+3]`（越界索引裁到合法范围），得到五个数；再取相邻绝对差，得到 4 维输入。

`4 → Linear(4,32) → ReLU → Linear(32,5) → softmax` 生成五个权重。对 `h[i−2:i+3]` 的合法帧加权，越界位置权重置零，再对合法权重归一化。一个候选生成一个 128 维向量。参数数为 `(4×32+32)+(32×5+5)=325`。

这不是固定 Hann 窗，也不是对每个候选单纯硬抽取。非流式融合可以读取同一句话内 i 后的合法邻帧；它们参与普通对称窗口计算，没有独立的“右上下文”身份或保护规则。

## 5. 非流式计算与复杂度

两块均沿用非因果 Conformer：注意力覆盖整条有效序列，卷积与五帧融合可以使用后续真实帧。因此本实验不主张流式、零前瞻或低算法延迟。

固定共同维度后，前后块注意力的主要长度项分别为 `2·n²` 和 `8·K²`；基线 K=n。动态方案的 K 每条语音不同，应记录实际长度分布。检测及固定窗口融合随 n、K 近似线性增长，Python/SciPy 检测含 CPU/GPU 同步开销，不能只由理论注意力复杂度推断 RTF。

## 6. 保持的训练对照

数据为 train-clean-100 + train-clean-360（约 460h），验证 dev-clean；固定 phone69 词表，保留大小写、重音和空格分隔 phone，过滤 spn。50 epoch，seed=42，无早停，单张 A100 80 GiB。Adam lr=0.0005、betas=(0.9,0.999)、eps=1e−8，warmup 10000 步，梯度裁剪 10，RNNT batch/sum、clamp=1，梯度累积 1。验证及验证 checkpoint 每 200 步，训练 checkpoint 每 500 步。

批次预算是 phone-token batch 的大小，不是动态压缩率。自然峰谷长度通常不同于旧的 27% 配额，旧预算不能直接照搬；启动前用真实 train/validation 批次的前向、RNNT loss、反向及优化器步测量显存，再将通过的预算写回各配置及预检记录。目标仍约 90% 设备显存，保留形状波动空间。不同预算带来每轮优化步数及 warmup 所占 epoch 比例差异，论文中需要披露，不能将其描述为完全等计算训练。

## 7. 启动门槛与恢复

启动前必须通过：四组配置一致性与维度计数、自然峰谷无配额测试、完整序列含尾部的选帧测试、长度与补齐测试、有限 loss/梯度测试，以及真实最坏形状显存预检。预检、预算和实际启动状态写入 `artifacts/runs/phone69_460h_native_offline_20261018/`，本报告随最终配置补全。

四个旧实验目录的 22 个 checkpoint（849,816,368 bytes，约 810.45 MiB）已按清单删除，本地对应目录无 checkpoint；旧日志与评测保留。新目录初次启动必须显示 `Starting from scratch`，`resume_checkpoint=null`。新 runner 在第一次 fit 开始前保存完整 epoch=0、step=0 初始状态，并在每个 epoch 完成时保存不可变的完整边界 checkpoint。后续中断仅可恢复本轮同定义、同实验的完整模型/优化器/调度器 checkpoint；改变批次划分只能从这些完整 epoch 边界恢复。checkpoint 中记录 offline_native_definition，旧方法或缺失定义的 checkpoint 会被拒绝。监控使用服务器 UTC 与服务器文件时间判断，禁止启动旧队列。

## 8. 实测配置与验证结果

以下结果已在新训练启动前完成；均使用真实 train-clean-100、train-clean-360、dev-clean 批次。样本选择覆盖全量批次的注意力长度上界、RNNT joint 上界、最多句数、最长音频以及各子集最坏注意力形状；相同批次去重。自然压缩的上界按 K≤n 计算，没有假设目标保留率。训练样本用训练增强，验证样本用无增强验证特征；验证形状额外执行反向，作为内存压力检查。所有样本均执行有限 RNNT loss、反向、梯度检查与优化器步，跨形状保留 allocator cache。

| 方案 | 最终 phone-token 预算 | 总参数 | 实批压力样本数 | PyTorch allocated 峰值 GiB | reserved 峰值 GiB | 设备使用采样最大 GiB |
|---|---:|---:|---:|---:|---:|---:|
| stride4_none | 23500 | 4,207,942 | 5 | 67.26 | 70.17 | 70.69 |
| stride1_cosd_learned | 6500 | 4,215,947 | 5 | 37.36 | 67.31 | 67.82 |
| stride2_cosd_learned | 16000 | 4,210,827 | 5 | 46.46 | 68.72 | 69.24 |
| stride4_cosd_learned | 32600 | 4,208,267 | 6 | 47.13 | 65.84 | 66.35 |

`allocated` 是活跃张量的分配峰值，`reserved` 包含 PyTorch 缓存池；设备使用还包含 CUDA 等开销。设备一列是各压力步末尾的最大采样，不伪称连续采样峰值。自然 K 与批次形状会波动，本次采用上述保守预算和余量；不因后续一次低采样增加预算。较大预算触发显存安全门槛的试测 JSON/log 全部保留，未用其结果启动训练。

三组动态方案在这些压力样本上的事后 K/n 范围分别为 40.30%–43.19%、40.39%–41.82%、40.99%–42.40%；这只是随机初始化/少量试探步的观测，既不是配置参数，也不代表全数据或收敛模型的压缩率。

验证结果：

- 服务器 `python -m unittest discover -s tests`：52 项通过，含历史行为、新自然峰谷与非流式路径、初始边界以及 epoch 边界改变 batch 后的优化器/调度器恢复；本机核心测试通过，完整 Lightning 测试以服务器运行环境为准。
- 四组实际 Conformer 主干参数均为 3,846,400；三组动态融合模块均为 325 参数；四组 101/83 帧可变长输入前向和反向有限。
- 四组共 21 个去重真实压力批次通过，所有批次的全长 joint 元素上界低于 1.9e9。
- 边界恢复回归：完成 4 个优化步后将 batch_size 从 1 改为 2，下一轮完成后 global_step、Adam step 和调度器 last_epoch 均为 6，没有重新计算旧轮中途偏移。
- `compileall`、串行脚本 `bash -n` 和 `git diff --check` 通过。

[维度与参数审计](../artifacts/runs/phone69_460h_native_offline_20261018/dimension_audit.json)、[显存验证](../artifacts/runs/phone69_460h_native_offline_20261018/memory_validation.json)、[52 项测试日志](../artifacts/runs/phone69_460h_native_offline_20261018/preflight_tests_with_boundary.log)、[旧 checkpoint 删除清单](../artifacts/runs/phone69_460h_native_offline_20261018/checkpoint_deletion.json)。

启动使用 `phone69_460h_offline_native` tmux 和 `scripts/server/phone69_460h_offline_native_pipeline.sh`。启动门槛 `prelaunch_validation.json` 校验四份 YAML、模型源码、runner、边界保存器和本报告的 SHA-256。实际开始时间、代码提交、起始状态和进度另存于 `startup_validation.json` 与每组 `run.json`，不将预检的临时优化器步数带入正式训练。
