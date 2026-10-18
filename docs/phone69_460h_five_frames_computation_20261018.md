# Phone69 460h：five_frames 三组动态压缩重训

用户批准 A05：三组动态方案统一采用 `five_frames`，停止当前 diff5 训练，删除其废弃实验目录，从随机初始化开始训练。随后明确要求已经完成的 stride4_none 基线保留不动。其它 A01–A14 审计项继续 pending，本次没有改动这些运算定义。

## 配置与范围

三份配置仍为 `configs/server_phone69_460h_offline_native_<方案>_s42_e50.yaml`；只更改 `weight_predictor_mode: five_frames` 以及新的实验输出路径。新实验名为 `exp_phone69_460h_offline_native_<方案>_five_frames_s42_e50`。顺序为 stride1、stride2、stride4，均为 cosine_diff + learned。

| 方案 | 输入投影 | 拼帧 stride | 主干宽度 | 输出维度 | 融合 MLP | 总参数 | phone-token 预算 |
|---|---:|---:|---:|---:|---|---:|---:|
| stride1_cosd_learned | 80→128 | 1 | 128 | 256 | 640→32→5 | 4,236,299 | 6500 |
| stride2_cosd_learned | 80→64 | 2 | 128 | 256 | 640→32→5 | 4,231,179 | 16000 |
| stride4_cosd_learned | 80→32 | 4 | 128 | 256 | 640→32→5 | 4,228,619 | 32600 |

已完成的 stride4_none 基线目录、配置、日志、run.json、最佳 checkpoint 和全部边界 checkpoint 均保留。启动前后以逐文件 SHA-256、大小和 mtime_ns 校验；它不在新训练队列内。

四组 Conformer 主干仍相同：内部128、2+8层、8头、FFN512、卷积核31、dropout0.1、输出256。基线总参数4,207,942；三组动态模块各20,677参数，比原diff5模块多20,352。公平性指主干维度和训练数据轮数一致，不声称总参数、更新步数、计算量完全相同。

## 数据与整句编码

固定 phone69：69个保留大小写和重音的完整ARPABET phone，blank69，过滤spn。train-clean-100 + train-clean-360，训练中验证dev-clean。16kHz音频，FFT400、hop160、80维Mel；原log变换、外部global_stats归一化、SpecAugment以及补齐规则均沿用。

输入无偏置投影至128/stride维，再拼接相邻stride帧，末尾零补齐，长度为ceil(T/stride)。两块Conformer使用完整非因果序列；没有时间分块、main/context划分或特殊尾部保留。新配置不含keep_rate、left_context_length、right_context_length。

## 选帧：自然余弦峰谷

第一块输出有效帧为H=(h0,…,h[n−1])，每帧128维。计算c[t]=1−cosine(h[t],h[t+1])。P=find_peaks(c)，V=find_peaks(−c)，候选I=sort(unique((P+1)∪V))，过滤越界位置。无额外阈值、top-k、指定数量或压缩率；无候选保留0，n≤1直接传递。

本次保持A04原定位规则，不改成差异点两侧平均，也不更改峰谷映射或去重。K=|I|由输入决定，observed_compression_lengths只记录实际长度。

## 融合：五帧特征预测五个权重

对每个中心i，按时间顺序拼接[h[i−2];h[i−1];h[i];h[i+1];h[i+2]]，得到640维输入。实际代码位于`conformer_rnnt/compression/mixers.py`，窗口沿用工厂默认5。

MLP为Linear(640,32)→ReLU→Linear(32,5)→softmax。每个位置输出5个标量权重，同一帧所有128通道共享其权重。输出z[i]=Σk α[i,k]·h[i+k]，仍为128维。参数数为640×32+32+32×5+5=20,677。

权重预测器不使用diff或差分值。压缩编排器仍会计算既有L1差分张量以维持公共检测接口，但cosine_diff检测器使用余弦距离，five_frames混合器在固定索引下忽略传入diff；不进行未经批准的接口/梯度路径重构。五帧输入及MLP通过RNNT反向传播；离散峰谷位置仍不反向传播。

边界保持既有实现：预测器输入的越界索引截到端点，融合时屏蔽越界位置并重新归一化。A02中的浮点稳定性修复未获批准，未加入。五帧左右邻居均是普通离线特征，没有独立上下文身份。

## 预测器、损失和优化

编码输出128→256并LayerNorm。phone embedding128，2层128维LayerNorm-LSTM，输出256；joiner加和→ReLU→70类。RNNT reduction=sum、clamp1、batch路径；Adam lr0.0005、betas0.9/0.999、eps1e−8、warmup10000、梯度裁剪10、累积1、seed42、50epoch、无早停。

验证与保存频率、按dev RNNT loss选最佳checkpoint、top-2以及每轮完整边界checkpoint均沿用。不同预算造成更新数和warmup所占epoch不同，这一既有协议没有改变。数据、增强、padding、归一化、LSTM、位置编码、损失缩放等其它审计项均未修改。

## 清理、预检与启动

删除范围严格限定三个旧动态目录`exp_phone69_460h_offline_native_<方案>_s42_e50`；基线不在清单中。先停止旧串行队列，保存配置/run/日志及文件删除清单，再删除旧动态checkpoint和目录。旧动态记录标为作废；共享旧审计目录仍保存有效基线证据。

三组重新初始化，from_checkpoint=null，不能从diff5的4维MLP checkpoint恢复640维MLP。新checkpoint继续保存offline_native_definition，其中weight_predictor_mode=five_frames；初始epoch0/step0及每轮完整模型/优化器/调度器状态均保存。

启动门槛在`artifacts/runs/phone69_460h_native_offline_five_frames_20261018/`：配置差异范围检查、实际维度/参数与有限前后向、回归测试、三组最坏真实批次GPU前后向/优化器压力检查、源码/配置/报告指纹、基线保留指纹及删除审计。预算保持6500/16000/32600，不因显存低采样提高；若预检失败，保留现场并通知，不擅自修改其它待审批项。

队列脚本为`scripts/server/phone69_460h_five_frames_pipeline.sh`，tmux为`phone69_460h_five_frames`。只运行训练及其中dev-clean验证，不启动独立评测。每小时监控沿用，其它审计项未批准前只读监控，不自动改配置或恢复失败任务。
