# 460h 固定 stride 1／2／8 串行补训

用户于 2026-11-08 指定固定投影 128、保留编码器层数，通过 stride1、2、8 补充固定降采样家族；每组试探最大 batch 后串行训练。原论文 E05 为 stride4（原归档 E09），继续复用。新运行使用名称 `stride{S}_proj128`，在结果完成前不占用论文 E 编号。

| 运行名称 | Stride | 每帧投影 | 主干宽度 | 编码层数 | 参数量 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stride1_proj128 | 1 | 128 | 128 | 10 | 4,215,622 | 已校准5300音素上限并开始训练 |
| stride2_proj128 | 2 | 128 | 256 | 10 | 10,378,310 | 串行排队，开跑前自动校准 |
| stride8_proj128 | 8 | 128 | 1024 | 10 | 95,523,398 | 串行排队，开跑前自动校准 |

三个 YAML 与原投影128的 stride4 基线逐字段比较，只有 stride 和独立实验路径不同。FFN512、注意力头8、卷积核31、输出256、predictor、joiner、增强、优化器、seed42、50轮及选模规则一致。运行时仅根据试探结果覆盖 `data.max_token_limit`。stride 是帧拼接，因此主干宽度及参数量随 stride 改变，不将其写成同主干宽度对照。

2026-11-11重启后已核实stride1、stride2完成50轮，并下载两组最佳checkpoint；stride8保持预算9700，从第8轮完整边界续训，本地评测队列已启动。原模型和训练源码SHA一致，详见[恢复与本地评测记录](phone69_stride_recovery_20261111.md)。上表为原始启动时的状态；当前状态以恢复记录为准，未完成的全量评测不填指标。

2026-11-08，stride1 从头开始正式训练，控制器PID17074、训练PID17432，运行源码提交 `354baeb03005729f7bc3ef950891565537a70ca1`。5300通过24个真实优化器步骤的确认，最高reserved为75.48 GiB；5400、5600、6100显存不足，7000触及实际RNNT张量元素边界。后两组的batch上限待各自校准，不以5300代替。[启动状态与全部校准记录](../artifacts/runs/phone69_460h_stride_sweep_proj128_20261108/startup.json)

## Batch 校准与串行流程

配置入口为 `configs/server_phone69_460h_stride_sweep_proj128_s42_e50.yaml`，控制脚本为 `scripts/server/phone69_stride_sweep.py`。串行顺序 stride1→stride2→stride8。先校准一组并启动其训练，完成50轮后才校准下一组。

每组从7,000个目标音素的 batch 上限开始：成功则乘1.4继续增大，失败则减小，找到通过／失败区间后以100音素为粒度细化。使用全部训练子集和 dev-clean 的真实时长及标签选出最大注意力、joint、激活、样本数、句长、目标长压力批；临时模型执行前向、反向、梯度裁剪与 Adam 更新。最终候选还经两轮压力批和每个训练子集3个固定随机批确认，保留约4%显存余量。记录所有失败和成功，最高通过值是已试探的稳定边界，不声称穷举了所有动态组批组合。

校准与正式训练使用相同显存分配环境，保留 allocator cache。批处理包含全部语句，不通过丢样本扩大名义 batch。RNNT int32 元素上限保持1.9e9。若增加 token 上限不再改变 guard 后的真实批结构，记录其约束；batch 数值按目标音素计，并非句数。

GPU锁为 `/tmp/phone69_460h_serial.lock`。等待空闲GPU，不中断其他任务。训练失败保留 checkpoint 和日志，停止队列；不在轮中改 batch 或无记录重启。

## 文件与本地验证

- 运行状态、逐次 probe、最终有效 YAML、选择报告：`output/phone69_460h_stride_sweep_proj128_20261108/`（服务器运行目录）。
- 各模型日志、完整轮边界 checkpoint、最佳验证 checkpoint：各自 `exp_phone69_460h_offline_native_stride{S}_none_proj128_s42_e50/`。
- 本地真实模型维度、输出长度与有限梯度验证：[local_validation.json](../artifacts/runs/phone69_460h_stride_sweep_proj128_20261108/local_validation.json)。回归与形状测试仅在本地运行，服务器只执行本次授权的训练及 batch 校准。
- 源码和配置先经 Git 同步至 Gitee、GitHub 与服务器同一提交，再启动。运行时有效 YAML 由已同步脚本生成并记录哈希，不复制代码部署。

## 论文报告

待训练完成，按最佳 dev-clean checkpoint，使用相同完整音频、FP32、batch1、beam20统计 PER 与总主要 MAC。当前没有这三组的 PER、含解码端的总 GMAC 或最佳 checkpoint；权重无关的编码器主要 MAC 已单独精确计算，见下节。最大可用 batch 是性能设置，会使更新次数和 warm-up 的数据暴露量不同；论文如实记录各组 token 预算、实际更新、轮次及 checkpoint 选择，避免声称训练暴露完全一致。[曲线设计](../papers/extremafuse_acl2027/compute_quality_curve_plan.md)

## 编码器固定形状计算量

使用冻结 test-clean 清单的2,620条真实音频长度，共19,452.480625秒、1,947,238个Mel帧，逐句解析统计后以总MAC／总音频时长归一化。包括输入投影、10层Conformer、输出投影，不含特征提取、归一化、softmax、激活、比较、访存和解码端。这里是batch1前向推理计算量，不是训练前反向总量。[完整结果与来源哈希](../artifacts/runs/phone69_460h_stride_sweep_proj128_20261108/encoder_compute.json)

| Stride | 主干宽度 | 编码器 GMAC／音频秒 | 相对stride4 |
| --- | ---: | ---: | ---: |
| 1 | 128 | 0.66752 | 83.23% |
| 2 | 256 | 0.64177 | 80.02% |
| 4 | 512 | 0.80207 | 100.00% |
| 8 | 1024 | 1.23010 | 153.37% |

固定Mel帧数T时，这些固定stride、无自适应压缩模型的编码器主要MAC与输入内容和权重无关，无需等待训练或checkpoint。令D=128s、L=ceil(T/s)，则总主要MAC为

```
C(T,s) = 80*128*T
       + 10 * [L*(4*D*512 + 7*D^2 + 31*D) + 2*D*L^2]
       + 256*D*L
```

全局注意力含L²项，故每秒计算量仍随句长变化；表中数值只对应上述共同语料的长度分布。忽略首尾帧及ceil取整、以每秒100个Mel帧近似，长度为d秒时，GMAC／音频秒约为 `0.2704128 + 0.114688*s + 0.0256*d/s`。增大stride会降低注意力项，同时增大主干矩阵运算项，因此不保证单调降低。当前语料上stride2的编码器最省，stride8比stride4多约53.37%；识别质量及解码端成本仍待评测。

随机初始化的四组实际模型各用101／1001帧、全零／随机输入做前向mock，共16次，现有执行计数hook与独立公式分项完全一致，且相同形状的不同内容计数相同。stride4还与已有全量计数的2,620条记录逐句精确一致。复现命令：

```sh
python scripts/evaluation/estimate_phone69_stride_encoder_compute.py --out /tmp/phone69-stride-encoder.json --verify-mock
```

### stride4 计算量增加来自哪里

当前比较实际是stride1的128维编码器与stride4的512维编码器。按同一完整语料分解，单位均为GMAC／音频秒：

| 组成 | stride1，D=128 | stride4，D=512 |
| --- | ---: | ---: |
| 两个FFN：4TD·512 | 0.26241 | 0.26294 |
| Q/K/V及输出投影：4TD² | 0.06560 | 0.26294 |
| 卷积中的逐点投影：3TD² | 0.04920 | 0.19720 |
| 深度卷积：31TD | 0.00397 | 0.00398 |
| 注意力矩阵乘法：2T²D | 0.28203 | 0.07070 |
| 编码器入口、出口投影 | 0.00431 | 0.00431 |
| 合计 | 0.66752 | 0.80207 |

表中T为各自进入编码层的时间步数；编码层内分项汇总10层，入口与出口投影各计一次。忽略取整，时间步降到1/4而D增为4倍，故TD近似不变、TD²增为4倍、T²D降到1/4。线性／卷积分项增加约0.34586，注意力项减少约0.21133，净增约0.13454，即20.16%。结果符合当前网络形状；它不能单独归因于四倍降采样。若保持同一主干宽度，压缩时间维将降低上述编码层计算量。

另对四个实际模型直接拦截底层mm、addmm、卷积和CPU flash attention的输入输出形状做独立算子级核对，1001帧输入下的总主要MAC与公式精确一致。[算子级审计](../artifacts/runs/phone69_460h_stride_sweep_proj128_20261108/encoder_operator_audit.json) · [可复现审计脚本](../scripts/diagnostics/audit_phone69_stride_encoder_ops.py)。
