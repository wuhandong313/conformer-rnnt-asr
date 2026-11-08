# 460h 固定 stride 1／2／8 串行补训

用户于 2026-11-08 指定固定投影 128、保留编码器层数，通过 stride1、2、8 补充固定降采样家族；每组试探最大 batch 后串行训练。原论文 E05 为 stride4（原归档 E09），继续复用。新运行使用名称 `stride{S}_proj128`，在结果完成前不占用论文 E 编号。

| 运行名称 | Stride | 每帧投影 | 主干宽度 | 编码层数 | 参数量 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stride1_proj128 | 1 | 128 | 128 | 10 | 4,215,622 | 已校准5300音素上限并开始训练 |
| stride2_proj128 | 2 | 128 | 256 | 10 | 10,378,310 | 串行排队，开跑前自动校准 |
| stride8_proj128 | 8 | 128 | 1024 | 10 | 95,523,398 | 串行排队，开跑前自动校准 |

三个 YAML 与原投影128的 stride4 基线逐字段比较，只有 stride 和独立实验路径不同。FFN512、注意力头8、卷积核31、输出256、predictor、joiner、增强、优化器、seed42、50轮及选模规则一致。运行时仅根据试探结果覆盖 `data.max_token_limit`。stride 是帧拼接，因此主干宽度及参数量随 stride 改变，不将其写成同主干宽度对照。

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

待训练完成，按最佳 dev-clean checkpoint，使用相同完整音频、FP32、batch1、beam20统计 PER 与分项主要 MAC。当前没有这三组的 PER、GMAC 或最佳 checkpoint。最大可用 batch 是性能设置，会使更新次数和 warm-up 的数据暴露量不同；论文如实记录各组 token 预算、实际更新、轮次及 checkpoint 选择，避免声称训练暴露完全一致。[曲线设计](../papers/extremafuse_acl2027/compute_quality_curve_plan.md)
