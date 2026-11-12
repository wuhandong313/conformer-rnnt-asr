# 固定 30 秒输入的理论主要 MAC

## 采用的论文口径

2026-11-12 用户要求：用真实数据测得各压缩层的保留率，再用固定算法计算统一输入长度的 MAC；缺失的逐层帧数在本地重新实测。主表采用 **30 秒、batch 1、编码器与压缩模块**，同时列 GMAC/30秒输入与 GMAC/输入音频秒。

- [Fast Conformer，ASRU 2023，§3.1.1、表4](https://arxiv.org/html/2305.05084v4#S3.SS1.SSS1)：使用单条30秒音频，DeepSpeed统计编码器MAC，表列GMAC；不包含RNNT decoder。
- [Zipformer，ICLR 2024，§4.0.2、表2](https://arxiv.org/html/2310.11230v4)：明确用30秒音频统计编码器GFLOPs，同时比较Conformer、Squeezeformer、Branchformer等。

30秒是本研究选择沿用的相关工作口径，不声称所有ASR论文使用同一时长。GMAC与GFLOPs不能直接混用；本报告一次乘加计1 MAC，不直接拿本项目的主要MAC与论文中所有FLOPs作数值排名。模型层数、宽度、输入投影和所含算子仍需一致才能作绝对比较。

## 输入与保留率

采用当前16kHz音频、10ms hop、centered STFT，长度为 `T0=floor(16000*d/160)+1`，30秒对应 **3001个Mel帧**，80维输入。不是把变长整集MAC除以总秒数，也不是将旧每秒结果乘30。另给1、5、10、20秒敏感性结果。

对每个checkpoint、每个压缩位置j，从冻结test-clean全部2620句取该层输入N和输出K：

`r_j = sum_i K_ij / sum_i N_ij`，称为**保留率**，压缩掉的比例为`1-r_j`。

多阶段分别计算条件保留率：L5阶段的分母是进入L5压缩层的长度，不能再除原始Mel帧数；hard邻域保留用最终去重后的输出帧数。保留每层总N、总K、逐句保留率均值和标准差、checkpoint及配置／数据来源哈希。

固定长度下按真实网络顺序传播：Mel压缩→输入投影→stride拼接→编码层→层后压缩→输出投影。压缩后长度为`clamp(round_half_up(r_j*N_j), 1, N_j)`；stride后长度为`ceil(N/s)`。取整用有理数实现。不同模型保留其真实宽度`D=projection_dim*stride`，不人为抹平stride导致的宽度变化。

保留率来自真实语音，但计算MAC无需再执行模型。该结果是**校准保留率下的固定长度理论估计**；不声称所有30秒语音有相同保留率，也不是`E[MAC]`：平方项使`MAC(E[T])`与`E[MAC(T)]`有差异。保留率仍依赖checkpoint、语料内容和原始长度分布，因此归档校准集与离散程度；不能把未完成训练的模型套用另一个模型的保留率。

## 逐项公式

在每个Conformer层，输入长度T、宽度D、FFN宽度F、卷积核k：

`C_layer = T*(4*D*F + 7*D^2 + D*k) + 2*D*T^2`。

前半计两个FFN、注意力Q/K/V及输出矩阵、卷积逐点矩阵及深度卷积；后半计QK与AV两个注意力矩阵乘法。逐层使用压缩前后的实际对应长度，不将全网都套用最终保留率。

- 输入投影：`T_projected * 80 * projection_dim`，Mel压缩发生在它之前。
- 输出投影：`T_final * D * encoding_dim`。
- 余弦检测器点积：`(N-1)*D_stage`；Mel处`D_stage=80`，编码层处为D。
- 五帧融合MLP：`K*(5*D_stage*32 + 32*5)`。
- 五帧加权融合：`K*5*D_stage`。
- hard选择不产生融合矩阵MAC；输入仅1帧时按实现旁路，不计检测和融合。

不计特征提取、归一化、softmax、激活、峰谷比较、排序、访存、Python控制流、训练反向和优化器。这是明确算子范围的主要MAC，不称为设备耗时或完整指令量。

## 解码端单列

主指标是编码器＋压缩层。固定时长和保留率无法唯一确定beam20的扩展次数，禁止把历史真实搜索量缩放后称为固定长度理论beam20成本。

附表提供一个所有模型相同的**规定单路径参考成本**：用同一参考音素速率`189023/19452.480625`确定`U=ceil(rate*d)`（30秒为292个音素，也可通过`--target-phones`显式固定U）。路径包含T_final次blank和U次音素发射，缓存predictor状态：predictor计算U+1次，joiner计算T_final+U行。

单次predictor成本由配置的LSTM门矩阵与输出投影解析求出；当前模型为294912 MAC/token。joiner每行是`256*70=17920 MAC`。这些列命名`reference_*`，不是greedy实测或beam20搜索预测，不与历史beam20总成本混成同一列。PER继续引用原始完整音频、FP32、batch1、beam20实测，不把它称为裁切30秒后的PER。

## 覆盖与复现

采用原归档E01–E11及新增S1/S2/S8，总计14个已评估模型；E编号保持原归档编号，另列论文编号避免混淆。六个新层位保留在清单中，缺少对应checkpoint完整保留率时明确pending，不输出虚构MAC。

```sh
python scripts/evaluation/theoretical_phone69_macs.py \
  --stage-calibration artifacts/research/phone69_fixed_duration_20261112/E11_stage_lengths.json
python -m unittest discover -s tests -p test_theoretical_macs.py
```

默认输出[结果JSON](../artifacts/research/phone69_fixed_duration_20261112/results.json)、[汇总CSV](../artifacts/research/phone69_fixed_duration_20261112/summary.csv)。没有stage-calibration时E11明确pending。

已完成[14组结果表](../artifacts/research/phone69_fixed_duration_20261112/REPORT.md)，每组包含1／5／10／20／30秒，共70个理论工作点。E11缺失帧数已在家中CPU按其冻结checkpoint补测完整2620句，Mel／L5／L8条件保留率分别为0.392838／0.384232／0.449261；[原始补测记录](../artifacts/research/phone69_fixed_duration_20261112/E11_stage_lengths.json)保留逐层观测与来源哈希。9项本机回归通过，[核验记录](../artifacts/research/phone69_fixed_duration_20261112/verification.json)列明范围和产物哈希。

缺失帧数的本地补测入口为`calibrate_phone69_stage_lengths.py --manifest <冻结manifest> --out <新文件>`：校验checkpoint、配置、源码、统计、词表和音频哈希，使用原CPU/FP32环境，实际观测每层输入帧数及各压缩模块前后长度，覆盖完整2620句。只执行编码器，不执行解码、不计时、不训练。脚本经Git同步到家中电脑后运行，服务器不执行测试或此补测。

公式通过原E01–E10全部26200条逐句记录的6个编码器／压缩MAC分项逐一精确核对；另测试多阶段条件比率、长短句加权、stride取整、Mel先压缩后投影、单帧旁路和规定解码路径。旧PER、计数及时间档案保留原样；本协议结果另存，不能直接覆盖旧beam20总GMAC曲线。
