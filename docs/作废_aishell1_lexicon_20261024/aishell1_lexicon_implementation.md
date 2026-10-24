> 作废的历史方案：当前已改为 AISHELL-3 + THCHS-30 原始带调音节；本文不再是运行说明。

# AISHELL-1 原始词典符号：实现与运行说明

本分支基于 `bae6d2c2170c1c1db9f4b1df4fa6743cc29b34f2`，名称为 `codex/aishell1-lexicon`。本地独立目录为 `/Users/wuhandong/Desktop/python/conformer-rnnt-asr/aishell1-lexicon`，服务器独立目录约定为 `/test21/conformer-rnnt-asr-aishell1`。原英文 `main` 的修改和实验不进入本分支。

**用户最新约束：先不下载数据集。** 本次只准备实现和生成音频夹具测试；没有获取 AISHELL 音频、没有生成真实语料标签、没有运行 A100 预检、没有正式训练或测评。两个配置同时设置 `allow_dataset_download: false`、`allow_training: false`。代码没有下载模式。不能把本地单元测试通过当成真实数据或显存检查通过。

## 1. 实现入口

| 内容 | 文件 |
|---|---|
| 完整 217 类固定词表 | `configs/aishell_lexicon_vocab.json` |
| 原始符号 tokenizer | `aishell_lexicon_tokenizer.py` |
| 词典解析、候选择读、g2pM 适配 | `conformer_rnnt/aishell/lexicon.py` |
| 本地数据审计、标签、训练统计、冻结 | `scripts/data/prepare_aishell1.py` |
| 通用清单、真实长度共同批次 | `conformer_rnnt/aishell/manifest.py` |
| 两组配置及配对检查 | `configs/server_aishell1_lexicon_*_s42_e50.yaml`、`conformer_rnnt/aishell/launch.py` |
| 真实 CUDA 压力预检 | `scripts/server/probe_aishell1.py` |
| 独立队列，默认仅检查 | `scripts/server/aishell1_lexicon_pipeline.py` |
| 严格门槛下的单组训练 | `scripts/server/train_aishell1_one.py` |
| 完整不可变 checkpoint | `conformer_rnnt/aishell/runtime.py` |
| 只读监控探针 | `scripts/server/status_aishell1.py` |
| 无 RTF 的独立测评与两组比较 | `scripts/evaluation/eval_aishell1.py` |

## 2. 官方资源与冻结条件

来源：[OpenSLR SLR33](https://www.openslr.org/33/)、[AISHELL-1 论文](https://arxiv.org/abs/1709.05522)。预期有转写样本数：train 120,098、dev 14,326、test 7,176。实际数量必须由本地文件逐条审计，不能以预期值代替实际检查。

此前词典核实的期望值：139,874 行、137,074 个不同词条、2,643 个具有多个不同发音的词条、217 个原始符号；47 行不满足“两符号对应一字”，其中包括合法 `er`、静音和非纯汉字词条。原始 `lexicon.txt` 的 SHA-256 固定为 `a01567fb7d0255164f7f97998cf1616ea40b30f1b3bccfd958df1a09e32f404c`。本次没有重新下载词典；真实标签准备时将重新核对已有文件的字节哈希及全部符号。原文件不改写，须留在外部数据目录。

音频审计保留官方词条边界，逐条验证 WAV/转写 ID、16kHz/单声道/PCM16、可完整解码、有效帧数、说话人隔离和跨划分 PCM 内容重复。无转写的额外音频单独列出；有转写但缺少有效音频、重复 ID、错误格式、跨划分音频或说话人泄漏均阻止通过。相同文字不会被去重。不会转换、删除、补齐或下载异常文件。

数据根目录默认 `/test21/aishell1/data_aishell`，预期包含 `wav/{train,dev,test}/说话人/*.wav` 和 `transcript/aishell_transcript_v0.8.txt`。所有清单中的音频路径相对此根目录。

## 3. 标签选择和人工审计

标签是词典右侧的完整符号，例：`中国 → zh ong1 g uo2`。保留 `aa/ee/ii/oo/uu/vv/ix/iy`、声调数字和边界，不转写成普通拼音或 IPA。

g2pM 固定为 `0.1.2.5`，仅在单独预处理环境安装。适配器记录安装包与随附模型文件的 SHA-256、NumPy 版本；不在训练环境安装或调用。输入是合并官方词条后的完整句子，输出必须能按字符区间对应回原始词条。[Charsiu 数据说明](https://github.com/lingjzhu/charsiu/blob/main/misc/data.md)提供 AISHELL 使用 g2pM 的先例，但这种标签仍是词典约束的自动转写，不能声称为人工逐音素标注。

每个词条只接受唯一与语境读音等价的原始候选。即使词典只有一个候选，预测冲突也进入审计；多个原始序列都等价时仍不能取第一条。完全相同的候选去重但保留全部原始行号。用于比较的拼音映射与模型标签分离，通过可变长精确匹配支持零声母、舌尖韵母和单独 `er`，不用模糊距离、置信度阈值或固定“两符号一音节”假设。明确的 `SIL`/`<SPOKEN_NOISE>` 词条只采用词典原有 `sil`。

`issues.jsonl` 保留录音、原文、词条、字符区间、候选及行号、预测、问题类型、稳定 issue ID 和处理结果。`label_audit.json` 按词条和类型聚合，提供计数与示例。人工决定文件是 JSON 字典：

```json
{
  "从 issues.jsonl 复制的完整 issue_id": {
    "issue_id": "同一个完整 issue_id",
    "candidate_phones": ["ii", "in2", "h", "ang2"],
    "reason": "独立核查后的具体依据",
    "version": "review-v1"
  }
}
```

只能选已有候选；缺词或词典没有正确候选时需另行决策。旧 issue ID、缺少理由/版本、未解决异常都会阻止冻结。不会剔除这些录音绕过问题。

冻结包包含三个 JSONL 清单、`statistics.json`、音频/标签/词典审计及 `label_protocol.json`，`freeze.json` 汇总每个文件的 SHA-256。协议包括词典、词表、g2pM 包与模型、比较映射、预处理代码、人工决定。两组共享完全同一个冻结包；训练和测评都不接受模型输出作为参考标签输入。

## 4. 完整计算逻辑

两组读取 16kHz PCM16，缩放到浮点振幅，不重采样。Mel 使用 FFT400、win400、Hann periodic、hop160、80 Mel、center=true、reflect、pad0、power2、normalized=false、HTK、f_min0、f_max8000（参数默认 None）、norm=None。实际帧数为 `floor(N/160)+1`；短于反射补齐要求的录音进入异常。

沿用 `common.piecewise_linear_log`：`u=Mel×GAIN`，`u>e` 时 `log(u)`，其余 `u/e`，GAIN 为已有 int16 幅度平方换算。没有换成常见 log1p，也未引入新的 epsilon/clamp。训练归一化逐维均值和逆标准差使用**中文 train 有效帧**、float64 在线合并、总体方差分母；dev/test 和补齐帧不参与。统计正性或有限性失败直接停止。

训练时频率遮挡两次（27）、时间遮挡两次（100、p0.2）；验证/测试无增强。批次补齐后重新将无效帧清零。增强 CPU 随机种子仅取 seed42、epoch、共同 batch ID 和批次指纹，独立于模型初始化、dropout 和 CUDA 随机流。Mel 缓存仅属于 CPU 特征处理，不进入中文模型 checkpoint。

| 计算项 | stride4 基线 | stride1 动态 |
|---|---|---|
| 输入投影 | 80→32 | 80→128 |
| 时间拼帧 | 4×32→128；末尾补零，ceil(T/4) | 1×128，长度 T |
| 第一块 | 2 层 Conformer | 同左 |
| 层2后 | 原序列 | 自然余弦峰谷＋five_frames |
| 第二块 | 8 层 Conformer | 同左 |
| 内部宽/头/FFN/卷积核/dropout | 128 / 8 / 512 / 31 / 0.1 | 同左 |
| 编码输出 | 256 | 256 |

Conformer 保留基点实现的两个半步 FFN、自注意力、卷积及各自归一化/残差。两组整句离线计算，没有分块、左右上下文切分或特殊尾部保留。静态拼帧的末尾零补齐仅用于张量对齐，不是右上下文。旧流式兼容代码保留供英文路径使用，不会因默认值进入中文配置。

动态位置与算法固定：第2层后计算相邻128维帧余弦距离，检测自然峰和谷，以峰位置+1、谷位置取中心，排序去重，保留所有候选。没有 keep_rate、配额、目标长度或新增阈值。无候选保留第0帧，长度≤1沿用传递定义。离散索引不反向传播。

每个中心取左右各2帧，按时间顺序形成640维输入；`Linear(640,32)→ReLU→Linear(32,5)→softmax`，五个权重融合五帧。边界预测输入沿用截断索引，实际融合屏蔽无效邻居并重新归一化。公共接口仍计算 L1 差分，**five_frames 的权重预测器不读取该差分**。融合参数和帧特征正常接收 RNNT 梯度。MLP 参数为20,677。`last_lengths` 仅记录实际压缩输入/输出，绝不回用作目标比例。

符号 embedding128；两层128维 LayerNorm-LSTM（epsilon0.001、dropout0.3），预测输出256；原 joiner 相加后 ReLU，输出218类。前217类原始词典符号，第217号 blank；无 BOS/EOS/UNK。`sil` 是普通可发射符号，不自动插入。目标补齐为 blank，由目标长度排除。

CPU 实例化实际统计：基线 **4,264,922** 参数，动态 **4,293,279** 参数。差额28,357，其中五帧权重 MLP20,677、输入投影多7,680。随机特征的前向仅证明尺寸和有限性，不代表真实语音压缩率或识别性能。

RNNT reduction=sum、clamp1、单个完整批次调用。Adam lr0.0005、betas0.9/0.999、eps1e-8，前10000步线性 warmup 后恒定；梯度裁剪10、累积1、FP32、seed42、50轮、无早停。开发损失以句数加权平均，按最低值保存 best。相同主干维度不代表总参数、有效帧数或计算量完全相等；参数实测见代码验证记录。

## 5. 共同批次和启动门槛

两组先固定同一个样本批次 JSON 和 seed42 每轮顺序，再训练；清单长度不依赖文件名、英语章节布局或字符串字符数。RNNT 守护使用 `B×max(real_Mel_T)×(max(U)+1)×218 < 1.9e9`。完整 Mel 长度是两个模型共同的安全上界，不假定动态压到某个比例。不能放下的单句明确报错，绝不丢弃。

显存候选依次6000、4500、3000、2000、1000。每个候选重新生成两组共同批次和配置副本，覆盖最长音频、最长标签、最大联合张量、注意力压力、最大样本数，以及开发集压力批次；训练压力至少执行两步以创建并覆盖 Adam 稳态。记录 allocated/reserved、设备显存、实际形状、梯度/参数有限性、真实观测长度。只接受 A10080GB 且显存上限≤90%。失败记录不覆盖；非 OOM 错误停止，不更改算法补救。

还必须通过完整回归测试、冻结资源检查、两组协议指纹相同、初始完整 checkpoint 保存及严格恢复、磁盘容量核算。初始空 Adam 状态与训练后状态都由测试覆盖。磁盘上界按每500步快照、每200步可能更新 best、每轮边界和50轮计算，按实测初始快照放大覆盖 Adam，再预留20%和2GiB。

`prelaunch_validation.json` 绑定源码、两组已解析配置、批次/数据/归一化、真实预检报告和测试日志。配置或源码变化后必须重做预检，不能手改 valid。训练代码还要求 `--execute`、配置 allow_training=true 和共享 GPU 锁。当前这三项均未具备。

## 6. 数据到位后的命令（本次未执行）

以下均在中文 checkout 根目录执行。依赖预处理环境与训练环境分开；准备脚本只读取已存在文件。

```sh
python scripts/data/prepare_aishell1.py lexicon --lexicon /test21/aishell1/resource_aishell/lexicon.txt --out /test21/aishell1/derived/dictionary_audit_v1.json
python scripts/data/prepare_aishell1.py audit-audio --data-root /test21/aishell1/data_aishell --out /test21/aishell1/derived/audio_audit_v1
python scripts/data/prepare_aishell1.py labels --audio-audit /test21/aishell1/derived/audio_audit_v1 --lexicon /test21/aishell1/resource_aishell/lexicon.txt --out /test21/aishell1/derived/labels_review_v1
```

存在异常时先处理独立决定文件，用新的 `--out` 和 `--resolutions` 重新生成；不能覆盖旧审计目录。全部已解决后：

```sh
python scripts/data/prepare_aishell1.py freeze --data-root /test21/aishell1/data_aishell --labels /test21/aishell1/derived/labels_review_v2 --out /test21/aishell1/derived/lexicon_v1/frozen
python scripts/server/aishell1_lexicon_pipeline.py --inspect
```

后续用户明确允许训练后，先将两组模板的 `execution.allow_training` 同时改为 true，再运行以下命令。预检生成的配置副本保持共同预算，正式队列使用副本而非模板。不要在预检后编辑副本。

```sh
/root/anaconda3/envs/whd/bin/python scripts/server/aishell1_lexicon_pipeline.py --preflight --audit-dir artifacts/runs/aishell1_lexicon_YYYYMMDD_attempt1 --wait-for-gpu
/root/anaconda3/envs/whd/bin/python -u scripts/server/aishell1_lexicon_pipeline.py --execute --gate artifacts/runs/aishell1_lexicon_YYYYMMDD_attempt1/prelaunch_validation.json --wait-for-gpu
```

与英文队列共用 `/tmp/phone69_460h_serial.lock`，并额外检查实际 GPU 进程；占用时等待或退出，不停止英文任务。代码路径由脚本位置决定，不会 cd 到英文目录。正式 OOM 或失败保留现场并暂停；不能单独改一组预算或自行换算法。显式 `--resume` 仅可恢复同一配置和批次的完整 epoch 边界。

完整 checkpoint 包含模型、优化器、调度器、循环进度、协议和随机状态；初始化、每轮边界、每500步及每次最佳验证均使用唯一文件名，不删旧文件。仅 latest/best 指针和 run 状态允许原子更新。

## 7. 监控、测评及同步

`status_aishell1.py --audit-dir <门槛目录>` 只读输出服务器 UTC、进程、GPU、queue/run、最新日志/进度/checkpoint 的服务器文件年龄。三者同时超过3600秒才列为可能停滞，仍需核对真实进程和GPU；单独验证耗时或连接失败不足以重启。当前没有中文训练，不创建空转监控；正式启动时按用户偏好启用每小时检查，正常推进静默，失败/明确停滞/完成/需处理才通知。没有自动重启入口。

队列依次训练基线与动态方案，各完成50轮后用其 dev-best、完整test、beam20评测；已完整完成的结果可验证后复用，部分失败结果不覆盖重跑。`per_sample.jsonl` 和 `per_sample_sdi.csv` 包含原文、完整参考/预测符号、S/D/I、去调错误、条件声调计数、真实音频秒数、十档和观测长度。总 PER 是总错误/总参考符号数；S/D/I 比率使用同一分母。去调率重新对齐去调后的完整符号；声调率仅比较去调对齐中基础发音匹配且双方带调的项，明确记录分母，不视为独立声调识别准确率。

语速为参考词典符号数/真实秒数，按 `(语速,录音ID)` 稳定排序分10个近等样本数区间，使用完整冻结 test 分档，两组完全相同。与 emformer/英文的单位不同，不能直接混称同一套语速档。统计实际压缩长度及分布，不输出 RTF。完整比较再次逐样本重算最小编辑对齐并检查相同参考、分档、数据与批次协议。

仅中文代码、配置、测试、报告和小记录进入同名 Gitee 分支；GitHub 沿既有镜像流程。数据、原词典、模型、凭据、checkpoint 和单文件超过10MiB的清单留服务器，提交外部文件的路径/数量/指纹清单。当前没有任何真实数据、标签或训练结果可同步，不能为交付生成占位“通过”报告。
