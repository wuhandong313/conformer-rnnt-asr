# AISHELL-3 + THCHS-30 带调音节联合训练

2026-10-24 用户明确采用 AISHELL-3，并要求同时支持 THCHS-30（读取第二行）。当前是代码准备阶段：不下载数据、不启动正式训练、不启用空转训练监控。中文工作树保持与英文 main 隔离，沿用已有中文 Git 分支；配置、实验、审计前缀改为 `mandarin_syllable_mix`。

中文分支为 `codex/aishell1-lexicon`（保留既有分支名以延续历史），本地工作树 `/Users/wuhandong/Desktop/python/conformer-rnnt-asr/aishell1-lexicon`，服务器独立 checkout `/test21/conformer-rnnt-asr-aishell1`。仅同步该分支，不自动合并 main；上述名称中的 aishell1 不表示当前使用 AISHELL-1 数据。

## 标签与 tokenizer

两套原始逐句拼音直接合并，不经过汉字查词典、g2pM、重新择读或拼音到声韵母转换。一个完整带调音节是一个 token，例如 `zhong1`、`guo2`、`de5`；保留原始声调、儿化及字母写法，不按字符拆分。AISHELL-1 的217类声韵母库存不再用于当前实验。

AISHELL-3 原文 `train/content.txt`、`test/content.txt` 的格式是 `录音ID<TAB>汉字 拼音 汉字 拼音 ...`。解析器提取拼音字段，保存原始行、文字、来源文件 SHA-256 和行号。无发音的标点单独记录，不进入音节目标；无效交替字段、不带调或非音节标签进入错误审计，不能静默过滤。

THCHS-30 的 `data/<录音ID>.wav.trn` 第1行保存文字，**第2行原样作为完整带调音节目标**，第3行声韵母完全不参与训练标签。官方 `train/dev/test` 音频清单保持原划分。测试夹具特意把第3行设置为非法内容，以验证不会误读这一行。

参考：[AISHELL-3 论文第2节](https://arxiv.org/html/2010.11567v2#S2)、[AISHELL-3 内容读取实现](https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/aishell3.py)、[THCHS-30 三行样例](https://github.com/lhotse-speech/lhotse/blob/master/lhotse/recipes/thchs_30.py)。这些说明不替代本地原始数据审计。

共享词表默认仅由合并训练集实际出现的音节建立，按字典序分配 `0..K-1`；RNN-T blank 为 `K`，总类别 `K+1`。没有 BOS/EOS/UNK，也不会用 blank 或 sil 代替未知目标。dev/test 中的未见音节单列审计并阻止冻结，可在独立核查后通过 `--inventory` 明确指定带指纹的完整音节库存，不能自动用测试标签扩词表。由于尚无真实数据，模板的 `model.num_symbols: null`，真实预检读取冻结词表后写入共同的实测 K，不编造类别数。

统一的是完整带调音节单位与共享 ID；不是重新生成两套语料的读音。相同字符串合并为同一 token，不同原始写法保留为不同 token。真实数据到位后须审查共享库存及每套语料的声调/符号分布，不能在尚未审计时宣称两套原始标注的拼写和读音约定完全一致。

## 两套数据一起使用的规则

- 每个训练 epoch 覆盖两套训练数据的全部样本一次，不额外复制、加权或按目标比例重采样。两组模型共用同一个样本批次列表、每轮顺序及增强种子。
- 样本 ID 和说话人 ID 带语料前缀，避免两套数据的短 ID 碰撞。清单保留 `corpus` 与 `official_split`，可在合并后独立统计和测评。
- AISHELL-3 公开划分为 train/test。为保留开发集选 best，从官方 train 的说话人中使用 seed42 固定抽取10%作为 dev，至少一名，保留正数训练说话人。抽取名单进入冻结协议；官方 test 原样保留。
- THCHS-30 保留官方 train/dev/test；其官方 train/dev 说话人重叠会明确报告并允许，不宣称这两部分说话人隔离。train/test、dev/test 的说话人重叠视为异常。
- 完整配对、重复 ID、缺失/多余音频、格式、内容一致性和跨划分完全相同的规范音频逐项审计。AISHELL-3 公开总量88,035条，THCHS-30 预期 train10,000/dev893/test2,495；任何版本数量差异均保留原因并阻止通过，不凑数、不删异常样本。实际到位后先核对发布版本。

## 音频与特征

AISHELL-3 原始44.1kHz、单声道 PCM16 在 CPU 数据读取时显式重采样到16kHz；THCHS-30 原始16kHz不重采样。原始 WAV 不改写。重采样使用 torchaudio 的 `sinc_interp_hann`，lowpass_filter_width6、rolloff0.99、beta=None，参数及库版本记录在协议中。逐条保存原始 PCM SHA-256、规范波形 SHA-256、原始采样率/采样点数、规范采样点数和实际 Mel 帧数。运行时再次核对；库版本改变导致结果不同会报错。

重采样后长度是 `ceil(N_original*16000/sample_rate_original)`，并用实际输出验证。语速分母使用 `N_original/sample_rate_original`，不以取整后的帧数估计秒数。

两组共用16kHz、FFT400、hop160、80维 Mel、Hann periodic、center/reflect、power2、HTK；沿用既有分段线性/对数变换。归一化仅由合并 train 的有效帧计算，dev/test 和补齐帧不参与。训练两次频率遮挡27、两次时间遮挡100/p0.2；验证测试无增强，无效补齐帧重新清零。

## 两组计算配置

| 项目 | stride4 基线 | stride1 动态 |
|---|---|---|
| 投影/拼帧 | 80→32；4帧拼成128 | 80→128；stride1 |
| 主干 | 2+8层、宽128、8头、FFN512、卷积31、dropout0.1 | 相同 |
| 层2后 | 原序列传递 | 自然余弦峰谷＋five_frames |
| 权重预测器 | 无 | 五帧640→32→5，softmax加权融合 |
| 编码输出 | 256 | 256 |
| 标签/输出类别 | 相同冻结音节词表，K+1类 | 相同 |

动态三点定义不变：峰位置+1、谷位置取候选，排序去重，全部自然候选保留；无候选保留第0帧，长度≤1沿用原实现。没有 keep_rate、数量配额、左右上下文切分、额外尾部保留或新增阈值。离散选帧不反传，五帧权重与特征反传；公共接口的 L1 差分不进入 five_frames 预测器。静态拼帧尾部补零取 ceil(T/4)，只属于张量对齐。

预测器 embedding128、两层128 LayerNorm-LSTM、输出256；joiner ReLU后输出K+1。Adam lr0.0005、betas0.9/0.999、eps1e-8；warmup10000后恒定；RNNT sum/clamp1、梯度裁剪10、累积1、FP32、seed42、50 epoch、无早停。best 按合并 dev 的句数加权平均损失选择。

两组主干维度相同，动态组额外多20,677个融合参数和7,680个投影参数。输出库存变化同时作用于两组，不宣称参数完全一致。

## 批次、预检与完整恢复

共同预算依次6000/4500/3000/2000/1000，指完整音节 token 数。联合张量守护使用 `B*max(real_Mel_T)*(max(U)+1)*(K+1)<1.9e9`，不再硬编码218类，也不假设动态压缩比例。两组必须采用同一个通过预检的预算，不能单独降低其中一组预算。单句超限不丢弃。

沿用真实批次 A10080GB 检查：最长音频、最长标签、最大联合张量和注意力压力、最大样本数、验证前向；每个训练压力形状至少两步以覆盖 Adam 状态创建与稳态。有限性、显存≤90%、参数/配置/源码/数据指纹、初始完整 checkpoint 恢复和快照磁盘容量均通过才生成启动门槛。

完整 checkpoint 保存模型、优化器、调度器、循环、随机状态、共同批次和新音节协议。旧 AISHELL-1/phone69/IPA checkpoint 缺少 `mandarin_syllable_protocol`，严格拒绝加载。初始、每轮边界、每500步和每次 best 使用唯一文件名；不删历史文件。默认仅从完整 epoch 边界恢复。

队列与英文训练共用 GPU 锁，检查实际 GPU 进程，不抢占现有任务。数据未准备和 `allow_training:false` 时拒绝启动。正常训练后才启用每小时监控，目前没有正式训练；异常保留现场，不改算法自动重跑。

## 命令（数据到位后，本次未执行）

```sh
python scripts/data/prepare_mandarin_syllables.py audit --data-root /test21 --aishell3-root /test21/aishell3 --thchs30-root /test21/data_thchs30 --out /test21/chinese_syllables/derived/audit_v1
python scripts/data/prepare_mandarin_syllables.py freeze --data-root /test21 --draft /test21/chinese_syllables/derived/audit_v1 --out /test21/chinese_syllables/derived/mix_v1/frozen
python scripts/server/mandarin_syllable_pipeline.py --inspect
```

取得正式训练授权并将两组 `execution.allow_training` 同时设为true后，使用新的审计目录运行 `--preflight --audit-dir ... --wait-for-gpu`；通过后以 `--execute --gate .../prelaunch_validation.json --wait-for-gpu` 开始队列。不能在通过预检后手改配置指纹或绕过门槛。

## 测评

各完成50轮后使用自己的 dev-best、beam20，在两套官方 test 上测评。保存 AISHELL-3、THCHS-30 各自结果和合并结果：带调 SyllableER、S/D/I 数量及相对参考音节总数的比率、重新对齐的去调 SyllableER、基础音节匹配条件下的声调错误和分母、实际压缩长度。合并错误率为总错误/总参考音节，不能平均两语料或逐句错误率。

每套语料分别按参考音节数/原始真实秒数排序，同速按带前缀录音ID排序，分十档，两模型共享完全相同分档。逐句结果包含文字、原始参考拼音、预测拼音、corpus、真实时长、错误数与观测长度。不输出 RTF，不把音节错误率称为 CER。

## 历史与验证范围

`configs/作废_aishell1_lexicon_20261024/`、`docs/作废_aishell1_lexicon_20261024/`、`artifacts/runs/作废_aishell1_lexicon_code_ready_20261024/` 保留旧方案，仅供历史和回归测试。当前默认入口只指向新联合音节配置。

本次测试使用临时生成的44.1kHz/16kHz音频和原始标签格式夹具，覆盖两语料合并、THCHS第二行、重采样、共享词表/批次/增强、有限损失梯度、完整恢复、分语料/合并指标和十档规则。真实语料数量、完整库存、正式GPU峰值、数据冻结、训练和最终测评尚未执行。
