# IPA / ArtVector 方向待完成事项

状态：2026-10-15，按用户要求暂停。当前只收尾本地代码、基础验证与工作树合并，
不继续服务器数据处理、正式训练或扩展实验。

## 已完成的基础

- 固定 38-token 完整 Epitran IPA 词表，保留 `ɹ̩` 和破擦音连接符，无 SPM。
- 外部标签读取：`<ipa_labels_path>/<下划线子集名>/ipa`，按语音 ID 与外部音频配对。
- 38×24 PanPhon 静态特征表，全部 token 完整识别；blank/padding 使用 mask。
- IPA RNN-T、语料级 PER、词表/特征表 checkpoint 指纹检查。
- 只读数据审计工具；无标签生成、复制或改写流程。
- 原 `configs/default.yaml` 保留，使用独立 IPA 配置显式启用此方向。

方法和读取配置详见 [ipa_vocabulary_review.md](ipa_vocabulary_review.md)。

## 恢复此方向时依次处理

1. **确认服务器完整 IPA 标签位置。** 音频已在 `/test21/librispeech/LibriSpeech/`，
   七个标准子集存在。检查到的外部标签主要是 `phone69.trans.txt`，500h 仅有
   `simple_ipa`，完整 38-token `ipa` 尚未找到。不要将两套标签混用。
2. **配置外部路径。** 调整 `configs/server_ipa.yaml` 的 `paths.ipa_labels_path`；
   仓库只读取数据，数据是否需要同步或生成，须另行确定，不在本轮执行。
3. **运行只读全量审计。** 核对七个子集的音频 ID、标签 ID、未知 token 和 SHA256。
   本地已有完整标签位于
   `/Users/wuhandong/Desktop/python/emformer_rnnt_copy/librispeech/Librispeech_phoneme_transcription`；
   之前扫描为 292,367 条语音、36,112,605 个 token、38 种 token。
4. **验证新的真实数据链路。** 用 38-token 配置做一次 CPU 小样本读取、损失/反向及
   checkpoint 保存恢复；不要把此前 41-token 版本的 smoke test 当作本版本的数据验证。
5. **决定是否加入 ArtV 训练。** 当前仅提供固定 24 维目标和 mask，没有 AP/VIB/ArtV
   辅助损失。若继续，先明确目标时间对齐、连接位置和损失权重，再做小规模对照。
6. **决定正式实验与同步。** 原 word 基线仍可显式运行；确定此方向后再推送最新
   `main`/`ipa` 并让服务器 IPA 工作树快进拉取，避免覆盖服务器主实验目录的本地改动。

## 本轮停止时的同步状态

最新 38-token 方案仅在本地收尾并合并。此前远程与服务器同步到 `997d07f`，
那是已弃用的 41-token 方案，不能拿来代表当前实现。
服务器 `/test21/conformer-rnnt-asr-ipa` 的旧字典/G2P 生成任务已终止；
未启动正式 IPA 训练，也未将数据搬入仓库。

## 基础验证

本地 23 项测试全部通过，包含小模型 RNN-T 损失/反向和外部读取验证。
本机训练依赖的临时补充包在 `/tmp/conformer-ipa-check-deps`；它只用于验证，
不属于项目或未来部署依赖。具备完整训练环境时运行：

```bash
python -m unittest discover -s tests
python -m compileall -q conformer_rnnt common.py train.py eval.py ipa_tokenizer.py ipa_labels.py ipa_features.py audit_ipa_data.py tests
bash -n scripts/ipa_pipeline.sh
```

基础测试涵盖外部读取、数据不被写入、标签缺失/ID 不匹配报错、38-token 编码、
24 维特征及 padding mask、RNN-T 小模型损失/反向、checkpoint 指纹和语料级 PER。
测试通过不代表已经完成服务器全量数据验证或有效性实验。
