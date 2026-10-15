# Epitran IPA 与 PanPhon 读取方案

此方向目前暂停。后续事项见 [IPA_TODO.md](IPA_TODO.md)。

## 当前选择

采用已有的完整 Epitran `ipa` 标签，保留音节化符号 `ɹ̩` 和破擦音连接符。
固定词表为 38 个 token，ID 0–37；RNN-T blank 为 38，输出共 39 类。
`configs/ipa_vocab.json` 的顺序是唯一的 ID 定义。

```text
a b d d͡ʒ e f h i j k l m n o p s t t͡ʃ u v w z
æ ð ŋ ɑ ɔ ə ɛ ɡ ɪ ɹ ɹ̩ ʃ ʊ ʌ ʒ θ
```

这是用户提供的英语标签的实际词表，不是 IPA 官方规定的唯一英语音素集合。
固定 tokenizer 按已有空格切分完整音素，不按 Unicode 字符拆分，不训练 SPM。
例如 `t͡ʃ`、`ɹ̩` 各为一个 token，`o w` 按现成标签保留为两个 token。

本地源目录：
`/Users/wuhandong/Desktop/python/emformer_rnnt_copy/librispeech/Librispeech_phoneme_transcription`。
其说明文件称这些标签由 Epitran 从文本生成。扫描七个子集得到 292,367 条语音、
36,112,605 个音素 token，每个子集都观察到上述 38 个 token，无重复 ID 或异常行。

`papers/artnet/ArtNet_camera_ready_20260611.pdf` 第 2 页 §2.1 和第 3 页 §3.1 明确采用
Epitran 文本→IPA，再通过 PanPhon 映射到 24 维特征。论文未公布完整词表或
附加符号清理参数，因此只能确认方法一致，不能据此断言现成标签逐项复现作者预处理。
本地参考代码中的 ARPABET→39 IPA 和早期 41 音素方案均不是本次采用的方案。

## 外部数据布局

音频和标签可以使用独立根目录：

```text
<dataset_path>/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac
<ipa_labels_path>/train_clean_100/ipa
```

`ipa` 文件每行是 `语音ID 空格分隔的IPA音素`。读取器将音频子集名中的连字符
转换成下划线查找标签目录，通过完整语音 ID 配对。每个子集标签一次读入缓存，
训练排序与取样共用该缓存。数据缺失、重复 ID、未知音素、音频/标签 ID 不一致都会报错。
程序不改写数据、不复制标签、不回退到英文文本、`phone69` 或 `simple_ipa`。

服务器检查到的音频根目录是 `/test21/librispeech`。其中只有部分子集的
`phone69.trans.txt` 和 `train_other_500/simple_ipa`，尚未找到完整 `ipa` 标签。
`configs/server_ipa.yaml` 的 `paths.ipa_labels_path` 必须指向后续确认的完整标签根目录。

## 24 维发音特征

`configs/ipa_articulatory_features.json` 用 PanPhon 0.22.2 离线生成，并对全部
38 个 token 验证：每个完整 token 恰好对应一个 PanPhon 音段、24 个特征。
运行时只读这张 JSON 表，不依赖 PanPhon 安装。

数值为 `-1/0/+1`；特征顺序保存于 `feature_names`。
`ɹ` 和 `ɹ̩` 的 `syl` 特征分别为 -1 和 +1，没有合并。
blank/padding 的向量为零并由 mask 排除，零向量不解释为真实发音。
checkpoint 同时检查词表与特征表指纹。

主训练仍是 IPA RNN-T，评估使用语料级 PER。
`_extract_articulatory_labels()` 可提供 `[B, U, 24]` 目标及 mask，
但尚未加入 ArtNet 的 AP、VIB 或辅助损失。

## 后续使用入口

保留原来的 `configs/default.yaml`，IPA 通过 `configs/ipa.yaml` 或
`configs/server_ipa.yaml` 显式启用。

```bash
# 只读审计；报告保存在仓库实验目录，数据目录不写入文件。
python audit_ipa_data.py /test21/librispeech --labels-root /path/to/external/ipa_labels --report exp_ipa_960h_none/ipa_audit.json

# 只有重建固定特征表才需要安装 PanPhon。
python -m pip install -r requirements-ipa-prep.txt
python ipa_features.py

# 方向恢复且数据审计通过后再运行。
python train.py configs/server_ipa.yaml
python eval.py configs/server_ipa.yaml --checkpoint /path/to/ipa.ckpt --metric per --out /tmp/ipa_eval.csv
```
