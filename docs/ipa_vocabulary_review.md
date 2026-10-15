# 固定 IPA 词表审核

本分支用 **41 个固定 IPA 音素 + 1 个 RNN-T blank** 替代 word-BPE 输出。
词表文件为 `configs/ipa_vocab.json`，ID 按 `tokens` 数组顺序固定为 0–40，
blank 为 41；`model.num_symbols: 41` 不包括 blank。

## 审核结论与适用范围

这是一套面向 LibriSpeech 英语的**美式英语宽式音素转写约定**，覆盖 CMUdict 的
39 个基础音素，并为 `AH0`、`ER0` 增加 `ə`、`ɚ` 两个区分。
它不是“IPA 官方规定的英语唯一词表”：IPA 是音标体系，英语口音和转写粒度会影响音素集合。

基础映射与本仓库参考代码
`phoneme_guided_separation/preprocessing/arpa_to_ipa.json` 一致；本分支另外保留两个弱读元音。
基于 [CMUdict 官方音素表](https://github.com/cmusphinx/cmudict/blob/master/cmudict.phones)
核对过 39 个源音素的覆盖，并参考 [IPA 官方音标表](https://www.internationalphoneticassociation.org/content/ipa-chart)
核对符号。映射规则由本项目明确选择，不应把 CMUdict 的 ARPABET 清单当作官方 IPA 映射。

供你重点审核的约定：

- `AH0 → ə`，`AH1/AH2 → ʌ`；`ER0 → ɚ`，`ER1/ER2 → ɝ`。
- 其他元音去掉 `0/1/2` 重音数字；不预测独立的主重音、次重音符号。
- `R → ɹ`；`IY → i`、`UW → u`，不另加长音号 `ː`。
- `CH → tʃ`、`JH → dʒ`，破擦音分别是一个 token。
- `AW/AY/EY/OW/OY → aʊ/aɪ/eɪ/oʊ/ɔɪ`，每个双元音也是一个 token。
- 不额外区分送气、闪音、音节化辅音或其他同位音；不合并 `ɑ/ɔ`。
- 没有词边界、静音、UNK、BOS 或 EOS 输出类；词间边界不会作为训练目标。
  blank 仅用于 RNN-T 对齐，并不代表真实的静音音素。

## 全部 41 个 token 与源音素

| ID | IPA | ARPABET 来源 |
| ---: | :--- | :--- |
| 0 | ɑ | AA0/AA1/AA2 |
| 1 | æ | AE0/AE1/AE2 |
| 2 | ʌ | AH1/AH2；无重音标记的 AH |
| 3 | ɔ | AO0/AO1/AO2 |
| 4 | aʊ | AW0/AW1/AW2 |
| 5 | aɪ | AY0/AY1/AY2 |
| 6 | b | B |
| 7 | tʃ | CH |
| 8 | d | D |
| 9 | ð | DH |
| 10 | ɛ | EH0/EH1/EH2 |
| 11 | ɝ | ER1/ER2；无重音标记的 ER |
| 12 | eɪ | EY0/EY1/EY2 |
| 13 | f | F |
| 14 | ɡ | G |
| 15 | h | HH |
| 16 | ɪ | IH0/IH1/IH2 |
| 17 | i | IY0/IY1/IY2 |
| 18 | dʒ | JH |
| 19 | k | K |
| 20 | l | L |
| 21 | m | M |
| 22 | n | N |
| 23 | ŋ | NG |
| 24 | oʊ | OW0/OW1/OW2 |
| 25 | ɔɪ | OY0/OY1/OY2 |
| 26 | p | P |
| 27 | ɹ | R |
| 28 | s | S |
| 29 | ʃ | SH |
| 30 | t | T |
| 31 | θ | TH |
| 32 | ʊ | UH0/UH1/UH2 |
| 33 | u | UW0/UW1/UW2 |
| 34 | v | V |
| 35 | w | W |
| 36 | j | Y |
| 37 | z | Z |
| 38 | ʒ | ZH |
| 39 | ə | AH0 |
| 40 | ɚ | ER0 |

辅音 24 个，元音类 token 17 个。无重音数字的元音也接受，使用对应基础映射。
仅规范化书写别名：`g → ɡ`，`t͡ʃ/t͜ʃ → tʃ`，`d͡ʒ/d͜ʒ → dʒ`。
不会把 `r`、`ɻ`、`kʰ`、`iː` 等不同约定静默改成已有 token。

## 服务器现有标签检查（2026-10-15）

根目录：`/test21/librispeech/LibriSpeech`。初次只读扫描结果：

| 子集 | 现有 -ph 文件 | 有音素标签的语音数 | 情况 |
| :--- | ---: | ---: | :--- |
| train-clean-100 | 585 | 28,539 | 带重音 ARPABET |
| train-clean-360 | 2,097 | 104,014 | 带重音 ARPABET |
| train-other-500 | 2,784 | 148,688 | 另一套 IPA 风格标签 |
| dev-clean | 97 | 2,703 | 带重音 ARPABET |
| dev-other | 0 | 0 | 91 个有音频的章节缺少 -ph 文件 |
| test-clean | 87 | 2,620 | 带重音 ARPABET |
| test-other | 0 | 0 | 90 个有音频的章节缺少 -ph 文件 |

有标签的章节中音频 ID 与标签 ID 对齐。ARPABET 部分观察到全部 39 个基础音素。
`train-other-500` 的旧约定包含 `a/e/o`、分开的 `j/w/ɹ`，还观察到一个 `spn`；
例如它把某些双元音写成 `o w`、`e j`，与本词表的 `oʊ/eɪ` 粒度不同。
因此不能把所有现有 `-ph` 文件直接按同一种编码转换，也不能简单按字符拆 IPA。

本分支默认从各子集原始 `*.trans.txt` 英文文本统一生成 `*-ipa.trans.txt`：

1. 按词查 CMUdict，固定选择第一条发音；相同词复用缓存。
2. 字典外词使用 [g2p-en](https://github.com/Kyubyong/g2p) 预测 ARPABET 发音。
3. 应用本词表的确定性映射，保留完整 IPA 音素边界。
4. 在写入前检查所有标签、未知音素及音频 ID；逐文件原子写入。

这得到的是基于文本的发音监督，**不是人工听音标注**。
同形异音词采用字典第一条发音，没有句子上下文消歧；字典外词也可能预测错误。
报告会列出所有字典外词、词典哈希、发音缓存哈希及词表哈希，方便复查。
旧英文文本、旧 `-ph` 标签和音频均保留；已有不同内容的 `-ipa` 文件默认拒绝覆盖。

网上的 [THU-SPMI LibriSpeech IPA 标签](https://huggingface.co/datasets/THU-SPMI/librispeech-phoneme-labels)
使用另一套 CMU→IPA 约定。本分支没有直接套用其 phone_list，也没有混合两套标签。

## 编码、训练和评估逻辑

```text
英文 CHILD → CH AY1 L D → tʃ aɪ l d → [7, 5, 20, 8]
声学输入 → Conformer → RNN-T joiner → 42 类概率（41 个音素 + blank）
RNN-T 损失对完整音素序列训练；解码时去掉 blank → tʃ aɪ l d
```

只按空格切**完整音素**。没有训练 BPE 模型，也不加载任何 `.model` tokenizer 文件。
训练时 target padding 使用 blank，实际 target length 排除 padding。
checkpoint 保存词表指纹，恢复时拒绝 word/SPM checkpoint 或不同 IPA 词表。

IPA 评估自动选择 **语料级 PER**：所有语音的音素编辑错误总数 / 参考音素总数。
`tʃ` 换成 `dʒ` 算一次替换，`aɪ` 算一个参考音素。
PER 与原来的 word WER 不可直接比较；本分支输出 IPA，本身没有 IPA→英文词解码器。

`configs/server_ipa.yaml` 保留 word-none 基线的 2+8 层 Conformer、LSTM predictor、
10ms 帧移和无层间压缩配置，仅替换目标与词表，并把初始 batch 音素预算设为 1000。
输出目录独立为 `exp_ipa_960h_none/`。`configs/default.yaml` 也已改为 IPA，保留其原有模型/声学参数。
旧 word/SPM 配置仍可显式运行。

## 运行方式

在仓库根目录，使用训练环境的 Python：

```bash
# 仅离线标签生成需要，模型训练不依赖 g2p-en。
python -m pip install -r requirements-ipa-prep.txt
python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng

# 审核完整数据；不写入新的 IPA 标签。
python prepare_ipa.py /test21/librispeech --source words --check-only --report exp_ipa_960h_none/ipa_audit.json

# 生成统一标签（不启动训练）。
python prepare_ipa.py /test21/librispeech --source words --report exp_ipa_960h_none/ipa_audit.json

python train.py configs/server_ipa.yaml
python eval.py configs/server_ipa.yaml --checkpoint /path/to/ipa.ckpt --metric per --max-utts 10 --out /tmp/ipa_eval.csv
```

`--source arpabet` 可单独转换确认是 CMU ARPABET 的子集，例如
`--subsets train-clean-100 dev-clean test-clean`。它会拒绝旧 IPA 风格标签和 `spn`，
因此服务器全量数据应采用默认的 `--source words`。
审核通过后也可运行 `bash scripts/ipa_pipeline.sh` 串联准备、训练、评估。
流水线支持 `PYTHON`（训练环境）与 `PREP_PYTHON`（离线标签环境）分别配置；
如果仓库中存在 `.venv-ipa-prep/`，会自动使用它准备标签，训练继续使用原环境。
