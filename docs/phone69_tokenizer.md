# 固定 phone69 tokenizer

## 词表和编码

`phone69_tokenizer.py` 和 `configs/phone69_vocab.json` 定义固定 ID，无需训练 tokenizer，也不加载 SentencePiece。

- 24 个辅音：`B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH`。
- 15 个元音：`AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW`，各保留重音 `0/1/2`，共 45 个。`0` 为无重音，`1` 为主重音，`2` 为次重音。
- 普通音素按字典序分配 ID 0–68，严格共有 69 个标签 token；编码时仅过滤原始噪声标记 `spn`。
- RNN-T blank 为 69，模型输出 70 类；blank 也用于有效长度之外的 padding。没有 UNK、BOS、EOS，也不把 ID 0 当作特殊符号。

例如 `CH AE1 P T ER0` 产生 5 个 token；`AH0`、`AH1`、`AH2` 是三个不同 token。只按空白分割完整音素，不转小写、不去重音、不进行 IPA 转换。未知音素和粘连字符串直接报错。

按用户确认，严格使用 69 个普通音素。clean-100 原始标签中有一次 `spn`，在训练编码和评估参考序列中显式过滤，不分配 ID、不参与监督或 PER 分母，也不转换成 blank。源标签文件保持原样。其他未知标记（包括 `SIL`、`SPN`）仍报错。

## 配置与读取

`configs/default.yaml`、`configs/server.yaml`、`configs/server_lstm_naive.yaml` 已移除音素 SPM 模型路径，使用：

```yaml
tokenizer:
  type: phone69
  vocab_path: configs/phone69_vocab.json
model:
  num_symbols: 69  # excludes blank; spn is filtered
data:
  transcript_suffix: "-ph"
  strip_spaces: false
  max_token_limit: 4000
```

词表路径相对于运行目录，训练和评估应在仓库根目录执行。`max_token_limit` 现在按真实音素数计数；服务器配方从原来的 16000 字符预算调整为 4000 音素，本地 default 保留较小的 1000 音素预算。精确批次和显存需求会变化，应在正式训练时核对。

音频和标签仍在仓库外，当前读取方式是：

```text
<paths.dataset_path>/LibriSpeech/<subset>/<speaker>/<chapter>/
  <utterance_id>.flac
  <speaker>-<chapter>-ph.trans.txt
```

每行格式是 `<utterance_id> <whitespace-delimited ARPABET phones>`。读取时不修改、复制或生成数据。独立的原始 `phone69.trans.txt` / `phone69` 平面文件仅用于词表核验，当前训练读取服务器已经分发好的 chapter 标签。

2026-10-15 对服务器 `/test21/librispeech/LibriSpeech` 只读核验：

| 子集 | 语句数 | 编码后音素 token 总数 | 固定词表之外的标签 |
| --- | ---: | ---: | --- |
| train-clean-100 | 28539 | 3515503 | 原始有 1 次 spn，编码时过滤 |
| train-clean-360 | 104014 | 12757024 | 无 |
| dev-clean | 2703 | 193062 | 无 |
| test-clean | 2620 | 189023 | 无 |

本地独立原始标签的上述四个子集也全部通过，dev-other 的 2864 句及 test-other 的 2939 句也符合词表。但服务器 dev-other/test-other 当前缺少 chapter `-ph` 标签，不能直接用于当前读取方式。

服务器 train-other-500 虽然有 `-ph.trans.txt`，内容实际是 simple IPA；本地也没有该子集的 phone69 文件。不要将其加入 phone69 训练列表。遇到这些 IPA 标签会报未知音素错误，不会自动混用。当前 phone69 默认训练 clean-100，可按需扩展为 clean-100 + clean-360。

## 训练、评估和迁移

固定 phone69 自动报告 corpus PER：全语料音素编辑错误数除以参考音素数。一个 `AH0 → AH1` 替换计一个音素错误。word-BPE 配置继续使用它们的 SPM 和 WER；IPA 配置维持独立模式。

Checkpoint 保存词表指纹与 token 顺序；训练恢复和评估均检查指纹。旧 74 类音素 SPM checkpoint 不能直接恢复为固定 phone69，需重新训练。新实验输出目录与旧实验分开，不覆盖旧结果。

## 已验证与待完成

- 31 项回归测试通过，包含小模型 RNN-T loss/backward、padding、解码 ID 0、批次音素计数、PER、词表指纹及 word-BPE 路径。
- 尚未开展真实音频完整评估或正式训练；未改动服务器代码、数据、checkpoint 或运行任务。
- 服务器 main 有 `eval.py` 和两个 pipeline 脚本的未提交修改；两个脚本与改造前本地 main 一致，`eval.py` 内容不同，更新前应保存并核对，不能直接覆盖。
- 服务器旧 ipa 工作树可在确认无人使用后清理；本地 main 已包含新版 IPA，无需再合并服务器旧实现。attn-predictor 保留独立，按具体需求挑选改动。
