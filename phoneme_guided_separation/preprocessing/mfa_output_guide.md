# MFA 结果怎么读（TextGrid）

对照：本条输出 [`_mfa_out/19-198-0001/19-198-0001.TextGrid`](./_mfa_out/19-198-0001/19-198-0001.TextGrid)、安装见 [mfa_installation.md](./mfa_installation.md)、流程见 [强制对齐怎么做.md](../新架构图/强制对齐怎么做.md)。

一句话：**TextGrid 不是听写结果，而是「已知这句话说了什么」之后，把每个词 / 每个音素钉到时间轴上。**

---

## 1. 这条结果对应什么？

| 项 | 值 |
|----|-----|
| 说话人句 id | `19-198-0001`（LibriSpeech） |
| 音频时长 | **11.765 s**（文件头 `xmax`） |
| 转写大意 | *This little work was finished in the year eighteen o three and intended for immediate publication. It was disposed of to a bookseller. It was even advertised.* |
| 词典 / 声学模型 | `english_us_arpa`（美式英语，ARPABET 音素） |
| 两层标注 | `words`（37 段）+ `phones`（117 段） |

Praat 打开同一文件，会看到上下两条 IntervalTier，和这里的文本是一回事。

---

## 2. 文件头在说什么？

```text
File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 11.765
size = 2
```

- `xmin / xmax`：整段音频的起止时间（秒），从 0 到结束。
- `size = 2`：两层（tier）。本仓库关心的就是这两层：

```text
item [1]  name = "words"    intervals: size = 37
item [2]  name = "phones"   intervals: size = 117
```

每一层都是一串 **区间**：`[xmin, xmax]` + `text`。区间首尾相接，铺满 `[0, 11.765]`，没有空洞。

---

## 3. `words` 层：词级时间戳

每个区间一个词（或静音占位）。读法：

```text
intervals [2]:
    xmin = 0.55
    xmax = 0.71
    text = "this"
```

→ 音频 **0.55～0.71 秒** 说的是单词 **this**。

本条开头几词：

| 时间 (s) | 词 |
|----------|-----|
| 0.00 – 0.55 | `<eps>`（静音，见下节） |
| 0.55 – 0.71 | `this` |
| 0.71 – 0.80 | `<eps>` |
| 0.80 – 1.12 | `little` |
| 1.12 – 1.43 | `work` |
| 1.46 – 1.63 | `was` |
| 1.63 – 2.16 | `finished` |
| … | … |
| 10.84 – 11.58 | `advertised` |
| 11.58 – 11.765 | `<eps>` |

词与词之间常会插一小段 `<eps>`，对应气口、停顿、爆破音前后的空白。

---

## 4. `phones` 层：音素级时间戳（后面算 ArtV 用这一层）

每个区间一个 **ARPABET 音素**（或 `sil`）。同一段 `this`：

| 时间 (s) | 音素 | 对应 |
|----------|------|------|
| 0.55 – 0.60 | `DH` | /ð/（this 的 th） |
| 0.60 – 0.68 | `IH1` | /ɪ/，**主重音** |
| 0.68 – 0.71 | `S` | /s/ |

`little`（0.80–1.12）拆成：`L IH1 T AH0 L`。  
`work`（1.12–1.43）拆成：`W ER1 K`。

**词层边界和音素层边界是对齐的**：一个词的 `[xmin, xmax]` 等于它内部若干音素区间的并。静音也是：words 的 `<eps>` 对应 phones 的 `sil`。

---

## 5. 特殊符号

| 符号 | 出现层 | 含义 |
|------|--------|------|
| `<eps>` | words | epsilon：没有词，通常是静音 / 停顿 |
| `sil` | phones | silence：静音音素，和上面的 `<eps>` 成对出现 |
| 空 `text = ""` | 有的对齐器会用 | 本条 MFA 输出主要用 `<eps>` / `sil`，不是空串 |

本条里有一段很长的静音：**6.42～8.09 s（约 1.67 s）**，夹在 `publication` 和 `it` 之间，对应转写里两句之间的停顿。这是正常现象，不是对齐坏了。

做帧级标签时：这段应标成 **静音**，不要当成某个音素拉长。

---

## 6. ARPABET 怎么读？（`english_us_arpa`）

标签是 **大写 ARPABET**，不是 IPA。

### 6.1 辅音（本条出现过的）

| 标签 | 大致 IPA | 例 |
|------|----------|-----|
| `DH` | ð | this |
| `TH` | θ | three |
| `SH` | ʃ | finished / publication |
| `S` `Z` `T` `D` `K` `P` `B` `F` `V` `M` `N` `L` `R` `W` `Y` | 与字母直觉接近 | — |

### 6.2 元音后面的数字 = 重音，不是另一个音

| 后缀 | 含义 |
|------|------|
| `1` | 主重音（primary） |
| `2` | 次重音（secondary） |
| `0` | 弱读 / 无重音 |

所以 **`IH1` 和 `IH0` 是同一个元音 /ɪ/**，只是重音不同：

- `this` → `IH1`
- `little` 里弱读的 schwa → `AH0`
- `publication` → `P AH2 B L IH0 K EY1 SH AH0 N`（次重音 + 主重音 + 弱读）

喂 PanPhon 之前通常要 **去掉 0/1/2**，再把 ARPABET 映成 IPA（例如 `DH` → `ð`，`IH` → `ɪ`）。为什么要这两步，见 [arpabet_to_ipa.md](./arpabet_to_ipa.md)；ArtV 怎么算见 [ArtVector（ArtV）怎么算呢.md](../新架构图/ArtVector（ArtV）怎么算呢.md)。

---

## 7. 用本文件走读一句

转写片段：*this little work was finished*

```text
时间轴 (s)
0.55      0.71  0.80           1.12      1.43  1.46     1.63            2.16
 |-- this --|sil|--  little  --|-- work --|sil|-- was --|-- finished --|
 |DH IH1 S |    |L IH1 T AH0 L| W ER1 K |    |W AH1 Z | F IH1 N IH0 SH T|
```

听音频时：若 0.55 秒附近确实是 “this”，1.12 附近是 “work”，对齐就合理。  
若整词错位（例如 “finished” 被标到后面静音里），才需要查 OOV、转写或采样率。

---

## 8. 和本项目后续步骤的关系

```text
TextGrid
  → 只取 phones 层（跳过 sil，或单独当 silence 类）
  → ARPABET 去重音 → IPA
  → 按帧展开：每一帧取当前音素
  → PanPhon 查表 → ArtV_gt[t]
```

训练时读的是 **离线缓存的帧级标签 / ArtV**，不会每 step 再跑 MFA。  
实验 4 里 DataLoader 的音素真值，来源就是这类 TextGrid（见 [实验4对应的代码编写.md](../实验计划4-新架构图2/实验4对应的代码编写.md) 里「MFA 帧级音素标签」）。

`mfa_preprocess.py` 打印时会跳过空标签；`<eps>` / `sil` 仍会打出来，方便核对停顿。

---

## 9. 抽查清单（看本条是否能当真值）

- [x] 词序列与 LibriSpeech 转写一致（本条是）
- [ ] 随机听：词边界是否切在真实词上，不要整词错位
- [ ] 长 `sil` 是否对应真实停顿（本条 6.42–8.09 应对应句间停顿）
- [ ] 极短音素（如 0.03 s 的 `sil`）过多时，边界可能偏碎，一般仍可用
- [ ] 不要把 ARPABET 和中文音素、IPA 混成同一套标签再喂 PanPhon
