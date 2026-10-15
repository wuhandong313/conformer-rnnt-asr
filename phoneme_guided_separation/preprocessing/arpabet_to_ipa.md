# 为什么喂 PanPhon 前要「去重音 + ARPABET → IPA」

对照：[mfa_output_guide.md](./mfa_output_guide.md)、[ArtVector（ArtV）怎么算呢.md](../新架构图/ArtVector（ArtV）怎么算呢.md)、[PanPhon库.md](../新架构图/PanPhon库.md)、[名词解释和翻译.md](../新架构图2/名词解释和翻译.md)

一句话：**MFA 吐出的是英语习惯写法（ARPABET + 重音数字），PanPhon 只认国际音标（IPA）。中间必须做一次「翻译」；重音数字先剥掉，因为它不是另一个音素。**

---

## 1. 这件事在整条流水线里干什么？

真值支路实际是：

```text
MFA TextGrid 的 phones 层
    DH  IH1  S     …     sil
        │
        │  ① 去掉末尾 0/1/2
        ▼
    DH  IH   S
        │
        │  ② 查表：ARPABET → IPA
        ▼
    ð   ɪ    s
        │
        │  ③ PanPhon 查发音特征
        ▼
    ArtV 向量（约 21 维，+1/-1/0）
```

①② 合在一起，就是「喂 PanPhon 前的符号对齐」。  
不做的话：PanPhon 不认识 `IH1`，也往往不认识 `DH` 这种英文电报码。

---

## 2. 三种东西，别混成一种

| 名字 | 是什么 | 例子（this 的第一个音） | 谁产出 / 谁消费 |
|------|--------|-------------------------|-----------------|
| **ARPABET** | 英语语音学/Kaldi/MFA 常用的 **ASCII 音素码** | `DH`、`IH1` | MFA `english_us_arpa` **产出** |
| **IPA** | **国际音标**，跨语言统一符号 | `ð`、`ɪ` | 映射表产出；**PanPhon 只认这个** |
| **ArtV** | IPA 再编码成的 **数值向量** | `[voi=+1, cont=+1, …]` | PanPhon **产出**；损失用这个 |

所以：

- ARPABET / IPA = **标签字符串**（给人看、给对齐器用）
- ArtV = **特征向量**（给网络算距离）

PanPhon 的输入必须是 IPA 音段，不是拼音，也不是 ARPABET。详见 [PanPhon库.md](../新架构图/PanPhon库.md)。

---

## 3. 第一步：为什么要去掉重音数字？

MFA 的 `english_us_arpa` 给元音加了后缀：

| 后缀 | 含义 | 例 |
|------|------|-----|
| `1` | 主重音 | `IH1`（this） |
| `2` | 次重音 | `AH2`（publication 开头） |
| `0` | 弱读 | `AH0`（little 里的弱元音） |

**重音描述的是「这个音节响不响、突不突出」，不是「嘴型换成了另一个音」。**

对 ArtV / PanPhon 来说：

- 特征维是：清浊、鼻音、舌位高低前后、圆唇……（**音段**怎么发）
- **不**编码英语词重音（和中文声调一样，属于超音段；PanPhon 基本不管）

因此实现上先：

```text
IH1 → IH
AH0 → AH
EY1 → EY
```

`IH1` 和 `IH0` 进 PanPhon 后应是 **同一个** `/ɪ/` 的原型点。若不去掉数字，它们会被当成两个未知符号，查表失败或变成两个无关类别。

`sil` / `<eps>` 没有重音数字，这一步原样跳过；后面也不要当普通音素去查 PanPhon（用全 0 或 mask，见 ArtV 文档）。

---

## 4. 第二步：为什么还要映成 IPA？

剥掉数字之后，剩下的仍是 **ARPABET**（`DH`、`IH`、`SH`），不是 IPA。

同一发音，两套写法：

| ARPABET | IPA | 在 this / finished 里 |
|---------|-----|------------------------|
| `DH` | `ð` | this 的 th（浊） |
| `IH` | `ɪ` | this 的 i |
| `S` | `s` | this 的 s |
| `SH` | `ʃ` | finished 的 -sh- |
| `TH` | `θ` | three 的 th（清） |
| `Y` | `j` | year 的 y |
| `R` | `ɹ`（常见） | year / work |

PanPhon 内部词典按 **IPA** 建。你若把 `DH` 喂进去，它不知道这是 `ð`，特征表是空的或报错。

英文走 ARPABET→IPA，中文走拼音/声韵→IPA，**最后都进同一张 PanPhon 表**，ArtV 才在同一空间里可比。这就是图注里「IPA / ArtV 整合」的前置步骤。

---

## 5. 用本条 TextGrid 走一遍

`this`（0.55–0.71 s）：

| MFA 原标签 | 去重音 | IPA | 再交给 PanPhon |
|------------|--------|-----|----------------|
| `DH` | `DH` | `ð` | 浊、齿间、擦音…… |
| `IH1` | `IH` | `ɪ` | 前、高、不圆唇…… |
| `S` | `S` | `s` | 清、齿龈、擦音…… |

`little`：`L IH1 T AH0 L` → `L IH T AH L` → 大致 `l ɪ t ə l`（`AH0` 习惯上当 schwa `ə`）。

到这里，才有「每个时间段一个 IPA 音素」，才能按帧查表得到 `ArtV_gt[t]`。

---

## 6. 不做会怎样？

| 漏哪一步 | 后果 |
|----------|------|
| 不去重音 | `IH0` / `IH1` / `IH2` 变成三个类；PanPhon 不认；CE 类别数虚高 |
| 不去 ARPABET→IPA | 英文标签和中文 IPA、PanPhon 对不上；ArtV 查表失败 |
| 两套符号混进同一张表 | `/ʃ/` 和 `SH` 被当成两个音，损失教歪 |

帧级 CE（实验 4 先跑通的那种）可以暂时 **继续用去重音后的 ARPABET ID**（不必 IPA）。  
**只有要算 PanPhon / ArtV-MSE 时，IPA 映射才是刚需。** 但映射表建议早点建好，避免两套 ID 后期对不齐。

---

## 7. 实现时注意两点

**（1）`AH` 的弱读有时要单独处理。**  
很多表把 `AH0` → `ə`（schwa），`AH1`/`AH2` → `ʌ`。这是少数「数字不只是重音、还暗示音质」的情况。先定一张表，全语料共用，不要有的句子当 `ə`、有的当 `ʌ`。

**（2）映射是查表，不是模型。**  
固定字典即可，例如：

```text
DH → ð
IH → ɪ
SH → ʃ
sil → （不查 PanPhon）
```

预计算 `phone_id → ArtV` 原型表 `[K, D]`，训练时只查、不反传。细节见 [ArtVector（ArtV）怎么算呢.md](../新架构图/ArtVector（ArtV）怎么算呢.md) 第 5 节。

---

## 8. 一句话

> **去重音**：把「同一个元音的轻/重读」收成一个音段。  
> **ARPABET → IPA**：把 MFA 的英语电报码译成 PanPhon 能查的国际音标。  
> 两步都做完，才能从 TextGrid 得到 ArtV 真值。

实现：词典 [`arpa_phone2id.json`](./arpa_phone2id.json)、IPA 表 [`arpa_to_ipa.json`](./arpa_to_ipa.json)、逻辑 [`arpa_inventory.py`](./arpa_inventory.py)；冒烟 [`mfa_preprocess.py`](./mfa_preprocess.py) 在对齐后会打印 ID 并写出 `artv.npy`。Python 依赖装在 **`convtasnet`**。
