负责处理语音到各种音素特征

```bash
python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch
python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch --filter-librimix   # 只处理 librimix 的 s1/s2 用到的句子
python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch --filter-librimix-small   # 只处理 librimix-small 的 s1/s2
python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch --num-jobs 8   # MFA 章节内并行，默认 4；已有 phone_ids.npy 会跳过
python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch --filter-librimix-small --skip-download  # 本地已有 MFA 模型、避免 GitHub 限流
```

## 结果

一条音频（例如 `.../dev-clean/1272/128104/1272-128104-0000.flac`）对齐成功后，**相对路径不变**，在 `libri_phoneme` 下得到同名文件：

```text
libri_phoneme/unzip/LibriSpeech/dev-clean/1272/128104/
  1272-128104-0000.TextGrid       # MFA 原始对齐（词 + 音素时间戳）
  1272-128104-0000.phone_ids.npy  # 去重音 ARPABET 离散 ID，shape [N]
  1272-128104-0000.phone_times.npy # 各音素区间 [xmin, xmax] 秒，shape [N, 2]
  1272-128104-0000.artv.npy       # PanPhon 发音向量，shape [N, 24]
```

`N` = phones 层区间个数（含静音）。本条约 5.86 s，`N=66`。

| 文件 | 一条音频里是什么 |
|------|------------------|
| **TextGrid** | 两层：`words`（mister / quilter / …）和 `phones`（`M IH1 S T ER0` …）。每个区间有起止秒。 |
| **phone_ids.npy** | 每个 phones 区间一个整数。`IH1`→`IH`→**18**；静音 / 空标签 → **0**（`sil`）。词典见 [`arpa_phone2id.json`](./arpa_phone2id.json)，共 42 类。 |
| **phone_times.npy** | 每个 phones 区间的起止秒 `[xmin, xmax]`。训练按 DataLoader 的 `clip_start/stop` 切 ArtV；没有则回退读 TextGrid。 |
| **artv.npy** | 每个区间一行 24 维（`+1/-1/0`）。`sil` 为全 0；同一音素 ID 对应同一行（查表，不是再识别）。 |

本条开头对应关系：

| 时间 (s) | MFA 原标签 | 去重音 | ID | ArtV |
|----------|------------|--------|----|------|
| 0.00–0.54 | （空/静音） | `sil` | 0 | `[0, 0, …]` |
| 0.54–… | `M` | `M` | 23 | 鼻音等特征 |
| … | `IH1` | `IH` | 18 | `/ɪ/` 的 24 维 |
| … | `S` | `S` | 30 | `/s/` 的 24 维 |

训练时 DataLoader 给出 `clip_start/stop`，`losses.py` 按时间切 ArtV / phone_ids 再对齐到探针帧长 T；CE 用对齐后的 **phone_ids**，ArtV 帧级目标同样对齐好（MSE 可后接）。根目录另有全局表 `arpa_phone2id.json`、`arpa_artv.npy`（`[42, 24]`）。
