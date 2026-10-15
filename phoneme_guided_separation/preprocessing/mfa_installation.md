# 安装 Montreal Forced Aligner（MFA）

本仓库音素对齐分两段环境：

| 做什么 | 环境 | 为什么 |
|--------|------|--------|
| **`mfa` 命令行**（Kaldi 对齐） | **`aligner`** | MFA 带 Kaldi 等二进制，不要装进训练栈 |
| **Python 脚本与后续依赖**（`mfa_preprocess.py`、soundfile、panphon、训练代码） | **`convtasnet`** | 与仓库其余代码同一环境；脚本会自动去 `aligner` 里找 `mfa.exe` |

官方 MFA 文档：[Installation](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html)

---

## 0. 先确认有 conda

PowerShell / 终端里：

```bash
conda --help
```

能打出一长串子命令即可。没有的话先装 [Miniforge](https://conda-forge.org/download/) 或 Miniconda，再开新终端。

本机若已是 Anaconda（例如 `C:\Users\feng\anaconda3`），不必再装一份。训练环境 `convtasnet` 见 [创建conda环境.md](../基础环境搭建/创建conda环境.md)。

---

## 1. 创建 `aligner` 并安装 MFA（只装这一次）

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner -y
conda activate aligner
mfa --help
```

若提示找不到 `conda-forge` 频道，先执行一次：

```bash
conda config --add channels conda-forge
```

Windows 上可执行文件一般在：

```text
C:\Users\feng\anaconda3\envs\aligner\Scripts\mfa.exe
```

`mfa_preprocess.py` 在 `convtasnet` 里跑时，会按 `PATH` → 当前环境 → **兄弟环境 `aligner`** 依次查找。

---

## 2. 下载本仓库用的预训练模型

仍在 **`aligner`** 里（只需一次）：

```bash
conda activate aligner
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
```

脚本默认词典 / 声学模型都是 **`english_us_arpa`**，装在：

```text
~/Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict
~/Documents/MFA/pretrained_models/acoustic/english_us_arpa.zip
```

`mfa_preprocess.py` **检测到本地已有就不会再 download**。`mfa model download` 每次都会请求 GitHub API，未登录时每小时约 60 次，超了会报 `ModelsConnectionError`（和模型在不在本地无关）。

已经下过、只想对齐：`--skip-download`。真要重新下载：等限流重置，或设置 `GITHUB_TOKEN` / `MFA_GITHUB_TOKEN`。

---

## 3. 在 `convtasnet` 里装 Python 依赖

后续代码（重采样、去重音 ID、PanPhon ArtV、训练）一律装在 **`convtasnet`**：

```bash
conda activate convtasnet
pip install soundfile numpy panphon
```

`soundfile`：把 8 kHz LibriMix 重采样到 16 kHz 再喂 MFA。  
`panphon`：去重音 ARPABET → IPA → 发音特征向量（ArtV）。

---

## 4. 跑通一条对齐 + ID + ArtV

在 **当前仓库根目录** 下：

```bash
conda activate convtasnet
python phoneme_guided_separation/preprocessing/mfa_preprocess.py
```

已有 TextGrid、只想重算 ID/ArtV：

```bash
python phoneme_guided_separation/preprocessing/mfa_preprocess.py --skip-align --skip-download
```

可选参数见 `python phoneme_guided_separation/preprocessing/mfa_preprocess.py --help`。

成功时会：

1. 打印 TextGrid 的 **words / phones** 时间戳  
2. 去重音后查 [`arpa_phone2id.json`](./arpa_phone2id.json) 得到离散 ID  
3. 经 [`arpa_to_ipa.json`](./arpa_to_ipa.json) + PanPhon 得到 ArtV（`sil`/`spn`/`unk` 为全 0）

输出目录默认 `phoneme_guided_separation/preprocessing/_mfa_out/<utt_id>/`。

### 批量 LibriSpeech（镜像目录 + 进度条 / 预计结束时间）

默认：`libri` → `libri_phoneme`，相对路径不变（`.flac` 换成 `.phone_ids.npy` / `.artv.npy` / `.TextGrid`）。按章节调 MFA，已有 `phone_ids.npy` 的句子会跳过，可中断后续跑。

```bash
conda activate convtasnet
python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch
```

先小跑验证：

```bash
python phoneme_guided_separation/preprocessing/mfa_preprocess.py --batch --splits dev-clean --limit 8 --skip-download
```

进度条 postfix 里有 **ETA 墙钟时间** 和剩余时长。全量约 4 万条，按机器可能要数小时到一两天。

| 参数 | 含义 |
|------|------|
| `--src` / `--dst` | 默认即上面两个 data 路径 |
| `--splits train-clean-100` | 只跑某个 split |
| `--limit N` | 最多 N 条 |
| `--overwrite` | 不跳过已有输出 |
| `--num-jobs 4` | MFA 并行进程数 |

---

## 5. 更新 MFA

在 **`aligner`** 里（不要在 convtasnet 里 `conda update` MFA）：

```bash
conda activate aligner
conda update -c conda-forge montreal-forced-aligner kalpy --update-deps
```

MFA 3.0.5 起也可以：

```bash
mfa_update
```

---

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| `未找到 mfa` | 确认已建 `aligner` 且存在 `...\envs\aligner\Scripts\mfa.exe`；Python 仍用 `conda activate convtasnet` |
| `未安装 panphon` | `conda activate convtasnet` 后 `pip install panphon` |
| 装 MFA 极慢 | 可在 base 里 `conda install -c conda-forge mamba`，再用 `mamba create -n aligner -c conda-forge montreal-forced-aligner -y` |
| 不要把 MFA 装进 `convtasnet` | Kaldi / MFA 与 PyTorch 训练栈容易冲突 |
| 不要 `pip install montreal-forced-aligner` 当主安装方式 | 官方主路径是 conda-forge |

对齐流程见：[强制对齐怎么做.md](../新架构图/强制对齐怎么做.md)。  
TextGrid 怎么读：[mfa_output_guide.md](./mfa_output_guide.md)。  
去重音 / IPA / ArtV：[arpabet_to_ipa.md](./arpabet_to_ipa.md)。
