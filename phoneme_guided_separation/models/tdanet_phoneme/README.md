# TDANetPhoneme（实验 4.1 音素锚定骨架）

## 命名

| 说法 | 实际 |
|------|------|
| 文案 / 计划里的 `TDANet-phonome` | 口语名（拼写 phonome） |
| 包目录 | `phoneme_guided_separation/models/tdanet_phoneme/`（下划线，可 import） |
| 类名 / `audionet_name` | **`TDANetPhoneme`** |

```yaml
audionet:
  audionet_name: TDANetPhoneme
  audionet_config:
    out_channels: 128
    in_channels: 512
    num_blocks: 16
    upsampling_depth: 5
    enc_kernel_size: 4
    num_sources: 2
    num_phonemes: 80          # 音素类数（含 sil），按 MFA 表改
    tf_num_layers: 2
    tf_nhead: 4
    tf_dim_feedforward: 512
    tf_dropout: 0.1
```

## 结构（相对 TDANet）

```text
encoder → ln → bottleneck → sm
  → channel_gen → [B,n_src,128,T]
  → 共享 PostSepTransformer
  → H
       ├─ ProbingNet → phoneme_logits [B,n_src,T,K]   # return_phoneme=True
       └─ mask_from_hidden → ×s → decoder → waveforms
```

## forward 约定

- `model(wav)` / `return_phoneme=False` → 仅 `waveforms` `[B, n_src, T]`（兼容现有 litmodule）
- `model(wav, return_phoneme=True)` → `(waveforms, phoneme_logits)`

## 损失辅助

见 `losses.py`：按 DataLoader 的 `clip_start/stop` 切 `{utt}.artv.npy`，对齐到探针帧长 T，再做 `PhonemeCELoss`（CE 用同区间的 `phone_ids`）。ArtV 帧级目标已对齐，MSE 可后续接上。
