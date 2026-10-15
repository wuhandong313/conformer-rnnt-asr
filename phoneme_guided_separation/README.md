# Phoneme-Guided Speech Separation

这组参考代码使用音素标签和发音特征向量（ArtV）辅助 TDANet 语音分离。

## 目录

- `models/`：TDANet 模型及其变体。
  - `tdanet.py`：基础 TDANet，对应 `TDANet` 类。
  - `tdanet_best.py`：对应原有 `TDANetBest` 变体。
  - `tdanet_v2.py`：对应 `TDANetV2` 变体。
  - `tdanet_legacy.py`：旧版实现，对应 `TDANetOld` 类。
  - `tdanet_phoneme/`：音素与 ArtV 探针、分离后 Transformer，以及辅助损失。
- `preprocessing/`：MFA 强制对齐、ARPABET 音素 ID、IPA 映射和 ArtV 原型表。
  - `mfa_preprocess.py`：单条冒烟与批量 LibriSpeech 预处理入口。
  - `arpa_inventory.py`：音素表、去重音、IPA 映射与 ArtV 原型表生成。
  - [mfa_installation.md](preprocessing/mfa_installation.md)：MFA 环境安装。
  - [mfa_smoke_test.md](preprocessing/mfa_smoke_test.md)：单条音频验证。
  - [mfa_output_guide.md](preprocessing/mfa_output_guide.md)：TextGrid 和对齐结果说明。
  - [arpabet_to_ipa.md](preprocessing/arpabet_to_ipa.md)：去重音与 IPA 转换说明。
- `training/`：Lightning 训练模块、优化器、学习率调度和两阶段早停。
  - `audio_lightning_module.py`：语音分离、音素 CE 与高斯 ArtV MSE 的联合训练。
  - `alpha_early_stopping.py`：首次早停后调整 alpha 并继续训练的回调。
- [gaussian_artv_loss.png](gaussian_artv_loss.png)：高斯 ArtV 目标与 MSE 损失的代码截图。

目录与 Python 模块使用小写 `snake_case`；模型类名、配置标识和数据表文件名保留原有约定。

## 来源依赖

这些文件来自独立的语音分离项目。`mfa_preprocess.py` 仍依赖原项目的
`system_paths.py` 和 `configs/basic_env.yaml`；部分文档也引用了未随本次文件一起提供的资料。
当前整理仅调整名称与相关引用，尚未将这组代码接入本仓库的 Conformer-RNN-T 训练流程。
