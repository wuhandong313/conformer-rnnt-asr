# Conformer-RNN-T 当前开发说明

本分支维护 LibriSpeech train-clean-100 + train-clean-360 的 460h Phone69 离线语音识别实验。当前配置包括不同 stride、五帧融合位置、特征平滑以及 Mel/第5层/第8层多阶段压缩。

- train.py 接受单个 YAML 配置路径；模型工厂位于 conformer_rnnt/model.py，Lightning 训练与数据处理位于 conformer_rnnt/lightning_module.py。
- configs/phone69_vocab.json 固定 69 个带重音 ARPABET 音素，blank ID 69。model.num_symbols 不包含 blank。标签按完整音素分割，保留大小写、重音和空格，过滤 spn。
- 当前 YAML 使用 model.sequence_mode: offline。整句有效序列参与自然峰谷检测，不设置 keep_rate 或左右上下文配额；padding 由有效长度屏蔽。
- 压缩代码位于 conformer_rnnt/compression/。staged_encoder.py 支持在 Mel 输入及全局 Conformer 层后插入压缩。
- MelSpectrogram 在 CPU 惰性创建，不进入模型 state_dict。保持现有 checkpoint 参数键和张量形状。
- 训练按音素 token 预算动态组批。修改批预算时使用完整 epoch 边界 checkpoint，保留优化器和调度器状态，并核验词表与压缩定义。
- 训练和评估环境需要 torch、torchaudio、pytorch-lightning、numpy、scipy、pyyaml；当前 tokenizer 不需要 SentencePiece。
- 所有软件测试在本地运行：python -m unittest discover -s tests。服务器仅运行明确授权的训练或评估，代码通过 Git 同步。

参见 [文件布局](docs/FILE_LAYOUT.md)、[Phone69 约定](docs/phone69_tokenizer.md)、[当前结果](docs/phone69_460h_results.md) 和 [权威评测归档](papers/phone69_460h_cpu_authoritative_20261108/README.md)。冻结结果目录内的源码仅用于复现实验，不随当前源码清理而改写。
