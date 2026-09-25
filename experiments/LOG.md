# 实验登记表（Experiment Log）

> 规则：**每启动一个实验必须先在此表登记一行；实验结束/失败必须回填结果。**
> 命名与 ckpt 规则见 [PROTOCOL.md](PROTOCOL.md)。

| ID | 启动日期 | 分支 | commit | 配置文件 | filter | seed | 状态 | 关键结果 | 日志/ckpt 位置（服务器） | 备注 |
|----|----------|------|--------|----------|--------|------|------|----------|--------------------------|------|
| a3-none-s42 | 2026-09-25 08:32 | exp/a3-baseline-seeds | 9264b5c | configs/exp/a3-none-s42.yaml | none | 42 | 运行中 | — | /test21/conformer-rnnt-asr/exp/a3-none-s42/ | PID 65270（nohup，conda env whd，torch 2.6.0+cu124）；A100-80G；epoch0 约 2.4 it/s（2634 batch/epoch，约 19 min/epoch + 验证 28s/200步）；ckpt 纪律 save_top_k=1 |
| a3-none-s1111 | — | exp/a3-baseline-seeds | 9264b5c | configs/exp/a3-none-s1111.yaml | none | 1111 | 排队 | — | /test21/conformer-rnnt-asr/exp/a3-none-s1111/ | 串行排队，等 s42 释放 GPU（单卡 A100） |
| a3-none-s2222 | — | exp/a3-baseline-seeds | 9264b5c | configs/exp/a3-none-s2222.yaml | none | 2222 | 排队 | — | /test21/conformer-rnnt-asr/exp/a3-none-s2222/ | 串行排队 |

## 状态取值
`排队 → 待启动 → 运行中 → 已完成 / 失败 / 已中止`

## 回填格式（实验结束后把"关键结果"换成）
`val_loss=…  PER=…%  WER=…%（若可用）  best_ckpt=<文件名>  墙钟时间=…h`
