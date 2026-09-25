# 实验登记表（Experiment Log）

> 规则：**每启动一个实验必须先在此表登记一行；实验结束/失败必须回填结果。**
> 命名与 ckpt 规则见 [PROTOCOL.md](PROTOCOL.md)。

| ID | 启动日期 | 分支 | commit | 配置文件 | filter | seed | 状态 | 关键结果 | 日志/ckpt 位置（服务器） | 备注 |
|----|----------|------|--------|----------|--------|------|------|----------|--------------------------|------|
| a3-none-s42 | 2026-09-25 | exp/a3-baseline-seeds | （见服务器部署记录） | configs/exp/a3-none-s42.yaml | none | 42 | 待启动 | — | /test21/conformer-rnnt-asr/exp/a3-none-s42/ | A3 锚点：无压缩基线，3 种子之一 |
| a3-none-s1111 | — | exp/a3-baseline-seeds | — | configs/exp/a3-none-s1111.yaml | none | 1111 | 排队 | — | /test21/conformer-rnnt-asr/exp/a3-none-s1111/ | 串行排队，等 s42 释放 GPU |
| a3-none-s2222 | — | exp/a3-baseline-seeds | — | configs/exp/a3-none-s2222.yaml | none | 2222 | 排队 | — | /test21/conformer-rnnt-asr/exp/a3-none-s2222/ | 串行排队 |

## 状态取值
`排队 → 待启动 → 运行中 → 已完成 / 失败 / 已中止`

## 回填格式（实验结束后把"关键结果"换成）
`val_loss=…  PER=…%  WER=…%（若可用）  best_ckpt=<文件名>  墙钟时间=…h`
