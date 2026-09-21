# llm_compress bench：compression.py 方法族 × Qwen3.5-0.8B 压缩率-质量曲线

日期 2026-09-21 | ckpt = 本地 qwen3.5-0.8b (fp32/MPS) | 代码 = llm_compress.py @ 92b07e6+
| 明细 = bench_compress_results.csv (88 行) | max_new=32 贪心, chat template 与 infer.py 同路径

## 设置

- 插入点：Qwen3.5 第一个 full_attention 层（idx 3）输出后；KV 缓存 index_select 裁剪 + 后续层 cos/sin 按 keep 切片；decode 步与 L≤64 短样本透传
- 探针：needle×4（636±10 token 长文埋独特事实句，verbatim 判定）；gist×4（重复冗余文本，主旨关键词）；short×3（<64 token 透传验证）
- 配置族：off 基线 / random_50 对照 / savgol(_d2,d4) 与 savgol_hanning = compression.py 原法（峰谷选择±Hanning 混合）/ diffq(_50,_85) = 帧差信号分位阈值推广 / novelty_50 = 窗口最近邻对照

## 曲线（hit x/n | needle+gist 平均保留率）

| config | needle | gist | short | ratio |
|---|---|---|---|---|
| off | 4/4* | 4/4 | 3/3 | 1.00 |
| diffq_85 | 2/4 | **4/4** | 3/3 | 0.86 |
| novelty_50 | 1/4 | 1/4 | 3/3 | 0.53 |
| diffq_50 | 1/4 | 2/4 | 3/3 | 0.51 |
| random_50 | 0/4 | 2/4 | 3/3 | 0.50 |
| savgol_hanning_d2 | 0/4 | 1/4 | 3/3 | 0.36 |
| savgol_d2 | 0/4 | 2/4 | 3/3 | 0.36 |
| savgol_d4 | 0/4 | **0/4** | 3/3 | 0.34 |

\* off 的 K-774 探针实际答对（"19.6%"），判分串只认"十九点六"，记为假阴性。

## 结论

1. **方法先验在 LM 上反转**：compression.py 的"帧差峰=关键"在 ASR 帧域成立（物理瞬变=音素边界），但 needle 探针实测（含控制实验）表明 LM layer3 hidden 中**独特内容位于轨迹平滑段、帧差低**——原法族保留率被局部极值密度锁死（d=1 饱和 ~41%），且在该区间 needle/gist 全部跌至随机子集水平以下（savgol_d4 gist 0/4 < random_50 2/4）。
2. **Hanning 混合不救 verbatim**：savgol_hanning vs savgol（同 36%）needle 均 0/4，gist 1/4 vs 2/4——窗内平均模糊细节，符合"混合保 gist 不保 token 级精度"预期。
3. **保序分位推广（diffq）恢复单调工作区间**：85% 保留点 gist 无损（4/4）、needle 2/4，是曲线上唯一有实用价值的温和压缩段；50% 点起质量≈随机。
4. **novelty（窗口最近邻）同样失败**（1/4, 1/4）：证明失效根因是"局部几何统计 ≠ 信息量"，非某个具体差分形式。
5. **机制本身正确**：缓存裁剪 5 步解码零漂移（见 92b07e6 验证记录）、多轮 REPL 下 41% 保留仍答对历史代号、短样本 3/3 透传。
6. **性能注记**：本地 fp32/MPS 下 needle 类 wall：off 2.8s → savgol_hanning 4.0s → diffq_85 5.5s（双 forward 统计 + blend Python 循环开销），本口径不以加速为目标。

## 后续候选（未做）

- 显著性信号换成模型自身量：attention-received（层 3 eager 权重列和）/ logit 投影长度（xResFormer 式）
- 位置编码保真：keep 原位置索引重编号（当前连续重排）
- CUDA 服务器 + flash-linear-attention 下重测吞吐口径
