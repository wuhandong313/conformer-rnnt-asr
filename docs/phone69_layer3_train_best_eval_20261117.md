# L3训练损失选模对照

2026-11-17用户要求将L3训练损失最低的已保存checkpoint下载到家中电脑，按之前的协议评测并比较效果。

最终完整第50轮边界（step93300）中的`ModelCheckpoint`状态确认：训练回调监控`Losses/train_loss`，每500个optimizer step检查一次，保存的best为`train-none-epoch=30-step=57500.ckpt`，得分9.764921188354492。该值对应保存检查点时的训练batch loss，不是整轮训练损失。文件SHA-256为`238c3e4afdb552d0d27b9f722aa38d3e066d167cea7f6746a45b5d018550e0e0`。文件名epoch从0起算，即第31轮。

训练CSV全部记录中的最小单步损失为4.002810001373291，对应global step84510，未保存该步权重。因此本次按训练checkpoint回调保存的best选择，不将未保存的日志最小值误认为可恢复的模型。

对照为已完成全量评测的验证损失best：`val-none-epoch=14-step=27324.ckpt`，test-clean corpus PER为6.393930897298213%。这是用户请求的checkpoint选模诊断对照；原验证选模评测与本次结果各自保存出处。

沿用完整test-clean 2620句、189023个参考音素、19452.480625秒音频，CPU／FP32／batch1／beam20、4线程／interop1、worker0、预热10句。复用L3的冻结训练源码、输入清单、归一化统计和词表，严格验证权重及全部输入SHA。完整解码同步记录逐层长度和主要MAC，并以独立公式核对编码器六个分项。

评测归档目录为`artifacts/evaluations/phone69_layer3_train_best_local_cpu_20261117/`。第一次启动因检测到重复OpenMP运行库，在解码前被一致性检查中止；原日志保留在`layer3_train_best/attempt_01/`。修正启动方式后使用独立的`attempt_02/`，直接启动评测Python并另设防休眠进程。独立核验显式选择已完成的attempt，保留所有失败记录。
