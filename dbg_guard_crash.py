import torch, time
from pytorch_lightning import seed_everything
from conformer_rnnt.lightning_module import LibriSpeechRNNTModule
import train as T

seed_everything(42, workers=True)
cfg = T.load_config("configs/server_lstm_word.yaml")
torch.set_float32_matmul_precision("high")
m = LibriSpeechRNNTModule.from_config(cfg)
dl = m.train_dataloader()
m = m.cuda().train()
it = iter(dl)
t0 = time.time()
for i in range(300):
    b = next(it)
    b = type(b)(*[x.cuda() if torch.is_tensor(x) else x for x in b])
    B, Tm, U = b.features.shape[0], b.features.shape[1], b.targets.shape[1]
    loss = m._step(b, i, "train")
    loss.backward()
    m.optimizer.zero_grad()
    if i % 10 == 0:
        print(f"{i}: B={B} Tmel={Tm} U={U} loss={float(loss):.1f} "
              f"alloc={torch.cuda.memory_allocated()/2**30:.1f}G "
              f"reserved={torch.cuda.memory_reserved()/2**30:.1f}G "
              f"max={torch.cuda.max_memory_allocated()/2**30:.1f}G "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
print("DBG_DONE_NO_CRASH", flush=True)
