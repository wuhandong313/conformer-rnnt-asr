#!/usr/bin/env python3
"""端到端一致性诊断: 单条 utterance 上 (1) TF RNNT loss, (2) 贪心解码逐步 logits。

若 TF loss 正常而贪心跑飞, 打印每步 top-3 logits 定位发散点。
"""

import glob
import os
import sys

import torch
import yaml

sys.path.insert(0, "/test21/conformer-rnnt-asr")

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule


def main() -> None:
    config = yaml.safe_load(open("/test21/conformer-rnnt-asr/configs/server.yaml"))
    config["paths"]["from_checkpoint"] = (
        "/test21/conformer-rnnt-asr/exp/checkpoints/val-epoch=42-step=26700.ckpt"
    )
    module = LibriSpeechRNNTModule.from_config(config)
    sd = torch.load(config["paths"]["from_checkpoint"], map_location="cpu")["state_dict"]
    module.load_state_dict(sd)
    module = module.cuda().eval()
    blank = module.blank_idx
    print(f"blank_idx={blank}")

    # ── 取一条测试 utterance ──
    dl = module.test_dataloader()
    batch = next(iter(dl))[0]
    features = batch.features.cuda()
    flens = batch.feature_lengths.cuda()
    targets = batch.targets.cuda()
    tlens = batch.target_lengths.cuda()

    with torch.no_grad():
        # ── 1. TF loss (与训练一致: blank 前置) ──
        prep = targets.new_zeros(targets.size(0), targets.size(1) + 1)
        prep[:, 1:] = targets
        prep[:, 0] = blank
        plens = tlens + 1
        logits, sl, tl, _, _, _ = module.model.forward_train_aux(
            features[:1], flens[:1], prep[:1], plens[:1]
        )
        sl = torch.clamp(sl, max=logits.size(1)).to(torch.int32)
        loss = module.loss(logits.float(), targets[:1], sl, tlens[:1].to(torch.int32))
        print(f"TF rnnt_loss (sum) = {loss.item():.3f}")

        # ── 2. 贪心解码逐步 ──
        enc, enc_len = module.model.transcribe(features[:1], flens[:1])
        T = int(enc_len[0])
        state = None
        emits = []
        t = 0
        step = 0
        while t < T and step < 4000:
            step += 1
            last_tok = emits[-1] if emits else blank
            tok_t = torch.tensor([[last_tok]], device="cuda")
            g, _, state = module.model.predict(tok_t, torch.ones(1, dtype=torch.long, device="cuda"), state)
            j = module.model.join(
                enc[:, t : t + 1, :],
                torch.ones(1, dtype=torch.long, device="cuda"),
                g, torch.ones(1, dtype=torch.long, device="cuda"),
            )[0]
            logits_step = torch.nn.functional.log_softmax(j[0, 0, 0, :].float(), dim=-1)
            top3 = torch.topk(logits_step, 3)
            pred = int(top3.indices[0])
            if pred == blank:
                t += 1
                continue
            emits.append(pred)
            if len(emits) <= 40:
                tops = " ".join(
                    f"{i}:{v:.2f}" for i, v in zip(top3.indices.tolist(), top3.values.tolist())
                )
                print(f"emit#{len(emits):>3} t={t:>4} tok={pred:>3} top3[{tops}]")

    print(f"\ntotal emits = {len(emits)} (T={T})")
    # 检查循环: 是否重复同一 token
    from collections import Counter
    c = Counter(emits)
    print("top-5 most emitted:", c.most_common(5))
    print("first 60 tokens:", emits[:60])


if __name__ == "__main__":
    main()
