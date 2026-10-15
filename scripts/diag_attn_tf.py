#!/usr/bin/env python3
"""隔离诊断: TF 全前向 vs 组件 API 逐 token, 同一真实 transcript 对比。"""

import sys

import torch

sys.path.insert(0, "/test21/conformer-rnnt-asr")
import yaml

import sentencepiece as spm  # noqa: E402

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule  # noqa: E402

with open("/test21/conformer-rnnt-asr/configs/server_eval_greedy.yaml") as f:
    config = yaml.safe_load(f)
model = LibriSpeechRNNTModule.from_config(config)
ckpt = "/test21/conformer-rnnt-asr/exp/checkpoints/val-epoch=42-step=26700.ckpt"
model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
device = torch.device("cuda")
model = model.to(device).eval()
blank = model.blank_idx
sp = spm.SentencePieceProcessor(model_file=config["paths"]["sp_model_path"])
print("blank_idx =", blank, "| token73 =", repr(sp.id_to_piece(73)))

for _batch_idx, (batch, transcripts) in enumerate(model.test_dataloader()):
    features = batch.features.to(device)
    feat_lens = batch.feature_lengths.to(device)
    targets = batch.targets.to(device)
    target_lens = batch.target_lengths.to(device)

    # TF 全前向
    with torch.no_grad():
        logits, src_len, tgt_len, _, _, _ = model.model.forward_train_aux(
            features[:1], feat_lens[:1], targets[:1], target_lens[:1]
        )
    tgt = targets[0, : target_lens[0]].tolist()
    print("ref tokens:", " ".join(sp.id_to_piece(t) for t in tgt[:12]))

    # TF 模拟贪心: 标准对齐检查 — (t,u) 列 argmax 是否倾向 targets[u]
    # 取 u=1 (predictor 已消费 blank+第一 token) 逐帧看 argmax
    for u in (0, 1, 2):
        col = logits[0, :, u, :].argmax(dim=-1)
        nonblank = [int(x) for x in col.tolist() if x != blank][:6]
        print(f"TF col u={u} first nonblank argmax:", nonblank)

    # 组件 API: 逐 token 喂 prepended targets, 对比 forward_full 的 tgt_enc
    prep = [blank] + tgt
    with torch.no_grad():
        tgt_enc_full, _, _ = model.model.predictor(
            torch.tensor([prep], device=device),
            torch.tensor([len(prep)], device=device),
            None,
        )
        g = None
        state = None
        max_diff = 0.0
        for u, tok in enumerate(prep):
            g, _, state = model.model.predict(
                torch.tensor([[tok]], device=device),
                torch.tensor([1], device=device),
                state,
            )
            d = (g[0, 0] - tgt_enc_full[0, u]).abs().max().item()
            max_diff = max(max_diff, d)
        print("component-API vs forward_full max |diff| =", max_diff)
    break
