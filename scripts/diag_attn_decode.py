#!/usr/bin/env python3
"""单句调试: torchaudio 解码 vs 手动贪心 (直接调 predictor) 对比。"""

import sys

import torch

sys.path.insert(0, "/test21/conformer-rnnt-asr-attn")
import yaml  # noqa: E402


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

from conformer_rnnt.lightning_module import LibriSpeechRNNTModule  # noqa: E402

config = load_config("/test21/conformer-rnnt-asr-attn/configs/server_eval_greedy.yaml")
model = LibriSpeechRNNTModule.from_config(config)
ckpt = "/test21/conformer-rnnt-asr/exp/checkpoints/val-epoch=42-step=26700.ckpt"
model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
device = torch.device("cuda")
model = model.to(device).eval()
blank = model.blank_idx

for _batch_idx, (batch, transcripts) in enumerate(model.test_dataloader()):
    features = batch.features.to(device)
    feat_lens = batch.feature_lengths.to(device)
    enc, enc_lens = model.model.transcribe(features, feat_lens)
    T = enc.size(1)
    print(f"utt0: T={T} ref={str(transcripts[0])[:60]}")

    # (a) torchaudio 路径 (eval.py 同款)
    with torch.no_grad():
        hyps = model.model.transcribe(features[:1], feat_lens[:1])
    print("(a) transcribe:", hyps[0][0][:20] if isinstance(hyps, tuple) else hyps[0][:20])

    # (b) 手动贪心: 直接调 predict (与 smoke/e2e 同构)
    with torch.no_grad():
        one = torch.tensor([1], device=device)
        g, _, state = model.model.predict(
            torch.tensor([[blank]], device=device), one, None
        )
        out_tokens = []
        for t in range(T):
            joint, _, _ = model.model.join(
                enc[:, t : t + 1], one, g, one
            )
            pred = int(joint.argmax(dim=-1).item())
            if pred != blank:
                out_tokens.append(pred)
                g, _, state = model.model.predict(
                    torch.tensor([[pred]], device=device), one, state
                )
    print("(b) manual greedy:", out_tokens[:20])

    # (b2) 完整贪心解码序列还原成 token 串看重复模式
    print("(b) len:", len(out_tokens), "first10:", out_tokens[:10])
    break
