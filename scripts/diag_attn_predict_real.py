#!/usr/bin/env python3
"""用真实配置/权重复核 PredictorAttn full vs incremental (CPU + CUDA, eval 模式)。

若真实模型也等价 ⇒ garbage 解码的根因不在 predictor, 需查解码路径
(transcribe / beam-search state batching / 特征归一化)。
"""

import glob
import sys

import torch
import yaml

sys.path.insert(0, "/test21/conformer-rnnt-asr")

from conformer_rnnt.predictor_attn import PredictorAttn


def check(pa: torch.nn.Module, tokens: torch.Tensor, tag: str) -> None:
    pa.eval()
    B, U = tokens.shape
    lengths = torch.full((B,), U, dtype=torch.long, device=tokens.device)
    with torch.no_grad():
        out_full, _, _ = pa(tokens, lengths)
        state = None
        outs = []
        for u in range(U):
            o, _, state = pa.predict(tokens[:, u : u + 1], lengths, state)
            outs.append(o[:, 0, :])
        inc = torch.stack(outs, dim=1)
    d = (inc - out_full).abs().max().item()
    print(f"[{tag}] B={B} U={U} predict-vs-full maxdiff = {d:.6f}")


def main() -> None:
    config = yaml.safe_load(open("/test21/conformer-rnnt-asr/configs/server.yaml"))
    n_sym = config["model"].get("num_symbols", 74) + 1  # +blank, 与 model.py 工厂一致
    pa = PredictorAttn(
        num_symbols=n_sym,
        output_dim=config["model"].get("encoding_dim", 512),
        symbol_embedding_dim=config["model"]["predictor"].get("symbol_embedding_dim", 128),
        d_model=config["model"]["predictor"].get("attn", {}).get("d_model", 256),
        num_layers=config["model"]["predictor"].get("attn", {}).get("num_layers", 4),
        num_heads=config["model"]["predictor"].get("attn", {}).get("num_heads", 4),
        ffn_dim=config["model"]["predictor"].get("attn", {}).get("ffn_dim", 1024),
        left_context=config["model"]["predictor"].get("attn", {}).get("left_context", 1024),
        dropout=config["model"]["predictor"].get("attn", {}).get("dropout", 0.1),
        max_len=config["model"]["predictor"].get("attn", {}).get("max_len", 4096),
    )
    print(f"predictor: d_model={pa.d_model} layers={len(pa.layers)} "
          f"left_context={pa.left_context} dropout={pa.layers[0].dropout.p}")

    # ckpt 权重 (lightning state_dict 里 predictor 前缀)
    ckpts = sorted(
        glob.glob("/test21/conformer-rnnt-asr/exp/checkpoints/val-epoch=42-*.ckpt")
    )
    loaded = False
    if ckpts:
        sd = torch.load(ckpts[0], map_location="cpu").get("state_dict", {})
        psd = {k[len("model.predictor."):]: v for k, v in sd.items()
               if k.startswith("model.predictor.")}
        missing, unexpected = pa.load_state_dict(psd, strict=True), None
        try:
            pa.load_state_dict(psd, strict=True)
            loaded = True
        except Exception as e:
            print(f"load ckpt predictor 失败: {e}")
        if loaded:
            print(f"loaded ckpt weights: {ckpts[0]} ({len(psd)} tensors)")

    torch.manual_seed(0)
    tokens = torch.randint(0, 74, (2, 64))
    tokens[:, 0] = 74
    check(pa.to("cpu"), tokens, "cpu " + ("ckpt" if loaded else "random"))

    if torch.cuda.is_available():
        check(pa.to("cuda"), tokens.cuda(), "cuda " + ("ckpt" if loaded else "random"))
        # 更长序列
        tl = torch.randint(0, 74, (3, 600))
        tl[:, 0] = 74
        check(pa.to("cuda"), tl.cuda(), "cuda long U=600")


if __name__ == "__main__":
    main()
