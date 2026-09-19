"""端到端构建检查：从 configs/server.yaml 构建 ConformerRNNTModel (predictor.type=attn) 并跑前向。"""
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conformer_rnnt.model import conformer_rnnt_model_from_config  # noqa: E402


def main() -> int:
    with open("configs/server.yaml") as f:
        cfg = yaml.safe_load(f)
    print("predictor.type =", cfg["model"]["predictor"].get("type"))
    print("predictor.attn =", {k: v for k, v in cfg["model"]["predictor"].get("attn", {}).items()})
    print("training.predictor_lr_scale =", cfg["training"].get("predictor_lr_scale"))
    print("from_checkpoint =", cfg["training"].get("from_checkpoint"))

    m = conformer_rnnt_model_from_config(cfg["model"])
    n_pred = sum(p.numel() for p in m.predictor.parameters())
    n_all = sum(p.numel() for p in m.parameters())
    print(f"predictor params = {n_pred/1e6:.2f}M, total = {n_all/1e6:.2f}M")

    m.eval()
    B, T, U = 2, 100, 7
    num_symbols = m.predictor.embedding.num_embeddings - 1
    feats = torch.randn(B, T, cfg["model"]["input_dim"])
    feat_lens = torch.tensor([80, 96], dtype=torch.long)
    tokens = torch.randint(0, num_symbols, (B, U))
    tok_lens = torch.full((B,), U, dtype=torch.long)
    with torch.no_grad():
        logits, _, _, _ = m(feats, feat_lens, tokens, tok_lens)
    print("joint logits shape =", tuple(logits.shape), "finite =", torch.isfinite(logits).all().item())

    # 增量解码路径（RNNTBeamSearch 契约）：predict 单 token + state 穿线
    g, _, state = m.predictor.predict(torch.zeros(B, 1, dtype=torch.long), None, None)
    print("predict out =", tuple(g.shape), "state layers =", len(state),
          "per-layer =", [len(s) for s in state[:1]])
    print("E2E_CHECK: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
