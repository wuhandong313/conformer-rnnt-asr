import torch, time
from conformer_rnnt.model import conformer_rnnt_model
from conformer_rnnt.spec_decode import greedy_decode, speculative_greedy_decode

torch.manual_seed(42)
m = conformer_rnnt_model(
    input_dim=80, encoding_dim=64, num_symbols=75,
    right_context_length=0, time_reduction_input_dim=16, time_reduction_stride=4,
    num_heads=4, ffn_dim=64, num_layers_block1=1, num_layers_block2=1,
    dropout=0.0, depthwise_conv_kernel_size=7,
    compression_filter="none",
    predictor_type="attn", attn_d_model=32, attn_num_layers=1, attn_num_heads=4,
    attn_ffn_dim=32, attn_left_context=8, attn_dropout=0.0,
    medusa_num_heads=2,
).eval()

for Tf in [8, 16, 32]:
    feat = torch.randn(1, Tf, 80)
    feat_len = torch.full((1,), Tf, dtype=torch.int32)
    with torch.no_grad():
        s = time.time()
        oracle = greedy_decode(m, feat, feat_len, 74, max_steps=3000)
        tg = time.time() - s
        s = time.time()
        spec = speculative_greedy_decode(m, feat, feat_len, 74, max_draft=2,
                                          frame_window=8, per_token_blank=4, max_steps=3000)
        ts = time.time() - s
    same = oracle == spec
    print(f"Tf={Tf} T_enc~{Tf//4}: oracle {len(oracle)}t {tg:.2f}s | spec {len(spec)}t {ts:.2f}s | same={same}", flush=True)
    if not same:
        print("  oracle:", oracle)
        print("  spec  :", spec)
        break
print("DONE", flush=True)
