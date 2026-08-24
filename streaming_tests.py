"""Standalone test runner (repo convention: no pytest dependency)."""
import sys
import traceback


def get_tests():
    import importlib
    ns = {}
    # 本文件内的 test_* 函数
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            ns[name] = obj
    # 外部依赖测试模块(若存在), 其内的 test_* 一并收集
    for dep in ("tests_tiny_model",):
        try:
            mod = importlib.import_module(dep)
        except ImportError:
            continue
        for name in dir(mod):
            if name.startswith("test_") and callable(getattr(mod, name)):
                ns[f"{dep}.{name}"] = getattr(mod, name)
    return ns


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    tests = get_tests()
    failed = 0
    for name in sorted(tests):
        if only and only not in name:
            continue
        try:
            tests[name]()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0

def test_chunk_plan_geometry():
    from conformer_rnnt.streaming.plan import ChunkPlan
    p = ChunkPlan(chunk_frames=16, left_context_frames=64,
                  right_context_frames=4, stride=4, total_frames=100)
    # chunk0: s=0 → 左侧零 pad 64, 右侧不需 pad
    assert p.window_for_chunk(0) == (0, 84, 64, 0), p.window_for_chunk(0)
    # chunk中段(s>=L): s=64,L=64 → 左文恰好是 mel[0:64], 无需 pad; 窗口全为真实帧
    p2 = ChunkPlan(chunk_frames=16, left_context_frames=64,
                   right_context_frames=4, stride=4, total_frames=200)
    assert p2.window_for_chunk(4) == (0, 84, 0, 0), p2.window_for_chunk(4)  # s=64


def test_chunk_plan_alignment_validation():
    from conformer_rnnt.streaming.plan import ChunkPlan
    try:
        ChunkPlan(chunk_frames=16, left_context_frames=65,
                  right_context_frames=4, stride=4, total_frames=100)
        raise AssertionError("应拒绝 65 (非 stride 倍数)")
    except ValueError:
        pass


def test_chunk_plan_first_and_last():
    from conformer_rnnt.streaming.plan import ChunkPlan
    p = ChunkPlan(chunk_frames=16, left_context_frames=64,
                  right_context_frames=4, stride=4, total_frames=36)
    # 首个 chunk: s=0 → left_pad 64; right_pad 0
    assert p.window_for_chunk(0) == (0, 84, 64, 0), p.window_for_chunk(0)
    # 末 chunk: s=32, c=4 → left_pad=32 right_pad=16 (缺 content 12 + rc 4)
    assert p.window_for_chunk(2) == (0, 84, 32, 16), p.window_for_chunk(2)
    assert p.chunk_content_frames(2) == 4


def test_feature_extractor_matches_offline_pipeline():
    import json
    import torch
    import torchaudio
    from conformer_rnnt.streaming.features import StreamingFeatureExtractor
    from common import piecewise_linear_log, GlobalStatsNormalization

    sr, n_fft, n_mels, hop = 16000, 400, 80, 160
    torch.manual_seed(0)
    pcm = torch.randn(1, 16000 * 4)  # 4s 随机音频

    stats = "/tmp/global_stats_test.json"
    json.dump({"mean": [0.0] * n_mels, "invstddev": [1.0] * n_mels}, open(stats, "w"))

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop)
    offline = GlobalStatsNormalization(stats)(piecewise_linear_log(mel(pcm)).transpose(1, 2))[0]

    ex = StreamingFeatureExtractor(sample_rate=sr, n_fft=n_fft, n_mels=n_mels,
                                   hop_length=hop, global_stats_path=stats)
    streamed = ex.compute_all(pcm[0])
    assert streamed.shape == offline.shape, (streamed.shape, offline.shape)
    assert torch.allclose(streamed, offline, atol=1e-6), (streamed - offline).abs().max()

    # 增量 push + flush 与一次性 compute_all 等价
    ex2 = StreamingFeatureExtractor(sample_rate=sr, n_fft=n_fft, n_mels=n_mels,
                                    hop_length=hop, global_stats_path=stats)
    chunk = 8000
    for i in range(0, pcm.numel(), chunk):
        ex2.push(pcm[0, i: i + chunk])
    ex2.flush()
    assert torch.allclose(ex2.frames(), streamed, atol=1e-6)


def test_chunked_encoder_none_commits_chunk():
    import torch
    from conformer_rnnt.streaming.encoder import ChunkedEncoder
    from tests_tiny_model import make_tiny_model

    model = make_tiny_model(compression_filter="none")
    enc = ChunkedEncoder(model=model, chunk_frames=16, left_context_frames=64,
                         right_context_frames=4, stride=4)
    torch.manual_seed(7)
    mel = torch.randn(1, 64 + 16 + 4, 80)  # 中段 chunk 窗口(左文真实)
    out, ids = enc.encode_window(mel, content_len=16 + 64)
    assert out.shape[1] == (64 + 16 + 4) // 4, out.shape            # 21 reduced
    assert tuple(ids) == tuple(range(21)), ids
    # 提交 = source 在 chunk 局部 [16, 20)
    committed = enc.committed_frames(out, ids, chunk_local_start=16, chunk_local_end=20)
    assert committed.shape == (1, 4, out.shape[-1]), committed.shape


def test_chunked_encoder_none_equals_full_transcribe():
    """相同窗口输入（content 80 + 尾部 rc 4 = 84 mel 帧），手动分 stage 的
    ChunkedEncoder 输出必须与整体 encoder.forward 逐位一致。"""
    import torch
    from conformer_rnnt.streaming.encoder import ChunkedEncoder
    from tests_tiny_model import make_tiny_model

    model = make_tiny_model(compression_filter="none")
    enc = ChunkedEncoder(model=model, chunk_frames=16, left_context_frames=64,
                         right_context_frames=4, stride=4)
    torch.manual_seed(2)
    W = 64 + 16 + 4                        # 84 mel 帧，含尾部 rc(4)
    mel = torch.randn(1, W, 80)
    out, ids = enc.encode_window(mel, content_len=64 + 16)
    ref, ref_len = model.transcriber(mel, torch.tensor([64 + 16], dtype=torch.int32))
    assert out.shape[1] == ref.shape[1], (out.shape, ref.shape)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_replicate_compression_equals_module_naive():
    import torch
    from conformer_rnnt.streaming.encoder import ChunkedEncoder, replicate_compression
    from tests_tiny_model import make_tiny_model

    model = make_tiny_model(compression_filter="naive")
    enc = ChunkedEncoder(model=model, chunk_frames=16, left_context_frames=64,
                         right_context_frames=4, stride=4)
    torch.manual_seed(11)
    W = 64 + 16 + 4
    mel = torch.randn(1, W, 80)
    # 先由 encode_window 拿到 conformer1 输出
    x = enc.enc.input_linear(mel)
    x, xl = enc.enc.time_reduction(x, torch.tensor([64 + 16], dtype=torch.int32))
    c1, _ = enc.enc.conformer1(x, xl + 1)
    # 复刻 vs 模块
    out, outlen, ids = replicate_compression(enc.comp, c1, c1.size(1))
    ref, reflen = model.transcriber.compression(
        c1, torch.tensor([c1.size(1)], dtype=torch.int32))
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    assert outlen.item() == reflen.item()


def test_chunked_encoder_naive_commits_ids():
    import torch
    from conformer_rnnt.streaming.encoder import ChunkedEncoder
    from tests_tiny_model import make_tiny_model

    model = make_tiny_model(compression_filter="naive")
    enc = ChunkedEncoder(model=model, chunk_frames=16, left_context_frames=64,
                         right_context_frames=4, stride=4)
    torch.manual_seed(13)
    mel = torch.randn(1, 64 + 16 + 4, 80)
    out, ids = enc.encode_window(mel, content_len=64 + 16)
    committed = enc.committed_frames(out, ids, chunk_local_start=16, chunk_local_end=20)
    committed_ids = [i for i, s in enumerate(ids) if 16 <= s < 20]
    assert len(committed_ids) > 0, "naive 窗口应有贡献帧"
    assert committed.shape[1] == len(committed_ids), (committed.shape[1], committed_ids)


def test_streaming_decoder_equals_greedy():
    import torch
    from conformer_rnnt.streaming.decoder import StreamingGreedyDecoder
    from conformer_rnnt.spec_decode import greedy_decode
    from tests_tiny_model import make_tiny_model

    model = make_tiny_model(compression_filter="none")
    blank = model.joiner.linear.out_features - 1   # 最后一个 logit 是 blank
    torch.manual_seed(3)
    mel = torch.randn(1, 200, 80)
    ln = torch.tensor([200], dtype=torch.int32)
    # 补 rc pad, 与 greedy_decode 的输入约定一致
    feat = torch.cat([mel, torch.zeros(1, 4, 80)], dim=1)
    oracle = greedy_decode(model, feat, ln, blank)

    enc, enc_len = model.transcribe(feat, ln)
    T = enc.size(1)
    dec = StreamingGreedyDecoder(model=model, blank_idx=blank)
    step = 7
    for t0 in range(0, T, step):
        dec.push(enc[:, t0: t0 + step, :])
    assert dec.partial() == oracle, ("streaming", dec.partial(), "offline", oracle)


def test_streaming_decoder_reset():
    import torch
    from conformer_rnnt.streaming.decoder import StreamingGreedyDecoder
    from tests_tiny_model import make_tiny_model
    model = make_tiny_model(compression_filter="none")
    dec = StreamingGreedyDecoder(model=model, blank_idx=model.joiner.linear.out_features - 1)
    dec.push(torch.randn(1, 5, model.transcriber.output_linear.out_features))
    dec._reset()
    assert dec.out == [] and dec.t == 0


def test_preview_readonly_and_shape():
    import torch
    from conformer_rnnt.streaming.preview import MedusaPreview
    from conformer_rnnt.streaming.decoder import StreamingGreedyDecoder
    from tests_tiny_model import make_tiny_model

    model = make_tiny_model(compression_filter="none", medusa_heads=2)
    blank = model.joiner.linear.out_features - 1
    dec = StreamingGreedyDecoder(model=model, blank_idx=blank)
    torch.manual_seed(5)
    feat = torch.cat([torch.randn(1, 120, 80), torch.zeros(1, 4, 80)], dim=1)
    enc, _ = model.transcribe(feat, torch.tensor([120], dtype=torch.int32))
    dec.push(enc[:, :40, :])

    prev = MedusaPreview(model=model)
    before = (dec.t, tuple(dec.out), dec.g_last.clone())
    drafts = prev.lookahead(decoder=dec, enc=dec._enc())
    assert len(drafts) == 2, drafts
    assert all(0 <= d <= blank for d in drafts)
    assert dec.t == before[0] and tuple(dec.out) == before[1]
    assert bool((dec.g_last == before[2]).all()), "g_last 被改写"


def test_recognizer_full_context_equals_offline():
    import torch
    from conformer_rnnt.streaming.recognizer import StreamingRecognizer
    from conformer_rnnt.spec_decode import greedy_decode
    from tests_tiny_model import make_tiny_model

    model = make_tiny_model(compression_filter="none")
    blank = model.joiner.linear.out_features - 1
    torch.manual_seed(9)
    T = 40  # 单个 chunk 恰好覆盖全句; L=0, C=T, R=4
    mel = torch.randn(1, T, 80)
    ln = torch.tensor([T], dtype=torch.int32)
    feat = torch.cat([mel, torch.zeros(1, 4, 80)], dim=1)

    rec = StreamingRecognizer(model=model, mel=mel[0], chunk_frames=T,
                              left_context_frames=0,
                              right_context_frames=4, stride=4, blank_idx=blank)
    final = rec.run_offline()
    oracle = greedy_decode(model, feat, ln, blank)
    assert final == oracle, ("streaming", final, "offline", oracle)


def test_preview_on_off_produces_identical_inference():
    """关键保证: medusa preview 只影响显示——开/关两次识别输出逐 token 一致。"""
    import torch
    from conformer_rnnt.streaming.recognizer import StreamingRecognizer
    from tests_tiny_model import make_tiny_model

    model = make_tiny_model(compression_filter="none", medusa_heads=2)
    blank = model.joiner.linear.out_features - 1
    torch.manual_seed(6)
    mel = torch.randn(40, 80)

    rec_on = StreamingRecognizer(model=model, mel=mel, chunk_frames=40,
                                 left_context_frames=0, right_context_frames=4,
                                 stride=4, blank_idx=blank, preview=True,
                                 preview_heads=2)
    rec_off = StreamingRecognizer(model=model, mel=mel, chunk_frames=40,
                                  left_context_frames=0, right_context_frames=4,
                                  stride=4, blank_idx=blank, preview=False)
    # 同一输入，同一识别逻辑：preview 开关不影响 decoder 轨迹
    assert rec_on.run_offline() == rec_off.run_offline(), "preview 影响了识别输出"


def test_preview_acceptance_metric():
    from conformer_rnnt.streaming.preview import preview_acceptance
    final = [1, 2, 3, 4, 5, 6]
    # 语义对齐 offline draft_after_emit: 采样在 emit 后, u = len(out)。
    # head j (0-based: drafts[j]) 检查 final_out[u + j]。
    # u=0(已发0 token, 发射后预测): h1->final[0]=1, h2->final[1]=2
    # u=2(已发2 token):            h1->final[2]=3, h2->final[3]=4
    log = [(0, [2, 9]), (2, [3, 5])]
    r = preview_acceptance(log, final)
    by = {a["head"]: a for a in r}
    # h1: [2!=1 ✗, 3==3 ✓] → 1/2
    assert by[1]["accepted"] == 1 and by[1]["drafts"] == 2, by[1]
    # h2: [9!=2 ✗, 5!=4 ✗] → 0/2
    assert by[2]["accepted"] == 0 and by[2]["drafts"] == 2, by[2]
    # 越界 draft (u 接近末尾) 不报错
    preview_acceptance([(4, [9, 9])], final)
    assert by[1]["rate"] == 0.5


if __name__ == "__main__":
    sys.exit(main())
