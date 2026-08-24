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


if __name__ == "__main__":
    sys.exit(main())
