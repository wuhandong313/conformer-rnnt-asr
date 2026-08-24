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


if __name__ == "__main__":
    sys.exit(main())
