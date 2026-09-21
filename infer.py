"""Qwen3.5-0.8B 本地交互推理（常驻进程 + 流式输出 + 多轮会话记忆）。

权重与配套文件在 ./qwen3.5-0.8b/。模型只加载一次；输入 exit/quit 退出，
clear 清空会话历史。纯文本路径，不喂图/视频。

用法:
    python infer.py                         # 交互模式
    python infer.py -p "单发问题"            # 单次输出后退出
    python infer.py --thinking              # 开启思维链
    python infer.py --sample                # nucleus 采样 (默认贪心)

注: macOS 无 triton 轮子 → linear-attn 走 torch 回退实现, 启动时的
    "fast path" 警告属预期; CUDA 服务器装 flash-linear-attention 后消失。
    MPS 上 auto 选 fp32: bf16 在该回退路径会触发 Metal 断言崩溃。
"""

import argparse
import threading
import time
from typing import List

import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

IM_END = "<|" + "im_end" + "|" + ">"
OPEN_THINK = "<" + "think" + ">"
CLOSE_THINK = "<" + "/think" + ">"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3.5-0.8b", help="本地模型目录")
    ap.add_argument("-p", "--prompt", default=None, help="单次提问（不给则进交互）")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"],
                    help="auto: MPS 用 fp32 (绕开 Metal bf16 命令编码器断言), 其余 bf16")
    ap.add_argument("--sample", action="store_true", help="nucleus 采样替代贪心")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--thinking", action="store_true", help="开启思维链")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compress", action="store_true",
                    help="在注意力层输出后插入 token 压缩 (llm_compress)")
    ap.add_argument("--compress-after", type=int, default=3,
                    help="压缩插入点: 该层输出之后 (须为 full_attention 层 3/7/11/...)")
    ap.add_argument("--compress-mode", default="savgol_hanning",
                    choices=["naive", "savgol", "hanning", "savgol_hanning",
                             "diffq", "diffq_hanning", "novelty", "novelty_hanning"],
                    help="compression.py 对齐族 + 分位/窗口泛化族 (bench 对照)")
    ap.add_argument("--keep-frac", type=float, default=0.5,
                    help="novelty 模式目标保留比例 (压缩率主旋钮)")
    ap.add_argument("--compress-distance", type=int, default=2,
                    help="峰检测最小间距 (diff 模式压缩率旋钮)")
    ap.add_argument("--keep-last", type=int, default=16,
                    help="无条件保留的尾部 token 数")
    ap.add_argument("--min-len", type=int, default=64,
                    help="短于此长度不压缩")
    return ap.parse_args()


def load(args: argparse.Namespace):
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    t0 = time.time()
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:  # auto
        # torch 2.12 + Qwen3.5 linear-attn torch 回退路径在 MPS bf16 下触发
        # Metal 命令编码器 coalescing 断言 (第二轮生成稳定复现)，fp32 正常。
        dtype = torch.float32 if device == "mps" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype
    ).to(device).eval()
    print(
        f"[加载完成 {time.time() - t0:.1f}s | device={device} | "
        f"{'fp32' if dtype == torch.float32 else 'bf16'}]",
        flush=True,
    )
    return tok, model, device


def encode_history(tok, messages: List[dict], device: str) -> dict:
    inputs = tok.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def generate_stream(model, tok, inputs: dict, args: argparse.Namespace) -> str:
    """流式生成并打印（自动隐藏思维链段），返回 assistant 全文。"""
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
    kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        streamer=streamer,
        pad_token_id=tok.pad_token_id
        if tok.pad_token_id is not None
        else tok.eos_token_id,
    )
    if args.sample:
        kwargs.update(temperature=args.temperature, top_p=args.top_p)

    worker = threading.Thread(target=model.generate, kwargs=kwargs, daemon=True)
    worker.start()

    buf = ""          # 原始累计 (含特殊 token)
    shown = 0         # 已打印的前缀长度 (已过滤)
    hiding = not args.thinking  # 默认把 think 段隐藏
    t0 = time.time()
    for piece in streamer:
        buf += piece
        keep_end = len(buf)
        stop = buf.find(IM_END)
        if stop != -1:
            keep_end = min(keep_end, stop)
        if hiding:
            o = buf.find(OPEN_THINK)
            c = buf.find(CLOSE_THINK)
            if o != -1 and (c == -1 or c > keep_end):
                keep_end = min(keep_end, o)
            if o != -1 and c != -1:
                hiding = False  # think 段结束, 后续恢复显示
        visible = buf[:keep_end]
        if len(visible) > shown:
            print(visible[shown:], end="", flush=True)
            shown = len(visible)
        if stop != -1:
            break  # 命中 eos, 提前终止打印
    print(f"\n[{time.time() - t0:.1f}s | greedy={not args.sample}]")
    return buf


def report_compression(comp) -> None:
    if comp is not None and comp.stats["events"]:
        k, t = comp.stats["kept"], comp.stats["total"]
        print(f"[压缩 kept/total = {k}/{t} ({k / t:.0%})]")


def main() -> None:
    args = parse_args()
    tok, model, device = load(args)

    comp = None
    if args.compress:
        from llm_compress import TokenSaliencyCompressor, install_compression
        comp = TokenSaliencyCompressor(
            mode=args.compress_mode,
            distance=args.compress_distance,
            keep_frac=args.keep_frac,
            keep_last=args.keep_last,
            min_len=args.min_len,
        )
        install_compression(model, comp, after_layer=args.compress_after)
        print(
            f"[压缩启用: layer{args.compress_after} 后 | mode={args.compress_mode}"
            f" | keep_frac={args.keep_frac} | distance={args.compress_distance}"
            f" | keep_last={args.keep_last} | min_len={args.min_len}]"
        )

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        generate_stream(model, tok, encode_history(tok, messages, device), args)
        report_compression(comp)
        return

    messages: List[dict] = []
    print("交互模式: 直接输入; exit/quit 退出, clear 清空历史, thinking on/off")
    while True:
        try:
            text = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        low = text.lower()
        if low in ("exit", "quit"):
            break
        if low == "clear":
            messages.clear()
            print("[历史已清空]")
            continue
        if low in ("thinking on", "thinking off"):
            args.thinking = low.endswith("on")
            print(f"[thinking = {args.thinking}]")
            continue
        messages.append({"role": "user", "content": text})
        print("Qwen > ", end="", flush=True)
        try:
            reply = generate_stream(
                model, tok, encode_history(tok, messages, device), args
            )
        except KeyboardInterrupt:
            print("\n[本轮已中断]")
            messages.pop()
            continue
        idx = reply.find(IM_END)
        if idx != -1:
            reply = reply[:idx]
        messages.append({"role": "assistant", "content": reply})
        report_compression(comp)


if __name__ == "__main__":
    main()
