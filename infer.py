"""Qwen3.5-0.8B 本地推理入口（纯文本路径，不喂图/视频）。

权重与配套文件在 ./qwen3.5-0.8b/，checkpoint 为 ConditionalGeneration
全量参数（language_model + visual + mtp），本脚本只用文本生成。

用法:
    python infer.py                          # 默认 prompt, 贪心
    python infer.py -p "解释一下音素" --max-new-tokens 256
    python infer.py --sample --thinking      # 采样 + 思维链
    python infer.py --device cpu             # 强制 CPU
"""

import argparse
import time

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen3.5-0.8b", help="本地模型目录")
    ap.add_argument("-p", "--prompt", default="用三句话介绍一下你自己。")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--sample", action="store_true", help="nucleus 采样替代贪心")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--thinking", action="store_true",
                    help="开启思维链（模板 enable_thinking）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to(device).eval()

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tok.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    ).to(device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    if args.sample:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p)

    with torch.no_grad():
        t0 = time.time()
        out = model.generate(**inputs, **gen_kwargs)
    dt = time.time() - t0

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    print(tok.decode(new_ids, skip_special_tokens=True))
    print(
        f"\n[{device} | greedy] {new_ids.numel()} tokens / {dt:.1f}s "
        f"= {new_ids.numel() / dt:.1f} tok/s"
    )


if __name__ == "__main__":
    main()
