#!/usr/bin/env python3
"""iter-2 训练监控 — 轮询 val checkpoint, 评估接受率, 输出一行状态。

由本地 CronCreate 定时通过 SSH 调用。输出格式 (一行):
  TRAIN step=… loss=… | CKPT <name> | h1=… h2=… h3=… | PER=… | lossless=…

也负责在训练进程死亡时报告 (不自动重启, 由本地会话决策)。
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = "/test21/conformer-rnnt-asr"
PY = "/root/anaconda3/envs/whd/bin/python"
CONFIG = os.path.join(REPO, "configs/server_chain.yaml")
CKPT_DIR = os.path.join(REPO, "exp_iter2/checkpoints")
EVAL_DIR = os.path.join(REPO, "exp_iter2/monitor_eval")
TRAIN_LOG = os.path.join(REPO, "exp_iter8h2_train.log")


def latest_ckpt() -> str | None:
    files = glob.glob(os.path.join(CKPT_DIR, "val-*.ckpt"))
    if not files:
        return None
    newest = max(files, key=os.path.getmtime)
    # Lightning 可能仍在写文件 — 太新的 ckpt 跳过 (防止读到半截权重,
    # 那会产生 PER 虚高被缓存的坏结果)。
    if time.time() - os.path.getmtime(newest) < 180:
        older = [f for f in files if f != newest]
        if not older:
            return None
        newest = max(older, key=os.path.getmtime)
    return newest


def eval_ckpt(ckpt: str) -> dict | None:
    os.makedirs(EVAL_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(ckpt))[0]
    # 缓存键含 ckpt 文件大小 — 不同 run 的同名 ckpt (epoch/step 相同)
    # 不会互相命中陈旧结果。
    base = f"{base}-{os.path.getsize(ckpt)}"
    out_json = os.path.join(EVAL_DIR, base + ".json")
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                return json.load(f)
        except Exception:
            pass
    cmd = [
        PY, os.path.join(REPO, "eval.py"), CONFIG,
        "--checkpoint", ckpt, "--max-samples", "100",
        "--output-json", out_json,
        "--per-sample-file", os.path.join(EVAL_DIR, base + ".csv"),
    ]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not os.path.exists(out_json):
        return None
    with open(out_json) as f:
        return json.load(f)


def train_status() -> tuple[str, str]:
    """Return (alive?, last-step) from screen session + log tail."""
    r = subprocess.run(
        ["screen", "-ls"], capture_output=True, text=True,
    )
    alive = "iter2-train" in (r.stdout + r.stderr)
    step = "?"
    loss = "?"
    try:
        with open(TRAIN_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200000))
            tail = f.read().decode(errors="replace")
        m = re.findall(r"Epoch (\d+):\s+\d+%[^ ]*\s+(\d+)/(\d+)", tail)
        if m:
            step = m[-1][1]
        m2 = re.findall(r"'?Losses/train_loss'?[^0-9]*([0-9.]+)", tail)
        if m2:
            loss = m2[-1]
    except FileNotFoundError:
        pass
    return ("ALIVE" if alive else "DEAD", f"{step} loss~{loss}")


def main() -> int:
    status, step = train_status()
    ckpt = latest_ckpt()
    if ckpt is None:
        print(f"TRAIN {status} {step} | no val ckpt yet")
        return 0
    metrics = eval_ckpt(ckpt)
    if metrics is None:
        print(f"TRAIN {status} {step} | CKPT {os.path.basename(ckpt)} | eval failed")
        return 0
    heads = metrics.get("medusa_per_head", [])
    h_txt = " ".join(
        f"h{i+1}={ph['rate']:.3f}" for i, ph in enumerate(heads)
    )
    per = metrics.get("per", float("nan"))
    # warm-start 重启后的前 ~400 步会有 PER 瞬态 (BN running stats 重整),
    # 会自愈 — 显著标注但不当作训练失败。
    per_flag = " <<PER_TRANSIENT?" if per > 0.2 else ""
    print(
        f"TRAIN {status} {step} | CKPT {os.path.basename(ckpt)} | "
        f"{h_txt} | overall={metrics.get('medusa_accept_rate', 0):.3f} | "
        f"PER={per:.4f}{per_flag} | "
        f"lossless_fail={metrics.get('lossless_failures', 0)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
