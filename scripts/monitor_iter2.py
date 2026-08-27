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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = "/test21/conformer-rnnt-asr"
PY = "/root/anaconda3/envs/whd/bin/python"
CONFIG = os.path.join(REPO, "configs/server_chain.yaml")
CKPT_DIR = os.path.join(REPO, "exp_iter2/checkpoints")
EVAL_DIR = os.path.join(REPO, "exp_iter2/monitor_eval")
TRAIN_LOG = os.path.join(REPO, "exp_iter7_train.log")


def latest_ckpt() -> str | None:
    files = glob.glob(os.path.join(CKPT_DIR, "val-*.ckpt"))
    return max(files, key=os.path.getmtime) if files else None


def eval_ckpt(ckpt: str) -> dict | None:
    os.makedirs(EVAL_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(ckpt))[0]
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
    """Return (alive?, last-step) from tmux + log tail."""
    r = subprocess.run(
        ["tmux", "has-session", "-t", "iter2-train"],
        capture_output=True,
    )
    alive = r.returncode == 0
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
