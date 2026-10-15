#!/usr/bin/env python3
"""attn-predictor 训练监控 — tmux 会话存活 + 训练日志进度 + 最新 ckpt。

attn 分支的 eval.py 只支持全量评估, 不适合高频循环, 故本脚本只做
轻量状态行; 全量 eval.py 由会话在里程碑手动触发。

输出一行: TRAIN <ALIVE|DEAD> epoch=… step=… loss~… | CKPT <name> <age_s>
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import time

REPO = "/test21/conformer-rnnt-asr"
# 分支 configs/server.yaml 的 exp_dir 指向主仓库 exp/, ckpt 落在那里
CKPT_DIR = "/test21/conformer-rnnt-asr/exp/checkpoints"
TRAIN_LOG = os.path.join(REPO, "artifacts/logs/attn_train.log")
SESSION = "attn-train"


def train_status() -> tuple[str, str, str]:
    r = subprocess.run(["tmux", "ls"], capture_output=True, text=True)
    alive = SESSION in (r.stdout + r.stderr)
    step = "?"
    loss = "?"
    epoch = "?"
    try:
        with open(TRAIN_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200000))
            tail = f.read().decode(errors="replace")
        m = re.findall(r"Epoch (\d+):\s+\d+%[^ ]*\s+(\d+)/(\d+)", tail)
        if m:
            epoch, step = m[-1][0], m[-1][1]
        m2 = re.findall(r"'?Losses/train_loss'?[^0-9]*([0-9.]+)", tail)
        if m2:
            loss = m2[-1]
    except FileNotFoundError:
        pass
    return ("ALIVE" if alive else "DEAD"), epoch, f"step={step} loss~{loss}"


def main() -> int:
    status, epoch, step = train_status()
    files = glob.glob(os.path.join(CKPT_DIR, "*.ckpt"))
    if files:
        newest = max(files, key=os.path.getmtime)
        age = int(time.time() - os.path.getmtime(newest))
        ck = f"CKPT {os.path.basename(newest)} age={age}s"
    else:
        ck = "no ckpt yet"
    print(f"TRAIN {status} epoch={epoch} {step} | {ck}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
