#!/bin/bash
# One-shot status viewer — designed to run over a single SSH connection
# (jump-host fail2ban punishes repeated logins).
#
#   bash scripts/status.sh                       # all sections
#   bash scripts/status.sh queue                 # only queue/results
#   bash scripts/status.sh tmux train            # tmux + training tails
set -u
SECTION="${1:-all}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

show_tmux() {
    echo "═══ tmux sessions ═══"
    tmux ls 2>/dev/null || echo "(no tmux sessions)"
    echo
}

show_queue() {
    echo "═══ results/results.csv ═══"
    if [ -f results/results.csv ]; then
        cat results/results.csv
    else
        echo "(no results yet)"
    fi
    echo
    echo "═══ queue_runner.log (tail 15) ═══"
    tail -n 15 results/queue_runner.log 2>/dev/null || echo "(no queue log)"
    echo
}

show_train() {
    echo "═══ training log tails ═══"
    for log in results/logs/*.train.log exp_baseline_none/train.log; do
        [ -f "$log" ] || continue
        echo "--- $log (tail 5) ---"
        tail -n 5 "$log"
        echo
    done
}

show_ckpt() {
    echo "═══ new checkpoints ═══"
    find exp_baseline_none exp_a4 exp_b exp_smoke -name 'val-*.ckpt' \
        -newer results/results.csv 2>/dev/null | head -20
    ls -lht exp_*/checkpoints/val-*.ckpt 2>/dev/null | head -15 || true
    echo
}

case "$SECTION" in
    all)    show_tmux; show_train; show_queue; show_ckpt ;;
    tmux)   show_tmux ;;
    queue)  show_queue ;;
    train)  show_train ;;
    ckpt)   show_ckpt ;;
    *) echo "usage: $0 [all|tmux|queue|train|ckpt]"; exit 2 ;;
esac
