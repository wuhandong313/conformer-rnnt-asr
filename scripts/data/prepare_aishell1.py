#!/usr/bin/env python3
"""Operate on already present data only. No subcommand downloads anything."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from conformer_rnnt.aishell.lexicon import Lexicon, PinnedG2PM
from conformer_rnnt.aishell.preparation import audit_audio, freeze_bundle, prepare_labels
from conformer_rnnt.aishell.protocol import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    lex = sub.add_parser("lexicon")
    lex.add_argument("--lexicon", required=True)
    lex.add_argument("--out", required=True)
    audio = sub.add_parser("audit-audio")
    audio.add_argument("--data-root", required=True)
    audio.add_argument("--out", required=True)
    labels = sub.add_parser("labels")
    labels.add_argument("--audio-audit", required=True)
    labels.add_argument("--lexicon", required=True)
    labels.add_argument("--resolutions", help="JSON mapping issue IDs to existing-candidate decisions")
    labels.add_argument("--out", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--data-root", required=True)
    freeze.add_argument("--labels", required=True)
    freeze.add_argument("--out", required=True)
    batches = sub.add_parser("batches")
    batches.add_argument("--bundle", required=True)
    batches.add_argument("--budget", required=True, type=int)
    batches.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.stage == "lexicon":
        result = Lexicon(args.lexicon).audit
        write_json(args.out, result)
    elif args.stage == "audit-audio":
        result = audit_audio(args.data_root, args.out)
    elif args.stage == "labels":
        resolutions = json.loads(Path(args.resolutions).read_text()) if args.resolutions else {}
        result = prepare_labels(args.audio_audit, args.lexicon, PinnedG2PM(), args.out, resolutions=resolutions)
    elif args.stage == "freeze":
        result = freeze_bundle(args.data_root, args.labels, args.out)
    else:
        from conformer_rnnt.aishell.manifest import create_batch_plan
        result = create_batch_plan(args.bundle, args.budget)
        write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"symbols", "groups", "batches", "files"}}, ensure_ascii=False))
    if result.get("valid") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
