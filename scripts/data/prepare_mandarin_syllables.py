#!/usr/bin/env python3
"""Prepare already-present AISHELL-3 and THCHS-30 directly annotated syllables."""
import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--aishell3-root", required=True)
    audit.add_argument("--thchs30-root", required=True)
    audit.add_argument("--out", required=True)
    audit.add_argument("--inventory", help="Optional independently reviewed original syllable inventory")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--data-root", required=True)
    freeze.add_argument("--draft", required=True)
    freeze.add_argument("--out", required=True)
    batches = sub.add_parser("batches")
    batches.add_argument("--bundle", required=True)
    batches.add_argument("--budget", required=True, type=int)
    batches.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.stage == "audit":
        from conformer_rnnt.mandarin.preparation import prepare_mix
        result = prepare_mix(args.data_root, args.aishell3_root, args.thchs30_root, args.out, inventory=args.inventory)
    elif args.stage == "freeze":
        from conformer_rnnt.mandarin.preparation import freeze_mix
        result = freeze_mix(args.data_root, args.draft, args.out)
    else:
        from conformer_rnnt.aishell.manifest import create_batch_plan
        from conformer_rnnt.aishell.protocol import write_json
        result = create_batch_plan(args.bundle, args.budget)
        write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k in {"valid", "counts", "counts_by_corpus", "vocabulary_size", "status"}}, ensure_ascii=False))
    if result.get("valid") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
