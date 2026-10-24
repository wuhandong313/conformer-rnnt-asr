#!/usr/bin/env python3
"""Per-corpus and pooled syllable S/D/I, SyllableER, tones and rate deciles."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.evaluation.eval_aishell1 import main

if __name__ == "__main__":
    main()
