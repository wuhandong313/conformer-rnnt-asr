#!/usr/bin/env python3
"""AISHELL-3 + THCHS-30 shared syllable queue. Default inspection never downloads."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.server.aishell1_lexicon_pipeline import main

if __name__ == "__main__":
    main()
