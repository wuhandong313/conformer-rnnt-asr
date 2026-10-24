#!/usr/bin/env python3
"""Train a gated combined Mandarin syllable experiment in this checkout only."""
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.server.train_aishell1_one import main

if __name__ == "__main__":
    main()
