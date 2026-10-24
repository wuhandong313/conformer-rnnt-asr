#!/usr/bin/env python3
"""Real-data CUDA preflight with the actual shared syllable vocabulary size."""
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.server.probe_aishell1 import main

if __name__ == "__main__":
    main()
