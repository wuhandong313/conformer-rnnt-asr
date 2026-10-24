#!/usr/bin/env python3
"""Read-only UTC/process/GPU/checkpoint status for combined Mandarin training."""
from pathlib import Path
import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.server.status_aishell1 import main

if __name__ == "__main__":
    main()
