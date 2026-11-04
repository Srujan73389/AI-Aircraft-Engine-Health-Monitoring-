"""
Step 5.9 Execution Script Wrapper
Entry point: scripts/run_step5_9_comparison.py

Roadmap Reference: Phase 5 - Step 5.9
"""

import os
import sys

# Ensure repository root is in python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from digital_twin.run_step5_9_comparison import run_step5_9_comparisons


if __name__ == "__main__":
    run_step5_9_comparisons()
