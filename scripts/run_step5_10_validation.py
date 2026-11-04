"""
Execution Script for Phase 5 Step 5.10: Validate Numerical Stability, Repeatability, and Edge Robustness
Mirror script in scripts/ directory.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from digital_twin.run_step5_10_validation import main

if __name__ == "__main__":
    main()
