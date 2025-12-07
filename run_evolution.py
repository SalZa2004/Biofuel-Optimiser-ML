#!/usr/bin/env python3
"""
Wrapper script to run molecular evolution with proper imports.
"""

import sys
import os

# Add paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cn-predictor-model"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Now run the evolution script
if __name__ == "__main__":
    import molecule_generator_crem
    molecule_generator_crem.main()