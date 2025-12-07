#!/usr/bin/env python3
"""
Wrapper script to run molecular evolution with proper imports.
"""

import sys
import os

# Add paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.append(PROJECT_ROOT)

# Now run the evolution script
if __name__ == "__main__":
    import src.molecule_generator_crem
    src.molecule_generator_crem.main()