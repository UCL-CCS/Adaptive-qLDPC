"""Output paths for Adaptive qLDPC experiments.

Run commands from the repository root so relative `outputs/` paths resolve.
"""

from __future__ import annotations

import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
