"""The real router -- imported, not reimplemented.

The plan named `video_router_final.py`; the working implementation is
`router_pipeline.py` at the repo root. `VideoRouter` there is the real thing:
boot window, 45-of-60 rolling majority vote, window reset on every switch.
Importing keeps one source of truth instead of a copy that drifts.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # before router_pipeline pulls in pyplot

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from router_pipeline import (  # noqa: E402,F401
    PIPE_ESC, PIPE_RGB, VideoRouter, degradation, full_metrics,
)

# Calibrated on IDD-AW. Never relabel a demo threshold as this number.
CALIBRATED_THRESHOLD = 0.80
