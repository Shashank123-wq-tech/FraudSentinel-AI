"""
src/common/time_utils.py
Shared temporal constants used across Phase 2 (and reusable by later phases).
"""

WINDOW_SIZES = {"5m": "5min", "15m": "15min", "60m": "60min"}

# rolling baseline lookback, in NUMBER OF BINS at each resolution — chosen so
# each baseline covers roughly the same real-world span (~24h) regardless of bin size
BASELINE_BINS = {"5m": 288, "15m": 96, "60m": 24}

EPSILON = 1e-6
