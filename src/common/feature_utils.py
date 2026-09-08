"""
src/common/feature_utils.py
Shared numeric utilities used across detector modules.
"""

from src.common.time_utils import EPSILON


def normalize_evidence(series, clip_at=10):
    """Min-max-ish normalization via clipped values -> [0, 1], robust to outliers."""
    clipped = series.clip(lower=-clip_at, upper=clip_at)
    return (clipped - clipped.min()) / (clipped.max() - clipped.min() + EPSILON)
