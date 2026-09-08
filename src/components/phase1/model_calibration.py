"""Phase-1 probability calibration."""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover
    FrozenEstimator = None


def calibrate_frozen(model, X_calib, y_calib, method="isotonic"):
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    else:
        # Compatibility fallback for older sklearn. Prefer the current FrozenEstimator path.
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated
