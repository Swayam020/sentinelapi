"""
Anomaly scoring against pre-trained Isolation Forest models.

Models are trained offline (app/ml/train.py) on a 100K-row corpus and loaded
once at startup. Inference NEVER refits — incoming values are scored against
the frozen forest, which makes per-point scoring deterministic and independent
of window composition (the property the per-point cache relies on).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import numpy as np
import joblib

from app.models.schemas import SeverityLevel

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
METRIC_TYPES = ("cpu", "memory", "latency")


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load() -> Dict[str, object]:
    reg_path = os.path.join(MODELS_DIR, "registry.joblib")
    if not os.path.exists(reg_path):
        raise ModelNotTrained(
            "Models not found. Run: python -m app.ml.generate_dataset "
            "&& python -m app.ml.train"
        )
    registry = joblib.load(reg_path)
    models = {m: joblib.load(os.path.join(MODELS_DIR, f"{m}.joblib")) for m in METRIC_TYPES}
    return {"registry": registry, "models": models}


def registry() -> dict:
    return _load()["registry"]


def threshold_for(metric_type: str, sensitivity: float = 1.0) -> float:
    """
    For decision_function, a point is anomalous when its score < 0 (the offset is
    already folded in). `sensitivity` shifts that cutoff without retraining:
    >1 raises it (flags more), <1 lowers it (flags fewer). Default 1.0 -> cutoff 0,
    matching the model's own predict().
    """
    _ = _load()["registry"]["metrics"][metric_type]
    return (sensitivity - 1.0) * 0.05


def score_values(metric_type: str, values: List[float]) -> np.ndarray:
    """Raw Isolation Forest decision scores; one batched call, per-point independent."""
    if metric_type not in METRIC_TYPES:
        raise ValueError(f"Unknown metric_type '{metric_type}'. Use one of {METRIC_TYPES}.")
    model = _load()["models"][metric_type]
    X = np.asarray(values, dtype=float).reshape(-1, 1)
    return model.decision_function(X)


def classify_severity(score: float, is_anomaly: bool) -> SeverityLevel:
    if not is_anomaly:
        return SeverityLevel.NORMAL
    magnitude = abs(score)
    if magnitude >= 0.08:
        return SeverityLevel.CRITICAL
    elif magnitude >= 0.05:
        return SeverityLevel.HIGH
    elif magnitude >= 0.025:
        return SeverityLevel.MEDIUM
    return SeverityLevel.LOW


def worst_severity(severities: List[SeverityLevel]) -> SeverityLevel:
    order = [
        SeverityLevel.NORMAL,
        SeverityLevel.LOW,
        SeverityLevel.MEDIUM,
        SeverityLevel.HIGH,
        SeverityLevel.CRITICAL,
    ]
    best = SeverityLevel.NORMAL
    for s in severities:
        if order.index(s) > order.index(best):
            best = s
    return best
