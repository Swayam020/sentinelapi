"""
Core anomaly detection logic using scikit-learn's Isolation Forest.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Tuple

from app.models.schemas import AnomalyDetail, SeverityLevel


def _classify_severity(score: float, is_anomaly: bool) -> SeverityLevel:
    """
    Map an Isolation Forest decision score to a human-readable severity.

    Isolation Forest scores are centred around 0:
      - Positive scores  → normal (closer to +0.5 = very normal)
      - Negative scores  → anomalous (closer to -0.5 = very anomalous)
    """
    if not is_anomaly:
        return SeverityLevel.NORMAL

    # score is negative for anomalies; we work with the absolute magnitude
    magnitude = abs(score)

    if magnitude >= 0.35:
        return SeverityLevel.CRITICAL
    elif magnitude >= 0.25:
        return SeverityLevel.HIGH
    elif magnitude >= 0.15:
        return SeverityLevel.MEDIUM
    else:
        return SeverityLevel.LOW


def _worst_severity(severities: List[SeverityLevel]) -> SeverityLevel:
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


def detect_anomalies(
    values: List[float], contamination: float = 0.05
) -> Tuple[List[AnomalyDetail], SeverityLevel]:
    """
    Run Isolation Forest on a list of numeric values.

    Returns a list of AnomalyDetail objects and the overall (worst) severity.
    """
    X = np.array(values).reshape(-1, 1)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    # decision_function: positive = normal, negative = anomaly
    scores: np.ndarray = model.decision_function(X)
    predictions: np.ndarray = model.predict(X)  # 1 = normal, -1 = anomaly

    details: List[AnomalyDetail] = []
    for i, (val, score, pred) in enumerate(zip(values, scores, predictions)):
        is_anomaly = pred == -1
        severity = _classify_severity(float(score), is_anomaly)
        details.append(
            AnomalyDetail(
                index=i,
                value=float(val),
                anomaly_score=round(float(score), 6),
                severity=severity,
                is_anomaly=is_anomaly,
            )
        )

    overall = _worst_severity([d.severity for d in details])
    return details, overall
