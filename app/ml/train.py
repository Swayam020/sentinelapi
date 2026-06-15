"""
Train one Isolation Forest per metric family on the 100K-row corpus and
persist the models + metadata with joblib.

Run:
    python -m app.ml.generate_dataset
    python -m app.ml.train

Artifacts written to app/ml/models/:
    <metric>.joblib        # trained IsolationForest
    registry.joblib        # metadata: training counts, contamination, thresholds, eval

The model is trained ONCE on a large corpus and loaded at API startup.
Inference never refits — incoming windows are scored against the frozen forest.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "metrics_corpus.csv")
MODELS_DIR = os.path.join(HERE, "models")

CONTAMINATION = 0.04
N_ESTIMATORS = 200


def _evaluate(model: IsolationForest, X: np.ndarray, y_true: np.ndarray) -> dict:
    pred = model.predict(X)
    y_pred = (pred == -1).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise SystemExit("Corpus not found. Run: python -m app.ml.generate_dataset")

    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    registry: dict = {
        "contamination": CONTAMINATION,
        "n_estimators": N_ESTIMATORS,
        "total_rows": int(len(df)),
        "metrics": {},
    }

    print(f"Loaded corpus: {len(df):,} rows")
    for metric in ["cpu", "memory", "latency"]:
        sub = df[df.metric_type == metric]
        X = sub[["value"]].to_numpy()
        y = sub["is_anomaly"].to_numpy()

        t0 = time.perf_counter()
        model = IsolationForest(
            contamination=CONTAMINATION,
            n_estimators=N_ESTIMATORS,
            random_state=42,
            n_jobs=-1,
        ).fit(X)
        fit_ms = (time.perf_counter() - t0) * 1000

        metrics = _evaluate(model, X, y)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{metric}.joblib"))
        registry["metrics"][metric] = {
            "train_rows": int(len(sub)),
            "anomaly_threshold": float(model.offset_),
            "fit_ms": round(fit_ms, 1),
            **metrics,
        }
        print(
            f"  {metric:<8} trained on {len(sub):>6,} rows  "
            f"P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']}  "
            f"({fit_ms:.0f} ms)"
        )

    joblib.dump(registry, os.path.join(MODELS_DIR, "registry.joblib"))
    print(f"\nSaved {len(registry['metrics'])} models + registry -> {MODELS_DIR}")
    print(f"Total training rows: {registry['total_rows']:,}")


if __name__ == "__main__":
    main()
