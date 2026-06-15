"""
Generate a synthetic-but-realistic 100,000-row metrics corpus used to
pre-train the SentinelAPI Isolation Forest models.

Three metric families are produced, each with its own distribution and an
injected anomaly fraction:

  - cpu      : utilisation %, diurnal baseline + noise, saturation spikes
  - memory   : utilisation %, slow upward drift + noise, leak spikes
  - latency  : response time ms, log-normal body + heavy-tail spikes

Output: app/ml/data/metrics_corpus.csv  (columns: metric_type, value, is_anomaly)
The is_anomaly column is a ground-truth label used only for offline evaluation
in train.py — it is never used by the live API.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COUNTS = {"cpu": 34_000, "memory": 33_000, "latency": 33_000}
ANOMALY_FRACTION = 0.04

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "metrics_corpus.csv")


def _cpu(n: int) -> np.ndarray:
    t = np.arange(n)
    diurnal = 35 + 18 * np.sin(2 * np.pi * t / 1440)
    noise = RNG.normal(0, 4, n)
    return np.clip(diurnal + noise, 0, 100)


def _memory(n: int) -> np.ndarray:
    drift = 55 + np.linspace(0, 12, n)
    noise = RNG.normal(0, 3, n)
    return np.clip(drift + noise, 0, 100)


def _latency(n: int) -> np.ndarray:
    body = RNG.lognormal(mean=np.log(120), sigma=0.25, size=n)
    return np.clip(body, 5, None)


GENERATORS = {"cpu": _cpu, "memory": _memory, "latency": _latency}


def _inject_anomalies(metric: str, values: np.ndarray) -> np.ndarray:
    n = len(values)
    k = int(n * ANOMALY_FRACTION)
    idx = RNG.choice(n, size=k, replace=False)
    labels = np.zeros(n, dtype=int)
    labels[idx] = 1

    if metric == "cpu":
        values[idx] = RNG.uniform(92, 100, k)
    elif metric == "memory":
        values[idx] = RNG.uniform(94, 100, k)
    else:
        values[idx] = RNG.uniform(600, 2500, k)
    return labels


def build() -> pd.DataFrame:
    frames = []
    for metric, n in COUNTS.items():
        values = GENERATORS[metric](n).astype(float)
        labels = _inject_anomalies(metric, values)
        frames.append(
            pd.DataFrame(
                {"metric_type": metric, "value": np.round(values, 3), "is_anomaly": labels}
            )
        )
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = build()
    total = len(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {total:,} rows -> {OUT_PATH}")
    for metric in COUNTS:
        sub = df[df.metric_type == metric]
        print(
            f"  {metric:<8} rows={len(sub):>6,}  "
            f"anomalies={int(sub.is_anomaly.sum()):>5,} "
            f"({sub.is_anomaly.mean() * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()
