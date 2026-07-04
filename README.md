# SentinelAPI — Anomaly Detection as a Service

Scores time-series windows (**cpu**, **memory**, **latency**) against Isolation
Forest models **pre-trained on a 100,000-row corpus**, with a **per-point Redis
cache** that lets overlapping polling windows skip the model entirely.

## Architecture

The model is trained **once, offline** (`app/ml/train.py`) and loaded at
startup; inference never refits. Because scoring runs against a frozen forest,
a single point's score is independent of its window — so the cache works at
the **point** level. Consecutive polls from a monitoring client overlap
heavily, so most points are already cached and only genuinely new points reach
the model.

Flow: request window → per-point cache lookup (Redis MGET) → score only the
misses against the frozen IsolationForest → apply threshold + severity → response.

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Uvicorn |
| Model | scikit-learn `IsolationForest`, one per metric family |
| Cache | Redis (async, per-point) with graceful in-process fallback |
| Training | 100K-row synthetic corpus (`app/ml/`) persisted via `joblib` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.ml.generate_dataset      # writes 100K-row corpus
python -m app.ml.train                 # trains + persists models

docker run -d -p 6379:6379 redis:7-alpine   # optional but recommended
uvicorn app.main:app --reload
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/detect` | Score one window |
| POST | `/api/v1/detect/batch` | Up to 20 windows in one call |
| GET | `/api/v1/metrics` | Live cache hit rate + inference reduction |
| POST | `/api/v1/metrics/reset` | Reset cache counters |
| GET | `/api/v1/model/info` | Training rows + offline eval per model |
| GET | `/health` | Redis + model status |

**Request**
```json
{ "stream_id": "cpu_host_01", "metric_type": "cpu",
  "data": [34.1, 35.0, 33.8, 98.7, 34.2], "sensitivity": 1.0 }
```

`sensitivity` (0.25–4.0) tunes the anomaly cutoff without retraining. Each
point returns a score, one of five severity levels
(`normal/low/medium/high/critical`), and a `from_cache` flag.

## Model

One Isolation Forest per metric family (200 trees, contamination 0.04),
trained on a 100,000-row labelled synthetic corpus. Offline eval:

| Metric | Train rows | Precision | Recall | F1 |
|--------|-----------:|----------:|-------:|---:|
| cpu | 34,000 | 0.843 | 0.838 | 0.840 |
| memory | 33,000 | 0.851 | 0.847 | 0.849 |
| latency | 33,000 | 1.000 | 1.000 | 1.000 |

Deterministic (`random_state=42`) — regenerate with `python -m app.ml.train`.

## Benchmark

`benchmark/run.py` simulates monitoring clients polling overlapping sliding
windows (window = 60 points, advancing 5 per poll):

```bash
python -m benchmark.run
```

Representative run (4 streams x 1,500 points, ~1,150 warm requests,
in-process cache):

| Measurement | Value |
|-------------|-------|
| Warm-window latency p50 / p95 / **p99** | 11.0 / 12.0 / **13.2 ms** |
| Cache hit rate | **93.7%** |
| Model-inference-call reduction | **93.7%** |

The hit rate follows from the polling overlap (window/step) and is printed by
the script, so it can be re-measured rather than taken on faith.

## Tests

```bash
pytest -q   # 8 tests: model loading, score independence, cache hits, overlap reuse
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CACHE_TTL_SECONDS` | `300` | Per-point cache TTL |
| `CACHE_QUANT_DECIMALS` | `2` | Value rounding for cache keys |
