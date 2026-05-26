# SentinelAPI — Anomaly Detection as a Service

> Accepts time-series data streams (server metrics, API response times, transaction volumes) and returns anomaly scores with severity classification using **Isolation Forest**.

## Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| ML Model | scikit-learn `IsolationForest` |
| Caching | Redis (async, optional) |
| Language | Python 3.12 |
| Containerisation | Docker + Docker Compose |

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/you/sentinelapi
cd sentinelapi
docker compose up --build
```

API is live at **http://localhost:8000**  
Swagger docs at **http://localhost:8000/docs**

### Option B — Local dev (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Optional: start Redis
docker run -d -p 6379:6379 redis:7-alpine

uvicorn app.main:app --reload
```

---

## API Endpoints

### `POST /api/v1/detect`

Detect anomalies in a single named stream.

**Request**
```json
{
  "stream_id": "api_response_time_prod",
  "data": [120.5, 118.2, 119.8, 500.1, 121.0, 119.5, 118.9, 120.2, 600.3, 119.7],
  "contamination": 0.05
}
```

**Response**
```json
{
  "stream_id": "api_response_time_prod",
  "total_points": 10,
  "anomaly_count": 2,
  "anomaly_rate": 0.2,
  "overall_severity": "high",
  "cached": false,
  "model_contamination": 0.05,
  "details": [
    { "index": 3, "value": 500.1, "anomaly_score": -0.182, "severity": "high", "is_anomaly": true },
    ...
  ]
}
```

### `POST /api/v1/detect/batch`

Process up to 20 streams in a single call.

```json
{
  "streams": [
    { "stream_id": "cpu_host_01", "data": [...], "contamination": 0.05 },
    { "stream_id": "tx_volume",   "data": [...], "contamination": 0.10 }
  ]
}
```

### `GET /api/v1/streams/{stream_id}/status`

Check Redis cache availability for a stream.

### `GET /health`

Returns `{ "status": "healthy"|"degraded", "redis": "connected"|"unavailable" }`.

---

## Severity Classification

| Severity | Condition |
|----------|-----------|
| `normal` | Not flagged as anomaly |
| `low` | Anomaly score magnitude < 0.15 |
| `medium` | 0.15 ≤ magnitude < 0.25 |
| `high` | 0.25 ≤ magnitude < 0.35 |
| `critical` | magnitude ≥ 0.35 |

---

## Caching Strategy

Results are cached in Redis keyed by **SHA-256(stream_id + data + contamination)**.  
TTL defaults to **5 minutes** (override with `CACHE_TTL_SECONDS` env var).  
If Redis is unavailable, the API degrades gracefully — detection still works, just without caching.

---

## Running Tests

```bash
pytest tests/ -v
```

All tests use `httpx.AsyncClient` with ASGI transport — no live server needed.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CACHE_TTL_SECONDS` | `300` | Cache TTL in seconds |

---

## Project Structure

```
sentinelapi/
├── app/
│   ├── main.py               # FastAPI app + lifespan
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response models
│   ├── routes/
│   │   └── anomaly.py        # /detect, /detect/batch, /streams/:id/status
│   └── services/
│       ├── detector.py       # Isolation Forest logic + severity classifier
│       └── cache.py          # Async Redis caching layer
├── tests/
│   └── test_sentinel.py      # Unit + integration tests
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```
