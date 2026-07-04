"""
Tests for SentinelAPI. Run: pytest -q
Requires trained models:
    python -m app.ml.generate_dataset && python -m app.ml.train
"""
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services import detector


def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture(autouse=True)
async def _isolate_cache():
    from app.services.cache import cache_service
    await cache_service.connect()
    if cache_service._client:
        await cache_service._client.flushall()
    cache_service._local.clear()
    await cache_service.reset_stats()
    yield


def test_models_load_with_100k_corpus():
    reg = detector.registry()
    assert reg["total_rows"] == 100_000
    assert set(reg["metrics"]) == {"cpu", "memory", "latency"}


def test_score_values_independent_of_window():
    solo = detector.score_values("cpu", [98.0])[0]
    in_window = detector.score_values("cpu", [34.0, 35.0, 98.0, 33.0])[2]
    assert abs(solo - in_window) < 1e-9


def test_spike_flagged_anomalous():
    scores = detector.score_values("latency", [120.0, 125.0, 2000.0])
    thr = detector.threshold_for("latency")
    assert scores[2] < thr
    assert scores[0] >= thr


@pytest.mark.asyncio
async def test_health():
    async with _client() as c:
        r = await c.get("/health")
    assert r.status_code == 200
    assert r.json()["models"] == "loaded"


@pytest.mark.asyncio
async def test_detect_flags_cpu_spike():
    payload = {"stream_id": "cpu1", "metric_type": "cpu",
               "data": [34, 35, 33, 99, 34, 36, 98, 35]}
    async with _client() as c:
        r = await c.post("/api/v1/detect", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["anomaly_count"] >= 1
    assert body["total_points"] == 8


@pytest.mark.asyncio
async def test_cache_hit_on_repeat_window():
    payload = {"stream_id": "cpu2", "metric_type": "cpu", "data": [34, 35, 33, 36, 34]}
    async with _client() as c:
        first = (await c.post("/api/v1/detect", json=payload)).json()
        second = (await c.post("/api/v1/detect", json=payload)).json()
    assert first["model_inferences"] == 5
    assert second["cache_hits"] == 5
    assert second["model_inferences"] == 0


@pytest.mark.asyncio
async def test_overlapping_windows_reuse_points():
    base = [34, 35, 33, 36, 34, 37, 32, 35]
    async with _client() as c:
        await c.post("/api/v1/detect",
                     json={"stream_id": "ov", "metric_type": "cpu", "data": base[:6]})
        second = (await c.post("/api/v1/detect",
                  json={"stream_id": "ov", "metric_type": "cpu", "data": base[2:8]})).json()
    assert second["cache_hits"] >= 4


@pytest.mark.asyncio
async def test_unknown_metric_rejected():
    async with _client() as c:
        r = await c.post("/api/v1/detect",
                         json={"stream_id": "x", "metric_type": "disk", "data": [1, 2, 3]})
    assert r.status_code == 422
