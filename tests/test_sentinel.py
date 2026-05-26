"""
Tests for SentinelAPI — anomaly detection routes and core detector logic.
Run with: pytest tests/ -v
"""
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services.detector import detect_anomalies
from app.models.schemas import SeverityLevel


# ---------------------------------------------------------------------------
# Detector unit tests
# ---------------------------------------------------------------------------

def make_normal_series(n: int = 50, base: float = 100.0, noise: float = 2.0):
    import random
    random.seed(42)
    return [base + random.gauss(0, noise) for _ in range(n)]


def inject_spikes(series, positions, spike_value=500.0):
    s = list(series)
    for p in positions:
        s[p] = spike_value
    return s


def test_detect_no_anomalies_returns_mostly_normal():
    data = make_normal_series(50)
    details, overall = detect_anomalies(data, contamination=0.05)
    assert len(details) == 50
    normals = [d for d in details if not d.is_anomaly]
    assert len(normals) >= 40  # at most 5 % flagged


def test_detect_spikes_flagged_as_anomalies():
    data = make_normal_series(50)
    data = inject_spikes(data, positions=[10, 30], spike_value=1000.0)
    details, overall = detect_anomalies(data, contamination=0.05)
    anomalies = [d for d in details if d.is_anomaly]
    anomaly_indices = {d.index for d in anomalies}
    # The two extreme spikes must be detected
    assert 10 in anomaly_indices or 30 in anomaly_indices


def test_overall_severity_escalates_with_spike():
    data = make_normal_series(50)
    data = inject_spikes(data, positions=[5], spike_value=9999.0)
    _, overall = detect_anomalies(data, contamination=0.05)
    assert overall in (
        SeverityLevel.MEDIUM,
        SeverityLevel.HIGH,
        SeverityLevel.CRITICAL,
    )


def test_anomaly_score_negative_for_anomalies():
    data = make_normal_series(50)
    data = inject_spikes(data, positions=[25], spike_value=5000.0)
    details, _ = detect_anomalies(data, contamination=0.05)
    for d in details:
        if d.is_anomaly:
            assert d.anomaly_score < 0


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_root_returns_200():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "SentinelAPI"


@pytest.mark.asyncio
async def test_detect_endpoint_normal_series():
    payload = {
        "stream_id": "test_cpu",
        "data": list(make_normal_series(20)),
        "contamination": 0.05,
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post("/api/v1/detect", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["stream_id"] == "test_cpu"
    assert body["total_points"] == 20
    assert "details" in body
    assert len(body["details"]) == 20


@pytest.mark.asyncio
async def test_detect_endpoint_flags_spike():
    data = make_normal_series(20)
    data[10] = 9999.0
    payload = {"stream_id": "spike_stream", "data": data, "contamination": 0.1}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post("/api/v1/detect", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["anomaly_count"] > 0


@pytest.mark.asyncio
async def test_detect_rejects_too_short_series():
    payload = {"stream_id": "short", "data": [1.0, 2.0], "contamination": 0.05}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post("/api/v1/detect", json=payload)
    assert r.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_batch_detect():
    data = list(make_normal_series(15))
    payload = {
        "streams": [
            {"stream_id": "s1", "data": data, "contamination": 0.05},
            {"stream_id": "s2", "data": data, "contamination": 0.05},
        ]
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post("/api/v1/detect/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["total_streams"] == 2
    assert len(body["results"]) == 2
