"""
Reproducible benchmark for SentinelAPI.

Simulates high-frequency monitoring clients polling OVERLAPPING sliding
windows (each poll re-sends the last W points, advanced by S). Measures:

  - server-side processing latency (p50 / p95 / p99) for warm windows
  - cache hit rate
  - model-inference-call reduction

Run:
    python -m benchmark.run
"""
from __future__ import annotations

import asyncio
import statistics
import time

import numpy as np
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services.cache import cache_service

RNG = np.random.default_rng(7)

STREAMS = [
    ("cpu_host_01", "cpu"),
    ("mem_host_01", "memory"),
    ("lat_api_prod", "latency"),
    ("cpu_host_02", "cpu"),
]
N_POINTS = 1500
WINDOW = 60
STEP = 5


def _make_stream(kind: str, n: int) -> list[float]:
    t = np.arange(n)
    if kind == "cpu":
        base = 35 + 18 * np.sin(2 * np.pi * t / 1440) + RNG.normal(0, 4, n)
        vals = np.clip(base, 0, 100)
    elif kind == "memory":
        vals = np.clip(60 + RNG.normal(0, 3, n), 0, 100)
    else:
        vals = np.clip(RNG.lognormal(np.log(120), 0.25, n), 5, None)
    idx = RNG.choice(n, size=int(n * 0.04), replace=False)
    vals[idx] = vals[idx] * RNG.uniform(3, 8, len(idx)) if kind == "latency" else 98.0
    return [round(float(v), 2) for v in vals]


async def main() -> None:
    await cache_service.connect()
    await cache_service.reset_stats()

    transport = ASGITransport(app=app)
    cold_lat, warm_lat = [], []

    async with AsyncClient(transport=transport, base_url="http://bench") as client:
        # Warm up model + sklearn so timings reflect request work, not cold import.
        for kind in ("cpu", "memory", "latency"):
            await client.post("/api/v1/detect",
                              json={"stream_id": "warmup", "metric_type": kind, "data": [1.0]})
        await client.post("/api/v1/metrics/reset")

        for stream_id, kind in STREAMS:
            series = _make_stream(kind, N_POINTS)
            first = True
            for start in range(0, N_POINTS - WINDOW, STEP):
                window = series[start:start + WINDOW]
                payload = {"stream_id": stream_id, "metric_type": kind, "data": window}
                t0 = time.perf_counter()
                r = await client.post("/api/v1/detect", json=payload)
                dt = (time.perf_counter() - t0) * 1000
                assert r.status_code == 200, r.text
                (cold_lat if first else warm_lat).append(dt)
                first = False

        stats = (await client.get("/api/v1/metrics")).json()

    def pct(xs, p):
        return round(statistics.quantiles(xs, n=100)[p - 1], 2)

    print("=" * 60)
    print("SentinelAPI benchmark")
    print("=" * 60)
    print(f"streams={len(STREAMS)}  stream_len={N_POINTS}  window={WINDOW}  step={STEP}")
    print(f"warm requests measured: {len(warm_lat):,}")
    print("-" * 60)
    print("Server-side latency, warm (overlapping) windows:")
    print(f"  p50 = {pct(warm_lat,50):.2f} ms")
    print(f"  p95 = {pct(warm_lat,95):.2f} ms")
    print(f"  p99 = {pct(warm_lat,99):.2f} ms")
    print(f"  cold-window p99 (first poll, all misses) = {pct(cold_lat,99):.2f} ms")
    print("-" * 60)
    print(f"points requested      : {stats['points_requested']:,}")
    print(f"cache hits            : {stats['cache_hits']:,}")
    print(f"cache hit rate        : {stats['cache_hit_rate']*100:.1f}%")
    print(f"model inference calls : {stats['model_inference_calls']:,}")
    print(f"inference reduction   : {stats['inference_reduction']*100:.1f}%")
    print(f"cache backend         : {stats['backend']}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
