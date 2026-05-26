from fastapi import APIRouter, HTTPException
from typing import List

from app.models.schemas import (
    TimeSeriesInput,
    DetectionResponse,
    BatchInput,
    BatchResponse,
)
from app.services.detector import detect_anomalies
from app.services.cache import cache_service

router = APIRouter()


async def _run_detection(payload: TimeSeriesInput) -> DetectionResponse:
    """Core detection logic with Redis cache check."""
    # 1. Cache lookup
    cached = await cache_service.get(
        payload.stream_id, payload.data, payload.contamination
    )
    if cached:
        response = DetectionResponse(**cached)
        response.cached = True
        return response

    # 2. Run model
    details, overall_severity = detect_anomalies(payload.data, payload.contamination)

    anomaly_count = sum(1 for d in details if d.is_anomaly)
    response = DetectionResponse(
        stream_id=payload.stream_id,
        total_points=len(payload.data),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / len(payload.data), 4),
        overall_severity=overall_severity,
        details=details,
        cached=False,
        model_contamination=payload.contamination,
    )

    # 3. Store in cache
    await cache_service.set(
        payload.stream_id, payload.data, payload.contamination, response.model_dump()
    )

    return response


@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect anomalies in a single time-series stream",
    response_description="Anomaly scores and severity classification for every data point.",
)
async def detect(payload: TimeSeriesInput) -> DetectionResponse:
    """
    Accepts a named time-series stream and returns per-point anomaly scores,
    severity classification (normal / low / medium / high / critical), and
    aggregate statistics.

    Results are cached in Redis for **5 minutes** — identical windows are
    served instantly without re-running the model.
    """
    try:
        return await _run_detection(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc


@router.post(
    "/detect/batch",
    response_model=BatchResponse,
    summary="Detect anomalies across multiple streams in one request",
)
async def detect_batch(payload: BatchInput) -> BatchResponse:
    """
    Process up to **20 streams** in a single call. Each stream is evaluated
    independently; results are cached individually.
    """
    results = []
    for stream in payload.streams:
        try:
            result = await _run_detection(stream)
            results.append(result)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Detection failed for stream '{stream.stream_id}': {exc}",
            ) from exc

    streams_with_anomalies = sum(1 for r in results if r.anomaly_count > 0)
    return BatchResponse(
        results=results,
        total_streams=len(results),
        streams_with_anomalies=streams_with_anomalies,
    )


@router.get(
    "/streams/{stream_id}/status",
    summary="Quick cache-hit check for a stream",
)
async def stream_status(stream_id: str):
    """
    Returns whether a cached result exists for the given stream ID.
    Useful for polling dashboards before submitting a full detection request.
    """
    redis_ok = await cache_service.ping()
    return {
        "stream_id": stream_id,
        "redis_available": redis_ok,
        "note": "Submit a /detect request to populate or refresh the cache.",
    }
