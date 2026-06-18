from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    TimeSeriesInput, DetectionResponse, AnomalyDetail,
    BatchInput, BatchResponse,
)
from app.services import detector
from app.services.cache import cache_service

router = APIRouter()


async def _run_detection(payload: TimeSeriesInput) -> DetectionResponse:
    metric = payload.metric_type.value
    values = payload.data
    cached_scores = await cache_service.get_scores(metric, values)
    miss_idx = [i for i, s in enumerate(cached_scores) if s is None]
    inferences = len(miss_idx)
    if miss_idx:
        miss_values = [values[i] for i in miss_idx]
        fresh = detector.score_values(metric, miss_values)
        to_store = {}
        for j, i in enumerate(miss_idx):
            cached_scores[i] = float(fresh[j])
            to_store[values[i]] = float(fresh[j])
        await cache_service.set_scores(metric, to_store)
    hits = len(values) - inferences
    await cache_service.record(hits=hits, misses=inferences, inferences=inferences)
    threshold = detector.threshold_for(metric, payload.sensitivity)
    miss_set = set(miss_idx)
    details, severities = [], []
    for i, (val, score) in enumerate(zip(values, cached_scores)):
        is_anom = score < threshold
        sev = detector.classify_severity(score, is_anom)
        severities.append(sev)
        details.append(AnomalyDetail(
            index=i, value=float(val), anomaly_score=round(float(score), 6),
            severity=sev, is_anomaly=is_anom, from_cache=(i not in miss_set),
        ))
    anomaly_count = sum(1 for d in details if d.is_anomaly)
    return DetectionResponse(
        stream_id=payload.stream_id,
        metric_type=payload.metric_type,
        total_points=len(values),
        anomaly_count=anomaly_count,
        anomaly_rate=round(anomaly_count / len(values), 4),
        overall_severity=detector.worst_severity(severities),
        details=details,
        cache_hits=hits,
        model_inferences=inferences,
    )


@router.post("/detect", response_model=DetectionResponse)
async def detect(payload: TimeSeriesInput) -> DetectionResponse:
    try:
        return await _run_detection(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc


@router.post("/detect/batch", response_model=BatchResponse)
async def detect_batch(payload: BatchInput) -> BatchResponse:
    results = []
    for stream in payload.streams:
        try:
            results.append(await _run_detection(stream))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Detection failed for '{stream.stream_id}': {exc}") from exc
    return BatchResponse(
        results=results,
        total_streams=len(results),
        streams_with_anomalies=sum(1 for r in results if r.anomaly_count > 0),
    )


@router.get("/metrics")
async def metrics():
    return await cache_service.stats()


@router.post("/metrics/reset")
async def reset_metrics():
    await cache_service.reset_stats()
    return {"status": "reset"}


@router.get("/model/info")
async def model_info():
    try:
        return detector.registry()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
