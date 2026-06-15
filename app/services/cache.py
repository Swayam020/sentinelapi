"""
Per-point score cache.

Because scoring now runs against a frozen pre-trained model, the score of a
single value is independent of the rest of its window. That lets us cache at
the POINT level keyed by (metric_type, quantized_value) instead of hashing the
whole window. Consecutive sliding windows from a polling client overlap
heavily, so most points are already cached — only genuinely new points hit the
model. This is what makes "overlapping time windows" benefit from the cache.

Tracks hits / misses / model-inference calls so the real hit rate and inference
reduction are observable via GET /api/v1/metrics. Degrades to an in-process
dict if Redis is unavailable, so the cache logic still works without Redis.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))
QUANT_DECIMALS = int(os.getenv("CACHE_QUANT_DECIMALS", "2"))

H_KEY, M_KEY, INF_KEY = "sentinel:stat:hits", "sentinel:stat:misses", "sentinel:stat:inferences"


def quantize(value: float) -> str:
    return f"{round(float(value), QUANT_DECIMALS):.{QUANT_DECIMALS}f}"


class CacheService:
    def __init__(self) -> None:
        self._client: Optional[aioredis.Redis] = None
        self._local: Dict[str, str] = {}
        self._local_stats = {"hits": 0, "misses": 0, "inferences": 0}

    async def connect(self) -> None:
        try:
            self._client = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
            await self._client.ping()
        except Exception:
            self._client = None

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def ping(self) -> bool:
        if not self._client:
            return False
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    @staticmethod
    def _key(metric_type: str, value: float) -> str:
        return f"sentinel:score:{metric_type}:{quantize(value)}"

    async def get_scores(self, metric_type: str, values: List[float]) -> List[Optional[float]]:
        keys = [self._key(metric_type, v) for v in values]
        if self._client:
            raw = await self._client.mget(keys)
        else:
            raw = [self._local.get(k) for k in keys]
        return [float(r) if r is not None else None for r in raw]

    async def set_scores(self, metric_type: str, pairs: Dict[float, float]) -> None:
        if not pairs:
            return
        if self._client:
            pipe = self._client.pipeline()
            for value, score in pairs.items():
                pipe.set(self._key(metric_type, value), str(score), ex=CACHE_TTL)
            await pipe.execute()
        else:
            for value, score in pairs.items():
                self._local[self._key(metric_type, value)] = str(score)

    async def record(self, hits: int, misses: int, inferences: int) -> None:
        if self._client:
            pipe = self._client.pipeline()
            pipe.incrby(H_KEY, hits)
            pipe.incrby(M_KEY, misses)
            pipe.incrby(INF_KEY, inferences)
            await pipe.execute()
        else:
            self._local_stats["hits"] += hits
            self._local_stats["misses"] += misses
            self._local_stats["inferences"] += inferences

    async def stats(self) -> dict:
        if self._client:
            h, m, inf = await self._client.mget(H_KEY, M_KEY, INF_KEY)
            hits, misses, inferences = int(h or 0), int(m or 0), int(inf or 0)
        else:
            hits = self._local_stats["hits"]
            misses = self._local_stats["misses"]
            inferences = self._local_stats["inferences"]
        requested = hits + misses
        hit_rate = hits / requested if requested else 0.0
        reduction = 1 - (inferences / requested) if requested else 0.0
        return {
            "points_requested": requested,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": round(hit_rate, 4),
            "model_inference_calls": inferences,
            "inference_reduction": round(reduction, 4),
            "backend": "redis" if self._client else "in-process",
        }

    async def reset_stats(self) -> None:
        if self._client:
            await self._client.delete(H_KEY, M_KEY, INF_KEY)
        else:
            self._local_stats = {"hits": 0, "misses": 0, "inferences": 0}


cache_service = CacheService()
