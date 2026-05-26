"""
Redis caching layer for repeat-query optimisation.

Cache key: SHA-256 of (stream_id + sorted data values + contamination).
TTL: 5 minutes (configurable via CACHE_TTL_SECONDS env var).
"""
import hashlib
import json
import os
from typing import Optional

import redis.asyncio as aioredis


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))


class CacheService:
    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        try:
            self._client = aioredis.from_url(
                REDIS_URL, encoding="utf-8", decode_responses=True
            )
            await self._client.ping()
        except Exception:
            # Redis is optional — API degrades gracefully without it
            self._client = None

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def ping(self) -> bool:
        if not self._client:
            return False
        try:
            return await self._client.ping()
        except Exception:
            return False

    def _make_key(self, stream_id: str, data: list, contamination: float) -> str:
        raw = json.dumps(
            {"stream_id": stream_id, "data": data, "contamination": contamination},
            sort_keys=True,
        )
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return f"sentinel:cache:{digest}"

    async def get(self, stream_id: str, data: list, contamination: float) -> Optional[dict]:
        if not self._client:
            return None
        try:
            key = self._make_key(stream_id, data, contamination)
            raw = await self._client.get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return None

    async def set(
        self, stream_id: str, data: list, contamination: float, result: dict
    ) -> None:
        if not self._client:
            return
        try:
            key = self._make_key(stream_id, data, contamination)
            await self._client.setex(key, CACHE_TTL, json.dumps(result))
        except Exception:
            pass


cache_service = CacheService()
