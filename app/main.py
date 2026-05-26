from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes import anomaly
from app.services.cache import cache_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache_service.connect()
    yield
    await cache_service.disconnect()


app = FastAPI(
    title="SentinelAPI",
    description=(
        "Anomaly Detection as a Service — accepts time-series data streams "
        "(server metrics, API response times, transaction volumes) and returns "
        "anomaly scores with severity classification using Isolation Forest."
    ),
    version="1.0.0",
    contact={"name": "SentinelAPI", "url": "https://github.com/sentinelapi"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(anomaly.router, prefix="/api/v1", tags=["Anomaly Detection"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "SentinelAPI",
        "status": "operational",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    redis_ok = await cache_service.ping()
    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "unavailable",
    }
