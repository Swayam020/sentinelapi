from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import anomaly
from app.services import detector
from app.services.cache import cache_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    await cache_service.connect()
    try:
        detector._load()
    except Exception as exc:
        print(f"[startup] WARNING: {exc}")
    yield
    await cache_service.disconnect()


app = FastAPI(
    title="SentinelAPI",
    description="Anomaly Detection as a Service. Scores time-series windows against Isolation Forest models pre-trained on a 100K-row corpus, with a per-point Redis cache that serves overlapping windows.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(anomaly.router, prefix="/api/v1", tags=["Anomaly Detection"])


@app.get("/", tags=["Health"])
async def root():
    return {"service": "SentinelAPI", "status": "operational", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", tags=["Health"])
async def health():
    redis_ok = await cache_service.ping()
    try:
        detector._load()
        model_ok = True
    except Exception:
        model_ok = False
    return {
        "status": "healthy" if model_ok else "degraded",
        "redis": "connected" if redis_ok else "unavailable",
        "models": "loaded" if model_ok else "missing",
    }
