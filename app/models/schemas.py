from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


class SeverityLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    LATENCY = "latency"


class TimeSeriesInput(BaseModel):
    stream_id: str = Field(..., examples=["api_response_time_prod"],
                           description="Unique identifier for this data stream.")
    metric_type: MetricType = Field(
        ..., description="Which pre-trained model to score against (cpu | memory | latency)."
    )
    data: List[float] = Field(
        ..., min_length=1,
        description="Ordered list of metric values for this window.",
        examples=[[34.1, 35.0, 33.8, 98.7, 34.2, 35.5, 99.1, 34.0]],
    )
    sensitivity: float = Field(
        default=1.0, ge=0.25, le=4.0,
        description="Tunes the anomaly cutoff without retraining (>1 = more sensitive).",
    )

    @field_validator("stream_id")
    @classmethod
    def _strip_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("stream_id must not be empty.")
        if len(v) > 128:
            raise ValueError("stream_id must be <= 128 characters.")
        return v


class AnomalyDetail(BaseModel):
    index: int
    value: float
    anomaly_score: float = Field(..., description="Isolation Forest score; more negative = more anomalous.")
    severity: SeverityLevel
    is_anomaly: bool
    from_cache: bool = Field(..., description="True if this point's score was served from cache.")


class DetectionResponse(BaseModel):
    stream_id: str
    metric_type: MetricType
    total_points: int
    anomaly_count: int
    anomaly_rate: float
    overall_severity: SeverityLevel
    details: List[AnomalyDetail]
    cache_hits: int = Field(..., description="Points in this request served from cache.")
    model_inferences: int = Field(..., description="Points in this request scored by the model.")


class BatchInput(BaseModel):
    streams: List[TimeSeriesInput] = Field(..., min_length=1, max_length=20)


class BatchResponse(BaseModel):
    results: List[DetectionResponse]
    total_streams: int
    streams_with_anomalies: int
