from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


class SeverityLevel(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataPoint(BaseModel):
    timestamp: Optional[float] = Field(
        default=None,
        description="Unix timestamp (optional). Auto-assigned if omitted.",
    )
    value: float = Field(..., description="Numeric metric value.")


class TimeSeriesInput(BaseModel):
    stream_id: str = Field(
        ...,
        description="Unique identifier for this data stream (e.g. 'cpu_host_01').",
        examples=["api_response_time_prod"],
    )
    data: List[float] = Field(
        ...,
        min_length=10,
        description="Ordered list of numeric values (min 10 points).",
        examples=[[120.5, 118.2, 119.8, 500.1, 121.0, 119.5, 118.9, 120.2, 600.3, 119.7]],
    )
    contamination: float = Field(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Expected proportion of anomalies (0.01–0.50). Default: 0.05.",
    )

    @field_validator("stream_id")
    @classmethod
    def validate_stream_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("stream_id must not be empty.")
        if len(v) > 128:
            raise ValueError("stream_id must be ≤ 128 characters.")
        return v


class AnomalyDetail(BaseModel):
    index: int = Field(..., description="Position in the input array.")
    value: float = Field(..., description="Original metric value.")
    anomaly_score: float = Field(
        ..., description="Isolation Forest score. More negative = more anomalous."
    )
    severity: SeverityLevel = Field(..., description="Classified severity level.")
    is_anomaly: bool = Field(..., description="True if classified as an anomaly.")


class DetectionResponse(BaseModel):
    stream_id: str
    total_points: int
    anomaly_count: int
    anomaly_rate: float = Field(..., description="Fraction of anomalous points.")
    overall_severity: SeverityLevel = Field(
        ..., description="Worst severity seen in this window."
    )
    details: List[AnomalyDetail]
    cached: bool = Field(default=False, description="True if result was served from cache.")
    model_contamination: float


class BatchInput(BaseModel):
    streams: List[TimeSeriesInput] = Field(
        ..., min_length=1, max_length=20, description="Up to 20 streams per batch."
    )


class BatchResponse(BaseModel):
    results: List[DetectionResponse]
    total_streams: int
    streams_with_anomalies: int
