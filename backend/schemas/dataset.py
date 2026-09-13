from pydantic import BaseModel
from typing import List, Optional, Dict

class FeatureMapping(BaseModel):
    source: str
    canonical: str
    method: str
    confidence: float

class SchemaFlags(BaseModel):
    timestamp: bool
    network_flow: bool
    packet_statistics: bool
    byte_statistics: bool
    duration: bool

class CompatibilityReport(BaseModel):
    status: str
    compatibility: str
    error_code: Optional[str] = None
    upload_id: Optional[str] = None
    reason: Optional[str] = None
    schema_flags: Optional[SchemaFlags] = None
    flow_count: Optional[int] = None
    usable_windows: Optional[int] = None
    usable_sequences: Optional[int] = None
    mapped_features: Optional[List[FeatureMapping]] = None
    derived_features: Optional[List[str]] = None
    missing_features: Optional[List[str]] = None
    missing_requirements: Optional[List[str]] = None
    timestamp_column: Optional[str] = None
    timestamp_detection_method: Optional[str] = None
    timestamp_parse_success_rate: Optional[float] = None
    temporal_range: Optional[str] = None
    window_seconds: int = 30
