from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class TopPrediction(BaseModel):
    class_name: str
    probability: float

class CurrentState(BaseModel):
    pred_class: int
    pred_label: str
    p_attack: float
    reliability: str
    top_predictions: List[TopPrediction]
    raw_probabilities: List[float]

class HorizonPrediction(BaseModel):
    k: int
    horizon_seconds: int
    pred_class: int
    pred_label: str
    p_attack: float
    posthoc_ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None

class AttackTrajectory(BaseModel):
    p_attack_curve: List[float]
    escalation_score: float
    trajectory_status: str

class LeadTime(BaseModel):
    status: str
    lead_seconds: Optional[int] = None
    trigger_k: Optional[int] = None
    onset_index: Optional[int] = None

class TemporalFeature(BaseModel):
    rank: int
    name: str
    relative_change: str
    color_code: str
    evidence_percent: float
    shap_value: float
    score: float

class MitreMapping(BaseModel):
    is_unclassified: bool
    tactic: str
    tactic_id: str
    confidence: str
    emoji: str
    color: str
    score: float

class NetworkState(BaseModel):
    feature_names: List[str]
    feature_values: List[float]

class PredictionResponse(BaseModel):
    engine: str = "native"
    feature_count: int = 71
    engine_label: str = "Deep-Packet Engine"
    current: CurrentState
    forecasts: List[HorizonPrediction]
    trajectory: AttackTrajectory
    lead_time: LeadTime
    evidence: List[TemporalFeature]
    mitre: Optional[MitreMapping] = None
    network_state: NetworkState
    is_attack: bool

class UploadResponse(BaseModel):
    status: str
    message: str
    prediction: Optional[PredictionResponse] = None

class HealthResponse(BaseModel):
    status: str
    inference_mode: str
    features: int
    classes: int
    sequence_length: int
    window_seconds: int
