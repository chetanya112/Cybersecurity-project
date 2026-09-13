export interface TopPrediction {
  class_name: string;
  probability: number;
}

export interface CurrentState {
  pred_class: number;
  pred_label: string;
  p_attack: number;
  reliability: string;
  top_predictions: TopPrediction[];
  raw_probabilities: number[];
}

export interface HorizonPrediction {
  k: number;
  horizon_seconds: number;
  pred_class: number;
  pred_label: string;
  p_attack: number;
  posthoc_ground_truth: string | null;
  is_correct: boolean | null;
}

export interface AttackTrajectory {
  p_attack_curve: number[];
  escalation_score: number;
  trajectory_status: string;
}

export interface LeadTime {
  status: string;
  lead_seconds: number | null;
  trigger_k: number | null;
  onset_index: number | null;
}

export interface TemporalFeature {
  rank: number;
  name: string;
  relative_change: string;
  color_code: string;
  evidence_percent: number;
  shap_value: number;
  score: number;
}

export interface MitreMapping {
  is_unclassified: boolean;
  tactic: string;
  tactic_id: string;
  confidence: string;
  emoji: string;
  color: string;
  score: number;
}

export interface NetworkState {
  feature_names: string[];
  feature_values: number[];
}

export interface PredictionResponse {
  engine: string;
  feature_count: number;
  engine_label: string;
  current: CurrentState;
  forecasts: HorizonPrediction[];
  trajectory: AttackTrajectory;
  lead_time: LeadTime;
  evidence: TemporalFeature[];
  mitre: MitreMapping | null;
  network_state: NetworkState;
  is_attack: boolean;
}

export interface DatasetInfo {
  num_sequences: number;
}

export interface FeatureMapping {
  source: string;
  canonical: string;
  method: string;
  confidence: number;
}

export interface SchemaFlags {
  timestamp: boolean;
  network_flow: boolean;
  packet_statistics: boolean;
  byte_statistics: boolean;
  duration: boolean;
}

export interface CompatibilityReport {
  status: string;
  compatibility: string;
  upload_id?: string;
  reason?: string;
  error_code?: string;
  schema_flags?: SchemaFlags;
  flow_count?: number;
  usable_windows?: number;
  usable_sequences?: number;
  mapped_features?: FeatureMapping[];
  derived_features?: string[];
  missing_features?: string[];
  missing_requirements?: string[];
}
