import json
import base64
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
from typing import Optional, List, Tuple

from backend.schemas.prediction import (
    PredictionResponse, CurrentState, TopPrediction, HorizonPrediction,
    AttackTrajectory, LeadTime, TemporalFeature, MitreMapping, NetworkState
)

# Reuse the existing model definitions
class AttackClassifierLSTM(nn.Module):
    def __init__(self, input_dim=71, hidden_dim=64, num_classes=15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class DirectFutureStateModel(nn.Module):
    def __init__(self, input_dim=71, hidden_dim=128, future_steps=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, future_steps * input_dim)
        self.future_steps = future_steps
        self.input_dim    = input_dim
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).view(-1, self.future_steps, self.input_dim)


# Global singletons
_CLF = None
_FS = None
_DATA = None
_META = None
_SCALER = None
_EXPLAINER = None

_FALLBACK_CLF = None
_FALLBACK_FS = None
_FALLBACK_META = None
_FALLBACK_SCALER = None

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data" / "processed" / "model_ready_v2_primary"
MDL_DIR = ROOT / "models" / "phase4"
FALLBACK_DIR = ROOT / "models" / "fallback"


def initialize_service():
    global _CLF, _FS, _DATA, _META, _SCALER, _EXPLAINER
    global _FALLBACK_CLF, _FALLBACK_FS, _FALLBACK_META, _FALLBACK_SCALER
    
    if _CLF is not None:
        return # Already initialized

    clf_path = MDL_DIR / "attack_classifier_protocolA_v1.pt"
    fs_path  = MDL_DIR / "state_transition_direct_v1.pt"
    
    if not clf_path.exists() or not fs_path.exists():
        raise RuntimeError("MODEL INFERENCE FAILED — Production models not found in models/phase4/")

    _CLF = AttackClassifierLSTM()
    _CLF.load_state_dict(torch.load(clf_path, map_location="cpu"))
    _CLF.eval()
    
    _FS = DirectFutureStateModel()
    _FS.load_state_dict(torch.load(fs_path, map_location="cpu"))
    _FS.eval()
    
    _DATA = np.load(DATA_DIR / "dataset.npz", allow_pickle=True)
    with open(DATA_DIR / "metadata.json") as f:
        _META = json.load(f)
    _SCALER = joblib.load(DATA_DIR / "scaler.joblib")
    
    # Load Explainer
    import sys
    sys.path.insert(0, str(SRC_DIR))
    from explain import build_explainer
    y_str = _DATA["ylab_train"][:, 0]
    y_bin = (y_str != "Benign").astype(int)
    _EXPLAINER = build_explainer(_CLF, _DATA["X_train"], y_bin, n_background=50)

    # Load Fallback Models
    f_clf_path = FALLBACK_DIR / "attack_classifier.pt"
    f_fs_path = FALLBACK_DIR / "state_transition_direct.pt"
    if f_clf_path.exists() and f_fs_path.exists():
        _FALLBACK_CLF = AttackClassifierLSTM(input_dim=6)
        _FALLBACK_CLF.load_state_dict(torch.load(f_clf_path, map_location="cpu"))
        _FALLBACK_CLF.eval()
        
        _FALLBACK_FS = DirectFutureStateModel(input_dim=6, future_steps=10)
        _FALLBACK_FS.load_state_dict(torch.load(f_fs_path, map_location="cpu"))
        _FALLBACK_FS.eval()
        
        with open(FALLBACK_DIR / "metadata.json") as f:
            _FALLBACK_META = json.load(f)
        _FALLBACK_SCALER = joblib.load(FALLBACK_DIR / "scaler.joblib")


def get_dataset_info():
    initialize_service()
    return {
        "num_sequences": len(_DATA["X_test"])
    }

def get_sequence(index: int) -> Tuple[np.ndarray, List[str]]:
    initialize_service()
    if index < 0 or index >= len(_DATA["X_test"]):
        raise ValueError(f"Sequence index {index} out of bounds")
    
    seq = _DATA["X_test"][index]
    ylab_test = _DATA["ylab_test"]
    y_true_seq = [l.replace("Infilteration", "Infiltration") for l in ylab_test[index]]
    return seq, y_true_seq

def build_context(x_seq: np.ndarray, pred_seq: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return x_seq
    h = 10 - (k - 1)
    return np.concatenate([x_seq[10 - h:], pred_seq[:k - 1, :]], axis=0)

def preprocess_csv_upload(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[str]]:
    initialize_service()
    feature_columns = _META["feature_columns_order"]
    
    available = [c for c in feature_columns if c in df.columns]
    if len(available) < 10:
        return None, f"Invalid schema: only {len(available)}/{len(feature_columns)} required feature columns found."
    
    mat = np.zeros((len(df), len(feature_columns)), dtype=np.float32)
    for j, col in enumerate(feature_columns):
        if col in df.columns:
            cleaned = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            mat[:, j] = np.clip(cleaned.values, np.finfo(np.float32).min, np.finfo(np.float32).max).astype(np.float32)
        else:
            mat[:, j] = _SCALER.mean_[j]
            
    scaled = _SCALER.transform(mat).astype(np.float32)
    if len(scaled) >= 10:
        return scaled[-10:], None
    
    pad = np.zeros((10 - len(scaled), len(feature_columns)), dtype=np.float32)
    return np.concatenate([pad, scaled], axis=0), None


def process_sequence(seq: np.ndarray, posthoc_ground_truth: Optional[List[str]] = None) -> PredictionResponse:
    initialize_service()
    
    import sys
    sys.path.insert(0, str(SRC_DIR))
    from explain import explain_sequence
    from mitre import score_mitre_tactics
    
    label_classes = [l.replace("Infilteration", "Infiltration") for l in _META["label_classes"]]
    feature_columns = _META["feature_columns_order"]
    benign_id = label_classes.index("Benign")
    
    # ── Production inference (Phase 5) ────────────────────
    n_classes = len(label_classes)  # 14 runtime labels; model output is 15 but class 14 is unused dead node
    with torch.no_grad():
        fs_preds = _FS(torch.tensor(seq[None], dtype=torch.float32)).cpu().numpy()[0]

    results = {}
    traj = []
    
    with torch.no_grad():
        for k in range(1, 11):
            ctx = build_context(seq, fs_preds, k)
            probs = torch.softmax(_CLF(torch.tensor(ctx[None], dtype=torch.float32)), dim=1).detach().cpu().numpy()[0]
            pc = int(np.argmax(probs))
            # Guard: if model predicts the unused 15th node (index 14), default to benign
            if pc >= n_classes:
                pc = benign_id
            pa = float(1.0 - probs[benign_id])
            results[k] = {
                "pred_label": label_classes[pc], 
                "p_attack": pa,
                "probs": probs[:n_classes],  # trim to known classes
                "pred_class": pc, 
                "ctx": ctx
            }
            traj.append(pa)

    k1 = results[1]
    is_atk = k1["pred_class"] != benign_id
    p1 = k1["p_attack"]

    # Phase 8 — escalation
    Kd = np.arange(1, 11) - 5.5
    ya = np.array(traj)
    esc = float(np.clip(np.sum(Kd * (ya - ya.mean())) / 82.5 * 9.0, -1.0, 1.0))
    if esc >= 0.20:
        traj_status = "ESCALATING"
    elif esc <= -0.20:
        traj_status = "DE-ESCALATING"
    else:
        traj_status = "STABLE"

    # Phase 10 — reliability (calibration-based)
    # HIGH: P(Attack) >= 0.90 (very high confidence attack signal)
    # HIGH: P(Attack) < 0.10 (very high confidence benign signal)
    # LOW: overconfident / uncertain region 0.10–0.90
    if p1 >= 0.90 or p1 < 0.10:
        rel_lbl = "HIGH"
    else:
        rel_lbl = "LOW"

    # Phase 7 — lead time (exact logic, no leakage)
    onset_idx = -1
    if posthoc_ground_truth is not None:
        onset_idx = next((i for i, l in enumerate(posthoc_ground_truth) if l != "Benign"), -1)
        
    valid_k = None
    if onset_idx >= 0 and posthoc_ground_truth is not None:
        for _k in [1, 3, 5, 10]:
            if results[_k]["pred_class"] != benign_id and posthoc_ground_truth[_k - 1] != "Benign":
                valid_k = _k
                break
                
    lead_time_status = ""
    lead_seconds = None
    if onset_idx == -1:
        lead_time_status = "NO THREAT"
    elif valid_k is not None:
        lead_time_status = "EARLY WARNING"
        lead_seconds = (onset_idx + 1) * 30
    else:
        lead_time_status = "MISSED"

    # Inverse-scaled observed state for network stats
    W_seq = _SCALER.inverse_transform(seq)
    curr_state = W_seq[-1]
    
    network_state = NetworkState(
        feature_names=["Fwd Pkts/s", "Bwd Pkts/s", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts"],
        feature_values=[]
    )
    for name in network_state.feature_names:
        if name in feature_columns:
            network_state.feature_values.append(float(curr_state[feature_columns.index(name)]))
        else:
            network_state.feature_values.append(0.0)

    # ── SHAP / MITRE ────────────────────
    mitre_obj = None
    feat_data = []

    if is_atk:
        try:
            shap_result = explain_sequence(_EXPLAINER, _CLF, seq, feature_columns,
                                           label_classes, benign_id, top_k=71)
            mitre_result = score_mitre_tactics(
                shap_result.ranked_indices.astype(int),
                shap_result.all_shap_values, feature_columns)

            Z_recent = seq[-5:]
            W_recent = W_seq[-5:]
            rc = Z_recent[-1] - Z_recent[0]
            delta_abs = W_recent[-1] - W_recent[-2]
            sv = shap_result.all_shap_values
            ns = np.abs(sv); ns = ns / max(np.max(ns), 1e-9)
            nc = np.abs(rc); nc = nc / max(np.max(nc), 1e-9)
            evidence = 0.5 * ns + 0.5 * nc
            top_idx = np.argsort(evidence)[::-1][:5]

            for rank, j in enumerate(top_idx):
                d_abs = delta_abs[j]
                dv = d_abs / max(abs(W_recent[-2, j]), 1e-6)
                rel = f"{dv*100:+.1f}%" if abs(dv) <= 1000 else "N/A"
                tc = "#ef4444" if d_abs >= 0 else "#0ea5e9"
                feat_data.append(TemporalFeature(
                    rank=rank + 1,
                    name=feature_columns[j],
                    relative_change=rel,
                    color_code=tc,
                    evidence_percent=min(float(evidence[j]) * 100, 100),
                    shap_value=float(sv[j]),
                    score=float(evidence[j])
                ))
                
            if mitre_result.unclassified:
                mitre_obj = MitreMapping(
                    is_unclassified=True, tactic="", tactic_id="", confidence="", emoji="", color="", score=0.0
                )
            else:
                from mitre import TACTIC_META
                tmeta = TACTIC_META.get(mitre_result.top_tactic, {"emoji": "⚠️", "id": "N/A", "color": "#f59e0b"})
                mitre_obj = MitreMapping(
                    is_unclassified=False,
                    tactic=mitre_result.top_tactic,
                    tactic_id=mitre_result.top_tactic_id,
                    confidence=mitre_result.confidence.upper(),
                    emoji=tmeta.get("emoji", "⚠️"),
                    color=tmeta.get("color", "#f59e0b"),
                    score=float(mitre_result.top_tactic_score)
                )
        except Exception:
            pass
    else:
        # Benign: show largest raw feature movements
        W_recent = W_seq[-5:]
        delta_abs = W_recent[-1] - W_recent[-2]
        top_raw = np.argsort(np.abs(delta_abs))[::-1][:3]
        for rank, j in enumerate(top_raw):
            d = delta_abs[j]
            dv = d / max(abs(W_recent[-2, j]), 1e-6)
            feat_data.append(TemporalFeature(
                rank=rank+1,
                name=feature_columns[j],
                relative_change=f"{dv*100:+.1f}%" if abs(dv) <= 1000 else "N/A",
                color_code="#ef4444" if d >= 0 else "#0ea5e9",
                evidence_percent=0.0,
                shap_value=0.0,
                score=0.0
            ))

    probs = k1["probs"]  # already trimmed to n_classes
    order = np.argsort(probs)[::-1]
    top_preds = []
    for idx in order[:3]:
        top_preds.append(TopPrediction(
            class_name=label_classes[idx],
            probability=float(probs[idx])
        ))

    # Forecasts
    forecasts = []
    for _k in [1, 3, 5, 10]:
        rk = results[_k]
        gt = posthoc_ground_truth[_k - 1] if posthoc_ground_truth else None
        forecasts.append(HorizonPrediction(
            k=_k,
            horizon_seconds=_k * 30,
            pred_class=rk["pred_class"],
            pred_label=rk["pred_label"],
            p_attack=rk["p_attack"],
            posthoc_ground_truth=gt,
            is_correct=(rk["pred_label"] == gt) if gt else None
        ))
    return PredictionResponse(
        current=CurrentState(
            pred_class=k1["pred_class"],
            pred_label=k1["pred_label"],
            p_attack=k1["p_attack"],
            reliability=rel_lbl,
            top_predictions=top_preds,
            raw_probabilities=probs.tolist()
        ),
        forecasts=forecasts,
        trajectory=AttackTrajectory(
            p_attack_curve=[float(x) for x in traj],
            escalation_score=esc,
            trajectory_status=traj_status
        ),
        lead_time=LeadTime(
            status=lead_time_status,
            lead_seconds=lead_seconds,
            trigger_k=valid_k,
            onset_index=onset_idx if onset_idx != -1 else None
        ),
        evidence=feat_data,
        mitre=mitre_obj,
        network_state=network_state,
        is_attack=is_atk
    )

def process_sequence_fallback(seq: np.ndarray, posthoc_ground_truth: Optional[List[str]] = None) -> PredictionResponse:
    initialize_service()
    
    label_classes = [l.replace("Infilteration", "Infiltration") for l in _FALLBACK_META["label_classes"]]
    feature_columns = _FALLBACK_META["feature_names"]
    benign_id = label_classes.index("Benign")
    
    n_classes = len(label_classes)  # 14 runtime labels
    with torch.no_grad():
        fs_preds = _FALLBACK_FS(torch.tensor(seq[None], dtype=torch.float32)).cpu().numpy()[0]

    results = {}
    traj = []
    
    with torch.no_grad():
        for k in range(1, 11):
            ctx = build_context(seq, fs_preds, k)
            probs = torch.softmax(_FALLBACK_CLF(torch.tensor(ctx[None], dtype=torch.float32)), dim=1).detach().cpu().numpy()[0]
            pc = int(np.argmax(probs))
            # Guard: if model predicts unused node >= n_classes, default to benign
            if pc >= n_classes:
                pc = benign_id
            pa = float(1.0 - probs[benign_id])
            results[k] = {
                "pred_label": label_classes[pc], 
                "p_attack": pa,
                "probs": probs[:n_classes],
                "pred_class": pc, 
                "ctx": ctx
            }
            traj.append(pa)

    k1 = results[1]
    is_atk = k1["pred_class"] != benign_id
    p1 = k1["p_attack"]

    Kd = np.arange(1, 11) - 5.5
    ya = np.array(traj)
    esc = float(np.clip(np.sum(Kd * (ya - ya.mean())) / 82.5 * 9.0, -1.0, 1.0))
    if esc >= 0.20:
        traj_status = "ESCALATING"
    elif esc <= -0.20:
        traj_status = "DE-ESCALATING"
    else:
        traj_status = "STABLE"

    if p1 >= 0.90 or p1 < 0.10: rel_lbl = "HIGH"
    else: rel_lbl = "LOW"

    lead_time_status = "MISSED"
    lead_seconds = None
    valid_k = None
    onset_idx = -1

    W_seq = _FALLBACK_SCALER.inverse_transform(seq)
    curr_state = W_seq[-1]
    
    network_state = NetworkState(
        feature_names=["Fwd Pkts/s", "Bwd Pkts/s", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts"],
        feature_values=[]
    )
    for name in network_state.feature_names:
        if name in feature_columns:
            network_state.feature_values.append(float(curr_state[feature_columns.index(name)]))
        else:
            network_state.feature_values.append(0.0)

    # Mitre heuristics
    mitre_obj = MitreMapping(
        is_unclassified=False,
        tactic="Fallback Analysis",
        tactic_id="TA-FB",
        confidence="LOW" if p1 < 0.5 else "HIGH",
        emoji="🔍",
        color="#8b5cf6",
        score=float(p1)
    )

    feat_data = []
    # Benign: show largest raw feature movements
    W_recent = W_seq[-5:]
    delta_abs = W_recent[-1] - W_recent[-2]
    top_raw = np.argsort(np.abs(delta_abs))[::-1][:3]
    for rank, j in enumerate(top_raw):
        d = delta_abs[j]
        dv = d / max(abs(W_recent[-2, j]), 1e-6)
        feat_data.append(TemporalFeature(
            rank=rank+1,
            name=feature_columns[j],
            relative_change=f"{dv*100:+.1f}%" if abs(dv) <= 1000 else "N/A",
            color_code="#ef4444" if d >= 0 else "#0ea5e9",
            evidence_percent=0.0,
            shap_value=0.0,
            score=0.0
        ))

    probs = k1["probs"]
    order = np.argsort(probs)[::-1]
    top_preds = []
    for idx in order[:3]:
        top_preds.append(TopPrediction(
            class_name=label_classes[idx],
            probability=float(probs[idx])
        ))

    forecasts = []
    for _k in [1, 3, 5, 10]:
        rk = results[_k]
        gt = posthoc_ground_truth[_k - 1] if posthoc_ground_truth else None
        forecasts.append(HorizonPrediction(
            k=_k,
            horizon_seconds=_k * 30,
            pred_class=rk["pred_class"],
            pred_label=rk["pred_label"],
            p_attack=rk["p_attack"],
            posthoc_ground_truth=gt,
            is_correct=(rk["pred_label"] == gt) if gt else None
        ))

    return PredictionResponse(
        engine="fallback",
        feature_count=6,
        engine_label="Generic Flow Engine",
        current=CurrentState(
            pred_class=k1["pred_class"],
            pred_label=k1["pred_label"],
            p_attack=k1["p_attack"],
            reliability=rel_lbl,
            top_predictions=top_preds,
            raw_probabilities=probs.tolist()
        ),
        forecasts=forecasts,
        trajectory=AttackTrajectory(
            p_attack_curve=[float(x) for x in traj],
            escalation_score=esc,
            trajectory_status=traj_status
        ),
        lead_time=LeadTime(
            status=lead_time_status,
            lead_seconds=lead_seconds,
            trigger_k=valid_k,
            onset_index=onset_idx if onset_idx != -1 else None
        ),
        evidence=feat_data,
        mitre=mitre_obj if is_atk else None,
        network_state=network_state,
        is_attack=is_atk
    )
