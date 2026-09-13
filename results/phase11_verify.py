import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import joblib

ROOT       = Path(".").resolve()
DATA_DIR_A = ROOT / "data" / "processed" / "model_ready_v2_primary"
MDL_DIR    = ROOT / "models" / "phase4"
OUT_DIR    = ROOT / "results" / "phase11"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        self.future_steps = future_steps; self.input_dim = input_dim
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).view(-1, self.future_steps, self.input_dim)

with open(DATA_DIR_A / "metadata.json") as f:
    meta = json.load(f)
label_classes = [l.replace("Infilteration","Infiltration") for l in meta["label_classes"]]
benign_id = label_classes.index("Benign")
feature_cols = meta["feature_columns_order"]

device = "cpu"
clf = AttackClassifierLSTM().to(device)
clf.load_state_dict(torch.load(MDL_DIR/"attack_classifier_protocolA_v1.pt", map_location=device))
clf.eval()

fs_model = DirectFutureStateModel().to(device)
fs_model.load_state_dict(torch.load(MDL_DIR/"state_transition_direct_v1.pt", map_location=device))
fs_model.eval()

scaler = joblib.load(DATA_DIR_A / "scaler.joblib")
data = np.load(DATA_DIR_A / "dataset.npz", allow_pickle=True)
X_test = data["X_test"]; ylab_test = data["ylab_test"]

def build_context(x_seq, pred_seq, k):
    if k == 1: return x_seq
    history_len = 10 - (k - 1)
    return np.concatenate([x_seq[10 - history_len:], pred_seq[:k-1, :]], axis=0)

# Pick a representative sequence: find one with an upcoming attack at K=3
seq_idx = 0
for i in range(len(X_test)):
    labels = [str(l).replace("Infilteration","Infiltration") for l in ylab_test[i]]
    if labels[0] == "Benign" and any(l != "Benign" for l in labels[1:4]):
        seq_idx = i; break

seq = X_test[seq_idx]
y_true_seq = [str(l).replace("Infilteration","Infiltration") for l in ylab_test[seq_idx]]

# Ground truth used ONLY after inference
print("=== PHASE 11 VERIFICATION ===")
print(f"Selected sequence: {seq_idx}")
print(f"Observed context: W(t-9) ... W(t)  shape={seq.shape}")
print(f"Ground truth used during inference: FALSE")

with torch.no_grad():
    fs_preds = fs_model(torch.tensor(seq[None], dtype=torch.float32)).cpu().numpy()[0]

results = {}
traj = []
for k in range(1, 11):
    ctx = build_context(seq, fs_preds, k)
    with torch.no_grad(): probs = torch.softmax(clf(torch.tensor(ctx[None], dtype=torch.float32)), dim=1).cpu().numpy()[0]
    pc = int(np.argmax(probs))
    results[k] = {"label": label_classes[pc], "p_attack": float(1.0 - probs[benign_id]), "pred_class": pc}
    traj.append(float(1.0 - probs[benign_id]))

# Phase 8 Escalation Score
K_arr = np.arange(1, 11); Kd = K_arr - 5.5
slope = np.sum(Kd * (np.array(traj) - np.mean(traj))) / 82.5
esc = float(np.clip(slope * 9.0, -1.0, 1.0))
traj_status = "ESCALATING" if esc >= 0.20 else ("DE-ESCALATING" if esc <= -0.20 else "STABLE")

# Phase 7 Lead Time
onset_idx = next((i for i, l in enumerate(y_true_seq) if l != "Benign"), -1)
valid_k = None
if onset_idx >= 0:
    for _k in [1,3,5,10]:
        if results[_k]["pred_class"] != benign_id and y_true_seq[_k-1] != "Benign":
            valid_k = _k; break
lt_str = f"+{(onset_idx+1)*30}s lead, K={valid_k}" if valid_k else "No early warning"

# Phase 9 What Changed
W = scaler.inverse_transform(seq)
Z_rec = seq[-5:]; W_rec = W[-5:]
rc = Z_rec[-1] - Z_rec[0]; da = W_rec[-1] - W_rec[-2]
Kd5 = np.array([0,1,2,3,4]) - 2.0
norm_c = np.abs(rc); norm_c = norm_c / max(np.max(norm_c), 1e-9)
# Use temporal slope as proxy for SHAP magnitude in verification (SHAP not run here to save time)
top5_change = np.argsort(np.abs(rc))[::-1][:5]

print(f"\nFuture forecast:")
for k in [1,3,5,10]:
    print(f"  K={k}: {results[k]['label']} P(Attack)={results[k]['p_attack']:.1%}  (GT post-hoc: {y_true_seq[k-1]})")

print(f"\nP(Attack) trajectory:")
print(f"  {[round(p,3) for p in traj]}")
print(f"\nEscalation score: {esc:+.3f}")
print(f"Trajectory:       {traj_status}")
print(f"Lead time:        {lt_str}")

print(f"\nWhat Changed — top temporal-change features:")
for rank, j in enumerate(top5_change):
    rel = da[j] / max(abs(W_rec[-2,j]), 1e-6)
    rel_s = f"{rel*100:+.1f}%" if abs(rel) <= 1000 else "N/A"
    print(f"  {rank+1}. {feature_cols[j]:30s}  Z-diff={rc[j]:+.3f}  Raw-change={rel_s}")

print(f"\nMITRE K=1: see dashboard SHAP panel (not re-computed in CLI verification)")
print(f"Ground truth used during inference: FALSE")

with open(OUT_DIR / "attack_replay_demo.md", "w", encoding="utf-8") as f:
    f.write("# Phase 11: Attack Replay / SIH Demo Mode\n\n")
    f.write("## Overview\n")
    f.write("The SIH Demo Mode provides a judge-friendly chronological replay of Protocol A test sequences, ")
    f.write("showing CHRONEX forecasting future attack states in real time.\n\n")
    f.write("Toggle: **🎬 SIH Demo / Replay Mode** in the sidebar.\n\n")

    f.write("## Model Artifacts Used\n")
    f.write("| Artifact | Path |\n|---|---|\n")
    f.write("| Attack Classifier | `models/phase4/attack_classifier_protocolA_v1.pt` |\n")
    f.write("| Direct Future-State Decoder | `models/phase4/state_transition_direct_v1.pt` |\n")
    f.write("| Scaler | `data/processed/model_ready_v2_primary/scaler.joblib` |\n")
    f.write("| Feature Order | 71 features from `metadata.json` |\n\n")

    f.write("## Navigation\n")
    f.write("- **Slider**: Select any test sequence index (0 to N-1)\n")
    f.write("- **⏮ Prev / Next ⏭**: Step one sequence at a time\n")
    f.write("- **↺ Reset**: Return to index 0\n\n")

    f.write("## Observed vs Post-Hoc Ground Truth\n")
    f.write("| Information | Source | Labelling |\n|---|---|---|\n")
    f.write("| Observed context W(t-9)…W(t) | Protocol A X_test | Used for inference |\n")
    f.write("| P(Attack) K=1..10 | Model softmax output | Displayed as forecast |\n")
    f.write("| Predicted class | argmax of softmax | Displayed as forecast |\n")
    f.write("| Actual future class | ylab_test | Shown as **Ground Truth (post-hoc)** |\n\n")

    f.write("## Reused Phase Metrics\n")
    f.write("| Phase | Metric | Demo Section |\n|---|---|---|\n")
    f.write("| Phase 8 | Escalation Score, Trajectory | Attack Risk Trajectory |\n")
    f.write("| Phase 7 | Lead Time, Early Warning | Forecast Lead Time |\n")
    f.write("| Phase 9 | Temporal Evidence Score, What Changed | SHAP + Evidence panel |\n")
    f.write("| Phase 10 | Forecast Reliability label | Current State card |\n\n")

    f.write("## Status Banner Logic\n")
    f.write("| Banner | Condition |\n|---|---|\n")
    f.write("| EARLY WARNING | `valid_k_demo is not None` (model predicts attack before onset) |\n")
    f.write("| ESCALATING RISK | `traj_status == 'ESCALATING'` |\n")
    f.write("| WATCH | `P(Attack at K=1) >= 0.30` |\n")
    f.write("| NORMAL | All others |\n\n")

    f.write("## Known Limitations\n")
    f.write("- Exact timestamps are not stored in dataset.npz; windows shown as relative offsets (+30s per step).\n")
    f.write("- SHAP computation adds ~2s latency per sequence when the attack classifier predicts a non-Benign class.\n")
    f.write("- Forecast uncertainty intervals are not modeled (Phase 4J limitation).\n")
    f.write("- Replay is Protocol A only; Protocol B requires separate scaler inversion step.\n")

print(f"\nDocumentation saved to {OUT_DIR}/attack_replay_demo.md")


