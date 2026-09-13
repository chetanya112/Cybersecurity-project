# CHRONEX — Network Attack Forecasting
## SIH Problem Statement 26153 | NTRO

[![SIH26153](https://img.shields.io/badge/SIH-26153-blue)](https://www.sih.gov.in/)
[![NTRO](https://img.shields.io/badge/Org-NTRO-red)](https://www.ntro.gov.in/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS--2018-green)](https://www.unb.ca/cic/datasets/ids-2018.html)

CHRONEX is a temporal network-traffic forecasting system that:
1. **Predicts future 71-D network states** up to 10 windows (5 minutes) ahead using a trained Direct Multi-Horizon Future-State Decoder.
2. **Forecasts future attack risk** at each horizon using a 15-class LSTM attack classifier applied to rolling predicted contexts.
3. **Provides interpretable security signals**: escalation score, forecast lead time, temporal evidence ranking, and MITRE ATT&CK attribution.

CHRONEX uses a **Dual-Engine Architecture**:
- **Primary Engine**: 71-feature NATIVE CIC-IDS representation (requires exact feature semantics)
- **Fallback Engine**: 6-feature GENERIC flow representation for standard external CSVs. If no timestamp is provided, CHRONEX synthesizes a logical clock for broad compatibility.

---

## Architecture

```
CIC-IDS-2018 Raw Flows (~16.6M)
  └─► clean_data.py        → cleaned flows
  └─► build_sequences_v2.py → 30-second windows → 71 features/window
  └─► combine_and_resplit.py → Protocol A/B chronological splits

Protocol A dataset.npz + scaler.joblib
  └─► train_phase4.py:
        ├─ Direct Multi-Horizon Future-State Decoder (LSTM enc → Linear)
        │    Input:  (batch, 10, 71)  observed history
        │    Output: (batch, 10, 71)  predicted W+1 ... W+10
        └─ 15-Class Attack Classifier (2-layer LSTM, hidden=64)
             Input:  (batch, 10, 71)  rolling context
             Output: (batch, 15)      attack class probabilities

At inference (K-step forecast):
  build_context(x_seq, fs_preds, k):
    context = [W(t+K-10) ... W(t)] + [Pred(t+1) ... Pred(t+K-1)]
  → 15-class softmax → P(Attack) = 1 − P(Benign)

  Phase 8: Escalation Score = clip(slope(P(k)) × 9, −1, +1)
  Phase 7: Lead Time = (onset_idx − forecast_idx) × 30s
  Phase 9: Temporal Evidence = 0.5*norm(|SHAP|) + 0.5*norm(|Z-change|)
  Phase 10: Reliability label derived from calibration study (ECE, Brier)

Dashboard: FastAPI Backend + React Frontend
  ├─ Normal / Analyst Mode (CSV Uploads via dual-engine inference)
  └─ 🎬 SIH Demo / Replay Mode (Historical ground-truth evaluation)
```

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
cd frontend && npm install
```

---

## Data Pipeline (skip if model_ready_v2_primary/ exists)

```bash
python src/clean_data.py
python src/build_sequences_v2.py
python src/combine_and_resplit.py
```

---

## Training

```bash
python src/train_phase4.py
# Artifacts saved to: models/phase4/
```

---

## Evaluation

```bash
python src/eval_phase6.py   # Protocol B (unseen generalization)
python src/eval_phase7.py   # Forecast lead time
python src/eval_phase8.py   # Attack trajectory / escalation score
python src/eval_phase9.py   # Temporal evidence (SHAP + Z-score)
python src/eval_phase10.py  # Calibration & reliability
python src/audit_phase12.py # Final system audit
```

---

## Dashboard

Start the backend:
```bash
python -m uvicorn backend.main:app --port 8000
```

Start the frontend (in a separate terminal):
```bash
cd frontend
npm run dev
```

**Features:**
- 🎬 **SIH Demo Mode**: chronological replay with status banner, visual timeline, lead-time display
- 📈 **K=1…10 Attack Trajectory**: escalation score and bar chart
- ⏱️ **Forecast Lead Time**: early-warning status for upcoming attacks
- 🔍 **What Changed?**: temporal evidence ranking (SHAP + Z-score change)
- 🎯 **MITRE ATT&CK**: K=1 heuristic tactic mapping
- 📊 **Forecast Reliability**: calibration-derived label (High/Moderate/Low)
- 📂 **CSV Upload**: live inference on external NetFlow CSVs (auto-routes to Native or Generic Fallback engine)

---

## Verified Results (held-out test sets)

| Metric | Value |
|--------|-------|
| Dataset | CIC-IDS-2018 · 10 days · ~16.6M raw flows |
| Features | 71 per window (70 traffic + flow_count) |
| Sequence | 10 windows × 30s = 5-minute context |
| Target classes | 15 (Benign + 14 attack types) |
| **Attack F1 at K=1 (Protocol B)** | **0.885** |
| **Attack F1 at K=10 (Protocol B)** | **0.795** |
| Future-state RMSE vs Persistence (K=1, Protocol B) | 0.802 vs 1.104 |
| P(Attack) Precision at P≥0.90 | 98.2% (Protocol A) |
| Brier Score | 0.062 (Protocol A), 0.062 (Protocol B) |
| Early-warning detection | 2/4 eligible test episodes (30s median lead time) |

**Important**: Macro F1 (15-class) is intentionally low (0.18–0.23) because it penalizes the full 15-class prediction difficulty, not just binary attack/benign discrimination.

---

## Key Limitations

1. **Forecast uncertainty intervals**: Not modeled (Phase 4J limitation).
2. **Exact-class attribution**: MITRE mapping is a SHAP-weighted heuristic, not ground-truth causal attribution.
3. **Lead time**: Only 4 contiguous eligible episodes extracted from Protocol A test set; population statistics require larger continuous captures.
4. **Calibration**: Well-calibrated at P(Attack) ≥ 0.90; overconfident in 0.10–0.90 range.
5. **P(Attack) interpretation**: Measures non-Benign probability only. Does NOT mean the predicted attack class is correct with that probability.

---

## Citation

Dataset: Sharafaldin, I., Habibi Lashkari, A., Ghorbani, A.A. (2018).
"Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization."
In ICISSP 2018.
