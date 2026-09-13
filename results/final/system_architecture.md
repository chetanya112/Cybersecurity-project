# CHRONEX — System Architecture
## SIH Problem Statement 26153 | NTRO

---

## Conceptual Data Flow

```
RAW NETWORK TRAFFIC (CIC-IDS-2018)
  └─► ~16.6M raw NetFlow records (10 capture days)

FLOWS
  └─► clean_data.py → remove inf/NaN, clip, drop exact-duplicate rows

30-SECOND WINDOWS
  └─► build_sequences_v2.py
      → aggregate flow-level features into fixed 30-second time windows
      → compute 70 statistical flow features + flow_count = 71 features per window

71-D NETWORK STATES  [shape: (71,)]
  └─► scaler.joblib (StandardScaler, fit on Protocol A training windows only)
      → standardized 71-D state vector

10-WINDOW TEMPORAL CONTEXT  [shape: (10, 71)]
  └─► sliding window over chronologically ordered states
      → each sequence = 5-minute observed history
      → Protocol A: train/val/test splits by calendar date (zero overlap)
      → Protocol B: held-out dates for unseen generalization evaluation

DIRECT MULTI-HORIZON FUTURE-STATE DECODER
  (state_transition_direct_v1.pt)
  Architecture: 2-layer LSTM (hidden=128) → Linear → reshape
  Input:  (batch, 10, 71)  — observed history
  Output: (batch, 10, 71)  — predicted future states W+1 … W+10

W+1 … W+10  [10 × 71-D predicted future states]
  └─► build_context(x_seq, fs_preds, k):
      For horizon K, construct rolling 10-window context:
        [W(t+K−10) … W(t)] + [Pred(t+1) … Pred(t+K−1)]
      Total context always = exactly 10 windows

15-CLASS ATTACK FORECASTING
  (attack_classifier_protocolA_v1.pt)
  Architecture: 2-layer LSTM (hidden=64) → Linear(15)
  Input:  (batch, 10, 71)  — rolling context
  Output: (batch, 15)      — 15-class logits

  Classes (14 attacks + Benign):
    Benign, Bot, Brute Force -Web, Brute Force -XSS,
    DDOS attack-HOIC, DDOS attack-LOIC-UDP, DDoS attacks-LOIC-HTTP,
    DoS attacks-GoldenEye, DoS attacks-SlowHTTPTest, DoS attacks-Slowloris,
    FTP-BruteForce, Infiltration, SQL Injection, SSH-Bruteforce

  P(Attack) = 1 − P(Benign class)   [NOT a multi-class probability]

RISK TRAJECTORY  [Phase 8]
  └─► P1 … P10 = P(Attack) at K=1 … K=10
  └─► Escalation Score = clip(slope × 9, −1, +1)
      where slope = linear regression of P(k) over k=1..10
  └─► Categories: ESCALATING (≥+0.20) | STABLE | DE-ESCALATING (≤−0.20)

LEAD TIME  [Phase 7]
  └─► Onset index = first actual attack window in forecast horizon
  └─► Lead time = (Onset_Index − Forecast_Index) × 30 seconds
  └─► Valid only when model predicts non-Benign AND target window is actual attack

TEMPORAL EVIDENCE  [Phase 9]
  └─► Observed only: W(t−4) … W(t)
  └─► Temporal Evidence Score = 0.5 × norm(|SHAP|) + 0.5 × norm(|Z-score change|)
  └─► NOTE: This is a ranking score, NOT a causal attribution

K=1 SHAP / MITRE ATT&CK  [Phase 5 / existing]
  └─► SHAP background: benign training sequences
  └─► MITRE tactic: aggregated SHAP score heuristic
  └─► Applied to K=1 (observed-context) prediction ONLY

FORECAST RELIABILITY  [Phase 10]
  └─► Derived from calibration study (ECE, Brier, bin-level gaps)
  └─► High: P(Attack) ≥ 0.90  (gap = 0.010, precision = 98.2%)
  └─► Moderate: P(Attack) < 0.10  (gap = 0.037)
  └─► Low: 0.10–0.90  (gaps 0.063–0.474)

SECURITY DECISION SUPPORT  [Phase 11 — SIH Demo Mode]
  └─► Status banner: NORMAL / WATCH / ESCALATING RISK / EARLY WARNING
  └─► Visual timeline: observed (blue) → forecast (risk-colored)
  └─► Ground truth displayed post-hoc only (not during inference)
```

---

## Key Design Constraints

| Constraint | Status |
|---|---|
| No future ground truth during inference | ✅ Enforced |
| No np.gradient in production path | ✅ Enforced (legacy path gated) |
| Protocol A scaler used for all inference | ✅ Enforced |
| Calibration transform NOT fit on test data | ✅ Evaluation only |
| Forecast uncertainty intervals | ❌ Not modeled (Phase 4J limitation) |
| MITRE attribution causal claim | ❌ Not made; heuristic only |

---

## Production Artifacts

| File | Purpose |
|---|---|
| `models/phase4/attack_classifier_protocolA_v1.pt` | 15-class LSTM attack classifier |
| `models/phase4/state_transition_direct_v1.pt` | Direct Multi-Horizon Future-State Decoder |
| `data/processed/model_ready_v2_primary/scaler.joblib` | StandardScaler (fit on Protocol A train) |
| `data/processed/model_ready_v2_primary/metadata.json` | Feature order, label classes, dataset stats |
