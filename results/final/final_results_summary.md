# CHRONEX — Final Results Summary

> All results are from held-out test sets. No calibration transforms were fit on test data.  
> Ground truth was never accessed during model inference.

---

## A. Future-State Forecasting (Protocol B — Unseen Generalization)

| Model | K | MAE | RMSE |
|---|---|---|---|
| **Persistence baseline** | 1 | 0.485 | 1.104 |
| **Persistence baseline** | 10 | 0.544 | 1.158 |
| **Direct Decoder** (production) | 1 | **0.394** | **0.802** |
| **Direct Decoder** (production) | 10 | **0.415** | **0.836** |
| Recursive LSTM (benchmark) | 1 | 0.395 | 0.797 |
| Recursive LSTM (benchmark) | 10 | 0.407 | 0.830 |

**Production choice**: Direct Decoder — lower RMSE, simpler inference, stable long-horizon.  
**Note**: Recursive LSTM has marginally lower MAE at K=3,5,10.

---

## B. Attack Forecasting (Protocol B Test — Direct Decoder)

| Horizon | Macro F1 (15-class) | Attack F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|
| K=1 | 0.229 | **0.885** | 0.956 | 0.823 | 0.022 |
| K=3 | 0.225 | 0.883 | 0.957 | 0.820 | 0.021 |
| K=5 | 0.218 | 0.876 | 0.967 | 0.802 | 0.016 |
| K=10 | 0.185 | 0.795 | 0.843 | 0.752 | 0.081 |

**Note**: Macro F1 uses fixed 15-class denominator to penalize class hallucination.

---

## C. Forecast Lead Time (Protocol A Test — Contiguous Episode Analysis)

| Metric | Value |
|---|---|
| Eligible attack episodes | 4 |
| Detected (binary early warning) | 2 |
| Detection rate | 50% |
| Median lead time | **30 seconds** |
| Episode types detected | FTP-BruteForce, Bot |
| Episode types missed | DoS-Slowloris, Infiltration |

> **Limitation**: Only 4 contiguous eligible episodes were identified due to strict chronological stitching. Population-level lead time statistics require a larger continuous capture window.

---

## D. Attack Trajectory (Escalation Score)

| Metric | Protocol A | Protocol B |
|---|---|---|
| Attack mean score | +0.100 | −0.061 |
| Benign mean score | +0.025 | +0.063 |
| **False Escalation Rate** | **8.7%** | **14.1%** |

**Formula**: `score = clip(slope(P(k) over k=1..10) × 9, −1, +1)`  
**Thresholds**: ESCALATING ≥ +0.20 | STABLE (−0.20, +0.20) | DE-ESCALATING ≤ −0.20

---

## E. Temporal Explanation

**Formula**: `evidence = 0.5 × norm(|SHAP|) + 0.5 × norm(|Z-score change|)`

| Metric | Protocol A | Protocol B |
|---|---|---|
| Attack mean temporal evidence | 0.2033 | 0.1963 |
| Benign mean temporal evidence | 0.2301 | 0.2100 |

> ⚠️ **IMPORTANT**: The Temporal Evidence Score is a **ranking mechanism** combining model attribution and recent standardized feature change. It is **NOT** causal attribution and does NOT imply that any feature caused the attack.

---

## F. Calibration (P(Attack) = 1 − P(Benign))

| Metric | Protocol A | Protocol B |
|---|---|---|
| Brier Score | 0.0624 | 0.0621 |
| ECE | 0.0773 | 0.0354 |
| Mean Abs Calibration Error | 0.197 | 0.083 |
| **P≥0.90 Precision** | **98.2%** | **98.8%** |
| **P≥0.90 Recall** | 70.7% | 79.2% |

**Key finding**: P(Attack) is well-calibrated in the extreme bins (gap = 0.010 at P≥0.90) but overconfident in the 0.10–0.90 range. The dashboard reliability label is derived from these measured calibration gaps.

> ⚠️ **IMPORTANT**: P(Attack) = "probability that the next window is non-Benign." It does NOT mean the predicted attack class is correct with that probability.
