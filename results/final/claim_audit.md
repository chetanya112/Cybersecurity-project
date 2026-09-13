# CHRONEX — Claim Audit
## SIH Presentation Claim Review

All claims are classified against measured, validated results only.

---

## SUPPORTED Claims

| Claim | Evidence |
|---|---|
| "CHRONEX forecasts future 71-dimensional network states up to 10 windows (5 minutes) ahead." | Direct Decoder: MAE=0.391, RMSE=0.800 at K=1–10 (Protocol A). Outperforms persistence baseline at all horizons. |
| "The downstream 15-class attack classifier achieves Attack F1 = 0.8604 at K=10 on Protocol A." | Measured and documented in Phase 4 results. |
| "CHRONEX predicts future attack risk as P(Attack) = 1 − P(Benign)." | This is the exact definition used throughout. |
| "CHRONEX identifies the currently predicted attack class from 14 attack families + Benign." | 15-class LSTM classifier with K=1 achieving Attack F1 ≈ 0.86. |
| "CHRONEX provides an Escalation Score in [−1, +1] indicating whether future risk is increasing." | Phase 8. Derived from linear slope of P(Attack) over K=1..10. Formula explicitly documented. |
| "The Direct Future-State Decoder outperforms persistence in RMSE at all tested horizons." | Phase 4 and Phase 6 (Protocol B). |
| "CHRONEX maintains high-confidence attack precision of 98.2% at P(Attack) ≥ 0.90." | Phase 10. ECE gap = 0.010 in the 0.90–1.00 bin. |
| "CHRONEX identifies the top features associated with current attack risk via SHAP." | Existing Phase 5 SHAP implementation. Background = benign training sequences. |
| "CHRONEX maps top SHAP features to MITRE ATT&CK tactics via a heuristic scoring function." | Phase 5 / mitre.py. Labelled as heuristic, not ground-truth attribution. |
| "The Recursive LSTM is retained as a benchmark; the Direct Decoder is selected for production." | Phase 4 design decision, documented. Lower RMSE and inference stability. |
| "Protocol B (unseen-attack) generalization shows consistent Attack F1 above 0.79." | Phase 6 results: Attack F1 = 0.885 at K=1, 0.795 at K=10 on Protocol B test. |

---

## PARTIALLY SUPPORTED Claims

| Claim | Qualification |
|---|---|
| "CHRONEX provides early warning of upcoming attacks." | **2 of 4 eligible test episodes** showed early-warning detection at 30-second median lead time. Not validated across a large continuous test window population. Use: *"CHRONEX demonstrated measurable early-warning capability in 2 of 4 eligible test episodes (50% detection rate) with 30-second median lead time in this evaluation."* |
| "CHRONEX explains why risk increased." | The Temporal Evidence Score ranks features by combined SHAP magnitude and recent standardized change. This is a **ranking mechanism**, not causal attribution. Use: *"CHRONEX identifies features with strong recent changes that are also associated with the current model prediction."* |
| "CHRONEX P(Attack) probabilities are calibrated." | Calibration is **bimodal**: well-calibrated at extremes (P≥0.90: gap=0.010; P<0.10: gap=0.037) but poorly calibrated in mid-range (0.10–0.90: gaps 0.063–0.474). Use: *"CHRONEX probability values are well-calibrated in the high-confidence regime (P≥0.90), but exhibit overconfidence in intermediate probability ranges."* |
| "CHRONEX generalizes to unseen attack patterns." | Protocol B uses different attack days with some novel traffic. Attack F1 drops from 0.86 to 0.79 at K=10. Some generalization is demonstrated, but exact-class recall on truly novel attack families was not separately isolated. |

---

## NOT SUPPORTED Claims

| Claim | Why Not Supported |
|---|---|
| "CHRONEX provides reliable 5-minute early warning." | Only 2/4 episodes detected; lead time was 30 seconds (1 window), not 5 minutes. |
| "CHRONEX provides X minutes of early warning." | Do not specify a minute value. The only measured lead time is 30 seconds in 2 episodes. |
| "The Temporal Evidence Score identifies the cause of the attack." | This is a ranking heuristic, not causal inference. |
| "P(Attack) = 98% means there is a 98% chance the predicted attack class is correct." | P(Attack) measures non-Benign probability only. Exact class prediction is separate (15-class). |
| "CHRONEX predicts attacks with world-model accuracy." | The system uses a learned future-state approximation with measured error (RMSE ≈ 0.80). It is not a perfect world model. |
| "CHRONEX has uncertainty quantification." | Phase 4J limitation: uncertainty/confidence intervals are NOT modeled. |

---

## Summary for SIH Presentation

**Safe to say:**
- CHRONEX predicts future 71-D network states up to 5 minutes ahead using a trained LSTM encoder-decoder.
- The downstream attack classifier achieves Attack F1 = 0.86 on Protocol A test and 0.88 on Protocol B test (K=1).
- P(Attack) values are well-calibrated at high confidence (P≥0.90), with 98.2% precision.
- The system provides an interpretable Escalation Score and temporal evidence ranking for SOC decision support.
- Early-warning capability was demonstrated in 2 of 4 eligible test episodes.

**Do not say:**
- CHRONEX guarantees early warning.
- CHRONEX provides causal explanations.
- P(Attack) = X% certainty of the specific attack class.
- The system has uncertainty quantification.
