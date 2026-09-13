# Phase 11: Attack Replay / SIH Demo Mode

## Overview
The SIH Demo Mode provides a judge-friendly chronological replay of Protocol A test sequences, showing CHRONEX forecasting future attack states in real time.

Toggle: **🎬 SIH Demo / Replay Mode** in the sidebar.

## Model Artifacts Used
| Artifact | Path |
|---|---|
| Attack Classifier | `models/phase4/attack_classifier_protocolA_v1.pt` |
| Direct Future-State Decoder | `models/phase4/state_transition_direct_v1.pt` |
| Scaler | `data/processed/model_ready_v2_primary/scaler.joblib` |
| Feature Order | 71 features from `metadata.json` |

## Navigation
- **Slider**: Select any test sequence index (0 to N-1)
- **⏮ Prev / Next ⏭**: Step one sequence at a time
- **↺ Reset**: Return to index 0

## Observed vs Post-Hoc Ground Truth
| Information | Source | Labelling |
|---|---|---|
| Observed context W(t-9)…W(t) | Protocol A X_test | Used for inference |
| P(Attack) K=1..10 | Model softmax output | Displayed as forecast |
| Predicted class | argmax of softmax | Displayed as forecast |
| Actual future class | ylab_test | Shown as **Ground Truth (post-hoc)** |

## Reused Phase Metrics
| Phase | Metric | Demo Section |
|---|---|---|
| Phase 8 | Escalation Score, Trajectory | Attack Risk Trajectory |
| Phase 7 | Lead Time, Early Warning | Forecast Lead Time |
| Phase 9 | Temporal Evidence Score, What Changed | SHAP + Evidence panel |
| Phase 10 | Forecast Reliability label | Current State card |

## Status Banner Logic
| Banner | Condition |
|---|---|
| EARLY WARNING | `valid_k_demo is not None` (model predicts attack before onset) |
| ESCALATING RISK | `traj_status == 'ESCALATING'` |
| WATCH | `P(Attack at K=1) >= 0.30` |
| NORMAL | All others |

## Known Limitations
- Exact timestamps are not stored in dataset.npz; windows shown as relative offsets (+30s per step).
- SHAP computation adds ~2s latency per sequence when the attack classifier predicts a non-Benign class.
- Forecast uncertainty intervals are not modeled (Phase 4J limitation).
- Replay is Protocol A only; Protocol B requires separate scaler inversion step.
