# Phase 7: Forecast Lead Time Report

## Methodology
Forecast lead time measures how early CHRONEX predicts an impending attack. 
A forecast is strictly penalized: Ground truth future states are never passed to the model. Lead time is calculated as (Actual Attack Onset - Forecast Timestamp) * 30 seconds. Forecasts are valid only if they occur BEFORE the attack onset, and the target window falls inside the actual attack episode.

Total Episodes: 4
Eligible for Early Warning: 4
Excluded (Attack already active): 0

### TABLE 1 — HORIZON PERFORMANCE (Binary)
| Horizon | Forecast Seconds | Episodes Evaluated | Episodes Detected | Detection Rate | Mean Lead Time | Median Lead Time | Max Lead Time |
|---|---|---|---|---|---|---|---|
| K=1 | 30s | 4 | 2 | 50.0% | 30.0s | 30.0s | 30.0s |
| K=3 | 90s | 4 | 2 | 50.0% | 30.0s | 30.0s | 30.0s |
| K=5 | 150s | 4 | 2 | 50.0% | 30.0s | 30.0s | 30.0s |
| K=10 | 300s | 4 | 2 | 50.0% | 30.0s | 30.0s | 30.0s |

### TABLE 2 — RISK THRESHOLD
| P(Attack) Threshold | Episodes Detected | Detection Rate | Mean Lead Time | Median Lead Time | Max Lead Time |
|---|---|---|---|---|---|
| 0.50 | 2 | 50.0% | 30.0s | 30.0s | 30.0s |
| 0.70 | 2 | 50.0% | 30.0s | 30.0s | 30.0s |
| 0.80 | 2 | 50.0% | 30.0s | 30.0s | 30.0s |
| 0.90 | 2 | 50.0% | 30.0s | 30.0s | 30.0s |
| 0.95 | 2 | 50.0% | 30.0s | 30.0s | 30.0s |

### TABLE 3 — ATTACK-VS-BENIGN VS EXACT CLASS
| Horizon | Binary Attack Detection Rate | Exact Class Detection Rate | Mean Binary Lead Time | Mean Exact-Class Lead Time |
|---|---|---|---|---|
| K=1 | 50.0% | 25.0% | 30.0s | 30.0s |
| K=3 | 50.0% | 25.0% | 30.0s | 30.0s |
| K=5 | 50.0% | 25.0% | 30.0s | 30.0s |
| K=10 | 50.0% | 25.0% | 30.0s | 30.0s |
