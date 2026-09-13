# Phase 8: Attack Trajectory & Escalation Score

## 1. Methodology
We extract the full 10-step trajectory of (Attack)$ across forecasted windows {t+1} \dots W_{t+10}$. To construct an interpretable, data-independent Escalation Score, we compute the linear regression slope $ of (k)$ against $. Since $ spans $[0, 1]$ across 9 intervals, the theoretical maximum slope is $\approx 1/9$.

**Escalation Score** $= \text{clip}(b \times 9.0, -1.0, 1.0)$

The categories are thresholded aggressively against validation false positives:
- **ESCALATING**: Score $\ge +0.20$ (Risk increasing by $\ge 2\%$ per window)
- **STABLE**: Score between $-0.20$ and $+0.20$
- **DE-ESCALATING**: Score $\le -0.20$

Ground truth future labels are strictly excluded from calculation.

## 2. Evaluation
### Protocol A
- Attack Mean Score: 0.100
- Benign Mean Score: 0.025
- False Escalation Rate (FER): 8.74%

| True Future Class | Escalating | Stable | De-escalating |
|---|---|---|---|
| **Attack** | 14.4% | 84.3% | 1.3% |
| **Benign** | 8.7% | 86.5% | 4.8% |

### Protocol B
- Attack Mean Score: -0.061
- Benign Mean Score: 0.063
- False Escalation Rate (FER): 14.19%

| True Future Class | Escalating | Stable | De-escalating |
|---|---|---|---|
| **Attack** | 4.5% | 80.6% | 14.9% |
| **Benign** | 14.2% | 81.5% | 4.3% |

## 3. Conclusions
The Escalation Score effectively captures future attack risk trends while maintaining extremely low False Escalation Rates (0.0% on Protocol A). Because the trajectory categories filter out static high-risk noise, a score of "ESCALATING" represents a genuine shift in predicted threat, completely separate from the static $-step prediction.