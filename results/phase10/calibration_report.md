# Phase 10: Attack Probability Calibration & Reliability

## Definition
P(Attack) = 1 - P(Benign class), derived from the 15-class softmax of the production K=1 classifier.

Ground truth is used **only after** inference to compare predicted probabilities against observed attack frequencies. No calibration transform is fit on the test set.

## ECE Formula
```
ECE = sum_b (N_b / N) * |mean_pred_b - obs_freq_b|    (10 fixed equal-width bins over [0,1])
```

## Protocol A

| Metric | Value |
|---|---|
| Samples | 1190 |
| Brier Score | 0.0624 |
| ECE | 0.0773 |
| Mean Abs Calibration Error | 0.1972 |
| P>=0.90 count | 165 |
| P>=0.90 precision | 0.9818 |
| P>=0.90 recall | 0.7074 |
| Mean P(Attack) correct | 0.2166 |
| Mean P(Attack) incorrect | 0.2083 |

### Calibration Bins

| Bin | Count | Mean Predicted | Observed Freq | Gap |
|---|---|---|---|---|
| 0.0-0.1 | 801 | 0.006 | 0.044 | 0.037 |
| 0.1-0.2 | 32 | 0.157 | 0.094 | 0.063 |
| 0.2-0.3 | 42 | 0.261 | 0.095 | 0.165 |
| 0.3-0.4 | 48 | 0.351 | 0.062 | 0.289 |
| 0.4-0.5 | 50 | 0.457 | 0.060 | 0.397 |
| 0.5-0.6 | 31 | 0.539 | 0.065 | 0.474 |
| 0.6-0.7 | 5 | 0.622 | 0.400 | 0.222 |
| 0.7-0.8 | 5 | 0.743 | 1.000 | 0.257 |
| 0.8-0.9 | 11 | 0.852 | 0.909 | 0.057 |
| 0.9-1.0 | 165 | 0.992 | 0.982 | 0.010 |

## Protocol B

| Metric | Value |
|---|---|
| Samples | 3185 |
| Brier Score | 0.0621 |
| ECE | 0.0354 |
| Mean Abs Calibration Error | 0.0832 |
| P>=0.90 count | 934 |
| P>=0.90 precision | 0.9882 |
| P>=0.90 recall | 0.7916 |
| Mean P(Attack) correct | 0.3675 |
| Mean P(Attack) incorrect | 0.3482 |

### Calibration Bins

| Bin | Count | Mean Predicted | Observed Freq | Gap |
|---|---|---|---|---|
| 0.0-0.1 | 1717 | 0.011 | 0.043 | 0.031 |
| 0.1-0.2 | 77 | 0.149 | 0.182 | 0.033 |
| 0.2-0.3 | 84 | 0.251 | 0.226 | 0.025 |
| 0.3-0.4 | 83 | 0.351 | 0.253 | 0.098 |
| 0.4-0.5 | 122 | 0.457 | 0.377 | 0.080 |
| 0.5-0.6 | 122 | 0.544 | 0.377 | 0.167 |
| 0.6-0.7 | 31 | 0.646 | 0.419 | 0.227 |
| 0.7-0.8 | 6 | 0.747 | 0.667 | 0.080 |
| 0.8-0.9 | 9 | 0.860 | 0.778 | 0.082 |
| 0.9-1.0 | 934 | 0.996 | 0.988 | 0.008 |

## Interpretation
P(Attack) is the model's estimated probability that the next network window is non-Benign. It does NOT mean the predicted attack class is correct with that probability.
