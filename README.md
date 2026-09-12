# Cybersecurity-project
# Network Attack Forecasting — SIH Problem Statement 26153

AI-based system that forecasts whether the next 30-second window of network traffic
is Benign or an Attack in progress, using an LSTM trained on the last 10 windows
(5 minutes) of traffic behavior. Includes SHAP-based explainability and a heuristic
MITRE ATT&CK stage-mapping layer.

## Repository structure

```
data/processed/model_ready/   - preprocessed dataset (windowed sequences), scaler, label encoder, metadata
models/                       - trained LSTM weights (trained_lstm_binary.pt)
notebooks/                    - full training + SHAP + MITRE analysis notebook (Chronex.ipynb)
src/                          - standalone scripts (training, explainability, dashboard)
```

## Setup

```bash
pip install -r requirements.txt
```

## How to run

**1. Train the model (optional — a trained checkpoint is already included in `models/`):**
```bash
python src/train_lstm.py
```

**2. Run the interactive dashboard (renders inside a Jupyter/Colab notebook, no server needed):**
Open `notebooks/Chronex.ipynb` in Jupyter or Google Colab and run all cells.

**3. Explainability + MITRE stage mapping:**
Included in the same notebook — SHAP feature attribution and rule-based MITRE ATT&CK
stage mapping run automatically for any "Attack" prediction.

## Dataset

CSE-CIC-IDS2018, processed into 30-second traffic-state windows, sequenced into
10-window (5-minute) sequences, used to forecast the label of the next window.

## Current build scope

- Binary (Attack vs Benign) forecasting: working end to end
- SHAP explainability: working
- Heuristic MITRE ATT&CK stage mapping (SHAP-driven): working
- Multiclass attack-type classification and true multi-step (K-step) forward rollout: next-phase work

## Requirements

See `requirements.txt`. Core dependencies: torch, numpy, scikit-learn, shap, joblib.
