"""
benchmark_baseline.py
----------------------
Mandatory PS deliverable: side-by-side benchmark comparison between
the CHRONEX LSTM World Model and a Logistic Regression baseline.

Updated for 15-class multiclass setup.

Produces:
  - Console table: Accuracy, Macro F1, and Binary Attack F1
  - models/benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

SRC_DIR   = Path(__file__).resolve().parent
ROOT      = SRC_DIR.parent
DATA_DIR  = ROOT / "data" / "processed" / "model_ready"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "phase4" / "attack_classifier_protocolA_v1.pt"

# ---------------------------------------------------------------------------
# Re-use model definition
# ---------------------------------------------------------------------------

class AttackClassifierLSTM(nn.Module):
    def __init__(self, input_dim=71, hidden_dim=64, num_classes=15):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, benign_id: int, name: str) -> dict:
    acc   = accuracy_score(y_true, y_pred)
    mf1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Calculate implicit binary "Attack F1" for the PS requirement gate
    y_true_b = (y_true != benign_id).astype(int)
    y_pred_b = (y_pred != benign_id).astype(int)
    bin_f1   = f1_score(y_true_b, y_pred_b, zero_division=0)
    
    return {
        "model":      name,
        "accuracy":   round(float(acc), 4),
        "macro_f1":   round(float(mf1), 4),
        "attack_f1":  round(float(bin_f1), 4),
    }


def print_comparison_table(results: list[dict]) -> None:
    cols = ["model", "accuracy", "macro_f1", "attack_f1"]
    labels = {
        "model": "Model",
        "accuracy": "15-Class Acc",
        "macro_f1": "15-Class Macro F1",
        "attack_f1": "Implicit Attack F1",
    }
    widths = {
        "model": 28, "accuracy": 14, "macro_f1": 18, "attack_f1": 20,
    }

    header = "  ".join(labels[c].ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)
    print("\n" + "=" * len(sep))
    print("BENCHMARK COMPARISON — LSTM World Model vs Logistic Regression Baseline")
    print("=" * len(sep))
    print(header)
    print(sep)
    for r in results:
        row = "  ".join(str(r[c]).ljust(widths[c]) for c in cols)
        print(row)
    print("=" * len(sep))

    lstm_r = next(r for r in results if "LSTM" in r["model"])
    lr_r   = next(r for r in results if "Logistic" in r["model"])
    delta_f1  = lstm_r["attack_f1"]  - lr_r["attack_f1"]
    print(f"\n  LSTM improvement over LR baseline (Implicit Binary Attack F1): {delta_f1:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmark(args):
    data_dir   = Path(args.data_dir)
    model_path = Path(args.model_path)

    print("=" * 70)
    print("CHRONEX BENCHMARK — Multiclass LSTM vs Logistic Regression")
    print("=" * 70)

    # ---- Load dataset ----
    npz_path  = data_dir / "dataset.npz"
    meta_path = data_dir / "metadata.json"

    if not npz_path.exists():
        sys.exit(f"ERROR: dataset.npz not found at {npz_path}")

    data = np.load(npz_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    label_classes   = metadata["label_classes"]
    feature_columns = metadata["feature_columns_order"]
    benign_id = label_classes.index("Benign")
    num_classes = len(label_classes)

    # ---- Defensive: drop attack_ratio ----
    if "attack_ratio" in feature_columns:
        idx = feature_columns.index("attack_ratio")
        keep = [i for i in range(len(feature_columns)) if i != idx]
        X_train = X_train[:, :, keep]
        X_test  = X_test[:, :, keep]
        feature_columns = [feature_columns[i] for i in keep]

    n_features = X_train.shape[2]
    n_train    = len(X_train)
    n_test     = len(X_test)
    print(f"\n  Train: {n_train} sequences | Test: {n_test} sequences")
    print(f"  Features per window: {n_features} | Sequence length: {X_train.shape[1]}")
    print(f"  Total classes: {num_classes}")

    results = []

    # ====================================================================
    # 1. LOGISTIC REGRESSION BASELINE
    # ====================================================================
    print("\n" + "-" * 50)
    print("Training Logistic Regression baseline...")
    print("  (Sequences flattened to [N, 710] — no temporal structure)")

    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat  = X_test.reshape(n_test, -1)

    t0 = time.perf_counter()
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        random_state=42,
    )
    lr.fit(X_train_flat, y_train)
    lr_time = time.perf_counter() - t0
    
    # Save the LR model so Streamlit doesn't have to train it on the fly
    import joblib
    lr_model_path = MODEL_DIR / "lr_baseline_multiclass.joblib"
    joblib.dump(lr, lr_model_path)
    print(f"  Saved LR model to {lr_model_path}")

    lr_preds = lr.predict(X_test_flat)
    lr_metrics = compute_metrics(y_test, lr_preds, benign_id, "Logistic Regression (baseline)")
    lr_metrics["train_time_s"] = round(lr_time, 2)
    results.append(lr_metrics)
    print(f"  Done in {lr_time:.1f}s")

    # ====================================================================
    # 2. LSTM WORLD MODEL
    # ====================================================================
    print("-" * 50)
    print("Evaluating LSTM World Model...")

    if not model_path.exists():
        print(f"  WARNING: Checkpoint not found at {model_path}")
        print("  Run: python src/train_lstm.py   first.")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location="cpu")

    ckpt_features = checkpoint.get("input_size", n_features) if isinstance(checkpoint, dict) else n_features
    model = AttackClassifierLSTM(
        input_dim=ckpt_features,
        hidden_dim=64,
        num_classes=num_classes
    )
    # The new state dict doesn't have "model_state_dict" wrapping, it's just the dict directly
    # or if it does, handle both.
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test_t)
        lstm_preds = logits.argmax(dim=1).numpy()

    lstm_metrics = compute_metrics(y_test, lstm_preds, benign_id, "LSTM World Model (CHRONEX)")
    lstm_metrics["train_time_s"] = "N/A (pre-trained)"
    results.append(lstm_metrics)

    # ====================================================================
    # 3. COMPARISON TABLE
    # ====================================================================
    print_comparison_table(results)

    # ====================================================================
    # 4. SAVE JSON
    # ====================================================================
    out_path = MODEL_DIR / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_obj = {
        "description": "LSTM World Model vs Logistic Regression baseline (15-class) on CIC-IDS-2018",
        "dataset": str(data_dir),
        "n_train": n_train,
        "n_test": n_test,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(save_obj, f, indent=2)

    print(f"\nBenchmark results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHRONEX benchmark: Multiclass LSTM vs LR")
    parser.add_argument("--data-dir",   default=str(DATA_DIR))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    args = parser.parse_args()
    run_benchmark(args)
