"""
explain.py
----------
CHRONEX explainability module — SHAP-based feature attribution.

KEY FIX from original notebook:
    The original code built the SHAP GradientExplainer background by randomly
    sampling 50 sequences from X_TEST.  This is wrong on two counts:
    1. It contaminates the reference baseline with ~23% attack traffic,
       diluting SHAP values for attack-driving features.
    2. It violates data hygiene — the explanation reference should never
       touch test data.

    CORRECTED: background is drawn exclusively from Benign training sequences
    (X_train[y_train_binary == 0]).  This correctly represents "normal
    network behaviour" as the SHAP reference point.  SHAP then measures
    "how much does each feature push THIS prediction away from normal?"

Usage:
    from explain import build_explainer, explain_sequence, ShapResult

    explainer = build_explainer(model, X_train, y_train_binary)
    result = explain_sequence(explainer, model, sequence, feature_columns)
    print(result.top_features)   # list of (name, contribution) tuples
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import shap
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)   # suppress SHAP version warnings

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ShapResult:
    """SHAP explanation for a single prediction."""
    predicted_label: str                      # "Benign" or "Attack"
    confidence: float                         # softmax P(predicted class)
    attack_prob: float                        # P(Attack) specifically
    top_features: list[tuple[str, float]]     # (feature_name, shap_value), sorted by |value|
    all_shap_values: np.ndarray               # shape (n_features,), attack-class SHAP
    ranked_indices: np.ndarray                # feature indices sorted by |shap| descending


# ---------------------------------------------------------------------------
# Build explainer (correct Benign-only baseline)
# ---------------------------------------------------------------------------


def build_explainer(
    model: nn.Module,
    X_train: np.ndarray,
    y_train_binary: np.ndarray,
    n_background: int = 100,
    seed: int = 42,
) -> shap.GradientExplainer:
    """
    Build a SHAP GradientExplainer with a Benign-only background distribution.

    Args:
        model:           Trained model (nn.Module), must be in eval mode
        X_train:         (N_train, seq_len, n_features) training sequences
        y_train_binary:  (N_train,) binary labels (0=Benign, 1=Attack)
        n_background:    Number of Benign background samples (default 100)
        seed:            RNG seed for reproducibility

    Returns:
        shap.GradientExplainer ready for use

    Raises:
        ValueError if fewer than 10 Benign training sequences found
    """
    model.eval()

    benign_idx = np.where(y_train_binary == 0)[0]
    if len(benign_idx) < 10:
        raise ValueError(
            f"Only {len(benign_idx)} Benign training sequences found. "
            "Need at least 10 for a meaningful SHAP background."
        )

    rng = np.random.default_rng(seed)
    n_bg = min(n_background, len(benign_idx))
    chosen = rng.choice(benign_idx, size=n_bg, replace=False)

    background = torch.tensor(X_train[chosen], dtype=torch.float32)
    print(f"  [SHAP] Background: {n_bg} Benign training sequences "
          f"(from {len(benign_idx)} available)")

    explainer = shap.GradientExplainer(model, background)
    return explainer


# ---------------------------------------------------------------------------
# Explain a single prediction
# ---------------------------------------------------------------------------


def _extract_class_shap(shap_values: object, class_idx: int) -> np.ndarray:
    """
    Robustly extract SHAP values for a specific class index across SHAP versions.

    Different SHAP releases return GradientExplainer output in different
    shapes for multi-class models.  This function handles all known variants.

    Returns:
        class_shap: (n_features,) float array, averaged over sequence timesteps
    """
    arr = np.array(shap_values)

    if arr.ndim == 5:
        # Shape: (1, batch, seq_len, n_features, n_classes) — list-wrapped
        arr = arr[0]                          # -> (batch, seq_len, n_features, n_classes)
        return arr[0, :, :, class_idx].mean(axis=0)  # average over seq_len

    if arr.ndim == 4 and arr.shape[0] > 1 and arr.shape[0] == len(arr):
        # Shape: (n_classes, batch, seq_len, n_features)
        return arr[class_idx, 0, :, :].mean(axis=0)

    if arr.ndim == 4:
        # Shape: (batch, seq_len, n_features, n_classes)
        return arr[0, :, :, class_idx].mean(axis=0)

    if arr.ndim == 3:
        # Shape: (batch, seq_len, n_features) — binary model, single output
        # If it's a single output array, we assume it's the target logit
        return arr[0].mean(axis=0)

    raise ValueError(
        f"Unexpected shap_values shape: {arr.shape}. "
        f"Check your installed SHAP version with: import shap; print(shap.__version__)"
    )


def explain_sequence(
    explainer: shap.GradientExplainer,
    model: nn.Module,
    sequence: np.ndarray,
    feature_columns: list[str],
    label_classes: list[str],
    benign_idx: int = 0,
    top_k: int = 8,
) -> ShapResult:
    """
    Compute SHAP explanation for a single sequence.

    Args:
        explainer:       Pre-built GradientExplainer (from build_explainer)
        model:           Trained model (in eval mode)
        sequence:        (seq_len, n_features) input sequence
        feature_columns: Feature names
        label_classes:   List of all 15 class names
        benign_idx:      Index of 'Benign' class
        top_k:           How many top features to return

    Returns:
        ShapResult with attribution details for the predicted class
    """
    model.eval()
    sequence = np.array(sequence, dtype=np.float32)

    if sequence.ndim != 2:
        raise ValueError(f"sequence must be (seq_len, n_features), got {sequence.shape}")

    x = torch.tensor(sequence[None], dtype=torch.float32)  # add batch dim

    # Get prediction
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).numpy()[0]

    pred_class = int(np.argmax(probs))
    predicted_label = label_classes[pred_class]
    confidence = float(probs[pred_class])
    attack_prob = 1.0 - float(probs[benign_idx])

    # Get SHAP values
    shap_values = explainer.shap_values(x)
    
    # We explain the predicted class
    class_shap = _extract_class_shap(shap_values, pred_class)

    # Rank by absolute contribution
    ranked_idx = np.argsort(np.abs(class_shap))[::-1][:top_k]

    top_features = []
    for i in ranked_idx:
        name = feature_columns[i] if i < len(feature_columns) else f"feat_{i}"
        top_features.append((name, float(class_shap[i])))

    return ShapResult(
        predicted_label=predicted_label,
        confidence=confidence,
        attack_prob=attack_prob,
        top_features=top_features,
        all_shap_values=class_shap,
        ranked_indices=ranked_idx,
    )


# ---------------------------------------------------------------------------
# Format for display
# ---------------------------------------------------------------------------


def format_shap_report(result: ShapResult) -> str:
    """Format ShapResult as a human-readable text block."""
    lines = [
        f"Prediction:  {result.predicted_label}  (confidence: {result.confidence*100:.1f}%)",
        f"P(Attack):   {result.attack_prob*100:.1f}%",
        "",
        f"Top {len(result.top_features)} features driving this prediction:",
        "-" * 55,
    ]
    for name, val in result.top_features:
        direction = "→ ATTACK" if val > 0 else "→ BENIGN"
        bar_len = int(abs(val) * 40 / 0.5)  # normalize bar to 40 chars
        bar = ("█" * min(bar_len, 40)).ljust(40)
        lines.append(f"  {name:30s}  {val:+.4f}  {direction}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    ROOT       = Path(__file__).resolve().parent.parent
    DATA_DIR   = ROOT / "data" / "processed" / "model_ready"
    MODEL_DIR  = ROOT / "models"

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_lstm import MulticlassAttackForecastLSTM

    model_path = MODEL_DIR / "trained_lstm_multiclass.pt"
    if not model_path.exists():
        print("Checkpoint not found. Run: python src/train_lstm.py  first.")
        sys.exit(1)

    ckpt = torch.load(model_path, map_location="cpu")
    model = MulticlassAttackForecastLSTM(
        input_size=ckpt["input_size"],
        hidden_size=ckpt.get("hidden_size", 64),
        num_layers=ckpt.get("num_layers", 2),
        num_classes=ckpt.get("num_classes", 15),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data = np.load(DATA_DIR / "dataset.npz")
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    meta = json.load(open(DATA_DIR / "metadata.json"))
    label_classes   = meta["label_classes"]
    feature_columns = meta["feature_columns_order"]
    benign_id = label_classes.index("Benign")

    print("Building explainer with Benign-only background...")
    explainer = build_explainer(model, X_train, y_train, n_background=50)

    # Pick a known attack sequence for demo
    attack_test_idx = np.where(y_test != benign_id)[0][0]
    seq = X_test[attack_test_idx]

    print(f"\nExplaining attack sequence #{attack_test_idx}...")
    result = explain_sequence(explainer, model, seq, feature_columns, label_classes, benign_idx, top_k=8)

    print("\n" + format_shap_report(result))
    print("\n✅ Explainability module OK")
