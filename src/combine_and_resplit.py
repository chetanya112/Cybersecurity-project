"""
CHRONEX dataset consolidation and stratified resplit pipeline.

Purpose
-------
1. Rebuild the current sequence pool from cleaned CSV files.
2. Optionally append future RAW/unscaled sequence .npz files.
3. Pool all sequences before splitting.
4. Create a stratified 70/15/15 train/validation/test split.
5. Guarantee that every canonical class with >= 3 samples is represented
   in train, validation, and test.
6. Derive clipping/fill statistics from TRAIN ONLY.
7. Fit StandardScaler on TRAIN ONLY.
8. Save corrected artifacts to model_ready_v2 without modifying model_ready.

Canonical model contract
-------------------------
- Window size: 30 seconds
- Sequence length: 10 windows
- Features per window: 71
- Features = 70 traffic features + flow_count
- attack_ratio is NOT a model input
- Existing 15-class label encoder is reused
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Make src/ importable when this file is executed as:
# python src\combine_and_resplit.py
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_sequences import (  # noqa: E402
    SEQUENCE_LENGTH,
    WINDOW_SECONDS,
    build_sequences,
    build_state_vectors,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CURRENT_MODEL_DIR = PROCESSED_DIR / "model_ready"
INCOMING_DIR = PROCESSED_DIR / "incoming_raw"
OUTPUT_DIR = PROCESSED_DIR / "model_ready_v2"

# ---------------------------------------------------------------------------
# Immutable project contract
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

RANDOM_STATE = 42

EXPECTED_N_FEATURES = 71
EXPECTED_SEQUENCE_LENGTH = 10
EXPECTED_WINDOW_SECONDS = 30
EXPECTED_N_CLASSES = 15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fail(message: str) -> None:
    """Raise a clear pipeline error."""
    raise RuntimeError(message)


def require_file(path: Path) -> None:
    """Require a file to exist."""
    if not path.exists():
        fail(f"Required file does not exist: {path}")


def validate_X(name: str, X: np.ndarray) -> None:
    """Validate the complete X tensor contract."""
    X = np.asarray(X)

    if X.ndim != 3:
        fail(
            f"{name} must have shape (N, seq_len, features), "
            f"but got {X.shape}"
        )

    if X.shape[1] != EXPECTED_SEQUENCE_LENGTH:
        fail(
            f"{name} has sequence length {X.shape[1]}; "
            f"expected {EXPECTED_SEQUENCE_LENGTH}"
        )

    if X.shape[2] != EXPECTED_N_FEATURES:
        fail(
            f"{name} has {X.shape[2]} features; "
            f"expected {EXPECTED_N_FEATURES}"
        )

    if not np.issubdtype(X.dtype, np.number):
        fail(f"{name} must contain numeric values; dtype={X.dtype}")

    if not np.isfinite(X).all():
        fail(f"{name} contains non-finite values")


def validate_y(name: str, y: np.ndarray, n_samples: int) -> None:
    """Validate a label vector."""
    y = np.asarray(y)

    if y.ndim != 1:
        fail(f"{name} must be 1-D, but got shape {y.shape}")

    if len(y) != n_samples:
        fail(
            f"{name} length {len(y)} does not match "
            f"X length {n_samples}"
        )


def class_counts(y: np.ndarray, encoder) -> Dict[str, int]:
    """Return class counts using canonical encoder names."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)

    result: Dict[str, int] = {}
    for class_id, count in zip(classes, counts):
        class_id = int(class_id)
        result[str(encoder.classes_[class_id])] = int(count)

    # Include zero-count canonical classes for complete metadata.
    for class_id, label in enumerate(encoder.classes_):
        result.setdefault(str(label), 0)

    return result


# ---------------------------------------------------------------------------
# Existing dataset contract
# ---------------------------------------------------------------------------


def load_contract():
    """
    Load the existing canonical label encoder and metadata.

    The existing encoder is reused. No new label vocabulary is fitted.
    """
    encoder_path = CURRENT_MODEL_DIR / "label_encoder.joblib"
    metadata_path = CURRENT_MODEL_DIR / "metadata.json"

    require_file(encoder_path)
    require_file(metadata_path)

    encoder = joblib.load(encoder_path)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_order = metadata.get("feature_columns_order", [])

    if len(feature_order) != EXPECTED_N_FEATURES:
        fail(
            f"metadata.json reports {len(feature_order)} features; "
            f"expected {EXPECTED_N_FEATURES}"
        )

    if metadata.get("sequence_length") != EXPECTED_SEQUENCE_LENGTH:
        fail(
            "metadata sequence_length does not match the required "
            f"{EXPECTED_SEQUENCE_LENGTH}"
        )

    if metadata.get("window_seconds") != EXPECTED_WINDOW_SECONDS:
        fail(
            "metadata window_seconds does not match the required "
            f"{EXPECTED_WINDOW_SECONDS}"
        )

    if len(encoder.classes_) != EXPECTED_N_CLASSES:
        fail(
            f"Existing label encoder has {len(encoder.classes_)} classes; "
            f"expected {EXPECTED_N_CLASSES}"
        )

    metadata_labels = metadata.get("label_classes")
    if metadata_labels is not None:
        encoder_labels = [str(x) for x in encoder.classes_]
        metadata_labels = [str(x) for x in metadata_labels]

        if encoder_labels != metadata_labels:
            fail(
                "label_encoder.joblib and metadata.json contain different "
                "class orders."
            )

    return encoder, metadata, feature_order


def encode_labels(
    y: np.ndarray,
    encoder,
    source: str,
) -> np.ndarray:
    """
    Encode labels using the existing canonical encoder.

    Accepts:
    - integer class IDs
    - canonical class-name strings
    """
    y = np.asarray(y)

    if y.ndim != 1:
        fail(f"{source} labels must be 1-D; got {y.shape}")

    if np.issubdtype(y.dtype, np.integer):
        y_int = y.astype(np.int64, copy=False)

        allowed = set(range(len(encoder.classes_)))
        observed = set(np.unique(y_int).tolist())
        unknown = sorted(observed - allowed)

        if unknown:
            fail(
                f"{source} contains unknown class IDs: {unknown}. "
                "The existing label encoder must be reused."
            )

        return y_int

    try:
        return encoder.transform(y.astype(str)).astype(np.int64)
    except ValueError as exc:
        fail(
            f"{source} contains labels outside the existing encoder: {exc}"
        )


# ---------------------------------------------------------------------------
# Build current RAW sequence pool
# ---------------------------------------------------------------------------


def build_current_raw_pool(
    cleaned_dir: Path,
    encoder,
    feature_order: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Rebuild sequences directly from cleaned CSVs.

    This avoids inverse-transforming the old scaled dataset, because the old
    pipeline clipped values before scaling.
    """
    files = sorted(cleaned_dir.glob("cleaned_*.csv"))

    if not files:
        fail(f"No cleaned_*.csv files found in {cleaned_dir}")

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    audit: List[Dict] = []

    for path in files:
        print(f"Processing {path.name} ...")

        try:
            df = pd.read_csv(path, parse_dates=["Timestamp"])
        except Exception as exc:
            fail(f"Could not read {path}: {exc}")

        if df.empty:
            print("  Skipping empty file.")
            continue

        required_columns = {"Timestamp", "Label"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            fail(
                f"{path.name} is missing required columns: "
                f"{missing_columns}"
            )

        state_df, feature_cols = build_state_vectors(df)

        actual_order = list(feature_cols) + ["flow_count"]

        if actual_order != feature_order:
            fail(
                f"Feature-order mismatch in {path.name}.\n"
                f"Expected: {feature_order}\n"
                f"Found:    {actual_order}"
            )

        if "attack_ratio" in actual_order:
            fail(
                f"attack_ratio unexpectedly appears in model feature order "
                f"for {path.name}. It is label-derived and must not be used."
            )

        if len(actual_order) != EXPECTED_N_FEATURES:
            fail(
                f"{path.name} produces {len(actual_order)} model features; "
                f"expected {EXPECTED_N_FEATURES}"
            )

        X, y_text = build_sequences(
            state_df,
            feature_cols,
            seq_len=EXPECTED_SEQUENCE_LENGTH,
        )

        if len(X) == 0:
            print("  No sequences produced.")
            continue

        y = encode_labels(y_text, encoder, f"{path.name}::y")

        validate_X(f"{path.name}::X", X)
        validate_y(f"{path.name}::y", y, len(X))

        X = X.astype(np.float32, copy=False)
        y = y.astype(np.int64, copy=False)

        xs.append(X)
        ys.append(y)

        audit.append(
            {
                "source": path.name,
                "flows": int(len(df)),
                "windows": int(len(state_df)),
                "sequences": int(len(X)),
            }
        )

        print(
            f"  flows={len(df):,}, "
            f"windows={len(state_df):,}, "
            f"sequences={len(X):,}"
        )

    if not xs:
        fail("No current sequences were produced from cleaned CSV files.")

    X_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)

    validate_X("current X pool", X_all)
    validate_y("current y pool", y_all, len(X_all))

    return X_all, y_all, audit


# ---------------------------------------------------------------------------
# Optional incoming RAW sequence pool
# ---------------------------------------------------------------------------


def load_incoming_raw(
    incoming_dir: Path,
    encoder,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Load optional future teammate NPZ files.

    Required NPZ arrays:
        X: (N, 10, 71), RAW/unscaled values
        y: (N,), canonical labels or existing integer IDs
    """
    if not incoming_dir.exists():
        return [], [], []

    files = sorted(incoming_dir.glob("*.npz"))

    if not files:
        return [], [], []

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    names: List[str] = []

    for path in files:
        print(f"Loading incoming raw file: {path.name}")

        try:
            with np.load(path, allow_pickle=True) as data:
                if "X" not in data.files or "y" not in data.files:
                    fail(
                        f"{path.name} must contain arrays named 'X' and 'y'."
                    )

                X = np.asarray(data["X"])
                y = np.asarray(data["y"])
        except RuntimeError:
            raise
        except Exception as exc:
            fail(f"Could not load {path}: {exc}")

        validate_X(f"{path.name}::X", X)
        validate_y(f"{path.name}::y", y, len(X))

        # Incoming X must be raw/unscaled. We cannot reliably infer whether
        # arbitrary data is scaled, so enforce the documented contract.
        y = encode_labels(y, encoder, f"{path.name}::y")

        X = X.astype(np.float32, copy=False)
        y = y.astype(np.int64, copy=False)

        xs.append(X)
        ys.append(y)
        names.append(path.name)

        print(f"  X={X.shape}, y={y.shape}")

    return xs, ys, names


# ---------------------------------------------------------------------------
# Robust stratified split
# ---------------------------------------------------------------------------


def _allocate_class_counts(
    n: int,
    fractions: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """
    Allocate one class across train/val/test.

    For classes with >= 3 samples, guarantee at least one sample in every
    split while keeping the allocation as close as possible to 70/15/15.
    """
    if n < 3:
        fail(
            "A class has fewer than 3 sequences, so it cannot be represented "
            "in train, validation, and test."
        )

    desired = np.asarray(fractions, dtype=float) * n

    counts = np.floor(desired).astype(int)

    # Guarantee one sample per split.
    counts = np.maximum(counts, 1)

    # Adjust until the sum is exactly n.
    while counts.sum() < n:
        remainders = desired - counts
        index = int(np.argmax(remainders))
        counts[index] += 1

    while counts.sum() > n:
        # Remove from the split with the largest excess, but never below 1.
        excess = counts - desired
        candidates = np.where(counts > 1)[0]

        if len(candidates) == 0:
            fail(f"Could not allocate {n} samples across three splits.")

        index = int(candidates[np.argmax(excess[candidates])])
        counts[index] -= 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
):
    """
    Perform a reproducible class-stratified 70/15/15 split.

    This implementation allocates each class independently, guaranteeing
    class presence in all three splits whenever that class has >= 3 samples.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    validate_X("split input X", X)
    validate_y("split input y", y, len(X))

    classes, counts = np.unique(y, return_counts=True)

    too_small = {
        int(class_id): int(count)
        for class_id, count in zip(classes, counts)
        if count < 3
    }

    if too_small:
        fail(
            "Cannot represent every class in train/val/test because these "
            f"classes have fewer than 3 sequences: {too_small}"
        )

    rng = np.random.default_rng(RANDOM_STATE)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for class_id, class_count in zip(classes, counts):
        class_id = int(class_id)
        class_indices = np.flatnonzero(y == class_id)
        rng.shuffle(class_indices)

        n_train, n_val, n_test = _allocate_class_counts(
            int(class_count),
            (TRAIN_FRAC, VAL_FRAC, TEST_FRAC),
        )

        train_indices.extend(class_indices[:n_train].tolist())
        val_indices.extend(
            class_indices[n_train:n_train + n_val].tolist()
        )
        test_indices.extend(
            class_indices[n_train + n_val:n_train + n_val + n_test].tolist()
        )

    # Shuffle within each split so classes are not grouped together.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test


def require_all_classes(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    encoder,
) -> None:
    """Fail unless every canonical class appears in all three splits."""
    expected = set(range(len(encoder.classes_)))

    for split_name, labels in (
        ("TRAIN", y_train),
        ("VAL", y_val),
        ("TEST", y_test),
    ):
        present = set(np.unique(labels).tolist())
        missing = sorted(expected - present)

        if missing:
            names = [str(encoder.classes_[i]) for i in missing]
            fail(
                f"{split_name} is missing classes: {names}"
            )

        print(
            f"{split_name}: "
            f"{len(present)}/{len(expected)} classes present"
        )


# ---------------------------------------------------------------------------
# Train-only preprocessing
# ---------------------------------------------------------------------------


def preprocess_train_only(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
):
    """
    Apply preprocessing using statistics learned from TRAIN ONLY.

    Steps:
        1. Convert non-finite values to NaN.
        2. Calculate 1st/99th percentile and median from TRAIN ONLY.
        3. Apply train-derived clipping and median fill to all splits.
        4. Fit StandardScaler on TRAIN ONLY.
        5. Transform all splits with the same scaler.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    validate_X("preprocessing X_train", X_train)
    validate_X("preprocessing X_val", X_val)
    validate_X("preprocessing X_test", X_test)

    n_features = X_train.shape[-1]

    def replace_nonfinite(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[~np.isfinite(X)] = np.nan
        return X

    X_train = replace_nonfinite(X_train)
    X_val = replace_nonfinite(X_val)
    X_test = replace_nonfinite(X_test)

    flat_train = X_train.reshape(-1, n_features)

    lower = np.nanpercentile(flat_train, 1, axis=0)
    upper = np.nanpercentile(flat_train, 99, axis=0)
    median = np.nanmedian(flat_train, axis=0)

    if not (
        np.isfinite(lower).all()
        and np.isfinite(upper).all()
        and np.isfinite(median).all()
    ):
        fail(
            "Train-derived preprocessing statistics contain non-finite "
            "values."
        )

    if np.any(lower > upper):
        fail("Some train-derived lower clipping bounds exceed upper bounds.")

    def clip_fill(X: np.ndarray) -> np.ndarray:
        shape = X.shape
        flat = X.reshape(-1, n_features).copy()

        for j in range(n_features):
            col = flat[:, j]
            nan_mask = np.isnan(col)

            if nan_mask.any():
                col[nan_mask] = median[j]

            flat[:, j] = np.clip(col, lower[j], upper[j])

        return flat.reshape(shape)

    X_train = clip_fill(X_train)
    X_val = clip_fill(X_val)
    X_test = clip_fill(X_test)

    # IMPORTANT: fit ONLY on training data.
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def scale(X: np.ndarray) -> np.ndarray:
        shape = X.shape
        scaled = scaler.transform(
            X.reshape(-1, n_features)
        ).reshape(shape)

        return scaled.astype(np.float32)

    X_train_s = scale(X_train)
    X_val_s = scale(X_val)
    X_test_s = scale(X_test)

    validate_X("final X_train", X_train_s)
    validate_X("final X_val", X_val_s)
    validate_X("final X_test", X_test_s)

    return (
        X_train_s,
        X_val_s,
        X_test_s,
        scaler,
        lower,
        upper,
        median,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_outputs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    encoder,
    metadata: Dict,
) -> None:
    """Save corrected versioned dataset artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = OUTPUT_DIR / "dataset.npz"
    scaler_path = OUTPUT_DIR / "scaler.joblib"
    encoder_path = OUTPUT_DIR / "label_encoder.joblib"
    metadata_path = OUTPUT_DIR / "metadata.json"

    np.savez_compressed(
        dataset_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Verify that the files were actually created.
    for path in (
        dataset_path,
        scaler_path,
        encoder_path,
        metadata_path,
    ):
        require_file(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("CHRONEX DATASET CONSOLIDATION + STRATIFIED RESPLIT")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Load canonical contract
    # -----------------------------------------------------------------------

    encoder, old_metadata, feature_order = load_contract()

    print("\nCanonical 15-class label map:")
    for i, label in enumerate(encoder.classes_):
        print(f"  {i:2d}: {label}")

    # -----------------------------------------------------------------------
    # Rebuild current raw sequences
    # -----------------------------------------------------------------------

    print("\nRebuilding current sequence pool from cleaned CSVs...")

    X_current, y_current, current_audit = build_current_raw_pool(
        PROCESSED_DIR,
        encoder,
        feature_order,
    )

    # -----------------------------------------------------------------------
    # Load optional incoming raw sequences
    # -----------------------------------------------------------------------

    incoming_X, incoming_y, incoming_names = load_incoming_raw(
        INCOMING_DIR,
        encoder,
    )

    # -----------------------------------------------------------------------
    # Combine all available raw sequences
    # -----------------------------------------------------------------------

    X_parts = [X_current]
    y_parts = [y_current]

    if incoming_X:
        X_parts.extend(incoming_X)
        y_parts.extend(incoming_y)

    X_all = np.concatenate(
        X_parts,
        axis=0,
    ).astype(np.float32, copy=False)

    y_all = np.concatenate(
        y_parts,
        axis=0,
    ).astype(np.int64, copy=False)

    validate_X("combined raw X", X_all)
    validate_y("combined raw y", y_all, len(X_all))

    print("\nCombined raw pool:")
    print(f"  X = {X_all.shape}")
    print(f"  y = {y_all.shape}")

    # -----------------------------------------------------------------------
    # Combined class counts
    # -----------------------------------------------------------------------

    print("\nCombined class counts:")

    combined_counts = class_counts(y_all, encoder)

    for class_name, count in combined_counts.items():
        print(f"  {class_name}: {count:,}")

    # -----------------------------------------------------------------------
    # Stratified split
    # -----------------------------------------------------------------------

    (
        X_train_raw,
        y_train,
        X_val_raw,
        y_val,
        X_test_raw,
        y_test,
    ) = stratified_split(X_all, y_all)

    print("\nSplit:")
    print(f"  TRAIN: {X_train_raw.shape}, {y_train.shape}")
    print(f"  VAL:   {X_val_raw.shape}, {y_val.shape}")
    print(f"  TEST:  {X_test_raw.shape}, {y_test.shape}")

    require_all_classes(
        y_train,
        y_val,
        y_test,
        encoder,
    )

    # -----------------------------------------------------------------------
    # Train-only preprocessing
    # -----------------------------------------------------------------------

    print("\nFitting clipping statistics + StandardScaler on TRAIN ONLY...")

    (
        X_train,
        X_val,
        X_test,
        scaler,
        lower,
        upper,
        median,
    ) = preprocess_train_only(
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )

    # -----------------------------------------------------------------------
    # Final contract validation
    # -----------------------------------------------------------------------

    validate_X("final X_train", X_train)
    validate_X("final X_val", X_val)
    validate_X("final X_test", X_test)

    validate_y("final y_train", y_train, len(X_train))
    validate_y("final y_val", y_val, len(X_val))
    validate_y("final y_test", y_test, len(X_test))

    if scaler.n_features_in_ != EXPECTED_N_FEATURES:
        fail(
            f"Scaler has {scaler.n_features_in_} input features; "
            f"expected {EXPECTED_N_FEATURES}"
        )

    require_all_classes(
        y_train,
        y_val,
        y_test,
        encoder,
    )

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    metadata = {
        "pipeline_version": "v2_stratified_train_only_scaling",
        "window_seconds": EXPECTED_WINDOW_SECONDS,
        "sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "n_features": EXPECTED_N_FEATURES,
        "feature_columns_order": feature_order,
        "label_classes": encoder.classes_.tolist(),
        "split": {
            "method": "class_stratified_sequence_split",
            "train_fraction": TRAIN_FRAC,
            "validation_fraction": VAL_FRAC,
            "test_fraction": TEST_FRAC,
            "random_state": RANDOM_STATE,
        },
        "preprocessing": {
            "source": (
                "raw sequences rebuilt from cleaned CSVs plus optional "
                "incoming raw NPZ files"
            ),
            "percentile_clip_lower": 1,
            "percentile_clip_upper": 99,
            "missing_value_fill": "train_median",
            "scaler": "StandardScaler",
            "scaler_fit_on": "train_only",
        },
        "sources": {
            "current_cleaned_files": [
                item["source"] for item in current_audit
            ],
            "incoming_raw_npz_files": incoming_names,
        },
        "shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "sequence_counts": {
            "combined": int(len(X_all)),
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "class_counts": {
            "combined": combined_counts,
            "train": class_counts(y_train, encoder),
            "validation": class_counts(y_val, encoder),
            "test": class_counts(y_test, encoder),
        },
        "train_preprocessing_statistics": {
            "lower_bound_shape": list(lower.shape),
            "upper_bound_shape": list(upper.shape),
            "median_shape": list(median.shape),
            "scaler_mean_shape": list(scaler.mean_.shape),
            "scaler_scale_shape": list(scaler.scale_.shape),
        },
        "notes": [
            "Existing data/processed/model_ready artifacts are not modified.",
            (
                "Current sequences are rebuilt from cleaned CSVs instead of "
                "inverse-transforming the old scaled dataset."
            ),
            (
                "Future incoming NPZ files must contain RAW/unscaled X with "
                "shape (N,10,71)."
            ),
            (
                "The existing label encoder is reused; no new label "
                "vocabulary is fitted."
            ),
            (
                "attack_ratio is excluded from the 71 model input features "
                "because it is label-derived."
            ),
            (
                "Clipping, missing-value statistics, and StandardScaler are "
                "derived from TRAIN ONLY."
            ),
        ],
    }

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    save_outputs(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        scaler,
        encoder,
        metadata,
    )

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("FINAL CHECK")
    print("=" * 72)

    require_all_classes(
        y_train,
        y_val,
        y_test,
        encoder,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")

    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape:   {y_val.shape}")
    print(f"y_test shape:  {y_test.shape}")

    print(f"Scaler features: {scaler.n_features_in_}")
    print(f"Feature count:   {EXPECTED_N_FEATURES}")
    print(f"Sequence length: {EXPECTED_SEQUENCE_LENGTH}")
    print(f"Window seconds:  {EXPECTED_WINDOW_SECONDS}")

    print("\nSaved versioned dataset to:")
    print(f"  {OUTPUT_DIR}")

    print("\nExisting model_ready directory was NOT modified.")
    print("DONE.")


if __name__ == "__main__":
    main()
