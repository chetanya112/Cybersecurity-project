"""
build_sequences.py
-------------------
Takes the cleaned per-day CSVs (from clean_data.py) and produces the final
model-ready dataset for the LSTM: time-windowed network state vectors,
chunked into sequences, split chronologically into train/val/test.

Pipeline:
  1. Load each cleaned day-file
  2. Group flows into fixed-size time windows (default: 30 seconds)
  3. Aggregate each window into one "state vector" (mean of numeric features
     + flow count + attack ratio + majority label)
  4. Build sliding sequences of consecutive windows (default length: 10)
     -> X = past 10 window-states, y = label of the NEXT window (forecasting)
  5. Split chronologically per day (70% train / 15% val / 15% test) to avoid
     leaking future information into training
  6. Scale features (fit scaler on train only, apply to val/test)
  7. Save everything as compressed .npz + a scaler + label encoder + metadata

Designed to be safe to re-run: if one file fails, it's skipped with a clear
message and the rest continue. Every intermediate check has a guard against
empty/degenerate input so it fails loudly and clearly instead of silently
producing a broken dataset.
"""

import pandas as pd
import numpy as np
import os
import json
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROCESSED_DIR = "../data/processed"
OUTPUT_DIR = "../data/processed/model_ready"
WINDOW_SECONDS = 30       # size of each time window
SEQUENCE_LENGTH = 10      # how many past windows the LSTM sees at once
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 goes to test

NON_FEATURE_COLS = ['Timestamp', 'Label']


# ---------------------------------------------------------------------------
# STEP 1: TIME WINDOWING + STATE VECTORS
# ---------------------------------------------------------------------------
def build_state_vectors(df):
    """Group flows into fixed time windows and aggregate each into one
    'network state vector' row. Returns (state_df, feature_cols)."""
    df = df.sort_values('Timestamp').reset_index(drop=True)

    if df['Timestamp'].isna().any():
        raise ValueError("Found NaT in Timestamp column - clean_data.py should have removed these")

    t0 = df['Timestamp'].min()
    df = df.copy()
    df['window_id'] = ((df['Timestamp'] - t0).dt.total_seconds() // WINDOW_SECONDS).astype(int)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + ['window_id']]
    if not feature_cols:
        raise ValueError("No feature columns found after excluding Timestamp/Label/window_id")

    grouped = df.groupby('window_id')

    agg = grouped[feature_cols].mean()
    agg['flow_count'] = grouped.size()
    agg['attack_ratio'] = grouped['Label'].apply(lambda s: (s != 'Benign').mean())
    agg['majority_label'] = grouped['Label'].agg(lambda s: s.value_counts().idxmax())
    agg['window_start_time'] = grouped['Timestamp'].min()

    agg = agg.reset_index().sort_values('window_id').reset_index(drop=True)

    # Sanity check: no NaNs should exist in the final aggregated table
    numeric_check_cols = feature_cols + ['flow_count', 'attack_ratio']
    if agg[numeric_check_cols].isna().any().any():
        raise ValueError("NaNs found in aggregated state vectors - investigate before proceeding")

    return agg, feature_cols


# ---------------------------------------------------------------------------
# STEP 2: BUILD SEQUENCES (forecasting: predict the NEXT window's label)
# ---------------------------------------------------------------------------
def build_sequences(state_df, feature_cols, seq_len=SEQUENCE_LENGTH):
    """From a chronological state-vector table, build sliding sequences.
    X[i] = seq_len consecutive window feature-vectors
    y[i] = majority_label of the window immediately AFTER the sequence
    """
    all_feature_cols = feature_cols + ['flow_count', 'attack_ratio']
    feature_matrix = state_df[all_feature_cols].to_numpy(dtype=np.float32)
    labels = state_df['majority_label'].to_numpy()

    n_windows = len(state_df)
    n_sequences = n_windows - seq_len
    if n_sequences <= 0:
        return np.empty((0, seq_len, len(all_feature_cols)), dtype=np.float32), np.empty((0,), dtype=labels.dtype)

    X = np.empty((n_sequences, seq_len, len(all_feature_cols)), dtype=np.float32)
    y = np.empty((n_sequences,), dtype=labels.dtype)

    for i in range(n_sequences):
        X[i] = feature_matrix[i:i + seq_len]
        y[i] = labels[i + seq_len]

    return X, y


# ---------------------------------------------------------------------------
# STEP 3: CHRONOLOGICAL SPLIT (per day, to avoid leakage)
# ---------------------------------------------------------------------------
def chrono_split(X, y):
    n = len(X)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])


def safe_concat(list_of_arrays, name):
    """Concatenate a list of numpy arrays, with a clear error if it's empty."""
    non_empty = [a for a in list_of_arrays if len(a) > 0]
    if not non_empty:
        raise RuntimeError(
            f"No data collected for '{name}'. All input files may have failed "
            f"or produced too few windows for SEQUENCE_LENGTH={SEQUENCE_LENGTH}. "
            f"Check the per-file logs above."
        )
    return np.concatenate(non_empty, axis=0)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(PROCESSED_DIR):
        print(f"ERROR: {PROCESSED_DIR} does not exist. Run clean_data.py first.")
        sys.exit(1)

    files = sorted(f for f in os.listdir(PROCESSED_DIR)
                    if f.startswith("cleaned_") and f.endswith(".csv"))

    if not files:
        print(f"ERROR: No cleaned_*.csv files found in {PROCESSED_DIR}. Run clean_data.py first.")
        sys.exit(1)

    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []
    feature_cols_reference = None
    files_processed = 0
    files_failed = []

    for fname in files:
        print(f"Processing {fname} ...")
        path = os.path.join(PROCESSED_DIR, fname)

        try:
            df = pd.read_csv(path, parse_dates=['Timestamp'])

            if len(df) == 0:
                print(f"  Skipping (empty file)\n")
                continue

            state_df, feature_cols = build_state_vectors(df)

            if feature_cols_reference is None:
                feature_cols_reference = feature_cols
            elif feature_cols != feature_cols_reference:
                raise ValueError(
                    f"Column mismatch: {fname} has different features than "
                    f"previous files. All cleaned files must have identical columns."
                )

            print(f"  {len(df)} flows -> {len(state_df)} time windows ({WINDOW_SECONDS}s each)")

            if len(state_df) <= SEQUENCE_LENGTH:
                print(f"  Skipping (only {len(state_df)} windows, need > {SEQUENCE_LENGTH})\n")
                continue

            X, y = build_sequences(state_df, feature_cols)
            X_tr, y_tr, X_va, y_va, X_te, y_te = chrono_split(X, y)

            all_X_train.append(X_tr); all_y_train.append(y_tr)
            all_X_val.append(X_va);   all_y_val.append(y_va)
            all_X_test.append(X_te);  all_y_test.append(y_te)

            print(f"  Sequences -> train:{len(X_tr)} val:{len(X_va)} test:{len(X_te)}\n")
            files_processed += 1

            del df, state_df, X, y

        except Exception as e:
            print(f"  ERROR processing {fname}: {e}\n")
            files_failed.append(fname)
            continue

    print(f"Files processed successfully: {files_processed}/{len(files)}")
    if files_failed:
        print(f"Files that failed: {files_failed}")

    # ---- Combine all days together (with clear errors if something's empty) ----
    X_train = safe_concat(all_X_train, "X_train")
    y_train = safe_concat(all_y_train, "y_train")
    X_val = safe_concat(all_X_val, "X_val")
    y_val = safe_concat(all_y_val, "y_val")
    X_test = safe_concat(all_X_test, "X_test")
    y_test = safe_concat(all_y_test, "y_test")

    print(f"\nTOTAL -> train:{X_train.shape} val:{X_val.shape} test:{X_test.shape}")

    # ---- Guard against inf/extreme values before scaling ----
    # Some flow-rate features (e.g. Flow Byts/s) can produce very large means
    # when Flow Duration is tiny, which can overflow float32 during scaling.
    # Replace inf with NaN, then clip each feature to a safe finite range
    # based on percentiles computed from the TRAIN split only (no leakage).
    n_features = X_train.shape[2]

    def replace_inf_with_nan(X):
        X = X.copy()
        X[~np.isfinite(X)] = np.nan
        return X

    X_train = replace_inf_with_nan(X_train)
    X_val = replace_inf_with_nan(X_val)
    X_test = replace_inf_with_nan(X_test)

    flat_train = X_train.reshape(-1, n_features)
    # Per-feature 1st/99th percentile computed ignoring NaNs, from train only
    lower = np.nanpercentile(flat_train, 1, axis=0)
    upper = np.nanpercentile(flat_train, 99, axis=0)
    # Per-feature median, used to fill any remaining NaNs (from original inf values)
    median = np.nanmedian(flat_train, axis=0)

    def clip_and_fill(X):
        shape = X.shape
        flat = X.reshape(-1, n_features).copy()
        for j in range(n_features):
            col = flat[:, j]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col[nan_mask] = median[j]
            flat[:, j] = np.clip(col, lower[j], upper[j])
        return flat.reshape(shape)

    X_train = clip_and_fill(X_train)
    X_val = clip_and_fill(X_val)
    X_test = clip_and_fill(X_test)

    n_inf_fixed = int((~np.isfinite(X_train)).sum())  # should be 0 now, sanity check
    print(f"Post-clip sanity check - remaining non-finite values in X_train: {n_inf_fixed}")

    # ---- Scale features (fit on train only, to avoid leakage) ----
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def scale(X):
        if len(X) == 0:
            return X
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_features)).reshape(shape).astype(np.float32)

    X_train_scaled = scale(X_train)
    X_val_scaled = scale(X_val)
    X_test_scaled = scale(X_test)

    # ---- Encode labels (text -> integer). Fit on the union of all splits so
    # every label seen anywhere has a stable, consistent integer code. ----
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_val, y_test]))
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    # ---- Save everything ----
    out_npz = os.path.join(OUTPUT_DIR, "dataset.npz")
    np.savez_compressed(
        out_npz,
        X_train=X_train_scaled, y_train=y_train_enc,
        X_val=X_val_scaled, y_val=y_val_enc,
        X_test=X_test_scaled, y_test=y_test_enc,
    )
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

    metadata = {
        "window_seconds": WINDOW_SECONDS,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_columns_order": feature_cols_reference + ["flow_count", "attack_ratio"],
        "n_features": n_features,
        "label_classes": label_encoder.classes_.tolist(),
        "files_processed": files_processed,
        "files_failed": files_failed,
        "shapes": {
            "X_train": list(X_train_scaled.shape),
            "X_val": list(X_val_scaled.shape),
            "X_test": list(X_test_scaled.shape),
        }
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved dataset.npz, scaler.joblib, label_encoder.joblib, metadata.json to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()