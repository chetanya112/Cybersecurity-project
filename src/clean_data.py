"""
clean_data.py
--------------
Cleans all raw CIC-IDS-2018 CSV files in data/raw/ and saves consistent,
model-ready versions into data/processed/.

Fixes applied (based on issues discovered during exploration):
1. Some files (e.g. 02-20-2018.csv) contain extra identity columns
   (Flow ID, Src IP, Src Port, Dst IP) not present in other files -> dropped
   so every output file has the same column set.
2. CIC-IDS-2018 CSVs sometimes contain the header row duplicated as a literal
   data row -> removed.
3. Infinite values (from divide-by-zero, e.g. Flow Byts/s) -> converted to NaN
   and dropped.
4. Non-numeric garbage in numeric columns -> coerced to NaN and dropped.
5. Timestamps parsed robustly, and a sanity filter drops any row outside the
   known valid range of this dataset (Feb-Mar 2018) -- this catches date
   parsing artifacts that would otherwise corrupt time-window grouping later.
6. Constant columns (zero variance in EVERY file) are dropped globally,
   after checking across all files -- not decided per-file, which would
   cause inconsistent column sets.
7. Numeric columns downcast to float32/smaller ints to reduce memory usage.
8. Large files are read and cleaned in chunks to avoid memory errors.
9. Already-cleaned files are skipped on re-run (safe to resume).
10. A cleaning log (data/processed/cleaning_log.csv) records what happened
    to each file, for reproducibility and reporting.
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
LOG_PATH = os.path.join(PROCESSED_DIR, "cleaning_log.csv")
CHUNK_SIZE = 200_000

ID_COLS_TO_DROP = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP']

GLOBAL_CONSTANT_COLS = [
    'Bwd Blk Rate Avg', 'Fwd Pkts/b Avg', 'Bwd Byts/b Avg', 'Bwd URG Flags',
    'Bwd PSH Flags', 'Fwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Fwd Blk Rate Avg'
]

VALID_DATE_MIN = '2018-02-01'
VALID_DATE_MAX = '2018-03-31'

NON_NUMERIC_COLS = ['Timestamp', 'Label']


# ---------------------------------------------------------------------------
# CLEANING LOGIC
# ---------------------------------------------------------------------------
def clean_chunk(df):
    """Apply all row/column-level cleaning steps to a single chunk."""
    stats = {}
    start = len(df)

    # 1. Normalize column names and Label text
    df.columns = df.columns.str.strip()
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip()

    # 2. Drop identity columns that only appear in some files
    df.drop(columns=[c for c in ID_COLS_TO_DROP if c in df.columns], inplace=True)

    # 3. Drop embedded duplicate header rows (header repeated as a data row)
    if 'Dst Port' in df.columns:
        df = df[df['Dst Port'] != 'Dst Port']

    # 4. Replace infinities with NaN (e.g. Flow Byts/s when Flow Duration=0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 5. Force numeric columns to be numeric; invalid entries -> NaN
    numeric_cols = [c for c in df.columns if c not in NON_NUMERIC_COLS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 6. Drop rows with any missing/invalid values
    df.dropna(inplace=True)
    stats['dropped_invalid'] = start - len(df)

    # 7. Parse Timestamp robustly, then sanity-filter to the known valid range
    if 'Timestamp' in df.columns:
        before_ts = len(df)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
        df.dropna(subset=['Timestamp'], inplace=True)
        df = df[(df['Timestamp'] >= VALID_DATE_MIN) & (df['Timestamp'] <= VALID_DATE_MAX)]
        stats['dropped_bad_timestamp'] = before_ts - len(df)
    else:
        stats['dropped_bad_timestamp'] = 0

    # 8. Downcast numeric types to reduce memory footprint
    for col in numeric_cols:
        if col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')

    return df, stats


def load_and_clean(filepath):
    """Read a CSV in chunks, clean each chunk, combine, dedupe, and drop
    globally-constant columns. Returns the cleaned DataFrame + stats."""
    print(f"Loading {filepath} ...")

    cleaned_chunks = []
    total_before = 0
    total_dropped_invalid = 0
    total_dropped_ts = 0

    for chunk in pd.read_csv(filepath, low_memory=False, chunksize=CHUNK_SIZE):
        total_before += len(chunk)
        cleaned, stats = clean_chunk(chunk)
        cleaned_chunks.append(cleaned)
        total_dropped_invalid += stats.get('dropped_invalid', 0)
        total_dropped_ts += stats.get('dropped_bad_timestamp', 0)

    df = pd.concat(cleaned_chunks, ignore_index=True)

    before_dup = len(df)
    df.drop_duplicates(inplace=True)
    dropped_dup = before_dup - len(df)

    df.drop(columns=[c for c in GLOBAL_CONSTANT_COLS if c in df.columns], inplace=True)

    print(f"  Rows before:              {total_before}")
    print(f"  Dropped invalid/missing:  {total_dropped_invalid}")
    print(f"  Dropped bad timestamps:   {total_dropped_ts}")
    print(f"  Dropped duplicates:       {dropped_dup}")
    print(f"  Final shape:              {df.shape}")

    stats = {
        'rows_before': total_before,
        'dropped_invalid': total_dropped_invalid,
        'dropped_bad_timestamp': total_dropped_ts,
        'dropped_duplicates': dropped_dup,
        'rows_after': len(df),
        'columns_after': df.shape[1],
    }
    return df, stats


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".csv"))

    log_rows = []

    for filename in files:
        out_path = os.path.join(PROCESSED_DIR, f"cleaned_{filename}")
        if os.path.exists(out_path):
            print(f"Skipping {filename} (already cleaned)\n")
            continue

        try:
            filepath = os.path.join(RAW_DIR, filename)
            cleaned_df, stats = load_and_clean(filepath)
            cleaned_df.to_csv(out_path, index=False)
            stats['filename'] = filename
            log_rows.append(stats)
            print(f"  Saved to {out_path}\n")
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}\n")
            continue

    if log_rows:
        log_df = pd.DataFrame(log_rows)
        if os.path.exists(LOG_PATH):
            old_log = pd.read_csv(LOG_PATH)
            log_df = pd.concat([old_log, log_df], ignore_index=True)
        log_df.to_csv(LOG_PATH, index=False)
        print(f"Cleaning log saved to {LOG_PATH}")

    print("\nAll files processed.")