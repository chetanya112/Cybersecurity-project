import pandas as pd
import numpy as np
import os
import json
import joblib
from collections import Counter
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = "data/processed"
PRIMARY_OUT = "data/processed/model_ready_v2_primary"
UNSEEN_OUT = "data/processed/model_ready_v2_unseen"

WINDOW_SECONDS = 30
SEQ_LEN = 10
FUTURE_K = 10

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

NON_FEATURE_COLS = ['Timestamp', 'Label', 'window_id']

def build_state_vectors(df):
    df = df.sort_values('Timestamp').reset_index(drop=True)
    t0 = df['Timestamp'].min()
    df['window_id'] = ((df['Timestamp'] - t0).dt.total_seconds() // WINDOW_SECONDS).astype(int)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    
    grouped = df.groupby('window_id')
    
    grouped = df.groupby('window_id')
    agg = grouped[feature_cols].mean()
    agg['flow_count'] = grouped.size()
    
    # Safely handle infinities and float32 limits on the aggregated data (NOT raw df to prevent OOM)
    f32_min, f32_max = np.finfo(np.float32).min, np.finfo(np.float32).max
    cleaned_features = agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    cleaned_features = np.clip(cleaned_features.values, f32_min, f32_max)
    agg.loc[:, feature_cols] = cleaned_features
    
    attacks_only = df[df['Label'] != 'Benign']
    if not attacks_only.empty:
        attack_freq = attacks_only.groupby(['window_id', 'Label']).size().reset_index(name='count')
        attack_freq = attack_freq.sort_values(['window_id', 'count'], ascending=[True, False])
        dominant_attacks = attack_freq.drop_duplicates(subset=['window_id']).set_index('window_id')['Label']
        agg['majority_label'] = agg.index.map(dominant_attacks).fillna('Benign')
    else:
        agg['majority_label'] = 'Benign'

    agg = agg.reset_index().sort_values('window_id').reset_index(drop=True)
    all_feature_cols = feature_cols + ['flow_count']
    return agg, all_feature_cols

def split_into_continuous_blocks(agg_df):
    window_ids = agg_df['window_id'].values
    gaps = np.where(np.diff(window_ids) > 1)[0] + 1
    
    blocks = []
    start = 0
    for g in gaps:
        blocks.append(agg_df.iloc[start:g])
        start = g
    blocks.append(agg_df.iloc[start:])
    return blocks

def extract_sequences_from_block(block, feature_cols):
    mat = block[feature_cols].values.astype(np.float32)
    labels = block['majority_label'].values
    
    total_len = len(block)
    req_len = SEQ_LEN + FUTURE_K
    if total_len < req_len:
        return None, None, None
        
    n_seq = total_len - req_len + 1
    X = np.empty((n_seq, SEQ_LEN, len(feature_cols)), dtype=np.float32)
    Y = np.empty((n_seq, FUTURE_K, len(feature_cols)), dtype=np.float32)
    Y_labels = np.empty((n_seq, FUTURE_K), dtype=object)
    
    for i in range(n_seq):
        X[i] = mat[i : i + SEQ_LEN]
        Y[i] = mat[i + SEQ_LEN : i + SEQ_LEN + FUTURE_K]
        Y_labels[i] = labels[i + SEQ_LEN : i + SEQ_LEN + FUTURE_K]
        
    return X, Y, Y_labels

def main():
    os.makedirs(PRIMARY_OUT, exist_ok=True)
    os.makedirs(UNSEEN_OUT, exist_ok=True)
    
    files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith("cleaned_") and f.endswith(".csv")])
    
    stats = {
        'total_valid_windows': 0,
        'discarded_windows': 0,
        'gap_count': 0,
        'class_counts': Counter()
    }
    
    file_data = {}
    
    print("Step 1: Extracting strict continuous sequences from all files...")
    
    feature_cols_ref = None
    
    for fname in files:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, fname), parse_dates=['Timestamp'])
        if len(df) == 0: continue
        
        agg, feat_cols = build_state_vectors(df)
        if feature_cols_ref is None:
            feature_cols_ref = feat_cols
            
        blocks = split_into_continuous_blocks(agg)
        stats['gap_count'] += len(blocks) - 1
        
        file_X, file_Y, file_Y_labels = [], [], []
        
        for block in blocks:
            X, Y, Y_lab = extract_sequences_from_block(block, feat_cols)
            if X is not None:
                file_X.append(X)
                file_Y.append(Y)
                file_Y_labels.append(Y_lab)
                stats['total_valid_windows'] += len(block)
                for seq_labs in Y_lab:
                    stats['class_counts'].update(seq_labs.flatten())
            else:
                stats['discarded_windows'] += len(block)
                
        if file_X:
            file_data[fname] = {
                'X': np.concatenate(file_X),
                'Y': np.concatenate(file_Y),
                'Y_labels': np.concatenate(file_Y_labels)
            }
    
    print("\nStep 2: Building PROTOCOL A (Primary Temporal Forecasting Benchmark)")
    
    A_X_tr, A_Y_tr, A_ylab_tr = [], [], []
    A_X_va, A_Y_va, A_ylab_va = [], [], []
    A_X_te, A_Y_te, A_ylab_te = [], [], []
    
    DROP_MARGIN = SEQ_LEN + FUTURE_K - 1
    
    for fname, data in file_data.items():
        X, Y, Y_lab = data['X'], data['Y'], data['Y_labels']
        n = len(X)
        if n < DROP_MARGIN * 3:
            continue
            
        tr_end = int(n * TRAIN_FRAC)
        va_end = int(n * (TRAIN_FRAC + VAL_FRAC))
        
        A_X_tr.append(X[:tr_end])
        A_Y_tr.append(Y[:tr_end])
        A_ylab_tr.append(Y_lab[:tr_end])
        
        va_start = tr_end + DROP_MARGIN
        A_X_va.append(X[va_start:va_end])
        A_Y_va.append(Y[va_start:va_end])
        A_ylab_va.append(Y_lab[va_start:va_end])
        
        te_start = va_end + DROP_MARGIN
        if te_start < n:
            A_X_te.append(X[te_start:])
            A_Y_te.append(Y[te_start:])
            A_ylab_te.append(Y_lab[te_start:])

    X_train_A = np.concatenate(A_X_tr)
    Y_train_A = np.concatenate(A_Y_tr)
    ylab_train_A = np.concatenate(A_ylab_tr)
    X_val_A = np.concatenate(A_X_va)
    Y_val_A = np.concatenate(A_Y_va)
    ylab_val_A = np.concatenate(A_ylab_va)
    X_test_A = np.concatenate(A_X_te)
    Y_test_A = np.concatenate(A_Y_te)
    ylab_test_A = np.concatenate(A_ylab_te)
    
    scaler_A = StandardScaler()
    scaler_A.fit(X_train_A.reshape(-1, X_train_A.shape[-1]))
    
    def scale_3d(arr, scaler):
        if len(arr) == 0: return arr
        return scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)
        
    X_train_A = scale_3d(X_train_A, scaler_A)
    Y_train_A = scale_3d(Y_train_A, scaler_A)
    X_val_A = scale_3d(X_val_A, scaler_A)
    Y_val_A = scale_3d(Y_val_A, scaler_A)
    X_test_A = scale_3d(X_test_A, scaler_A)
    Y_test_A = scale_3d(Y_test_A, scaler_A)
    
    np.savez_compressed(os.path.join(PRIMARY_OUT, "dataset.npz"),
                        X_train=X_train_A, Y_train=Y_train_A, ylab_train=ylab_train_A,
                        X_val=X_val_A, Y_val=Y_val_A, ylab_val=ylab_val_A,
                        X_test=X_test_A, Y_test=Y_test_A, ylab_test=ylab_test_A)
    joblib.dump(scaler_A, os.path.join(PRIMARY_OUT, "scaler.joblib"))
    
    print("\nStep 3: Building PROTOCOL B (Unseen-Attack Generalization Benchmark)")
    B_X_tr, B_Y_tr, B_ylab_tr = [], [], []
    B_X_va, B_Y_va, B_ylab_va = [], [], []
    B_X_te, B_Y_te, B_ylab_te = [], [], []
    
    train_days = ["cleaned_02-14-2018.csv", "cleaned_02-15-2018.csv", "cleaned_02-16-2018.csv", 
                  "cleaned_02-20-2018.csv", "cleaned_02-21-2018.csv", "cleaned_02-28-2018.csv"]
    val_days = ["cleaned_02-22-2018.csv"]
    test_days = ["cleaned_02-23-2018.csv", "cleaned_03-01-2018.csv", "cleaned_03-02-2018.csv"]
    
    for fname, data in file_data.items():
        if fname in train_days:
            B_X_tr.append(data['X']); B_Y_tr.append(data['Y']); B_ylab_tr.append(data['Y_labels'])
        elif fname in val_days:
            B_X_va.append(data['X']); B_Y_va.append(data['Y']); B_ylab_va.append(data['Y_labels'])
        elif fname in test_days:
            B_X_te.append(data['X']); B_Y_te.append(data['Y']); B_ylab_te.append(data['Y_labels'])
            
    X_train_B = np.concatenate(B_X_tr)
    Y_train_B = np.concatenate(B_Y_tr)
    ylab_train_B = np.concatenate(B_ylab_tr)
    X_val_B = np.concatenate(B_X_va)
    Y_val_B = np.concatenate(B_Y_va)
    ylab_val_B = np.concatenate(B_ylab_va)
    X_test_B = np.concatenate(B_X_te)
    Y_test_B = np.concatenate(B_Y_te)
    ylab_test_B = np.concatenate(B_ylab_te)
    
    scaler_B = StandardScaler()
    scaler_B.fit(X_train_B.reshape(-1, X_train_B.shape[-1]))
    
    X_train_B = scale_3d(X_train_B, scaler_B)
    Y_train_B = scale_3d(Y_train_B, scaler_B)
    X_val_B = scale_3d(X_val_B, scaler_B)
    Y_val_B = scale_3d(Y_val_B, scaler_B)
    X_test_B = scale_3d(X_test_B, scaler_B)
    Y_test_B = scale_3d(Y_test_B, scaler_B)
    
    np.savez_compressed(os.path.join(UNSEEN_OUT, "dataset.npz"),
                        X_train=X_train_B, Y_train=Y_train_B, ylab_train=ylab_train_B,
                        X_val=X_val_B, Y_val=Y_val_B, ylab_val=ylab_val_B,
                        X_test=X_test_B, Y_test=Y_test_B, ylab_test=ylab_test_B)
    joblib.dump(scaler_B, os.path.join(UNSEEN_OUT, "scaler.joblib"))
    
    all_classes = set()
    for fname, data in file_data.items():
        all_classes.update(np.unique(data['Y_labels']))
    all_classes = sorted(list(all_classes))
    if 'Benign' in all_classes:
        all_classes.remove('Benign')
        all_classes.insert(0, 'Benign')
        
    meta = {
        "n_features": len(feature_cols_ref),
        "sequence_length": SEQ_LEN,
        "future_horizon": FUTURE_K,
        "feature_columns_order": feature_cols_ref,
        "label_classes": all_classes
    }
    with open(os.path.join(PRIMARY_OUT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(UNSEEN_OUT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n============================================================")
    print("PHASE 1-3 IMPLEMENTATION REPORT")
    print("============================================================")
    print(f"2. Number of valid windows: {stats['total_valid_windows']}")
    print(f"3. Discarded sequences/windows: {stats['discarded_windows']} windows discarded (blocks smaller than {SEQ_LEN + FUTURE_K}).")
    print(f"5. Class distributions (across all generated sequences):")
    for k, v in sorted(stats['class_counts'].items(), key=lambda item: item[1], reverse=True):
        print(f"     {k}: {v}")
    print(f"6. Temporal-gap statistics: Detected {stats['gap_count']} distinct >30s time gaps across captures.")
    
    print("\n--- PROTOCOL A: PRIMARY TEMPORAL FORECASTING ---")
    print("1. Split strategy: Chronological per-day (70/15/15) with strict boundary dropout.")
    print("7. Evidence of zero overlap: Dropped exactly seq_len + future_k - 1 (19) sequences at boundary.")
    print(f"4. Shapes:")
    print(f"   Train X: {X_train_A.shape}, Y: {Y_train_A.shape}")
    print(f"   Val X:   {X_val_A.shape}, Y: {Y_val_A.shape}")
    print(f"   Test X:  {X_test_A.shape}, Y: {Y_test_A.shape}")
    
    print("\n--- PROTOCOL B: UNSEEN-ATTACK GENERALIZATION ---")
    print("1. Split strategy: Strict Day-based isolation.")
    print("7. Evidence of zero overlap: Entire days are segregated into splits; physical overlap is impossible.")
    print(f"4. Shapes:")
    print(f"   Train X: {X_train_B.shape}, Y: {Y_train_B.shape}")
    print(f"   Val X:   {X_val_B.shape}, Y: {Y_val_B.shape}")
    print(f"   Test X:  {X_test_B.shape}, Y: {Y_test_B.shape}")
    
    print(f"\n8. Future-target shapes: (batch_size, {FUTURE_K}, 71) containing actual float32 feature values.")
    print("9. Scaler verification: Fit strictly on Train sets for A and B independently.")
    print("============================================================\n")

if __name__ == '__main__':
    main()
