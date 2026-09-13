import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from backend.services.dataset_adapter import process_upload
from backend.services.chronex_service import initialize_service
import backend.services.chronex_service as chronex_service
from backend.test_csv_ingestion import df_to_fake_file

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"

def test_native_cicids_equivalence():
    initialize_service()
    
    csv_file = DATA_DIR / "cleaned_02-14-2018.csv"
    if not csv_file.exists():
        pytest.skip("Test file not found")
        
    df = pd.read_csv(csv_file, nrows=50000)
    
    # 1. Process via upload chunked pipeline
    report = process_upload(df_to_fake_file(df))
    assert report.compatibility == "NATIVE"
    
    upload_mat = np.load(ROOT / "data" / "temp_uploads" / f"{report.upload_id}.npy")
    
    # 2. Process via exact original dataframe operations (equivalent to production chronex)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    df['__parsed_ts'] = df['Timestamp'].dt.floor('30s')
    
    canon = chronex_service._META['feature_columns_order']
    
    # Clean up df for numeric means exactly like original
    numeric_cols = [c for c in canon if c != 'flow_count']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0)
        
    df['flow_count'] = 1.0
    
    grouped = df.groupby('__parsed_ts')
    df_sums = grouped[canon].sum()
    df_counts = grouped[canon].count()
    
    df_counts = df_counts.replace(0, 1)
    df_means = df_sums / df_counts
    df_means['flow_count'] = df_sums['flow_count'] # Override flow_count mean to sum
    
    sorted_ts = sorted(df_means.index)
    
    # Find longest block
    best_start, best_len = 0, 1
    curr_start, curr_len = 0, 1
    for i in range(1, len(sorted_ts)):
        if np.isclose((sorted_ts[i] - sorted_ts[i-1]).total_seconds(), 30.0, atol=1.0):
            curr_len += 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = curr_start
        else:
            curr_start = i
            curr_len = 1
            
    assert best_len >= 10
    best_block = df_means.iloc[best_start:best_start+best_len].values.astype(np.float32)
    best_block_scaled = chronex_service._SCALER.transform(best_block).astype(np.float32)
    
    assert upload_mat.shape == best_block_scaled.shape
    assert np.allclose(upload_mat, best_block_scaled, rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    pytest.main([__file__])
