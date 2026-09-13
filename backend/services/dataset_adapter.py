import os
import uuid
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

from backend.schemas.dataset import CompatibilityReport, FeatureMapping, SchemaFlags
from backend.services import chronex_service

TEMP_DIR = chronex_service.ROOT / "data" / "temp_uploads"

# NATIVE aliases
ALIAS_REGISTRY = {
    "Src Port": ["Source Port", "source_port", "src_port"],
    "Dst Port": ["Destination Port", "destination_port", "dst_port", "Dest Port"],
    "Protocol": ["protocol", "Proto", "proto"],
    "Flow Duration": ["flow_duration", "duration", "Duration", "dur"],
    "Tot Fwd Pkts": ["Total Fwd Packets", "total_fwd_packets", "fwd_pkts", "Total Forward Packets"],
    "Tot Bwd Pkts": ["Total Backward Packets", "total_bwd_packets", "bwd_pkts", "Total Bwd Packets"],
    "TotLen Fwd Pkts": ["Total Length of Fwd Packets", "fwd_pkt_len_tot", "fwd_bytes", "Total Forward Packet Length"],
    "TotLen Bwd Pkts": ["Total Length of Bwd Packets", "bwd_pkt_len_tot", "bwd_bytes"],
    "Fwd Pkts/s": ["Forward Packets/s", "fwd_pkts_s"],
    "Bwd Pkts/s": ["Backward Packets/s", "bwd_pkts_s"],
    "Flow Byts/s": ["Flow Bytes/s", "flow_bytes_s"],
    "Flow Pkts/s": ["Flow Packets/s", "flow_pkts_s"],
}

TIMESTAMP_ALIASES = [
    "timestamp", "time", "datetime", "date", "starttime", "start time", 
    "flow start", "flow start time", "flowstarttime"
]

FALLBACK_FEATURES = [
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts"
]

PROTOCOL_ENCODING = {
    "TCP": 6, "tcp": 6, "6": 6, "6.0": 6,
    "UDP": 17, "udp": 17, "17": 17, "17.0": 17,
    "ICMP": 1, "icmp": 1, "1": 1, "1.0": 1,
    "ARP": 0, "arp": 0, "0": 0, "0.0": 0
}

def _detect_timestamp_column(df: pd.DataFrame) -> Dict[str, Any]:
    cols = list(df.columns)
    lower_cols = [c.lower().strip() for c in cols]
    
    date_col = None
    time_col = None
    for c, lc in zip(cols, lower_cols):
        if lc == 'date': date_col = c
        if lc == 'time': time_col = c
        
    if date_col and time_col:
        sample = df.head(100).copy()
        try:
            combined = sample[date_col].astype(str) + " " + sample[time_col].astype(str)
            parsed = pd.to_datetime(combined, errors='coerce')
            success_rate = parsed.notna().mean()
            if success_rate > 0.5:
                return {
                    "is_valid": True,
                    "columns": [date_col, time_col],
                    "method": "date_plus_time",
                    "success_rate": float(success_rate)
                }
        except Exception:
            pass

    candidates = []
    for alias in TIMESTAMP_ALIASES:
        for c, lc in zip(cols, lower_cols):
            if lc == alias or alias in lc:
                if c not in candidates:
                    candidates.append(c)
                    
    best_col = None
    best_rate = -1.0
    
    sample = df.head(100)
    best_method = "alias_match"
    for c in candidates:
        try:
            # First try standard parsing
            parsed = pd.to_datetime(sample[c], errors='coerce')
            
            # If standard parsing puts everything in 1970, it might be a float/int UNIX timestamp parsed as ns
            if parsed.notna().any() and (parsed.dt.year == 1970).all():
                try:
                    num_sample = pd.to_numeric(sample[c], errors='coerce')
                    parsed_s = pd.to_datetime(num_sample, unit='s', errors='coerce')
                    if (parsed_s.dt.year > 1980).any():
                        parsed = parsed_s
                        best_method = "unix_epoch"
                except Exception:
                    pass
            
            rate = parsed.notna().mean()
            if rate > best_rate:
                best_rate = rate
                best_col = c
        except Exception:
            pass
            
    if best_col and best_rate > 0.5:
        return {
            "is_valid": True,
            "columns": [best_col],
            "method": best_method,
            "success_rate": float(best_rate)
        }
        
    return {"is_valid": False}

def encode_protocol(val: Any) -> float:
    s_val = str(val).strip()
    if s_val in PROTOCOL_ENCODING:
        return float(PROTOCOL_ENCODING[s_val])
    # Fallback encoding for unknown
    try:
        return float(val)
    except ValueError:
        return 255.0

def process_upload(file_obj: Any) -> CompatibilityReport:
    chronex_service.initialize_service()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    canonical_features = chronex_service._META["feature_columns_order"]
    
    chunk_iter = pd.read_csv(file_obj, chunksize=50000, low_memory=False)
    
    is_first = True
    schema_flags = None
    mapped_features: List[FeatureMapping] = []
    derived_features: List[str] = []
    missing_requirements = []
    ts_cols = []
    ts_method = None
    ts_success_rate = None
    canonical_map = {} # mapped col -> original col
    is_generic = False
    
    bins = {}
    total_flow_count = 0
    
    for chunk in chunk_iter:
        total_flow_count += len(chunk)
        columns = list(chunk.columns)
        
        if is_first:
            is_first = False
            
            ts_info = _detect_timestamp_column(chunk)
            ts_cols = ts_info.get("columns", [])
            ts_method = ts_info.get("method")
            ts_success_rate = ts_info.get("success_rate")
            
            schema_flags = SchemaFlags(
                timestamp=ts_info["is_valid"],
                network_flow=False,
                packet_statistics=False,
                byte_statistics=False,
                duration=False
            )
            
            if not schema_flags.timestamp:
                pass # Will be synthesized if network flow requirements are met
            
            available_canonical = set()
            for col in columns:
                if col in canonical_features:
                    available_canonical.add(col)
                    canonical_map[col] = col
                    mapped_features.append(FeatureMapping(source=col, canonical=col, method="exact", confidence=1.0))
                else:
                    for can, aliases in ALIAS_REGISTRY.items():
                        if col in aliases and can not in available_canonical:
                            available_canonical.add(can)
                            canonical_map[can] = col
                            mapped_features.append(FeatureMapping(source=col, canonical=can, method="alias", confidence=0.95))
                            break
                            
            if "Src Port" in available_canonical or "Dst Port" in available_canonical or "Protocol" in available_canonical:
                schema_flags.network_flow = True
            if "Tot Fwd Pkts" in available_canonical or "Tot Bwd Pkts" in available_canonical:
                schema_flags.packet_statistics = True
            if "TotLen Fwd Pkts" in available_canonical or "TotLen Bwd Pkts" in available_canonical:
                schema_flags.byte_statistics = True
            if "Flow Duration" in available_canonical:
                schema_flags.duration = True
                
            if not schema_flags.network_flow: missing_requirements.append("Network flow identifiers (Ports/Protocol)")
            if not schema_flags.packet_statistics: missing_requirements.append("Packet statistics")
            
            if len(missing_requirements) > 0:
                if not schema_flags.timestamp:
                    missing_requirements.append("Timestamp column")
                return CompatibilityReport(
                    status="error", compatibility="UNSUPPORTED", error_code="UNSUPPORTED_SCHEMA",
                    reason="Insufficient network-flow semantics.", missing_requirements=missing_requirements
                )
                
            # Derivations for Native
            if "Subflow Fwd Pkts" not in available_canonical and "Tot Fwd Pkts" in available_canonical:
                available_canonical.add("Subflow Fwd Pkts")
                derived_features.append("Subflow Fwd Pkts derived from Tot Fwd Pkts")
            if "Subflow Bwd Pkts" not in available_canonical and "Tot Bwd Pkts" in available_canonical:
                available_canonical.add("Subflow Bwd Pkts")
                derived_features.append("Subflow Bwd Pkts derived from Tot Bwd Pkts")
            if "flow_count" not in available_canonical:
                available_canonical.add("flow_count")
                derived_features.append("flow_count derived as 1 per raw flow")
                
            missing_native = [f for f in canonical_features if f not in available_canonical]
            if len(missing_native) > 0:
                # Check for Generic
                missing_generic = [f for f in FALLBACK_FEATURES if f not in available_canonical]
                if len(missing_generic) > 0:
                    return CompatibilityReport(
                        status="error", compatibility="UNSUPPORTED", error_code="MISSING_REQUIRED_FEATURE",
                        reason="Dataset is missing required canonical features that cannot be safely derived.",
                        missing_features=missing_generic,
                        missing_requirements=["Either all 71 NATIVE features or 6 GENERIC features must be present."]
                    )
                else:
                    is_generic = True
            
        # 1. Parse timestamp
        if not schema_flags.timestamp:
            base_time = pd.Timestamp("2024-01-01 00:00:00")
            current_flow_index = total_flow_count - len(chunk)
            offsets = np.arange(len(chunk)) * 0.5 + (current_flow_index * 0.5)
            parsed_ts = base_time + pd.to_timedelta(offsets, unit='s')
            ts_method = "SYNTHETIC_LOGICAL_CLOCK"
            ts_success_rate = 1.0
            ts_cols = ["Synthetic (Assumed Sequential)"]
        else:
            try:
                if ts_method == "unix_epoch":
                    num_ts = pd.to_numeric(chunk[ts_cols[0]], errors='coerce')
                    parsed_ts = pd.to_datetime(num_ts, unit='s', errors='coerce')
                elif len(ts_cols) == 2:
                    combined = chunk[ts_cols[0]].astype(str) + " " + chunk[ts_cols[1]].astype(str)
                    parsed_ts = pd.to_datetime(combined, errors='coerce')
                else:
                    parsed_ts = pd.to_datetime(chunk[ts_cols[0]], errors='coerce')
            except Exception:
                return CompatibilityReport(status="error", compatibility="UNSUPPORTED", error_code="INVALID_TIMESTAMP", reason="TEMPORAL DATA REQUIRED")
            
        chunk = chunk[parsed_ts.notna()].copy()
        if len(chunk) == 0: continue
            
        parsed_ts = parsed_ts.dropna()
        chunk['__parsed_ts'] = parsed_ts.dt.floor('30s')
        
        target_features = FALLBACK_FEATURES if is_generic else canonical_features
        
        # 2. Extract features
        for can in target_features:
            if can in canonical_map:
                chunk[can] = chunk[canonical_map[can]]
            elif can == "Subflow Fwd Pkts":
                chunk[can] = chunk["Tot Fwd Pkts"]
            elif can == "Subflow Bwd Pkts":
                chunk[can] = chunk["Tot Bwd Pkts"]
            elif can == "flow_count" and can not in chunk.columns:
                chunk[can] = 1.0
                
        # Handle Protocol Encoding for Generic (and Native if it has strings)
        if "Protocol" in target_features:
            chunk["Protocol"] = chunk["Protocol"].apply(encode_protocol)
                
        # 3. Clean numeric
        for can in target_features:
            if can == "flow_count": continue
            chunk[can] = pd.to_numeric(chunk[can], errors='coerce').fillna(0)
            chunk[can] = chunk[can].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        # 4. Aggregate
        grouped = chunk.groupby('__parsed_ts')[target_features].agg(['sum', 'count'])
        
        for ts, row in grouped.iterrows():
            ts_val = ts.timestamp()
            sums = row.xs('sum', level=1).values
            counts = row.xs('count', level=1).values
            
            if ts_val not in bins:
                bins[ts_val] = {'sum': np.zeros(len(target_features), dtype=np.float64), 'count': np.zeros(len(target_features), dtype=np.float64)}
                
            bins[ts_val]['sum'] += sums
            bins[ts_val]['count'] += counts

    if not bins:
        return CompatibilityReport(status="error", compatibility="UNSUPPORTED", error_code="INVALID_CSV", reason="No valid temporal traffic found.")
        
    sorted_ts = sorted(bins.keys())
    target_features = FALLBACK_FEATURES if is_generic else canonical_features
    
    bin_matrix = np.zeros((len(sorted_ts), len(target_features)), dtype=np.float32)
    for i, ts in enumerate(sorted_ts):
        c_count = bins[ts]['count']
        c_sum = bins[ts]['sum']
        safe_count = np.where(c_count == 0, 1, c_count)
        means = c_sum / safe_count
        
        if "flow_count" in target_features:
            flow_count_idx = target_features.index("flow_count")
            means[flow_count_idx] = c_sum[flow_count_idx]
            
        bin_matrix[i, :] = means.astype(np.float32)
        
    best_start, best_len = 0, 1
    curr_start, curr_len = 0, 1
    for i in range(1, len(sorted_ts)):
        if np.isclose(sorted_ts[i] - sorted_ts[i-1], 30.0, atol=1.0):
            curr_len += 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = curr_start
        else:
            curr_start = i
            curr_len = 1
            
    if best_len < 10:
        return CompatibilityReport(
            status="error", compatibility="UNSUPPORTED", error_code="INSUFFICIENT_HISTORY",
            reason=f"Requires 10 consecutive 30-second states. Detected: {best_len}.",
            missing_requirements=["10 consecutive 30-second states"]
        )

    contiguous_block = bin_matrix[best_start:best_start+best_len]
    
    if is_generic:
        assert contiguous_block.shape[1] == 6
        scaled_mat = chronex_service._FALLBACK_SCALER.transform(contiguous_block).astype(np.float32)
        compatibility = "GENERIC"
    else:
        assert contiguous_block.shape[1] == 71
        scaled_mat = chronex_service._SCALER.transform(contiguous_block).astype(np.float32)
        is_native = len([m for m in mapped_features if m.method == "alias"]) == 0
        compatibility = "NATIVE" if is_native else "ADAPTABLE"
    
    upload_id = str(uuid.uuid4())
    save_path = TEMP_DIR / f"{upload_id}.npy"
    np.save(save_path, scaled_mat)
    
    temporal_range = ""
    if len(sorted_ts) > 0:
        start_ts = pd.Timestamp(sorted_ts[0], unit='s').strftime('%Y-%m-%d %H:%M:%S')
        end_ts = pd.Timestamp(sorted_ts[-1], unit='s').strftime('%Y-%m-%d %H:%M:%S')
        temporal_range = f"{start_ts} to {end_ts}"
    
    return CompatibilityReport(
        status="ready", compatibility=compatibility, upload_id=upload_id,
        schema_flags=schema_flags, flow_count=total_flow_count,
        usable_windows=best_len, usable_sequences=max(0, best_len - 9),
        mapped_features=mapped_features, derived_features=derived_features, missing_features=[],
        timestamp_column=" + ".join(ts_cols),
        timestamp_detection_method=ts_method,
        timestamp_parse_success_rate=ts_success_rate,
        temporal_range=temporal_range,
        window_seconds=30
    )

def load_upload(upload_id: str) -> np.ndarray:
    save_path = TEMP_DIR / f"{upload_id}.npy"
    if not save_path.exists(): raise FileNotFoundError("Upload expired")
    return np.load(save_path)
