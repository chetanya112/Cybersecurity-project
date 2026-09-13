import io
import os
import time
import shutil
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from backend.schemas.prediction import PredictionResponse, HealthResponse
from backend.schemas.dataset import CompatibilityReport
from backend.services.chronex_service import (
    initialize_service, get_dataset_info, get_sequence, 
    process_sequence, process_sequence_fallback
)
from backend.services.dataset_adapter import process_upload, load_upload, TEMP_DIR

app = FastAPI(title="CHRONEX API", description="Network Threat Intelligence Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHRONEX_MAX_UPLOAD_MB = int(os.getenv("CHRONEX_MAX_UPLOAD_MB", 500))

def cleanup_temp_uploads():
    """Remove uploads older than 1 hour or on startup."""
    if TEMP_DIR.exists():
        now = time.time()
        for f in TEMP_DIR.glob("*.npy"):
            try:
                if os.path.getmtime(f) < now - 3600:
                    f.unlink()
            except Exception:
                pass

@app.on_event("startup")
async def startup_event():
    # Pre-load heavy PyTorch models and SHAP explainers at startup
    initialize_service()
    
    # Cleanup abandoned temporary uploads
    if TEMP_DIR.exists():
        for f in TEMP_DIR.glob("*.npy"):
            try:
                f.unlink()
            except Exception:
                pass

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        inference_mode="Production (Direct Future-State Decoder)",
        features=71,
        classes=14,
        sequence_length=10,
        window_seconds=30
    )

@app.get("/api/dataset/info")
async def dataset_info() -> Dict[str, Any]:
    try:
        return get_dataset_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/sequence/{index}", response_model=PredictionResponse)
async def dataset_sequence(index: int):
    try:
        seq, ground_truth = get_sequence(index)
        response = process_sequence(seq, posthoc_ground_truth=ground_truth)
        return response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload", response_model=CompatibilityReport)
async def upload_csv(file: UploadFile = File(...)):
    # File validation
    if file.size and file.size > CHRONEX_MAX_UPLOAD_MB * 1024 * 1024:
        size_mb = round(file.size / (1024 * 1024), 1)
        return CompatibilityReport(
            status="error", 
            compatibility="UNSUPPORTED", 
            error_code="UPLOAD_TOO_LARGE",
            reason=f"File size {size_mb} MB exceeds limit of {CHRONEX_MAX_UPLOAD_MB} MB."
        )
        
    try:
        if not file.filename.endswith('.csv'):
            return CompatibilityReport(
                status="error", 
                compatibility="UNSUPPORTED", 
                error_code="INVALID_CSV",
                reason="Invalid file format. Please upload a CSV."
            )
            
        report = process_upload(file.file)
        
        # Cleanup old files
        cleanup_temp_uploads()
        
        return report
        
    except Exception as e:
        return CompatibilityReport(
            status="error", 
            compatibility="UNSUPPORTED", 
            error_code="PROCESSING_ERROR",
            reason=f"Unexpected error: {str(e)}"
        )

@app.post("/api/analyze/{upload_id}", response_model=PredictionResponse)
async def analyze_upload(upload_id: str):
    """
    Consume the prepared canonical sequence and run CHRONEX inference.
    """
    try:
        if not upload_id or ".." in upload_id or "/" in upload_id or "\\" in upload_id:
            raise HTTPException(status_code=400, detail="Invalid upload_id")
            
        # Load prepared state
        seq = load_upload(upload_id)
        
        # Delete file after successful load
        try:
            (TEMP_DIR / f"{upload_id}.npy").unlink()
        except Exception:
            pass
            
        # Ensure we only use the last 10 windows
        seq = seq[-10:]
        
        if seq.shape[1] == 71:
            if seq.shape != (10, 71):
                raise HTTPException(status_code=400, detail=f"Invalid NATIVE sequence shape: {seq.shape}. Expected (10, 71)")
            return process_sequence(seq, posthoc_ground_truth=None)
        elif seq.shape[1] == 6:
            if seq.shape != (10, 6):
                raise HTTPException(status_code=400, detail=f"Invalid GENERIC sequence shape: {seq.shape}. Expected (10, 6)")
            return process_sequence_fallback(seq, posthoc_ground_truth=None)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid sequence feature count: {seq.shape[1]}")
            
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
