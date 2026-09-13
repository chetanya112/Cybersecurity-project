# CHRONEX Repository Cleanup Report (SIH Final Release)

## Objective
Finalize the repository by safely removing obsolete prototyping code, Streamlit legacy files, and old iterations of models/training scripts, establishing the **FastAPI + React Dual-Engine Architecture** as the single source of truth for the release.

## Actions Executed

1. **Obsolete App Files Removed**:
   - `src/app.py`, `src/app_backup.py`, `src/app_new.py`

2. **Obsolete Scripts Removed**:
   - `src/train_lstm.py` (Superseded by `train_phase4.py`)
   - `src/rollout.py`, `src/build_sequences.py`
   - `extract_logo.py`, `clean_data_current.txt`
   - `Chronex.ipynb` (Superseded by `01_explore_data.ipynb`)
   - `src/audit_counts.py`, `src/audit_sequences.py`

3. **Superseded Model Checkpoints Removed**:
   - `models/trained_lstm_binary.pt`
   - `models/benign_train_indices.npy`
   *(Note: The old `trained_lstm_multiclass.pt` was removed, but `benchmark_baseline.py` was updated to properly use the new `phase4` classifier model for valid comparisons).*

4. **Retained for SIH Reproducibility & Evaluation**:
   - `models/lr_baseline_multiclass.joblib`
   - `models/benchmark_results.json`
   - `src/benchmark_baseline.py` (Adapted to benchmark against the latest `phase4` world model)
   - `notebooks/01_explore_data.ipynb`

5. **Scratch & Local Data Policy Applied**:
   - `.gitignore` was updated to protect `scratch/` files locally while ensuring they aren't pushed to the final repo.
   - Raw CSVs and processed intermediate artifacts are correctly ignored.

## Verification
- **Frontend Build**: `npm run build` passes with zero functional errors. Types fixed.
- **Backend Build**: FastAPI successfully initializes the dual ML models.
- **Audits**: `audit_runner.py` executed confirming data leakage constraints, shape contracts, scaling correctness, and probability distributions.
- **Regression Tests**: `test_csv_ingestion.py` confirms that the backend correctly routes to the new NATIVE, ADAPTABLE (Fallback), or UNSUPPORTED paths.

The repository is now fully aligned with the Phase 12 architecture and is **READY TO STAGE**.
