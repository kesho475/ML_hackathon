# PathSnap: map-matching (ML steps 1-6)

Scope: EDA, features, 5-model comparison, model selection, Optuna tuning. No HMM/API/UI yet.

```
pip install -r requirements.txt
```
Data: put trips under `data/raw/{train,student_test}/<trip_id>/` (gitignored). Settings: `configs/config.yaml`. Feature columns: `docs/feature_schema.md`.

## Data setup
1. Put the given `Student folder/` in the project root (it has `train/` and `student_test/`).
2. `data/raw/` is gitignored, so create it locally (symlinks or copies):
```
mkdir -p data/raw
ln -s "$PWD/Student folder/train" data/raw/train
ln -s "$PWD/Student folder/student_test" data/raw/student_test
```
3. Environment:
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run the API
Laptop 1 (model + API). Needs `models/final_model.joblib` and the trips in `data/raw/` (see Data setup).
```
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
- `GET /health`, `GET /trips`, `POST /predict` with `{"trip_id": "00000000"}`.
- The trips served come from `eval_trips_dir` in `configs/config.yaml` (default `data/raw/student_test`).
- Every request is logged to the terminal (time, client IP, endpoint, trip_id, n_points, top1, processing time).
- The decoder is `hmm` (HMM + Viterbi, `src/hmm.py`). The classifier-only decoder is still available: set `decoder: classifier` in `configs/config.yaml`. New decoders register in `DECODERS` in `src/pipeline.py`.

## Run the UI
Laptop 2 (evaluator UI). It reads no data or models, only the API.
```
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run ui/app.py
```
Open the page, set the API URL in the sidebar (default `http://localhost:8000`; use Laptop 1's LAN address, e.g. `http://192.168.x.x:8000`), click "Check connection", pick a trip and click "Run prediction".
