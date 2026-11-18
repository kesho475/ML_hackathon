# PathSnap: map-matching

Scope: EDA, features, 5-model comparison, Optuna tuning, HMM + Viterbi decoding, FastAPI backend (Laptop 1) and Streamlit UI (Laptop 2).

Python 3.14 (developed and tested on 3.14.4). Create the virtual environment first, then install:
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
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

## Artifacts
Everything the API needs at inference is in these files plus the code in `src/` (nothing is read from `results/`):

| File | What it is | Produced by |
|---|---|---|
| `models/final_model.joblib` | The tuned `HistGradientBoostingClassifier` (scikit-learn 1.7.2, 9 features), loaded once by `src/pipeline.py` | `python src/bayes_opt.py --model hgb` (refit on all train with the best parameters) |
| `configs/config.yaml` | Runtime settings: `K`, `radius_m`, `eval_trips_dir`, `decoder`, and the HMM parameters under `hmm:` | Edited by hand. The `hmm:` values come from the grid search in `python -m src.evaluate_hmm` (`results/csv/hmm_param_search.csv`) |
| `configs/best_params.yaml` | The best Optuna hyperparameters (a record of the tuning; not read at inference) | `src/bayes_opt.py` |
| `models/metadata.json` | Export record: library and Python versions, the 9 feature names, hyperparameters | Maintained by hand when the model is re-saved. Update it after re-running `bayes_opt.py` |

- The other `models/*.joblib` files are the comparison models from `src/train_models.py`. They are not used at inference.
- **Features are rebuilt at inference** by `build_features` in `src/build_dataset.py` from the raw trip folder (candidate arcs plus the 9 features). The saved model is the bare classifier: HGB needs no fitted scaler or imputer.
- Keep scikit-learn at 1.7.2 (pinned in `requirements.txt`); the model file was pickled with it.

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
