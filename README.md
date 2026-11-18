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
