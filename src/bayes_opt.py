"""Bayesian hyperparameter search for PathSnap candidate classifiers."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "point_idx", "cand_arc_id", "dist_m", "dist_rank", "dist_minus_min",
    "heading_diff", "arc_len", "pos_on_arc", "speed", "dt", "n_cands",
]


def make_model(model: str, params: dict[str, Any] | None = None) -> Any:
    params = dict(params or {})
    if model == "hgb":
        return HistGradientBoostingClassifier(random_state=42, **params)
    if model == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(solver="saga", max_iter=500, random_state=42, **params)),
        ])
    raise ValueError("--model must be hgb or logreg")


def point_top1(y: np.ndarray, probabilities: np.ndarray, points: pd.Series) -> float:
    frame = pd.DataFrame({"point": points.to_numpy(), "y": y, "p": probabilities})
    chosen = frame.loc[frame.groupby("point")["p"].idxmax()]
    return float((chosen["y"].to_numpy() == 1).mean())


def evaluate(model: Any, data: pd.DataFrame, groups: pd.Series | None = None) -> dict[str, float]:
    X = data[FEATURES].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=np.int8)
    point_ids = data["trip_id"].astype(str) + "::" + data["point_idx"].astype(str)
    if groups is None:
        groups = data["trip_id"]
    splitter = GroupKFold(n_splits=3)
    f1s: list[float] = []
    for train_idx, valid_idx in splitter.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[valid_idx])
        f1s.append(float(f1_score(y[valid_idx], pred, average="macro", zero_division=0)))
    return {"cv_macro_f1": float(np.mean(f1s))}


def objective_factory(data: pd.DataFrame, model_name: str):
    X = data[FEATURES].to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=np.int8)
    groups = data["trip_id"].to_numpy()
    splitter = GroupKFold(n_splits=3)

    def objective(trial: optuna.Trial) -> float:
        if model_name == "hgb":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),  # covers slow to fast boosting.
                "max_iter": trial.suggest_int("max_iter", 100, 500),  # spans compact to high-capacity ensembles.
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),  # spans shallow to expressive trees.
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),  # controls noisy small leaves.
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True),  # spans negligible to strong shrinkage.
            }
        else:
            params = {
                "C": trial.suggest_float("C", 1e-3, 100, log=True),  # spans strong to weak regularization.
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),  # compares sparse and dense coefficients.
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),  # tests prevalence-aware weighting.
            }
        scores: list[float] = []
        for train_idx, valid_idx in splitter.split(X, y, groups):
            estimator = make_model(model_name, params)
            estimator.fit(X[train_idx], y[train_idx])
            pred = estimator.predict(X[valid_idx])
            scores.append(float(f1_score(y[valid_idx], pred, average="macro", zero_division=0)))
        return float(np.mean(scores))

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hgb", "logreg"], required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()
    t0 = time.perf_counter()
    train = pd.read_parquet(ROOT / "results/features_train.parquet")
    student = pd.read_parquet(ROOT / "results/features_student_test.parquet")
    (ROOT / "results/csv").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/plots").mkdir(parents=True, exist_ok=True)
    (ROOT / "configs").mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_factory(train, args.model), n_trials=args.n_trials, timeout=480)
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Trials finished: {len(finished)}")

    trials = study.trials_dataframe(attrs=("number", "value", "state", "duration", "params"))
    trials.to_csv(ROOT / "results/csv/optuna_trials.csv", index=False)
    best_params = dict(study.best_trial.params)
    with (ROOT / "configs/best_params.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"model": args.model, "params": best_params}, handle, sort_keys=False)

    tuned = make_model(args.model, best_params)
    untuned = make_model(args.model)
    cv_tuned = evaluate(tuned, train)
    cv_untuned = evaluate(untuned, train)

    def final_metrics(estimator: Any) -> dict[str, float]:
        X = train[FEATURES].to_numpy(dtype=np.float32)
        y = train["label"].to_numpy(dtype=np.int8)
        estimator.fit(X, y)
        Xs = student[FEATURES].to_numpy(dtype=np.float32)
        ys = student["label"].to_numpy(dtype=np.int8)
        started = time.perf_counter()
        probabilities = estimator.predict_proba(Xs)[:, 1]
        elapsed = time.perf_counter() - started
        predictions = (probabilities >= 0.5).astype(np.int8)
        points = student["trip_id"].astype(str) + "::" + student["point_idx"].astype(str)
        return {"top1_accuracy": point_top1(ys, probabilities, points),
                "macro_f1": float(f1_score(ys, predictions, average="macro", zero_division=0)),
                "log_loss": float(log_loss(ys, probabilities, labels=[0, 1])),
                "inference_time_s": elapsed}

    rows = []
    for name, estimator, cv in [("tuned", tuned, cv_tuned), ("untuned", untuned, cv_untuned)]:
        metrics = final_metrics(estimator)
        rows.append({"model": name, **cv, **metrics, "total_optimisation_time_s": time.perf_counter() - t0,
                     "n_trials": len(finished)})
    pd.DataFrame(rows).to_csv(ROOT / "results/csv/tuned_vs_untuned.csv", index=False)
    joblib.dump(tuned, ROOT / "models/final_model.joblib")

    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    ax = plot_optimization_history(study); ax.figure.savefig(ROOT / "results/plots/optimization_history.png"); plt.close(ax.figure)
    if len(finished) > 1:
        ax = plot_param_importances(study); ax.figure.savefig(ROOT / "results/plots/parameter_importance.png"); plt.close(ax.figure)
    comparison = pd.DataFrame({"metric": ["cv_macro_f1", "top1_accuracy", "macro_f1", "log_loss"],
                               "tuned": [rows[0]["cv_macro_f1"], rows[0]["top1_accuracy"], rows[0]["macro_f1"], rows[0]["log_loss"]],
                               "untuned": [rows[1]["cv_macro_f1"], rows[1]["top1_accuracy"], rows[1]["macro_f1"], rows[1]["log_loss"]]})
    comparison.set_index("metric").plot(kind="bar", figsize=(9, 5)); plt.tight_layout(); plt.savefig(ROOT / "results/plots/tuned_vs_untuned.png"); plt.close()
    print(f"Total optimisation time: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
