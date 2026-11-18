"""Bayesian hyperparameter search (Optuna TPE) for the selected PathSnap candidate classifier."""

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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metrics import candidate_metrics, top1_accuracy, top1_table

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
# exactly the 9 features in docs/feature_schema.md; trip_id / point_idx / cand_arc_id are identifiers, not features
FEATURES = ["dist_m", "dist_rank", "dist_minus_min", "heading_diff", "arc_len", "pos_on_arc", "speed", "dt", "n_cands"]
SEED = 42
# Objective: mean candidate-level macro F1 over 3-fold GroupKFold by trip_id, TRAIN partition only (student_test is never
# used here). 3 folds, not the 5 used for model selection, to fit the time budget; tuned and untuned use the same split.
N_SPLITS = 3


def make_model(model: str, params: dict[str, Any] | None = None) -> Any:
    params = dict(params or {})
    if model == "hgb":
        return HistGradientBoostingClassifier(random_state=SEED, **params)
    if model == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(solver="saga", max_iter=500, random_state=SEED, **params)),
        ])
    raise ValueError("--model must be hgb or logreg")


def cv_macro_f1(model_name: str, params: dict[str, Any] | None, X, y, groups) -> float:
    scores = []
    for tr, va in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        est = make_model(model_name, params).fit(X[tr], y[tr])
        scores.append(candidate_metrics(y[va], est.predict_proba(X[va])[:, 1])["macro_f1"])
    return float(np.mean(scores))


def objective_factory(train: pd.DataFrame, model_name: str):
    X = train[FEATURES].to_numpy(dtype=np.float32)
    y = train["label"].to_numpy(dtype=np.int8)
    groups = train["trip_id"].to_numpy()

    def objective(trial: optuna.Trial) -> float:
        if model_name == "hgb":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),  # covers slow to fast boosting.
                "max_iter": trial.suggest_int("max_iter", 100, 300),  # compact to mid-size ensembles, bounded for time.
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),  # spans shallow to expressive trees.
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),  # controls noisy small leaves.
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10, log=True),  # negligible to strong shrinkage.
            }
        else:
            params = {
                "C": trial.suggest_float("C", 1e-3, 100, log=True),  # spans strong to weak regularization.
                "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),  # compares sparse and dense coefficients.
                "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),  # prevalence-aware weighting.
            }
        return cv_macro_f1(model_name, params, X, y, groups)

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["hgb", "logreg"], required=True)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=2400, help="seconds")
    args = parser.parse_args()
    t0 = time.perf_counter()
    train = pd.read_parquet(ROOT / "results/features_train.parquet")
    student = pd.read_parquet(ROOT / "results/features_student_test.parquet")
    for d in ("results/csv", "results/plots", "configs", "models"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective_factory(train, args.model), n_trials=args.n_trials, timeout=args.timeout)
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Trials finished: {len(finished)}")
    study.trials_dataframe(attrs=("number", "value", "state", "duration", "params")).to_csv(
        ROOT / "results/csv/optuna_trials.csv", index=False)
    best_params = dict(study.best_trial.params)
    with (ROOT / "configs/best_params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({"model": args.model, "params": best_params}, f, sort_keys=False)

    X = train[FEATURES].to_numpy(dtype=np.float32)
    y = train["label"].to_numpy(dtype=np.int8)
    groups = train["trip_id"].to_numpy()
    Xs = student[FEATURES].to_numpy(dtype=np.float32)
    ys = student["label"].to_numpy(dtype=np.int8)

    rows, fitted = [], {}
    for name, params in [("tuned", best_params), ("untuned", None)]:
        # same CV function and split for both; the tuned score equals the best trial's value
        cv = study.best_value if name == "tuned" else cv_macro_f1(args.model, None, X, y, groups)
        est = make_model(args.model, params).fit(X, y)  # refit on all train, evaluated once on student_test
        t1 = time.perf_counter()
        p = est.predict_proba(Xs)[:, 1]
        inf = time.perf_counter() - t1
        m = candidate_metrics(ys, p)
        rows.append({"model": name, "cv_macro_f1": cv, "top1_accuracy": top1_accuracy(top1_table(student, p)),
                     "macro_f1": m["macro_f1"], "log_loss": m["log_loss"], "inference_time_s": inf,
                     "total_optimisation_time_s": time.perf_counter() - t0, "n_trials": len(finished)})
        fitted[name] = est
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results/csv/tuned_vs_untuned.csv", index=False)
    joblib.dump(fitted["tuned"], ROOT / "models/final_model.joblib")

    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    ax = plot_optimization_history(study); ax.figure.savefig(ROOT / "results/plots/optimization_history.png"); plt.close(ax.figure)
    if len(finished) > 1:
        ax = plot_param_importances(study); ax.figure.savefig(ROOT / "results/plots/parameter_importance.png"); plt.close(ax.figure)
    cols = ["cv_macro_f1", "top1_accuracy", "macro_f1", "log_loss"]
    out.set_index("model")[cols].T.plot(kind="bar", figsize=(9, 5), title="Tuned vs untuned")
    plt.tight_layout(); plt.savefig(ROOT / "results/plots/tuned_vs_untuned.png"); plt.close()

    print(f"Total optimisation time: {time.perf_counter() - t0:.1f} s")
    print(out[["model", *cols, "inference_time_s"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("Best params:", best_params)


if __name__ == "__main__":
    main()
