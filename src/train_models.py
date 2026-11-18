import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from metrics import (
    candidate_metrics,
    top1_accuracy,
    route_continuity
)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(ROOT, "results", "features_train.parquet")
TEST_PATH = os.path.join(ROOT, "results", "features_student_test.parquet")

MODEL_DIR = os.path.join(ROOT, "models")
CSV_DIR = os.path.join(ROOT, "results", "csv")
PLOT_DIR = os.path.join(ROOT, "results", "plots")

for folder in [MODEL_DIR, CSV_DIR, PLOT_DIR]:
    os.makedirs(folder, exist_ok=True)


FEATURES = [
    "dist_m",
    "dist_rank",
    "dist_minus_min",
    "heading_diff",
    "arc_len",
    "pos_on_arc",
    "speed",
    "dt",
    "n_cands"
]

TARGET = "label"
KNN_LIMIT = 100_000
RANDOM_STATE = 42


def load_data(quick=False):
    print("Loading Parquet datasets...")

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    required = FEATURES + [TARGET, "trip_id", "point_idx", "cand_arc_id"]

    for name, df in [("Training", train), ("Student test", test)]:
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"{name} Parquet is missing columns: {missing}")

    if quick:
        selected_trips = train["trip_id"].drop_duplicates().head(10)
        train = train[train["trip_id"].isin(selected_trips)].copy()
        print(f"QUICK MODE: using {len(selected_trips)} training trips.")

    # Ensure numeric features and valid binary labels.
    for df in [train, test]:
        df[FEATURES] = df[FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df.dropna(subset=[TARGET], inplace=True)
        df[TARGET] = df[TARGET].astype(int)

        if not set(df[TARGET].unique()).issubset({0, 1}):
            raise ValueError("Label column must contain only 0 and 1.")

    print(f"Training rows: {len(train):,}")
    print(f"Training trips: {train['trip_id'].nunique()}")
    print(f"Student-test rows: {len(test):,}")
    print(f"Student-test trips: {test['trip_id'].nunique()}")

    return train, test


def build_models():
    return {
        "GaussianNB": {
            "family": "Probabilistic",
            "config": "GaussianNB()",
            "model": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GaussianNB())
            ])
        },

        "DecisionTree": {
            "family": "Tree-based",
            "config": "DecisionTreeClassifier(random_state=42)",
            "model": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                ))
            ])
        },

        "LogisticRegression": {
            "family": "Linear",
            "config": "LogisticRegression(max_iter=300)",
            "model": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=300,
                    random_state=RANDOM_STATE
                ))
            ])
        },

        "HistGradientBoosting": {
            "family": "Boosting",
            "config": "HistGradientBoostingClassifier(random_state=42)",
            "model": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE
                ))
            ])
        },

        "KNN": {
            "family": "Instance-based",
            "config": "KNeighborsClassifier(n_neighbors=15)",
            "model": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(
                    n_neighbors=15,
                    n_jobs=-1
                ))
            ])
        }
    }


def sample_knn_data(X, y):
    if len(X) <= KNN_LIMIT:
        return X, y

    rng = np.random.RandomState(RANDOM_STATE)
    indices = rng.choice(len(X), KNN_LIMIT, replace=False)

    return X.iloc[indices], y.iloc[indices]


def fit_model(model, name, X, y):
    model = clone(model)

    if name == "KNN":
        X_fit, y_fit = sample_knn_data(X, y)
    else:
        X_fit, y_fit = X, y

    start = time.perf_counter()
    model.fit(X_fit, y_fit)
    elapsed = time.perf_counter() - start

    return model, elapsed


def get_positive_probabilities(model, X):
    probabilities = model.predict_proba(X)

    classes = model.classes_
    positive_index = list(classes).index(1)

    return probabilities[:, positive_index]


def run_cross_validation(train, name, model):
    print(f"\nRunning 5-fold CV: {name}")

    X = train[FEATURES]
    y = train[TARGET]
    groups = train["trip_id"]

    splitter = GroupKFold(n_splits=5)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if y_train.nunique() < 2:
            raise ValueError(
                f"{name}, fold {fold}: training fold has only one class."
            )

        fitted_model, train_time = fit_model(
            model, name, X_train, y_train
        )

        probabilities = get_positive_probabilities(
            fitted_model, X_val
        )

        metrics = candidate_metrics(y_val, probabilities)

        fold_results.append({
            "Model": name,
            "Fold": fold,
            "Macro F1": metrics["macro_f1"],
            "Log Loss": metrics["log_loss"],
            "Train Time": train_time
        })

        print(
            f"  Fold {fold}: "
            f"Macro F1={metrics['macro_f1']:.4f}, "
            f"Log Loss={metrics['log_loss']:.4f}"
        )

    return fold_results


def evaluate_test(model, name, train, test):
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print(f"\nTraining final {name} model...")

    fitted_model, train_time = fit_model(
        model, name, X_train, y_train
    )

    start = time.perf_counter()
    probabilities = get_positive_probabilities(
        fitted_model, X_test
    )
    inference_time = time.perf_counter() - start

    metrics = candidate_metrics(y_test, probabilities)

    top1 = top1_accuracy(test, probabilities)
    continuity = route_continuity(test, probabilities)

    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
    joblib.dump(fitted_model, model_path)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    if model_size_mb > 90:
        print(
            f"NOTE: {name}.joblib is {model_size_mb:.1f} MB. "
            "Add this file to .gitignore."
        )

    print(
        f"{name}: Test Macro F1={metrics['macro_f1']:.4f}, "
        f"Top-1={top1:.4f}, "
        f"Log Loss={metrics['log_loss']:.4f}"
    )

    return {
        "Model": name,
        "Test Macro F1": metrics["macro_f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Log Loss": metrics["log_loss"],
        "Top-1 Accuracy": top1,
        "Route Continuity": continuity,
        "Train Time": train_time,
        "Inference Time": inference_time,
        "Confusion Matrix": metrics["confusion_matrix"],
        "CV Macro F1": np.nan,
        "CV Std": np.nan
    }


def create_plots(results, cv_df):
    names = [r["Model"] for r in results]

    # 1. Cross-validation Macro F1 with fold error bars.
    cv_summary = cv_df.groupby("Model")["Macro F1"].agg(
        ["mean", "std"]
    ).reindex(names)

    plt.figure(figsize=(10, 5))
    plt.bar(
        names,
        cv_summary["mean"],
        yerr=cv_summary["std"].fillna(0),
        capsize=5
    )
    plt.ylabel("CV Macro F1")
    plt.title("5-Fold GroupKFold Macro F1")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cv_macro_f1.png"))
    plt.close()

    # 2. Student-test Top-1 and Macro F1.
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width / 2,
        [r["Top-1 Accuracy"] for r in results],
        width,
        label="Top-1 Accuracy"
    )
    plt.bar(
        x + width / 2,
        [r["Test Macro F1"] for r in results],
        width,
        label="Macro F1"
    )
    plt.xticks(x, names, rotation=20)
    plt.ylabel("Score")
    plt.title("Student-Test Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "student_test_scores.png"))
    plt.close()

    # 3. Log Loss.
    plt.figure(figsize=(10, 5))
    plt.bar(names, [r["Log Loss"] for r in results])
    plt.ylabel("Log Loss")
    plt.title("Student-Test Log Loss")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "log_loss.png"))
    plt.close()

    # 4. Confusion matrices.
    for r in results:
        cm = r["Confusion Matrix"]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, interpolation="nearest")
        ax.set_title(f"{r['Model']} Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        fig.tight_layout()
        fig.savefig(
            os.path.join(PLOT_DIR, f"{r['Model']}_confusion_matrix.png")
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only the first 10 training trips."
    )
    args = parser.parse_args()

    train, test = load_data(quick=args.quick)

    models = build_models()
    all_cv_results = []
    final_results = []

    for name, details in models.items():
        model = details["model"]

        cv_results = run_cross_validation(train, name, model)
        all_cv_results.extend(cv_results)

        result = evaluate_test(model, name, train, test)

        scores = [x["Macro F1"] for x in cv_results]
        result["CV Macro F1"] = float(np.mean(scores))
        result["CV Std"] = float(np.std(scores, ddof=1))

        result["Family"] = details["family"]
        result["Key Config"] = details["config"]

        final_results.append(result)

    cv_df = pd.DataFrame(all_cv_results)
    cv_df.to_csv(
        os.path.join(CSV_DIR, "cv_folds.csv"),
        index=False
    )

    create_plots(final_results, cv_df)

    comparison = pd.DataFrame([
        {
            "Model": r["Model"],
            "Family": r["Family"],
            "Key Config": r["Key Config"],
            "CV Macro F1 (mean±std)": (
                f"{r['CV Macro F1']:.4f} ± {r['CV Std']:.4f}"
            ),
            "Top-1 Accuracy": r["Top-1 Accuracy"],
            "Route Continuity": "N/A - arc connectivity unavailable",
            "Test Macro F1": r["Test Macro F1"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "Log Loss": r["Log Loss"],
            "Train Time": r["Train Time"],
            "Inference Time": r["Inference Time"]
        }
        for r in final_results
    ])

    comparison.to_csv(
        os.path.join(CSV_DIR, "model_comparison.csv"),
        index=False
    )

    print("\n========== MODEL COMPARISON ==========")
    print(comparison.to_string(index=False))

    # Primary selection: Top-1 accuracy.
    # Tie-break: lower log loss.
    ranked = sorted(
        final_results,
        key=lambda r: (
            r["Top-1 Accuracy"],
            -r["Log Loss"]
        ),
        reverse=True
    )

    best = ranked[0]

    print("\n========== SELECTED MODEL ==========")
    print(f"Selected model: {best['Model']}")
    print(
        f"Reason: Highest student-test Top-1 accuracy "
        f"({best['Top-1 Accuracy']:.4f})."
    )
    print(
        f"Tie-break / probability quality: Log Loss "
        f"({best['Log Loss']:.4f})."
    )

    print("\n========== GENERALISATION CHECK ==========")
    for r in final_results:
        gap = r["CV Macro F1"] - r["Test Macro F1"]

        if gap > 0.10:
            status = "Possible overfitting"
        elif gap < -0.10:
            status = "Test score exceeds CV; investigate split variation"
        else:
            status = "No large CV-to-test gap"

        print(
            f"{r['Model']}: CV-Test Macro F1 gap={gap:.4f} "
            f"-> {status}"
        )

    print("\nRoute continuity: N/A until road-arc connectivity is available.")
    print("All model files, CSVs and plots have been saved.")


if __name__ == "__main__":
    main()