"""Train and compare 5 candidate-arc classifiers. Run from project root: python src/train_models.py [--quick]"""
import argparse, os, sys, time
from functools import lru_cache
import joblib, matplotlib, numpy as np, pandas as pd, yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
sys.path.insert(0, os.path.dirname(__file__))
from io_utils import load_trip
from metrics import candidate_metrics, route_continuity, top1_accuracy, top1_table

cfg = yaml.safe_load(open("configs/config.yaml"))
SEED = cfg["seed"]
FEATURES = ["dist_m", "dist_rank", "dist_minus_min", "heading_diff", "arc_len", "pos_on_arc", "speed", "dt", "n_cands"]
KNN_N = 100_000

# name: (family, key config, estimator, fit on a 100k subsample?)
MODELS = {
    "GaussianNB": ("Linear/probabilistic", "defaults", GaussianNB(), False),
    "DecisionTree": ("Tree", "max_depth=20", DecisionTreeClassifier(max_depth=20, random_state=SEED), False),
    "LogisticRegression": ("Linear", "max_iter=300, StandardScaler",
                           make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=SEED)), False),
    "HistGradientBoosting": ("Boosting", "defaults", HistGradientBoostingClassifier(random_state=SEED), False),
    "KNN": ("Instance-based", f"k=15, StandardScaler, {KNN_N // 1000}k-row train subsample",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, n_jobs=-1)), True),
}


def subsample(X, y, seed=SEED):
    if len(X) <= KNN_N:
        return X, y
    i = np.random.default_rng(seed).choice(len(X), KNN_N, replace=False)
    return X[i], y[i]


def fit(est, X, y, sub):
    est = clone(est)
    return est.fit(*subsample(X, y)) if sub else est.fit(X, y)


def bar(ax_vals, labels, title, path, yerr=None, ylabel="", log=False):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    b = ax.bar(labels, ax_vals, yerr=yerr, capsize=4, color="tab:blue")
    ax.bar_label(b, fmt="%.3f", padding=2, fontsize=8)
    ax.set_title(title); ax.set_ylabel(ylabel); plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    if log: ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true", help="use 10 train trips")
    quick = ap.parse_args().quick
    sfx = "_quick" if quick else ""
    tr = pd.read_parquet("results/features_train.parquet")
    te = pd.read_parquet("results/features_student_test.parquet")
    if quick:
        tr = tr[tr.trip_id.isin(sorted(tr.trip_id.unique())[:10])]
    print(f"train: {tr.trip_id.nunique()} trips, {len(tr)} rows | student_test: {te.trip_id.nunique()} trips, {len(te)} rows")
    print(f"NOTE: KNN is fit on a {KNN_N:,}-row random subsample of train (seed {SEED}), also inside each CV fold.")
    Xtr, ytr, g = tr[FEATURES].to_numpy("float32"), tr.label.to_numpy(), tr.trip_id.to_numpy()
    Xte, yte = te[FEATURES].to_numpy("float32"), te.label.to_numpy()

    @lru_cache(None)
    def arcs_of(tid):
        return load_trip(os.path.join(cfg["test_dir"], tid))[2]

    rows, folds = [], []
    for name, (family, conf, est, sub) in MODELS.items():
        t0 = time.time()
        f1s, lls = [], []
        for k, (a, b) in enumerate(GroupKFold(n_splits=cfg["cv_folds"]).split(Xtr, ytr, g)):
            p = fit(est, Xtr[a], ytr[a], sub).predict_proba(Xtr[b])[:, 1]
            m = candidate_metrics(ytr[b], p)
            f1s.append(m["macro_f1"]); lls.append(m["log_loss"])
            folds.append({"Model": name, "Fold": k + 1, "Macro F1": m["macro_f1"], "Log Loss": m["log_loss"]})
        t1 = time.perf_counter(); model = fit(est, Xtr, ytr, sub); train_s = time.perf_counter() - t1
        t1 = time.perf_counter(); p = model.predict_proba(Xte)[:, 1]; inf_s = time.perf_counter() - t1
        m = candidate_metrics(yte, p); top = top1_table(te, p)
        cm = confusion_matrix(yte, (p >= .5).astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(3.6, 3.4)); ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{name} (student_test)", fontsize=9); fig.tight_layout()
        fig.savefig(f"results/plots/confusion_{name}{sfx}.png", dpi=110); plt.close(fig)
        if not quick:
            path = f"models/{name}.joblib"; joblib.dump(model, path)
            mb = os.path.getsize(path) / 1e6
            if mb > 90:
                with open(".gitignore", "a") as f: f.write(f"{path}\n")
                print(f"NOTE: {path} is {mb:.0f} MB (> 90 MB), added to .gitignore")
        rows.append({"Model": name, "Family": family, "Key Config": conf,
                     "CV Macro F1 (mean┬▒std)": f"{np.mean(f1s):.4f}┬▒{np.std(f1s):.4f}", "_cv_f1": np.mean(f1s),
                     "Top-1 Accuracy": top1_accuracy(top), "Route Continuity": route_continuity(top, arcs_of),
                     "Test Macro F1": m["macro_f1"], "Precision": m["precision"], "Recall": m["recall"], "Log Loss": m["log_loss"],
                     "Train Time": train_s, "Inference Time": inf_s, "_cv_std": np.std(f1s)})
        print(f"{name}: done in {time.time() - t0:.0f}s (CV F1 {np.mean(f1s):.4f}, test Top-1 {rows[-1]['Top-1 Accuracy']:.4f})", flush=True)

    R = pd.DataFrame(rows)
    cols = ["Model", "Family", "Key Config", "CV Macro F1 (mean┬▒std)", "Top-1 Accuracy", "Route Continuity", "Test Macro F1",
            "Precision", "Recall", "Log Loss", "Train Time", "Inference Time"]
    R[cols].to_csv(f"results/csv/model_comparison{sfx}.csv", index=False, float_format="%.5f")
    pd.DataFrame(folds).to_csv(f"results/csv/cv_folds{sfx}.csv", index=False, float_format="%.5f")
    n = R.Model.tolist()
    bar(R._cv_f1, n, "CV macro F1, GroupKFold by trip (train)", f"results/plots/cv_macro_f1{sfx}.png", yerr=R._cv_std, ylabel="macro F1")
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(n))
    for off, c, lab in [(-.2, "Top-1 Accuracy", "Top-1 accuracy"), (.2, "Test Macro F1", "macro F1")]:
        b = ax.bar(x + off, R[c], .4, label=lab); ax.bar_label(b, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(n, rotation=20, ha="right"); ax.legend(); ax.set_title("student_test: Top-1 accuracy and macro F1")
    fig.tight_layout(); fig.savefig(f"results/plots/test_top1_f1{sfx}.png", dpi=110); plt.close(fig)
    bar(R["Log Loss"], n, "student_test log loss (lower is better)", f"results/plots/log_loss{sfx}.png", ylabel="log loss",
        log=R["Log Loss"].max() / R["Log Loss"].min() > 20)

    # selection: primary = Top-1 accuracy (rounded to 4 dp to define ties), tie-break = log loss
    R["_t1"] = R["Top-1 Accuracy"].round(4)
    S = R.sort_values(["_t1", "Log Loss"], ascending=[False, True]); best = S.iloc[0]
    print("\n" + R[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nCV macro F1 minus student-test macro F1 (overfitting / shift check):")
    for _, r in R.iterrows(): print(f"  {r.Model:22s} gap = {r._cv_f1 - r['Test Macro F1']:+.4f}")
    print(f"\nCHOSEN: {best.Model}\n  Highest student-test Top-1 accuracy ({best['Top-1 Accuracy']:.4f}); route continuity {best['Route Continuity']:.4f}, "
          f"log loss {best['Log Loss']:.4f} (tie-break metric).\n  CV macro F1 {best['CV Macro F1 (mean┬▒std)']}, train {best['Train Time']:.1f}s, inference {best['Inference Time']:.2f}s.")


if __name__ == "__main__":
    main()
