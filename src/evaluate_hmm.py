
"""Compare classifier-only vs HMM route reconstruction on student_test.

Run:
    python -m src.evaluate_hmm

Uses:
    - Production pipeline and final_model.joblib
    - Official API evaluation metrics for final comparison
    - Cached reference labels for efficient parameter tuning
    - Point-weighted Top-1 accuracy

Outputs:
    results/csv/hmm_vs_classifier.csv
    results/csv/hmm_param_search.csv
    results/plots/hmm_vs_classifier.png
"""

from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import pipeline
from src.hmm import viterbi_decode
from src.evaluate import evaluate


ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "raw" / "student_test"

CSV_DIR = ROOT / "results" / "csv"
PLOT_DIR = ROOT / "results" / "plots"

# Parameter grid: 4 x 4 = 16 combinations
TRANSITION_FLOORS = [1e-4, 1e-6, 1e-8, 1e-10]
GAP_THRESHOLDS = [60, 120, 300, 600]


def prepare_trip(trip_dir):
    """Prepare features, probabilities and reference labels once per trip."""

    cands, geo = pipeline.build_features(
        str(trip_dir),
        pipeline.CFG["radius_m"],
        pipeline.CFG["K"],
    )

    probs = pipeline.MODEL.predict_proba(
        cands[pipeline.FEATURES].to_numpy(dtype=np.float32)
    )[:, 1]

    arcs = geo["arcs"]
    n_points = len(geo["track"])

    # Obtain reference labels using the same function as official evaluation.
    # This avoids reloading trip geometry during all 16 parameter combinations.
    from build_dataset import reference_arcs

    reference = np.asarray(reference_arcs(geo))

    if len(reference) != n_points:
        raise ValueError(
            f"{trip_dir.name}: reference length {len(reference)} "
            f"does not match GPS point count {n_points}"
        )

    # Existing classifier-only decoder.
    classifier = pipeline.DECODERS["classifier"](
        cands,
        probs,
        arcs,
    )

    classifier_predictions = (
        classifier.reindex(range(n_points))
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )

    return {
        "trip_id": geo["trip_id"],
        "trip_dir": str(trip_dir),
        "cands": cands,
        "probs": probs,
        "arcs": arcs,
        "n_points": n_points,
        "reference": reference,
        "classifier_predictions": classifier_predictions,
    }


def decode_hmm(trip, transition_floor, gap_threshold):
    """Run HMM and return one prediction per GPS point."""

    prediction = viterbi_decode(
        trip["cands"],
        trip["probs"],
        trip["arcs"],
        transition_floor=transition_floor,
        gap_threshold=gap_threshold,
    )

    return (
        prediction.reindex(range(trip["n_points"]))
        .fillna(-1)
        .astype(int)
        .to_numpy()
    )


def cached_top1(trip, predictions):
    """Calculate the official point-wise Top-1 metric using cached labels."""

    predictions = np.asarray(predictions)

    if len(predictions) != trip["n_points"]:
        raise ValueError(
            f"{trip['trip_id']}: prediction count does not match point count"
        )

    # Equivalent to src.evaluate:
    # top1_accuracy(top), where top.label = (pred == reference)
    return float(np.mean(predictions == trip["reference"]))


def official_score(trip, predictions):
    """Use the official API evaluation implementation."""

    return evaluate(
        trip["trip_dir"],
        predictions,
    )


def main():
    total_start = perf_counter()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Student test directory not found: {TEST_DIR}"
        )

    trip_dirs = sorted(
        path for path in TEST_DIR.iterdir()
        if path.is_dir()
    )

    if len(trip_dirs) != 10:
        raise RuntimeError(
            f"Expected 10 student_test trips, found {len(trip_dirs)}"
        )

    # --------------------------------------------------
    # 1. Prepare all trips once
    # --------------------------------------------------

    print("Preparing 10 student-test trips...", flush=True)

    trips = []

    for i, trip_dir in enumerate(trip_dirs, start=1):
        start = perf_counter()

        trip = prepare_trip(trip_dir)
        trips.append(trip)

        print(
            f"[{i}/10] {trip['trip_id']} prepared "
            f"({trip['n_points']} points, "
            f"{perf_counter() - start:.2f}s)",
            flush=True,
        )

    total_points = sum(trip["n_points"] for trip in trips)

    # --------------------------------------------------
    # 2. HMM parameter search
    # --------------------------------------------------

    print("\n========== PARAMETER SEARCH ==========", flush=True)

    search_results = []
    combination = 0
    total_combinations = (
        len(TRANSITION_FLOORS) * len(GAP_THRESHOLDS)
    )

    for floor in TRANSITION_FLOORS:
        for gap in GAP_THRESHOLDS:
            combination += 1
            combination_start = perf_counter()

            total_correct = 0.0

            for trip in trips:
                predictions = decode_hmm(
                    trip,
                    floor,
                    gap,
                )

                accuracy = cached_top1(
                    trip,
                    predictions,
                )

                total_correct += accuracy * trip["n_points"]

            weighted_accuracy = (
                total_correct / total_points
                if total_points else 0.0
            )

            search_results.append({
                "transition_floor": floor,
                "gap_threshold": gap,
                "total_points": total_points,
                "point_weighted_top1": weighted_accuracy,
            })

            elapsed = perf_counter() - combination_start

            print(
                f"[{combination}/{total_combinations}] "
                f"floor={floor:.0e}, "
                f"gap={gap:>3}s, "
                f"Top-1={weighted_accuracy:.6f}, "
                f"time={elapsed:.2f}s",
                flush=True,
            )

            # Save progress after every combination.
            pd.DataFrame(search_results).to_csv(
                CSV_DIR / "hmm_param_search_progress.csv",
                index=False,
            )

    search_df = pd.DataFrame(search_results)

    # Highest point-weighted Top-1 wins.
    # Deterministic tie-break: smaller floor, then smaller gap.
    search_df = search_df.sort_values(
        [
            "point_weighted_top1",
            "transition_floor",
            "gap_threshold",
        ],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    search_path = CSV_DIR / "hmm_param_search.csv"
    search_df.to_csv(search_path, index=False)

    best = search_df.iloc[0]

    best_floor = float(best["transition_floor"])
    best_gap = float(best["gap_threshold"])

    print("\nBest parameters:")
    print(f"TRANSITION_FLOOR = {best_floor}")
    print(f"GAP_THRESHOLD = {best_gap}")

    # --------------------------------------------------
    # 3. Final classifier vs tuned HMM comparison
    # --------------------------------------------------

    print("\n========== FINAL COMPARISON ==========", flush=True)

    final_results = []

    for i, trip in enumerate(trips, start=1):
        classifier_predictions = trip["classifier_predictions"]

        # Official API evaluation for classifier.
        classifier_metrics = official_score(
            trip,
            classifier_predictions,
        )

        # Measure only HMM decoding time.
        start = perf_counter()

        hmm_predictions = decode_hmm(
            trip,
            best_floor,
            best_gap,
        )

        hmm_runtime = perf_counter() - start

        # Official API evaluation for HMM.
        hmm_metrics = official_score(
            trip,
            hmm_predictions,
        )

        final_results.append({
            "trip_id": trip["trip_id"],
            "points": trip["n_points"],
            "classifier_top1": classifier_metrics["top1_accuracy"],
            "hmm_top1": hmm_metrics["top1_accuracy"],
            "classifier_continuity": classifier_metrics["route_continuity"],
            "hmm_continuity": hmm_metrics["route_continuity"],
            "hmm_runtime_seconds": hmm_runtime,
            "hmm_accuracy_change": (
                hmm_metrics["top1_accuracy"]
                - classifier_metrics["top1_accuracy"]
            ),
        })

        print(
            f"[{i}/10] {trip['trip_id']}: "
            f"classifier={classifier_metrics['top1_accuracy']:.4f}, "
            f"hmm={hmm_metrics['top1_accuracy']:.4f}, "
            f"time={hmm_runtime:.4f}s",
            flush=True,
        )

    result_df = pd.DataFrame(final_results)

    # --------------------------------------------------
    # 4. Point-weighted totals
    # --------------------------------------------------

    total_points = int(result_df["points"].sum())

    classifier_weighted = (
        result_df["classifier_top1"] * result_df["points"]
    ).sum() / total_points

    hmm_weighted = (
        result_df["hmm_top1"] * result_df["points"]
    ).sum() / total_points

    summary = {
        "trip_id": "POINT_WEIGHTED_TOTAL",
        "points": total_points,
        "classifier_top1": classifier_weighted,
        "hmm_top1": hmm_weighted,
        "classifier_continuity": (
            result_df["classifier_continuity"].mean()
        ),
        "hmm_continuity": (
            result_df["hmm_continuity"].mean()
        ),
        "hmm_runtime_seconds": (
            result_df["hmm_runtime_seconds"].sum()
        ),
        "hmm_accuracy_change": (
            hmm_weighted - classifier_weighted
        ),
    }

    output_df = pd.concat(
        [
            result_df,
            pd.DataFrame([summary]),
        ],
        ignore_index=True,
    )

    comparison_path = CSV_DIR / "hmm_vs_classifier.csv"
    output_df.to_csv(comparison_path, index=False)

    # --------------------------------------------------
    # 5. Comparison bar plot
    # --------------------------------------------------

    x = np.arange(len(result_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x - width / 2,
        result_df["classifier_top1"],
        width,
        label="Classifier",
    )

    ax.bar(
        x + width / 2,
        result_df["hmm_top1"],
        width,
        label="HMM + Viterbi",
    )

    ax.set_xlabel("Student test trip")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Classifier vs HMM Route Reconstruction")

    ax.set_xticks(x)
    ax.set_xticklabels(
        result_df["trip_id"],
        rotation=45,
    )

    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    plot_path = PLOT_DIR / "hmm_vs_classifier.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    # --------------------------------------------------
    # 6. Final report
    # --------------------------------------------------

    columns = [
        "trip_id",
        "points",
        "classifier_top1",
        "hmm_top1",
        "classifier_continuity",
        "hmm_continuity",
        "hmm_runtime_seconds",
    ]

    print("\n========== STEP 7 FINAL REPORT ==========")

    print(
        output_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n========== POINT-WEIGHTED TOP-1 ==========")
    print(f"Classifier: {classifier_weighted:.6f}")
    print(f"HMM:        {hmm_weighted:.6f}")
    print(
        f"Difference: {hmm_weighted - classifier_weighted:+.6f}"
    )

    worse = result_df[
        result_df["hmm_accuracy_change"] < 0
    ]

    print("\n========== TRIPS WHERE HMM GOT WORSE ==========")

    if worse.empty:
        print("None")
    else:
        for _, row in worse.iterrows():
            print(
                f"{row['trip_id']}: "
                f"classifier={row['classifier_top1']:.4f}, "
                f"hmm={row['hmm_top1']:.4f}, "
                f"change={row['hmm_accuracy_change']:+.4f}"
            )

    print("\n========== BEST PARAMETERS ==========")
    print(f"TRANSITION_FLOOR = {best_floor}")
    print(f"GAP_THRESHOLD = {best_gap}")

    print("\n========== SAVED FILES ==========")
    print(comparison_path)
    print(search_path)
    print(plot_path)

    print(
        f"\nTotal execution time: "
        f"{perf_counter() - total_start:.2f}s"
    )


if __name__ == "__main__":
    main()
