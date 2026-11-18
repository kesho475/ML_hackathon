
"""Evaluate classifier-only versus HMM/Viterbi route reconstruction."""

import time
import joblib
import pandas as pd

from src.hmm import viterbi_decode
from src.io_utils import load_trip

FEATURES = [
    "dist_m", "dist_rank", "dist_minus_min",
    "heading_diff", "arc_len", "pos_on_arc",
    "speed", "dt", "n_cands"
]

DATASET = "results/features_train.parquet"
MODEL = "models/HistGradientBoosting.joblib"


def route_continuity(predictions, arcs):
    """Fraction of consecutive predicted arcs that are connected."""
    arc_from = arcs.from_node.to_dict()
    arc_to = arcs.to_node.to_dict()

    values = predictions.tolist()

    if len(values) < 2:
        return float("nan")

    connected = 0

    for a, b in zip(values[:-1], values[1:]):
        a, b = int(a), int(b)

        if a == b or (
            a in arc_to
            and b in arc_from
            and arc_to[a] == arc_from[b]
        ):
            connected += 1

    return connected / (len(values) - 1)


def main():
    df = pd.read_parquet(DATASET)
    model = joblib.load(MODEL)

    results = []
    start_time = time.time()

    for trip_id, candidates in df.groupby("trip_id", sort=True):
        candidates = candidates.copy().reset_index(drop=True)

        probs = model.predict_proba(candidates[FEATURES])[:, 1]

        # Classifier-only prediction
        candidates["_prob"] = probs

        baseline = (
            candidates.sort_values(
                ["point_idx", "_prob"],
                ascending=[True, False],
                kind="stable"
            )
            .drop_duplicates("point_idx")
            .set_index("point_idx")
            .cand_arc_id
        )

        # HMM prediction
        trip_dir = f"data/raw/train/{trip_id}"
        _, _, arcs, _ = load_trip(trip_dir)

        hmm = viterbi_decode(candidates, probs, arcs)

        # Ground truth: observations with a positive candidate label
        truth = (
            candidates[candidates.label == 1]
            .drop_duplicates("point_idx")
            .set_index("point_idx")
            .cand_arc_id
        )

        common = truth.index.intersection(baseline.index)

        if len(common) == 0:
            continue

        baseline_accuracy = (
            baseline.loc[common].to_numpy()
            == truth.loc[common].to_numpy()
        ).mean()

        hmm_accuracy = (
            hmm.loc[common].to_numpy()
            == truth.loc[common].to_numpy()
        ).mean()

        results.append({
            "trip_id": trip_id,
            "evaluated_points": len(common),
            "classifier_accuracy": baseline_accuracy,
            "hmm_accuracy": hmm_accuracy,
            "classifier_continuity": route_continuity(
                baseline.sort_index(), arcs
            ),
            "hmm_continuity": route_continuity(
                hmm.sort_index(), arcs
            ),
        })

        print(
            f"{trip_id}: "
            f"classifier={baseline_accuracy:.4f}, "
            f"hmm={hmm_accuracy:.4f}"
        )

    result_df = pd.DataFrame(results)

    if result_df.empty:
        raise RuntimeError("No trips could be evaluated.")

    result_df.to_csv(
        "results/csv/hmm_evaluation.csv",
        index=False
    )

    summary = result_df[
        [
            "classifier_accuracy",
            "hmm_accuracy",
            "classifier_continuity",
            "hmm_continuity",
        ]
    ].mean()

    print("\n========== STEP 7 EVALUATION ==========")
    print(summary)
    print("\nTrips evaluated:", len(result_df))
    print("Runtime:", round(time.time() - start_time, 2), "seconds")
    print("Saved: results/csv/hmm_evaluation.csv")


if __name__ == "__main__":
    main()
