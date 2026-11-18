
"""Candidate-level, observation-level and route-level evaluation metrics."""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def candidate_metrics(y_true, probabilities, threshold=0.5):
    """Calculate binary candidate-classification metrics."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)

    predictions = (probabilities >= threshold).astype(np.int8)

    # Keep probabilities valid for log loss.
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)

    return {
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "log_loss": float(
            log_loss(y_true, probabilities, labels=[0, 1])
        ),
    }


def top1_table(df, probabilities):
    """Select the highest-probability candidate for each GPS point."""

    result = df.copy()
    result["probability"] = np.asarray(probabilities, dtype=float)

    if "point_idx" not in result.columns:
        raise ValueError("DataFrame must contain point_idx")

    if "cand_arc_id" not in result.columns:
        raise ValueError("DataFrame must contain cand_arc_id")

    result = result.sort_values(
        ["point_idx", "probability"],
        ascending=[True, False],
        kind="stable",
    )

    return result.drop_duplicates(
        subset=["point_idx"],
        keep="first",
    ).reset_index(drop=True)


def top1_accuracy(top):
    """Calculate observation-level Top-1 accuracy."""

    if top.empty:
        return 0.0

    if "label" not in top.columns:
        raise ValueError("DataFrame must contain label")

    return float(top["label"].mean())


def route_continuity(top, arcs_of):
    """
    Calculate the fraction of consecutive predicted arcs that connect.

    arcs_of(trip_id) must return the road-arc DataFrame for that trip.
    Two consecutive predictions are continuous when the previous arc's
    destination node equals the current arc's source node.
    """

    if top.empty:
        return 0.0

    required = {"trip_id", "point_idx", "cand_arc_id"}
    missing = required - set(top.columns)

    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )

    total_pairs = 0
    connected_pairs = 0

    for trip_id, trip in top.groupby("trip_id"):
        trip = trip.sort_values("point_idx")

        arcs = arcs_of(trip_id)

        if arcs is None or arcs.empty:
            continue

        if not {"from_node", "to_node"}.issubset(arcs.columns):
            raise ValueError(
                "Arc DataFrame must contain from_node and to_node"
            )

        arc_from = arcs["from_node"].to_dict()
        arc_to = arcs["to_node"].to_dict()

        predictions = trip[
            ["point_idx", "cand_arc_id"]
        ].to_numpy()

        for i in range(1, len(predictions)):
            previous_point, previous_arc = predictions[i - 1]
            current_point, current_arc = predictions[i]

            # Only compare predictions for adjacent GPS points.
            if current_point != previous_point + 1:
                continue

            previous_arc = int(previous_arc)
            current_arc = int(current_arc)

            if (
                previous_arc not in arc_to
                or current_arc not in arc_from
            ):
                continue

            total_pairs += 1

            if arc_to[previous_arc] == arc_from[current_arc]:
                connected_pairs += 1

    if total_pairs == 0:
        return 0.0

    return float(connected_pairs / total_pairs)
