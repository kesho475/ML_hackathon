import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, log_loss,
    confusion_matrix
)


def candidate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "macro_f1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        )
    }


def top1_accuracy(data, probabilities):
    temp = data[["trip_id", "point_idx", "label"]].copy()
    temp["probability"] = probabilities

    # Select the highest-probability candidate for each GPS observation.
    selected = temp.loc[
        temp.groupby(["trip_id", "point_idx"])["probability"].idxmax()
    ]

    # A point is correct only if its selected candidate has label 1.
    return float(selected["label"].eq(1).mean())


def route_continuity(data, probabilities):
    # Exact connectivity requires arcs.from_node and arcs.to_node.
    # These are not present in the feature Parquet files.
    return np.nan