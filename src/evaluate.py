
"""Server-side evaluation.

Reference labels are accessed ONLY inside evaluate(), after prediction.
"""

import os
import sys

# Allow existing flat imports such as io_utils and metrics
# used by build_dataset.py.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd

from build_dataset import reference_arcs, trip_geometry
from metrics import route_continuity, top1_accuracy


def evaluate(trip_dir, predicted_arcs):
    """Evaluate predicted road arcs using official project metrics."""

    geo = trip_geometry(trip_dir)
    ref = np.asarray(reference_arcs(geo))
    pred = np.asarray(predicted_arcs)

    if pred.ndim != 1:
        raise ValueError(
            "predicted_arcs must be a one-dimensional sequence"
        )

    if len(pred) != len(ref):
        raise ValueError(
            f"Prediction count ({len(pred)}) does not match "
            f"GPS point count ({len(ref)})"
        )

    # One row per GPS observation.
    # Missing predictions (-1) count as incorrect.
    top = pd.DataFrame({
        "trip_id": geo["trip_id"],
        "point_idx": np.arange(len(pred)),
        "cand_arc_id": pred,
        "label": (pred == ref).astype(np.int8),
    })

    accuracy = top1_accuracy(top)

    valid_predictions = top[top["cand_arc_id"] >= 0]

    continuity = route_continuity(
        valid_predictions,
        lambda _tid: geo["arcs"],
    )

    return {
        "top1_accuracy": accuracy,
        "route_continuity": continuity,
    }
