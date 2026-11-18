"""Server-side evaluation. Reference labels exist ONLY inside evaluate(), after prediction, and never leave it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from build_dataset import reference_arcs, trip_geometry
from metrics import route_continuity, top1_accuracy


def evaluate(trip_dir, predicted_arcs):
    geo = trip_geometry(trip_dir)
    ref = reference_arcs(geo)  # nearest route arc per GPS point (same rule as build_dataset.py)
    pred = np.asarray(predicted_arcs)
    top = pd.DataFrame({"trip_id": geo["trip_id"], "point_idx": np.arange(len(pred)), "cand_arc_id": pred,
                        "label": (pred == ref).astype(np.int8)})
    return {"top1_accuracy": top1_accuracy(top),
            "route_continuity": route_continuity(top[top.cand_arc_id >= 0], lambda _tid: geo["arcs"])}
