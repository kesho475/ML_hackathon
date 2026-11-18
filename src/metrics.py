"""Candidate-level, observation-level and route-level metrics for candidate-arc classifiers."""
import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y, p, thr=0.5):
    """Candidate-level macro F1 / precision / recall at threshold thr, plus log loss."""
    yh = (p >= thr).astype(int)
    return {"macro_f1": f1_score(y, yh, average="macro", zero_division=0),
            "precision": precision_score(y, yh, average="macro", zero_division=0),
            "recall": recall_score(y, yh, average="macro", zero_division=0),
            "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])}


def top1_table(df, p):
    """Highest-probability candidate per (trip_id, point_idx). Columns: trip_id, point_idx, cand_arc_id, label."""
    d = df[["trip_id", "point_idx", "cand_arc_id", "label"]].copy()
    d["p"] = p
    d = d.sort_values(["trip_id", "point_idx", "p"], ascending=[True, True, False], kind="stable")
    return d.drop_duplicates(["trip_id", "point_idx"])


def top1_accuracy(top):
    # a point whose candidates contain no label=1 row has top.label == 0, so it counts as wrong
    return float(top.label.mean())


def route_continuity(top, arcs_of):
    """Per trip, in point order: share of consecutive Top-1 arcs that are the same arc or connected
    (to_node of previous == from_node of next). Returns the mean over trips.
    arcs_of(trip_id) -> DataFrame(from_node, to_node) indexed by arc_id."""
    out = []
    for tid, g in top.groupby("trip_id"):
        if len(g) < 2:
            continue
        arcs = arcs_of(tid)
        a = g.sort_values("point_idx").cand_arc_id.to_numpy()
        f, t = arcs.from_node.to_numpy(), arcs.to_node.to_numpy()
        ok = (a[:-1] == a[1:]) | (t[a[:-1]] == f[a[1:]])
        out.append(ok.mean())
    return float(np.mean(out))
