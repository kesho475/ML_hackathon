"""Prediction pipeline for the API: raw trip folder -> predicted arc per GPS point.
Features come from build_dataset.build_features, which never touches the .route file or labels."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # flat imports work under uvicorn src.api.main:app

import joblib
import numpy as np
import pandas as pd
import yaml

from build_dataset import FEATURES, build_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml")))
MODEL = joblib.load(os.path.join(ROOT, "models", "final_model.joblib"))  # loaded once at import


def _classifier_argmax(cands, probs, arcs):
    """Default decoder: highest-probability candidate per GPS point."""
    d = pd.DataFrame({"point_idx": cands["point_idx"].to_numpy(), "arc": cands["cand_arc_id"].to_numpy(), "p": probs})
    d = d.sort_values(["point_idx", "p"], ascending=[True, False], kind="stable").drop_duplicates("point_idx")
    return d.set_index("point_idx")["arc"]


# HOOK for B's HMM: register DECODERS["hmm"] = fn(cands_df, probs, arcs) -> pd.Series (index point_idx, value arc_id),
# then set `decoder: hmm` in configs/config.yaml. Nothing else in the API changes.
from hmm import viterbi_decode

DECODERS = {
    "classifier": _classifier_argmax,
    "hmm": viterbi_decode,
}


def decode(cands, probs, arcs, decoder=None):
    """Returns (decoder_name, Series of predicted arc_id indexed by point_idx)."""
    name = decoder or CFG.get("decoder", "classifier")
    if name not in DECODERS:
        raise ValueError(f"decoder '{name}' is not registered (have: {list(DECODERS)})")
    return name, DECODERS[name](cands, probs, arcs)


def predict_trip(trip_dir):
    cands, geo = build_features(trip_dir, CFG["radius_m"], CFG["K"])
    probs = MODEL.predict_proba(cands[FEATURES].to_numpy(dtype=np.float32))[:, 1]
    arcs, nodes, track = geo["arcs"], geo["nodes"], geo["track"]
    name, pred = decode(cands, probs, arcs)
    n = len(track)
    predicted = pred.reindex(range(n)).fillna(-1).astype(int).to_numpy()  # -1 = no prediction (no candidates)
    keep = predicted >= 0
    seq = predicted[keep]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]] if len(seq) else seq  # drop consecutive duplicates
    ll = nodes[["lat", "lon"]].to_numpy()  # response coordinates are [lat, lon]
    f, t = arcs["from_node"].to_numpy()[seq], arcs["to_node"].to_numpy()[seq]
    return {
        "trip_id": geo["trip_id"], "decoder": name, "n_points": n,
        "gps_points": track[["lat", "lon"]].to_numpy().tolist(),
        "predicted_arcs": predicted.tolist(),
        "predicted_route": np.stack([ll[f], ll[t]], axis=1).tolist() if len(seq) else [],
    }
