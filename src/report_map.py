"""Report map (sections 5 and 6, student_test only): road network near the route, reference route, classifier-only
and HMM predicted routes, GPS points.  Usage (project root): python src/report_map.py --trip_id 00000076
Offline report tool: it reads the reference route on purpose; the API never does."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import folium
import numpy as np
from scipy.spatial import cKDTree

import pipeline
from build_dataset import trip_geometry
from evaluate import evaluate  # same reference rule as the API; used here for the legend metrics

NEAR_DEG = 0.0015  # ~150 m: network arcs this close to the reference route are drawn


def predicted_segments(cands, probs, geo, decoder):
    """Predicted arc per point from pipeline.decode -> (per-point arc array, deduped [lat,lon] segments)."""
    _, pred = pipeline.decode(cands, probs, geo["arcs"], decoder=decoder)
    arcs_pp = pred.reindex(range(len(geo["track"]))).fillna(-1).astype(int).to_numpy()
    seq = arcs_pp[arcs_pp >= 0]
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    return arcs_pp, arc_segments(geo, seq)


def arc_segments(geo, arc_ids):
    ll = geo["nodes"][["lat", "lon"]].to_numpy()
    a = geo["arcs"]
    f, t = a["from_node"].to_numpy()[arc_ids], a["to_node"].to_numpy()[arc_ids]
    return np.stack([ll[f], ll[t]], axis=1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip_id", required=True)
    tid = ap.parse_args().trip_id
    trip_dir = os.path.join(pipeline.ROOT, "data", "raw", "student_test", tid)  # student_test only
    if not os.path.isdir(trip_dir):
        sys.exit(f"trip {tid} not found in data/raw/student_test")

    cands, geo = pipeline.build_features(trip_dir, pipeline.CFG["radius_m"], pipeline.CFG["K"])
    probs = pipeline.MODEL.predict_proba(cands[pipeline.FEATURES].to_numpy(dtype=np.float32))[:, 1]
    clf_arcs, clf_seg = predicted_segments(cands, probs, geo, "classifier")
    hmm_arcs, hmm_seg = predicted_segments(cands, probs, geo, "hmm")
    m_clf, m_hmm = evaluate(trip_dir, clf_arcs), evaluate(trip_dir, hmm_arcs)

    route_ids = np.unique(np.asarray(geo["route"], dtype=np.int64))  # reference route arcs (report only)
    route_seg = arc_segments(geo, np.asarray(geo["route"], dtype=np.int64))
    nodes_ll = geo["nodes"][["lat", "lon"]].to_numpy()
    arcs = geo["arcs"]
    route_nodes = np.unique(np.r_[arcs["from_node"].to_numpy()[route_ids], arcs["to_node"].to_numpy()[route_ids]])
    d, _ = cKDTree(nodes_ll[route_nodes]).query(nodes_ll[arcs["from_node"].to_numpy()])
    net_seg = arc_segments(geo, np.flatnonzero(d < NEAR_DEG))  # full network is too big for a map

    gps = geo["track"][["lat", "lon"]].to_numpy()
    fmap = folium.Map(tiles="OpenStreetMap")
    for name, seg, color, w in [("Road network near route (grey)", net_seg, "#888888", 1),
                                ("Reference route (green)", route_seg, "#1a9a1a", 7),
                                ("Classifier-only route (orange)", clf_seg, "#ff8c00", 4),
                                ("HMM route (red)", hmm_seg, "#e00000", 3)]:
        g = folium.FeatureGroup(name)
        folium.PolyLine(seg, color=color, weight=w, opacity=0.85 if w > 1 else 0.6).add_to(g)
        g.add_to(fmap)
    g = folium.FeatureGroup("GPS points (blue)")
    for lat, lon in gps:
        folium.CircleMarker((lat, lon), radius=2, color="blue", weight=1, fill=True, fill_opacity=0.7).add_to(g)
    g.add_to(fmap)
    fmap.fit_bounds([gps.min(0).tolist(), gps.max(0).tolist()])

    def row(color, text):
        return f'<div><span style="display:inline-block;width:22px;height:5px;background:{color};margin-right:6px"></span>{text}</div>'
    legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;color:#222;padding:8px 10px;'
              'border:1px solid #444;font:13px sans-serif">'
              f'<b>Trip {tid} (student_test)</b>'
              + row("#888888", "Road network near route")
              + row("#1a9a1a", "Reference route")
              + row("#ff8c00", f"Classifier only: Top-1 {m_clf['top1_accuracy']:.1%}, continuity {m_clf['route_continuity']:.1%}")
              + row("#e00000", f"HMM + Viterbi: Top-1 {m_hmm['top1_accuracy']:.1%}, continuity {m_hmm['route_continuity']:.1%}")
              + '<div><span style="color:blue">&#9679;</span> GPS points</div></div>')
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    out = os.path.join(pipeline.ROOT, "results", "maps", f"report_{tid}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fmap.save(out)
    print(f"{tid}: classifier Top-1 {m_clf['top1_accuracy']:.4f} cont {m_clf['route_continuity']:.4f} | "
          f"HMM Top-1 {m_hmm['top1_accuracy']:.4f} cont {m_hmm['route_continuity']:.4f} | "
          f"network arcs drawn {len(net_seg)} | saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
