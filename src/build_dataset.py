"""Build candidate-arc feature tables: one row per (GPS point, candidate arc). Run from project root."""
import os, sys, time
import numpy as np, pandas as pd, shapely, yaml
from pyproj import Transformer
from shapely import STRtree
sys.path.insert(0, os.path.dirname(__file__))
from io_utils import load_trip, get_utm_crs

cfg = yaml.safe_load(open("configs/config.yaml"))
K, RADIUS = cfg["K"], cfg["radius_m"]


def bearing(dx, dy):
    return np.degrees(np.arctan2(dx, dy)) % 360  # compass bearing, 0 = north


def trip_features(trip_dir):
    track, nodes, arcs, route = load_trip(trip_dir)
    tid = os.path.basename(os.path.normpath(trip_dir))
    # per-trip UTM projection -> metres
    tf = Transformer.from_crs("EPSG:4326", get_utm_crs(track.lon.mean(), track.lat.mean()), always_xy=True)
    px, py = tf.transform(track.lon.to_numpy(), track.lat.to_numpy())
    nx_, ny_ = tf.transform(nodes.lon.to_numpy(), nodes.lat.to_numpy())
    node_xy = np.column_stack([nx_, ny_])
    a0, a1 = node_xy[arcs.from_node.to_numpy()], node_xy[arcs.to_node.to_numpy()]
    lines = shapely.linestrings(np.stack([a0, a1], axis=1))  # (n_arcs,) LineStrings in metres
    arc_len = shapely.length(lines)
    arc_brg = bearing(a1[:, 0] - a0[:, 0], a1[:, 1] - a0[:, 1])
    pts = shapely.points(px, py)
    n = len(pts)

    # LABEL (our assumption): reference arc = nearest arc among this trip's route arcs.
    # .route is only an ordered arc list with no per-point alignment, so this can be wrong
    # at junctions and on parallel roads.
    route_ids = np.unique(np.asarray(route, dtype=np.int64))
    ref = route_ids[STRtree(lines[route_ids]).query_nearest(pts, all_matches=False)[1]]

    # CANDIDATES: K nearest arcs within RADIUS metres, one STRtree for the whole network
    pi, ai = STRtree(lines).query(pts, predicate="dwithin", distance=RADIUS)
    d = shapely.distance(pts[pi], lines[ai])
    o = np.lexsort((d, pi))  # by point, then distance
    pi, ai, d = pi[o], ai[o], d[o]
    start = np.searchsorted(pi, np.arange(n))  # first row of each point
    rank = np.arange(len(pi)) - start[pi]
    keep = rank < K
    pi, ai, d, rank = pi[keep], ai[keep], d[keep], rank[keep]
    n_c = np.bincount(pi, minlength=n)
    first_d = np.full(n, np.nan); first_d[pi[rank == 0]] = d[rank == 0]

    hit = np.zeros(n, bool); hit[pi[ai == ref[pi]]] = True  # reference arc in candidate set?

    # point motion: speed, dt, GPS bearing prev->current (first point: current->next)
    t = track.t.to_numpy()
    dt = np.r_[0.0, np.diff(t)]
    step = np.r_[0.0, np.hypot(np.diff(px), np.diff(py))]
    speed = np.divide(step, dt, out=np.zeros(n), where=dt > 0)  # first point / dt==0 -> 0
    dx, dy = np.r_[np.diff(px), 0.0], np.r_[np.diff(py), 0.0]   # step i -> i+1
    dx_p, dy_p = np.r_[dx[0], dx[:-1]], np.r_[dy[0], dy[:-1]]   # step i-1 -> i (first: reuse first step)
    moved = np.hypot(dx_p, dy_p) >= 1.0
    gps_brg = bearing(dx_p, dy_p)

    diff = np.abs(gps_brg[pi] - arc_brg[ai]) % 360
    diff = np.minimum(diff, 360 - diff)
    # heading undefined (moved < 1 m or zero-length arc) -> 90, the uninformative midpoint
    diff = np.where(moved[pi] & (arc_len[ai] > 0), diff, 90.0)
    pos = shapely.line_locate_point(lines[ai], pts[pi])
    pos = np.divide(pos, arc_len[ai], out=np.zeros(len(pi)), where=arc_len[ai] > 0)

    df = pd.DataFrame({
        "trip_id": tid, "point_idx": pi.astype(np.int32), "cand_arc_id": ai.astype(np.int32),
        "label": (ai == ref[pi]).astype(np.int8),
        "dist_m": d, "dist_rank": rank.astype(np.int8), "dist_minus_min": d - first_d[pi],
        "heading_diff": diff, "arc_len": arc_len[ai], "pos_on_arc": np.clip(pos, 0, 1),
        "speed": speed[pi], "dt": dt[pi], "n_cands": n_c[pi].astype(np.int8)})
    f32 = ["dist_m", "dist_minus_min", "heading_diff", "arc_len", "pos_on_arc", "speed", "dt"]
    df[f32] = df[f32].astype("float32")
    return df, n, int(hit.sum()), int((n_c == 0).sum())


def main():
    t0 = time.time(); miss = []
    os.makedirs("results/csv", exist_ok=True)
    for part, key in [("train", "train_dir"), ("student_test", "test_dir")]:
        dfs, n_pts, n_hit, n_empty = [], 0, 0, 0
        for tid in sorted(os.listdir(cfg[key])):
            df, n, h, e = trip_features(os.path.join(cfg[key], tid))
            dfs.append(df); n_pts += n; n_hit += h; n_empty += e
        out = pd.concat(dfs, ignore_index=True)
        out.to_parquet(f"results/features_{part}.parquet", index=False)
        miss.append({"partition": part, "n_trips": len(dfs), "n_points": n_pts, "n_rows": len(out), "pos_ratio": out.label.mean(),
                     "points_ref_in_candidates": n_hit, "points_no_candidates": n_empty,
                     "candidate_miss_rate": 1 - n_hit / n_pts, "K": K, "radius_m": RADIUS})
        print(f"{part}: {len(out)} rows, {n_pts} points, pos ratio {out.label.mean():.4f}, miss rate {1 - n_hit / n_pts:.4f}  [{time.time() - t0:.0f}s]")
    pd.DataFrame(miss).to_csv("results/csv/candidate_miss_rate.csv", index=False)
    print(pd.DataFrame(miss).T.to_string()); print(f"total run time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
