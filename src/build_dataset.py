import os
import glob
import time
import yaml
import numpy as np
import pandas as pd
import shapely

from shapely import STRtree
from pyproj import Transformer

from io_utils import load_trip, get_utm_crs


FEATURES = [
    "dist_m",
    "dist_rank",
    "dist_minus_min",
    "heading_diff",
    "arc_len",
    "pos_on_arc",
    "speed",
    "dt",
    "n_cands"
]

OUTPUT_COLUMNS = [
    "trip_id",
    "point_idx",
    "cand_arc_id",
    "label",
    *FEATURES
]

FLOAT_COLUMNS = [
    "dist_m",
    "dist_minus_min",
    "heading_diff",
    "arc_len",
    "pos_on_arc",
    "speed",
    "dt"
]

INT32_COLUMNS = [
    "point_idx",
    "cand_arc_id",
    "n_cands"
]


def compact_dtypes(frame):
    for column in FLOAT_COLUMNS:
        frame[column] = frame[column].astype(np.float32)

    if "label" in frame:
        frame["label"] = frame["label"].astype(np.int8)
    frame["dist_rank"] = frame["dist_rank"].astype(np.int8)

    for column in INT32_COLUMNS:
        frame[column] = frame[column].astype(np.int32)

    return frame


def fallback_candidates(tree, points, point_id, radius, k, line_count):
    point = points[point_id]

    nearest_idx, nearest_dist = tree.query_nearest(
        point,
        all_matches=True,
        return_distance=True
    )

    nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
    nearest_dist = np.asarray(nearest_dist, dtype=float).reshape(-1)

    if len(nearest_idx) >= k:
        order = np.argsort(nearest_dist, kind="stable")[:k]
        return nearest_idx[order], nearest_dist[order]

    search_radius = max(
        float(radius) * 2.0,
        float(nearest_dist.min()) + 1e-6
    )

    while True:
        pairs = tree.query(
            point,
            predicate="dwithin",
            distance=search_radius
        )

        candidate_idx = np.asarray(pairs, dtype=np.int64).reshape(-1)

        if len(candidate_idx) >= min(k, line_count):
            break

        if search_radius > 1e8:
            candidate_idx = np.arange(line_count, dtype=np.int64)
            break

        search_radius *= 2.0

    candidate_idx = np.unique(
        np.concatenate((candidate_idx, nearest_idx))
    )

    candidate_dist = shapely.distance(
        point,
        tree.geometries.take(candidate_idx)
    )

    order = np.argsort(candidate_dist, kind="stable")[:k]

    return candidate_idx[order], candidate_dist[order]


def trip_geometry(trip_dir):
    """Load one trip and project it to its own UTM zone; road arcs become LineStrings in metres.
    The .route list is loaded (io_utils reads it if present) but only reference_arcs() ever uses it."""
    trip_id = os.path.basename(os.path.normpath(trip_dir))

    track, nodes, arcs, route = load_trip(trip_dir)

    if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
        raise ValueError(f"Empty data in trip {trip_id}")

    lon = track["lon"].to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)
    timestamps = track["t"].to_numpy(dtype=float)

    if not (
        np.isfinite(lon).all()
        and np.isfinite(lat).all()
        and np.isfinite(timestamps).all()
    ):
        raise ValueError(f"Invalid GPS or timestamp values in {trip_id}")

    utm = get_utm_crs(
        float(np.median(lon)),
        float(np.median(lat))
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        utm,
        always_xy=True
    )

    node_x, node_y = transformer.transform(
        nodes["lon"].to_numpy(dtype=float),
        nodes["lat"].to_numpy(dtype=float)
    )

    node_xy = np.column_stack((node_x, node_y))
    arc_nodes = arcs.to_numpy(dtype=np.int64)

    valid_arcs = (
        (arc_nodes[:, 0] >= 0)
        & (arc_nodes[:, 1] >= 0)
        & (arc_nodes[:, 0] < len(nodes))
        & (arc_nodes[:, 1] < len(nodes))
    )

    arc_ids = np.flatnonzero(valid_arcs)
    arc_nodes = arc_nodes[valid_arcs]
    coords = node_xy[arc_nodes]

    lines = shapely.linestrings(coords)
    arc_lengths = shapely.length(lines)

    valid_lines = (
        np.isfinite(arc_lengths)
        & (arc_lengths > 0)
    )

    arc_ids = arc_ids[valid_lines]
    lines = lines[valid_lines]
    arc_lengths = arc_lengths[valid_lines]
    coords = coords[valid_lines]

    if len(lines) == 0:
        raise ValueError(f"No valid road segments in {trip_id}")

    px, py = transformer.transform(lon, lat)
    points = shapely.points(np.column_stack((px, py)))

    return {
        "trip_id": trip_id, "track": track, "nodes": nodes, "arcs": arcs, "route": route,
        "timestamps": timestamps, "arc_ids": arc_ids, "coords": coords, "lines": lines,
        "arc_lengths": arc_lengths, "px": px, "py": py, "points": points,
    }


def build_features(trip_dir, radius, k):
    """Candidates + the 9 features for EVERY GPS point of a trip. Never touches the route / labels.
    Returns (frame without the label column, geometry dict from trip_geometry)."""
    geo = trip_geometry(trip_dir)
    trip_id, arc_ids, coords = geo["trip_id"], geo["arc_ids"], geo["coords"]
    lines, arc_lengths, px, py = geo["lines"], geo["arc_lengths"], geo["px"], geo["py"]
    points, timestamps = geo["points"], geo["timestamps"]

    tree = STRtree(lines)

    pairs = tree.query(
        points,
        predicate="dwithin",
        distance=radius
    )

    point_idx = pairs[0]
    line_idx = pairs[1]

    distances = shapely.distance(
        points[point_idx],
        lines[line_idx]
    )

    missing = np.setdiff1d(
        np.arange(len(points)),
        np.unique(point_idx)
    )

    if len(missing):
        fallback_point_parts = []
        fallback_line_parts = []
        fallback_distance_parts = []

        for point_id in missing:
            nearest_lines, nearest_distances = fallback_candidates(
                tree,
                points,
                int(point_id),
                radius,
                k,
                len(lines)
            )

            fallback_point_parts.append(
                np.full(len(nearest_lines), point_id, dtype=np.int64)
            )
            fallback_line_parts.append(nearest_lines)
            fallback_distance_parts.append(nearest_distances)

        point_idx = np.concatenate(
            (point_idx, *fallback_point_parts)
        )
        line_idx = np.concatenate(
            (line_idx, *fallback_line_parts)
        )
        distances = np.concatenate(
            (distances, *fallback_distance_parts)
        )

    order = np.lexsort((distances, point_idx))

    point_idx = point_idx[order]
    line_idx = line_idx[order]
    distances = distances[order]

    unique_points, starts, counts = np.unique(
        point_idx,
        return_index=True,
        return_counts=True
    )

    min_dist = np.full(len(points), np.nan, dtype=float)
    min_dist[unique_points] = distances[starts]

    ranks = (
        np.arange(len(point_idx))
        - np.repeat(starts, counts)
    )

    keep = ranks < k

    point_idx = point_idx[keep]
    line_idx = line_idx[keep]
    distances = distances[keep]
    ranks = ranks[keep]

    n_cands = np.bincount(
        point_idx,
        minlength=len(points)
    )

    candidate_arc_ids = arc_ids[line_idx]

    delta_t = np.diff(
        timestamps,
        prepend=timestamps[0]
    )
    dt = np.maximum(delta_t, 0)

    movement = np.hypot(
        np.diff(px, prepend=px[0]),
        np.diff(py, prepend=py[0])
    )

    speed = np.zeros(len(points), dtype=float)
    valid_speed = delta_t > 0
    speed[valid_speed] = (
        movement[valid_speed] / delta_t[valid_speed]
    )

    gps_heading = np.degrees(
        np.arctan2(
            np.diff(py, prepend=py[0]),
            np.diff(px, prepend=px[0])
        )
    )

    arc_dx = (
        coords[line_idx, 1, 0]
        - coords[line_idx, 0, 0]
    )
    arc_dy = (
        coords[line_idx, 1, 1]
        - coords[line_idx, 0, 1]
    )

    arc_heading = np.degrees(
        np.arctan2(arc_dy, arc_dx)
    )

    heading_diff = np.abs(
        (gps_heading[point_idx] - arc_heading + 180) % 360 - 180
    )

    neutral_heading = (
        (point_idx == 0)
        | (movement[point_idx] < 1.0)
        | (arc_lengths[line_idx] <= 0)
    )

    heading_diff[neutral_heading] = 90.0

    candidate_lines = lines[line_idx]
    candidate_points = points[point_idx]

    positions = shapely.line_locate_point(
        candidate_lines,
        candidate_points
    ) / arc_lengths[line_idx]

    positions = np.clip(positions, 0, 1)

    result = pd.DataFrame({
        "trip_id": trip_id,
        "point_idx": point_idx,
        "cand_arc_id": candidate_arc_ids,
        "dist_m": distances,
        "dist_rank": ranks,
        "dist_minus_min": distances - min_dist[point_idx],
        "heading_diff": heading_diff,
        "arc_len": arc_lengths[line_idx],
        "pos_on_arc": positions,
        "speed": speed[point_idx],
        "dt": dt[point_idx],
        "n_cands": n_cands[point_idx]
    })

    return compact_dtypes(result), geo


def reference_arcs(geo):
    """Reference arc per GPS point = nearest arc among the trip's ROUTE arcs (our labelling assumption).
    Uses the .route list: training labels and server-side evaluation only, never prediction."""
    arc_ids, lines, points = geo["arc_ids"], geo["lines"], geo["points"]

    route_ids = np.asarray(geo["route"], dtype=np.int64)
    route_ids = np.unique(
        route_ids[
            (route_ids >= 0)
            & (route_ids < len(geo["arcs"]))
        ]
    )

    route_mask = np.isin(arc_ids, route_ids)
    route_lines = lines[route_mask]
    route_arc_ids = arc_ids[route_mask]

    if len(route_lines) == 0:
        raise ValueError(
            f"No valid reference-route geometries in {geo['trip_id']}"
        )

    route_tree = STRtree(route_lines)
    nearest_pairs = route_tree.query_nearest(
        points,
        all_matches=False
    )

    reference_arc = np.full(
        len(points),
        -1,
        dtype=np.int64
    )

    reference_arc[nearest_pairs[0]] = (
        route_arc_ids[nearest_pairs[1]]
    )

    return reference_arc


def build_trip(trip_dir, radius, k):
    result, geo = build_features(trip_dir, radius, k)
    reference_arc = reference_arcs(geo)

    point_idx = result["point_idx"].to_numpy()
    labels = (
        result["cand_arc_id"].to_numpy() == reference_arc[point_idx]
    ).astype(np.int8)

    n_points = len(geo["points"])
    candidate_hit = np.zeros(n_points, dtype=bool)
    valid_points = reference_arc[point_idx] >= 0

    np.logical_or.at(
        candidate_hit,
        point_idx[valid_points],
        labels[valid_points].astype(bool)
    )

    missed = int(np.count_nonzero(~candidate_hit))

    result.insert(3, "label", labels)
    result = compact_dtypes(result)[OUTPUT_COLUMNS]

    return result, missed, n_points


def build_partition(directory, radius, k):
    trip_dirs = sorted(
        path
        for path in glob.glob(os.path.join(directory, "*"))
        if os.path.isdir(path)
    )

    if not trip_dirs:
        raise ValueError(f"No trip folders found in {directory}")

    frames = []
    trip_stats = []

    for trip_dir in trip_dirs:
        frame, missed, point_count = build_trip(
            trip_dir,
            radius,
            k
        )

        frames.append(frame)
        trip_id = os.path.basename(trip_dir)

        trip_stats.append({
            "trip_id": trip_id,
            "total_points": point_count,
            "missed_points": missed
        })

        print(
            trip_id,
            "rows:", len(frame),
            "missed:", missed
        )

    combined = pd.concat(frames, ignore_index=True)
    stats = pd.DataFrame(trip_stats)

    return combined, stats


def summarize_partition(name, frame, stats):
    total_points = int(stats["total_points"].sum())
    missed_points = int(stats["missed_points"].sum())

    return {
        "partition": name,
        "total_points": total_points,
        "missed_points": missed_points,
        "candidate_miss_rate": (
            missed_points / total_points if total_points else 0.0
        ),
        "rows": len(frame),
        "positive_ratio": float(frame["label"].mean()) if len(frame) else 0.0,
        "nan_count": int(frame.isna().sum().sum())
    }


def main():
    start_time = time.perf_counter()

    with open("configs/config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    radius = float(config["radius_m"])
    k = int(config["K"])
    excluded_train_trips = {
        str(trip_id)
        for trip_id in config.get("exclude_train_trips", [])
    }

    os.makedirs("results/csv", exist_ok=True)

    train_all, train_stats_all = build_partition(
        config["train_dir"],
        radius,
        k
    )

    train_before_summary = summarize_partition(
        "train_before_drop",
        train_all,
        train_stats_all
    )

    keep_mask = ~train_all["trip_id"].isin(excluded_train_trips)
    train = train_all.loc[keep_mask].copy()

    kept_stats = train_stats_all[
        ~train_stats_all["trip_id"].isin(excluded_train_trips)
    ].copy()

    missing_exclusions = excluded_train_trips - set(train_stats_all["trip_id"])
    if missing_exclusions:
        raise ValueError(
            "Configured train trips were not found: "
            + ", ".join(sorted(missing_exclusions))
        )

    train_after_summary = summarize_partition(
        "train_after_drop",
        train,
        kept_stats
    )

    student_test, test_stats = build_partition(
        config["test_dir"],
        radius,
        k
    )

    test_summary = summarize_partition(
        "student_test",
        student_test,
        test_stats
    )

    train.to_parquet(
        "results/features_train.parquet",
        index=False
    )

    student_test.to_parquet(
        "results/features_student_test.parquet",
        index=False
    )

    miss_report = pd.DataFrame([
        train_before_summary,
        train_after_summary,
        test_summary
    ])

    miss_report.to_csv(
        "results/csv/candidate_miss_rate.csv",
        index=False
    )

    elapsed = time.perf_counter() - start_time

    print("\nFINAL SUMMARY")
    for summary in [
        train_before_summary,
        train_after_summary,
        test_summary
    ]:
        print(f"\n{summary['partition']}")
        print("Rows:", summary["rows"])
        print("Positive ratio:", summary["positive_ratio"])
        print("Candidate miss rate:", summary["candidate_miss_rate"])
        print("NaN count:", summary["nan_count"])
        print("Unique trips:", (
            train_all["trip_id"].nunique()
            if summary["partition"] == "train_before_drop"
            else train["trip_id"].nunique()
            if summary["partition"] == "train_after_drop"
            else student_test["trip_id"].nunique()
        ))

    print("\nRUNTIME_SECONDS:", round(elapsed, 2))

    for path in [
        "results/features_train.parquet",
        "results/features_student_test.parquet",
        "results/csv/candidate_miss_rate.csv"
    ]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{path}: {size_mb:.2f} MB")

    print("\nCandidate miss report:")
    print(miss_report.to_string(index=False))


if __name__ == "__main__":
    main()
