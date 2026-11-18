
import os
import glob
import yaml
import numpy as np
import pandas as pd
import shapely

from shapely import STRtree
from scipy.spatial import cKDTree
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


def build_trip(trip_dir, radius, k):
    trip_id = os.path.basename(os.path.normpath(trip_dir))

    track, nodes, arcs, route = load_trip(trip_dir)

    if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
        raise ValueError(f"Empty data in trip {trip_id}")

    if len(route) == 0:
        raise ValueError(f"Missing reference route in trip {trip_id}")

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

    points = shapely.points(
        np.column_stack((px, py))
    )

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
        midpoints = coords.mean(axis=1)
        midpoint_tree = cKDTree(midpoints)

        _, nearest = midpoint_tree.query(
            np.column_stack((px[missing], py[missing])),
            k=min(k, len(lines))
        )

        nearest = np.asarray(nearest)

        if nearest.ndim == 1:
            nearest = nearest[:, None]

        fallback_points = np.repeat(
            missing,
            nearest.shape[1]
        )

        fallback_lines = nearest.reshape(-1)

        fallback_distances = shapely.distance(
            points[fallback_points],
            lines[fallback_lines]
        )

        point_idx = np.concatenate(
            (point_idx, fallback_points)
        )

        line_idx = np.concatenate(
            (line_idx, fallback_lines)
        )

        distances = np.concatenate(
            (distances, fallback_distances)
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

    min_dist = np.full(
        len(points),
        np.nan,
        dtype=float
    )

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

    route_ids = np.asarray(
        route,
        dtype=np.int64
    )

    route_ids = np.unique(
        route_ids[
            (route_ids >= 0)
            & (route_ids < len(arcs))
        ]
    )

    route_mask = np.isin(
        arc_ids,
        route_ids
    )

    route_lines = lines[route_mask]
    route_arc_ids = arc_ids[route_mask]

    if len(route_lines) == 0:
        raise ValueError(
            f"No valid reference-route geometries in {trip_id}"
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

    labels = (
        candidate_arc_ids
        == reference_arc[point_idx]
    ).astype(np.int8)

    candidate_hit = np.zeros(
        len(points),
        dtype=bool
    )

    valid_points = reference_arc[point_idx] >= 0

    np.logical_or.at(
        candidate_hit,
        point_idx[valid_points],
        candidate_arc_ids[valid_points]
        == reference_arc[point_idx[valid_points]]
    )

    missed = np.count_nonzero(~candidate_hit)

    delta_t = np.diff(
        timestamps,
        prepend=timestamps[0]
    )

    dt = np.maximum(delta_t, 0)

    movement = np.hypot(
        np.diff(px, prepend=px[0]),
        np.diff(py, prepend=py[0])
    )

    speed = np.zeros(
        len(points),
        dtype=float
    )

    valid_speed = delta_t > 0

    speed[valid_speed] = (
        movement[valid_speed]
        / delta_t[valid_speed]
    )

    gps_heading = np.degrees(
        np.arctan2(
            np.diff(py, prepend=py[0]),
            np.diff(px, prepend=px[0])
        )
    )

    arc_heading = np.degrees(
        np.arctan2(
            coords[line_idx, 1, 1]
            - coords[line_idx, 0, 1],
            coords[line_idx, 1, 0]
            - coords[line_idx, 0, 0]
        )
    )

    heading_diff = np.abs(
        (
            gps_heading[point_idx]
            - arc_heading
            + 180
        ) % 360 - 180
    )

    candidate_lines = lines[line_idx]
    candidate_points = points[point_idx]

    positions = shapely.line_locate_point(
        candidate_lines,
        candidate_points
    ) / arc_lengths[line_idx]

    positions = np.clip(
        positions,
        0,
        1
    )

    result = pd.DataFrame({
        "trip_id": trip_id,
        "point_idx": point_idx,
        "cand_arc_id": candidate_arc_ids,
        "label": labels,
        "dist_m": distances,
        "dist_rank": ranks,
        "dist_minus_min": (
            distances - min_dist[point_idx]
        ),
        "heading_diff": heading_diff,
        "arc_len": arc_lengths[line_idx],
        "pos_on_arc": positions,
        "speed": speed[point_idx],
        "dt": dt[point_idx],
        "n_cands": n_cands[point_idx]
    })

    return result[OUTPUT_COLUMNS], missed, len(points)


def build_partition(directory, radius, k):
    trip_dirs = sorted(
        path
        for path in glob.glob(
            os.path.join(directory, "*")
        )
        if os.path.isdir(path)
    )

    if not trip_dirs:
        raise ValueError(
            f"No trip folders found in {directory}"
        )

    frames = []
    total_missed = 0
    total_points = 0

    for trip_dir in trip_dirs:
        frame, missed, count = build_trip(
            trip_dir,
            radius,
            k
        )

        frames.append(frame)
        total_missed += missed
        total_points += count

        print(
            os.path.basename(trip_dir),
            "rows:", len(frame),
            "missed:", missed
        )

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    return combined, total_missed, total_points


def main():
    with open(
        "configs/config.yaml",
        encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    radius = float(config["radius_m"])
    k = int(config["K"])

    os.makedirs("results/csv", exist_ok=True)

    train, train_missed, train_points = build_partition(
        config["train_dir"],
        radius,
        k
    )

    student_test, test_missed, test_points = build_partition(
        config["test_dir"],
        radius,
        k
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
        {
            "partition": "train",
            "total_points": train_points,
            "missed_points": train_missed,
            "candidate_miss_rate": (
                train_missed / train_points
                if train_points else 0
            )
        },
        {
            "partition": "student_test",
            "total_points": test_points,
            "missed_points": test_missed,
            "candidate_miss_rate": (
                test_missed / test_points
                if test_points else 0
            )
        }
    ])

    miss_report.to_csv(
        "results/csv/candidate_miss_rate.csv",
        index=False
    )

    for name, frame in [
        ("train", train),
        ("student_test", student_test)
    ]:
        print(f"\n{name}")
        print("Rows:", len(frame))
        print(
            "Positive ratio:",
            frame["label"].mean()
        )
        print(
            "Unique trips:",
            frame["trip_id"].nunique()
        )
        print(
            "Maximum candidates per point:",
            frame.groupby(
                ["trip_id", "point_idx"]
            ).size().max()
        )

    print("\nCandidate miss rates:")
    print(miss_report.to_string(index=False))


if __name__ == "__main__":
    main()
