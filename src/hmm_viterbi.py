
import os
import time
from functools import lru_cache

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from pyproj import Transformer

try:
    from .io_utils import load_trip, get_utm_crs
except ImportError:
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


def build_road_graph(trip_dir):
    track, nodes, arcs, route = load_trip(trip_dir)

    lon = track["lon"].to_numpy(dtype=float)
    lat = track["lat"].to_numpy(dtype=float)

    crs = get_utm_crs(
        float(np.median(lon)),
        float(np.median(lat))
    )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    x, y = transformer.transform(
        nodes["lon"].to_numpy(dtype=float),
        nodes["lat"].to_numpy(dtype=float)
    )

    node_xy = np.column_stack((x, y))
    arc_array = arcs.to_numpy(dtype=np.int64)

    graph = nx.DiGraph()
    arc_info = {}

    for arc_id, (u, v) in enumerate(arc_array):
        if (
            u < 0 or v < 0
            or u >= len(nodes)
            or v >= len(nodes)
        ):
            continue

        length = float(
            np.linalg.norm(node_xy[v] - node_xy[u])
        )

        if not np.isfinite(length) or length <= 0:
            continue

        graph.add_edge(u, v, weight=length)

        arc_info[arc_id] = {
            "from_node": int(u),
            "to_node": int(v),
            "length": length
        }

    px, py = transformer.transform(lon, lat)

    gps_xy = np.column_stack((px, py))

    return graph, arc_info, track, gps_xy


def calculate_emissions(model, candidates):
    missing = set(FEATURES) - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing model features: {sorted(missing)}"
        )

    probabilities = model.predict_proba(
        candidates[FEATURES]
    )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model does not contain positive class 1.")

    positive_index = classes.index(1)

    probabilities = probabilities[:, positive_index]

    return np.clip(
        probabilities.astype(float),
        1e-12,
        1.0
    )


def viterbi_decode(
    candidates,
    emissions,
    graph,
    arc_info,
    gps_xy,
    beta=50.0,
    transition_floor=1e-8,
    gap_threshold=300.0
):
    if beta <= 0:
        raise ValueError("beta must be positive.")

    if len(candidates) != len(emissions):
        raise ValueError("Candidate and emission lengths differ.")

    data = candidates.copy()
    data["_emission"] = emissions

    data = data.sort_values(
        ["point_idx", "dist_rank"],
        kind="stable"
    )

    groups = [
        group.copy()
        for _, group in data.groupby("point_idx", sort=True)
    ]

    if not groups:
        return pd.DataFrame(
            columns=["point_idx", "cand_arc_id", "probability"]
        )

    @lru_cache(maxsize=100000)
    def network_distance(source, target):
        if source == target:
            return 0.0

        try:
            return float(
                nx.shortest_path_length(
                    graph,
                    source,
                    target,
                    weight="weight"
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return np.inf

    def transition_distance(previous, current):
        previous_id = int(previous["cand_arc_id"])
        current_id = int(current["cand_arc_id"])

        a = arc_info.get(previous_id)
        b = arc_info.get(current_id)

        if a is None or b is None:
            return np.inf

        previous_pos = float(
            np.clip(previous["pos_on_arc"], 0, 1)
        )

        current_pos = float(
            np.clip(current["pos_on_arc"], 0, 1)
        )

        if previous_id == current_id:
            if current_pos >= previous_pos:
                return (
                    current_pos - previous_pos
                ) * a["length"]

        middle = network_distance(
            a["to_node"],
            b["from_node"]
        )

        if not np.isfinite(middle):
            return np.inf

        remaining = (
            1.0 - previous_pos
        ) * a["length"]

        beginning = current_pos * b["length"]

        return remaining + middle + beginning

    predictions = []

    for segment_start in range(len(groups)):
        if segment_start > 0:
            previous_idx = int(
                groups[segment_start - 1]["point_idx"].iloc[0]
            )
            current_idx = int(
                groups[segment_start]["point_idx"].iloc[0]
            )

            if current_idx != previous_idx + 1:
                pass

            gps_gap = float(
                np.linalg.norm(
                    gps_xy[current_idx] - gps_xy[previous_idx]
                )
            )

            previous_candidates = groups[segment_start - 1]
            current_candidates = groups[segment_start]

            if (
                current_idx != previous_idx + 1
                or gps_gap > gap_threshold
            ):
                continue

        break

    segments = []
    current_segment = [groups[0]]

    for i in range(1, len(groups)):
        previous_idx = int(groups[i - 1]["point_idx"].iloc[0])
        current_idx = int(groups[i]["point_idx"].iloc[0])

        gps_gap = float(
            np.linalg.norm(
                gps_xy[current_idx] - gps_xy[previous_idx]
            )
        )

        if (
            current_idx != previous_idx + 1
            or gps_gap > gap_threshold
        ):
            segments.append(current_segment)
            current_segment = []

        current_segment.append(groups[i])

    if current_segment:
        segments.append(current_segment)

    for sequence in segments:
        states = [
            group.reset_index(drop=True)
            for group in sequence
        ]

        scores = [
            np.log(
                np.clip(
                    group["_emission"].to_numpy(dtype=float),
                    1e-12,
                    1.0
                )
            )
            for group in states
        ]

        backpointers = [
            np.full(len(states[0]), -1, dtype=np.int32)
        ]

        previous_scores = scores[0].copy()

        for t in range(1, len(states)):
            previous = states[t - 1]
            current = states[t]

            previous_idx = int(previous["point_idx"].iloc[0])
            current_idx = int(current["point_idx"].iloc[0])

            observed_distance = float(
                np.linalg.norm(
                    gps_xy[current_idx] - gps_xy[previous_idx]
                )
            )

            transition_matrix = np.empty(
                (len(previous), len(current)),
                dtype=float
            )

            for i, (_, prev_row) in enumerate(previous.iterrows()):
                for j, (_, curr_row) in enumerate(current.iterrows()):
                    road_distance = transition_distance(
                        prev_row,
                        curr_row
                    )

                    if np.isfinite(road_distance):
                        cost = abs(
                            road_distance - observed_distance
                        )

                        transition_matrix[i, j] = (
                            -cost / beta
                        )
                    else:
                        transition_matrix[i, j] = np.log(
                            transition_floor
                        )

            values = (
                previous_scores[:, None]
                + transition_matrix
            )

            best_previous = np.argmax(values, axis=0)

            previous_scores = (
                values[best_previous, np.arange(len(current))]
                + scores[t]
            )

            backpointers.append(
                best_previous.astype(np.int32)
            )

        state_index = int(np.argmax(previous_scores))

        path = [state_index]

        for t in range(len(states) - 1, 0, -1):
            state_index = int(
                backpointers[t][state_index]
            )
            path.append(state_index)

        path.reverse()

        for group, selected in zip(states, path):
            row = group.iloc[selected]

            predictions.append({
                "point_idx": int(row["point_idx"]),
                "cand_arc_id": int(row["cand_arc_id"]),
                "probability": float(row["_emission"])
            })

    return pd.DataFrame(predictions).sort_values(
        "point_idx"
    ).reset_index(drop=True)


def reconstruct_trip(
    trip_dir,
    features,
    model_path="models/HistGradientBoosting.joblib",
    beta=50.0,
    gap_threshold=300.0
):
    start = time.perf_counter()

    trip_id = os.path.basename(
        os.path.normpath(trip_dir)
    )

    candidates = features[
        features["trip_id"].astype(str) == trip_id
    ].copy()

    if candidates.empty:
        raise ValueError(
            f"No candidate features found for trip {trip_id}"
        )

    model = joblib.load(model_path)

    emissions = calculate_emissions(
        model,
        candidates
    )

    graph, arc_info, track, gps_xy = build_road_graph(
        trip_dir
    )

    predictions = viterbi_decode(
        candidates,
        emissions,
        graph,
        arc_info,
        gps_xy,
        beta=beta,
        gap_threshold=gap_threshold
    )

    predictions.insert(
        0,
        "trip_id",
        trip_id
    )

    predictions.attrs["inference_time"] = (
        time.perf_counter() - start
    )

    return predictions


def classifier_only_predictions(features, model):
    candidates = features.copy()

    emissions = calculate_emissions(
        model,
        candidates
    )

    candidates["_probability"] = emissions

    result = (
        candidates.sort_values(
            ["trip_id", "point_idx", "_probability"],
            ascending=[True, True, False],
            kind="stable"
        )
        .drop_duplicates(
            ["trip_id", "point_idx"]
        )
    )

    return result[
        ["trip_id", "point_idx", "cand_arc_id", "_probability"]
    ].rename(
        columns={"_probability": "probability"}
    ).reset_index(drop=True)


def evaluate_predictions(predictions, features):
    truth = (
        features.sort_values(
            ["trip_id", "point_idx", "label"],
            ascending=[True, True, False]
        )
        .drop_duplicates(
            ["trip_id", "point_idx"]
        )
        [["trip_id", "point_idx", "cand_arc_id"]]
        .rename(columns={"cand_arc_id": "reference_arc"})
    )

    merged = predictions.merge(
        truth,
        on=["trip_id", "point_idx"],
        how="inner"
    )

    merged["correct"] = (
        merged["cand_arc_id"]
        == merged["reference_arc"]
    )

    pointwise_accuracy = float(
        merged["correct"].mean()
    ) if len(merged) else 0.0

    return {
        "pointwise_accuracy": pointwise_accuracy,
        "predicted_points": len(merged),
        "correct_points": int(merged["correct"].sum())
    }
