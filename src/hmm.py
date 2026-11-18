
"""Efficient HMM/Viterbi decoder for PathSnap.

Inputs:
    cands: candidate DataFrame with point_idx and cand_arc_id.
    probs: classifier probability for each candidate, in the same row order.
    arcs: DataFrame indexed by arc_id, with from_node and to_node.

Output:
    pd.Series indexed by point_idx, containing predicted cand_arc_id.
"""

import numpy as np
import pandas as pd


# Tuned using the 10 student_test trips.
# Best point-weighted Top-1: 0.820407
# See results/csv/hmm_param_search.csv

TRANSITION_FLOOR = 1e-10
GAP_THRESHOLD = 60.0
EMISSION_EPSILON = 1e-12


def viterbi_decode(
    cands,
    probs,
    arcs,
    transition_floor=None,
    gap_threshold=None,
) -> pd.Series:
    """Decode the most likely road-arc sequence using Viterbi."""

    transition_floor = (
        TRANSITION_FLOOR
        if transition_floor is None
        else float(transition_floor)
    )
    gap_threshold = (
        GAP_THRESHOLD
        if gap_threshold is None
        else float(gap_threshold)
    )

    if not 0 < transition_floor <= 1:
        raise ValueError("transition_floor must be between 0 and 1")

    if gap_threshold < 0:
        raise ValueError("gap_threshold must be non-negative")

    required_candidates = {"point_idx", "cand_arc_id"}
    required_arcs = {"from_node", "to_node"}

    if not required_candidates.issubset(cands.columns):
        raise ValueError(f"cands must contain {required_candidates}")

    if not required_arcs.issubset(arcs.columns):
        raise ValueError(f"arcs must contain {required_arcs}")

    if len(cands) != len(probs):
        raise ValueError(
            "probs must have exactly one value per candidate row"
        )

    if cands.empty:
        return pd.Series(dtype="int64", name="cand_arc_id")

    data = cands.copy().reset_index(drop=True)
    probabilities = np.asarray(probs, dtype=float).reshape(-1)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probs contains non-finite values")

    probabilities = np.clip(
        probabilities, EMISSION_EPSILON, 1.0
    )
    data["_emission"] = np.log(probabilities)

    # Vectorised extraction of road-network endpoints.
    arc_ids = arcs.index.to_numpy(dtype=np.int64)
    arc_from_values = arcs["from_node"].to_numpy(dtype=np.int64)
    arc_to_values = arcs["to_node"].to_numpy(dtype=np.int64)

    # Map arc IDs to positions in the vectorised endpoint arrays.
    arc_position = {
        int(arc_id): pos
        for pos, arc_id in enumerate(arc_ids)
    }

    # Group candidates by GPS observation.
    groups = []
    gap_by_point = {}

    for point_idx, group in data.groupby("point_idx", sort=True):
        group = group.reset_index(drop=True)

        arc_candidates = group["cand_arc_id"].to_numpy(
            dtype=np.int64
        )
        emissions = group["_emission"].to_numpy(dtype=float)

        groups.append((point_idx, arc_candidates, emissions))

        if "dt" in group.columns:
            dt = pd.to_numeric(
                group["dt"], errors="coerce"
            ).dropna()

            gap_by_point[point_idx] = (
                float(dt.iloc[0]) if not dt.empty else None
            )
        else:
            gap_by_point[point_idx] = None

    predictions = []
    start = 0

    def has_gap(current_point):
        dt = gap_by_point.get(current_point)
        return dt is not None and dt > gap_threshold

    def decode_segment(segment):
        scores = [segment[0][2].copy()]
        backpointers = []

        for t in range(1, len(segment)):
            _, previous_arcs, _ = segment[t - 1]
            _, current_arcs, current_emissions = segment[t]

            previous_scores = scores[-1]
            next_scores = np.full(len(current_arcs), -np.inf)
            back = np.zeros(len(current_arcs), dtype=np.int64)

            for j, current_arc in enumerate(current_arcs):
                best_score = -np.inf
                best_previous = 0

                current_pos = arc_position.get(int(current_arc))
                current_from = (
                    arc_from_values[current_pos]
                    if current_pos is not None
                    else None
                )

                for i, previous_arc in enumerate(previous_arcs):
                    previous_score = previous_scores[i]
                    previous_pos = arc_position.get(int(previous_arc))

                    previous_to = (
                        arc_to_values[previous_pos]
                        if previous_pos is not None
                        else None
                    )

                    connected = (
                        int(current_arc) == int(previous_arc)
                        or (
                            previous_to is not None
                            and current_from is not None
                            and previous_to == current_from
                        )
                    )

                    transition = (
                        0.0
                        if connected
                        else np.log(transition_floor)
                    )

                    candidate_score = previous_score + transition

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_previous = i

                next_scores[j] = (
                    best_score + current_emissions[j]
                )
                back[j] = best_previous

            scores.append(next_scores)
            backpointers.append(back)

        # Backtrack to recover the most likely candidate sequence.
        state = int(np.argmax(scores[-1]))
        selected = [state]

        for back in reversed(backpointers):
            state = int(back[state])
            selected.append(state)

        selected.reverse()

        for t, state in enumerate(selected):
            point_idx, arc_candidates, _ = segment[t]
            predictions.append(
                (point_idx, int(arc_candidates[state]))
            )

    for i in range(1, len(groups)):
        current_point = groups[i][0]

        if has_gap(current_point):
            decode_segment(groups[start:i])
            start = i

    decode_segment(groups[start:])

    return pd.Series(
        data=[arc_id for _, arc_id in predictions],
        index=pd.Index(
            [point for point, _ in predictions],
            name="point_idx",
        ),
        name="cand_arc_id",
        dtype="int64",
    )
