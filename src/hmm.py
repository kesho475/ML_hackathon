
"""Efficient HMM/Viterbi decoder for PathSnap.

Inputs:
    cands: candidate DataFrame with point_idx and cand_arc_id.
    probs: HGB probability for each candidate, in the same row order.
    arcs: DataFrame indexed by arc_id, with from_node and to_node.

Output:
    pd.Series indexed by point_idx, containing predicted cand_arc_id.
"""

import numpy as np
import pandas as pd


# HMM parameters
TRANSITION_FLOOR = 1e-8
GAP_THRESHOLD_SECONDS = 300.0
EMISSION_EPSILON = 1e-12


def viterbi_decode(cands, probs, arcs) -> pd.Series:
    """Decode the most likely road-arc sequence using Viterbi."""

    required_candidates = {"point_idx", "cand_arc_id"}
    required_arcs = {"from_node", "to_node"}

    if not required_candidates.issubset(cands.columns):
        raise ValueError(
            f"cands must contain {required_candidates}"
        )

    if not required_arcs.issubset(arcs.columns):
        raise ValueError(
            f"arcs must contain {required_arcs}"
        )

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

    # Build a fast arc connectivity lookup.
    arc_from = {}
    arc_to = {}

    for arc_id, row in arcs.iterrows():
        arc_id = int(arc_id)
        arc_from[arc_id] = int(row["from_node"])
        arc_to[arc_id] = int(row["to_node"])

    # Group candidates by GPS observation.
    groups = []

    for point_idx, group in data.groupby(
        "point_idx", sort=True
    ):
        group = group.reset_index(drop=True)

        arc_ids = group["cand_arc_id"].to_numpy(dtype=np.int64)
        emissions = group["_emission"].to_numpy(dtype=float)

        groups.append((point_idx, arc_ids, emissions))

    if not groups:
        return pd.Series(dtype="int64", name="cand_arc_id")

    predictions = []
    start = 0

    # Split into independent sequences when there is a GPS gap.
    def has_gap(previous_point, current_point):
        if "dt" not in data.columns:
            return False

        current_rows = data.loc[
            data["point_idx"] == current_point, "dt"
        ]

        if current_rows.empty:
            return False

        dt = pd.to_numeric(
            current_rows, errors="coerce"
        ).dropna()

        return not dt.empty and float(dt.iloc[0]) > GAP_THRESHOLD_SECONDS

    def decode_segment(segment):
        scores = [segment[0][2].copy()]
        backpointers = []

        for t in range(1, len(segment)):
            _, previous_arcs, _ = segment[t - 1]
            _, current_arcs, current_emissions = segment[t]

            previous_scores = scores[-1]

            next_scores = np.full(
                len(current_arcs), -np.inf
            )

            back = np.zeros(
                len(current_arcs), dtype=np.int64
            )

            for j, current_arc in enumerate(current_arcs):
                best_score = -np.inf
                best_previous = 0

                current_from = arc_from.get(
                    int(current_arc)
                )

                for i, previous_arc in enumerate(previous_arcs):
                    previous_arc = int(previous_arc)

                    previous_score = previous_scores[i]

                    previous_to = arc_to.get(previous_arc)

                    # Same arc or directly connected arcs.
                    connected = (
                        int(current_arc) == previous_arc
                        or (
                            previous_to is not None
                            and current_from is not None
                            and previous_to == current_from
                        )
                    )

                    transition = (
                        0.0
                        if connected
                        else np.log(TRANSITION_FLOOR)
                    )

                    candidate_score = (
                        previous_score + transition
                    )

                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_previous = i

                next_scores[j] = (
                    best_score + current_emissions[j]
                )
                back[j] = best_previous

            scores.append(next_scores)
            backpointers.append(back)

        # Backtrack.
        state = int(np.argmax(scores[-1]))
        selected = [state]

        for back in reversed(backpointers):
            state = int(back[state])
            selected.append(state)

        selected.reverse()

        for t, state in enumerate(selected):
            point_idx, arc_ids, _ = segment[t]

            predictions.append(
                (point_idx, int(arc_ids[state]))
            )

    for i in range(1, len(groups)):
        previous_point = groups[i - 1][0]
        current_point = groups[i][0]

        if has_gap(previous_point, current_point):
            decode_segment(groups[start:i])
            start = i

    decode_segment(groups[start:])

    result = pd.Series(
        data=[arc_id for _, arc_id in predictions],
        index=pd.Index(
            [point for point, _ in predictions],
            name="point_idx"
        ),
        name="cand_arc_id",
        dtype="int64"
    )

    return result
