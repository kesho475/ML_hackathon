# Feature schema

Files: `results/features_train.parquet`, `results/features_student_test.parquet` (built by `src/build_dataset.py`).
One row per (GPS point, candidate arc). Each trip is projected to its own UTM zone (metres).

| column | meaning |
|---|---|
| trip_id | trip folder name |
| point_idx | row index in .track |
| cand_arc_id | candidate arc (row index in .arcs; unique only within a trip) |
| label | 1 if cand_arc_id is the point's reference arc, else 0 |
| dist_m | metres from GPS point to arc |
| dist_rank | rank of dist_m among this point's candidates (0 = nearest) |
| dist_minus_min | dist_m minus the nearest candidate's dist_m |
| heading_diff | angle 0-180 deg between GPS bearing (prev -> current) and arc bearing (from_node -> to_node) |
| arc_len | arc length, m |
| pos_on_arc | projected position along arc / arc length, 0-1 |
| speed | m/s from previous point; 0 for the first point or dt == 0 |
| dt | seconds since previous point; 0 for the first point |
| n_cands | number of candidates for this point (<= K) |

## Rules
- **Candidates:** the K nearest arcs within `radius_m` (config: K=8, radius_m=200). A point with no arc in range has no rows (counted as a miss).
- **Label (assumption):** reference arc = nearest arc among the trip's `.route` arcs. `.route` has no per-point alignment, so this can be wrong at junctions and on parallel roads.
- **heading_diff:** the first point uses the first step; if the GPS moved < 1 m or the arc has zero length, it is set to 90 (uninformative).
- **Candidate miss rate:** `results/csv/candidate_miss_rate.csv`.

## Known data issue
Train trips 00000033 and 00000054: GPS lies a median of about 120-140 m from the reference route, so most labels are "miss" (miss 66%). Without them the train miss rate is 0.7%.
