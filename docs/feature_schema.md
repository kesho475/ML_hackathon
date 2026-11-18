# Feature schema

Files: `results/features_train.parquet`, `results/features_student_test.parquet`.
One row per (GPS point, candidate arc).

| column | meaning |
|---|---|
| trip_id | trip folder name |
| point_idx | row index in .track |
| cand_arc_id | candidate arc (row index in .arcs) |
| label | 1 if cand_arc_id is in the reference route, else 0 |
| dist_m | metres from GPS point to arc (projected CRS, config `crs`) |
| dist_rank | rank of dist_m among this point's candidates (0 = nearest) |
| dist_minus_min | dist_m minus nearest candidate's dist_m |
| heading_diff | abs angle (deg, 0-180) between GPS motion direction and arc direction |
| arc_len | arc length, m |
| pos_on_arc | projection position along arc, 0-1 |
| speed | m/s from previous point |
| dt | seconds since previous point |
| n_cands | number of candidates for this point |

Candidates: arcs within `radius_m`, else K nearest, capped at `K`.
