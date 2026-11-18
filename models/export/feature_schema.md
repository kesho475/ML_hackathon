# Feature schema

Files: `results/features_train.parquet`, `results/features_student_test.parquet`.
One row per (GPS point, candidate arc).

| column | meaning |
|---|---|
| trip_id | trip folder name |
| point_idx | row index in .track |
| cand_arc_id | candidate arc (row index in .arcs) |
| label | 1 if cand_arc_id is the nearest reference-route arc for this GPS point, else 0 |
| dist_m | metres from GPS point to arc, using the trip-specific UTM projection |
| dist_rank | rank of dist_m among this point's candidates (0 = nearest) |
| dist_minus_min | dist_m minus nearest candidate's dist_m |
| heading_diff | absolute angle (degrees, 0-180) between GPS motion direction and arc direction; set to 90 for the first GPS point, GPS movement under 1 m, or a zero-length arc |
| arc_len | arc length, metres |
| pos_on_arc | projection position along arc, 0-1 |
| speed | m/s from previous point; 0 when the time difference is zero |
| dt | non-negative seconds since previous point |
| n_cands | number of candidates for this point |

## Candidate generation

Candidates are arcs within `radius_m` (200 m), capped at `K` (8). If a GPS point has no arcs within the radius, fallback candidates are found using exact point-to-line distances. `STRtree.query_nearest` identifies the true nearest line, and the search radius is expanded until up to `K` nearby lines are available; candidates are ranked by their actual point-to-line distances.

Each trip is projected using its own UTM CRS selected from its GPS coordinates. No fixed CRS is used.

Zero-length and invalid arcs are excluded before building the spatial index and feature rows.

## Labels and heading

For each GPS point, the reference label is the nearest valid arc among the trip's reference-route arcs. A candidate receives label 1 when its arc ID equals that nearest reference-route arc ID. This is a nearest-route-arc approximation and may be ambiguous at junctions or parallel roads.

Heading difference is neutralized to 90 degrees for the first GPS point, when movement from the previous point is less than 1 metre, or when the candidate arc has zero length. Zero-length arcs are filtered, but the condition is retained as a defensive safeguard.

## Excluded training trips

Trips `00000033` and `00000054` are excluded from the training partition only. Their GPS median distances from the reference route are approximately 122 m and 139 m, respectively; their reference routes are too short for their tracks, and approximately 66% of points miss candidate arcs. Their labels are therefore considered unreliable. They remain in the raw dataset and are not removed from `student_test`.

The candidate miss-rate report records training miss rate before exclusion, training miss rate after exclusion, and student-test miss rate.
