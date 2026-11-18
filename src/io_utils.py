import os
import pandas as pd


def _read(path, names, **kw):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, **kw)


def load_trip(trip_dir):
    """Return track (lon,lat,t), nodes (lon,lat; index=node_id), arcs (from_node,to_node; index=arc_id), route (list of arc_id)."""
    tid = os.path.basename(os.path.normpath(trip_dir))
    p = os.path.join(trip_dir, tid)
    track = _read(p + ".track", ["lon", "lat", "t"])
    nodes = _read(p + ".nodes", ["lon", "lat"])
    arcs = _read(p + ".arcs", ["from_node", "to_node"], dtype="int64")
    route_file = p + ".route"
    route = _read(route_file, ["arc_id"])["arc_id"].astype(int).tolist() if os.path.exists(route_file) else []
    return track, nodes, arcs, route


def get_utm_crs(lon, lat):
    # Every trip is projected to its own UTM zone: trips span many countries and both
    # hemispheres, so one fixed EPSG would distort distances. Pass the trip's mean track lon/lat.
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"
