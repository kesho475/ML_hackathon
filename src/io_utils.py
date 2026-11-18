
import os
import pandas as pd
from pyproj import CRS


def get_utm_crs(lon, lat):
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _read(path, names, **kw):
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=names,
        **kw
    )


def load_trip(trip_dir):
    tid = os.path.basename(os.path.normpath(trip_dir))
    p = os.path.join(trip_dir, tid)

    track = _read(p + ".track", ["lon", "lat", "t"])
    nodes = _read(p + ".nodes", ["lon", "lat"])
    arcs = _read(
        p + ".arcs",
        ["from_node", "to_node"],
        dtype="int64"
    )

    route_file = p + ".route"

    route = (
        _read(route_file, ["arc_id"])["arc_id"]
        .astype(int)
        .tolist()
        if os.path.exists(route_file)
        else []
    )

    return track, nodes, arcs, route
