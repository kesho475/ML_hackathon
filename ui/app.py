"""PathSnap evaluator UI (Laptop 2). Everything comes from the API; this app reads no data files or models.
Run: streamlit run ui/app.py"""
import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: map-matching evaluation")

api = st.sidebar.text_input("API URL", "http://localhost:8000").rstrip("/")
if st.sidebar.button("Check connection", key="check"):
    try:
        r = requests.get(f"{api}/health", timeout=5)
        if r.ok:
            st.sidebar.success(f"Connected: {r.json()}")
        else:
            st.sidebar.error(f"API answered HTTP {r.status_code}")
    except requests.RequestException as e:
        st.sidebar.error(f"Cannot reach the API at {api}: {type(e).__name__}")


def api_error(resp):
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    return f"API error HTTP {resp.status_code}: {detail}"


try:
    r = requests.get(f"{api}/trips", timeout=10)
except requests.RequestException:
    st.error(f"Cannot reach the API at {api}. Start it on Laptop 1 and check the URL and network.")
    st.stop()
if not r.ok:
    st.error(api_error(r))
    st.stop()
trips = r.json()

trip_id = st.selectbox("Trip", trips)
if st.button("Run prediction", type="primary", key="run"):
    try:
        with st.spinner("Running the prediction pipeline on the API..."):
            resp = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=300)
        if resp.ok:
            st.session_state["result"] = resp.json()
        else:
            st.session_state.pop("result", None)
            st.error(api_error(resp))
    except requests.RequestException as e:
        st.session_state.pop("result", None)
        st.error(f"Request to the API failed ({type(e).__name__}). Is it still running?")

res = st.session_state.get("result")
if res:
    st.subheader(f"Trip {res['trip_id']}")
    m = res["metrics"]
    c = st.columns(5)
    c[0].metric("Top-1 accuracy", f"{m['top1_accuracy']:.2%}")
    c[1].metric("Route continuity", f"{m['route_continuity']:.2%}")
    c[2].metric("GPS points", f"{res['n_points']:,}")
    c[3].metric("Processing time", f"{m['processing_time_s']:.2f} s")
    c[4].metric("Decoder", res["decoder"])

    gps, route = res["gps_points"], res["predicted_route"]  # already [lat, lon]
    fmap = folium.Map(tiles="cartodbpositron")
    pts = folium.FeatureGroup("GPS points (blue)")
    for lat, lon in gps:
        folium.CircleMarker((lat, lon), radius=2, color="blue", weight=1, fill=True, fill_opacity=0.7).add_to(pts)
    line = folium.FeatureGroup("Predicted route (red)")
    if route:
        folium.PolyLine(route, color="red", weight=4, opacity=0.8).add_to(line)
    pts.add_to(fmap)
    line.add_to(fmap)
    lats, lons = [p[0] for p in gps], [p[1] for p in gps]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    fmap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;'
        'border:1px solid #888;font-size:13px"><b>Legend</b><br><span style="color:blue">&#9679;</span> GPS points'
        '<br><span style="color:red">&#9473;</span> Predicted route</div>'))
    folium.LayerControl().add_to(fmap)
    st_folium(fmap, height=620, use_container_width=True, returned_objects=[], key=f"map_{res['trip_id']}")
