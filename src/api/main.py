"""PathSnap API (Laptop 1). Run from the project root: uvicorn src.api.main:app --host 0.0.0.0 --port 8000"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from evaluate import evaluate
from pipeline import CFG, ROOT, predict_trip

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

TRIPS_DIR = os.path.join(ROOT, CFG.get("eval_trips_dir", "data/raw/student_test"))

app = FastAPI(title="PathSnap")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])  # LAN demo


class PredictRequest(BaseModel):
    trip_id: str


def trip_ids():
    return sorted(d for d in os.listdir(TRIPS_DIR) if os.path.isdir(os.path.join(TRIPS_DIR, d)))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("REQUEST ip=%s %s %s -> %s (%.0f ms)", request.client.host if request.client else "?", request.method,
             request.url.path, response.status_code, (time.perf_counter() - t0) * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips")
def trips():
    return trip_ids()


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    t0 = time.perf_counter()
    ip = request.client.host if request.client else "?"
    if req.trip_id not in trip_ids():  # also blocks path tricks: only listed folder names are accepted
        log.warning("PREDICT ip=%s trip_id=%s -> 404 unknown trip", ip, req.trip_id)
        raise HTTPException(status_code=404, detail=f"unknown trip_id '{req.trip_id}'")
    trip_dir = os.path.join(TRIPS_DIR, req.trip_id)
    try:
        out = predict_trip(trip_dir)                          # prediction first, no labels involved
        metrics = evaluate(trip_dir, out.pop("predicted_arcs"))  # labels only inside evaluate()
    except Exception:
        log.exception("PREDICT ip=%s trip_id=%s failed", ip, req.trip_id)
        raise HTTPException(status_code=500, detail="prediction failed")
    metrics["processing_time_s"] = round(time.perf_counter() - t0, 3)
    log.info("PREDICT ip=%s endpoint=/predict trip_id=%s decoder=%s n_points=%d top1_accuracy=%.4f "
             "route_continuity=%.4f processing_time=%.3fs", ip, req.trip_id, out["decoder"], out["n_points"],
             metrics["top1_accuracy"], metrics["route_continuity"], metrics["processing_time_s"])
    return {**out, "metrics": metrics}
