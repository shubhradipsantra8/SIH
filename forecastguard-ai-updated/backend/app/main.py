import csv
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional


from app.services.predictor import ForecastGuardModel, MODEL_DIR
from app.services.feature_synth import STATE_META

app = FastAPI(
    title="ForecastGuard AI",
    description="AI-based forecast bust detection for medium-range weather forecasts (SIH 26079)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500",
        "http://localhost:5500",],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model():
    try:
        return ForecastGuardModel.instance()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifacts not found ({e}). Run the ml/ training pipeline first: "
                   f"generate_sample_data.py -> build_features_and_labels.py -> train.py -> explain.py -> analogues.py",
        )


@app.get("/health")
def health():
    """Reports real component status - not a static ping. Used by the
    dashboard's system-health panel. Each field reflects whether that
    artifact/module actually loaded, not a hard-coded 'ready'."""
    try:
        m = ForecastGuardModel.instance()
        return {
            "status": "operational",
            "service": "ForecastGuard AI",
            "components": {
                "backend_api": "online",
                "model": "loaded",
                "forecast_data": "available" if len(m.available_dates) > 0 else "empty",
                "explanation_engine": "ready" if m.explainer is not None else "unavailable",
                "analogue_engine": "ready" if m.analogue_finder is not None else "unavailable",
            },
        }
    except FileNotFoundError as e:
        return {
            "status": "degraded",
            "service": "ForecastGuard AI",
            "detail": str(e),
            "components": {
                "backend_api": "online",
                "model": "unavailable",
                "forecast_data": "unknown",
                "explanation_engine": "unavailable",
                "analogue_engine": "unavailable",
            },
        }


@app.get("/model/confusion_matrix")
def confusion_matrix():
    """Real (not per-lead-day - the CSV holds one overall matrix from the
    held-out 2024-25 test set) confusion matrix at the default 0.5 threshold.
    Order in the CSV is sklearn's confusion_matrix(y_true, y_pred) layout:
    row0=[TN,FP], row1=[FN,TP]."""
    path = f"{MODEL_DIR}/confusion_matrix_test.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail="confusion_matrix_test.csv not found - run ml/train.py")
    with open(path) as f:
        rows = list(csv.reader(f))
    tn, fp = int(rows[0][0]), int(rows[0][1])
    fn, tp = int(rows[1][0]), int(rows[1][1])
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "threshold": 0.5, "note": "overall test-set matrix, not split by lead day"}


@app.get("/model/metrics")
def model_metrics():
    m = get_model()
    return m.metadata


@app.get("/model/states")
def list_states():
    return {"states": list(STATE_META.keys())}


@app.get("/model/available_dates")
def available_dates():
    """The forecast issue (init) dates that actually exist in the real
    HRES prototype dataset. The dashboard should restrict its date picker
    to these - any other date has no real forecast row to predict from."""
    m = get_model()
    return {"dates": m.available_dates}


@app.get("/forecast/predict")
def predict(
    state: str = Query(..., description="Indian state name, e.g. 'Odisha'"),
    date: str = Query(..., description="Forecast issue date, YYYY-MM-DD"),
    lead_day: int = Query(..., ge=1, le=10),
):
    m = get_model()
    if state not in STATE_META:
        raise HTTPException(status_code=400, detail=f"Unknown state '{state}'. See /model/states.")
    try:
        return m.predict_for_state(state, date, lead_day)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/forecast/map")
def forecast_map(
    date: str = Query(..., description="Forecast issue date, YYYY-MM-DD"),
    lead_day: int = Query(..., ge=1, le=10),
):
    m = get_model()
    return m.predict_map(date, lead_day)


@app.get("/forecast/explanation")
def forecast_explanation(
    state: str = Query(...),
    date: str = Query(...),
    lead_day: int = Query(..., ge=1, le=10),
):
    m = get_model()
    result = m.predict_for_state(state, date, lead_day)
    return {
        "state": state, "date": date, "lead_day": lead_day,
        "bust_probability": result["bust_probability"],
        "risk_level": result["risk_level"],
        "drivers": result["drivers"],
        "explanation_method": result["explanation_method"],
    }


@app.get("/forecast/analogues")
def forecast_analogues(
    state: str = Query(...),
    date: str = Query(...),
    lead_day: int = Query(..., ge=1, le=10),
):
    m = get_model()
    result = m.predict_for_state(state, date, lead_day)
    return {
        "state": state, "date": date, "lead_day": lead_day,
        "historical_analogues": result["historical_analogues"],
    }


@app.get("/forecast/day_profile")
def day_profile(
    state: str = Query(...),
    date: str = Query(...),
):
    """Bust probability for Day 1 through Day 10 for a single state - powers
    the lead-time bar chart in the dashboard."""
    m = get_model()
    profile = []
    for lead_day in range(1, 11):
        r = m.predict_for_state(state, date, lead_day)
        profile.append({"lead_day": lead_day, "bust_probability": r["bust_probability"],
                         "reliability": r["reliability"], "risk_level": r["risk_level"]})
    return {"state": state, "date": date, "profile": profile}
