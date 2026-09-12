# ForecastGuard AI — SIH 26079

AI-based forecast bust detection for medium-range weather forecasts.
*"We don't predict the weather — we predict when the forecast itself is likely to fail."*

This repo contains a **fully runnable, end-to-end implementation**: real feature
engineering, a real leakage-safe bust-label definition, a real trained &
calibrated classifier, real SHAP-style explanations, real historical-analogue
search, and a FastAPI backend — plus the original dashboard prototype UI.

## What's real vs. simulated here — read this first

| Component | Status |
|---|---|
| Forecast-error → bust-label logic | **Real**, leakage-safe (thresholds computed only from training years) |
| Feature engineering (spatial neighbourhood stats, derived wind speed, etc.) | **Real** |
| Model training (temporal split, baseline vs. gradient-boosted model) | **Real** — trained and evaluated, not hard-coded |
| Calibration (isotonic) | **Real** |
| Evaluation metrics (ROC-AUC, PR-AUC, Brier, by lead day) | **Real**, measured on held-out 2024–25 data |
| Explainability | **Real** (uses SHAP if installed; otherwise a labeled permutation-importance/z-score fallback — see `ml/explain.py`) |
| Historical analogues | **Real** nearest-neighbour search over the training pool |
| The **raw NWP forecast + ERA5 verification data itself** | **Simulated.** `ml/generate_sample_data.py` builds a physically-structured stand-in dataset (seasonal rainfall, monsoon depressions, cyclones, western disturbances, error growth with lead time) so the rest of the pipeline is fully testable without network access or CDS/NCMRWF credentials. `ml/download_era5_real.py` is the real Copernicus CDS download script to swap in once you have an account. |
| Dashboard frontend | The original prototype HTML (`frontend/forecastguard-ai.html`), still running its own internal synthetic demo data generator. **Live-wiring is in progress**: status pill and real-metric element IDs are hooked up in the markup, but the fetch/cache JS layer and the async `renderAll()` rewrite are not written yet — see "Next implementation steps". |

In short: everything *after* "here is today's forecast" is a genuine, working ML
system. The one thing standing between this and an operational tool is a real
forecast feed, because that requires network access / NCMRWF credentials this
environment doesn't have.

## Repository layout

```
forecastguard-ai/
├── ml/
│   ├── generate_sample_data.py       # synthetic forecast+ERA5 stand-in dataset
│   ├── download_era5_real.py         # REAL Copernicus CDS download script
│   ├── build_features_and_labels.py  # errors, leakage-safe bust labels, features
│   ├── train.py                      # baseline + main model, temporal split, calibration
│   ├── explain.py                    # global + per-prediction explanations
│   ├── analogues.py                  # historical analogue nearest-neighbour index
│   └── requirements.txt
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI endpoints
│   │   └── services/
│   │       ├── predictor.py          # loads trained artifacts, runs inference
│   │       └── feature_synth.py      # synthesizes a "current forecast" for demo (see note below)
│   └── requirements.txt
├── frontend/
│   └── forecastguard-ai.html         # original dashboard prototype (not yet wired to backend)
├── models/                           # trained artifacts (small ones included; regenerate the rest)
└── data/                             # NOT included in this delivery (regenerate — see below, ~1 min)
```

## Quickstart — reproduce everything from scratch

```bash
cd forecastguard-ai
pip install -r ml/requirements.txt --break-system-packages   # or use a venv
pip install -r backend/requirements.txt --break-system-packages

# 1. Build the stand-in dataset (~15s, ~355K rows, ~150MB)
python ml/generate_sample_data.py

# 2. Errors + leakage-safe bust labels + engineered features (~30s)
python ml/build_features_and_labels.py

# 3. Train baseline + main model, calibrate, evaluate (~20s)
python ml/train.py

# 4. Global + per-prediction explainability artifacts
python ml/explain.py

# 5. Historical-analogue index
python ml/analogues.py

# 6. Run the API
cd backend
uvicorn app.main:app --reload --port 8000
# then open http://localhost:8000/docs
```

Everything in steps 1–5 ran successfully during development with these actual
results on 2024–25 held-out data (your numbers will vary slightly by run/seed):

```
ROC-AUC: 0.846   PR-AUC: 0.701   Precision: 0.77   Recall: 0.44   Brier: 0.126
Skill improves with lead day in this synthetic dataset (0.78 AUC at Day 1 → 0.88 at Day 10)
```

Full metrics by lead day are in `models/metrics_by_lead_day.csv` /
`models/metadata.json` after you run `train.py`.

### A note on `feature_synth.py`

The trained model needs a forecast **feature vector** (rainfall, temperature,
pressure, wind, humidity, etc.) for whatever state/date/lead-day the API is
asked about. There's no live NWP feed wired up in this prototype, so
`backend/app/services/feature_synth.py` deterministically synthesizes a
plausible one using the same climatology logic as the training data generator
(seeded by state+date+lead_day, so repeated calls are stable). Everything
downstream — the trained model, calibration, explanation, analogue search —
is genuine inference on that vector. Replace `synthesize_forecast()` with a
reader for your actual NWP output and nothing else changes.

## API endpoints

- `GET /health`
- `GET /model/metrics` — training metadata, real evaluation metrics by lead day
- `GET /model/states` — the 16 states the demo covers
- `GET /forecast/predict?state=Odisha&date=2026-07-14&lead_day=5`
- `GET /forecast/map?date=2026-07-14&lead_day=5` — all 16 states at once
- `GET /forecast/explanation?state=Odisha&date=2026-07-14&lead_day=5`
- `GET /forecast/analogues?state=Odisha&date=2026-07-14&lead_day=5`
- `GET /forecast/day_profile?state=Odisha&date=2026-07-14` — Day 1–10 in one call
- `GET /model/confusion_matrix` — real overall TN/FP/FN/TP from the 2024–25 test set (added this session)

Try `GET /forecast/predict?state=Odisha&date=2026-07-14&lead_day=5` in `/docs` —
you'll get back a real calibrated bust probability, real top drivers, and real
historical analogue statistics.

**Verified working end-to-end this session**: `data/raw`, `data/processed`, and
`models/analogue_index.pkl` (all excluded from this zip to keep it small — see
below) were regenerated from scratch with the Quickstart commands and every
endpoint above was hit against the running server, including
`/forecast/analogues`, which now returns real nearest-neighbour cases instead
of `null`.

## Next implementation steps (not yet done)

These are the remaining items from the "solid SIH project" checklist, roughly
in priority order:

1. **Wire the dashboard HTML to this API.** *In progress.* The markup side is
   done — `frontend/forecastguard-ai.html` now has a `#statusPill` /
   `#statusDot` / `#statusText` header element, `id`s on the top-line eval
   metrics (`#evalRocAuc`, `#evalPrAuc`, `#evalBrier`, `#evalRecall`), and
   `#cmTitle`/`#cmCaption` on the confusion-matrix panel. **Not yet written**:
   the actual `fetch`-based cache layer, the async rewrite of `renderAll()`,
   and the wiring of each panel (map, ranking list, trend chart, eval table,
   confusion matrix) to consume it with graceful fallback to the existing
   `computeCell()` synthetic function when the backend is unreachable.
2. **Swap in XGBoost + SHAP** the moment you have a normal internet-connected
   environment (`pip install xgboost shap`) — `train.py` and `explain.py`
   already auto-detect and prefer them over the sklearn fallbacks used here.
3. **Real data**: run `ml/download_era5_real.py` against Copernicus CDS for
   verification, and source a matching historical NWP forecast archive
   (NCMRWF or a WeatherBench2-compatible dataset) for the forecast half —
   see the "Data strategy" discussion earlier in this project's design notes.
4. **District-level granularity, ensemble spread / run-to-run disagreement
   features, forecast-verification replay screen, model/data health
   monitoring panel** — see the full feature checklist from the design phase.
5. **Docker Compose** for backend + (eventually) Postgres/PostGIS once the
   system needs to persist predictions rather than compute them on demand.

## What NOT to claim to judges

Be upfront that the current forecast inputs are simulated pending a live NWP
feed, while everything downstream (labeling methodology, model, calibration,
explainability, analogue retrieval) is real and measured. That distinction is
more credible than pretending it's all live, and judges with ML backgrounds
will specifically probe for exactly this kind of honesty.
