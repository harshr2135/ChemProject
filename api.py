"""HTTP API for colour prediction, camera analysis, and stored results."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from lab_store import analyze_beaker_colour, list_analyses, save_analysis, sql_editor_url, using_cloud
from predictor import model_info, predict

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_info()
    yield


app = FastAPI(title="Chemistry Colour Analysis", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _cloud_status() -> dict:
    if not using_cloud():
        return {"cloud": False, "schema_error": None, "sql_editor_url": None}
    try:
        list_analyses(limit=1)
    except Exception as exc:
        return {"cloud": True, "schema_error": str(exc), "sql_editor_url": sql_editor_url()}
    return {"cloud": True, "schema_error": None, "sql_editor_url": sql_editor_url()}


@app.get("/api/config")
def config():
    return {
        "model": model_info(),
        "default_concentration": 1.5,
        "cloud": using_cloud(),
        "schema_error": None,
        "sql_editor_url": sql_editor_url() if using_cloud() else None,
    }


@app.post("/api/predict")
def predict_colour(body: dict):
    try:
        concentration = float(body["concentration"])
        return predict(concentration)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analyze")
async def analyze_colour(image: UploadFile = File(...)):
    try:
        return analyze_beaker_colour(await image.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/analyses")
def analyses():
    try:
        return {"records": list_analyses(), **_cloud_status()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/analyses")
async def store_analysis(
    concentration: float = Form(...),
    image: UploadFile = File(...),
):
    image_bytes = await image.read()
    try:
        analysed = analyze_beaker_colour(image_bytes)
        predicted = predict(concentration)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        record_id = save_analysis(
            model_variant=predicted["label"],
            volume_ml=None,
            concentration=concentration,
            predicted_absorbance=predicted["predicted_absorbance"],
            predicted_rgb=tuple(predicted["predicted_rgb"]),
            predicted_hex=predicted["predicted_hex"],
            analyzed_rgb=tuple(analysed["analyzed_rgb"]),
            analyzed_hex=analysed["analyzed_hex"],
            image_bytes=image_bytes,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"id": record_id, "prediction": predicted, "analysed": analysed}


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
