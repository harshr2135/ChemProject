"""Load the trained absorbance and colour models and score one sample."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lab_store import rgb_to_hex

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"

MODEL = {
    "label": "Concentration only",
    "abs": ARTIFACT_DIR / "absorbance_model_no_volume.joblib",
    "rgb": ARTIFACT_DIR / "colour_model_rgb_no_volume.joblib",
    "meta": ARTIFACT_DIR / "model_metadata_no_volume.joblib",
}

_CACHE: tuple | None = None


def load_model():
    global _CACHE
    if _CACHE is None:
        if not MODEL["abs"].exists() or not MODEL["rgb"].exists():
            raise FileNotFoundError(
                "Concentration-only model files are missing. "
                "Run the notebook training cells first."
            )
        metadata = joblib.load(MODEL["meta"]) if MODEL["meta"].exists() else {}
        _CACHE = (joblib.load(MODEL["abs"]), joblib.load(MODEL["rgb"]), metadata)
    return _CACHE


def model_info() -> dict:
    _abs_model, _rgb_model, metadata = load_model()
    metadata = metadata if isinstance(metadata, dict) else {}
    return {
        "label": MODEL["label"],
        "abs_algorithm": metadata.get("abs_algorithm", "unknown"),
        "color_algorithm": metadata.get("color_algorithm", "unknown"),
    }


def predict(concentration: float) -> dict:
    abs_model, rgb_model, _metadata = load_model()
    sample = pd.DataFrame({"conc": [concentration]})
    predicted_abs = float(abs_model.predict(sample)[0])
    predicted_rgb = np.rint(
        np.clip(rgb_model.predict(sample.assign(abs_pred=[predicted_abs]))[0], 0, 255)
    ).astype(int)
    return {
        "predicted_absorbance": predicted_abs,
        "predicted_rgb": [int(value) for value in predicted_rgb],
        "predicted_hex": rgb_to_hex(predicted_rgb),
        **model_info(),
    }
