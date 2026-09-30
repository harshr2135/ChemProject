"""Load the trained absorbance and colour models and score one sample."""

from __future__ import annotations

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from lab_store import rgb_to_hex

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"

VARIANTS = {
    "volume_concentration": {
        "label": "Volume + concentration",
        "abs": ARTIFACT_DIR / "absorbance_model.joblib",
        "rgb": ARTIFACT_DIR / "colour_model_rgb.joblib",
        "meta": ARTIFACT_DIR / "model_metadata.joblib",
        "uses_volume": True,
    },
    "concentration_only": {
        "label": "Concentration only",
        "abs": ARTIFACT_DIR / "absorbance_model_no_volume.joblib",
        "rgb": ARTIFACT_DIR / "colour_model_rgb_no_volume.joblib",
        "meta": ARTIFACT_DIR / "model_metadata_no_volume.joblib",
        "uses_volume": False,
    },
}

_CACHE: dict[str, tuple] = {}


def _volume_range(metadata) -> tuple[float | None, float | None]:
    if not isinstance(metadata, dict):
        return None, None
    values = []
    for label in metadata.get("volume_labels", []):
        match = re.search(r"(\d+(?:\.\d+)?)", str(label))
        if match:
            values.append(float(match.group(1)))
    if not values:
        return None, None
    return min(values), max(values)


def load_variant(variant_id: str):
    if variant_id not in VARIANTS:
        raise KeyError(variant_id)
    if variant_id not in _CACHE:
        config = VARIANTS[variant_id]
        if not config["abs"].exists() or not config["rgb"].exists():
            raise FileNotFoundError(
                f"Model files for '{config['label']}' are missing. "
                "Run the notebook training cells first."
            )
        metadata = joblib.load(config["meta"]) if config["meta"].exists() else {}
        _CACHE[variant_id] = (
            joblib.load(config["abs"]),
            joblib.load(config["rgb"]),
            metadata,
            config,
        )
    return _CACHE[variant_id]


def variant_info(variant_id: str) -> dict:
    _abs_model, _rgb_model, metadata, config = load_variant(variant_id)
    volume_min, volume_max = _volume_range(metadata)
    return {
        "id": variant_id,
        "label": config["label"],
        "uses_volume": config["uses_volume"],
        "volume_min": volume_min,
        "volume_max": volume_max,
        "abs_algorithm": metadata.get("abs_algorithm", "unknown") if isinstance(metadata, dict) else "unknown",
        "color_algorithm": metadata.get("color_algorithm", "unknown") if isinstance(metadata, dict) else "unknown",
    }


def predict(variant_id: str, concentration: float, volume_ml: float | None = None) -> dict:
    abs_model, rgb_model, _metadata, config = load_variant(variant_id)
    uses_volume = config["uses_volume"]
    if uses_volume and volume_ml is None:
        raise ValueError("Volume is required for this model.")

    if uses_volume:
        sample = pd.DataFrame({"volume_ml": [volume_ml], "conc": [concentration]})
    else:
        sample = pd.DataFrame({"conc": [concentration]})

    predicted_abs = float(abs_model.predict(sample)[0])
    predicted_rgb = np.rint(np.clip(rgb_model.predict(sample.assign(abs_pred=[predicted_abs]))[0], 0, 255)).astype(int)
    info = variant_info(variant_id)
    outside = (
        uses_volume
        and info["volume_min"] is not None
        and info["volume_max"] is not None
        and volume_ml is not None
        and not (info["volume_min"] <= volume_ml <= info["volume_max"])
    )
    return {
        "predicted_absorbance": predicted_abs,
        "predicted_rgb": [int(value) for value in predicted_rgb],
        "predicted_hex": rgb_to_hex(predicted_rgb),
        "volume_outside_range": outside,
        **info,
    }
