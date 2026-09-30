"""Colour measurement from a beaker photo, and storage for confirmed analyses.

Saves go to Supabase when SUPABASE_URL and SUPABASE_KEY are set. Otherwise they
stay in the local SQLite file data/analyses.db.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "analyses.db"
CAPTURE_DIR = DATA_DIR / "captures"


def rgb_to_hex(rgb_values) -> str:
    red, green, blue = (int(np.clip(round(value), 0, 255)) for value in rgb_values)
    return f"#{red:02X}{green:02X}{blue:02X}"


def rgb_distance(left, right) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def analyze_beaker_colour(image_bytes: bytes) -> dict:
    """Read the solution colour from the centre of a beaker photo.

    Near-white glass and near-black scale markings are left out. When the
    frame has too little colour, the median of the centre crop is used and
    ``low_colour_signal`` is set so the app can ask for another photo.
    """
    if not image_bytes:
        raise ValueError("The captured image is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Could not read that image. Take the photo again.") from exc

    image.thumbnail((900, 900))
    pixels = np.asarray(image, dtype=np.float32)
    height, width = pixels.shape[:2]
    if height < 8 or width < 8:
        raise ValueError("The photo is too small to analyse.")

    y0, y1 = int(height * 0.32), int(height * 0.78)
    x0, x1 = int(width * 0.30), int(width * 0.70)
    crop = pixels[y0:y1, x0:x1].reshape(-1, 3)
    if len(crop) == 0:
        raise ValueError("Could not find a region to analyse in that photo.")

    luminance = 0.2126 * crop[:, 0] + 0.7152 * crop[:, 1] + 0.0722 * crop[:, 2]
    usable = crop[(luminance > 18) & (luminance < 242)]
    if len(usable) < 30:
        usable = crop

    chroma = usable.max(axis=1) - usable.min(axis=1)
    coloured = usable[chroma >= 15]
    low_colour_signal = len(coloured) < max(40, int(0.08 * len(usable)))
    chosen = usable if low_colour_signal else coloured

    median = np.median(chosen, axis=0)
    rgb = tuple(int(np.clip(round(channel), 0, 255)) for channel in median)
    return {
        "analyzed_rgb": rgb,
        "analyzed_hex": rgb_to_hex(rgb),
        "low_colour_signal": low_colour_signal,
    }


def _read_dotenv() -> dict[str, str]:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _read_streamlit_secrets() -> dict[str, str]:
    try:
        from streamlit.runtime.secrets import secrets_singleton

        if not secrets_singleton.load_if_toml_exists():
            return {}
        return {
            "SUPABASE_URL": str(secrets_singleton.get("SUPABASE_URL", "") or ""),
            "SUPABASE_KEY": str(secrets_singleton.get("SUPABASE_KEY", "") or ""),
        }
    except Exception:
        return {}


def cloud_credentials() -> tuple[str, str] | None:
    """Return (project URL, anon key) when cloud storage is configured."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        secrets = _read_streamlit_secrets()
        url = url or secrets.get("SUPABASE_URL", "").strip()
        key = key or secrets.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        file_values = _read_dotenv()
        url = url or file_values.get("SUPABASE_URL", "").strip()
        key = key or file_values.get("SUPABASE_KEY", "").strip()
    url = url.rstrip("/")
    if url and key:
        return url, key
    return None


def using_cloud() -> bool:
    return cloud_credentials() is not None


def sql_editor_url() -> str:
    creds = cloud_credentials()
    if not creds:
        return "https://supabase.com/dashboard"
    host = creds[0].split("//", 1)[-1].split("/", 1)[0]
    project_ref = host.split(".")[0]
    return f"https://supabase.com/dashboard/project/{project_ref}/sql/new"


def _friendly_cloud_error(status: int, detail: str) -> str:
    lowered = detail.lower()
    if "schema cache" in lowered or "does not exist" in lowered or "pgrst205" in lowered:
        return (
            "Supabase accepted the key, but the analyses table has not been created yet. "
            f"Open {sql_editor_url()}, paste supabase_schema.sql, and click Run."
        )
    if status in (401, 403) or "jwt" in lowered:
        return (
            "Supabase rejected the API key. In Project Settings → API Keys, copy the "
            "publishable key into SUPABASE_KEY. Do not use the secret key."
        )
    return f"Cloud database error ({status}): {detail[:400]}"


def _api(method, url, key, payload=None, content_type="application/json", extra_headers=None):
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8") if content_type == "application/json" else payload
        headers["Content-Type"] = content_type
    if extra_headers:
        headers.update(extra_headers)

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_friendly_cloud_error(exc.code, detail)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the cloud database. {exc.reason}") from exc


def _jpeg_bytes(image_bytes: bytes, max_side: int = 900) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def _analysis_payload(
    *,
    created_at,
    model_variant,
    volume_ml,
    concentration,
    predicted_absorbance,
    predicted_rgb,
    predicted_hex,
    analyzed_rgb,
    analyzed_hex,
    image_path,
) -> dict:
    return {
        "created_at": created_at,
        "model_variant": model_variant,
        "volume_ml": volume_ml,
        "concentration": concentration,
        "predicted_absorbance": predicted_absorbance,
        "predicted_r": int(predicted_rgb[0]),
        "predicted_g": int(predicted_rgb[1]),
        "predicted_b": int(predicted_rgb[2]),
        "predicted_hex": predicted_hex,
        "analyzed_r": int(analyzed_rgb[0]),
        "analyzed_g": int(analyzed_rgb[1]),
        "analyzed_b": int(analyzed_rgb[2]),
        "analyzed_hex": analyzed_hex,
        "rgb_distance": round(rgb_distance(predicted_rgb, analyzed_rgb), 2),
        "image_path": image_path,
    }


def _save_cloud(
    *,
    model_variant,
    volume_ml,
    concentration,
    predicted_absorbance,
    predicted_rgb,
    predicted_hex,
    analyzed_rgb,
    analyzed_hex,
    image_bytes,
) -> int:
    url, key = cloud_credentials()
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    encoded = base64.b64encode(_jpeg_bytes(image_bytes)).decode("ascii")
    body = _api(
        "POST",
        f"{url}/rest/v1/analyses",
        key,
        payload=_analysis_payload(
            created_at=created_at,
            model_variant=model_variant,
            volume_ml=volume_ml,
            concentration=concentration,
            predicted_absorbance=predicted_absorbance,
            predicted_rgb=predicted_rgb,
            predicted_hex=predicted_hex,
            analyzed_rgb=analyzed_rgb,
            analyzed_hex=analyzed_hex,
            image_path=f"data:image/jpeg;base64,{encoded}",
        ),
        extra_headers={"Prefer": "return=representation"},
    )

    rows = json.loads(body.decode("utf-8"))
    if not rows:
        raise RuntimeError("The cloud database did not return the saved row.")
    return int(rows[0]["id"])


def _list_cloud(limit: int) -> list[dict]:
    url, key = cloud_credentials()
    query = urllib.parse.urlencode(
        {
            "select": (
                "id,created_at,model_variant,volume_ml,concentration,"
                "predicted_absorbance,predicted_hex,analyzed_hex,rgb_distance"
            ),
            "order": "id.desc",
            "limit": str(limit),
        }
    )
    body = _api("GET", f"{url}/rest/v1/analyses?{query}", key)
    return json.loads(body.decode("utf-8"))


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    if using_cloud():
        return
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_variant TEXT NOT NULL,
                volume_ml REAL,
                concentration REAL NOT NULL,
                predicted_absorbance REAL NOT NULL,
                predicted_r INTEGER NOT NULL,
                predicted_g INTEGER NOT NULL,
                predicted_b INTEGER NOT NULL,
                predicted_hex TEXT NOT NULL,
                analyzed_r INTEGER NOT NULL,
                analyzed_g INTEGER NOT NULL,
                analyzed_b INTEGER NOT NULL,
                analyzed_hex TEXT NOT NULL,
                rgb_distance REAL,
                image_path TEXT
            )
            """
        )


def save_analysis(
    *,
    model_variant: str,
    volume_ml: float | None,
    concentration: float,
    predicted_absorbance: float,
    predicted_rgb: tuple[int, int, int],
    predicted_hex: str,
    analyzed_rgb: tuple[int, int, int],
    analyzed_hex: str,
    image_bytes: bytes,
) -> int:
    """Save a confirmed analysis and the captured photo. Returns the new row id."""
    if using_cloud():
        return _save_cloud(
            model_variant=model_variant,
            volume_ml=volume_ml,
            concentration=concentration,
            predicted_absorbance=predicted_absorbance,
            predicted_rgb=predicted_rgb,
            predicted_hex=predicted_hex,
            analyzed_rgb=analyzed_rgb,
            analyzed_hex=analyzed_hex,
            image_bytes=image_bytes,
        )

    init_db()
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.jpg"
    image_path = CAPTURE_DIR / filename
    Image.open(io.BytesIO(image_bytes)).convert("RGB").save(image_path, format="JPEG", quality=90)
    relative_path = image_path.relative_to(ROOT).as_posix()

    try:
        with _connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analyses (
                    created_at, model_variant, volume_ml, concentration,
                    predicted_absorbance, predicted_r, predicted_g, predicted_b, predicted_hex,
                    analyzed_r, analyzed_g, analyzed_b, analyzed_hex,
                    rgb_distance, image_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    model_variant,
                    volume_ml,
                    concentration,
                    predicted_absorbance,
                    int(predicted_rgb[0]),
                    int(predicted_rgb[1]),
                    int(predicted_rgb[2]),
                    predicted_hex,
                    int(analyzed_rgb[0]),
                    int(analyzed_rgb[1]),
                    int(analyzed_rgb[2]),
                    analyzed_hex,
                    round(rgb_distance(predicted_rgb, analyzed_rgb), 2),
                    relative_path,
                ),
            )
            return int(cursor.lastrowid)
    except Exception:
        image_path.unlink(missing_ok=True)
        raise


def list_analyses(limit: int = 25) -> list[dict]:
    if using_cloud():
        return _list_cloud(limit)
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT
                id, created_at, model_variant, volume_ml, concentration,
                predicted_absorbance, predicted_hex, analyzed_hex, rgb_distance
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
