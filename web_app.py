from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from lab_store import (
    analyze_beaker_colour,
    init_db,
    list_analyses,
    rgb_to_hex,
    save_analysis,
    sql_editor_url,
    using_cloud,
)

st.set_page_config(
    page_title="Chemistry Colour Analysis",
    page_icon="🧪",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"

MODEL = {
    "label": "Concentration only",
    "abs": ARTIFACT_DIR / "absorbance_model_no_volume.joblib",
    "rgb": ARTIFACT_DIR / "colour_model_rgb_no_volume.joblib",
    "meta": ARTIFACT_DIR / "model_metadata_no_volume.joblib",
}


@st.cache_resource
def load_models(cache_version: str = "conc_only_v1"):
    # cache_version busts Streamlit's model cache when artifacts are swapped
    _ = cache_version
    abs_path = MODEL["abs"]
    rgb_path = MODEL["rgb"]

    if not abs_path.exists() or not rgb_path.exists():
        raise FileNotFoundError(
            "Concentration-only model files are missing. "
            "Run the notebook training cells first to create artifacts."
        )

    abs_model = joblib.load(abs_path)
    rgb_model = joblib.load(rgb_path)
    metadata = {}
    if MODEL["meta"].exists():
        metadata = joblib.load(MODEL["meta"])

    return abs_model, rgb_model, metadata


def predict_sample(abs_model, rgb_model, concentration):
    sample = pd.DataFrame({"conc": [concentration]})
    predicted_abs = float(abs_model.predict(sample)[0])
    sample_color = sample.assign(abs_pred=[predicted_abs])
    predicted_rgb = np.rint(np.clip(rgb_model.predict(sample_color)[0], 0, 255)).astype(int)
    predicted_hex = rgb_to_hex(predicted_rgb)

    return {
        "predicted_absorbance": predicted_abs,
        "predicted_rgb": tuple(int(value) for value in predicted_rgb),
        "predicted_hex": predicted_hex,
    }


def ink_for(hex_color: str) -> str:
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#1c1c1c" if luminance > 170 else "#ffffff"


def colour_card(title: str, hex_color: str | None, detail: str, empty_message: str | None = None) -> str:
    if hex_color is None:
        background = "#eef0f6"
        ink = "#8b909a"
        primary = empty_message or "—"
        secondary = ""
    else:
        background = hex_color
        ink = ink_for(hex_color)
        primary = hex_color
        secondary = f'<div class="colour-detail">{detail}</div>' if detail else ""

    return f"""
    <div class="colour-card" style="background:{background}; color:{ink};">
        <div class="colour-title">{title}</div>
        <div class="colour-hex">{primary}</div>
        {secondary}
    </div>
    """


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, button, input, label {
        font-family: Inter, "Segoe UI", sans-serif;
    }

    .stApp {
        background: #e8edf3;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    footer { display: none; }

    section.main > div.block-container {
        max-width: 980px;
        background: #ffffff;
        border-radius: 28px;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.06);
        padding: 1.6rem 1.7rem 1.2rem;
        margin-top: 1.1rem;
    }

    .field-label {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1c1c1c;
        margin: 0 0 0.15rem 0;
    }

    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #1c1c1c;
        color: #ffffff;
        border-radius: 999px;
        padding: 0.38rem 0.75rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ff3b30;
        box-shadow: 0 0 0 0 rgba(255, 59, 48, 0.7);
        animation: pulse 1.4s infinite;
    }

    @keyframes pulse {
        70% { box-shadow: 0 0 0 7px rgba(255, 59, 48, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0); }
    }

    .camera-note {
        color: #667085;
        font-size: 0.92rem;
        margin: 0.35rem 0 0.7rem 0;
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 12px;
        border: 1px solid #e4e7ec;
        min-height: 48px;
        background: #fbfcfe;
        color: #1c1c1c;
    }

    .st-key-camera_panel {
        border: 1px solid #e6e8ee;
        border-radius: 18px;
        padding: 0.45rem 0.7rem 0.7rem;
        background: #f7f8fb;
    }

    .st-key-camera_panel video,
    .st-key-camera_panel img {
        border-radius: 14px;
    }

    [data-testid="stCameraInputWebcamStyledBox"],
    [data-testid="stCameraInput"] div:has(> img[alt="Snapshot"]) {
        position: relative !important;
    }

    [data-testid="stCameraInputWebcamStyledBox"]::before,
    [data-testid="stCameraInput"] div:has(> img[alt="Snapshot"])::before {
        content: "";
        position: absolute;
        left: 30%;
        top: 32%;
        width: 40%;
        height: 46%;
        border: 2px solid rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.45);
        pointer-events: none;
        z-index: 4;
    }

    [data-testid="stCameraInputWebcamStyledBox"]::after,
    [data-testid="stCameraInput"] div:has(> img[alt="Snapshot"])::after {
        content: "";
        position: absolute;
        left: 50%;
        top: 50%;
        width: 34px;
        height: 34px;
        transform: translate(-50%, -50%);
        border: 2px solid #ffffff;
        border-radius: 50%;
        background:
            linear-gradient(#ffffff, #ffffff) center / 2px 34px no-repeat,
            linear-gradient(#ffffff, #ffffff) center / 34px 2px no-repeat;
        filter: drop-shadow(0 0 1px rgba(0, 0, 0, 0.9));
        pointer-events: none;
        z-index: 5;
    }

    .colour-card {
        border-radius: 18px;
        min-height: 158px;
        padding: 1.35rem 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        box-shadow: 0 10px 24px rgba(20, 30, 60, 0.06);
        margin: 0.35rem 0 0.2rem;
    }

    .colour-title {
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.15;
    }

    .colour-hex {
        margin-top: 0.55rem;
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .colour-detail {
        margin-top: 0.28rem;
        font-size: 0.98rem;
        font-weight: 600;
        opacity: 0.95;
    }

    .st-key-store_wrap button {
        border-radius: 14px !important;
        min-height: 3.15rem;
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: #ffffff !important;
    }

    .st-key-store_wrap button:not(:disabled) {
        background-color: #189a56 !important;
        border: 1px solid #189a56 !important;
    }

    .st-key-store_wrap button:not(:disabled):hover {
        background-color: #138349 !important;
        border-color: #138349 !important;
        color: #ffffff !important;
    }

    div[data-testid="stCaptionContainer"] p {
        color: #667085 !important;
        font-weight: 500;
    }

    @media (max-width: 720px) {
        section.main > div.block-container {
            border-radius: 0;
            margin-top: 0;
            padding: 1rem 0.9rem 1rem;
        }

        section.main div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }

        section.main div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db()

schema_error = None
if using_cloud():
    try:
        list_analyses(limit=1)
    except Exception as exc:
        schema_error = str(exc)

if schema_error:
    st.error(schema_error)
    st.link_button("Open the SQL Editor", sql_editor_url())
    st.caption("Paste this SQL, click Run, then refresh this page.")
    st.code((ROOT / "supabase_schema.sql").read_text(encoding="utf-8"), language="sql")

try:
    abs_model, rgb_model, metadata = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

conc_default = 1.5

st.markdown('<div class="field-label">Enter Concentration</div>', unsafe_allow_html=True)
concentration = st.number_input(
    "Concentration",
    min_value=0.0,
    value=conc_default,
    step=0.1,
    format="%.2f",
    label_visibility="collapsed",
    key="concentration",
    help="Use the same concentration units as the training sheet (about 0.2 to 10).",
)
st.caption("Concentration in the training sheet runs from 0.2 to 10.")

result = predict_sample(abs_model, rgb_model, concentration)
pred_abs = result["predicted_absorbance"]
pred_rgb = result["predicted_rgb"]
pred_hex = result["predicted_hex"]

st.markdown(
    '<div class="live-badge"><span class="live-dot"></span>LIVE CAMERA</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="camera-note">Line the coloured solution up with the centre mark, then take a photo. '
    "Colour is read from inside the box in the middle of the frame.</div>",
    unsafe_allow_html=True,
)
with st.container(key="camera_panel"):
    photo = st.camera_input("Capture and analyse colour", key="beaker_camera")

analyzed = None
photo_bytes = None
if photo is not None:
    photo_bytes = photo.getvalue()
    try:
        analyzed = analyze_beaker_colour(photo_bytes)
    except ValueError as exc:
        st.error(str(exc))

if analyzed and analyzed["low_colour_signal"]:
    st.warning(
        "The photo looks mostly blank or grey. Centre the coloured solution in the frame and take another photo."
    )

predicted_detail = f"Abs {pred_abs:.4f}"
analyzed_hex = analyzed["analyzed_hex"] if analyzed else None
left_card, right_card = st.columns(2)
with left_card:
    st.markdown(
        colour_card("Predicted<br>Colour", pred_hex, predicted_detail),
        unsafe_allow_html=True,
    )
with right_card:
    st.markdown(
        colour_card(
            "Analyzed<br>Colour",
            analyzed_hex,
            "",
            empty_message="Take a photo",
        ),
        unsafe_allow_html=True,
    )

if analyzed:
    distance = float(
        np.linalg.norm(np.asarray(pred_rgb, dtype=float) - np.asarray(analyzed["analyzed_rgb"], dtype=float))
    )
    st.caption(
        f"Predicted absorbance {pred_abs:.4f}. "
        f"Camera colour {analyzed_hex} is {distance:.1f} RGB units from the predicted colour {pred_hex}."
    )
else:
    st.caption(
        f"Predicted absorbance {pred_abs:.4f}. Capture a photo to compare it with the solution in the beaker."
    )

signature = None
if analyzed and photo_bytes is not None:
    signature = (
        round(float(concentration), 4),
        pred_hex,
        analyzed_hex,
        sha256(photo_bytes).hexdigest(),
    )

already_saved = signature is not None and st.session_state.get("saved_signature") == signature
if already_saved:
    destination = "Supabase" if using_cloud() else "this computer"
    st.success(f"Stored analysis #{st.session_state.get('last_saved_id')} in {destination}.")

_spacer, store_col = st.columns([1.35, 0.85])
with store_col:
    with st.container(key="store_wrap"):
        store_clicked = st.button(
            "Store Analysis to Database",
            key="store_analysis",
            type="primary",
            width="stretch",
            disabled=analyzed is None or photo_bytes is None or already_saved,
            help="Saves concentration, predicted absorbance and colour, and the colour measured from the photo.",
        )

if store_clicked:
    if analyzed is None or photo_bytes is None:
        st.warning("Take a photo of the beaker before storing the analysis.")
    else:
        try:
            record_id = save_analysis(
                model_variant=MODEL["label"],
                volume_ml=None,
                concentration=float(concentration),
                predicted_absorbance=pred_abs,
                predicted_rgb=pred_rgb,
                predicted_hex=pred_hex,
                analyzed_rgb=analyzed["analyzed_rgb"],
                analyzed_hex=analyzed_hex,
                image_bytes=photo_bytes,
            )
        except Exception as exc:
            st.error(f"Could not store this analysis. {exc}")
        else:
            st.session_state["saved_signature"] = signature
            st.session_state["last_saved_id"] = record_id
            st.rerun()

abs_algo = metadata.get("abs_algorithm", "unknown") if isinstance(metadata, dict) else "unknown"
color_algo = metadata.get("color_algorithm", "unknown") if isinstance(metadata, dict) else "unknown"
st.caption(f"Active algorithms: absorbance = {abs_algo}, colour = {color_algo}.")

if using_cloud():
    st.caption("New analyses are stored in Supabase, so they remain available after the app is deployed.")
else:
    st.caption("New analyses are stored on this computer only.")
    with st.expander("Connect a cloud database"):
        st.markdown(
            """
            A deployed website or phone app cannot read the database file on this PC.
            Supabase keeps every confirmed analysis online, where you can export it later.

            1. Create a free project at [supabase.com](https://supabase.com).
            2. Open **SQL Editor**, paste the contents of `supabase_schema.sql`, and run it.
            3. In **Project Settings → API**, copy the project URL and the **anon public** key.
            4. Create a `.env` file in this project folder:

            ```
            SUPABASE_URL=https://YOUR_PROJECT.supabase.co
            SUPABASE_KEY=your_anon_public_key
            ```

            Refresh this page after saving `.env`. On Streamlit Community Cloud, put those same two names in the app Secrets. A future Android app can use this same URL and anon key. Do not put the service_role key in the app.
            """
        )

if schema_error:
    records = []
else:
    try:
        records = list_analyses()
    except Exception as exc:
        records = []
        st.error(str(exc))
with st.expander("Saved analyses", expanded=already_saved):
    if not records:
        st.caption("Nothing stored yet. Capture a photo, check the two colours, then store the analysis.")
    else:
        table = pd.DataFrame(records).drop(columns=["volume_ml", "model_variant"], errors="ignore").rename(
            columns={
                "id": "ID",
                "created_at": "Saved",
                "concentration": "Concentration",
                "predicted_absorbance": "Absorbance",
                "predicted_hex": "Predicted",
                "analyzed_hex": "Analyzed",
                "rgb_distance": "Colour difference",
            }
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Absorbance": st.column_config.NumberColumn(format="%.4f"),
                "Concentration": st.column_config.NumberColumn(format="%.2f"),
                "Colour difference": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        if using_cloud():
            st.caption("Rows and photos are stored in the Supabase table analyses.")
        else:
            st.caption("Photos are saved under data/captures. The table is stored in data/analyses.db.")
