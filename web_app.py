from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Chemistry Predictor",
    page_icon="🧪",
    layout="centered",
)

ARTIFACT_DIR = Path("artifacts")
ABS_MODEL_PATH = ARTIFACT_DIR / "absorbance_model.joblib"
RGB_MODEL_PATH = ARTIFACT_DIR / "colour_model_rgb.joblib"
META_PATH = ARTIFACT_DIR / "model_metadata.joblib"


@st.cache_resource
def load_models():
    if not ABS_MODEL_PATH.exists() or not RGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Run the notebook training cells first to create artifacts."
        )

    abs_model = joblib.load(ABS_MODEL_PATH)
    rgb_model = joblib.load(RGB_MODEL_PATH)
    metadata = {}
    if META_PATH.exists():
        metadata = joblib.load(META_PATH)

    return abs_model, rgb_model, metadata


def rgb_to_hex(rgb_values):
    r, g, b = (int(np.clip(round(value), 0, 255)) for value in rgb_values)
    return f"#{r:02X}{g:02X}{b:02X}"


def predict_sample(abs_model, rgb_model, volume_ml, concentration):
    sample = pd.DataFrame({"volume_ml": [volume_ml], "conc": [concentration]})
    predicted_abs = float(abs_model.predict(sample)[0])

    sample_color = sample.assign(abs_pred=[predicted_abs])
    predicted_rgb = np.rint(np.clip(rgb_model.predict(sample_color)[0], 0, 255)).astype(int)
    predicted_hex = rgb_to_hex(predicted_rgb)

    return {
        "predicted_absorbance": predicted_abs,
        "predicted_rgb": tuple(int(v) for v in predicted_rgb),
        "predicted_hex": predicted_hex,
    }


def get_volume_training_range(metadata):
    if not isinstance(metadata, dict):
        return None, None, []

    labels = metadata.get("volume_labels", [])
    parsed_values = []
    for label in labels:
        match = re.search(r"(\d+(?:\.\d+)?)", str(label))
        if match:
            parsed_values.append(float(match.group(1)))

    if not parsed_values:
        return None, None, []

    unique_values = sorted(set(parsed_values))
    return min(unique_values), max(unique_values), unique_values


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Serif+Display:ital@0;1&display=swap');

    .stApp {
        background:
            radial-gradient(1200px 500px at -10% -10%, rgba(46, 196, 182, 0.22), transparent 60%),
            radial-gradient(1000px 400px at 110% 10%, rgba(255, 159, 67, 0.2), transparent 55%),
            linear-gradient(135deg, #f9f7f3 0%, #fffdf8 100%);
    }

    .main-card {
        border: 1px solid #d7d3c8;
        border-radius: 20px;
        padding: 1.2rem 1.2rem 1.4rem;
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(6px);
        box-shadow: 0 12px 35px rgba(39, 45, 52, 0.08);
        animation: floatIn 0.5s ease-out;
    }

    .title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.1rem;
        color: #0E4D64;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        font-family: 'Space Grotesk', sans-serif;
        color: #2F4858;
        margin-bottom: 1.1rem;
        line-height: 1.4;
    }

    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        color: #0B3C4A;
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        color: #0f2f35;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .swatch {
        width: 100%;
        height: 90px;
        border-radius: 14px;
        border: 1px solid rgba(18, 52, 59, 0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.6px;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
    }

    @keyframes floatIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }

    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stNumberInput"] label p {
        color: #0B3C4A;
        font-weight: 700;
        font-size: 1.02rem;
        opacity: 1;
    }

    div[data-testid="stCaptionContainer"] p {
        color: #174E63 !important;
        font-weight: 700;
        font-size: 0.95rem;
        opacity: 1;
    }

    .stButton > button {
        border-radius: 10px;
        background: linear-gradient(90deg, #128277, #15a48f);
        color: #ffffff;
        border: none;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #0f6d64, #118a78);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title">Chemistry Color Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict absorbance and resulting solution color from volume and concentration.</div>',
    unsafe_allow_html=True,
)

try:
    abs_model, rgb_model, metadata = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

volume_min, volume_max, volume_values = get_volume_training_range(metadata)

if volume_min is not None and volume_max is not None:
    volume_default = float(np.clip(3.0, volume_min, volume_max))
else:
    volume_default = 3.0

conc_default = 1.5

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        volume_ml = st.number_input(
            "Volume (mL)",
            min_value=0.0,
            value=volume_default,
            step=0.1,
            format="%.2f",
            help="Try to stay within the trained volume range for more reliable predictions.",
            label_visibility="visible",
        )
        if volume_min is not None and volume_max is not None:
            st.caption(f"Recommended range: {volume_min:.2f} to {volume_max:.2f} mL")
    with c2:
        concentration = st.number_input(
            "Concentration",
            min_value=0.0,
            value=conc_default,
            step=0.1,
            format="%.2f",
            label_visibility="visible",
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    result = predict_sample(abs_model, rgb_model, volume_ml, concentration)
else:
    result = predict_sample(abs_model, rgb_model, volume_default, conc_default)

if volume_min is not None and volume_max is not None and not (volume_min <= volume_ml <= volume_max):
    st.warning(
        f"Volume {volume_ml:.2f} mL is outside the trained range ({volume_min:.2f}-{volume_max:.2f} mL). "
        "Predictions may be less reliable."
    )

pred_abs = result["predicted_absorbance"]
pred_rgb = result["predicted_rgb"]
pred_hex = result["predicted_hex"]

m1, m2 = st.columns(2)
with m1:
    st.markdown('<div class="metric-label">Predicted absorbance</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{pred_abs:.4f}</div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-label">Predicted RGB</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric-value">({pred_rgb[0]}, {pred_rgb[1]}, {pred_rgb[2]})</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="swatch" style="background: {pred_hex};">{pred_hex}</div>',
    unsafe_allow_html=True,
)

volume_labels = metadata.get("volume_labels", []) if isinstance(metadata, dict) else []
if volume_labels:
    st.caption(f"Known volumes in training data: {', '.join(volume_labels)}")

st.markdown('</div>', unsafe_allow_html=True)
