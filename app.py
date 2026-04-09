from pathlib import Path
import sys

import joblib
import matplotlib
import numpy as np
import streamlit as st
from xgboost import plot_importance

matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP_ICON = "\U0001F30D"
MODEL_FILENAME = "xgboost_aqi_model.pkl"
DATA_FILENAME = "city_day.csv"
APP_TITLE = "Air Quality Prediction"


def resource_path(filename: str) -> Path:
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_dir / filename


@st.cache_resource
def load_model(model_file: Path):
    return joblib.load(model_file)


st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon=APP_ICON,
    layout="wide",
)

model_path = resource_path(MODEL_FILENAME)
data_path = resource_path(DATA_FILENAME)
model = None

try:
    model = load_model(model_path)
except FileNotFoundError:
    pass

st.sidebar.title("About Project")
st.sidebar.info(
    f"""
{APP_TITLE}

Model Used:
XGBoost Regressor

Input Features:
PM2.5
PM10
NO2
SO2
CO
O3

Output:
Predicted Air Quality Index (AQI)
"""
)

if model is None:
    st.warning(f"Model file `{MODEL_FILENAME}` was not found.")
    if data_path.exists():
        st.info("Run `python train_model.py` to create the model file, then refresh the app.")
    else:
        st.info(
            f"Add `{DATA_FILENAME}` to `{data_path.parent}`, run `python train_model.py`, "
            "then refresh the app."
        )

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}
[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}
.block-container{
padding-top:2rem;
}
.title{
text-align:center;
font-size:45px;
font-weight:bold;
color:#f8fafc;
}
.subtitle{
text-align:center;
font-size:20px;
margin-bottom:30px;
color:#dbeafe;
}
.section-heading{
color:#f8fafc;
font-size:1.5rem;
font-weight:700;
margin:1.4rem 0 0.9rem;
text-shadow:0 2px 8px rgba(0,0,0,0.25);
}
.aqi-card{
background:rgba(255,255,255,0.12);
border:1px solid rgba(255,255,255,0.2);
border-radius:18px;
overflow:hidden;
backdrop-filter:blur(10px);
box-shadow:0 16px 40px rgba(15,23,42,0.18);
margin-bottom:1.3rem;
}
.aqi-table{
width:100%;
border-collapse:collapse;
color:#f8fafc;
}
.aqi-table th,
.aqi-table td{
padding:0.9rem 1rem;
text-align:left;
border-bottom:1px solid rgba(255,255,255,0.14);
}
.aqi-table th{
background:rgba(15,23,42,0.45);
font-weight:700;
letter-spacing:0.02em;
}
.aqi-table tr:last-child td{
border-bottom:none;
}
.aqi-range{
color:#e2e8f0;
font-weight:600;
}
.category-good{
color:#86efac;
font-weight:700;
}
.category-moderate{
color:#fde047;
font-weight:700;
}
.category-poor{
color:#fdba74;
font-weight:700;
}
.category-hazardous{
color:#fca5a5;
font-weight:700;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInput"] label p{
color:#f8fafc !important;
font-weight:700 !important;
}
div[data-testid="stNumberInput"] [data-baseweb="input"]{
background:transparent !important;
border:none !important;
box-shadow:none !important;
}
div[data-testid="stNumberInput"] [data-baseweb="input"] > div{
background:rgba(15,23,42,0.82) !important;
border:1px solid rgba(191,219,254,0.3) !important;
border-radius:14px !important;
}
div[data-testid="stNumberInput"] input{
color:#f8fafc !important;
-webkit-text-fill-color:#f8fafc !important;
background:transparent !important;
font-weight:600 !important;
}
div[data-testid="stNumberInput"] input::placeholder{
color:#cbd5e1 !important;
opacity:1 !important;
}
div[data-testid="stNumberInput"] button{
color:#f8fafc !important;
background:rgba(30,41,59,0.95) !important;
border-left:1px solid rgba(191,219,254,0.24) !important;
opacity:1 !important;
}
div[data-testid="stNumberInput"] button:hover{
background:rgba(51,65,85,0.95) !important;
}
div[data-testid="stNumberInput"] button:disabled,
div[data-testid="stNumberInput"] button[disabled]{
color:#e2e8f0 !important;
background:rgba(30,41,59,0.95) !important;
opacity:1 !important;
}
div[data-testid="stNumberInput"] button svg,
div[data-testid="stNumberInput"] button span{
color:inherit !important;
fill:currentColor !important;
stroke:currentColor !important;
opacity:1 !important;
}
div[data-testid="stNumberInput"] button:disabled svg,
div[data-testid="stNumberInput"] button[disabled] svg,
div[data-testid="stNumberInput"] button:disabled span,
div[data-testid="stNumberInput"] button[disabled] span{
color:#e2e8f0 !important;
fill:currentColor !important;
stroke:currentColor !important;
opacity:1 !important;
}
div[data-testid="stButton"] > button{
background:linear-gradient(135deg,#f8fafc,#bfdbfe) !important;
color:#0f172a !important;
border:none !important;
border-radius:12px !important;
font-weight:700 !important;
padding:0.7rem 1.5rem !important;
box-shadow:0 14px 28px rgba(15,23,42,0.22);
}
div[data-testid="stButton"] > button:hover{
background:linear-gradient(135deg,#ffffff,#dbeafe) !important;
color:#020617 !important;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)
st.markdown(
    f'<p class="title">{APP_ICON} {APP_TITLE}</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Predict AQI using XGBoost Machine Learning Model</p>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">AQI Categories</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="aqi-card">
  <table class="aqi-table">
    <thead>
      <tr>
        <th>AQI Range</th>
        <th>Category</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="aqi-range">0-50</td>
        <td class="category-good">Good 🟢</td>
      </tr>
      <tr>
        <td class="aqi-range">51-100</td>
        <td class="category-moderate">Moderate 🟡</td>
      </tr>
      <tr>
        <td class="aqi-range">101-200</td>
        <td class="category-poor">Poor 🟠</td>
      </tr>
      <tr>
        <td class="aqi-range">201+</td>
        <td class="category-hazardous">Hazardous 🔴</td>
      </tr>
    </tbody>
  </table>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Enter Pollution Levels</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0)
    pm10 = st.number_input("PM10", min_value=0.0)
    no2 = st.number_input("NO2", min_value=0.0)

with col2:
    so2 = st.number_input("SO2", min_value=0.0)
    co = st.number_input("CO", min_value=0.0)
    o3 = st.number_input("O3", min_value=0.0)

if st.button("Predict AQI"):
    if model is None:
        st.error(
            "Prediction is unavailable because the trained model file is missing. "
            "Please generate or add `xgboost_aqi_model.pkl` first."
        )
        st.stop()

    data = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(data)[0]

    st.subheader(f"Predicted AQI: {prediction:.2f}")

    if prediction <= 50:
        st.success("Air Quality: Good \U0001F7E2")
    elif prediction <= 100:
        st.info("Air Quality: Moderate \U0001F7E1")
    elif prediction <= 200:
        st.warning("Air Quality: Poor \U0001F7E0")
    else:
        st.error("Air Quality: Hazardous \U0001F534")

st.markdown("### Model Feature Importance")
if model is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_importance(model, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Feature importance will appear here after the trained model is available.")

st.markdown("---")
st.markdown(
    "Machine Learning Project | Air Quality Prediction using XGBoost"
)
