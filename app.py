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
.title{
text-align:center;
font-size:45px;
font-weight:bold;
}
.subtitle{
text-align:center;
font-size:20px;
margin-bottom:30px;
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

st.markdown("### AQI Categories")
st.table(
    {
        "AQI Range": ["0-50", "51-100", "101-200", "201+"],
        "Category": [
            "Good \U0001F7E2",
            "Moderate \U0001F7E1",
            "Poor \U0001F7E0",
            "Hazardous \U0001F534",
        ],
    }
)

st.markdown("### Enter Pollution Levels")
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
