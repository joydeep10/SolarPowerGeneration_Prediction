import json
import io
import pandas as pd
import numpy as np
import streamlit as st
import joblib

st.set_page_config(page_title="Solar Power Predictor", page_icon="⚡", layout="centered")

@st.cache_resource
def load_artifacts():
    pipe = joblib.load("xgb_pipeline.joblib")       # <- the saved sklearn Pipeline
    with open("feature_order.json") as f:
        num_cols = json.load(f)["num_cols"]
    return pipe, num_cols

pipe, NUM_COLS = load_artifacts()

st.title("⚡ Solar Power Generation Predictor")
st.caption("XGBoost model in a scikit-learn Pipeline (scaler + model).")

# ----- Single prediction -----
st.subheader("Single Prediction")

defaults = {
    "distance_to_solar_noon": 0.48,
    "temperature": 59.0,
    "wind_direction": 27.0,
    "wind_speed": 7.5,
    "sky_cover": 1.0,
    "visibility": 10.0,
    "humidity": 40.0,
    "average_wind_speed_period": 3.0,
    "average_pressure_period": 30.00
}

cols = st.columns(3)
row = {}
for i, col in enumerate(NUM_COLS):
    with cols[i % 3]:
        row[col] = st.number_input(col, value=float(defaults.get(col, 0.0)), step=0.1)

if st.button("Predict (Joules)", type="primary"):
    X = pd.DataFrame([row], columns=NUM_COLS)
    pred = pipe.predict(X)[0]
    st.success(f"Prediction: {pred:,.0f} Joules")

# ----- Batch prediction -----
st.subheader("Batch (CSV)")
st.write("CSV must contain these columns (any order):")
st.code(", ".join(NUM_COLS))

# download template
example = pd.DataFrame([{c: defaults.get(c, 0.0) for c in NUM_COLS}])
buf = io.BytesIO(); example.to_csv(buf, index=False)
st.download_button("Download example_input.csv", buf.getvalue(), file_name="example_input.csv", mime="text/csv")

file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    try:
        df = pd.read_csv(file)
        # reindex to expected columns (this also orders columns)
        X = df.reindex(columns=NUM_COLS)
        preds = pipe.predict(X)
        out = df.copy()
        out["prediction"] = preds
        st.dataframe(out.head(20), use_container_width=True)
        out_buf = io.BytesIO(); out.to_csv(out_buf, index=False)
        st.download_button("Download predictions.csv", out_buf.getvalue(), file_name="predictions.csv", mime="text/csv")
        st.success(f"Predicted {len(out)} rows.")
    except Exception as e:
        st.error(f"Error: {e}")