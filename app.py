import io
 
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
 
st.set_page_config(page_title="Power Plant Output Predictor", layout="wide")
 
 
@st.cache_resource
def load_artifact():
    return joblib.load("power_plant_model.joblib")
 
 
art = load_artifact()
model = art["model"]
FEATURES = art["features"]          # ['AT', 'V', 'AP', 'RH']
RANGES = art["ranges"]              # training min/max per feature
 
LABELS = {
    "AT": "Ambient temperature (°C)",
    "V": "Exhaust vacuum (cm Hg)",
    "AP": "Ambient pressure (millibar)",
    "RH": "Relative humidity (%)",
}
 
st.title("Combined-Cycle Power Plant: Electrical Output Predictor")
st.caption(
    "Predicts the plant's net hourly electrical output (MW) from four ambient "
    "measurements, using a tuned gradient-boosting regression model."
)
 
with st.sidebar:
    st.header("About the model")
    st.markdown(
        """
- **Model:** Histogram Gradient Boosting (tuned by grid search)
- **Data:** UCI Combined Cycle Power Plant, 9,527 unique hourly readings
- **Held-out test result:**
  - RMSE **3.26 MW** (95% CI 2.85 to 3.72)
  - MAE **2.20 MW**
  - R² **0.964**
- Predictions are only reliable inside the ranges the model was trained on.
"""
    )
    sample = pd.DataFrame(
        {
            "AT": [14.96, 25.18, 5.11, 20.86, 10.82],
            "V": [41.76, 62.96, 39.40, 57.32, 37.50],
            "AP": [1024.07, 1020.04, 1012.16, 1010.24, 1009.23],
            "RH": [73.17, 59.08, 92.14, 76.64, 96.62],
            "PE": [463.26, 444.37, 488.56, 446.48, 473.90],
        }
    )
    st.download_button(
        "Download sample CSV", sample.to_csv(index=False), "sample.csv", "text/csv"
    )
 
tab1, tab2 = st.tabs(["Single prediction", "Upload CSV"])
 
# ---------------------------------------------------------------- single
with tab1:
    st.subheader("Enter the ambient conditions")
    cols = st.columns(4)
    values = {}
    for col, f in zip(cols, FEATURES):
        lo, hi = RANGES[f]
        values[f] = col.slider(
            LABELS[f], float(lo), float(hi), float(round((lo + hi) / 2, 2)), step=0.01
        )
    x = pd.DataFrame([values])[FEATURES]
    pred = float(model.predict(x)[0])
    c1, c2 = st.columns(2)
    c1.metric("Predicted net electrical output", f"{pred:.1f} MW")
    c2.metric("Typical error (test RMSE)", "± 3.3 MW")
 
# ---------------------------------------------------------------- csv
with tab2:
    st.subheader("Predict many rows from a CSV")
    st.write(
        f"Required columns: `{', '.join(FEATURES)}`. "
        "An optional `PE` column (actual output) adds accuracy metrics and a plot."
    )
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is None:
        st.info("Upload a CSV, or download the sample from the sidebar.")
        st.stop()
 
    try:
        df = pd.read_csv(file)
    except Exception as e:  # unreadable file
        st.error(f"Could not read the file: {e}")
        st.stop()
 
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
 
    for c in FEATURES + (["PE"] if "PE" in df.columns else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=FEATURES + (["PE"] if "PE" in df.columns else [])).reset_index(drop=True)
    if len(df) < before:
        st.warning(f"Dropped {before - len(df)} rows with missing or non-numeric values.")
    if df.empty:
        st.error("No usable rows left.")
        st.stop()
 
    out_of_range = np.zeros(len(df), dtype=bool)
    for f in FEATURES:
        lo, hi = RANGES[f]
        out_of_range |= (df[f] < lo).values | (df[f] > hi).values
    if out_of_range.any():
        st.warning(
            f"{int(out_of_range.sum())} rows fall outside the training ranges; "
            "their predictions are less reliable."
        )
 
    df["predicted_PE"] = model.predict(df[FEATURES])
    df["outside_training_range"] = out_of_range
 
    st.write(f"Predicted **{len(df)}** rows.")
    st.dataframe(df.head(200), use_container_width=True)
 
    if "PE" in df.columns:
        err = df["PE"] - df["predicted_PE"]
        rmse = float(np.sqrt((err**2).mean()))
        mae = float(err.abs().mean())
        ss_res = float((err**2).sum())
        ss_tot = float(((df["PE"] - df["PE"].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE (MW)", f"{rmse:.2f}")
        m2.metric("MAE (MW)", f"{mae:.2f}")
        m3.metric("R²", f"{r2:.3f}" if len(df) > 1 else "n/a")
 
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(df["PE"], df["predicted_PE"], s=8, alpha=0.5)
        lo_, hi_ = df["PE"].min(), df["PE"].max()
        ax[0].plot([lo_, hi_], [lo_, hi_], "r--")
        ax[0].set_xlabel("Actual PE (MW)")
        ax[0].set_ylabel("Predicted PE (MW)")
        ax[0].set_title("Predicted vs actual")
        ax[1].hist(err, bins=30)
        ax[1].set_xlabel("Error (actual - predicted, MW)")
        ax[1].set_title("Error distribution")
        plt.tight_layout()
        st.pyplot(fig)
 
    st.download_button(
        "Download predictions CSV", df.to_csv(index=False), "predictions.csv", "text/csv"
    )
 