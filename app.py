import streamlit as st
import pickle
import numpy as np
import sklearn

st.set_page_config(page_title="Spotify Popularity Predictor", page_icon="🎵", layout="centered")

st.title("🎵 Spotify Popularity Predictor")
st.write("Enter song features to predict popularity score.")

# Show the scikit-learn version so you can track compatibility
st.caption(f"scikit-learn version: {sklearn.__version__}")

# Load the trained bundle
@st.cache_resource
def load_bundle():
    with open("spotify_rf_bundle.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]

# ---- Safety monkeypatch: ensure each tree estimator has monotonic_cst attribute ----
# This fixes AttributeError when model was trained under a different sklearn version.
try:
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            if not hasattr(est, "monotonic_cst"):
                est.monotonic_cst = None
except Exception as e:
    # Show a small warning in the app (keeps the user informed)
    st.warning("Warning: failed to patch model estimators (non-fatal). See logs.")
    st.text(str(e))
# ------------------------------------------------------------------------------------

# Input form for features
inputs = {}
st.header("Song Features Input")

for feature in features:
    inputs[feature] = st.number_input(
        f"{feature}",
        value=0.0,
        step=0.1,
        format="%.2f"
    )

# Convert to array and scale
input_values = np.array([inputs[f] for f in features]).reshape(1, -1)

# wrap prediction in try/except so streamlit shows a helpful error message
if st.button("Predict Popularity"):
    try:
        scaled_input = scaler.transform(input_values)
        prediction = model.predict(scaled_input)[0]
        st.success(f"Predicted Spotify Popularity Score: {prediction:.2f}")
    except Exception as e:
        st.error("Error while predicting. See details below.")
        st.exception(e)
        # Optional: show a hint so you know this is likely a sklearn version mismatch
        st.info("Hint: If this is an AttributeError mentioning 'monotonic_cst', retrain the model "
                "in this environment or install the sklearn version used at training.")

