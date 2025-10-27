import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("🌾 Crop Yield Prediction App")

# --- Load trained model and required files ---
try:
    model = pickle.load(open('model_compressed.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    st.success("✅ Model, Scaler, and Features loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model files: {e}")
    st.stop()

# --- File upload ---
uploaded_file = st.file_uploader("📂 Upload CSV file for prediction (same format as training data)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # --- Encode categorical columns (convert text → numeric) ---
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # --- Check and align feature columns ---
    missing_cols = [c for c in features if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in features]

    if missing_cols:
        st.error(f"⚠️ Missing columns in uploaded file: {missing_cols}")
        st.stop()

    if extra_cols:
        st.warning(f"⚠️ Ignoring extra columns: {extra_cols}")
        df = df[features]

    df = df[features]  # Ensure same order as training

    # --- Scale and predict ---
    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)
    df["Predicted_Value"] = preds

    st.success("✅ Prediction Completed Successfully!")
    st.write("### 📊 Predicted Results")
    st.dataframe(df)

    # --- Download option ---
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="predicted_yield.csv",
        mime="text/csv",
    )

else:
    st.info("👉 Please upload your CSV file (same features as training).")
