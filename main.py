import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load trained Random Forest model
# -----------------------------
model = joblib.load("crop_rf_model.pkl")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🌾 Crop Recommendation System")
st.markdown(
    "Predict the **best crop** based on soil nutrients and climatic conditions "
    "using a **Random Forest Machine Learning model**."
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("🔍 Enter Soil & Climate Parameters")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 90)
    P = st.number_input("Phosphorus (P)", 0, 200, 42)
    K = st.number_input("Potassium (K)", 0, 200, 43)
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)

with col2:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🌱 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)

    st.success(f"✅ Recommended Crop: **{prediction[0].capitalize()}**")

st.divider()

# -----------------------------
# Feature Importance Section
# -----------------------------
st.subheader("📊 Feature Importance (Random Forest)")

feature_names = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Temperature", "Humidity", "Soil pH", "Rainfall"
]

importances = model.feature_importances_

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Show table
st.dataframe(fi_df, use_container_width=True)

# Plot bar chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(fi_df["Feature"], fi_df["Importance"])
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance Analysis")
ax.invert_yaxis()

st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.markdown(
    """
    👨‍🌾 **Crop Recommendation System**  
    🔧 Model: Random Forest Classifier  
    📊 Features: Soil nutrients & climate conditions  
    """
)
