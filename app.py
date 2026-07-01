import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoders
model = joblib.load("fertilizer_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

st.set_page_config(
    page_title="AI Fertilizer Recommendation Agent",
    page_icon="🌱",
    layout="wide"
)

st.sidebar.title("🌱 AI Agriculture")
st.sidebar.write("Fertilizer Recommendation Agent")

st.title("🌱 AI Fertilizer Recommendation Agent")
st.write("Enter the crop and soil details below.")
st.subheader("Enter Farm Details")

temperature = st.number_input("Temperature", 0, 100, 25)

humidity = st.number_input("Humidity", 0, 100, 50)

moisture = st.number_input("Moisture", 0, 100, 40)

soil = st.selectbox(
    "Soil Type",
    soil_encoder.classes_
)

crop = st.selectbox(
    "Crop Type",
    crop_encoder.classes_
)

nitrogen = st.number_input("Nitrogen", 0, 150, 20)

potassium = st.number_input("Potassium", 0, 150, 20)

phosphorous = st.number_input("Phosphorous", 0, 150, 20)

if st.button("🌱 Recommend Fertilizer"):

    soil_value = soil_encoder.transform([soil])[0]
    crop_value = crop_encoder.transform([crop])[0]

    sample = pd.DataFrame([{

        "Temparature": temperature,
        "Humidity": humidity,
        "Moisture": moisture,
        "Soil Type": soil_value,
        "Crop Type": crop_value,
        "Nitrogen": nitrogen,
        "Potassium": potassium,
        "Phosphorous": phosphorous

    }])

    prediction = model.predict(sample)

    fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

    st.success(f"Recommended Fertilizer: {fertilizer}")