import streamlit as st
import requests

st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌾 Smart Crop Recommendation System")

# --- User Inputs ---
with st.form("crop_form"):
    st.subheader("🧪 Enter Soil and Location Info")
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43)
    pH = st.number_input("Soil pH", min_value=3.0, max_value=9.0, value=6.5)
    city = st.text_input("City (for weather)", value="")

    submit = st.form_submit_button("🔍 Predict Crop")

# --- Handle Submit ---
if submit:
    st.info("⏳ Sending data to prediction engine...")

    # Make request to Flask backend
    try:
        res = requests.post(
            "http://127.0.0.1:5000/predict",  # or your deployed backend URL
            json={"N": N, "P": P, "K": K, "pH": pH, "city": city}
        )
        if res.status_code != 200:
            st.error("❌ Backend error. Please check server.")
        else:
            result = res.json()
            st.success(f"✅ Recommended Crop: {result['predicted_crop'].title()}")
            st.metric("Confidence", f"{result['confidence']}%")

            st.subheader("📊 Top 3 Crop Suggestions")
            for item in result['top_3_predictions']:
                st.write(f"- {item['crop'].title()}: {item['confidence']}%")

            st.subheader("🌦️ Weather Used (5-day avg)")
            st.write(f"Temperature: {result['weather_used']['avg_temperature']} °C")
            st.write(f"Humidity: {result['weather_used']['avg_humidity']} %")
            st.write(f"Rainfall: {result['weather_used']['total_rainfall_mm']} mm")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
