import os
import joblib
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import numpy as np

# Load env variables
load_dotenv()
OPEN_WEATHER_KEY = os.getenv('OPEN_WEATHER_API_KEY')

app = Flask(__name__)

# Load the trained model
model = joblib.load('rf_crop_model.pkl')


@app.route('/predict', methods=['POST'])
def predict_crop():
    # Inputs from frontend (real user input)
    data = request.json
    city = data['city']  # default for testing
    n = data['N']
    p = data['P']
    k = data['K']
    ph = data['pH']

    # Fetch 5-day weather forecast
    url = f'https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPEN_WEATHER_KEY}&units=metric'
    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({'error': 'Weather API failed'}), 500

    forecast = response.json()

    # Aggregate weather data
    temps, humidities, rainfall = [], [], 0.0
    for entry in forecast['list']:
        temps.append(entry['main']['temp'])
        humidities.append(entry['main']['humidity'])
        rainfall += entry.get('rain', {}).get('3h', 0)

    avg_temp = round(sum(temps) / len(temps), 2)
    avg_humidity = round(sum(humidities) / len(humidities), 2)
    total_rainfall = round(rainfall, 2)

    # Create input vector
    features = [[n, p, k, avg_temp, avg_humidity, ph, total_rainfall]]

    # Predict crop and confidence
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = round(max(probs) * 100, 2)

    # Top 3 crops
    classes = model.classes_
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [{"crop": classes[i], "confidence": round(probs[i] * 100, 2)} for i in top3_idx]

    return jsonify({
        "predicted_crop": prediction,
        "confidence": confidence,
        "top_3_predictions": top3,
        "weather_used": {
            "avg_temperature": avg_temp,
            "avg_humidity": avg_humidity,
            "total_rainfall_mm": total_rainfall
        }
    })


if __name__ == '__main__':
    app.run(debug=True)
