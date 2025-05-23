# Qetu shenohet main python code
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:5173"}})

# Load models at startup
with open('AI/model1.pkl', 'rb') as f:
    model1 = pickle.load(f)

with open('AI/model2.pkl', 'rb') as f:
    model2 = pickle.load(f)

with open('AI/model3.pkl', 'rb') as f:
    model3 = pickle.load(f)

@app.route('/predict_traffic_level', methods=['POST'])
def predict_traffic_level():
    data = request.json
    print(data)
    label_encoders = {}
    for col in ['start_city', 'end_city', 'day_of_week']:
        le = LabelEncoder()
        data[col] = le.fit_transform([data[col]])[0]
        label_encoders[col] = le
    
    features = np.array([[data['start_city'], data['end_city'], data['day_of_week'], data['hour'], data['minute']]])
    prediction = model1.predict(features)

    return jsonify({'traffic_level': int(prediction[0])})

@app.route('/predict_traffic_wait', methods=['POST'])
def predict_traffic_wait():
    data = request.json
    label_encoders = {}
    for col in ['start_city', 'end_city', 'day_of_week']:
        le = LabelEncoder()
        data[col] = le.fit_transform([data[col]])[0]
        label_encoders[col] = le

    features = np.array([[data['start_city'], data['end_city'], data['day_of_week'], data['hour'], data['minute'], data['traffic_level']]])
    prediction = model2.predict(features)
    return jsonify({'traffic_wait_minutes': int(prediction[0])})

@app.route('/predict_passenger_count', methods=['POST'])
def predict_passenger_count():
    data = request.json
    label_encoders = {}
    for col in ['start_city', 'end_city', 'day_of_week']:
        le = LabelEncoder()
        data[col] = le.fit_transform([data[col]])[0]
        label_encoders[col] = le

    features = np.array([[data['start_city'], data['end_city'], data['day_of_week']]])
    prediction = model3.predict(features)
    return jsonify({'passenger_count': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
