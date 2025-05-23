import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# from xgboost import XGBRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import os

# Get absolute path to CSV
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "../bus_trips.csv")

# Load data
df = pd.read_csv(csv_path)
df.dropna(inplace=True)
# print(df)

df['trip_start_time'] = pd.to_datetime(df['trip_start_time'])

# # Feature engineering
df['hour'] = df['trip_start_time'].dt.hour
df['minute'] = df['trip_start_time'].dt.minute

data = {
    'start_city': df["start_city"],
    'end_city': df["end_city"],
    'trip_duration_minutes': df["trip_duration_minutes"],
    'traffic_level': df["traffic_level"],
    'traffic_wait_minutes': df["traffic_wait_minutes"],
    'day_of_week': df["day_of_week"],
    'hour': df["hour"],
    'minute': df["minute"],
    "passenger_count": df["passenger_count"]
}

df = pd.DataFrame(data)
# print(df)

# # Encode categorical features
label_encoders = {}
for col in ['start_city', 'end_city', 'traffic_level', 'day_of_week']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# # Split features/target
X1 = df[['start_city', 'end_city', 'day_of_week', 'hour', 'minute']]
y1 = df['traffic_level']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state=24)

model1 = XGBClassifier()
model1.fit(X1_train, y1_train)
print("Model 1 trained - traffic_level")

# ==== Model 2 =====
df['predicted_traffic_level'] = model1.predict(X1)

X2 = df[['start_city', 'end_city', 'day_of_week', 'hour', 'minute', 'predicted_traffic_level']]
y2 = df['traffic_wait_minutes']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = XGBRegressor()
model2.fit(X2_train, y2_train)
print("Model 2 trained - traffic_wait_minutes")

# ==== Model 3 ====
df['predicted_traffic_wait'] = model2.predict(X2)

# 3. Predict passenger_count (regression)
X3 = df[['start_city', 'end_city', 'day_of_week']]
y3 = df['passenger_count']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=10)

model3 = XGBRegressor()
model3.fit(X3_train, y3_train)
print("Model 3 trained - passenger_count")

# Predictions
y_pred1 = model1.predict(X1_test)
acc = accuracy_score(y1_test, y_pred1)
print(f"Accuracy: {acc:.2f}, model 1 (traffic_level classifier)")

y_pred2 = model2.predict(X2_test)
mae = mean_absolute_error(y2_test, y_pred2)
print(f"Mean Absolute Error: {mae:.2f} minutes, model 2")

y_pred3 = model3.predict(X3_test)
mae = mean_absolute_error(y3_test, y_pred3)
print(f"Mean Absolute Error: {mae:.2f} passangers, model 3")

# ==== Save ====
with open('AI/model1.pkl', 'wb') as f:
    pickle.dump(model1, f)

with open('AI/model2.pkl', 'wb') as f:
    pickle.dump(model2, f)

with open('AI/model3.pkl', 'wb') as f:
    pickle.dump(model3, f)