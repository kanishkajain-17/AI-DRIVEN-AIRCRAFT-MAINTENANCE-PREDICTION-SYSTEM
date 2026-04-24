import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import os

TRAIN_FILES = ["X_final_preprocessed_normalized.csv"]

FEATURES = [
    "time_in_cycles", "sensor_measurement_2", "sensor_measurement_3",
    "sensor_measurement_4", "sensor_measurement_7",
    "sensor_measurement_9", "sensor_measurement_11",
    "sensor_measurement_12", "sensor_measurement_17",
    "sensor_measurement_20", "sensor_measurement_21"
]

dfs = [pd.read_csv(f) for f in TRAIN_FILES]
df = pd.concat(dfs, ignore_index=True)

X = df[FEATURES]

X_train, _ = train_test_split(X, test_size=0.2, random_state=42)

model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_train)

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print("Isolation Forest model saved")