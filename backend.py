import pandas as pd
import pickle

MODEL_PATH = "models/model.pkl"
FEATURE_PATH = "models/feature_names.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def load_features():
    with open(FEATURE_PATH, "rb") as f:
        return pickle.load(f)

def predict_anomaly(df: pd.DataFrame):
    model = load_model()
    features = load_features()
    X = df[features]

    df["anomaly"] = model.predict(X)
    return df