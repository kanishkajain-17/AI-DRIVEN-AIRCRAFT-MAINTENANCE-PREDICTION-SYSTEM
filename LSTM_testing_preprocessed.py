# -------------------------------
# Basic libraries
# -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Preprocessing
# -------------------------------
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# -------------------------------
# Deep Learning (LSTM)
# -------------------------------
from keras.models import load_model
from sklearn.cluster import KMeans

import os


def run_lstm_pipeline(file_path):

    # -------------------------------
    # LOAD DATA
    # -------------------------------
    df = pd.read_csv(file_path)

    print(df.head())
    print(df.shape)
    print(df.columns)

    # -------------------------------
    # CLEANING
    # -------------------------------
    if 'max_cycle' in df.columns:
        df = df.drop(columns=['max_cycle'])

    df['anomaly'] = df['anomaly'].map({-1: 1, 1: 0})

    # -------------------------------
    # FEATURE SELECTION
    # -------------------------------
    feature_cols = [
        col for col in df.columns
        if col not in ['unit_number', 'time_in_cycles', 'RUL']
    ]

    # -------------------------------
    # SCALING
    # -------------------------------
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.to_csv("outputs/test_FD001_after_anomaly_scaled.csv", index=False)

    # -------------------------------
    # CREATE SEQUENCES
    # -------------------------------
    WINDOW_SIZE = 30

    def create_test_sequences(df, feature_cols, window_size):

        X_test = []
        seq_per_engine = []

        for uid in df['unit_number'].unique():

            engine_data = df[df['unit_number'] == uid]
            data = engine_data[feature_cols].values

            seq_count = 0

            for i in range(len(data) - window_size):
                X_test.append(data[i:i+window_size])
                seq_count += 1

            seq_per_engine.append(seq_count)

        return np.array(X_test), seq_per_engine

    X_test, seq_per_engine = create_test_sequences(
        df,
        feature_cols,
        WINDOW_SIZE
    )

    print("X_test shape:", X_test.shape)

    # -------------------------------
    # LOAD MODEL
    # -------------------------------
    model = load_model("models/LSTM_model.keras")

    # -------------------------------
    # PREDICTION
    # -------------------------------
    y_pred_seq = model.predict(X_test)

    print("Prediction shape:", y_pred_seq.shape)

    # -------------------------------
    # FINAL PREDICTION PER ENGINE
    # -------------------------------
    final_predictions = []

    start = 0

    for seq_count in seq_per_engine:

        last_index = start + seq_count - 1
        final_predictions.append(y_pred_seq[last_index][0])

        start += seq_count

    print("Total engines:", len(final_predictions))

    # -------------------------------
    # RESULTS
    # -------------------------------
    results_df = pd.DataFrame({
        "unit_number": df['unit_number'].unique(),
        "Predicted_RUL": final_predictions
    })

    print(results_df.head())

    # -------------------------------
    # CLUSTERING
    # -------------------------------
    X_cluster = results_df[['Predicted_RUL']]

    kmeans = KMeans(n_clusters=3, random_state=42)

    results_df['cluster'] = kmeans.fit_predict(X_cluster)

    centers = kmeans.cluster_centers_.flatten()

    cluster_order = np.argsort(centers)

    health_map = {
        cluster_order[0]: "Critical",
        cluster_order[1]: "Warning",
        cluster_order[2]: "Safe"
    }

    results_df['Health_Status'] = results_df['cluster'].map(health_map)

    print(results_df.head())

    # -------------------------------
    # SAVE OUTPUT
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)

    results_df.to_csv("outputs/final_output.csv", index=False)

    print("✅ Final output saved")

    return results_df