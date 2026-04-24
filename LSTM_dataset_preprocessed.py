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
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

import os


# ================================
# MAIN FUNCTION
# ================================
def train_lstm(file_path):

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

    # Optional save (kept from your code)
    df.to_csv("outputs/FD001_after_anomaly_scaled.csv", index=False)

    # -------------------------------
    # SEQUENCE CREATION
    # -------------------------------
    def create_sequences(df, feature_cols, window_size=30):
        X, y = [], []

        for uid in df['unit_number'].unique():
            engine_data = df[df['unit_number'] == uid]

            for i in range(len(engine_data) - window_size):
                X.append(
                    engine_data.iloc[i:i+window_size][feature_cols].values
                )
                y.append(
                    engine_data.iloc[i+window_size]['RUL']
                )

        return np.array(X), np.array(y)

    WINDOW_SIZE = 30

    X_train, y_train = create_sequences(
        df,
        feature_cols,
        WINDOW_SIZE
    )

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # -------------------------------
    # MODEL
    # -------------------------------
    model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),

        LSTM(32),
        Dropout(0.2),

        Dense(1)   # RUL
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.summary()

    # -------------------------------
    # TRAINING
    # -------------------------------
    early_stop = EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # -------------------------------
    # SAVE MODEL
    # -------------------------------
    os.makedirs("models", exist_ok=True)

    model.save("models/LSTM_model.keras")

    print("LSTM model saved successfully")


# ================================
# RUN (ONLY WHEN FILE EXECUTED DIRECTLY)
# ================================
if __name__ == "__main__":
    train_lstm("outputs/test_anomaly_RUL_results.csv")