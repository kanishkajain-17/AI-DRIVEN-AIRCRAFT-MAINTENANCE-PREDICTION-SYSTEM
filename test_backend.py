import pandas as pd
from backend import predict_anomaly
from LSTM_testing_preprocessed import run_lstm_pipeline

def run_backend(file_path):

    df = pd.read_csv(file_path)

    # -------------------------------
    # STEP 1: ANOMALY DETECTION
    # -------------------------------
    result = predict_anomaly(df)

    # Save output
    result.to_csv("outputs/test_anomaly_results.csv", index=False)

    print("Backend processing complete")

    # -------------------------------
    # STEP 2: LSTM PIPELINE
    # -------------------------------
    final_output = run_lstm_pipeline("outputs/test_anomaly_results.csv")

    return final_output