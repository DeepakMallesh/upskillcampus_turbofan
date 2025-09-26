import streamlit as st
import torch
import numpy as np
import os
import pandas as pd

from src.data_loader import load_dataset
from src.feature_engineering import preprocess_data
from src.train_evaluate import (
    train_lstm, evaluate_lstm,
    train_random_forest, evaluate_random_forest,
    create_dataloader, save_lstm_model
)
from src.utils import plot_rul_predictions

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_best.pth")
PRED_PATH = os.path.join(OUTPUT_DIR, "predictions.csv")
RESULT_LOG = os.path.join(OUTPUT_DIR, "results.log")

def main():
    st.title("Turbofan Engine Remaining Useful Life (RUL) Prediction")
    dataset_id = st.selectbox("Select Dataset", ['FD001', 'FD002', 'FD003', 'FD004'])
    seq_length = st.slider("Sequence Length", min_value=10, max_value=60, value=30, step=1)

    if st.button("Load & Preprocess Data"):
        with st.spinner("Loading data..."):
            train_df, test_df, rul_array = load_dataset(dataset_id)
        with st.spinner("Preprocessing data..."):
            X_train, y_train, test_scaled, scaler = preprocess_data(train_df, test_df, seq_length=seq_length)
        # Show diagnostic info and NaN stats
        st.write(f"Training samples: {X_train.shape[0]}, Sequence length: {seq_length}, Features: {X_train.shape[2]}")
        st.write(f"NaNs in X_train: {np.isnan(X_train).sum()}, NaNs in y_train: {np.isnan(y_train).sum()}")
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            st.error("Data still contains NaNs. Fix preprocessing!")
            st.stop()
        st.session_state['X_train'] = X_train
        st.session_state['y_train'] = y_train
        st.session_state['train_loader'] = create_dataloader(X_train, y_train, batch_size=64)
        st.success("Data loaded and preprocessed.")

    if 'X_train' in st.session_state and 'train_loader' in st.session_state:
        model_choice = st.selectbox("Choose Model", ["LSTM", "Random Forest"])
        if st.button("Train & Evaluate Model"):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            train_loader = st.session_state['train_loader']
            result_lines = []
            if model_choice == "LSTM":
                from src.model import LSTMRegressor
                st.info("Training LSTM model...")
                lstm_model = LSTMRegressor(input_size=X_train.shape[2])
                lstm_model = train_lstm(lstm_model, train_loader, epochs=20, device=DEVICE)
                save_lstm_model(lstm_model, MODEL_PATH)
                st.success(f"LSTM training completed! Model saved as: {MODEL_PATH}")
                rmse, r2, preds = evaluate_lstm(lstm_model, X_train, y_train, device=DEVICE)
                st.metric("RMSE", f"{rmse:.4f}")
                st.metric("R² Score", f"{r2:.4f}")
                plot_rul_predictions(y_train, preds, "LSTM Training RUL Prediction vs True")
                pd.DataFrame({"True_RUL": y_train, "Predicted_RUL": preds}).to_csv(PRED_PATH, index=False)
                st.success(f"Predictions saved as: {PRED_PATH}")
                result_lines.append(f"LSTM Model - RMSE: {rmse:.4f}, R2: {r2:.4f}\n")
            else:
                st.info("Training Random Forest model...")
                X_train_rf = X_train.mean(axis=1)
                rf_model = train_random_forest(X_train_rf, y_train)
                st.success("Random Forest training completed!")
                rmse, r2, preds = evaluate_random_forest(rf_model, X_train_rf, y_train)
                st.metric("RMSE", f"{rmse:.4f}")
                st.metric("R² Score", f"{r2:.4f}")
                plot_rul_predictions(y_train, preds, "Random Forest Training RUL Prediction vs True")
                pd.DataFrame({"True_RUL": y_train, "Predicted_RUL": preds}).to_csv(PRED_PATH, index=False)
                st.success(f"Predictions saved as: {PRED_PATH}")
                result_lines.append(f"Random Forest - RMSE: {rmse:.4f}, R2: {r2:.4f}\n")
            with open(RESULT_LOG, "a") as f:
                f.writelines(result_lines)
            st.success(f"Results logged at: {RESULT_LOG}")

if __name__ == "__main__":
    st.set_page_config(page_title="Turbofan RUL Predictor", layout="wide")
    main()
