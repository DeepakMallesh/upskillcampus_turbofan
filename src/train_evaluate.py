import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from src.model import LSTMRegressor, get_random_forest


def train_lstm(model, train_loader, val_loader=None, epochs=30, lr=1e-3, device='cpu'):
    """
    Train LSTM model.
    Args:
      model: LSTMRegressor instance
      train_loader: DataLoader for training data
      val_loader: Optional DataLoader for validation
      epochs: Number of epochs
      lr: learning rate
      device: 'cuda' or 'cpu'

    Returns:
      trained model
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).float()
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device).float(), y_val.to(device).float()
                    y_pred_val = model(X_val)
                    val_loss = criterion(y_pred_val, y_val)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f"Epoch {epoch}/{epochs} - Train RMSE: {np.sqrt(avg_train_loss):.4f} - Val RMSE: {np.sqrt(avg_val_loss):.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} - Train RMSE: {np.sqrt(avg_train_loss):.4f}")

    return model


def evaluate_lstm(model, X, y, device='cpu'):
    """
    Evaluate LSTM model on given data.
    Args:
      model: trained LSTMRegressor
      X: numpy array (samples, seq_len, features)
      y: true labels
      device: 'cuda' or 'cpu'

    Returns:
      RMSE, R2, predictions numpy array
    """
    model.eval()
    model.to(device)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return rmse, r2, preds


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Train Random Forest regressor.
    """
    rf = get_random_forest(n_estimators=n_estimators, max_depth=max_depth)
    rf.fit(X_train, y_train)
    return rf


def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluate Random Forest regressor.
    """
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return rmse, r2, preds


def save_lstm_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"LSTM model saved at {path}")


def load_lstm_model(input_size, path, device='cpu'):
    model = LSTMRegressor(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"LSTM model loaded from {path}")
    return model


def create_dataloader(X, y, batch_size=64, shuffle=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == '__main__':
    print("Run this module by importing functions in your training pipeline or app.py")
