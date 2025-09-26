import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination).
    """
    return r2_score(y_true, y_pred)

def plot_rul_predictions(y_true, y_pred, title="RUL Prediction vs True", save_path=None):
    """
    Plot true vs predicted Remaining Useful Life.
    """
    plt.figure(figsize=(10,6))
    plt.plot(y_true, label='True RUL')
    plt.plot(y_pred, label='Predicted RUL')
    plt.xlabel('Sample Index')
    plt.ylabel('RUL')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def print_evaluation_metrics(y_true, y_pred):
    """
    Print RMSE and R² metrics formatted.
    """
    rmse = calculate_rmse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

if __name__ == '__main__':
    print("utils.py loaded: Provides metrics and plotting utilities for RUL prediction")
