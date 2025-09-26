import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        LSTM model for sequence regression.
        Args:
            input_size: Number of input features per timestep.
            hidden_size: Number of LSTM hidden units.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability for regularization.
        """
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Output single RUL value

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_len, hidden_size)
        out = out[:, -1, :]               # take last time step
        out = self.fc(out)                # shape: (batch_size, 1)
        return out.squeeze(1)             # shape: (batch_size,)

def get_random_forest(n_estimators=100, max_depth=None, random_state=42):
    """
    Returns a sklearn RandomForestRegressor instance with given hyperparameters.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                               random_state=random_state, n_jobs=-1)
    return rf

if __name__ == '__main__':
    # Quick model test run
    x_dummy = torch.randn(16, 30, 14)  # batch=16, seq=30, features=14
    model = LSTMRegressor(input_size=14)
    y_out = model(x_dummy)
    print("LSTM output shape:", y_out.shape)  # Should be (16,)
