import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def drop_constant_sensors(df, threshold=0.01):
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    variances = df[sensor_cols].var()
    constant_sensors = variances[variances < threshold].index.tolist()
    # Remove ALL-NaN sensors too
    nan_sensors = [col for col in sensor_cols if df[col].isna().all()]
    drop_sensors = list(set(constant_sensors + nan_sensors))
    df_cleaned = df.drop(columns=drop_sensors)
    return df_cleaned, drop_sensors

def add_rul_column(train_df):
    train_df = train_df.copy()
    max_cycle = train_df.groupby('unit')['cycle'].transform('max')
    train_df['RUL'] = max_cycle - train_df['cycle']
    return train_df

def normalize_sensors(train_df, test_df):
    features = [col for col in train_df.columns if col.startswith('sensor_') or col.startswith('op_setting_')]
    # Fill NaNs before scaling
    train_df[features] = train_df[features].fillna(0)
    test_df[features] = test_df[features].fillna(0)
    scaler = MinMaxScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    scaler.fit(train_df[features])
    train_df_scaled[features] = scaler.transform(train_df[features])
    test_df_scaled[features] = scaler.transform(test_df[features])
    return train_df_scaled, test_df_scaled, scaler

def create_sequences(df, seq_length=30, feature_cols=None, label_col='RUL'):
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col.startswith('sensor_') or col.startswith('op_setting_')]
    units = df['unit'].unique()
    X, y = [], []
    for unit in units:
        unit_df = df[df['unit'] == unit].sort_values('cycle')
        features = unit_df[feature_cols].values
        labels = unit_df[label_col].values
        for i in range(len(unit_df) - seq_length + 1):
            seq_x = features[i:i+seq_length]
            seq_y = labels[i+seq_length-1]
            X.append(seq_x)
            y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    # Replace any remaining NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y

def add_smoothed_rul(train_df, window=20, max_rul=130):
    train_df = train_df.copy()
    train_df = add_rul_column(train_df)
    smoothed = train_df.groupby('unit')['RUL'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    train_df['RUL_smoothed'] = smoothed.clip(upper=max_rul)
    return train_df

def preprocess_data(train_df, test_df, seq_length=30):
    train_df, dropped_sensors = drop_constant_sensors(train_df)
    test_df = test_df.drop(columns=dropped_sensors, errors='ignore')
    train_df = add_smoothed_rul(train_df)
    train_scaled, test_scaled, scaler = normalize_sensors(train_df, test_df)
    feature_cols = [col for col in train_scaled.columns if col.startswith('sensor_') or col.startswith('op_setting_')]
    X_train, y_train = create_sequences(train_scaled, seq_length=seq_length, feature_cols=feature_cols, label_col='RUL_smoothed')
    # Defensive final clean
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train, y_train, test_scaled, scaler

if __name__ == '__main__':
    print("feature_engineering.py module loaded. Use preprocessing functions to transform your data.")
