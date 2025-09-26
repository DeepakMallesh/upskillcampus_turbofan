import os
import pandas as pd
import numpy as np

# Base data folder path
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def read_cmapss_file(filename, n_cols):
    """
    Reads CMAPSS dataset file, returns a pandas DataFrame.
    `filename`: file name (e.g., 'train_FD001.txt')
    `n_cols`: number of columns in the file (26 for CMAPSS)
    """
    file_path = os.path.join(DATA_DIR, filename)
    col_names = ['unit', 'cycle'] + \
                [f'op_setting_{i}' for i in range(1,4)] + \
                [f'sensor_{i}' for i in range(1, n_cols-4+1)]
    df = pd.read_csv(file_path, sep=' ', header=None, names=col_names, engine='python')
    # Remove last two empty columns created by space separation
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def read_rul_file(filename):
    """
    Reads RUL target file, returns a numpy array.
    Each row is the true RUL for one test engine (in order).
    """
    file_path = os.path.join(DATA_DIR, filename)
    return np.loadtxt(file_path)

def load_dataset(dataset_id='FD001'):
    """
    Loads train, test and RUL files for a given subset.
    Returns train_df, test_df, rul_array
    """
    n_cols = 26
    train_file = f'train_{dataset_id}.txt'
    test_file = f'test_{dataset_id}.txt'
    rul_file = f'RUL_{dataset_id}.txt'
    train_df = read_cmapss_file(train_file, n_cols)
    test_df = read_cmapss_file(test_file, n_cols)
    rul_array = read_rul_file(rul_file)
    return train_df, test_df, rul_array

# Example usage (for FD001):
if __name__ == '__main__':
    train_df, test_df, rul_array = load_dataset('FD001')
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"RUL shape: {rul_array.shape}")
    print(train_df.head())
    print(test_df.head())
    print(rul_array[:10])
