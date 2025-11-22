# evaluation.py
import numpy as np
import pandas as pd

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def evaluate_series(y_true, y_pred):
    return {
        'MAE': float(mae(y_true, y_pred)),
        'RMSE': float(rmse(y_true, y_pred)),
        'MAPE': float(mape(y_true, y_pred))
    }
