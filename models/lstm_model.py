# lstm_model.py
import numpy as np
import pandas as pd
import os
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def get_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU detected: using GPU.")
        return 'GPU'
    else:
        logger.info("No GPU detected: using CPU.")
        return 'CPU'

class LSTMModel:
    def __init__(self, window_size=30, epochs=50, batch_size=32):
        self.window = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler()

    def create_windowed(self, series: np.ndarray):
        X, y = [], []
        for i in range(len(series) - self.window):
            X.append(series[i:i+self.window])
            y.append(series[i+self.window])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    def fit(self, train_series: pd.Series):
        arr = train_series.values.reshape(-1,1).astype(float)
        arr_scaled = self.scaler.fit_transform(arr)
        X, y = self.create_windowed(arr_scaled.flatten())

        model = Sequential()
        model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        logger.info("Training LSTM...")
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, callbacks=[es], verbose=1)
        self.model = model
        logger.info("LSTM trained.")

    def forecast(self, last_series: pd.Series, periods: int):
        vals = last_series.values.reshape(-1,1).astype(float)
        vals_scaled = self.scaler.transform(vals)
        window = list(vals_scaled.flatten()[-self.window:])
        preds = []
        for _ in range(periods):
            x = np.array(window[-self.window:]).reshape((1, self.window, 1))
            p = self.model.predict(x)[0,0]
            preds.append(p)
            window.append(p)
        preds = np.array(preds).reshape(-1,1)
        preds_inv = self.scaler.inverse_transform(preds).flatten()
        return preds_inv
