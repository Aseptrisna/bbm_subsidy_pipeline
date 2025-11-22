# arima_model.py
import pandas as pd
import pmdarima as pm
from loguru import logger

class ARIMAModel:
    def __init__(self):
        self.model = None

    def fit(self, train_series: pd.Series):
        logger.info("Fitting ARIMA Non-Seasonal...")

        # Jika data terlalu sedikit, ARIMA rawan error
        if len(train_series) < 6:
            raise ValueError("Data terlalu sedikit untuk ARIMA (min 6 bulan).")

        # ARIMA untuk data sparse: non-seasonal
        self.model = pm.auto_arima(
            train_series,
            seasonal=False,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            d=None,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True
        )

        logger.info(f"ARIMA fitted: order={self.model.order}")

    def forecast(self, periods: int):
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(n_periods=periods)
