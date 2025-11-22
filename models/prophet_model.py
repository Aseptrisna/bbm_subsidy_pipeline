# prophet_model.py
from prophet import Prophet
import pandas as pd
from loguru import logger

class ProphetModel:
    def __init__(self):
        self.model = Prophet()

    def fit(self, train_df: pd.DataFrame):
        logger.info("Fitting Prophet model...")

        df = train_df.copy()

        # HILANGKAN TIMEZONE WAJIB
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

        self.model.fit(df[["ds", "y"]])

        logger.info("Prophet fitted.")

    def forecast(self, periods: int, freq='D'):
        # Buat future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Pastikan future['ds'] juga tanpa timezone
        future["ds"] = future["ds"].dt.tz_localize(None)

        fcst = self.model.predict(future)

        # Ambil hanya periode future yang kita butuhkan
        df_out = fcst[["ds", "yhat"]].tail(periods).reset_index(drop=True)

        return df_out
