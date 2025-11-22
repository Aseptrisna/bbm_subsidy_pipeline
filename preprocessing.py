# # preprocessing.py
# # Versi CSV — tanpa database
# # Kompatibel dengan run_pipeline_csv.py

# import pandas as pd


# # ================================
# # LOAD TRANSACTIONS FROM CSV
# # ================================
# def load_transactions_csv(path: str):
#     """
#     Load transaksi dari file CSV.
#     CSV harus memiliki kolom:
#     - id
#     - vehicle_id
#     - waktu_transaksi
#     - jenis_bbm
#     - volume_liter
#     """
#     df = pd.read_csv(path)

#     if df.empty:
#         return df

#     df["waktu_transaksi"] = pd.to_datetime(df["waktu_transaksi"], utc=True)
#     return df


# # ================================
# # RESAMPLE SERIES PER VEHICLE
# # ================================
# def aggregate_series(df, vehicle_id, period="D"):
#     """
#     Mengubah transaksi kendaraan menjadi time-series agregasi harian/bulanan.
#     Menghasilkan range 2025-01-01 sampai 2025-12-31.
#     """

#     veh = df[df["vehicle_id"] == vehicle_id].copy()

#     # Jika kendaraan tidak punya data → isi 0 selama 1 tahun
#     if veh.empty:
#         idx = pd.date_range(
#             start="2025-01-01", end="2025-12-31", freq=period, tz="UTC"
#         )
#         s = pd.Series(0.0, index=idx)
#         s.index.name = "ds"
#         return s

#     veh = veh.set_index("waktu_transaksi")

#     s = veh["volume_liter"].resample(period).sum()

#     # Pastikan tahun penuh
#     start = pd.Timestamp("2025-01-01", tz="UTC")
#     end = pd.Timestamp("2025-12-31", tz="UTC")

#     idx = pd.date_range(start=start, end=end, freq=period, tz="UTC")
#     s = s.reindex(idx, fill_value=0.0)
#     s.index.name = "ds"

#     return s.sort_index()


# # ================================
# # BUILD FEATURES FOR ML MODELS
# # ================================
# def make_features(df_ts):
#     """
#     Input: Series time-series
#     Output: DataFrame dengan:
#         - ds
#         - y
#         - dayofweek
#         - is_weekend
#         - month
#     """
#     df = pd.DataFrame({"ds": df_ts.index, "y": df_ts.values})
#     df["dayofweek"] = df["ds"].dt.dayofweek
#     df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
#     df["month"] = df["ds"].dt.month
#     return df


# # ================================
# # TRAIN/TEST SPLIT BY DATE
# # ================================
# def train_test_split_by_date(df, split_date="2025-10-01"):
#     split_dt = pd.to_datetime(split_date, utc=True)

#     train = df[df["ds"] < split_dt].reset_index(drop=True)
#     test = df[df["ds"] >= split_dt].reset_index(drop=True)

#     return train, test


# preprocessing.py — versi stabil untuk Prediction Pipeline
# smoothing + log-transform untuk stabilitas model

import pandas as pd
import numpy as np

# ============================================
# LOAD CSV
# ============================================
def load_transactions_csv(path: str):
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["waktu_transaksi"] = pd.to_datetime(df["waktu_transaksi"], utc=True)
    return df


# ============================================
# MOVING AVERAGE SMOOTHING
# ============================================
def smooth_series(series, window=7):
    """
    Menghaluskan time-series agar tidak menyebabkan ML overfit
    dan menghindari prediksi ekstrem.
    """
    return series.rolling(window=window, min_periods=1).mean()


# ============================================
# LIMIT HARUS MASUK AKAL
# ============================================
def clip_realistic(series, max_daily=80):
    """
    Membatasi konsumsi harian agar tidak melebihi batas realistis.
    default max_daily=80 liter/hari (configurable)
    """
    return series.clip(lower=0, upper=max_daily)


# ============================================
# AGGREGATE SERIES
# ============================================
def aggregate_series(df, vehicle_id, period="D", smooth_window=7, max_daily=80):
    """
    Mengubah transaksi kendaraan menjadi time-series agregasi harian/bulanan.
    Menghasilkan range 2025-01-01 sampai 2025-12-31.
    - smoothing (moving average)
    - clipping untuk nilai tidak realistis
    """

    veh = df[df["vehicle_id"] == vehicle_id].copy()

    # full-year index
    idx = pd.date_range(start="2025-01-01", end="2025-12-31", freq=period, tz="UTC")

    if veh.empty:
        s = pd.Series(0.0, index=idx)
        s.index.name = "ds"
        return s

    veh = veh.set_index("waktu_transaksi")

    s = veh["volume_liter"].resample(period).sum()
    s = s.reindex(idx, fill_value=0.0)

    # 1) smoothing
    s = smooth_series(s, window=smooth_window)

    # 2) clipping to realistic daily max (avoid extreme outliers)
    s = clip_realistic(s, max_daily=max_daily)

    s.index.name = "ds"
    return s.sort_index()


# ============================================
# MAKE FEATURES
# ============================================
def make_features(series, do_log_transform=True):
    """
    Build features from series.
    If do_log_transform==True, apply y = log1p(y) for stability.
    """
    df = pd.DataFrame({"ds": series.index, "y": series.values})

    if do_log_transform:
        # log-transform for stable modeling (inverse with expm1 after prediction)
        df["y"] = np.log1p(df["y"].astype(float))

    df["dayofweek"] = df["ds"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["month"] = df["ds"].dt.month

    return df


# ============================================
# TRAIN / TEST SPLIT
# ============================================
def train_test_split_by_date(df, split_date="2025-10-01"):
    split_dt = pd.to_datetime(split_date, utc=True)
    train = df[df["ds"] < split_dt].reset_index(drop=True)
    test = df[df["ds"] >= split_dt].reset_index(drop=True)
    return train, test
