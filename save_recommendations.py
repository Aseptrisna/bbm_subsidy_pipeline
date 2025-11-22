# save_recommendations.py
import pandas as pd
import numpy as np
from datetime import timedelta

def build_recommendations_df(vehicle_id: int, method: str, period_type: str, start_dates, values):
    rows = []
    for s, v in zip(start_dates, values):
        if period_type == 'daily':
            period_end = s + timedelta(days=1)
        elif period_type == 'monthly':
            period_end = (s + pd.DateOffset(months=1))
        else:
            period_end = (s + pd.DateOffset(years=1))
        rows.append({
            'vehicle_id': int(vehicle_id),
            'method': method,
            'period_type': period_type,
            'period_start': pd.to_datetime(s),
            'period_end': pd.to_datetime(period_end),
            'recommended_liter': float(v)
        })
    df = pd.DataFrame(rows)
    return df

def aggregate_to_monthly(dates, values):
    s = pd.Series(values, index=pd.to_datetime(dates))
    monthly = s.resample('M').sum()
    return monthly.index.to_list(), monthly.values

def aggregate_to_yearly(dates, values):
    s = pd.Series(values, index=pd.to_datetime(dates))
    yearly = s.resample('Y').sum()
    return yearly.index.to_list(), yearly.values
