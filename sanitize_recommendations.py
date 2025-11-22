# sanitize_recommendations.py
import pandas as pd
import numpy as np

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def sanitize_recommendations(
    rec_df: pd.DataFrame,
    vehicles_df: pd.DataFrame,
    *,
    cap_daily_multiplier: float = 3.0,
    cap_monthly_multiplier: float = 1.5,
    cap_yearly_multiplier: float = 1.2,
    min_daily_floor: float = 1.0,
    max_daily_hardcap: float = 10000.0,
    historical_max_dict: dict = None
) -> pd.DataFrame:
    """
    Sanitize rekomendasi:
    - rec_df: DataFrame with columns
        ['vehicle_id','method','period_type','period_start','period_end','recommended_liter']
    - vehicles_df: DataFrame with vehicle info containing 'id' and 'kuota_bulanan_liter'
    Returns sanitized DataFrame with extra columns:
      - original_value
      - recommended_liter (sanitized)
      - flag_over_quota
      - flag_outlier
    """

    df = rec_df.copy()
    df.columns = [c.strip() for c in df.columns]

    # types
    df['vehicle_id'] = df['vehicle_id'].astype(int)
    df['recommended_liter'] = df['recommended_liter'].apply(_safe_float).fillna(0.0)

    # prepare vehicles df
    veh = vehicles_df.copy()
    if 'id' in veh.columns:
        veh = veh.rename(columns={'id': 'vehicle_id'})
    if 'kuota_bulanan_liter' not in veh.columns:
        raise ValueError("vehicles_df must contain 'kuota_bulanan_liter'")

    df = df.merge(veh[['vehicle_id', 'kuota_bulanan_liter']], on='vehicle_id', how='left')

    if historical_max_dict is None:
        historical_max_dict = {}

    def compute_daily_cap(kuota_monthly):
        if pd.isna(kuota_monthly):
            avg_daily = 0.0
        else:
            avg_daily = float(kuota_monthly) / 30.0
        cap = max(avg_daily * cap_daily_multiplier, min_daily_floor)
        cap = min(cap, max_daily_hardcap)
        return cap

    def compute_monthly_cap(kuota_monthly):
        if pd.isna(kuota_monthly):
            return np.inf
        return float(kuota_monthly) * cap_monthly_multiplier

    def compute_yearly_cap(kuota_monthly):
        if pd.isna(kuota_monthly):
            return np.inf
        return float(kuota_monthly) * 12.0 * cap_yearly_multiplier

    df['original_value'] = df['recommended_liter']
    df['flag_over_quota'] = False
    df['flag_outlier'] = False

    sanitized = []
    for _, row in df.iterrows():
        pid = int(row['vehicle_id'])
        ptype = row['period_type']
        val = float(row['recommended_liter'])
        kuota_monthly = _safe_float(row.get('kuota_bulanan_liter', np.nan))

        if ptype == 'daily':
            cap = compute_daily_cap(kuota_monthly)
            hist_max = historical_max_dict.get(pid, None)
            if hist_max is not None and hist_max > 0:
                cap = min(cap, hist_max * 3.0)
            newval = min(val, cap)
            flag_over = newval < val
            flag_out = False
            if hist_max is not None and val > hist_max * 3.0:
                flag_out = True

        elif ptype == 'monthly':
            cap = compute_monthly_cap(kuota_monthly)
            newval = min(val, cap)
            flag_over = newval < val
            flag_out = False

        elif ptype == 'yearly':
            cap = compute_yearly_cap(kuota_monthly)
            newval = min(val, cap)
            flag_over = newval < val
            flag_out = False

        else:
            newval = val
            flag_over = False
            flag_out = False

        newval = round(float(newval), 2)

        sanitized.append({
            **row.to_dict(),
            'recommended_liter': newval,
            'original_value': round(float(row['original_value']), 2),
            'flag_over_quota': bool(flag_over),
            'flag_outlier': bool(flag_out)
        })

    out = pd.DataFrame(sanitized)
    out['period_start'] = pd.to_datetime(out['period_start'])
    out['period_end'] = pd.to_datetime(out['period_end'])

    # reorder
    keep = ['vehicle_id','method','period_type','period_start','period_end',
            'recommended_liter','original_value','flag_over_quota','flag_outlier','kuota_bulanan_liter']
    cols = [c for c in keep if c in out.columns] + [c for c in out.columns if c not in keep]
    out = out[cols]
    return out
