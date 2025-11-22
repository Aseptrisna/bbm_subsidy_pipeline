import argparse
import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta
from loguru import logger

from preprocessing import (
    load_transactions_csv,
    aggregate_series,
    make_features,
    train_test_split_by_date
)

from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.lstm_model import LSTMModel

from evaluation import evaluate_series
from save_recommendations import (
    build_recommendations_df,
    aggregate_to_monthly,
    aggregate_to_yearly
)
from sanitize_recommendations import sanitize_recommendations

logger.add("pipeline.log")


async def process_vehicle(vehicle_id, df_transactions, method_list, horizon_days=365):
    results = {}

    # aggregate (this returns raw liters smoothed & clipped already)
    series_daily = aggregate_series(df_transactions, vehicle_id, period="D")

    if len(series_daily) < 5:
        logger.warning(f"Vehicle {vehicle_id}: data terlalu sedikit, dilewati.")
        return {}

    # build features (this applies log1p)
    df_daily = make_features(series_daily, do_log_transform=True)
    train, test = train_test_split_by_date(df_daily, split_date="2025-10-01")

    train_series = pd.Series(train["y"].values, index=train["ds"])
    test_series = pd.Series(test["y"].values, index=test["ds"])

    total_periods = len(test_series) + horizon_days

    for method in method_list:
        # ARIMA
        if method == "arima":
            m = ARIMAModel()
            m.fit(train_series)
            preds = m.forecast(periods=total_periods)
            # preds are in log-space (because trained on log1p)
            test_pred = preds[: len(test_series)]
            future_pred = preds[len(test_series): len(test_series) + horizon_days]

            # inverse -> liters
            test_pred_l = np.expm1(test_pred)
            future_pred_l = np.expm1(future_pred)

            metrics = evaluate_series(np.expm1(test_series.values), test_pred_l)
            results[method] = (metrics, test_pred_l, future_pred_l)

        # Prophet
        elif method == "prophet":
            pm = ProphetModel()
            # pm.fit(train.reset_index(drop=True)[["ds", "y"]])
            tmp = train[['ds', 'y']].copy()
            tmp["ds"] = tmp["ds"].dt.tz_localize(None)
            pm.fit(tmp)

            fcst = pm.forecast(periods=total_periods, freq="D")
            yhat = fcst["yhat"].values

            test_pred = yhat[: len(test_series)]
            future_pred = yhat[len(test_series): len(test_series) + horizon_days]

            test_pred_l = np.expm1(test_pred)
            future_pred_l = np.expm1(future_pred)

            metrics = evaluate_series(np.expm1(test_series.values), test_pred_l)
            results[method] = (metrics, test_pred_l, future_pred_l)

        # LSTM
        elif method == "lstm":
            lm = LSTMModel(window_size=30, epochs=30)
            lm.fit(train_series)
            last_series = pd.concat([train_series, test_series])
            preds_all = lm.forecast(last_series, periods=total_periods)

            test_pred = preds_all[: len(test_series)]
            future_pred = preds_all[len(test_series): len(test_series) + horizon_days]

            test_pred_l = np.expm1(test_pred)
            future_pred_l = np.expm1(future_pred)

            metrics = evaluate_series(np.expm1(test_series.values), test_pred_l)
            results[method] = (metrics, test_pred_l, future_pred_l)

    return results


async def main(methods):
    logger.info("===================================")
    logger.info(" RUNNING PIPELINE CSV-ONLY VERSION ")
    logger.info("===================================")

    # load CSV
    logger.info("Loading transactions from CSV...")
    df = load_transactions_csv("data/fuel_transactions.csv")
    logger.info("Loading vehicles from CSV...")
    dfv = pd.read_csv("data/vehicles.csv")

    vehicle_ids = dfv["id"].unique().tolist()
    all_results = {}

    for vid in vehicle_ids:
        logger.info(f"Processing vehicle {vid} ...")
        res = await process_vehicle(vid, df, methods)
        if not res:
            continue

        for method, (metrics, test_pred, future_pred) in res.items():
            logger.info(f"VID {vid} - {method.upper()} metrics: {metrics}")

            start_date = pd.Timestamp("2026-01-01")
            future_index = pd.date_range(start=start_date, periods=len(future_pred), freq="D")

            # Daily
            daily_df = build_recommendations_df(
                vid,
                method,
                "daily",
                future_index,
                future_pred
            )

            # Monthly
            monthly_idx, monthly_vals = aggregate_to_monthly(future_index, future_pred)
            monthly_df = build_recommendations_df(
                vid, method, "monthly", monthly_idx, monthly_vals
            )

            # Yearly
            yearly_idx, yearly_vals = aggregate_to_yearly(future_index, future_pred)
            yearly_df = build_recommendations_df(
                vid, method, "yearly", yearly_idx, yearly_vals
            )

            final_df = pd.concat([daily_df, monthly_df, yearly_df], ignore_index=True)

            # sanitize with vehicles quotas
            final_df_sanitized = sanitize_recommendations(final_df, dfv)

            if method not in all_results:
                all_results[method] = []
            all_results[method].append(final_df_sanitized)

    # export per method
    for method, listdfs in all_results.items():
        full = pd.concat(listdfs, ignore_index=True)
        fname = f"result/rekomendasi_{method}.csv"
        full.to_csv(fname, index=False)
        logger.info(f"Saved {fname} with {len(full)} rows")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="all", choices=["all", "arima", "prophet", "lstm"])
    args = parser.parse_args()
    methods = [args.method] if args.method != 'all' else ["arima", "prophet", "lstm"]
    asyncio.run(main(methods))

