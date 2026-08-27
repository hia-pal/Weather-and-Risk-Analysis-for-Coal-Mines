# -*- coding: utf-8 -*-
"""
DYNAMIC PRODUCTION LOSS FORECAST PIPELINE - LIGHTGBM VERSION
==============================================================
Fetches weather forecast from Open-Meteo, builds the FULL feature set
(weather + engineered weather + lag/rolling + offtake) exactly as used
in training, and calls each area's trained LightGBM model to generate
Predicted_Loss_Pct for the next FORECAST_HORIZON_DAYS days.

Because lag/rolling features depend on recent history that doesn't
exist yet for future dates, this script forecasts RECURSIVELY:
day 1 is predicted from real history, then day 1's prediction is
appended to that area's history so day 2 can be predicted, etc.

Risk scoring is still handled separately by calculate_risk.py.
"""

import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration"""
    BASE_PATH = "."
    MODELS_PATH = os.path.join(BASE_PATH, "Model_Outputs", "Models")
    MASTER_PATH = os.path.join(BASE_PATH, "Model_Outputs")
    FORECAST_PATH = os.path.join(BASE_PATH, "Forecast_Outputs")
    COORDS_PATH = os.path.join(BASE_PATH, "Preliminary Datasets", "Areas.xlsx")

    EXCLUDED_AREAS = [5190, 6003, 8351, 4151, 4251]
    LAG_SEED_DAYS = 30          # how many days of real history to seed lag/rolling features
    ARCHIVE_LAG_DAYS = 5
    FORECAST_HORIZON_DAYS = 15

    OPEN_METEO_DAILY_VARS = (
        "precipitation_sum,precipitation_hours,temperature_2m_max,"
        "temperature_2m_min,temperature_2m_mean,wind_speed_10m_max,"
        "wind_gusts_10m_max"
    )

    OPEN_METEO_COLUMN_MAP = {
        "precipitation_sum": "total_rain_mm",
        "precipitation_hours": "precipitation_hours",
        "temperature_2m_max": "max_temp_c",
        "temperature_2m_min": "min_temp_c",
        "temperature_2m_mean": "mean_temp_c",
        "wind_speed_10m_max": "max_wind_speed_kmh",
        "wind_gusts_10m_max": "max_wind_gust_kmh",
    }

    @classmethod
    def get_dates(cls):
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        archive_end = today - timedelta(days=cls.ARCHIVE_LAG_DAYS)
        forecast_start = archive_end + timedelta(days=1)
        forecast_end = forecast_start + timedelta(days=cls.FORECAST_HORIZON_DAYS - 1)
        return {
            "today": today,
            "archive_end": archive_end,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
        }


# =============================================================================
# HELPER FUNCTIONS (unchanged from original)
# =============================================================================

def safe_file_load(filepath, **kwargs):
    try:
        if filepath.endswith(".csv"):
            return pd.read_csv(filepath, **kwargs)
        elif filepath.endswith(".xlsx"):
            return pd.read_excel(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
    except FileNotFoundError:
        print(f"  ⚠️ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"  ⚠️ Error loading {filepath}: {e}")
        return None


def dms_to_decimal(dms_str):
    if pd.isna(dms_str) or dms_str == "":
        return None
    dms_str = str(dms_str).strip()
    try:
        return float(dms_str)
    except Exception:
        pass
    direction = dms_str[-1]
    if direction in ["N", "S", "E", "W"]:
        dms_str = dms_str[:-1].strip()
    else:
        direction = None
    dms_str = dms_str.replace("°", " ").replace("'", " ").replace('"', " ")
    dms_str = dms_str.replace("′", " ").replace("″", " ")
    parts = [p for p in dms_str.split() if p.strip()]
    if len(parts) == 3:
        degrees, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
    elif len(parts) == 2:
        degrees, minutes, seconds = float(parts[0]), float(parts[1]), 0
    elif len(parts) == 1:
        return float(parts[0])
    else:
        return None
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ["S", "W"]:
        decimal = -decimal
    return decimal


def load_master_data():
    file_path = os.path.join(Config.MASTER_PATH, "Master_Dataset.csv")
    print(f"  Loading master data from: {file_path}")
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[~df["Area_Code"].isin(Config.EXCLUDED_AREAS)].copy()
    print(f"  ✅ Loaded: {len(df):,} rows")
    print(f"     Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def load_models():
    """Loads the {'model','scaler','features',...} dict saved per area."""
    model_path = Config.MODELS_PATH
    print(f"  Loading models from: {model_path}")
    models = {}
    if os.path.exists(model_path):
        model_files = glob.glob(os.path.join(model_path, "*.pkl"))
        for model_file in model_files:
            area_code = int(os.path.basename(model_file).replace("_LightGBM.pkl", ""))
            if area_code not in Config.EXCLUDED_AREAS:
                with open(model_file, "rb") as f:
                    models[area_code] = pickle.load(f)
        print(f"  ✅ Loaded: {len(models)} models")
    else:
        print(f"  ⚠️ Models folder not found: {model_path}")
    return models


def load_feature_manifest():
    manifest_path = os.path.join(Config.MASTER_PATH, "Feature_Manifest.csv")
    print(f"  Loading feature manifest from: {manifest_path}")
    if os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path).sort_values("Position_Index")
        features = manifest["Feature"].tolist()
        print(f"  ✅ Loaded: {len(features)} features")
        return features
    print("  ⚠️ Feature manifest not found - models cannot be scored reliably")
    return []


def load_coordinates():
    print(f"  Loading coordinates from: {Config.COORDS_PATH}")
    df = safe_file_load(Config.COORDS_PATH)
    if df is None:
        print("  ⚠️ Could not load coordinates")
        return pd.DataFrame()

    area_col = next((c for c in df.columns if "area" in c.lower() or "code" in c.lower()), None)
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)

    if not area_col or not lat_col or not lon_col:
        print(f"  ⚠️ Could not find coordinate columns. Available: {df.columns.tolist()}")
        return pd.DataFrame()

    coords = df[[area_col, lat_col, lon_col]].copy()
    coords.columns = ["Area_Code", "Latitude_Original", "Longitude_Original"]

    print("  Converting DMS to decimal...")
    coords["Latitude"] = coords["Latitude_Original"].apply(dms_to_decimal)
    coords["Longitude"] = coords["Longitude_Original"].apply(dms_to_decimal)
    coords = coords.dropna(subset=["Latitude", "Longitude"])

    coords = coords[["Area_Code", "Latitude", "Longitude"]].copy()
    coords = coords[~coords["Area_Code"].isin(Config.EXCLUDED_AREAS)].copy()
    print(f"  ✅ Final: {len(coords)} areas with valid coordinates")
    return coords


# =============================================================================
# WEATHER FETCHING (unchanged - forecast mode only)
# =============================================================================

def fetch_open_meteo(lat, lon, start_date, end_date, mode):
    if mode == "archive":
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "daily": Config.OPEN_METEO_DAILY_VARS,
            "timezone": "Asia/Kolkata",
        }
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": Config.OPEN_METEO_DAILY_VARS,
            "forecast_days": Config.FORECAST_HORIZON_DAYS,
            "timezone": "Asia/Kolkata",
        }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("daily", {})
        if not data or "time" not in data:
            return pd.DataFrame()
        df = pd.DataFrame({"date": pd.to_datetime(data["time"])})
        for om_col, out_col in Config.OPEN_METEO_COLUMN_MAP.items():
            df[out_col] = data.get(om_col, np.nan)
        return df
    except Exception as e:
        print(f"    [WARN] Open-Meteo {mode} fetch failed for lat={lat}, lon={lon}: {e}")
        return pd.DataFrame()


def fetch_weather_for_areas(area_list, coords_df, dates):
    print(f"\n  Fetching weather for {len(area_list)} areas...")
    forecast_frames = []
    for i, area in enumerate(area_list):
        area_coords = coords_df[coords_df["Area_Code"] == area]
        if len(area_coords) == 0:
            continue
        lat = area_coords.iloc[0]["Latitude"]
        lon = area_coords.iloc[0]["Longitude"]
        fc_df = fetch_open_meteo(lat, lon, dates["forecast_start"], dates["forecast_end"], mode="forecast")
        if len(fc_df) > 0:
            fc_df["Area_Code"] = area
            forecast_frames.append(fc_df)
        if (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(area_list)} areas queried")

    forecast_weather_df = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()
    if len(forecast_weather_df) > 0:
        forecast_weather_df = forecast_weather_df.rename(columns={"date": "Date"})
    print(f"\n  ✅ Forecast weather: {len(forecast_weather_df):,} rows")
    return forecast_weather_df


# =============================================================================
# WEATHER FEATURE ENGINEERING (identical logic to the training script)
# =============================================================================

def engineer_weather_features(df):
    if len(df) == 0:
        return df
    df = df.copy().sort_values(["Area_Code", "Date"])
    g = df.groupby("Area_Code")

    if "Rainfall_mm" not in df.columns:
        df["Rainfall_mm"] = df.get("total_rain_mm", 0.0)
    df["Precipitation_Hours"] = df.get("precipitation_hours", 0.0)
    df["Max_Temp_C"] = df.get("max_temp_c", 25.0)
    df["Min_Temp_C"] = df.get("min_temp_c", 15.0)
    df["Mean_Temp_C"] = df.get("mean_temp_c", (df["Max_Temp_C"] + df["Min_Temp_C"]) / 2)
    df["Max_Wind_Speed_kmh"] = df.get("max_wind_speed_kmh", 0.0)
    df["Max_Wind_Gust_kmh"] = df.get("max_wind_gust_kmh", 0.0)

    df["Prev_Day_Rainfall_mm"] = g["Rainfall_mm"].shift(1).fillna(0)
    df["Rainfall_3day_Sum"] = g["Rainfall_mm"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df["Rainfall_7day_Sum"] = g["Rainfall_mm"].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df["Rainfall_15day_Sum"] = g["Rainfall_mm"].transform(lambda x: x.rolling(15, min_periods=1).sum())
    df["Rainfall_EMA_7"] = g["Rainfall_mm"].transform(lambda x: x.ewm(span=7, adjust=False).mean())
    df["Waterlogging_Index"] = (df["Rainfall_3day_Sum"] / 50).clip(0, 1)

    rain_flag = (df["Rainfall_mm"] > 0.5).astype(int)
    df["Consecutive_Rainy_Days"] = (
        rain_flag.groupby(df["Area_Code"]).transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumsum())
    )

    df["Rainfall_Intensity"] = (df["Rainfall_mm"] / (df["Precipitation_Hours"] + 1e-3)).fillna(0)
    df["Temp_Range"] = df["Max_Temp_C"] - df["Min_Temp_C"]

    df["Rain_Flag"] = (df["Rainfall_mm"] > 2.5).astype(int)
    df["Heavy_Rain_Flag"] = (df["Rainfall_mm"] > 35.0).astype(int)
    df["Heat_Stress_Flag"] = (df["Max_Temp_C"] > 42.0).astype(int)
    df["High_Wind_Flag"] = (df["Max_Wind_Speed_kmh"] > 60.0).astype(int)

    df["Month"] = df["Date"].dt.month
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Day_of_Month"] = df["Date"].dt.day
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["Is_Sunday"] = (df["Day_of_Week"] == 6).astype(int)
    df["Is_Monsoon_Season"] = df["Month"].isin([6, 7, 8, 9]).astype(int)
    df["Is_Quarter_End_Month"] = df["Month"].isin([3, 6, 9, 12]).astype(int)

    def _season(m):
        if m in [3, 4, 5]:
            return "Summer"
        if m in [6, 7, 8, 9]:
            return "Monsoon"
        if m in [10, 11]:
            return "Post_Monsoon"
        return "Winter"

    season = df["Month"].apply(_season)
    df["Season_Summer"] = (season == "Summer").astype(int)
    df["Season_Monsoon"] = (season == "Monsoon").astype(int)
    df["Season_Post_Monsoon"] = (season == "Post_Monsoon").astype(int)
    df["Season_Winter"] = (season == "Winter").astype(int)

    df["Rainfall_x_Monsoon"] = df["Rainfall_mm"] * df["Is_Monsoon_Season"]
    df["Rain_x_Temp"] = df["Rainfall_mm"] * df["Max_Temp_C"]
    df["Waterlogging_x_Monsoon"] = df["Waterlogging_Index"] * df["Is_Monsoon_Season"]
    df["Heat_x_Summer"] = df["Heat_Stress_Flag"] * df["Season_Summer"]
    df["Wind_x_Winter"] = df["High_Wind_Flag"] * df["Season_Winter"]

    return df


# =============================================================================
# RECURSIVE LAG / ROLLING FEATURE BUILDER
# =============================================================================
# NOTE: unlike training (where lag features are computed once over a full
# historical dataframe), forecasting has to build these ONE DAY AT A TIME,
# because Loss_Lag_1 / Prod_Lag_1 / etc. for day N depend on the (predicted)
# value for day N-1, which doesn't exist until we've predicted it.

def compute_lag_row(history, prod_col="ONDT_ACT", loss_col="Production_Loss_Pct",
                     offtake_loss_col="Offtake_Loss_Pct", offtake_act_col="Offtake_Actual"):
    """
    history: DataFrame sorted by Date, containing all days up to and
    including 'yesterday' (real + previously predicted), for ONE area.
    Returns a dict of lag/rolling feature values for the NEXT day.
    """
    feat = {}

    prod = history[prod_col] if prod_col in history.columns else pd.Series(dtype=float)
    loss = history[loss_col] if loss_col in history.columns else pd.Series(dtype=float)
    oft_loss = history[offtake_loss_col] if offtake_loss_col in history.columns else pd.Series(dtype=float)
    oft_act = history[offtake_act_col] if offtake_act_col in history.columns else pd.Series(dtype=float)

    def _lag(series, n):
        return float(series.iloc[-n]) if len(series) >= n else 0.0

    for lag in [1, 2, 3, 7, 14, 30]:
        feat[f"Prod_Lag_{lag}"] = _lag(prod, lag)
    for period in [3, 7, 14]:
        if len(prod) > period and prod.iloc[-(period + 1)] != 0:
            feat[f"Prod_Momentum_{period}"] = (prod.iloc[-1] - prod.iloc[-(period + 1)]) / prod.iloc[-(period + 1)]
        else:
            feat[f"Prod_Momentum_{period}"] = 0.0

    for lag in [1, 2, 3, 7, 14]:
        feat[f"Loss_Lag_{lag}"] = _lag(loss, lag)

    feat["Loss_7day_Avg"] = float(loss.tail(7).mean()) if len(loss) else 0.0
    feat["Loss_30day_Avg"] = float(loss.tail(30).mean()) if len(loss) else 0.0
    feat["Loss_7day_Std"] = float(loss.tail(7).std()) if len(loss) > 1 else 0.0

    for diff in [1, 3, 7]:
        feat[f"Loss_Change_{diff}"] = (loss.iloc[-1] - loss.iloc[-(diff + 1)]) if len(loss) > diff else 0.0

    if len(loss) > 2:
        feat["Loss_Acceleration"] = (loss.iloc[-1] - loss.iloc[-2]) - (loss.iloc[-2] - loss.iloc[-3])
    else:
        feat["Loss_Acceleration"] = 0.0

    for lag in [1, 2, 3, 7, 14]:
        feat[f"Offtake_Lag_{lag}"] = _lag(oft_loss, lag)
    feat["Offtake_7day_Avg"] = float(oft_loss.tail(7).mean()) if len(oft_loss) else 0.0
    feat["Offtake_30day_Avg"] = float(oft_loss.tail(30).mean()) if len(oft_loss) else 0.0
    feat["Offtake_7day_Std"] = float(oft_loss.tail(7).std()) if len(oft_loss) > 1 else 0.0

    for period in [3, 7, 14]:
        if len(oft_act) > period and oft_act.iloc[-(period + 1)] != 0:
            feat[f"Offtake_Momentum_{period}"] = (oft_act.iloc[-1] - oft_act.iloc[-(period + 1)]) / oft_act.iloc[-(period + 1)]
        else:
            feat[f"Offtake_Momentum_{period}"] = 0.0

    if len(prod) and len(oft_act):
        ratio = prod.iloc[-1] / (oft_act.iloc[-1] + 1e-3)
        gap = (loss.iloc[-1] if len(loss) else 0.0) - (oft_loss.iloc[-1] if len(oft_loss) else 0.0)
        feat["Prod_Offtake_Ratio"] = ratio
        feat["Prod_Offtake_Loss_Gap"] = gap
        # rolling avg of ratio/gap needs the historical series, approximate using last 7 rows
        ratio_series = (prod / (oft_act + 1e-3)).tail(7)
        gap_series = (loss - oft_loss).tail(7) if len(loss) == len(oft_loss) else pd.Series([gap])
        feat["Ratio_7day_Avg"] = float(ratio_series.mean())
        feat["Gap_7day_Avg"] = float(gap_series.mean())
    else:
        feat["Prod_Offtake_Ratio"] = 0.0
        feat["Prod_Offtake_Loss_Gap"] = 0.0
        feat["Ratio_7day_Avg"] = 0.0
        feat["Gap_7day_Avg"] = 0.0

    return feat


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_forecast():
    dates = Config.get_dates()

    print("=" * 80)
    print("DYNAMIC PRODUCTION LOSS FORECAST PIPELINE (LightGBM)")
    print("=" * 80)
    print(f"\n  Today                : {dates['today'].date()}")
    print(f"  Forecast window      : {dates['forecast_start'].date()} to {dates['forecast_end'].date()}")
    print("=" * 80)

    print("\n📂 LOADING DATA")
    print("-" * 80)
    master_df = load_master_data()
    models = load_models()
    features = load_feature_manifest()
    coords_df = load_coordinates()

    area_list = [a for a in models.keys() if a in master_df["Area_Code"].values]
    if len(coords_df) > 0:
        area_list = [a for a in area_list if a in coords_df["Area_Code"].values]
    print(f"\n  Areas available: {len(area_list)}")

    print("\n🌤️ FETCHING WEATHER")
    print("-" * 80)
    forecast_weather = fetch_weather_for_areas(area_list, coords_df, dates)

    print("\n🔧 ENGINEERING WEATHER FEATURES")
    print("-" * 80)
    forecast_weather_eng = engineer_weather_features(forecast_weather)
    print(f"  ✅ Weather engineered: {len(forecast_weather_eng)} rows")

    if len(forecast_weather_eng) == 0:
        print("\n❌ No weather data fetched.")
        return None

    print("\n📊 CALCULATING TARGETS")
    print("-" * 80)
    area_targets = {}
    for area in area_list:
        area_hist = master_df[master_df["Area_Code"] == area]
        if "ONDT_TARGET (Thousand Tonnes)" in area_hist.columns:
            mask = area_hist["ONDT_TARGET (Thousand Tonnes)"] > 0
            filtered = area_hist[mask].sort_values("Date").tail(90)
            area_targets[area] = filtered["ONDT_TARGET (Thousand Tonnes)"].mean() if len(filtered) else 0.0
        else:
            area_targets[area] = 0.0
    print(f"  ✅ Targets calculated for {len(area_targets)} areas")

    print("\n🤖 GENERATING LIGHTGBM FORECASTS (recursive, day-by-day)")
    print("-" * 80)

    history_cols = ["Date", "ONDT_ACT", "Production_Loss_Pct", "Offtake_Loss_Pct", "Offtake_Actual"]
    all_predictions = []
    skipped_no_model, skipped_no_target, skipped_no_history = 0, 0, 0

    for area in area_list:
        mdata = models.get(area)
        if mdata is None or "model" not in mdata or "scaler" not in mdata:
            skipped_no_model += 1
            continue

        area_model = mdata["model"]
        area_scaler = mdata["scaler"]
        area_features = mdata.get("features", features)

        daily_target = area_targets.get(area, 0.0)
        if daily_target <= 0:
            skipped_no_target += 1
            continue

        # Seed history with the last LAG_SEED_DAYS of REAL data for this area
        for col in history_cols:
            if col not in master_df.columns:
                master_df[col] = 0.0
        area_hist_full = (
            master_df[master_df["Area_Code"] == area][history_cols]
            .sort_values("Date")
            .tail(Config.LAG_SEED_DAYS)
            .reset_index(drop=True)
        )
        if len(area_hist_full) == 0:
            skipped_no_history += 1
            continue

        history = area_hist_full.copy()

        area_fc_wx = (
            forecast_weather_eng[
                (forecast_weather_eng["Area_Code"] == area)
                & (forecast_weather_eng["Date"] >= dates["forecast_start"])
            ]
            .sort_values("Date")
            .reset_index(drop=True)
        )
        if len(area_fc_wx) == 0:
            continue

        for _, wx_row in area_fc_wx.iterrows():
            lag_feat = compute_lag_row(history)

            feature_row = {}
            for feat_name in area_features:
                if feat_name in wx_row.index:
                    feature_row[feat_name] = wx_row[feat_name]
                elif feat_name in lag_feat:
                    feature_row[feat_name] = lag_feat[feat_name]
                else:
                    feature_row[feat_name] = 0.0

            X = pd.DataFrame([feature_row])[area_features].replace([np.inf, -np.inf], 0).fillna(0)
            X_scaled = area_scaler.transform(X)
            pred_loss = float(np.clip(area_model.predict(X_scaled)[0], 0, 100))
            predicted_production = daily_target * (1 - pred_loss / 100)

            all_predictions.append({
                "Area_Code": area,
                "Date": wx_row["Date"],
                "Predicted_Loss_Pct": round(pred_loss, 3),
                "Predicted_Production": round(predicted_production, 4),
                "Daily_Production_Target": daily_target,
                "Rainfall_mm": wx_row.get("Rainfall_mm", 0.0),
                "Max_Temp_C": wx_row.get("Max_Temp_C", 0.0),
            })

            # Feed this prediction back into history for the NEXT day's lag features.
            # Offtake has no future signal, so it's carried forward flat (last known value).
            new_row = {
                "Date": wx_row["Date"],
                "ONDT_ACT": predicted_production,
                "Production_Loss_Pct": pred_loss,
                "Offtake_Loss_Pct": history["Offtake_Loss_Pct"].iloc[-1] if len(history) else 0.0,
                "Offtake_Actual": history["Offtake_Actual"].iloc[-1] if len(history) else 0.0,
            }
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    df_predictions = pd.DataFrame(all_predictions)

    print(f"\n  Areas skipped (no model): {skipped_no_model}")
    print(f"  Areas skipped (no target): {skipped_no_target}")
    print(f"  Areas skipped (no history to seed lags): {skipped_no_history}")

    if len(df_predictions) > 0:
        print(f"\n  ✅ Generated {len(df_predictions):,} predictions")
        print(f"     Areas: {df_predictions['Area_Code'].nunique()}")
        print(f"     Date range: {df_predictions['Date'].min().date()} to {df_predictions['Date'].max().date()}")
    else:
        print("  ❌ No predictions generated!")
        return None

    print("\n💾 SAVING OUTPUTS")
    print("-" * 80)
    os.makedirs(Config.FORECAST_PATH, exist_ok=True)
    pred_file = os.path.join(Config.FORECAST_PATH, "Forecast_Predictions.csv")
    df_predictions.to_csv(pred_file, index=False)
    print(f"  ✅ {pred_file}")

    print("\n" + "=" * 80)
    print("✅ FORECAST PIPELINE COMPLETE (LightGBM-driven)")
    print("=" * 80)
    print("\n📌 Next step: Run 'python calculate_risk.py' to add risk scores")
    print("=" * 80)

    return df_predictions


if __name__ == "__main__":
    predictions = run_forecast()
