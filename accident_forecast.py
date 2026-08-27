# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import traceback


class Config:
    BASE_PATH = r"D:\OneDrive\Desktop\COAL PROJECT DEMO"
    WEATHER_PATH = os.path.join(BASE_PATH, "Weather_Forecasts", "Visual_Crossing_Weather_Historical_Forecast.csv")
    ROAD_DATA_PATH = os.path.join(BASE_PATH, "Preliminary Datasets", "Road_Material_Data.xlsx")
    OUTPUT_PATH = os.path.join(BASE_PATH, "Risk_Outputs")

    DEFAULT_SOIL = {
        "GC": {"LL": 35, "PL": 20, "PI": 15},
        "SC": {"LL": 30, "PL": 18, "PI": 12},
        "OL": {"LL": 60, "PL": 35, "PI": 25},
        "CH": {"LL": 70, "PL": 30, "PI": 40},
        "LAT": {"LL": 40, "PL": 22, "PI": 18},
        "OB_MIX": {"LL": 33, "PL": 19, "PI": 14},
        "default": {"LL": 33, "PL": 19, "PI": 14},
    }
    COALFIELD_SOIL_HINT = {}
    DRAIN_RATE_SOIL_MULTIPLIER = {}

    CONFIDENCE = {
        "road_slippery": "High",
        "lightning_strike": "High",
        "ballast_washout": "High",
        "wind_dust": "High",
        "wind_structural": "High",
        "water_inrush": "Low",
        "slope_critical": "Medium",
    }

    ROAD_SLIPPERY_SATURATION_MM = 100.0
    ROAD_SLIPPERY_CONSEC_MAX_BOOST = 0.15
    ROAD_SLIPPERY_CONSEC_BOOST_PER_DAY = 0.03

    FAST_RETAIN_BASE = 0.35
    FAST_RETAIN_MIN = 0.15
    FAST_RETAIN_MAX = 0.55

    SLOW_RETAIN_BASE = 0.85
    SLOW_RETAIN_MIN = 0.75
    SLOW_RETAIN_MAX = 0.95
    SLOW_BUCKET_SENSITIVITY = 0.4

    DECAY_HUMIDITY_REF_PCT = 60.0
    DECAY_WIND_REF_KMH = 10.0
    DECAY_SOLAR_REF_MJM2 = 15.0

    DRY_STREAK_ACCEL_DAYS = 2
    DRY_STREAK_ACCEL_MULT = 0.70

    MIN_PRECIP_CONFIDENCE_WEIGHT = 0.35
    LEAD_TIME_FULL_CONFIDENCE_DAYS = 3
    LEAD_TIME_ZERO_CONFIDENCE_DAYS = 15
    LEAD_TIME_MIN_CONFIDENCE_WEIGHT = 0.5

    PRESSURE_DROP_THRESHOLD_HPA = 4.0
    HUMIDITY_INSTABILITY_THRESHOLD_PCT = 70.0
    CLOUD_COVER_INSTABILITY_THRESHOLD_PCT = 80.0
    SOLAR_HEATING_THRESHOLD_MJM2 = 20.0

    FOG_DEW_POINT_DEPRESSION_C = 2.5
    DUST_SUPPRESSION_RAIN_MM = 5.0

    RAIN_INTENSITY_HEAVY_MM = 35.0
    RAIN_INTENSITY_EXTREME_MM = 65.0
    BURST_WINDOW_FAST_DAYS = 3
    BURST_WINDOW_SLOW_DAYS = 15

    WIND_STRUCTURAL_SUSTAINED_MULTIPLIER = 1.3
    WIND_STRUCTURAL_PRESSURE_BOOST_MULT = 1.10

    WATER_INRUSH_SURGE_RATIO = 0.5

    WET_DRY_CYCLE_LOOKBACK_DAYS = 15
    WET_DRY_CYCLE_BOOST_THRESHOLD = 4
    WET_DRY_CYCLE_MAX_BOOST_LEVELS = 1

    RISK_INDEX_NAMES = [
        "road_slippery", "lightning_strike", "ballast_washout",
        "wind_dust", "wind_structural", "water_inrush", "slope_critical"
    ]
    RISK_INDEX_LABELS = {
        "road_slippery": "Road Slippery",
        "lightning_strike": "Lightning Strike",
        "ballast_washout": "Ballast Washout",
        "wind_dust": "Wind-Dust Visibility",
        "wind_structural": "Wind-Structural",
        "water_inrush": "Water Inrush",
        "slope_critical": "Slope-Critical",
    }


def _num(row, key, default=0.0):
    val = row.get(key, default)
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def load_weather_data():
    print(f"  Loading weather: {Config.WEATHER_PATH}")
    if not os.path.exists(Config.WEATHER_PATH):
        print("  Weather file not found. Run the weather fetcher first.")
        return None
    df = pd.read_csv(Config.WEATHER_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    if "Forecast_Type" not in df.columns:
        df["Forecast_Type"] = "Forecast"
    n_hist = (df["Forecast_Type"] == "Historical").sum()
    n_fcst = (df["Forecast_Type"] == "Forecast").sum()
    print(f"  Loaded {len(df):,} rows, {df['Area_Code'].nunique()} areas "
          f"({n_hist:,} historical / {n_fcst:,} forecast)")
    return df


def load_road_data():
    print(f"  Loading road data: {Config.ROAD_DATA_PATH}")
    if not os.path.exists(Config.ROAD_DATA_PATH):
        print("  Road data not found - using default soil properties.")
        return None
    df = pd.read_excel(Config.ROAD_DATA_PATH)
    print(f"  Loaded {len(df)} road segments")
    return df


def get_soil_properties(classification, road_df=None):
    if road_df is not None and "Soil_Classification" in road_df.columns:
        match = road_df[road_df["Soil_Classification"] == classification]
        if not match.empty:
            row = match.iloc[0]
            LL = row.get("Liquid_Limit_LL")
            PL = row.get("Plastic_Limit_PL")
            if pd.notna(LL) and pd.notna(PL):
                return {"LL": float(LL), "PL": float(PL), "PI": float(LL - PL)}
    return Config.DEFAULT_SOIL.get(classification, Config.DEFAULT_SOIL["default"]).copy()


def _bucket_recurrence(precip: np.ndarray, retain: np.ndarray) -> np.ndarray:
    n = len(precip)
    out = np.zeros(n)
    level = 0.0
    for t in range(n):
        level = level * retain[t] + precip[t]
        out[t] = level
    return out


def _consecutive_streak(flag: pd.Series) -> pd.Series:
    run_id = (flag != flag.shift()).cumsum()
    streak_len = flag.groupby(run_id).cumcount() + 1
    return streak_len * flag


def compute_bucket_features(df):
    df = df.copy().sort_values(["Area_Code", "Date"]).reset_index(drop=True)

    if "Precipitation_mm" not in df.columns:
        df["Precipitation_mm"] = 0.0
    df["Precipitation_mm"] = df["Precipitation_mm"].fillna(0.0)

    if "Precipitation_Prob_pct" in df.columns:
        prob = df["Precipitation_Prob_pct"].fillna(100.0).clip(0, 100)
    else:
        prob = pd.Series(100.0, index=df.index)
    confidence_weight = np.sqrt(prob / 100.0).clip(lower=Config.MIN_PRECIP_CONFIDENCE_WEIGHT)
    confidence_weight = confidence_weight.where(df["Forecast_Type"] == "Forecast", 1.0)
    df["Precip_Confidence_Weight"] = confidence_weight
    df["Precipitation_Effective_mm"] = df["Precipitation_mm"] * confidence_weight

    today = pd.Timestamp(datetime.now().date())
    lead_time = (df["Date"] - today).dt.days.clip(lower=0)
    df["Lead_Time_Days"] = lead_time
    span = max(Config.LEAD_TIME_ZERO_CONFIDENCE_DAYS - Config.LEAD_TIME_FULL_CONFIDENCE_DAYS, 1)
    raw_ramp = 1.0 - (lead_time - Config.LEAD_TIME_FULL_CONFIDENCE_DAYS).clip(lower=0) / span
    non_precip_conf = raw_ramp.clip(lower=Config.LEAD_TIME_MIN_CONFIDENCE_WEIGHT, upper=1.0)
    non_precip_conf = non_precip_conf.where(df["Forecast_Type"] == "Forecast", 1.0)
    df["Non_Precip_Confidence_Weight"] = non_precip_conf

    humidity = df.get("Relative_Humidity_pct", pd.Series(Config.DECAY_HUMIDITY_REF_PCT, index=df.index))
    humidity = humidity.fillna(Config.DECAY_HUMIDITY_REF_PCT)
    wind = df.get("Wind_Speed_kmh", pd.Series(Config.DECAY_WIND_REF_KMH, index=df.index))
    wind = wind.fillna(Config.DECAY_WIND_REF_KMH)
    if "Solar_Radiation_MJm2" in df.columns:
        solar = df["Solar_Radiation_MJm2"].fillna(Config.DECAY_SOLAR_REF_MJM2)
    elif "Solar_Energy_MJm2" in df.columns:
        solar = df["Solar_Energy_MJm2"].fillna(Config.DECAY_SOLAR_REF_MJM2)
    else:
        solar = pd.Series(Config.DECAY_SOLAR_REF_MJM2, index=df.index)

    humidity_factor = 0.7 + 0.3 * (humidity / 100.0)
    wind_factor = (1 - (wind - Config.DECAY_WIND_REF_KMH) / 200.0).clip(0.85, 1.05)
    solar_factor = (1 - (solar - Config.DECAY_SOLAR_REF_MJM2) / 40.0).clip(0.85, 1.05)

    fast_retain = (Config.FAST_RETAIN_BASE * humidity_factor * wind_factor * solar_factor).clip(
        Config.FAST_RETAIN_MIN, Config.FAST_RETAIN_MAX
    )

    s = Config.SLOW_BUCKET_SENSITIVITY
    humidity_factor_slow = 1 + (humidity_factor - 1) * s
    wind_factor_slow = 1 + (wind_factor - 1) * s
    solar_factor_slow = 1 + (solar_factor - 1) * s
    slow_retain = (Config.SLOW_RETAIN_BASE * humidity_factor_slow * wind_factor_slow * solar_factor_slow).clip(
        Config.SLOW_RETAIN_MIN, Config.SLOW_RETAIN_MAX
    )

    g = df.groupby("Area_Code")

    df["Rainfall_3day_Sum"] = g["Precipitation_mm"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df["Rainfall_7day_Sum"] = g["Precipitation_mm"].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df["Rainfall_15day_Sum"] = g["Precipitation_mm"].transform(lambda x: x.rolling(15, min_periods=1).sum())

    df["Burst_Fast_mm"] = g["Precipitation_Effective_mm"].transform(
        lambda x: x.rolling(Config.BURST_WINDOW_FAST_DAYS, min_periods=1).max()
    )
    df["Burst_Slow_mm"] = g["Precipitation_Effective_mm"].transform(
        lambda x: x.rolling(Config.BURST_WINDOW_SLOW_DAYS, min_periods=1).max()
    )

    fast_bucket_parts, slow_bucket_parts = [], []
    for area, grp in g:
        idx = grp.index
        fast_retain_arr = fast_retain.loc[idx].to_numpy()
        slow_retain_arr = slow_retain.loc[idx].to_numpy()
        fast_bucket_parts.append(pd.Series(fast_retain_arr, index=idx))
        slow_bucket_parts.append(pd.Series(slow_retain_arr, index=idx))

    df["Fast_Retain_PreAccel"] = pd.concat(fast_bucket_parts).sort_index()
    df["Slow_Retain"] = pd.concat(slow_bucket_parts).sort_index()

    rain_flag = (df["Precipitation_mm"] > 0.5).astype(int)
    df["Rain_Flag"] = rain_flag
    df["Consecutive_Rainy_Days"] = rain_flag.groupby(df["Area_Code"]).transform(_consecutive_streak)
    dry_flag = 1 - rain_flag
    df["Consecutive_Dry_Days"] = dry_flag.groupby(df["Area_Code"]).transform(_consecutive_streak)

    accel_mask = df["Consecutive_Dry_Days"] >= Config.DRY_STREAK_ACCEL_DAYS
    df["Fast_Retain"] = np.where(
        accel_mask,
        (df["Fast_Retain_PreAccel"] * Config.DRY_STREAK_ACCEL_MULT).clip(lower=0.05),
        df["Fast_Retain_PreAccel"],
    )

    fast_bucket_final, slow_bucket_final = [], []
    for area, grp in df.groupby("Area_Code"):
        idx = grp.index
        precip_arr = grp["Precipitation_Effective_mm"].to_numpy()
        fast_r = grp["Fast_Retain"].to_numpy()
        slow_r = grp["Slow_Retain"].to_numpy()
        fast_bucket_final.append(pd.Series(_bucket_recurrence(precip_arr, fast_r), index=idx))
        slow_bucket_final.append(pd.Series(_bucket_recurrence(precip_arr, slow_r), index=idx))

    df["Fast_Bucket_mm"] = pd.concat(fast_bucket_final).sort_index()
    df["Slow_Bucket_mm"] = pd.concat(slow_bucket_final).sort_index()

    df["Prev_Day_Rainfall_mm"] = g["Precipitation_mm"].shift(1).fillna(0)

    prev_flag = df.groupby("Area_Code")["Rain_Flag"].shift(1).fillna(0)
    wet_start = ((df["Rain_Flag"] == 1) & (prev_flag == 0)).astype(int)
    df["_wet_start"] = wet_start
    df["Wet_Dry_Cycles_15d"] = df.groupby("Area_Code")["_wet_start"].transform(
        lambda x: x.rolling(Config.WET_DRY_CYCLE_LOOKBACK_DAYS, min_periods=1).sum()
    )
    df.drop(columns=["_wet_start"], inplace=True)

    if "Pressure_hPa" in df.columns:
        df["Pressure_Drop_24h_hPa"] = g["Pressure_hPa"].transform(lambda x: -x.diff(1)).fillna(0.0)
    else:
        df["Pressure_Drop_24h_hPa"] = 0.0

    if "Dew_Point_C" in df.columns and "Temperature_Mean_C" in df.columns:
        df["Dew_Point_Depression_C"] = (df["Temperature_Mean_C"] - df["Dew_Point_C"]).fillna(5.0)
    else:
        df["Dew_Point_Depression_C"] = 5.0

    return df


def calc_road_slippery(row, road_df=None):
    soil = row.get("Soil_Classification", "default")
    props = get_soil_properties(soil, road_df)
    LL, PL = props["LL"], props["PL"]
    PI = props["PI"] if props["PI"] else max(LL - PL, 1.0)

    fast_bucket = _num(row, "Fast_Bucket_mm", 0.0)
    consec = _num(row, "Consecutive_Rainy_Days", 0)
    cycles = _num(row, "Wet_Dry_Cycles_15d", 0)

    saturation_fraction = 1.0 - np.exp(-fast_bucket / Config.ROAD_SLIPPERY_SATURATION_MM)
    consec_boost = min(consec * Config.ROAD_SLIPPERY_CONSEC_BOOST_PER_DAY, Config.ROAD_SLIPPERY_CONSEC_MAX_BOOST)
    moisture_fraction = min(saturation_fraction + consec_boost, 1.0)
    moisture = PL + PI * moisture_fraction
    LI = (moisture - PL) / PI if PI > 0 else 0.0
    LI = max(0.0, min(LI, 1.2))

    if LI < 0.25:
        cat, score = "Low", 0
    elif LI < 0.50:
        cat, score = "Moderate", 1
    elif LI < 0.75:
        cat, score = "High", 2
    else:
        cat, score = "Severe", 3

    if cycles >= Config.WET_DRY_CYCLE_BOOST_THRESHOLD and score < 3:
        score = min(score + Config.WET_DRY_CYCLE_MAX_BOOST_LEVELS, 3)
        cat = ["Low", "Moderate", "High", "Severe"][score]

    return cat, score


def calc_lightning(row):
    flag = _num(row, "Thunderstorm_Flag", 0)
    severe_flag = _num(row, "Severe_Weather_Flag", 0)
    risk = _num(row, "Severe_Risk", 0)
    pressure_drop = _num(row, "Pressure_Drop_24h_hPa", 0.0)
    humidity = _num(row, "Relative_Humidity_pct", 0.0)
    cloud = _num(row, "Cloud_Cover_pct", 0.0)
    solar = _num(row, "Solar_Radiation_MJm2", _num(row, "Solar_Energy_MJm2", 0.0))

    pressure_humidity_instability = (
        pressure_drop >= Config.PRESSURE_DROP_THRESHOLD_HPA
        and humidity >= Config.HUMIDITY_INSTABILITY_THRESHOLD_PCT
    )
    cloud_support = cloud >= Config.CLOUD_COVER_INSTABILITY_THRESHOLD_PCT
    convective_heating = (
        solar >= Config.SOLAR_HEATING_THRESHOLD_MJM2
        and pressure_drop >= Config.PRESSURE_DROP_THRESHOLD_HPA
    )

    instability_boost = 0
    if pressure_humidity_instability:
        instability_boost += 2
        if cloud_support:
            instability_boost += 1
    if convective_heating:
        instability_boost += 1

    if flag == 0 and severe_flag == 0 and instability_boost == 0:
        return "Low", 0

    effective_risk = risk + instability_boost + (2 if severe_flag else 0)

    if flag == 1 and effective_risk == 0:
        return "Moderate", 1
    if effective_risk <= 4:
        return "Moderate", 1
    if effective_risk <= 7:
        return "High", 2
    return "Severe", 3


def calc_ballast_washout(row):
    fast_bucket = _num(row, "Fast_Bucket_mm", 0.0)
    peak = _num(row, "Burst_Fast_mm", 0.0)
    cycles = _num(row, "Wet_Dry_Cycles_15d", 0)

    if fast_bucket < 50:
        cat, score = "Low", 0
    elif fast_bucket < 100:
        cat, score = "Moderate", 1
    elif fast_bucket < 150:
        cat, score = "High", 2
    else:
        cat, score = "Severe", 3

    if peak >= Config.RAIN_INTENSITY_HEAVY_MM and score < 3:
        score += 1
        cat = ["Low", "Moderate", "High", "Severe"][score]

    if cycles >= Config.WET_DRY_CYCLE_BOOST_THRESHOLD and score < 3:
        score = min(score + Config.WET_DRY_CYCLE_MAX_BOOST_LEVELS, 3)
        cat = ["Low", "Moderate", "High", "Severe"][score]

    return cat, score


def calc_wind_dust(row):
    vis = _num(row, "Visibility_km", 10.0)
    wind = _num(row, "Wind_Speed_kmh", 0.0)
    dew_dep = _num(row, "Dew_Point_Depression_C", 5.0)
    fast_bucket = _num(row, "Fast_Bucket_mm", 0.0)
    non_precip_conf = _num(row, "Non_Precip_Confidence_Weight", 1.0)

    fog_likely = dew_dep < Config.FOG_DEW_POINT_DEPRESSION_C
    ground_wet = fast_bucket >= Config.DUST_SUPPRESSION_RAIN_MM
    dust_possible = wind > 25 and not ground_wet

    if vis < 2:
        return "Severe", 3
    if vis < 5:
        return "High", 2
    if vis <= 8:
        if dust_possible or fog_likely:
            if non_precip_conf < 0.7:
                return "Low", 0
            return "Moderate", 1
        return "Low", 0
    return "Low", 0


def calc_wind_structural(row):
    gust = _num(row, "Wind_Gust_kmh", 0.0)
    sustained = _num(row, "Wind_Speed_kmh", 0.0)
    pressure_drop = _num(row, "Pressure_Drop_24h_hPa", 0.0)

    estimated_gust = sustained * Config.WIND_STRUCTURAL_SUSTAINED_MULTIPLIER
    effective_gust = max(gust, estimated_gust)

    if pressure_drop >= Config.PRESSURE_DROP_THRESHOLD_HPA:
        effective_gust *= Config.WIND_STRUCTURAL_PRESSURE_BOOST_MULT

    if effective_gust < 40:
        return "Low", 0
    elif effective_gust < 60:
        return "Moderate", 1
    elif effective_gust < 70:
        return "High", 2
    else:
        return "Severe", 3


def calc_water_inrush(row):
    slow_bucket = _num(row, "Slow_Bucket_mm", 0.0)
    fast_bucket = _num(row, "Fast_Bucket_mm", 0.0)

    if slow_bucket < 100:
        cat, score = "Low", 0
    elif slow_bucket < 150:
        cat, score = "Moderate", 1
    elif slow_bucket < 250:
        cat, score = "High", 2
    else:
        cat, score = "Severe", 3

    surge_ratio = (fast_bucket / slow_bucket) if slow_bucket > 0 else 0.0
    if surge_ratio >= Config.WATER_INRUSH_SURGE_RATIO and slow_bucket >= 100 and score < 3:
        score += 1
        cat = ["Low", "Moderate", "High", "Severe"][score]

    return cat, score


def calc_slope_critical(row):
    slow_bucket = _num(row, "Slow_Bucket_mm", 0.0)
    consec = _num(row, "Consecutive_Rainy_Days", 0)
    peak = _num(row, "Burst_Slow_mm", 0.0)

    if slow_bucket < 50:
        cat, score = "Low", 0
    elif slow_bucket < 80:
        cat, score = "Moderate", 1
    elif slow_bucket < 100:
        cat, score = "Moderate", 1
    else:
        cat, score = "High", 2

    persistent = consec >= 3
    if persistent and score < 2:
        cat, score = "High", 2
    elif persistent and slow_bucket >= 150:
        cat, score = "Severe", 3

    if peak >= Config.RAIN_INTENSITY_EXTREME_MM and score < 3:
        cat, score = "Severe", 3

    return cat, score


def compute_composite(scores_dict, row=None):
    ordered_indices = Config.RISK_INDEX_NAMES
    scores = [scores_dict[idx][1] for idx in ordered_indices]

    avg = sum(scores) / len(scores)
    composite = 100.0 * avg / 3.0

    severe_indices = [idx for idx, (cat, _) in scores_dict.items() if cat == "Severe"]
    if len(severe_indices) >= 2:
        composite = min(100.0, composite + 10.0)
    if len(severe_indices) >= 3:
        composite = max(composite, 75.0)
    critical_severe = any(idx in severe_indices for idx in ["road_slippery", "slope_critical"])
    if critical_severe:
        composite = max(composite, 50.0)
    if row is not None and _num(row, "Severe_Weather_Flag", 0):
        composite = max(composite, 50.0)

    composite = min(100.0, max(0.0, composite))

    if composite < 25:
        cat = "Low"
    elif composite < 50:
        cat = "Moderate"
    elif composite < 75:
        cat = "High"
    else:
        cat = "Severe"

    return composite, cat


def get_weather_regime(scores_dict):
    water_scores = [
        scores_dict["road_slippery"][1], scores_dict["ballast_washout"][1],
        scores_dict["water_inrush"][1], scores_dict["slope_critical"][1]
    ]
    wind_scores = [scores_dict["wind_dust"][1], scores_dict["wind_structural"][1]]

    water_high = any(s >= 2 for s in water_scores)
    wind_high = any(s >= 2 for s in wind_scores)

    if water_high and wind_high:
        return "Stormy (Wet & Windy)"
    elif water_high and not wind_high:
        return "Wet & Calm"
    elif not water_high and wind_high:
        return "Dry & Windy"
    else:
        return "Mild Conditions"


def build_dashboard_payload(df_out):
    today = pd.Timestamp(datetime.now().date())

    index_names = Config.RISK_INDEX_NAMES
    latest = []
    timeseries = {}

    for area_code, grp in df_out.groupby("Area_Code"):
        grp = grp.sort_values("Date")

        series = []
        for _, row in grp.iterrows():
            point = {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "type": row["Forecast_Type"],
                "composite_score": None if pd.isna(row.get("Composite_Score")) else float(row["Composite_Score"]),
                "composite_category": row.get("Composite_Category"),
                "weather_regime": row.get("Weather_Regime"),
                "fast_bucket_mm": None if pd.isna(row.get("Fast_Bucket_mm")) else float(row["Fast_Bucket_mm"]),
                "slow_bucket_mm": None if pd.isna(row.get("Slow_Bucket_mm")) else float(row["Slow_Bucket_mm"]),
            }
            for name in index_names:
                point[f"{name}_score"] = int(row.get(f"{name}_score", 0))
                point[f"{name}_category"] = row.get(f"{name}_category")
            series.append(point)
        timeseries[str(int(area_code))] = series

        fcst_grp = grp[grp["Forecast_Type"] == "Forecast"]
        today_row = grp[grp["Date"] == today]
        if today_row.empty:
            today_row = fcst_grp.head(1) if len(fcst_grp) > 0 else grp.tail(1)

        if not today_row.empty:
            snap = today_row.iloc[0]
            indices_breakdown = {
                name: {
                    "label": Config.RISK_INDEX_LABELS[name],
                    "category": snap.get(f"{name}_category"),
                    "score": int(snap.get(f"{name}_score", 0)),
                    "confidence": snap.get(f"{name}_confidence"),
                }
                for name in index_names
            }
            entry = {
                "area_code": int(area_code),
                "date": snap["Date"].strftime("%Y-%m-%d"),
                "forecast_type": snap["Forecast_Type"],
                "composite_score": None if pd.isna(snap.get("Composite_Score")) else round(float(snap["Composite_Score"]), 1),
                "composite_category": snap.get("Composite_Category"),
                "weather_regime": snap.get("Weather_Regime"),
                "fast_bucket_mm": None if pd.isna(snap.get("Fast_Bucket_mm")) else round(float(snap["Fast_Bucket_mm"]), 1),
                "slow_bucket_mm": None if pd.isna(snap.get("Slow_Bucket_mm")) else round(float(snap["Slow_Bucket_mm"]), 1),
                "indices": indices_breakdown,
            }
            latest.append(entry)

        if len(fcst_grp) > 0:
            worst = fcst_grp.loc[fcst_grp["Composite_Score"].idxmax()]
            for entry in latest:
                if entry["area_code"] == int(area_code):
                    entry["forecast_peak"] = {
                        "date": worst["Date"].strftime("%Y-%m-%d"),
                        "composite_score": round(float(worst["Composite_Score"]), 1),
                        "composite_category": worst.get("Composite_Category"),
                    }

    latest.sort(key=lambda e: e.get("composite_score") or 0, reverse=True)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "risk_indices": [
            {"key": name, "label": Config.RISK_INDEX_LABELS[name], "confidence": Config.CONFIDENCE[name]}
            for name in index_names
        ],
        "latest": latest,
        "timeseries": timeseries,
    }
    return payload


def print_final_summary(df_out, success, error_msg=None, output_files=None):
    print("\n" + "=" * 80)
    print("FINAL EXECUTION SUMMARY")
    print("=" * 80)

    if success:
        print("  STATUS: SUCCESS")
    else:
        print("  STATUS: FAILED")
        print(f"  Error: {error_msg}")
        if df_out is None or len(df_out) == 0:
            print("  No output data was produced.")
            print("=" * 80)
            return

    if df_out is not None and len(df_out) > 0:
        n_rows = len(df_out)
        n_areas = df_out["Area_Code"].nunique()
        date_min = df_out["Date"].min()
        date_max = df_out["Date"].max()
        print(f"\n  DATA VOLUME:")
        print(f"     Total rows: {n_rows:,} (Historical + Forecast)")
        print(f"     Unique areas: {n_areas}")
        print(f"     Date range: {date_min.date()} to {date_max.date()}")

        fcst = df_out[df_out["Forecast_Type"] == "Forecast"]
        if len(fcst) > 0:
            print(f"\n  RISK DISTRIBUTION - FORECAST WINDOW ONLY (Composite Category):")
            for cat in ["Low", "Moderate", "High", "Severe"]:
                cnt = (fcst["Composite_Category"] == cat).sum()
                pct = 100 * cnt / len(fcst)
                print(f"     {cat:10s}: {cnt:5,} ({pct:.1f}%)")

            print(f"\n  TOP 5 HIGHEST RISK AREAS - FORECAST WINDOW (Average Composite Score):")
            top = fcst.groupby("Area_Code")["Composite_Score"].mean().sort_values(ascending=False).head(5)
            for area, score in top.items():
                print(f"     Area {area}: {score:.1f}")

    if output_files:
        print(f"\n  OUTPUT FILES:")
        for fname in output_files:
            exists = os.path.exists(fname)
            size = os.path.getsize(fname) if exists else 0
            status = "OK" if exists and size > 0 else "MISSING"
            print(f"     [{status}] {os.path.basename(fname)} ({size:,} bytes)")

    print("\n" + "=" * 80)


def main():
    start_time = datetime.now()
    print("=" * 80)
    print("CIL WEATHER-ACCIDENT RISK SCORING (WIND & WATER ONLY) - v5 (dashboard-ready)")
    print("=" * 80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    success = False
    error_msg = None
    df_out = None

    try:
        os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

        print("\nLOADING DATA")
        print("-" * 80)
        weather = load_weather_data()
        if weather is None:
            raise Exception("Weather data could not be loaded.")
        road_df = load_road_data()

        print("\nCOMPUTING TWO-BUCKET MOISTURE FEATURES")
        print("-" * 80)
        weather = compute_bucket_features(weather)
        print("  Fast/slow leaky-bucket moisture levels computed across Historical+Forecast timeline.")

        print("\nCALCULATING RISK INDICES (Wind & Water only, all 7 hardened)")
        print("-" * 80)

        results = []
        total = len(weather)

        for idx, row in weather.iterrows():
            if idx % 100 == 0:
                print(f"  Processing: {idx+1:,}/{total:,} rows")

            indices = {
                "road_slippery": calc_road_slippery(row, road_df),
                "lightning_strike": calc_lightning(row),
                "ballast_washout": calc_ballast_washout(row),
                "wind_dust": calc_wind_dust(row),
                "wind_structural": calc_wind_structural(row),
                "water_inrush": calc_water_inrush(row),
                "slope_critical": calc_slope_critical(row),
            }

            comp_score, comp_cat = compute_composite(indices, row)
            regime = get_weather_regime(indices)

            record = {
                "Area_Code": row.get("Area_Code"),
                "Date": row.get("Date"),
                "Forecast_Type": row.get("Forecast_Type", "Forecast"),
                "Lead_Time_Days": row.get("Lead_Time_Days"),
                "Temperature_Max_C": row.get("Temperature_Max_C"),
                "Temperature_Mean_C": row.get("Temperature_Mean_C"),
                "Relative_Humidity_pct": row.get("Relative_Humidity_pct"),
                "Precipitation_mm": row.get("Precipitation_mm"),
                "Precipitation_Prob_pct": row.get("Precipitation_Prob_pct"),
                "Precip_Confidence_Weight": round(_num(row, "Precip_Confidence_Weight", 1.0), 3),
                "Non_Precip_Confidence_Weight": round(_num(row, "Non_Precip_Confidence_Weight", 1.0), 3),
                "Fast_Retain": round(_num(row, "Fast_Retain"), 3),
                "Slow_Retain": round(_num(row, "Slow_Retain"), 3),
                "Fast_Bucket_mm": round(_num(row, "Fast_Bucket_mm"), 2),
                "Slow_Bucket_mm": round(_num(row, "Slow_Bucket_mm"), 2),
                "Burst_Fast_mm": round(_num(row, "Burst_Fast_mm"), 2),
                "Burst_Slow_mm": round(_num(row, "Burst_Slow_mm"), 2),
                "Rainfall_3day_Sum": row.get("Rainfall_3day_Sum"),
                "Rainfall_7day_Sum": row.get("Rainfall_7day_Sum"),
                "Rainfall_15day_Sum": row.get("Rainfall_15day_Sum"),
                "Consecutive_Rainy_Days": row.get("Consecutive_Rainy_Days"),
                "Consecutive_Dry_Days": row.get("Consecutive_Dry_Days"),
                "Wet_Dry_Cycles_15d": round(_num(row, "Wet_Dry_Cycles_15d"), 1),
                "Pressure_Drop_24h_hPa": round(_num(row, "Pressure_Drop_24h_hPa"), 2),
                "Dew_Point_Depression_C": round(_num(row, "Dew_Point_Depression_C"), 2),
                "Severe_Risk": row.get("Severe_Risk", 0),
                "Thunderstorm_Flag": row.get("Thunderstorm_Flag", 0),
                "Severe_Weather_Flag": row.get("Severe_Weather_Flag", 0),
                "Solar_Radiation_MJm2": row.get("Solar_Radiation_MJm2"),
                "Wind_Speed_kmh": row.get("Wind_Speed_kmh"),
                "Wind_Gust_kmh": row.get("Wind_Gust_kmh"),
                "Visibility_km": row.get("Visibility_km"),
                "Weather_Regime": regime,
            }

            for name, (cat, score) in indices.items():
                record[f"{name}_category"] = cat
                record[f"{name}_score"] = score
                record[f"{name}_confidence"] = Config.CONFIDENCE.get(name, "Medium")

            record["Composite_Score"] = round(comp_score, 2)
            record["Composite_Category"] = comp_cat

            results.append(record)

        df_out = pd.DataFrame(results)
        print(f"\n  Calculated {len(df_out):,} risk records.")

        print("\nSAVING OUTPUTS")
        print("-" * 80)

        main_file = os.path.join(Config.OUTPUT_PATH, "Weather_Accident_Risk_Scores.csv")
        df_out.to_csv(main_file, index=False)
        print(f"  Main scores (Historical + Forecast): {main_file}")

        fcst_out = df_out[df_out["Forecast_Type"] == "Forecast"]
        if len(fcst_out) > 0:
            summary = fcst_out.groupby("Area_Code").agg({
                "Composite_Score": ["mean", "max", "std"],
                "Composite_Category": lambda x: x.mode()[0] if len(x) > 0 else "Low",
            })
            summary.columns = ["Composite_Avg", "Composite_Max", "Composite_Std", "Most_Common_Category"]
        else:
            summary = pd.DataFrame()
        summary_file = os.path.join(Config.OUTPUT_PATH, "Risk_Summary_Stats.csv")
        summary.to_csv(summary_file)
        print(f"  Summary stats (Forecast window only): {summary_file}")

        limitations = pd.DataFrame([
            {"Index": "Road Slippery", "Confidence": "High",
             "Limitation": "Uses a FAST leaky-bucket moisture proxy (not measured); drain rate is a uniform "
                            "placeholder from humidity/wind/solar, not yet soil-type or coal-material specific. "
                            "Soil LL/PL/PI falls back to a coarse OB_MIX/CH/LAT placeholder table, not per-mine "
                            "lab data. Wet-dry cycle (slaking) boost is a heuristic, not a measured degradation model."},
            {"Index": "Lightning Strike", "Confidence": "High",
             "Limitation": "Pressure-drop/humidity/cloud/solar-heating instability signal is a heuristic, not a "
                            "numerical weather model."},
            {"Index": "Ballast Washout", "Confidence": "High",
             "Limitation": "Uses the same FAST bucket as Road Slippery; burst-intensity and wet-dry-cycle "
                            "escalations use fixed thresholds, not per-segment drainage capacity or ballast material data."},
            {"Index": "Wind-Dust Visibility", "Confidence": "High",
             "Limitation": "Fog vs. dust distinction uses dew point depression and the FAST bucket as proxies, not "
                            "direct observation. Long-lead-time forecasts get a confidence-based downgrade on "
                            "borderline calls only, not a full recalibration."},
            {"Index": "Wind-Structural", "Confidence": "High",
             "Limitation": "Sustained-wind fallback estimates gust when the API gust field is missing/implausible; "
                            "pressure-drop boost is a heuristic; not a substitute for site anemometry."},
            {"Index": "Water Inrush", "Confidence": "Low",
             "Limitation": "Uses the SLOW leaky-bucket as a rainfall-only proxy for subsurface saturation - missing "
                            "actual subsurface hydrology/old-workings data. Surge detection compares FAST vs SLOW "
                            "bucket levels, still rainfall-only."},
            {"Index": "Slope-Critical", "Confidence": "Medium",
             "Limitation": "Uses the SLOW leaky-bucket as a rainfall-only proxy - missing slope angle/geotechnical "
                            "data per mine. Burst-intensity override uses a fixed threshold, not per-slope stability analysis."},
        ])
        lim_file = os.path.join(Config.OUTPUT_PATH, "Confidence_Documentation.csv")
        limitations.to_csv(lim_file, index=False)
        print(f"  Confidence documentation: {lim_file}")

        dashboard_payload = build_dashboard_payload(df_out)
        dashboard_file = os.path.join(Config.OUTPUT_PATH, "risk_dashboard_data.json")
        with open(dashboard_file, "w", encoding="utf-8") as f:
            json.dump(dashboard_payload, f, ensure_ascii=False)
        print(f"  Dashboard payload: {dashboard_file}")

        latest_flat = []
        for entry in dashboard_payload["latest"]:
            flat = {
                "area_code": entry["area_code"],
                "date": entry["date"],
                "forecast_type": entry["forecast_type"],
                "composite_score": entry["composite_score"],
                "composite_category": entry["composite_category"],
                "weather_regime": entry["weather_regime"],
                "fast_bucket_mm": entry["fast_bucket_mm"],
                "slow_bucket_mm": entry["slow_bucket_mm"],
            }
            for name, info in entry["indices"].items():
                flat[f"{name}_category"] = info["category"]
                flat[f"{name}_score"] = info["score"]
            if "forecast_peak" in entry:
                flat["forecast_peak_date"] = entry["forecast_peak"]["date"]
                flat["forecast_peak_score"] = entry["forecast_peak"]["composite_score"]
                flat["forecast_peak_category"] = entry["forecast_peak"]["composite_category"]
            latest_flat.append(flat)
        latest_file = os.path.join(Config.OUTPUT_PATH, "risk_dashboard_latest.csv")
        pd.DataFrame(latest_flat).to_csv(latest_file, index=False)
        print(f"  Dashboard latest snapshot: {latest_file}")

        print("\nSAMPLE OUTPUT - FORECAST WINDOW (first 5 rows)")
        print("-" * 80)
        sample_cols = ["Area_Code", "Date", "Weather_Regime", "Fast_Bucket_mm", "Slow_Bucket_mm",
                       "Composite_Score", "Composite_Category"]
        if len(fcst_out) > 0:
            print(fcst_out[sample_cols].head(5).to_string(index=False))
        else:
            print("  (No forecast rows in output)")

        success = True

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}")
        print(traceback.format_exc())
        success = False

    finally:
        output_files = [
            os.path.join(Config.OUTPUT_PATH, "Weather_Accident_Risk_Scores.csv"),
            os.path.join(Config.OUTPUT_PATH, "Risk_Summary_Stats.csv"),
            os.path.join(Config.OUTPUT_PATH, "Confidence_Documentation.csv"),
            os.path.join(Config.OUTPUT_PATH, "risk_dashboard_data.json"),
            os.path.join(Config.OUTPUT_PATH, "risk_dashboard_latest.csv"),
        ] if os.path.exists(Config.OUTPUT_PATH) else None

        print_final_summary(df_out, success, error_msg, output_files)
        elapsed = datetime.now() - start_time
        print(f"\nTotal execution time: {elapsed.total_seconds():.1f} seconds")
        print("=" * 80)


if __name__ == "__main__":
    main()
