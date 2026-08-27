# -*- coding: utf-8 -*-
"""
Coal India - Production Loss Prediction Model
==============================================
LightGBM per-area model predicting daily production loss %.
Features: weather (engineered) + lag/rolling + offtake cross-signal.
Train: Synthetic FY2020-23 + Real FY2023  |  Test: Real FY2024
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# CONFIGURATION - CHANGED FOR LOCAL VS CODE
# =============================================================================

class Config:
    """Configuration management with robust path handling"""

    # LOCAL PATHS - Changed from Colab
    BASE_PATH = "."  # Current directory (COAL PROJECT folder)

    # Subdirectories
    OUTPUT_DIR = "Model_Outputs"
    MODELS_DIR = "Models"
    SYNTH_DIR = "Synthetic_Outputs"

    # File paths - CHANGED for local
    SYNTH_FILE = "Synthetic_Production_Offtake_FY2020_FY2023.csv"
    REAL_FILE = "Preliminary Datasets/Fixed Production Dispatch Dataset.xlsx"
    WEATHER_FILES = [
        "Preliminary Datasets/area_hq_weather_daily_FY2023-24.csv",
        "Preliminary Datasets/area_hq_weather_daily_FY2024-25.csv"
    ]

    # Model parameters
    EXCLUDED_AREAS = [5190, 6003, 8351, 4151, 4251]
    MIN_TRAIN_ROWS = 30
    MIN_TEST_ROWS = 5
    REAL_WEIGHT = 3.0

    # Risk scoring weights
    RISK_WEIGHTS = {
        "loss": 0.35,
        "rainfall": 0.15,
        "waterlogging": 0.12,
        "consecutive": 0.10,
        "heat": 0.08,
        "offtake": 0.20,
    }

def setup_directories():
    """Create necessary directories"""
    paths = {}

    # Use current directory as base
    base_path = os.getcwd()
    print(f"  Base directory: {base_path}")

    # Create output directories
    output_dir = os.path.join(base_path, Config.OUTPUT_DIR)
    models_dir = os.path.join(output_dir, Config.MODELS_DIR)
    synth_dir = os.path.join(base_path, Config.SYNTH_DIR)

    for directory in [output_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created/verified: {directory}")

    paths['base'] = base_path
    paths['output'] = output_dir
    paths['models'] = models_dir
    paths['synth'] = synth_dir

    return paths

# Setup paths
PATHS = setup_directories()

# File paths with proper joining
SYNTH_PATH = os.path.join(PATHS['synth'], Config.SYNTH_FILE)
REAL_PATH = Config.REAL_FILE  # Already has path
WEATHER_PATHS = Config.WEATHER_FILES  # Already have paths

# Output paths
OUTPUT_DIR = PATHS['output']
MODELS_DIR = PATHS['models']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_file_load(filepath, **kwargs):
    """Safely load a file with error handling"""
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath, **kwargs)
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
    except FileNotFoundError:
        print(f"  ⚠️ File not found: {filepath}")
        return None
    except Exception as e:
        print(f"  ⚠️ Error loading {filepath}: {e}")
        return None

def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all weather-derived columns used as model features"""
    df = df.copy().sort_values(["Area_Code", "Date"])
    g = df.groupby("Area_Code")

    # Core renames
    df["Rainfall_mm"] = df["total_precipitation_mm"]
    df["Precipitation_Hours"] = df["precipitation_hours"]
    df["Max_Temp_C"] = df["max_temp_c"]
    df["Min_Temp_C"] = df["min_temp_c"]
    df["Mean_Temp_C"] = df["mean_temp_c"]
    df["Max_Wind_Speed_kmh"] = df["max_wind_speed_kmh"]
    df["Max_Wind_Gust_kmh"] = df["max_wind_gust_kmh"]

    # Rainfall rolling aggregates
    df["Prev_Day_Rainfall_mm"] = g["Rainfall_mm"].shift(1).fillna(0)
    df["Rainfall_3day_Sum"] = g["Rainfall_mm"].transform(
        lambda x: x.rolling(3, min_periods=1).sum())
    df["Rainfall_7day_Sum"] = g["Rainfall_mm"].transform(
        lambda x: x.rolling(7, min_periods=1).sum())
    df["Rainfall_15day_Sum"] = g["Rainfall_mm"].transform(
        lambda x: x.rolling(15, min_periods=1).sum())
    df["Rainfall_EMA_7"] = g["Rainfall_mm"].transform(
        lambda x: x.ewm(span=7, adjust=False).mean())

    # Waterlogging index
    df["Waterlogging_Index"] = (df["Rainfall_3day_Sum"] / 50).clip(0, 1)

    # Consecutive rainy days
    rain_flag = (df["Rainfall_mm"] > 0.5).astype(int)
    df["Consecutive_Rainy_Days"] = (
        rain_flag
        .groupby(df["Area_Code"])
        .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumsum())
    )

    # Intensity and temperature range
    df["Rainfall_Intensity"] = (
        df["Rainfall_mm"] / (df["Precipitation_Hours"] + 1e-3)
    ).fillna(0)
    df["Temp_Range"] = df["Max_Temp_C"] - df["Min_Temp_C"]

    # Binary weather flags
    df["Rain_Flag"] = (df["Rainfall_mm"] > 2.5).astype(int)
    df["Heavy_Rain_Flag"] = (df["Rainfall_mm"] > 35.0).astype(int)
    df["Heat_Stress_Flag"] = (df["Max_Temp_C"] > 42.0).astype(int)
    df["High_Wind_Flag"] = (df["Max_Wind_Speed_kmh"] > 60.0).astype(int)

    # Calendar features
    df["Month"] = df["Date"].dt.month
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Day_of_Month"] = df["Date"].dt.day
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["Is_Sunday"] = (df["Day_of_Week"] == 6).astype(int)
    df["Is_Monsoon_Season"] = df["Month"].isin([6, 7, 8, 9]).astype(int)
    df["Is_Quarter_End_Month"] = df["Month"].isin([3, 6, 9, 12]).astype(int)

    # Season dummies
    def _season(m):
        if m in [3, 4, 5]: return "Summer"
        if m in [6, 7, 8, 9]: return "Monsoon"
        if m in [10, 11]: return "Post_Monsoon"
        return "Winter"

    season = df["Month"].apply(_season)
    df["Season_Summer"] = (season == "Summer").astype(int)
    df["Season_Monsoon"] = (season == "Monsoon").astype(int)
    df["Season_Post_Monsoon"] = (season == "Post_Monsoon").astype(int)
    df["Season_Winter"] = (season == "Winter").astype(int)

    # Interaction features
    df["Rainfall_x_Monsoon"] = df["Rainfall_mm"] * df["Is_Monsoon_Season"]
    df["Rain_x_Temp"] = df["Rainfall_mm"] * df["Max_Temp_C"]
    df["Waterlogging_x_Monsoon"] = df["Waterlogging_Index"] * df["Is_Monsoon_Season"]
    df["Heat_x_Summer"] = df["Heat_Stress_Flag"] * df["Season_Summer"]
    df["Wind_x_Winter"] = df["High_Wind_Flag"] * df["Season_Winter"]

    return df

def calc_loss(df, target_col="ONDT_TARGET (Thousand Tonnes)", actual_col="ONDT_ACT"):
    """Calculate production loss percentage"""
    mask = df[target_col] > 0
    loss = pd.Series(0.0, index=df.index)
    loss.loc[mask] = (
        (df.loc[mask, target_col] - df.loc[mask, actual_col])
        / df.loc[mask, target_col] * 100
    )
    return loss.clip(0, 100).fillna(0)

def get_fy(date):
    """Get financial year from date"""
    return date.year if date.month >= 4 else date.year - 1

def engineer_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add production lag, rolling-window, and offtake cross-features"""
    df = df.copy().sort_values(["Area_Code", "Date"]).reset_index(drop=True)
    g = df.groupby("Area_Code")

    # Identify production volume column
    prod_vol_col = (
        "ONDT_ACT" if "ONDT_ACT" in df.columns
        else "Daily_Production_Actual" if "Daily_Production_Actual" in df.columns
        else None
    )

    # Production volume lags
    if prod_vol_col:
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f"Prod_Lag_{lag}"] = g[prod_vol_col].shift(lag).fillna(0)
        for period in [3, 7, 14]:
            df[f"Prod_Momentum_{period}"] = (
                g[prod_vol_col].pct_change(periods=period).fillna(0)
                .replace([np.inf, -np.inf], 0)
            )
    else:
        for col in (
            [f"Prod_Lag_{l}" for l in [1,2,3,7,14,30]]
            + [f"Prod_Momentum_{p}" for p in [3,7,14]]
        ):
            df[col] = 0.0

    # Production loss lags & rolling stats
    lc = "Production_Loss_Pct"
    for lag in [1, 2, 3, 7, 14]:
        df[f"Loss_Lag_{lag}"] = g[lc].shift(lag).fillna(0)

    df["Loss_7day_Avg"] = g[lc].transform(lambda x: x.rolling(7, min_periods=1).mean()).fillna(0)
    df["Loss_30day_Avg"] = g[lc].transform(lambda x: x.rolling(30, min_periods=1).mean()).fillna(0)
    df["Loss_7day_Std"] = g[lc].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0)).fillna(0)

    for diff in [1, 3, 7]:
        df[f"Loss_Change_{diff}"] = g[lc].diff(diff).fillna(0)
    df["Loss_Acceleration"] = g[lc].diff().diff().fillna(0)

    # Offtake cross-features
    oc = "Offtake_Loss_Pct"
    if oc in df.columns:
        for lag in [1, 2, 3, 7, 14]:
            df[f"Offtake_Lag_{lag}"] = g[oc].shift(lag).fillna(0)
        df["Offtake_7day_Avg"] = g[oc].transform(lambda x: x.rolling(7, min_periods=1).mean()).fillna(0)
        df["Offtake_30day_Avg"] = g[oc].transform(lambda x: x.rolling(30, min_periods=1).mean()).fillna(0)
        df["Offtake_7day_Std"] = g[oc].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0)).fillna(0)
    else:
        for col in (
            [f"Offtake_Lag_{l}" for l in [1,2,3,7,14]]
            + ["Offtake_7day_Avg", "Offtake_30day_Avg", "Offtake_7day_Std"]
        ):
            df[col] = 0.0

    if "Offtake_Actual" in df.columns:
        for period in [3, 7, 14]:
            df[f"Offtake_Momentum_{period}"] = (
                g["Offtake_Actual"].pct_change(periods=period).fillna(0)
                .replace([np.inf, -np.inf], 0)
            )
    else:
        for col in [f"Offtake_Momentum_{p}" for p in [3,7,14]]:
            df[col] = 0.0

    # Prod-Offtake relationship
    if prod_vol_col and "Offtake_Actual" in df.columns:
        df["Prod_Offtake_Ratio"] = (df[prod_vol_col] / (df["Offtake_Actual"] + 1e-3)).fillna(0)
        df["Prod_Offtake_Loss_Gap"] = (df[lc] - df.get(oc, 0)).fillna(0)
        df["Ratio_7day_Avg"] = g["Prod_Offtake_Ratio"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()).fillna(0)
        df["Gap_7day_Avg"] = g["Prod_Offtake_Loss_Gap"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()).fillna(0)
    else:
        for col in ["Prod_Offtake_Ratio","Prod_Offtake_Loss_Gap","Ratio_7day_Avg","Gap_7day_Avg"]:
            df[col] = 0.0

    return df

def compute_risk_score(pred_loss, rain_3day, waterlogging, consec_rain,
                       max_temp, offtake_loss, weights=None) -> float:
    """Weighted composite risk score in [0, 100]"""
    if weights is None:
        weights = Config.RISK_WEIGHTS

    c_loss = weights["loss"] * np.clip(pred_loss, 0, 100)
    c_rain = weights["rainfall"] * np.clip(rain_3day * 1.8, 0, 100)
    c_water = weights["waterlogging"] * np.clip(waterlogging * 100, 0, 100)
    c_consec = weights["consecutive"] * np.clip(consec_rain * 4, 0, 100)
    c_heat = weights["heat"] * (np.clip((max_temp - 35) * 3, 0, 100) if max_temp > 35 else 0)
    c_oft = weights["offtake"] * np.clip(offtake_loss * 0.5, 0, 100)

    return c_loss + c_rain + c_water + c_consec + c_heat + c_oft

def risk_category(score: float) -> str:
    """Categorize risk score"""
    if score < 25: return "Low"
    if score < 50: return "Moderate"
    if score < 75: return "High"
    return "Severe"

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

WEATHER_FEATURES = [
    "Rainfall_mm", "Rainfall_3day_Sum", "Rainfall_7day_Sum", "Rainfall_15day_Sum",
    "Prev_Day_Rainfall_mm", "Waterlogging_Index", "Consecutive_Rainy_Days",
    "Max_Temp_C", "Min_Temp_C", "Mean_Temp_C", "Max_Wind_Speed_kmh",
    "Precipitation_Hours", "Rainfall_EMA_7", "Rainfall_Intensity", "Temp_Range",
    "Is_Monsoon_Season", "Day_of_Week", "Is_Sunday", "Day_of_Month",
    "Day_of_Year", "Week_of_Year", "Quarter", "Is_Quarter_End_Month",
    "Season_Summer", "Season_Monsoon", "Season_Post_Monsoon", "Season_Winter",
    "Rainfall_x_Monsoon", "Rain_x_Temp", "Waterlogging_x_Monsoon",
    "Heat_x_Summer", "Wind_x_Winter",
    "Rain_Flag", "Heavy_Rain_Flag", "Heat_Stress_Flag", "High_Wind_Flag",
]

LAG_FEATURES = [
    "Prod_Lag_1", "Prod_Lag_2", "Prod_Lag_3", "Prod_Lag_7", "Prod_Lag_14", "Prod_Lag_30",
    "Prod_Momentum_3", "Prod_Momentum_7", "Prod_Momentum_14",
    "Loss_Lag_1", "Loss_Lag_2", "Loss_Lag_3", "Loss_Lag_7", "Loss_Lag_14",
    "Loss_7day_Avg", "Loss_30day_Avg", "Loss_7day_Std",
    "Loss_Change_1", "Loss_Change_3", "Loss_Change_7", "Loss_Acceleration",
]

OFFTAKE_FEATURES = [
    "Offtake_Lag_1", "Offtake_Lag_2", "Offtake_Lag_3", "Offtake_Lag_7", "Offtake_Lag_14",
    "Offtake_7day_Avg", "Offtake_30day_Avg", "Offtake_7day_Std",
    "Offtake_Momentum_3", "Offtake_Momentum_7", "Offtake_Momentum_14",
    "Prod_Offtake_Ratio", "Prod_Offtake_Loss_Gap", "Ratio_7day_Avg", "Gap_7day_Avg",
]

ALL_FEATURES = WEATHER_FEATURES + LAG_FEATURES + OFFTAKE_FEATURES

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main execution pipeline"""

    print("=" * 70)
    print("COAL INDIA - PRODUCTION LOSS PREDICTION MODEL")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print("=" * 70)

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================

    print("\nSTEP 1: LOADING DATA")
    print("-" * 70)

    # Load real data
    real_raw = safe_file_load(REAL_PATH)
    if real_raw is None:
        print("  ❌ Failed to load real data. Exiting.")
        return

    real_raw.rename(columns={"Area Code": "Area_Code"}, inplace=True)
    real_raw["Date"] = pd.to_datetime(real_raw["S_DATE"], format="%Y%m%d")
    real_raw["Data_Source"] = "Real"

    real_prod = real_raw[real_raw["MATERIAL"] == "PROD"].copy()
    real_offtake = real_raw[real_raw["MATERIAL"] == "OFFTAKE"].copy()

    # Load synthetic data
    synth_raw = safe_file_load(SYNTH_PATH)
    if synth_raw is None:
        print("  ❌ Failed to load synthetic data. Exiting.")
        return

    synth_raw["Date"] = pd.to_datetime(synth_raw["Date"])
    synth_raw["Data_Source"] = "Synthetic"

    synth_prod = synth_raw[synth_raw["Material_Type"] == "PROD"].copy()
    synth_offtake = synth_raw[synth_raw["Material_Type"] == "OFFTAKE"].copy()

    # Load weather data
    weather_dfs = []
    for wpath in WEATHER_PATHS:
        df = safe_file_load(wpath)
        if df is not None:
            weather_dfs.append(df)

    if not weather_dfs:
        print("  ❌ Failed to load weather data. Exiting.")
        return

    weather_raw = pd.concat(weather_dfs, ignore_index=True)
    weather_raw["Date"] = pd.to_datetime(weather_raw["date"])
    weather_raw["Area_Code"] = weather_raw["area_code"]

    # Drop excluded areas
    for df in [real_prod, real_offtake, synth_prod, synth_offtake]:
        df.drop(df[df["Area_Code"].isin(Config.EXCLUDED_AREAS)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    print(f"  Real PROD: {len(real_prod):,} rows, {real_prod['Area_Code'].nunique()} areas")
    print(f"  Real OFFTAKE: {len(real_offtake):,} rows")
    print(f"  Synthetic PROD: {len(synth_prod):,} rows")
    print(f"  Synthetic OFFTAKE: {len(synth_offtake):,} rows")
    print(f"  Weather: {len(weather_raw):,} rows")

    # =========================================================================
    # STEP 2: WEATHER FEATURE ENGINEERING
    # =========================================================================

    print("\nSTEP 2: WEATHER FEATURE ENGINEERING")
    print("-" * 70)

    weather_eng = engineer_weather_features(weather_raw)
    print(f"  Weather engineered: {len(weather_eng):,} rows, {len(weather_eng.columns)} columns")

    # =========================================================================
    # STEP 3: CALCULATE LOSS % & MERGE WEATHER
    # =========================================================================

    print("\nSTEP 3: CALCULATE LOSS % & MERGE WEATHER")
    print("-" * 70)

    # Weather columns for merge
    WEATHER_FEATURE_COLS = [
        "Rainfall_mm", "Prev_Day_Rainfall_mm",
        "Rainfall_3day_Sum", "Rainfall_7day_Sum", "Rainfall_15day_Sum",
        "Rainfall_EMA_7", "Precipitation_Hours",
        "Max_Temp_C", "Min_Temp_C", "Mean_Temp_C",
        "Max_Wind_Speed_kmh", "Max_Wind_Gust_kmh",
        "Waterlogging_Index", "Consecutive_Rainy_Days",
        "Rainfall_Intensity", "Temp_Range",
        "Rain_Flag", "Heavy_Rain_Flag", "Heat_Stress_Flag", "High_Wind_Flag",
        "Is_Monsoon_Season", "Is_Quarter_End_Month",
        "Day_of_Week", "Day_of_Month", "Day_of_Year",
        "Week_of_Year", "Quarter", "Is_Sunday",
        "Season_Summer", "Season_Monsoon", "Season_Post_Monsoon", "Season_Winter",
        "Rainfall_x_Monsoon", "Rain_x_Temp", "Waterlogging_x_Monsoon",
        "Heat_x_Summer", "Wind_x_Winter",
    ]

    weather_merge = weather_eng[["Date", "Area_Code"] + WEATHER_FEATURE_COLS].copy()

    # Real production
    real_prod["Production_Loss_Pct"] = calc_loss(real_prod)
    real_prod["Financial_Year"] = real_prod["Date"].apply(get_fy)
    real_prod = real_prod.merge(weather_merge, on=["Date", "Area_Code"], how="inner")

    # Real offtake
    real_offtake["Offtake_Loss_Pct"] = calc_loss(real_offtake)
    real_offtake["Financial_Year"] = real_offtake["Date"].apply(get_fy)

    # Merge offtake into real
    real_combined = real_prod.merge(
        real_offtake[["Date", "Area_Code", "Offtake_Loss_Pct", "ONDT_ACT"]]
        .rename(columns={"ONDT_ACT": "Offtake_Actual"}),
        on=["Date", "Area_Code"], how="left"
    )
    real_combined["Offtake_Loss_Pct"] = real_combined["Offtake_Loss_Pct"].fillna(0)
    real_combined["Offtake_Actual"] = real_combined["Offtake_Actual"].fillna(0)

    print(f"  Real combined: {len(real_combined):,} rows")

    # Synthetic production
    if "Final_Loss_Pct" in synth_prod.columns:
        synth_prod["Production_Loss_Pct"] = synth_prod["Final_Loss_Pct"].fillna(0)
    else:
        synth_prod["Production_Loss_Pct"] = calc_loss(
            synth_prod, "Daily_Production_Target", "Daily_Production_Actual"
        )
    synth_prod["Financial_Year"] = synth_prod["Date"].apply(get_fy)

    # Synthetic offtake
    if "Final_Loss_Pct" in synth_offtake.columns:
        synth_offtake["Offtake_Loss_Pct"] = synth_offtake["Final_Loss_Pct"].fillna(0)
    else:
        synth_offtake["Offtake_Loss_Pct"] = 0.0

    # Merge weather into synthetic
    synth_combined = synth_prod.merge(
        weather_merge, on=["Date", "Area_Code"], how="left"
    )
    for col in WEATHER_FEATURE_COLS:
        if col in synth_combined.columns:
            synth_combined[col] = synth_combined[col].fillna(0)

    # Merge offtake signal
    synth_offtake_slim = (
        synth_offtake[["Date", "Area_Code", "Offtake_Loss_Pct"]]
        .assign(Offtake_Actual=0)
    )
    synth_combined = synth_combined.merge(
        synth_offtake_slim, on=["Date", "Area_Code"], how="left"
    )
    synth_combined["Offtake_Loss_Pct"] = synth_combined["Offtake_Loss_Pct"].fillna(0)
    synth_combined["Offtake_Actual"] = synth_combined["Offtake_Actual"].fillna(0)

    print(f"  Synth combined: {len(synth_combined):,} rows")

    # =========================================================================
    # STEP 4: LAG & ROLLING FEATURE ENGINEERING
    # =========================================================================

    print("\nSTEP 4: LAG & ROLLING FEATURE ENGINEERING")
    print("-" * 70)

    real_combined = engineer_lag_features(real_combined)
    synth_combined = engineer_lag_features(synth_combined)

    print(f"  Real lag-engineered: {len(real_combined):,} rows")
    print(f"  Synth lag-engineered: {len(synth_combined):,} rows")
    print(f"  Features: Weather={len(WEATHER_FEATURES)}, "
          f"Lag={len(LAG_FEATURES)}, Offtake={len(OFFTAKE_FEATURES)}, "
          f"Total={len(ALL_FEATURES)}")

    # Ensure all features exist
    for df in [real_combined, synth_combined]:
        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

    # =========================================================================
    # STEP 5: TRAIN LIGHTGBM MODELS
    # =========================================================================

    print("\nSTEP 5: TRAINING LIGHTGBM MODELS")
    print("-" * 70)

    def prepare_area_data(area: int):
        """Prepare train/test splits for one area"""
        area_synth = synth_combined[synth_combined["Area_Code"] == area]
        area_real = real_combined[real_combined["Area_Code"] == area]

        if len(area_real) == 0:
            return (None,) * 7

        fy23_real = area_real[area_real["Financial_Year"] == 2023]
        fy24_real = area_real[area_real["Financial_Year"] == 2024]

        train = (
            pd.concat([area_synth, fy23_real], ignore_index=True)
            if len(area_synth) > 0
            else fy23_real.copy()
        )
        test = fy24_real.copy()

        train = train.dropna(subset=["Production_Loss_Pct"])
        test = test.dropna(subset=["Production_Loss_Pct"])

        if len(train) < Config.MIN_TRAIN_ROWS or len(test) < Config.MIN_TEST_ROWS:
            return (None,) * 7

        X_train = train[ALL_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
        y_train = train["Production_Loss_Pct"].clip(0, 100)
        X_test = test[ALL_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
        y_test = test["Production_Loss_Pct"].clip(0, 100).values
        dates = test["Date"].values
        weights = np.where(train["Data_Source"] == "Real", Config.REAL_WEIGHT, 1.0)

        return X_train, y_train, weights, X_test, y_test, dates, train["Date"].values

    def train_area_model(X_train, y_train, weights, X_test, area: int):
        """Train LightGBM model for one area"""
        try:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)

            model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(X_tr, y_train, sample_weight=weights)
            y_pred = np.clip(model.predict(X_te), 0, 100)

            return model, scaler, y_pred
        except Exception as e:
            print(f"  ✗ Area {area}: training failed - {e}")
            return None, None, None

    areas = sorted(set(real_combined["Area_Code"].unique()))
    results = []
    preds = []
    models = {}

    for i, area in enumerate(areas, 1):
        data = prepare_area_data(area)
        if data[0] is None:
            continue

        X_train, y_train, weights, X_test, y_test, dates, train_dates = data
        model, scaler, y_pred = train_area_model(X_train, y_train, weights, X_test, area)

        if model is None:
            continue

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Area_Code": area,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Train_Samples": len(X_train),
            "Test_Samples": len(X_test),
        })

        models[area] = {
            "model": model,
            "scaler": scaler,
            "features": ALL_FEATURES,
            "train_dates": train_dates,
            "test_dates": dates,
        }

        for j, date in enumerate(dates):
            preds.append({
                "Area_Code": area,
                "Date": date,
                "Actual_Loss": y_test[j],
                "Predicted_Loss": y_pred[j],
            })

        print(f"  ✓ Area {area:5d}  MAE={mae:.2f}%  RMSE={rmse:.2f}%  "
              f"R²={r2:.3f}  [train={len(X_train)}, test={len(X_test)}]")

    results_df = pd.DataFrame(results)
    preds_df = pd.DataFrame(preds)

    print(f"\n  Models trained: {len(results_df)}")
    if len(results_df):
        print(f"  Avg R²: {results_df['R2'].mean():.3f}")
        print(f"  Avg MAE: {results_df['MAE'].mean():.2f}%")
        print(f"  Avg RMSE: {results_df['RMSE'].mean():.2f}%")

    # =========================================================================
    # STEP 6: SAVE MODELS & RESULTS
    # =========================================================================

    print("\nSTEP 6: SAVING MODELS & RESULTS")
    print("-" * 70)

    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, "LightGBM_Results.csv"), index=False)
    preds_df.to_csv(os.path.join(OUTPUT_DIR, "LightGBM_Predictions.csv"), index=False)

    # Save models
    saved_models = 0
    for area, mdata in models.items():
        try:
            path = os.path.join(MODELS_DIR, f"{area}_LightGBM.pkl")
            with open(path, "wb") as f:
                pickle.dump(mdata, f)
            saved_models += 1
        except Exception as e:
            print(f"  ⚠️ Failed to save model for area {area}: {e}")

    print(f"  ✓ Models saved: {saved_models}/{len(models)}")

    # Save feature manifest
    feature_manifest = pd.DataFrame({
        "Feature": ALL_FEATURES,
        "Group": (
            ["weather"] * len(WEATHER_FEATURES)
            + ["lag"] * len(LAG_FEATURES)
            + ["offtake"] * len(OFFTAKE_FEATURES)
        ),
        "Position_Index": range(len(ALL_FEATURES)),
    })
    feature_manifest.to_csv(os.path.join(OUTPUT_DIR, "Feature_Manifest.csv"), index=False)

    # =========================================================================
    # STEP 7: RISK SCORING
    # =========================================================================

    print("\nSTEP 7: RISK SCORING")
    print("-" * 70)

    risk_rows = []
    for _, row in preds_df.iterrows():
        area = row["Area_Code"]
        date = row["Date"]

        match = real_combined[
            (real_combined["Area_Code"] == area) &
            (real_combined["Date"] == date)
        ]

        if len(match):
            m = match.iloc[0]
            rain = m.get("Rainfall_3day_Sum", 0)
            water = m.get("Waterlogging_Index", 0)
            consec = m.get("Consecutive_Rainy_Days", 0)
            temp = m.get("Max_Temp_C", 25)
            oft = m.get("Offtake_Loss_Pct", 0)
        else:
            rain, water, consec, temp, oft = 0, 0, 0, 25, 0

        score = compute_risk_score(row["Predicted_Loss"], rain, water, consec, temp, oft)
        cat = risk_category(score)

        risk_rows.append({
            "Area_Code": area,
            "Date": date,
            "Predicted_Loss_Pct": row["Predicted_Loss"],
            "Actual_Loss_Pct": row["Actual_Loss"],
            "Risk_Score": round(score, 2),
            "Risk_Category": cat,
            "Rainfall_3day_Sum": rain,
            "Waterlogging_Index": water,
            "Consecutive_Rainy_Days": consec,
            "Max_Temp_C": temp,
            "Offtake_Loss_Pct": oft,
            "Loss_Component": Config.RISK_WEIGHTS["loss"] * np.clip(row["Predicted_Loss"], 0, 100),
            "Rainfall_Component": Config.RISK_WEIGHTS["rainfall"] * np.clip(rain * 1.8, 0, 100),
            "Waterlogging_Component": Config.RISK_WEIGHTS["waterlogging"] * np.clip(water * 100, 0, 100),
            "Consecutive_Component": Config.RISK_WEIGHTS["consecutive"] * np.clip(consec * 4, 0, 100),
            "Heat_Component": Config.RISK_WEIGHTS["heat"] * (np.clip((temp - 35) * 3, 0, 100) if temp > 35 else 0),
            "Offtake_Component": Config.RISK_WEIGHTS["offtake"] * np.clip(oft * 0.5, 0, 100),
        })

    risk_df = pd.DataFrame(risk_rows)
    risk_df.to_csv(os.path.join(OUTPUT_DIR, "Risk_Scores.csv"), index=False)

    print(f"  ✓ Risk scores: {len(risk_df):,} records")

    print("\n  Risk Distribution:")
    for cat in ["Low", "Moderate", "High", "Severe"]:
        n = (risk_df["Risk_Category"] == cat).sum()
        pct = n / len(risk_df) * 100 if len(risk_df) else 0
        print(f"    {cat:9s}: {n:5,} ({pct:.1f}%)")

    print("\n  Top 10 highest average risk areas:")
    top = risk_df.groupby("Area_Code")["Risk_Score"].mean().sort_values(ascending=False).head(10)
    for area, score in top.items():
        print(f"    Area {area}: {score:.1f} ({risk_category(score)})")

    # =========================================================================
    # STEP 8: MASTER DATASET
    # =========================================================================

    print("\nSTEP 8: CREATING MASTER DATASET")
    print("-" * 70)

    master = real_combined.merge(
        preds_df.rename(columns={
            "Actual_Loss": "Predicted_Actual_Loss_Pct",
            "Predicted_Loss": "Predicted_Loss_Pct",
        }),
        on=["Area_Code", "Date"], how="left"
    )

    master = master.merge(
        risk_df[["Area_Code", "Date", "Risk_Score", "Risk_Category"]],
        on=["Area_Code", "Date"], how="left"
    )

    if "ONDT_TARGET (Thousand Tonnes)" in master.columns:
        master["Predicted_Loss_Tonnes"] = (
            master["Predicted_Loss_Pct"].fillna(0) / 100
            * master["ONDT_TARGET (Thousand Tonnes)"]
        )

    master.to_csv(os.path.join(OUTPUT_DIR, "Master_Dataset.csv"), index=False)
    print(f"  ✓ Master dataset: {len(master):,} rows, {len(master.columns)} columns")

    # =========================================================================
    # STEP 9: SHAP EXPLAINABILITY
    # =========================================================================

    print("\nSTEP 9: SHAP EXPLAINABILITY")
    print("-" * 70)

    shap_records = []
    processed, failed = 0, 0

    for area, mdata in models.items():
        try:
            area_test = real_combined[
                (real_combined["Area_Code"] == area) &
                (real_combined["Financial_Year"] == 2024)
            ].copy()

            if len(area_test) == 0:
                failed += 1
                continue

            sample_size = min(30, len(area_test))
            X_sample = (
                area_test[ALL_FEATURES].head(sample_size)
                .replace([np.inf, -np.inf], 0).fillna(0)
            )
            X_scaled = mdata["scaler"].transform(X_sample)

            explainer = shap.TreeExplainer(mdata["model"])
            shap_vals = explainer.shap_values(X_scaled)

            for i in range(len(X_sample)):
                shap_row = pd.Series(shap_vals[i], index=ALL_FEATURES)
                top_pos = shap_row[shap_row > 0].nlargest(5)
                top_neg = shap_row[shap_row < 0].nsmallest(5)

                for feat, val in pd.concat([top_pos, top_neg]).items():
                    shap_records.append({
                        "Area_Code": area,
                        "Sample_Index": i,
                        "Feature": feat,
                        "SHAP_Value": round(val, 5),
                        "Direction": "Increases Loss" if val > 0 else "Decreases Loss",
                    })

            processed += 1

        except Exception as e:
            failed += 1
            print(f"  ✗ Area {area}: SHAP failed - {e}")

    shap_df = pd.DataFrame(shap_records)
    if len(shap_df):
        shap_df.to_csv(os.path.join(OUTPUT_DIR, "SHAP_Explanations.csv"), index=False)
        print(f"  ✓ SHAP processed: {processed} areas | failed: {failed}")
        print(f"  ✓ Records saved: {len(shap_df):,}")

        print("\n  Features that most increase loss (avg SHAP):")
        pos = (
            shap_df[shap_df["Direction"] == "Increases Loss"]
            .groupby("Feature")["SHAP_Value"].mean()
            .sort_values(ascending=False).head(10)
        )
        for feat, val in pos.items():
            print(f"    +{val:.4f}  {feat}")

        print("\n  Features that most reduce loss:")
        neg = (
            shap_df[shap_df["Direction"] == "Decreases Loss"]
            .groupby("Feature")["SHAP_Value"].mean()
            .sort_values().head(10)
        )
        for feat, val in neg.items():
            print(f"    {val:.4f}  {feat}")
    else:
        print("  ⚠️ No SHAP results generated")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    if len(results_df):
        print(f"""
  Performance Summary:
    Areas trained: {len(results_df)}
    Avg R²: {results_df['R2'].mean():.3f}
    Avg MAE: {results_df['MAE'].mean():.2f}%
    Avg RMSE: {results_df['RMSE'].mean():.2f}%
    Features: {len(ALL_FEATURES)}

  Outputs saved to: {OUTPUT_DIR}
    ✓ Master_Dataset.csv
    ✓ LightGBM_Results.csv
    ✓ LightGBM_Predictions.csv
    ✓ Risk_Scores.csv
    ✓ SHAP_Explanations.csv
    ✓ Feature_Manifest.csv
    ✓ Models/*.pkl ({saved_models} files)
""")

    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()
