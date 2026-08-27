# -*- coding: utf-8 -*-
"""
UNIFIED RISK SCORING SYSTEM
============================
Single file combining:
1. Risk scoring formula (compute_risk_score, risk_category)
2. Forecast risk scoring pipeline (rebuilds rolling weather, computes scores)
3. Testing/validation against historical outcomes

This is the SINGLE SOURCE OF TRUTH for all risk scoring operations.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# =============================================================================
# PART 1: RISK SCORING FORMULA (SINGLE SOURCE OF TRUTH)
# =============================================================================

# Tune these if you want stronger/weaker bonuses
RISK_PARAMS = {
    "rain_bonus_divisor": 10,   # Rainfall_3day_Sum / this, capped below
    "rain_bonus_cap": 8,
    "consec_bonus_rate": 0.5,   # Consecutive_Rainy_Days * this, capped below
    "consec_bonus_cap": 4,
    "heat_threshold_c": 38,     # bonus only kicks in above this temp
    "heat_bonus_rate": 1.0,
    "heat_bonus_cap": 5,
    "offtake_bonus_rate": 0.08, # Offtake_Loss_Pct * this, capped below
    "offtake_bonus_cap": 8,
}

def compute_risk_score(pred_loss, rain_3day, consec_rain, max_temp, offtake_loss, params=None) -> float:
    """
    Returns a single risk score in [0, 100].
    
    Args:
        pred_loss:     Predicted_Loss_Pct from the LightGBM model (0-100)
        rain_3day:     Rainfall_3day_Sum in mm
        consec_rain:   Consecutive_Rainy_Days
        max_temp:      Max_Temp_C
        offtake_loss:  Offtake_Loss_Pct (0-100)
    """
    if params is None:
        params = RISK_PARAMS
    
    base = np.clip(pred_loss, 0, 100)
    
    rain_bonus = np.clip(rain_3day / params["rain_bonus_divisor"], 0, params["rain_bonus_cap"])
    consec_bonus = np.clip(consec_rain * params["consec_bonus_rate"], 0, params["consec_bonus_cap"])
    heat_bonus = (
        np.clip((max_temp - params["heat_threshold_c"]) * params["heat_bonus_rate"],
                0, params["heat_bonus_cap"])
        if max_temp > params["heat_threshold_c"] else 0.0
    )
    offtake_bonus = np.clip(offtake_loss * params["offtake_bonus_rate"], 0, params["offtake_bonus_cap"])
    
    score = base + rain_bonus + consec_bonus + heat_bonus + offtake_bonus
    return float(np.clip(score, 0, 100))

def risk_category(score: float) -> str:
    """Categorize a 0-100 risk score."""
    if score < 25: return "Low"
    if score < 50: return "Moderate"
    if score < 75: return "High"
    return "Severe"

# =============================================================================
# PART 2: FORECAST RISK SCORING PIPELINE
# =============================================================================

class ForecastConfig:
    """Configuration for forecast risk scoring"""
    BASE_PATH = "."
    MASTER_FILE = os.path.join(BASE_PATH, "Model_Outputs", "Master_Dataset.csv")
    FORECAST_FILE = os.path.join(BASE_PATH, "Forecast_Outputs", "Forecast_Predictions.csv")
    OUTPUT_PATH = os.path.join(BASE_PATH, "Forecast_Outputs")
    OUTPUT_FILE = "Forecast_Risk_Scores.csv"
    
    EXCLUDED_AREAS = [5190, 6003, 8351, 4151, 4251]
    
    RAIN_HISTORY_SEED_DAYS = 14   # real days of rainfall history to seed rolling stats
    RAIN_WET_DAY_THRESHOLD = 0.5  # mm threshold used for "rainy day" flag
    
    TOP_N_AREAS = 10
    HIGH_RISK_THRESHOLD = 50  # "Moderate" and up, for the alert summary

def load_forecast():
    """Load forecast predictions"""
    print(f"  Loading forecast predictions from: {ForecastConfig.FORECAST_FILE}")
    if not os.path.exists(ForecastConfig.FORECAST_FILE):
        raise FileNotFoundError(
            f"{ForecastConfig.FORECAST_FILE} not found. Run the forecast pipeline first."
        )
    df = pd.read_csv(ForecastConfig.FORECAST_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[~df["Area_Code"].isin(ForecastConfig.EXCLUDED_AREAS)].copy()
    print(f"  ✅ Loaded: {len(df):,} rows, {df['Area_Code'].nunique()} areas")
    print(f"     Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df

def load_master():
    """Load master dataset"""
    print(f"  Loading master dataset from: {ForecastConfig.MASTER_FILE}")
    if not os.path.exists(ForecastConfig.MASTER_FILE):
        raise FileNotFoundError(
            f"{ForecastConfig.MASTER_FILE} not found. Run the training pipeline first."
        )
    df = pd.read_csv(ForecastConfig.MASTER_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[~df["Area_Code"].isin(ForecastConfig.EXCLUDED_AREAS)].copy()
    print(f"  ✅ Loaded: {len(df):,} rows")
    return df

def rebuild_rolling_weather(area_forecast, area_master):
    """
    For ONE area: stitch the last RAIN_HISTORY_SEED_DAYS of real Rainfall_mm
    (from master) onto the forecasted Rainfall_mm, recompute rolling stats.
    """
    area_forecast = area_forecast.sort_values("Date").reset_index(drop=True)
    
    if "Rainfall_mm" in area_master.columns:
        hist = (
            area_master[["Date", "Rainfall_mm"]]
            .sort_values("Date")
            .tail(ForecastConfig.RAIN_HISTORY_SEED_DAYS)
            .copy()
        )
    else:
        hist = pd.DataFrame(columns=["Date", "Rainfall_mm"])
    
    hist["Source"] = "history"
    fc = area_forecast[["Date", "Rainfall_mm"]].copy()
    fc["Source"] = "forecast"
    
    combined = pd.concat([hist, fc], ignore_index=True).sort_values("Date").reset_index(drop=True)
    combined["Rainfall_mm"] = combined["Rainfall_mm"].fillna(0)
    
    # Rolling rainfall stats - identical windows to training
    combined["Rainfall_3day_Sum"] = combined["Rainfall_mm"].rolling(3, min_periods=1).sum()
    combined["Waterlogging_Index"] = (combined["Rainfall_3day_Sum"] / 50).clip(0, 1)
    
    rain_flag = (combined["Rainfall_mm"] > ForecastConfig.RAIN_WET_DAY_THRESHOLD).astype(int)
    combined["Consecutive_Rainy_Days"] = (
        rain_flag.groupby((rain_flag != rain_flag.shift()).cumsum()).cumsum()
    )
    combined["Consecutive_Rainy_Days"] = np.where(rain_flag == 1, combined["Consecutive_Rainy_Days"], 0)
    
    forecast_only = combined[combined["Source"] == "forecast"][
        ["Date", "Rainfall_3day_Sum", "Waterlogging_Index", "Consecutive_Rainy_Days"]
    ].reset_index(drop=True)
    
    return area_forecast.merge(forecast_only, on="Date", how="left")

def get_last_offtake(area_master):
    """Last known real Offtake_Loss_Pct for an area, carried forward flat."""
    if "Offtake_Loss_Pct" not in area_master.columns or len(area_master) == 0:
        return 0.0
    sorted_master = area_master.sort_values("Date")
    val = sorted_master["Offtake_Loss_Pct"].dropna()
    return float(val.iloc[-1]) if len(val) else 0.0

def run_forecast_risk_scoring():
    """
    Main function for forecast risk scoring pipeline.
    Returns the risk scores DataFrame.
    """
    print("=" * 80)
    print("FORECAST RISK SCORING")
    print("=" * 80)
    
    print("\n📂 LOADING DATA")
    print("-" * 80)
    forecast_df = load_forecast()
    master_df = load_master()
    
    print("\n🔧 RECONSTRUCTING ROLLING WEATHER + OFFTAKE SIGNALS PER AREA")
    print("-" * 80)
    
    enriched_frames = []
    areas = sorted(forecast_df["Area_Code"].unique())
    
    for area in areas:
        area_forecast = forecast_df[forecast_df["Area_Code"] == area].copy()
        area_master = master_df[master_df["Area_Code"] == area].copy()
        
        area_forecast = rebuild_rolling_weather(area_forecast, area_master)
        area_forecast["Offtake_Loss_Pct"] = get_last_offtake(area_master)
        
        enriched_frames.append(area_forecast)
    
    enriched_df = pd.concat(enriched_frames, ignore_index=True)
    
    for col in ["Rainfall_3day_Sum", "Waterlogging_Index", "Consecutive_Rainy_Days",
                "Max_Temp_C", "Offtake_Loss_Pct", "Predicted_Loss_Pct"]:
        if col not in enriched_df.columns:
            enriched_df[col] = 0.0
        enriched_df[col] = enriched_df[col].fillna(0)
    
    print(f"  ✅ Enriched: {len(enriched_df):,} rows across {len(areas)} areas")
    
    print("\n⚠️  COMPUTING RISK SCORES")
    print("-" * 80)
    
    p = RISK_PARAMS
    risk_rows = []
    for _, row in enriched_df.iterrows():
        pred_loss = row["Predicted_Loss_Pct"]
        rain = row["Rainfall_3day_Sum"]
        consec = row["Consecutive_Rainy_Days"]
        temp = row["Max_Temp_C"]
        oft = row["Offtake_Loss_Pct"]
        
        score = compute_risk_score(pred_loss, rain, consec, temp, oft)
        cat = risk_category(score)
        
        rain_bonus = round(float(np.clip(rain / p["rain_bonus_divisor"], 0, p["rain_bonus_cap"])), 2)
        consec_bonus = round(float(np.clip(consec * p["consec_bonus_rate"], 0, p["consec_bonus_cap"])), 2)
        heat_bonus = round(
            float(np.clip((temp - p["heat_threshold_c"]) * p["heat_bonus_rate"], 0, p["heat_bonus_cap"]))
            if temp > p["heat_threshold_c"] else 0.0, 2
        )
        offtake_bonus = round(float(np.clip(oft * p["offtake_bonus_rate"], 0, p["offtake_bonus_cap"])), 2)
        
        risk_rows.append({
            "Area_Code": row["Area_Code"],
            "Date": row["Date"],
            "Predicted_Loss_Pct": pred_loss,
            "Predicted_Production": row.get("Predicted_Production", np.nan),
            "Risk_Score": round(score, 2),
            "Risk_Category": cat,
            "Base_Loss_Score": round(float(np.clip(pred_loss, 0, 100)), 2),
            "Rain_Bonus": rain_bonus,
            "Consecutive_Bonus": consec_bonus,
            "Heat_Bonus": heat_bonus,
            "Offtake_Bonus": offtake_bonus,
            "Rainfall_mm": row.get("Rainfall_mm", 0.0),
            "Rainfall_3day_Sum": round(rain, 2),
            "Waterlogging_Index": round(row["Waterlogging_Index"], 3),
            "Consecutive_Rainy_Days": int(consec),
            "Max_Temp_C": round(temp, 2),
            "Offtake_Loss_Pct": round(oft, 2),
        })
    
    risk_df = pd.DataFrame(risk_rows).sort_values(["Area_Code", "Date"]).reset_index(drop=True)
    
    print(f"  ✅ Risk scores computed: {len(risk_df):,} records")
    
    print("\n💾 SAVING OUTPUT")
    print("-" * 80)
    os.makedirs(ForecastConfig.OUTPUT_PATH, exist_ok=True)
    out_path = os.path.join(ForecastConfig.OUTPUT_PATH, ForecastConfig.OUTPUT_FILE)
    risk_df.to_csv(out_path, index=False)
    print(f"  ✅ {out_path}")
    
    # Summary
    print("\n📊 RISK DISTRIBUTION (all forecast days, all areas)")
    print("-" * 80)
    for cat in ["Low", "Moderate", "High", "Severe"]:
        n = (risk_df["Risk_Category"] == cat).sum()
        pct = n / len(risk_df) * 100 if len(risk_df) else 0
        print(f"  {cat:9s}: {n:5,} ({pct:.1f}%)")
    
    print(f"\n🔝 TOP {ForecastConfig.TOP_N_AREAS} HIGHEST AVG RISK AREAS (across forecast window)")
    print("-" * 80)
    top = (
        risk_df.groupby("Area_Code")["Risk_Score"]
        .mean()
        .sort_values(ascending=False)
        .head(ForecastConfig.TOP_N_AREAS)
    )
    for area, score in top.items():
        print(f"  Area {area}: {score:.1f} ({risk_category(score)})")
    
    print(f"\n🚨 UPCOMING HIGH-RISK DAYS (Risk_Score >= {ForecastConfig.HIGH_RISK_THRESHOLD})")
    print("-" * 80)
    alerts = (
        risk_df[risk_df["Risk_Score"] >= ForecastConfig.HIGH_RISK_THRESHOLD]
        .sort_values(["Date", "Risk_Score"], ascending=[True, False])
    )
    if len(alerts):
        for _, r in alerts.head(20).iterrows():
            print(f"  {r['Date'].date()}  Area {r['Area_Code']:5d}  "
                  f"Risk={r['Risk_Score']:.1f} ({r['Risk_Category']})  "
                  f"Loss={r['Predicted_Loss_Pct']:.1f}%")
        if len(alerts) > 20:
            print(f"  ... and {len(alerts) - 20} more (see {ForecastConfig.OUTPUT_FILE})")
    else:
        print("  None - all forecasted days are Low risk.")
    
    print("\n" + "=" * 80)
    print("✅ RISK SCORING COMPLETE")
    print("=" * 80)
    
    return risk_df

# =============================================================================
# PART 3: TESTING/VALIDATION
# =============================================================================

class TestConfig:
    """Configuration for testing"""
    RISK_SCORES_PATH = os.path.join(".", "Model_Outputs", "Risk_Scores.csv")

def test_risk_score_formula():
    """
    Tests whether the risk formula is actually good, using real FY2024 outcomes.
    """
    print("=" * 70)
    print("TESTING RISK SCORE FORMULA AGAINST REAL FY2024 OUTCOMES")
    print("=" * 70)
    
    if not os.path.exists(TestConfig.RISK_SCORES_PATH):
        print(f"\n❌ {TestConfig.RISK_SCORES_PATH} not found. Run the training script first.")
        return None
    
    df = pd.read_csv(TestConfig.RISK_SCORES_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"\n  Loaded {len(df):,} real historical area-days ({df['Area_Code'].nunique()} areas)")
    
    # Recompute the score fresh using the current formula
    df["Risk_Score"] = df.apply(
        lambda r: compute_risk_score(
            r["Predicted_Loss_Pct"], r["Rainfall_3day_Sum"],
            r["Consecutive_Rainy_Days"], r["Max_Temp_C"], r["Offtake_Loss_Pct"],
        ), axis=1
    )
    df["Risk_Category"] = df["Risk_Score"].apply(risk_category)
    
    # --- 1. Correlation with real outcomes ---
    print("\n" + "-" * 70)
    print("1. DOES THE SCORE TRACK REAL LOSSES?")
    print("-" * 70)
    rho_score, _ = spearmanr(df["Risk_Score"], df["Actual_Loss_Pct"])
    rho_pred, _ = spearmanr(df["Predicted_Loss_Pct"], df["Actual_Loss_Pct"])
    print(f"  Predicted_Loss_Pct alone : {rho_pred:.3f}  (best possible reference)")
    print(f"  Risk_Score (current)     : {rho_score:.3f}")
    
    if rho_score >= rho_pred - 0.02:
        print("  ✅ GOOD - the risk score preserves the model's predictive power.")
    else:
        print("  ⚠️  The risk score is weaker than the raw model prediction - "
              "the bonuses may be too large relative to the base.")
    
    # --- 2. Are the categories meaningful? ---
    print("\n" + "-" * 70)
    print("2. DO THE CATEGORIES MEAN WHAT THEY CLAIM?")
    print("-" * 70)
    cat_order = ["Low", "Moderate", "High", "Severe"]
    summary = df.groupby("Risk_Category")["Actual_Loss_Pct"].agg(["mean", "count"]).reindex(cat_order)
    print(summary.to_string())
    
    means = summary["mean"].dropna().tolist()
    if all(means[i] <= means[i + 1] for i in range(len(means) - 1)):
        print("\n  ✅ GOOD - each category has a higher average real loss than the one below it.")
    else:
        print("\n  ⚠️  Categories are NOT ordered correctly - thresholds may need adjusting.")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return df

# =============================================================================
# MAIN - RUN ALL OR SELECTED COMPONENTS
# =============================================================================

def main():
    """Main function - runs all components or selected ones based on arguments"""
    import sys
    
    print("=" * 80)
    print("UNIFIED RISK SCORING SYSTEM")
    print("=" * 80)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "forecast":
            print("\n🚀 Running Forecast Risk Scoring only\n")
            return run_forecast_risk_scoring()
        elif mode == "test":
            print("\n🧪 Running Test/Validation only\n")
            return test_risk_score_formula()
        elif mode == "all":
            print("\n🚀 Running all components\n")
            # Run test first
            test_result = test_risk_score_formula()
            print("\n" + "=" * 80 + "\n")
            # Then run forecast
            forecast_result = run_forecast_risk_scoring()
            return forecast_result
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: 'forecast', 'test', 'all'")
            return None
    
    # Default: interactive mode - ask user what to run
    print("\nWhat would you like to run?")
    print("  1. Forecast Risk Scoring (compute scores for future dates)")
    print("  2. Test/Validate formula against historical data")
    print("  3. Run all (test then forecast)")
    print("  4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        return run_forecast_risk_scoring()
    elif choice == "2":
        return test_risk_score_formula()
    elif choice == "3":
        test_result = test_risk_score_formula()
        print("\n" + "=" * 80 + "\n")
        return run_forecast_risk_scoring()
    else:
        print("Exiting...")
        return None

if __name__ == "__main__":
    result = main()
