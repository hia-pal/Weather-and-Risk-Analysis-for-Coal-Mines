# -*- coding: utf-8 -*-
"""
VALIDATE FORECAST MODELS
=========================
Loads the trained per-area LightGBM models and re-scores them against
real, already-known days from Master_Dataset.csv (days where the true
Production_Loss_Pct is known). Reports MAE/RMSE/R2 per area and overall,
so you can confirm the loaded models actually predict well before
trusting the forecast pipeline's output.
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MASTER_PATH = os.path.join(".", "Model_Outputs", "Master_Dataset.csv")
MODELS_PATH = os.path.join(".", "Model_Outputs", "Models")
N_TEST_DAYS = 15          # most recent N real days per area, held out for validation
EXCLUDED_AREAS = [5190, 6003, 8351, 4151, 4251]

# R2 is unstable/meaningless when the true values barely move across the test
# window (small denominator -> huge or NaN R2). Only report R2 when the true
# values have at least this much spread; otherwise leave it blank.
MIN_TRUE_STD_FOR_R2 = 1.0   # in loss-percentage points

# MAE thresholds used to flag an area as worth a closer look
MAE_WARN_THRESHOLD = 5.0    # percentage points
MAE_FLAG_THRESHOLD = 10.0   # percentage points


def load_models():
    models = {}
    for f in glob.glob(os.path.join(MODELS_PATH, "*.pkl")):
        area = int(os.path.basename(f).replace("_LightGBM.pkl", ""))
        if area not in EXCLUDED_AREAS:
            with open(f, "rb") as fh:
                models[area] = pickle.load(fh)
    return models


def main():
    master = pd.read_csv(MASTER_PATH)
    master["Date"] = pd.to_datetime(master["Date"])
    master = master[~master["Area_Code"].isin(EXCLUDED_AREAS)]

    models = load_models()
    print(f"Loaded {len(models)} models\n")

    rows = []
    for area, mdata in models.items():
        area_df = master[master["Area_Code"] == area].sort_values("Date")
        area_df = area_df.dropna(subset=["Production_Loss_Pct"])
        if len(area_df) < N_TEST_DAYS:
            continue

        test = area_df.tail(N_TEST_DAYS)
        features = mdata.get("features", [])
        missing = [f for f in features if f not in test.columns]
        if missing:
            print(f"  ⚠️ Area {area}: missing {len(missing)} features in master data, skipping")
            continue

        X = test[features].replace([np.inf, -np.inf], 0).fillna(0)
        y_true = test["Production_Loss_Pct"].clip(0, 100).values

        X_scaled = mdata["scaler"].transform(X)
        y_pred = np.clip(mdata["model"].predict(X_scaled), 0, 100)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Only report R2 when the true values have enough spread for it to
        # mean anything; otherwise it's a divide-by-a-tiny-number artifact.
        true_std = float(np.std(y_true))
        r2 = r2_score(y_true, y_pred) if true_std >= MIN_TRUE_STD_FOR_R2 else np.nan

        if mae >= MAE_FLAG_THRESHOLD:
            status = "REVIEW"
        elif mae >= MAE_WARN_THRESHOLD:
            status = "WATCH"
        else:
            status = "OK"

        rows.append({
            "Area_Code": area,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 3) if not np.isnan(r2) else None,
            "N": len(test),
            "Status": status,
        })

        r2_display = f"{r2:6.3f}" if not np.isnan(r2) else "   n/a"
        print(f"  Area {area:5d}  MAE={mae:6.2f}%  RMSE={rmse:6.2f}%  R2={r2_display}  [{status:6s}]  (n={len(test)})")

    if not rows:
        print("\n❌ No areas could be validated (check paths / feature columns).")
        return

    results = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print(f"Areas validated : {len(results)}")
    print(f"Avg MAE         : {results['MAE'].mean():.2f}%")
    print(f"Avg RMSE        : {results['RMSE'].mean():.2f}%")

    r2_valid = results["R2"].dropna()
    if len(r2_valid) > 0:
        print(f"Avg R2 (where meaningful, {len(r2_valid)}/{len(results)} areas): {r2_valid.mean():.3f}")
    else:
        print("Avg R2: not shown - no area had enough variation in true values")

    n_review = (results["Status"] == "REVIEW").sum()
    n_watch = (results["Status"] == "WATCH").sum()
    print(f"\nAreas needing review (MAE >= {MAE_FLAG_THRESHOLD:.0f}%): {n_review}")
    print(f"Areas to watch      (MAE >= {MAE_WARN_THRESHOLD:.0f}%): {n_watch}")
    if n_review > 0:
        print("  -> " + ", ".join(str(a) for a in results.loc[results["Status"] == "REVIEW", "Area_Code"]))
    print("=" * 60)
    print("\nCompare Avg MAE/RMSE to training-time metrics (MAE 1.86%, RMSE 3.94%).")
    print("R2 is intentionally hidden for areas with near-flat true values, where")
    print("it becomes a divide-by-almost-zero artifact rather than a real signal.")

    results = results.sort_values("MAE", ascending=False)
    results.to_csv("Model_Validation_Report.csv", index=False)
    print("\n✅ Saved: Model_Validation_Report.csv (sorted by MAE, worst first)")


if __name__ == "__main__":
    main()
