import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


DATA_DIR = Path(".")
TARGET = "demand"
ID_COL = "Index"
RANDOM_STATE = 42


def parse_timestamp_features(df):
    out = df.copy()
    parts = out["timestamp"].astype(str).str.split(":", expand=True)
    out["hour"] = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    out["minute"] = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    out["time_index"] = out["hour"] * 4 + (out["minute"] // 15)
    out["total_minutes"] = out["hour"] * 60 + out["minute"]

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["time_index_sin"] = np.sin(2 * np.pi * out["time_index"] / 96)
    out["time_index_cos"] = np.cos(2 * np.pi * out["time_index"] / 96)
    out["is_morning_peak"] = out["hour"].between(7, 10).astype(int)
    out["is_evening_peak"] = out["hour"].between(16, 19).astype(int)
    out["is_late_night"] = out["hour"].between(0, 5).astype(int)
    out["is_peak_hour"] = ((out["is_morning_peak"] == 1) | (out["is_evening_peak"] == 1)).astype(int)
    return out


def decode_geohash(value):
    if pd.isna(value):
        return np.nan, np.nan
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    bits = [16, 8, 4, 2, 1]
    lat = [-90.0, 90.0]
    lon = [-180.0, 180.0]
    even = True
    try:
        for char in str(value).lower():
            cd = base32.index(char)
            for mask in bits:
                if even:
                    mid = (lon[0] + lon[1]) / 2
                    if cd & mask:
                        lon[0] = mid
                    else:
                        lon[1] = mid
                else:
                    mid = (lat[0] + lat[1]) / 2
                    if cd & mask:
                        lat[0] = mid
                    else:
                        lat[1] = mid
                even = not even
        return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2
    except ValueError:
        return np.nan, np.nan


def add_geohash_features(df):
    out = df.copy()
    out["geohash"] = out["geohash"].astype(str)
    out["geohash_prefix_3"] = out["geohash"].str[:3]
    out["geohash_prefix_4"] = out["geohash"].str[:4]
    out["geohash_prefix_5"] = out["geohash"].str[:5]

    geo = pd.DataFrame({"geohash": out["geohash"].drop_duplicates()})
    geo[["geo_lat", "geo_lon"]] = pd.DataFrame(geo["geohash"].map(decode_geohash).tolist(), index=geo.index)
    return out.merge(geo, on="geohash", how="left")


def clean_base(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    train["is_train"] = 1
    test["is_train"] = 0
    test[TARGET] = np.nan
    combined = pd.concat([train, test], axis=0, ignore_index=True)

    combined = parse_timestamp_features(combined)
    combined = add_geohash_features(combined)

    for col in ["RoadType", "Weather", "Temperature"]:
        combined[f"{col}_was_missing"] = combined[col].isna().astype(int)

    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
        combined[col] = combined[col].fillna("Unknown").astype(str).str.strip()

    combined["NumberofLanes"] = pd.to_numeric(combined["NumberofLanes"], errors="coerce")
    combined["NumberofLanes"] = combined["NumberofLanes"].fillna(train_df["NumberofLanes"].median()).astype(int)
    combined["has_many_lanes"] = (combined["NumberofLanes"] >= 4).astype(int)

    combined["Temperature"] = pd.to_numeric(combined["Temperature"], errors="coerce")
    train_part = combined[combined["is_train"] == 1].copy()
    temp_by_weather_hour = train_part.groupby(["Weather", "hour"])["Temperature"].median()
    temp_by_weather = train_part.groupby("Weather")["Temperature"].median()
    global_temp = train_part["Temperature"].median()

    def fill_temp(row):
        if pd.notna(row["Temperature"]):
            return row["Temperature"]
        key = (row["Weather"], row["hour"])
        if key in temp_by_weather_hour.index and pd.notna(temp_by_weather_hour.loc[key]):
            return temp_by_weather_hour.loc[key]
        if row["Weather"] in temp_by_weather.index and pd.notna(temp_by_weather.loc[row["Weather"]]):
            return temp_by_weather.loc[row["Weather"]]
        return global_temp

    combined["Temperature"] = combined.apply(fill_temp, axis=1)
    combined["temp_capped"] = combined["Temperature"].clip(
        train_part["Temperature"].quantile(0.01),
        train_part["Temperature"].quantile(0.99),
    )

    combined["large_vehicle_allowed"] = (combined["LargeVehicles"].str.lower() == "allowed").astype(int)
    combined["has_landmark"] = (combined["Landmarks"].str.lower() == "yes").astype(int)
    combined["road_weather"] = combined["RoadType"] + "_" + combined["Weather"]
    combined["lane_vehicle_interaction"] = combined["NumberofLanes"] * combined["large_vehicle_allowed"]
    combined["peak_lane_interaction"] = combined["is_peak_hour"] * combined["NumberofLanes"]

    train_clean = combined[combined["is_train"] == 1].drop(columns=["is_train"]).copy()
    test_clean = combined[combined["is_train"] == 0].drop(columns=["is_train", TARGET]).copy()
    return train_clean, test_clean


def add_frequency_features(train_df, test_df, cols):
    train = train_df.copy()
    test = test_df.copy()
    combined = pd.concat([train[cols], test[cols]], axis=0)
    total_rows = len(combined)
    for col in cols:
        counts = combined[col].value_counts(dropna=False)
        train[f"{col}_freq"] = train[col].map(counts).fillna(0) / total_rows
        test[f"{col}_freq"] = test[col].map(counts).fillna(0) / total_rows
    return train, test


def add_target_encoding_by_day(train_df, test_df, cols, smoothing=30):
    train = train_df.copy()
    test = test_df.copy()
    global_mean = train[TARGET].mean()
    days = sorted(train["day"].unique())

    for col in cols:
        encoded = pd.Series(index=train.index, dtype=float)
        for day in days:
            tr_mask = train["day"] < day
            val_mask = train["day"] == day
            if tr_mask.sum() == 0:
                encoded.loc[val_mask] = global_mean
                continue
            stats = train.loc[tr_mask].groupby(col)[TARGET].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            encoded.loc[val_mask] = train.loc[val_mask, col].map(smooth)

        full_stats = train.groupby(col)[TARGET].agg(["mean", "count"])
        full_smooth = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (
            full_stats["count"] + smoothing
        )
        train[f"{col}_target_mean"] = encoded.fillna(global_mean)
        test[f"{col}_target_mean"] = test[col].map(full_smooth).fillna(global_mean)
    return train, test


def add_day48_history(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    history = train[train["day"] == 48].copy()
    global_mean = history[TARGET].mean()

    specs = [
        ("hist_geo_time", ["geohash", "time_index"]),
        ("hist_geo_hour", ["geohash", "hour"]),
        ("hist_geo", ["geohash"]),
        ("hist_prefix5_time", ["geohash_prefix_5", "time_index"]),
        ("hist_prefix4_time", ["geohash_prefix_4", "time_index"]),
        ("hist_road_weather_time", ["road_weather", "time_index"]),
        ("hist_lane_time", ["NumberofLanes", "time_index"]),
        ("hist_time", ["time_index"]),
        ("hist_hour", ["hour"]),
    ]

    for feature_name, keys in specs:
        lookup = history.groupby(keys)[TARGET].mean().rename(feature_name).reset_index()
        train = train.merge(lookup, on=keys, how="left")
        test = test.merge(lookup, on=keys, how="left")

        # Day 48 has no previous-day label history. Mask it to avoid direct target leakage.
        train.loc[train["day"] == 48, feature_name] = np.nan

        fill_value = train.loc[train["day"] > 48, feature_name].median()
        if pd.isna(fill_value):
            fill_value = global_mean
        train[feature_name] = train[feature_name].fillna(fill_value)
        test[feature_name] = test[feature_name].fillna(fill_value)

    train["hist_geo_time_diff_from_geo"] = train["hist_geo_time"] - train["hist_geo"]
    test["hist_geo_time_diff_from_geo"] = test["hist_geo_time"] - test["hist_geo"]
    train["hist_geo_time_ratio"] = train["hist_geo_time"] / (train["hist_geo"] + 1e-6)
    test["hist_geo_time_ratio"] = test["hist_geo_time"] / (test["hist_geo"] + 1e-6)
    return train, test


def encode_categories(train_df, test_df, categorical_cols):
    train = train_df.copy()
    test = test_df.copy()
    for col in categorical_cols:
        combined = pd.concat([train[col], test[col]], axis=0).astype(str).fillna("Unknown")
        mapping = {value: idx for idx, value in enumerate(pd.Series(combined).drop_duplicates().sort_values())}
        train[col] = train[col].astype(str).fillna("Unknown").map(mapping).astype(int)
        test[col] = test[col].astype(str).fillna("Unknown").map(mapping).fillna(-1).astype(int)
    return train, test


def build_features(train_raw, test_raw):
    train, test = clean_base(train_raw, test_raw)
    train, test = add_frequency_features(
        train,
        test,
        ["geohash", "geohash_prefix_4", "geohash_prefix_5", "road_weather"],
    )
    train, test = add_target_encoding_by_day(
        train,
        test,
        ["geohash", "geohash_prefix_4", "geohash_prefix_5", "RoadType", "Weather", "road_weather"],
    )
    train, test = add_day48_history(train, test)

    categorical_features = [
        "geohash",
        "geohash_prefix_3",
        "geohash_prefix_4",
        "geohash_prefix_5",
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "Weather",
        "road_weather",
    ]
    numeric_features = [
        "day",
        "hour",
        "minute",
        "time_index",
        "total_minutes",
        "hour_sin",
        "hour_cos",
        "time_index_sin",
        "time_index_cos",
        "is_morning_peak",
        "is_evening_peak",
        "is_late_night",
        "is_peak_hour",
        "NumberofLanes",
        "has_many_lanes",
        "Temperature",
        "temp_capped",
        "RoadType_was_missing",
        "Weather_was_missing",
        "Temperature_was_missing",
        "large_vehicle_allowed",
        "has_landmark",
        "lane_vehicle_interaction",
        "peak_lane_interaction",
        "geo_lat",
        "geo_lon",
        "geohash_freq",
        "geohash_prefix_4_freq",
        "geohash_prefix_5_freq",
        "road_weather_freq",
        "geohash_target_mean",
        "geohash_prefix_4_target_mean",
        "geohash_prefix_5_target_mean",
        "RoadType_target_mean",
        "Weather_target_mean",
        "road_weather_target_mean",
        "hist_geo_time",
        "hist_geo_hour",
        "hist_geo",
        "hist_prefix5_time",
        "hist_prefix4_time",
        "hist_road_weather_time",
        "hist_lane_time",
        "hist_time",
        "hist_hour",
        "hist_geo_time_diff_from_geo",
        "hist_geo_time_ratio",
    ]
    features = numeric_features + categorical_features

    train_model = train[[ID_COL, TARGET] + features].copy()
    test_model = test[[ID_COL] + features].copy()
    train_model, test_model = encode_categories(train_model, test_model, categorical_features)

    for col in features:
        train_model[col] = pd.to_numeric(train_model[col], errors="coerce")
        test_model[col] = pd.to_numeric(test_model[col], errors="coerce")
        median = train_model[col].median()
        train_model[col] = train_model[col].fillna(median)
        test_model[col] = test_model[col].fillna(median)

    return train_model, test_model, features


def metrics(name, y_true, preds):
    rmse = mean_squared_error(y_true, preds) ** 0.5
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    print(f"{name:18s} RMSE={rmse:.6f} MAE={mae:.6f} R2={r2:.6f}")
    return rmse, mae, r2


def find_best_blend(y_true, pred_dict, step=0.02):
    names = list(pred_dict)
    best = (float("inf"), None, None)
    grid = np.arange(0, 1 + step / 2, step)
    for w0 in grid:
        for w1 in grid:
            w2 = 1 - w0 - w1
            if w2 < -1e-9:
                continue
            weights = np.array([w0, w1, w2])
            preds = sum(weights[i] * pred_dict[names[i]] for i in range(3))
            rmse = mean_squared_error(y_true, np.clip(preds, 0, 1)) ** 0.5
            if rmse < best[0]:
                best = (rmse, weights, np.clip(preds, 0, 1))
    return dict(zip(names, best[1])), best[2], best[0]


def main():
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_model, test_model, features = build_features(train_raw, test_raw)

    train_mask = train_model["day"] == 48
    valid_mask = train_model["day"] == 49
    X_train = train_model.loc[train_mask, features]
    y_train = train_model.loc[train_mask, TARGET]
    X_valid = train_model.loc[valid_mask, features]
    y_valid = train_model.loc[valid_mask, TARGET]
    X_full = train_model[features]
    y_full = train_model[TARGET]
    X_test = test_model[features]

    print(f"Train rows: {X_train.shape}, validation rows: {X_valid.shape}, test rows: {X_test.shape}")
    print(f"Feature count: {len(features)}")

    lgb_params = dict(
        n_estimators=2600,
        learning_rate=0.025,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.2,
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    xgb_params = dict(
        n_estimators=2200,
        learning_rate=0.025,
        max_depth=8,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.03,
        reg_lambda=0.35,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cb_params = dict(
        iterations=2200,
        learning_rate=0.025,
        depth=8,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )

    print("\nValidation on day 49 early rows:")
    lgb_val = LGBMRegressor(**lgb_params)
    xgb_val = XGBRegressor(**xgb_params)
    cb_val = CatBoostRegressor(**cb_params)
    lgb_val.fit(X_train, y_train)
    xgb_val.fit(X_train, y_train)
    cb_val.fit(X_train, y_train)

    valid_lgb = np.clip(lgb_val.predict(X_valid), 0, 1)
    valid_xgb = np.clip(xgb_val.predict(X_valid), 0, 1)
    valid_cb = np.clip(cb_val.predict(X_valid), 0, 1)
    blend_weights, valid_ensemble, _ = find_best_blend(
        y_valid,
        {"LightGBM": valid_lgb, "XGBoost": valid_xgb, "CatBoost": valid_cb},
        step=0.02,
    )

    metrics("LightGBM", y_valid, valid_lgb)
    metrics("XGBoost", y_valid, valid_xgb)
    metrics("CatBoost", y_valid, valid_cb)
    metrics("Ensemble", y_valid, valid_ensemble)
    print("Best blend weights:", {k: round(v, 3) for k, v in blend_weights.items()})

    print("\nTraining final models on all available training rows...")
    lgb_final = LGBMRegressor(**lgb_params)
    xgb_final = XGBRegressor(**xgb_params)
    cb_final = CatBoostRegressor(**cb_params)
    lgb_final.fit(X_full, y_full)
    xgb_final.fit(X_full, y_full)
    cb_final.fit(X_full, y_full)

    pred_lgb = np.clip(lgb_final.predict(X_test), 0, 1)
    pred_xgb = np.clip(xgb_final.predict(X_test), 0, 1)
    pred_cb = np.clip(cb_final.predict(X_test), 0, 1)
    final_pred = np.clip(
        blend_weights["LightGBM"] * pred_lgb
        + blend_weights["XGBoost"] * pred_xgb
        + blend_weights["CatBoost"] * pred_cb,
        0,
        1,
    )

    submission = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: final_pred})
    submission.to_csv(DATA_DIR / "submission.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(DATA_DIR / "model_feature_list.csv", index=False)
    print("\nSaved submission.csv")
    print(submission.head().to_string(index=False))
    print(submission[TARGET].describe().to_string())


if __name__ == "__main__":
    main()
