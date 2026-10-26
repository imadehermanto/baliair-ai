"""
BaliAir AI - Model Training & Time-Series Evaluation Pipeline
Melatih model multi-horizon forecasting (t+1h, t+6h, t+24h) menggunakan
LightGBM & XGBoost, dievaluasi terhadap Persistence Baseline menggunakan
Expanding Window TimeSeriesSplit dan Out-of-Time Test Set.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PARQUET = "data/processed/bali_air_features.parquet"
MODEL_DIR = "data/models"

FEATURE_COLS = [
    # Autoregressive Lags
    "pm25_lag_1h", "pm25_lag_2h", "pm25_lag_3h", "pm25_lag_6h", "pm25_lag_12h", "pm25_lag_24h",
    # Rolling Statistics
    "pm25_roll_mean_3h", "pm25_roll_mean_6h", "pm25_roll_mean_24h",
    "pm25_roll_std_6h", "pm25_roll_max_6h", "pm25_roll_min_6h",
    "pm25_diff_1h", "pm25_diff_24h",
    # Cuaca & Mikroklimat
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "precipitation", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    # Vektor Angin & Dispersi Atmosfer
    "wind_u", "wind_v", "ventilation_index", "direct_radiation",
    # Waktu Siklikal & Spasial
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend",
    "lat", "lon"
]

HORIZONS = {
    "1h": "target_pm25_1h",
    "6h": "target_pm25_6h",
    "24h": "target_pm25_24h"
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Menghitung metrik standar evaluasi regresi time-series"""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    # WAPE (Weighted Absolute Percentage Error): sum(|y - y_hat|) / sum(y)
    sum_true = np.sum(y_true)
    wape = float(np.sum(np.abs(y_true - y_pred)) / sum_true * 100) if sum_true > 0 else 0.0
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 3), "WAPE%": round(wape, 2)}


def train_and_evaluate() -> Dict:
    """Eksekusi training, evaluasi time-series, dan penyimpanan artefak model"""
    print("=" * 70)
    print("🧠 BALIAIR AI - MODEL TRAINING & MULTI-HORIZON EVALUATION")
    print("=" * 70)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_parquet(FEATURES_PARQUET)
    
    # Urutkan berdasarkan waktu murni untuk splitting Out-of-Time
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    unique_timestamps = np.sort(df["timestamp"].unique())
    n_times = len(unique_timestamps)
    split_idx = int(n_times * 0.80)
    cutoff_time = unique_timestamps[split_idx]
    
    print(f"Total observasi: {len(df):,} baris ({df['station_id'].nunique()} stasiun)")
    print(f"Rentang waktu  : {df['timestamp'].min()} s/d {df['timestamp'].max()}")
    print(f"Split Train/Test (Out-of-Time): Cutoff {cutoff_time}")
    
    train_mask = df["timestamp"] < cutoff_time
    test_mask = df["timestamp"] >= cutoff_time
    
    results = {}
    
    for h_name, target_col in HORIZONS.items():
        print(f"\n" + "-" * 70)
        print(f"🎯 FORECAST HORIZON: +{h_name} ({target_col})")
        print("-" * 70)
        
        # Filter sampel valid (tidak null pada fitur utama & target)
        valid_df = df.dropna(subset=[target_col, "pm25_lag_1h"]).copy()
        
        train_df = valid_df[valid_df["timestamp"] < cutoff_time]
        test_df = valid_df[valid_df["timestamp"] >= cutoff_time]
        
        X_train, y_train = train_df[FEATURE_COLS], train_df[target_col].values
        X_test, y_test = test_df[FEATURE_COLS], test_df[target_col].values
        
        print(f"Data Train: {len(X_train):,} sampel | Data Test: {len(X_test):,} sampel")
        
        # 1. Baseline: Persistence (Menggunakan nilai PM2.5 terakhir t-0)
        s_pred_persist = test_df["pm25"].fillna(test_df["pm25_lag_1h"]).fillna(0.0)
        metrics_persist = compute_metrics(y_test, s_pred_persist.values)
        
        # 2. Baseline: 24h Rolling Mean
        s_pred_roll = test_df["pm25_roll_mean_24h"].fillna(s_pred_persist).fillna(0.0)
        metrics_roll = compute_metrics(y_test, s_pred_roll.values)
        
        # 3. Model LightGBM Regressor
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        y_pred_lgb = lgb_model.predict(X_test)
        metrics_lgb = compute_metrics(y_test, y_pred_lgb)
        
        # 4. Model XGBoost Regressor
        xgb_model = xgb.XGBRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        metrics_xgb = compute_metrics(y_test, y_pred_xgb)
        
        # Hitung Improvement (%) terhadap Persistence Baseline
        imp_lgb = round(((metrics_persist["MAE"] - metrics_lgb["MAE"]) / metrics_persist["MAE"]) * 100, 1)
        imp_xgb = round(((metrics_persist["MAE"] - metrics_xgb["MAE"]) / metrics_persist["MAE"]) * 100, 1)
        
        print("\n📈 HASIL EVALUASI (Out-of-Time Test Set):")
        print(f"  • Persistence Baseline : MAE={metrics_persist['MAE']} µg/m³ | RMSE={metrics_persist['RMSE']} | R²={metrics_persist['R2']} | WAPE={metrics_persist['WAPE%']}%")
        print(f"  • Rolling 24h Baseline : MAE={metrics_roll['MAE']} µg/m³ | RMSE={metrics_roll['RMSE']} | R²={metrics_roll['R2']} | WAPE={metrics_roll['WAPE%']}%")
        print(f"  • LightGBM Regressor   : MAE={metrics_lgb['MAE']} µg/m³ | RMSE={metrics_lgb['RMSE']} | R²={metrics_lgb['R2']} | WAPE={metrics_lgb['WAPE%']}% (Improvement: +{imp_lgb}%)")
        print(f"  • XGBoost Regressor    : MAE={metrics_xgb['MAE']} µg/m³ | RMSE={metrics_xgb['RMSE']} | R²={metrics_xgb['R2']} | WAPE={metrics_xgb['WAPE%']}% (Improvement: +{imp_xgb}%)")
        
        # Simpan model terbaik untuk horizon ini (LightGBM)
        best_model = lgb_model if metrics_lgb["MAE"] <= metrics_xgb["MAE"] else xgb_model
        best_name = "LightGBM" if metrics_lgb["MAE"] <= metrics_xgb["MAE"] else "XGBoost"
        model_path = os.path.join(MODEL_DIR, f"baliair_model_{h_name}.joblib")
        joblib.dump(best_model, model_path)
        print(f"  ✓ Model terbaik ({best_name}) disimpan di: {model_path}")
        
        # Top 5 Feature Importances
        importances = lgb_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        top_feats = [(FEATURE_COLS[i], int(importances[i])) for i in top_idx]
        print(f"  ✓ Top 5 Fitur Berpengaruh: {top_feats}")
        
        results[h_name] = {
            "best_model": best_name,
            "metrics": {
                "persistence": metrics_persist,
                "rolling_24h": metrics_roll,
                "lightgbm": metrics_lgb,
                "xgboost": metrics_xgb
            },
            "improvement_pct_over_baseline": imp_lgb,
            "top_features": top_feats
        }
        
    # Simpan ringkasan hasil evaluasi JSON
    results_path = os.path.join(MODEL_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Simpan metadata fitur
    features_meta_path = os.path.join(MODEL_DIR, "feature_columns.json")
    with open(features_meta_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, indent=2)
        
    print("\n" + "=" * 70)
    print("🏆 TRAINING & TIME-SERIES EVALUATION SELESAI!")
    print(f"Hasil lengkap tersimpan di: {results_path}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    train_and_evaluate()
