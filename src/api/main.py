"""
BaliAir AI - Production REST API for Air Quality Multi-Horizon Forecasting
FastAPI service melayani inference real-time untuk prediksi PM2.5 di Bali (+1h, +6h, +24h)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.collector import get_pm25_color_and_category

MODEL_DIR = Path("data/models")
FEATURES_PARQUET = Path("data/processed/bali_air_features.parquet")

# Inisialisasi FastAPI
app = FastAPI(
    title="BaliAir AI — Air Quality Forecasting API",
    description="End-to-end Machine Learning REST API for Multi-Horizon PM2.5 Concentration Forecasting in Bali.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Artifacts & Feature Schema
models = {}
feature_cols = []
eval_results = {}


@app.on_event("startup")
def load_artifacts():
    global models, feature_cols, eval_results
    for h in ["1h", "6h", "24h"]:
        model_path = MODEL_DIR / f"baliair_model_{h}.joblib"
        if model_path.exists():
            models[h] = joblib.load(model_path)
            print(f"✓ Model horizon +{h} berhasil dimuat.")
            
    cols_path = MODEL_DIR / "feature_columns.json"
    if cols_path.exists():
        with open(cols_path, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
            
    eval_path = MODEL_DIR / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_results = json.load(f)


# Schemas
class ForecastItem(BaseModel):
    horizon: str = Field(..., description="Waktu proyeksi (+1h, +6h, +24h)")
    predicted_pm25: float = Field(..., description="Konsentrasi PM2.5 hasil prediksi (µg/m³)")
    ci_lower: float = Field(..., description="Batas bawah interval kepercayaan 90%")
    ci_upper: float = Field(..., description="Batas atas interval kepercayaan 90%")
    category: str = Field(..., description="Kategori kualitas udara standar WHO/EPA")
    color_hex: str = Field(..., description="Kode warna visualisasi")


class StationForecastResponse(BaseModel):
    station_id: str
    station_name: str
    source: str
    coordinates: Dict[str, float]
    current_observed_pm25: Optional[float]
    latest_timestamp_utc: str
    forecasts: List[ForecastItem]
    model_metadata: Dict[str, Any]


@app.get("/api/v1/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint dan status ketersediaan model"""
    return {
        "status": "healthy",
        "service": "BaliAir AI Forecasting API",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "loaded_models": list(models.keys()),
        "total_features": len(feature_cols)
    }


@app.get("/api/v1/metrics", tags=["Evaluation"])
def get_model_metrics():
    """Mengambil hasil evaluasi time-series out-of-time untuk seluruh horizon"""
    if not eval_results:
        raise HTTPException(status_code=404, detail="Hasil evaluasi belum ditemukan.")
    return eval_results


@app.get("/api/v1/forecast/{station_id}", response_model=StationForecastResponse, tags=["Inference"])
def get_station_forecast(station_id: str):
    """
    Melakukan inference prediksi PM2.5 multi-horizon (+1 jam, +6 jam, +24 jam)
    berdasarkan data sensor & meteorologi terkini stasiun.
    """
    if not FEATURES_PARQUET.exists():
        raise HTTPException(status_code=503, detail="Dataset fitur belum siap.")
        
    df = pd.read_parquet(FEATURES_PARQUET)
    stn_df = df[df["station_id"] == station_id].sort_values("timestamp")
    
    if stn_df.empty:
        raise HTTPException(status_code=404, detail=f"Stasiun dengan ID '{station_id}' tidak ditemukan.")
        
    latest_row = stn_df.iloc[-1]
    X_latest = pd.DataFrame([latest_row[feature_cols]])
    
    forecast_items = []
    
    for h in ["1h", "6h", "24h"]:
        if h not in models:
            continue
        model = models[h]
        pred_val = float(model.predict(X_latest)[0])
        pred_val = max(0.0, round(pred_val, 2))  # Polutan tidak boleh negatif
        
        # Uncertainty Interval (1.645 * RMSE untuk ~90% CI)
        rmse_val = eval_results.get(h, {}).get("metrics", {}).get("lightgbm", {}).get("RMSE", 10.0)
        ci_lower = max(0.0, round(pred_val - 1.645 * rmse_val, 2))
        ci_upper = round(pred_val + 1.645 * rmse_val, 2)
        
        color, cat_label = get_pm25_color_and_category(pred_val)
        
        forecast_items.append(ForecastItem(
            horizon=f"+{h}",
            predicted_pm25=pred_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            category=cat_label,
            color_hex=color
        ))
        
    return StationForecastResponse(
        station_id=station_id,
        station_name=str(latest_row.get("station_name", "Unknown")),
        source=str(latest_row.get("source", "Unknown")),
        coordinates={
            "lat": float(latest_row["lat"]),
            "lon": float(latest_row["lon"])
        },
        current_observed_pm25=float(latest_row["pm25"]) if pd.notna(latest_row["pm25"]) else None,
        latest_timestamp_utc=str(latest_row["timestamp"]),
        forecasts=forecast_items,
        model_metadata={
            "engine": "LightGBM & XGBoost Multi-Horizon Regressor",
            "feature_count": len(feature_cols),
            "meteorological_covariates": "Open-Meteo High Resolution Weather"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
