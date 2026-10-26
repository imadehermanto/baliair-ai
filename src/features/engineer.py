"""
BaliAir AI - Time-Series Feature Engineering Module
Menghasilkan matriks fitur lengkap untuk pemodelan time-series multi-horizon forecasting
(Lags, Rolling Aggregations, Wind Vectors, Cyclical Time, dan Multi-Horizon Targets)
Bebas data leakage (komputasi group-wise per stasiun hanya melihat data masa lalu).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PARQUET = "data/processed/bali_air_weather_hourly.parquet"
OUTPUT_PARQUET = "data/processed/bali_air_features.parquet"


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Membangun fitur time-series per stasiun tanpa data leakage"""
    print(f"Memproses {len(df):,} baris data awal...")
    
    # Pastikan urutan kronologis per stasiun
    df = df.sort_values(by=["station_id", "timestamp"]).copy()
    
    # 1. Multi-Horizon Forecasting Targets (Masa Depan: t+1h, t+6h, t+24h)
    print("  [1/5] Membuat Multi-Horizon Targets (t+1h, t+6h, t+24h)...")
    grouped = df.groupby("station_id")
    df["target_pm25_1h"] = grouped["pm25"].shift(-1)
    df["target_pm25_6h"] = grouped["pm25"].shift(-6)
    df["target_pm25_24h"] = grouped["pm25"].shift(-24)
    
    # 2. Autoregressive Lag Features (Masa Lalu: t-1 s/d t-24)
    print("  [2/5] Menghasilkan Autoregressive Lags (t-1h, t-2h, t-3h, t-6h, t-12h, t-24h)...")
    df["pm25_lag_1h"] = grouped["pm25"].shift(1)
    df["pm25_lag_2h"] = grouped["pm25"].shift(2)
    df["pm25_lag_3h"] = grouped["pm25"].shift(3)
    df["pm25_lag_6h"] = grouped["pm25"].shift(6)
    df["pm25_lag_12h"] = grouped["pm25"].shift(12)
    df["pm25_lag_24h"] = grouped["pm25"].shift(24)
    
    df["temp_lag_1h"] = grouped["temperature_2m"].shift(1)
    df["wind_speed_lag_1h"] = grouped["wind_speed_10m"].shift(1)
    
    # 3. Rolling Window Statistics (Hanya melihat ke belakang)
    print("  [3/5] Menghitung Rolling Statistics (mean, std, min, max, delta)...")
    # Rolling 3 jam
    df["pm25_roll_mean_3h"] = grouped["pm25"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    # Rolling 6 jam
    df["pm25_roll_mean_6h"] = grouped["pm25"].transform(lambda x: x.shift(1).rolling(6, min_periods=2).mean())
    df["pm25_roll_std_6h"] = grouped["pm25"].transform(lambda x: x.shift(1).rolling(6, min_periods=2).std())
    df["pm25_roll_max_6h"] = grouped["pm25"].transform(lambda x: x.shift(1).rolling(6, min_periods=2).max())
    df["pm25_roll_min_6h"] = grouped["pm25"].transform(lambda x: x.shift(1).rolling(6, min_periods=2).min())
    # Rolling 24 jam (Baseline harian)
    df["pm25_roll_mean_24h"] = grouped["pm25"].transform(lambda x: x.shift(1).rolling(24, min_periods=6).mean())
    
    # Delta / Kecepatan Perubahan Konsentrasi
    df["pm25_diff_1h"] = df["pm25_lag_1h"] - df["pm25_lag_2h"]
    df["pm25_diff_24h"] = df["pm25_lag_1h"] - df["pm25_lag_24h"]
    
    # 4. Interaksi Meteorologi & Vektor Angin (Atmospheric Physics)
    print("  [4/5] Mengurai Vektor Angin (u, v) dan Indikator Dispersi Atmosfer...")
    rad = np.radians(df["wind_direction_10m"])
    # u = komponen zonal (Barat -> Timur), v = komponen meridional (Selatan -> Utara)
    df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
    df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)
    
    # Indeks Ventilasi (perkalian kecepatan angin dan selisih titik embun)
    # Nilai rendah mengindikasikan polutan terperangkap dekat permukaan
    dew_spread = np.maximum(df["temperature_2m"] - df["dew_point_2m"], 0.1)
    df["ventilation_index"] = df["wind_speed_10m"] * dew_spread
    
    # 5. Fitur Kalender Siklikal (Diurnal Cycle)
    print("  [5/5] Membuat Cyclical Time Encodings (sin/cos hour, day of week)...")
    hour = df["timestamp"].dt.hour
    dayofweek = df["timestamp"].dt.day_of_week
    
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
    df["is_weekend"] = (dayofweek >= 5).astype(int)
    
    return df


def run_feature_pipeline() -> pd.DataFrame:
    """Eksekusi pipeline rekayasa fitur dan simpan ke parquet"""
    print("=" * 60)
    print("⚙️ BALIAIR AI - FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    if not Path(INPUT_PARQUET).exists():
        raise FileNotFoundError(f"File {INPUT_PARQUET} belum tersedia. Jalankan harmonizer.py terlebih dahulu.")
        
    raw_df = pd.read_parquet(INPUT_PARQUET)
    feature_df = build_feature_matrix(raw_df)
    
    # Simpan hasil
    feature_df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", index=False)
    
    print("=" * 60)
    print(f"✅ FEATURE PIPELINE SELESAI!")
    print(f"Dataset fitur tersimpan di : {OUTPUT_PARQUET}")
    print(f"Dimensi matriks fitur      : {feature_df.shape[0]:,} baris x {feature_df.shape[1]} kolom")
    print(f"Fitur baru yang ditambahkan: {feature_df.shape[1] - raw_df.shape[1]} fitur")
    print("=" * 60)
    return feature_df


if __name__ == "__main__":
    run_feature_pipeline()
