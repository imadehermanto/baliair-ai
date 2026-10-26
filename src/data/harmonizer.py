"""
BaliAir AI - Data Harmonizer & Cleanser (ETL Pipeline)
Mengintegrasikan data sensor 15-menitan dari baliairdispatch dengan data cuaca Open-Meteo,
melakukan resampling ke frekuensi 1-jam (hourly), imputasi terbatas, dan menyimpan ke format Apache Parquet.
"""

import os
import sys
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_history import download_all_histories, RAW_DIR
from src.data.weather_enrichment import fetch_weather_for_grid

PROCESSED_FILE = "data/processed/bali_air_weather_hourly.parquet"


def load_raw_sensor_data() -> List[Dict]:
    """Membaca seluruh file JSON histori sensor yang tersimpan"""
    files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    stations_data = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
                if d and "station" in d and "points" in d and d["points"]:
                    stations_data.append({
                        "station": d["station"],
                        "points": d["points"]
                    })
        except Exception as e:
            print(f"Error membaca {fp}: {e}")
    return stations_data


def process_station_timeseries(station_info: Dict, points: List[Dict]) -> Optional[pd.DataFrame]:
    """Mengonversi titik 15-menitan ke agregasi tingkat jam (hourly)"""
    stn_id = station_info.get("station_id")
    lat = station_info.get("lat")
    lon = station_info.get("lon")
    name = station_info.get("name")
    source = station_info.get("source")
    stn_type = station_info.get("type")
    
    if lat is None or lon is None or not points:
        return None
        
    df_pts = pd.DataFrame(points)
    
    # Standardisasi timestamp
    if "ts" in df_pts.columns:
        df_pts["datetime"] = pd.to_datetime(df_pts["ts"], unit="s", utc=True)
    elif "station_till" in df_pts.columns:
        df_pts["datetime"] = pd.to_datetime(df_pts["station_till"], utc=True)
    else:
        return None
        
    df_pts = df_pts.sort_values("datetime").drop_duplicates(subset=["datetime"])
    df_pts.set_index("datetime", inplace=True)
    
    # Kolom numerik yang relevan
    num_cols = ["pm25", "pm10", "pm1", "temperature", "humidity"]
    for c in num_cols:
        if c not in df_pts.columns:
            df_pts[c] = np.nan
        else:
            df_pts[c] = pd.to_numeric(df_pts[c], errors="coerce")
            
    # Pembersihan noise sensor
    df_pts.loc[df_pts["pm25"] < 0, "pm25"] = np.nan
    df_pts.loc[df_pts["pm25"] > 800, "pm25"] = np.nan  # Filter sensor glitched
    
    # Resample ke interval 1 jam
    hourly = df_pts.resample("1h").agg({
        "pm25": ["mean", "max", "min", "std"],
        "pm10": "mean",
        "pm1": "mean",
        "temperature": "mean",
        "humidity": "mean"
    })
    
    # Flatten multiindex columns
    hourly.columns = [
        "pm25", "pm25_max", "pm25_min", "pm25_std",
        "pm10", "pm1", "temp_sensor", "humidity_sensor"
    ]
    
    # Forward-fill untuk gap kecil (maksimal 2 jam berturut-turut)
    hourly["pm25"] = hourly["pm25"].ffill(limit=2)
    
    # Tambahkan metadata stasiun
    hourly["station_id"] = stn_id
    hourly["station_name"] = name
    hourly["source"] = source
    hourly["station_type"] = stn_type
    hourly["lat"] = lat
    hourly["lon"] = lon
    hourly["grid_lat"] = round(lat, 2)
    hourly["grid_lon"] = round(lon, 2)
    
    hourly.reset_index(inplace=True)
    hourly.rename(columns={"datetime": "timestamp"}, inplace=True)
    return hourly


def run_etl_pipeline(force_download: bool = False) -> pd.DataFrame:
    """Eksekusi pipeline lengkap: Download Sensor -> Resample Hourly -> Fetch Weather -> Join Parquet"""
    print("=" * 60)
    print("🚀 BALIAIR AI - DATA ENGINEERING & HARMONIZATION PIPELINE")
    print("=" * 60)
    
    # Langkah 1: Download / Validasi Data Historis
    raw_files = glob.glob(os.path.join(RAW_DIR, "*.json"))
    if force_download or len(raw_files) < 10:
        download_all_histories(force=force_download)
        
    stations_data = load_raw_sensor_data()
    print(f"\n[1/4] Memproses data sensor dari {len(stations_data)} stasiun...")
    
    all_hourly_stations = []
    for s in stations_data:
        df_stn = process_station_timeseries(s["station"], s["points"])
        if df_stn is not None and not df_stn.empty:
            all_hourly_stations.append(df_stn)
            
    if not all_hourly_stations:
        raise ValueError("Tidak ada data sensor yang berhasil diproses!")
        
    df_sensors = pd.concat(all_hourly_stations, ignore_index=True)
    print(f"  ✓ Terkumpul {len(df_sensors):,} baris observasi per jam.")
    print(f"  ✓ Rentang waktu: {df_sensors['timestamp'].min()} s/d {df_sensors['timestamp'].max()}")
    
    # Langkah 2: Ambil Data Cuaca Open-Meteo untuk Setiap Grid
    unique_grids = df_sensors[["grid_lat", "grid_lon"]].drop_duplicates().values
    print(f"\n[2/4] Mengambil variabel meteorologi untuk {len(unique_grids)} grid di Bali...")
    
    weather_dfs = []
    for g_lat, g_lon in unique_grids:
        try:
            w_df = fetch_weather_for_grid(g_lat, g_lon, past_days=7)
            weather_dfs.append(w_df)
        except Exception as e:
            print(f"  Warning: Cuaca gagal untuk ({g_lat}, {g_lon}): {e}")
            
    if not weather_dfs:
        raise ValueError("Gagal mengambil data cuaca dari Open-Meteo.")
        
    df_weather = pd.concat(weather_dfs, ignore_index=True)
    df_weather.rename(columns={"time": "timestamp"}, inplace=True)
    print(f"  ✓ Data cuaca terkumpul: {len(df_weather):,} baris.")
    
    # Langkah 3: Merge Sensor & Weather Covariates
    print("\n[3/4] Melakukan harmonisasi & penggabungan fitur (Merge on timestamp + grid)...")
    merged = pd.merge(
        df_sensors,
        df_weather,
        on=["timestamp", "grid_lat", "grid_lon"],
        how="inner"
    )
    
    # Sort secara time-series per stasiun
    merged.sort_values(by=["station_id", "timestamp"], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    
    # Langkah 4: Simpan ke Parquet
    print("\n[4/4] Validasi akhir & penyimpanan ke Parquet...")
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    merged.to_parquet(PROCESSED_FILE, engine="pyarrow", index=False)
    
    print("=" * 60)
    print(f"✅ ETL PIPELINE BERHASIL!")
    print(f"Dataset tersimpan di: {PROCESSED_FILE}")
    print(f"Dimensi dataset : {merged.shape[0]:,} baris x {merged.shape[1]} kolom")
    print(f"Jumlah stasiun  : {merged['station_id'].nunique()} stasiun")
    print(f"Fitur tersedia  : {list(merged.columns)}")
    print(f"Missing PM2.5   : {merged['pm25'].isna().sum()} ({merged['pm25'].isna().mean()*100:.2f}%)")
    print("=" * 60)
    return merged


if __name__ == "__main__":
    run_etl_pipeline(force_download=False)
