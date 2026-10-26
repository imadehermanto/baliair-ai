"""
BaliAir AI - Weather Enrichment Pipeline (Open-Meteo API)
Mengunduh variabel meteorologi tingkat jam (hourly) untuk setiap zona lokasi sensor di Bali
"""

import os
import json
import time
from typing import Dict, List, Tuple
import pandas as pd
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_RAW_DIR = "data/raw/weather"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "direct_radiation",
]


def cluster_coordinates(coords: List[Tuple[float, float]], precision: int = 2) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
    """
    Mengelompokkan koordinat stasiun yang berdekatan (grid ~1 km) 
    untuk efisiensi kuota API cuaca Open-Meteo.
    """
    clusters = {}
    for lat, lon in coords:
        grid_key = (round(lat, precision), round(lon, precision))
        if grid_key not in clusters:
            clusters[grid_key] = []
        clusters[grid_key].append((lat, lon))
    return clusters


def fetch_weather_for_grid(lat: float, lon: float, past_days: int = 7) -> pd.DataFrame:
    """Mengambil time-series cuaca per grid koordinat"""
    os.makedirs(WEATHER_RAW_DIR, exist_ok=True)
    cache_path = os.path.join(WEATHER_RAW_DIR, f"weather_{lat:.2f}_{lon:.2f}.json")
    
    # Cek cache
    if os.path.exists(cache_path):
        file_age = time.time() - os.path.getmtime(cache_path)
        if file_age < 3 * 3600:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                hourly = data.get("hourly", {})
                df = pd.DataFrame(hourly)
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df["grid_lat"] = lat
                df["grid_lon"] = lon
                return df
                
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": WEATHER_VARS,
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "UTC"
    }
    
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["grid_lat"] = lat
    df["grid_lon"] = lon
    return df


def fetch_weather_for_all_stations(stations_metadata: List[Dict]) -> pd.DataFrame:
    """Mengunduh dan menggabungkan data cuaca untuk seluruh titik stasiun di Bali"""
    coords = [(s["lat"], s["lon"]) for s in stations_metadata if s.get("lat") is not None and s.get("lon") is not None]
    clusters = cluster_coordinates(coords, precision=2)
    
    print(f"Mengelompokkan {len(coords)} stasiun menjadi {len(clusters)} zona grid cuaca di Bali...")
    
    dfs = []
    for idx, (grid_lat, grid_lon) in enumerate(clusters.keys(), 1):
        print(f"  [{idx}/{len(clusters)}] Mengunduh cuaca zona ({grid_lat:.2f}, {grid_lon:.2f})...", end=" ", flush=True)
        try:
            df_grid = fetch_weather_for_grid(grid_lat, grid_lon, past_days=7)
            dfs.append(df_grid)
            print(f"✓ ({len(df_grid)} jam observasi cuaca)")
        except Exception as e:
            print(f"✗ Gagal: {e}")
        time.sleep(0.1)
        
    if dfs:
        all_weather = pd.concat(dfs, ignore_index=True)
        return all_weather
    return pd.DataFrame()


if __name__ == "__main__":
    test_lat, test_lon = -8.65, 115.21
    df = fetch_weather_for_grid(test_lat, test_lon)
    print(f"Test cuaca Denpasar ({test_lat}, {test_lon}):")
    print(df.head(3))
