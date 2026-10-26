"""
BaliAir AI - Data Collector Module
Mengambil data pemantauan kualitas udara langsung dari Bali Air Dispatch API
(https://baliairdispatch.com/api/live)
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional, Tuple
import pandas as pd
import requests

API_URL = "https://baliairdispatch.com/api/live"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# Standar Warna & Kategori AQI (US EPA & WHO)
AQI_COLOR_MAP = {
    "good": {"color": "#4f7a3f", "label": "Good (Baik)", "range": "0 - 12 µg/m³"},
    "mod": {"color": "#b8892a", "label": "Moderate (Sedang)", "range": "12.1 - 35.4 µg/m³"},
    "usg": {"color": "#c25d22", "label": "Unhealthy for Sensitive Groups", "range": "35.5 - 55.4 µg/m³"},
    "unh": {"color": "#a32020", "label": "Unhealthy (Tidak Sehat)", "range": "55.5 - 150.4 µg/m³"},
    "vunh": {"color": "#6e1414", "label": "Very Unhealthy (Sangat Tidak Sehat)", "range": "150.5 - 250.4 µg/m³"},
    "haz": {"color": "#4a1142", "label": "Hazardous (Berbahaya)", "range": "> 250.4 µg/m³"},
    "offline": {"color": "#7f8c8d", "label": "Offline", "range": "No recent data"},
}


def get_pm25_color_and_category(pm25: Optional[float], is_offline: bool = False) -> Tuple[str, str]:
    """Menentukan kode warna HEX dan label kategori berdasarkan nilai PM2.5"""
    if is_offline or pm25 is None or pd.isna(pm25):
        return AQI_COLOR_MAP["offline"]["color"], AQI_COLOR_MAP["offline"]["label"]
    
    if pm25 <= 12.0:
        return AQI_COLOR_MAP["good"]["color"], AQI_COLOR_MAP["good"]["label"]
    elif pm25 <= 35.4:
        return AQI_COLOR_MAP["mod"]["color"], AQI_COLOR_MAP["mod"]["label"]
    elif pm25 <= 55.4:
        return AQI_COLOR_MAP["usg"]["color"], AQI_COLOR_MAP["usg"]["label"]
    elif pm25 <= 150.4:
        return AQI_COLOR_MAP["unh"]["color"], AQI_COLOR_MAP["unh"]["label"]
    elif pm25 <= 250.4:
        return AQI_COLOR_MAP["vunh"]["color"], AQI_COLOR_MAP["vunh"]["label"]
    else:
        return AQI_COLOR_MAP["haz"]["color"], AQI_COLOR_MAP["haz"]["label"]


def fetch_live_stations(save_raw: bool = True) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Mengambil data stasiun live dari baliairdispatch.com
    Mengembalikan (DataFrame stasiun, timestamp_data)
    """
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=12)
        response.raise_for_status()
        data = response.json()
        
        timestamp = data.get("ts", datetime.now(timezone.utc).isoformat())
        raw_stations = data.get("stations", [])
        
        if save_raw:
            os.makedirs("data/raw", exist_ok=True)
            with open("data/raw/live_latest.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        records = []
        for s in raw_stations:
            lat = s.get("lat")
            lon = s.get("lon")
            if lat is None or lon is None:
                continue
            
            is_offline = bool(s.get("off", False)) or s.get("category") == "Offline"
            pm25 = s.get("pm25")
            color, cat_label = get_pm25_color_and_category(pm25, is_offline)
            
            records.append({
                "id": s.get("id"),
                "name": s.get("name", "Tanpa Nama"),
                "source": s.get("source", "Unknown"),
                "type": s.get("type", "Sensor"),
                "lat": float(lat),
                "lon": float(lon),
                "pm25": float(pm25) if pm25 is not None else None,
                "pm10": float(s.get("pm10")) if s.get("pm10") is not None else None,
                "pm1": float(s.get("pm1")) if s.get("pm1") is not None else None,
                "temperature": float(s.get("temperature")) if s.get("temperature") is not None else None,
                "humidity": float(s.get("humidity")) if s.get("humidity") is not None else None,
                "aqi": s.get("aqi"),
                "category": s.get("category", cat_label),
                "category_label": cat_label,
                "color": color,
                "is_offline": is_offline,
                "last_seen": s.get("lastSeen"),
                "feed": s.get("feed", "api")
            })
            
        df = pd.DataFrame(records)
        return df, timestamp

    except Exception as e:
        print(f"Error fetching live stations: {e}")
        # Coba fallback ke file lokal jika ada
        if os.path.exists("data/raw/live_latest.json"):
            with open("data/raw/live_latest.json", "r", encoding="utf-8") as f:
                cached = json.load(f)
                return pd.DataFrame(cached.get("stations", [])), cached.get("ts")
        return pd.DataFrame(), None


if __name__ == "__main__":
    df, ts = fetch_live_stations()
    print(f"Berhasil mengunduh {len(df)} stasiun. Waktu data: {ts}")
    print(df.head(3))
