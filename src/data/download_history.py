"""
BaliAir AI - Ingestion Script: Unduh Data Time-Series Historis (7 Hari)
Mengunduh time-series 15-menitan dari seluruh stasiun aktif di baliairdispatch.com
"""

import os
import json
import time
from typing import Dict, List, Any
import requests
from datetime import datetime, timezone

API_BASE = "https://baliairdispatch.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}
RAW_DIR = "data/raw/history_7d"


def get_active_stations() -> List[Dict[str, Any]]:
    """Mengambil daftar stasiun aktif dari endpoint live"""
    url = f"{API_BASE}/api/live"
    resp = requests.get(url, headers=HEADERS, timeout=12)
    resp.raise_for_status()
    data = resp.json()
    stations = data.get("stations", [])
    # Filter stasiun yang tidak offline permanen (bukan tombstone)
    active = [s for s in stations if not s.get("off", False) and s.get("id")]
    return active


def download_station_history(station_id: str, force: bool = False) -> Dict[str, Any]:
    """Mengunduh history 7d resolusi 15 menit per stasiun"""
    os.makedirs(RAW_DIR, exist_ok=True)
    cache_file = os.path.join(RAW_DIR, f"{station_id}.json")
    
    # Cek cache lokal jika sudah ada dan masih baru (< 3 jam)
    if not force and os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < 3 * 3600:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
                
    url = f"{API_BASE}/api/history?id={requests.utils.quote(station_id)}&range=7d"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data
    else:
        print(f"  [Warning] HTTP {resp.status_code} untuk stasiun {station_id}")
        return {}


def download_all_histories(force: bool = False) -> Dict[str, Any]:
    """Download data historis seluruh stasiun aktif"""
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Mengambil daftar stasiun aktif...")
    stations = get_active_stations()
    print(f"Ditemukan {len(stations)} stasiun aktif.")
    
    all_data = {}
    success_count = 0
    total_points = 0
    
    for idx, s in enumerate(stations, 1):
        stn_id = s["id"]
        stn_name = s.get("name", "Tanpa Nama")
        print(f"[{idx}/{len(stations)}] Mengunduh: {stn_name} ({stn_id})...", end=" ", flush=True)
        
        hist = download_station_history(stn_id, force=force)
        pts = hist.get("points", [])
        if pts:
            success_count += 1
            total_points += len(pts)
            all_data[stn_id] = {
                "metadata": {
                    "id": stn_id,
                    "name": stn_name,
                    "source": s.get("source"),
                    "type": s.get("type"),
                    "lat": s.get("lat"),
                    "lon": s.get("lon"),
                },
                "points": pts
            }
            print(f"✓ ({len(pts)} titik observasi)")
        else:
            print("✗ (0 titik / kosong)")
            
        # Jeda sejenak untuk rate limiting etis
        time.sleep(0.15)
        
    print(f"\n==========================================")
    print(f"Selesai! Berhasil mengunduh {success_count}/{len(stations)} stasiun.")
    print(f"Total observasi terkumpul: {total_points:,} titik data 15-menitan.")
    print(f"File tersimpan di: {RAW_DIR}/")
    print(f"==========================================")
    return all_data


if __name__ == "__main__":
    download_all_histories(force=False)
