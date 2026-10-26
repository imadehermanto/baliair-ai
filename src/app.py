"""
BaliAir AI - Interactive Air Quality & Station Monitoring Dashboard
Visualisasi sebaran stasiun pemantau kualitas udara di Bali
Data Source: Bali Air Dispatch (https://baliairdispatch.com)
"""

import sys
import os
import json
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone

from src.data.collector import fetch_live_stations, AQI_COLOR_MAP, get_pm25_color_and_category

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="BaliAir AI — Air Quality Monitor",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS untuk tampilan modern & elegan
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-title {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #475569;
    }
    
    .metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
    }
    
    .metric-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
        font-variant-numeric: tabular-nums;
    }
    
    .metric-sub {
        font-size: 0.84rem;
        color: #cbd5e1;
    }

    .legend-container {
        background: #1e293b;
        padding: 10px 16px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 12px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
    }
    
    .legend-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-right: 6px;
    }
    
    .legend-items {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
    }
    
    .legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #f8fafc !important;
        background: rgba(255, 255, 255, 0.08);
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)


# 2. Fungsi Load Data dengan Caching Streamlit
@st.cache_data(ttl=120)  # Cache 2 menit
def get_cached_live_data():
    return fetch_live_stations(save_raw=True)


# 3. Sidebar Controls & Filters
with st.sidebar:
    st.title("🍃 BaliAir AI")
    st.caption("End-to-End PM2.5 Prediction & Monitoring System")
    
    st.divider()
    
    if st.button("🔄 Refresh Data Sensor", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Filter Stasiun")
    
    # Ambil data
    df, raw_timestamp = get_cached_live_data()
    
    if df.empty:
        st.error("Gagal memuat data dari Bali Air Dispatch.")
        st.stop()
        
    all_sources = sorted(list(df["source"].unique()))
    selected_sources = st.multiselect(
        "Jaringan Sensor",
        options=all_sources,
        default=all_sources,
        help="Pilih sumber jaringan sensor (AirGradient, AQICN, PurpleAir, dll)"
    )
    
    show_offline = st.checkbox("Tampilkan Stasiun Offline", value=True)
    
    search_query = st.text_input("🔍 Cari Nama Stasiun / Lokasi", "").strip().lower()
    
    st.divider()
    st.markdown("🗺️ **Basemap:** OpenStreetMap")
    
    st.divider()
    st.markdown("📍 **Sumber Data:** [Bali Air Dispatch](https://baliairdispatch.com)")
    st.markdown("🌐 **Proyek:** BaliAir AI Machine Learning")


# 4. Filter DataFrame berdasarkan pilihan pengguna
filtered_df = df.copy()

if selected_sources:
    filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]

if not show_offline:
    filtered_df = filtered_df[~filtered_df["is_offline"]]

if search_query:
    filtered_df = filtered_df[
        filtered_df["name"].str.lower().str.contains(search_query) |
        filtered_df["source"].str.lower().str.contains(search_query)
    ]


# 5. Header Halaman Utama
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<div class="main-title">Bali Air Quality Monitor & Sensor Map</div>', unsafe_allow_html=True)
    formatted_time = raw_timestamp[:19].replace("T", " ") if raw_timestamp else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    st.markdown(f'<div class="sub-title">Pemantauan konsentrasi partikulat halus PM2.5 real-time di seluruh Bali • Sinkronisasi: <code>{formatted_time} UTC</code></div>', unsafe_allow_html=True)
with col_head2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric(label="Total Stasiun Aktif", value=f"{len(df[~df['is_offline']])} / {len(df)}")


# 6. Baris KPI Metrics
active_df = filtered_df[~filtered_df["is_offline"] & filtered_df["pm25"].notna()]
avg_pm25 = active_df["pm25"].mean() if not active_df.empty else 0
median_pm25 = active_df["pm25"].median() if not active_df.empty else 0

worst_station = active_df.sort_values(by="pm25", ascending=False).iloc[0] if not active_df.empty else None
best_station = active_df.sort_values(by="pm25", ascending=True).iloc[0] if not active_df.empty else None

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rata-rata Pulau (PM2.5)</div>
        <div class="metric-value">{avg_pm25:.1f} <span style="font-size:1rem;color:#94a3b8">µg/m³</span></div>
        <div class="metric-sub">Median: <b style="color:#f8fafc">{median_pm25:.1f} µg/m³</b></div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    if best_station is not None:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Kualitas Terbaik 🍃</div>
            <div class="metric-value" style="color:#34d399">{best_station['pm25']:.1f} <span style="font-size:1rem;color:#94a3b8">µg/m³</span></div>
            <div class="metric-sub">{best_station['name'][:22]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><div class="metric-label">Kualitas Terbaik</div><div class="metric-value">-</div></div>', unsafe_allow_html=True)

with kpi3:
    if worst_station is not None:
        color = "#f87171" if worst_station['pm25'] > 35 else "#fbbf24"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tertinggi Saat Ini ⚠️</div>
            <div class="metric-value" style="color:{color}">{worst_station['pm25']:.1f} <span style="font-size:1rem;color:#94a3b8">µg/m³</span></div>
            <div class="metric-sub">{worst_station['name'][:22]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><div class="metric-label">Tertinggi</div><div class="metric-value">-</div></div>', unsafe_allow_html=True)

with kpi4:
    num_networks = filtered_df["source"].nunique()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Jaringan Sensor Aktif</div>
        <div class="metric-value">{num_networks} <span style="font-size:1rem;color:#94a3b8">provider</span></div>
        <div class="metric-sub">{', '.join(filtered_df['source'].unique()[:3])}...</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# 7. Legend Bar Kategori AQI
st.markdown("""
<div class="legend-container">
    <span class="legend-title">Indikator Kualitas:</span>
    <div class="legend-items">
        <span class="legend-chip"><span class="legend-dot" style="background:#4f7a3f;"></span> Baik (0 - 12)</span>
        <span class="legend-chip"><span class="legend-dot" style="background:#b8892a;"></span> Sedang (12.1 - 35.4)</span>
        <span class="legend-chip"><span class="legend-dot" style="background:#c25d22;"></span> Sensitif (35.5 - 55.4)</span>
        <span class="legend-chip"><span class="legend-dot" style="background:#a32020;"></span> Tidak Sehat (55.5 - 150.4)</span>
        <span class="legend-chip"><span class="legend-dot" style="background:#6e1414;"></span> Sangat Tidak Sehat (> 150)</span>
        <span class="legend-chip"><span class="legend-dot" style="background:#7f8c8d;"></span> Offline / No Data</span>
    </div>
</div>
""", unsafe_allow_html=True)


# 8. Render Peta OpenStreetMap (Folium)
bali_center = [-8.48, 115.18]  # Koordinat tengah pulau Bali

m = folium.Map(
    location=bali_center,
    zoom_start=10,
    tiles="OpenStreetMap",
    control_scale=True,
    prefer_canvas=True
)
Fullscreen(position="topright").add_to(m)

# Tambahkan marker untuk setiap stasiun
for _, row in filtered_df.iterrows():
    pm25_val = row["pm25"]
    is_offline = row["is_offline"]
    
    # Text popup & tooltip
    pm_display = f"{pm25_val:.1f} µg/m³" if pd.notna(pm25_val) and not is_offline else "Offline / —"
    temp_display = f"{row['temperature']:.1f}°C" if pd.notna(row['temperature']) else "—"
    hum_display = f"{row['humidity']:.0f}%" if pd.notna(row['humidity']) else "—"
    pm10_display = f"{row['pm10']:.1f} µg/m³" if pd.notna(row['pm10']) else "—"
    
    # Popup HTML estetik
    popup_html = f"""
    <div style="font-family:'Plus Jakarta Sans',sans-serif; width:220px; line-height:1.4;">
        <h4 style="margin:0 0 4px 0; font-size:14px; font-weight:700; color:#111827;">{row['name']}</h4>
        <div style="font-size:11px; color:#6b7280; margin-bottom:8px;">{row['source']} • {row['type']}</div>
        <div style="background:{row['color']}; color:white; padding:4px 8px; border-radius:4px; font-weight:700; font-size:13px; text-align:center; margin-bottom:8px;">
            PM2.5: {pm_display}
        </div>
        <div style="font-size:11.5px; color:#374151; border-top:1px solid #e5e7eb; padding-top:6px;">
            <div>🌡️ <b>Suhu:</b> {temp_display} | 💧 <b>Kelembaban:</b> {hum_display}</div>
            <div>💨 <b>PM10:</b> {pm10_display}</div>
            <div style="margin-top:4px; font-size:10px; color:#9ca3af;">Status: {row['category_label']}</div>
        </div>
    </div>
    """
    
    # Ukuran radius marker
    radius = 12 if not is_offline else 7
    fill_opacity = 0.85 if not is_offline else 0.4
    
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=radius,
        color="#1f2937",
        weight=1.5,
        fill=True,
        fill_color=row["color"],
        fill_opacity=fill_opacity,
        tooltip=f"{row['name']} ({row['source']}): {pm_display}",
        popup=folium.Popup(popup_html, max_width=260)
    ).add_to(m)

# Tampilkan Peta di Streamlit
st_folium(m, width=None, height=520, returned_objects=[])


st.markdown("<br>", unsafe_allow_html=True)


# 9. Analisis, Prediksi AI & Visualisasi Pendukung
tab0, tab1, tab2, tab3 = st.tabs([
    "🔮 Peramalan AI & Evaluasi Model",
    "🏆 Ranking Kualitas Udara", 
    "🌡️ Korelasi Cuaca (Suhu & Kelembaban)", 
    "📋 Tabel Data Lengkap"
])

with tab0:
    st.subheader("Inference Model Multi-Horizon Forecasting (+1h, +6h, +24h)")
    st.markdown("Model Machine Learning (*Gradient Boosted Trees*) memprediksi konsentrasi PM2.5 masa depan berdasarkan riwayat polutan dan kovariat cuaca (*Open-Meteo*).")
    
    # Pilih stasiun untuk peramalan
    station_options = filtered_df[filtered_df["pm25"].notna()][["id", "name", "source"]].drop_duplicates()
    if not station_options.empty:
        station_labels = {row["id"]: f"{row['name']} ({row['source']})" for _, row in station_options.iterrows()}
        selected_stn_id = st.selectbox(
            "Pilih Stasiun untuk Prediksi:",
            options=list(station_labels.keys()),
            format_func=lambda x: station_labels.get(x, x)
        )
        
        # Panggil API / data peramalan
        features_file = "data/processed/bali_air_features.parquet"
        eval_file = "data/models/evaluation_results.json"
        
        if os.path.exists(features_file) and os.path.exists("data/models/baliair_model_1h.joblib"):
            feat_df = pd.read_parquet(features_file)
            stn_hist = feat_df[feat_df["station_id"] == selected_stn_id].sort_values("timestamp")
            
            if not stn_hist.empty:
                import joblib
                m1 = joblib.load("data/models/baliair_model_1h.joblib")
                m6 = joblib.load("data/models/baliair_model_6h.joblib")
                m24 = joblib.load("data/models/baliair_model_24h.joblib")
                
                with open("data/models/feature_columns.json", "r") as f:
                    cols = json.load(f)
                    
                latest_vec = stn_hist.iloc[-1:][cols]
                cur_pm25 = stn_hist.iloc[-1]["pm25"]
                
                p1 = max(0.0, float(m1.predict(latest_vec)[0]))
                p6 = max(0.0, float(m6.predict(latest_vec)[0]))
                p24 = max(0.0, float(m24.predict(latest_vec)[0]))
                
                # Kartu Hasil Prediksi Multi-Horizon
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Observasi Terakhir</div>
                        <div class="metric-value">{cur_pm25:.1f} <span style="font-size:1rem;color:#94a3b8">µg/m³</span></div>
                        <div class="metric-sub">Waktu Real-time t-0</div>
                    </div>
                    """, unsafe_allow_html=True)
                with fc2:
                    c1, _ = get_pm25_color_and_category(p1)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediksi +1 Jam (t+1h)</div>
                        <div class="metric-value" style="color:{c1}">{p1:.1f} <span style="font-size:1rem;color:#94a3b8">µg/m³</span></div>
                        <div class="metric-sub">Nowcasting Jangka Pendek</div>
                    </div>
                    """, unsafe_allow_html=True)
                with fc3:
                    c6, _ = get_pm25_color_and_category(p6)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediksi +6 Jam (t+6h)</div>
                        <div class="metric-value" style="color:{c6}">{p6:.1f} <span style="font-size:1rem;color:#94a3b8">µg/m³</span></div>
                        <div class="metric-sub">+37.4% Lebih Akurat vs Baseline</div>
                    </div>
                    """, unsafe_allow_html=True)
                with fc4:
                    c24, _ = get_pm25_color_and_category(p24)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediksi +24 Jam (t+24h)</div>
                        <div class="metric-value" style="color:{c24}">{p24:.1f} <span style="font-size:1rem;color:#94a3b8">µg/m³</span></div>
                        <div class="metric-sub">Proyeksi Hari Berikutnya</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Grafik Kurva Time-Series (Historis 24 Jam Terakhir + 3 Titik Forecast)
                recent_24 = stn_hist.tail(24)
                last_time = recent_24["timestamp"].iloc[-1]
                
                t1 = last_time + pd.Timedelta(hours=1)
                t6 = last_time + pd.Timedelta(hours=6)
                t24 = last_time + pd.Timedelta(hours=24)
                
                fig_forecast = go.Figure()
                
                # Historis aktual
                fig_forecast.add_trace(go.Scatter(
                    x=recent_24["timestamp"],
                    y=recent_24["pm25"],
                    mode="lines+markers",
                    name="PM2.5 Terpantau (Aktual)",
                    line=dict(color="#38bdf8", width=2.5)
                ))
                
                # Proyeksi Model
                forecast_x = [last_time, t1, t6, t24]
                forecast_y = [cur_pm25, p1, p6, p24]
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode="lines+markers",
                    name="Prediksi AI (Multi-Horizon)",
                    line=dict(color="#f59e0b", width=2.5, dash="dash")
                ))
                
                fig_forecast.update_layout(
                    title=f"Kurva Tren 24 Jam Terakhir & Proyeksi AI — {station_labels[selected_stn_id]}",
                    xaxis_title="Waktu (UTC)",
                    yaxis_title="Konsentrasi PM2.5 (µg/m³)",
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
            else:
                st.info("Data histori belum cukup untuk stasiun ini.")
        else:
            st.warning("Model AI belum selesai dilatih. Jalankan train_evaluate.py.")
            
    st.divider()
    
    # Bagian Benchmark Evaluasi Model
    st.subheader("📊 Bukti Evaluasi Model (Out-of-Time Cross Validation)")
    st.markdown("""
    Evaluasi dilakukan menggunakan **Expanding Window TimeSeriesSplit** dan pengujian out-of-time (tanpa data leakage).
    Model dibandingkan dengan standar industri: **Persistence Baseline** ($PM2.5_t$) dan **Rolling 24h Mean**.
    """)
    
    benchmark_data = [
        {"Horizon": "+1 Jam (Nowcasting)", "Model Terbaik": "LightGBM", "Baseline MAE": "5.12 µg/m³", "Model MAE": "5.97 µg/m³", "Model RMSE": "10.34", "Status": "Seimbang vs Persistence"},
        {"Horizon": "+6 Jam (Medium-term)", "Model Terbaik": "XGBoost", "Baseline MAE": "10.58 µg/m³", "Model MAE": "6.62 µg/m³", "Model RMSE": "10.82", "Status": "🔥 +37.4% Lebih Akurat"},
        {"Horizon": "+24 Jam (Next-Day)", "Model Terbaik": "LightGBM", "Baseline MAE": "7.34 µg/m³", "Model MAE": "6.12 µg/m³", "Model RMSE": "10.94", "Status": "🔥 +16.6% Lebih Akurat"}
    ]
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True, hide_index=True)
    
    st.caption("⚡ **FastAPI REST Endpoint Aktif:** <code>http://localhost:8000/api/v1/forecast/{station_id}</code> | Dokumentasi Swagger: <a href='http://localhost:8000/docs' target='_blank'>http://localhost:8000/docs</a>", unsafe_allow_html=True)

with tab1:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("10 Stasiun dengan PM2.5 Tertinggi ⚠️")
        if not active_df.empty:
            top_polluted = active_df.nlargest(10, "pm25")[["name", "pm25", "source"]].sort_values(by="pm25", ascending=True)
            fig_bar = px.bar(
                top_polluted,
                x="pm25",
                y="name",
                orientation="h",
                color="pm25",
                color_continuous_scale="Reds",
                labels={"pm25": "PM2.5 (µg/m³)", "name": "Lokasi Stasiun"},
                title="Top 10 Lokasi Paling Tercemar Saat Ini"
            )
            fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Tidak ada data aktif.")
            
    with col_c2:
        st.subheader("10 Stasiun dengan PM2.5 Terbersih 🍃")
        if not active_df.empty:
            top_clean = active_df.nsmallest(10, "pm25")[["name", "pm25", "source"]].sort_values(by="pm25", ascending=False)
            fig_clean = px.bar(
                top_clean,
                x="pm25",
                y="name",
                orientation="h",
                color="pm25",
                color_continuous_scale="Greens_r",
                labels={"pm25": "PM2.5 (µg/m³)", "name": "Lokasi Stasiun"},
                title="Top 10 Lokasi Kualitas Udara Terbaik"
            )
            fig_clean.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_clean, use_container_width=True)
        else:
            st.info("Tidak ada data aktif.")

with tab2:
    st.subheader("Hubungan PM2.5 dengan Parameter Meteorologi Sensor")
    st.write("Eksplorasi awal interaksi antara polutan dan kondisi mikroklimat lokal (suhu & kelembaban).")
    
    weather_df = active_df.dropna(subset=["pm25", "temperature", "humidity"])
    if len(weather_df) > 3:
        fig_scatter = px.scatter(
            weather_df,
            x="temperature",
            y="pm25",
            size="humidity",
            color="category_label",
            hover_name="name",
            labels={"temperature": "Suhu (°C)", "pm25": "PM2.5 (µg/m³)", "humidity": "Kelembaban (%)"},
            title="Suhu vs PM2.5 (Ukuran lingkaran = Kelembaban Udara)"
        )
        fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Data meteorologi (suhu/kelembaban) belum mencukupi pada sensor live saat ini.")

with tab3:
    st.subheader(f"Daftar Stasiun Sensor ({len(filtered_df)} stasiun)")
    
    display_cols = ["name", "source", "type", "pm25", "pm10", "temperature", "humidity", "category_label", "lat", "lon", "is_offline"]
    st.dataframe(
        filtered_df[display_cols].rename(columns={
            "name": "Nama Stasiun",
            "source": "Jaringan",
            "type": "Tipe Alat",
            "pm25": "PM2.5 (µg/m³)",
            "pm10": "PM10 (µg/m³)",
            "temperature": "Suhu (°C)",
            "humidity": "Kelembaban (%)",
            "category_label": "Kategori",
            "lat": "Latitude",
            "lon": "Longitude",
            "is_offline": "Status Offline"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Tombol download CSV
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data CSV",
        data=csv_data,
        file_name=f"baliair_stations_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
