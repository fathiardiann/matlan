# =============================================
# 🏠 Prediksi Harga Rumah California — Responsive + Map
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Prediksi Harga Rumah California",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Styling CSS (ringan)
# ---------------------------
st.markdown("""
<style>
    body {background-color: #0E1117; color: #E6E6E6;}
    .card {background-color: #161B22; border-radius: 12px; padding: 16px;}
    .predict-btn .stButton>button {background: linear-gradient(90deg,#0078FF,#0057CC); border-radius:8px;}
    .small-muted {color: #9aa3b2; font-size:12px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load data & model (cached)
# ---------------------------
@st.cache_data
def load_data_and_model():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df[data.feature_names]
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return df, data.feature_names, model, X_train, X_test, y_test, y_pred, mse, r2

df, feature_names, model, X_train, X_test, y_test, y_pred, mse, r2 = load_data_and_model()

# ---------------------------
# Header
# ---------------------------
st.markdown("<h1 style='text-align:center'>🏠 Prediksi Harga Rumah California</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: #9aa3b2'>Model regresi linier demo — dengan visual responsif dan peta interaktif.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Tabs
# ---------------------------
tab_pred, tab_eval, tab_vis = st.tabs(["🔮 Prediksi", "📊 Evaluasi Model", "📈 Visualisasi & Map"])

# ---------------------------
# Sidebar Inputs (shared)
# ---------------------------
with st.sidebar:
    st.header("Masukkan Data Rumah")
    MedInc = st.slider("Pendapatan Median (x10.000 USD)", 0.0, 15.0, 8.3, step=0.1)
    HouseAge = st.slider("Usia Rumah (tahun)", 1, 50, 20)
    AveRooms = st.slider("Rata-rata Jumlah Ruangan", 1.0, 10.0, 6.5, step=0.1)
    AveBedrms = st.slider("Rata-rata Jumlah Kamar Tidur", 0.5, 5.0, 1.0, step=0.1)
    Population = st.number_input("Populasi", min_value=100, max_value=50000, value=1200, step=50)
    AveOccup = st.slider("Rata-rata Penghuni per Rumah", 1.0, 10.0, 2.8, step=0.1)
    Latitude = st.slider("Latitude", 32.0, 42.0, 37.88, step=0.01)
    Longitude = st.slider("Longitude", -125.0, -114.0, -122.25, step=0.01)

# Helper: build sample and predict
def make_prediction():
    sample = pd.DataFrame([{
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }])
    pred = model.predict(sample)[0] * 100000  # convert to USD
    # simple CI (approx) — adjust per model choice
    lower, upper = pred * 0.95, pred * 1.05
    return float(pred), float(lower), float(upper), sample

# ---------------------------
# TAB: Prediksi
# ---------------------------
with tab_pred:
    st.markdown("### 🔮 Estimasi Harga Rumah")
    col_left, col_right = st.columns([2,1])

    with col_left:
        st.markdown("Sesuaikan parameter di sidebar lalu tekan tombol prediksi.")
        predict_btn = st.button("Prediksi Sekarang", key="predict_now")

        if predict_btn:
            with st.spinner("🔍 Menghitung estimasi..."):
                time.sleep(0.8)  # memberi feel responsif
                pred, lower, upper, sample = make_prediction()
                # simpan hasil ke session_state agar bisa dipakai di tab lain
                st.session_state['last_pred'] = pred
                st.session_state['last_lower'] = lower
                st.session_state['last_upper'] = upper
                st.session_state['last_lat'] = Latitude
                st.session_state['last_lon'] = Longitude
            st.success("✅ Prediksi selesai!")

            # Hasil kartu besar
            st.markdown(f"""
            <div class="card">
                <h3>🏡 Estimasi Harga Rumah</h3>
                <p style='font-size:28px; color:#00FFAA; margin:6px 0'><b>${pred:,.0f}</b></p>
                <p class='small-muted'>Rentang estimasi: ${lower:,.0f} – ${upper:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

            # Insight kontekstual (rata-rata area)
            # kumpulkan rata-rata di area kecil ±0.5 deg
            area_mask = (
                (df["Latitude"].between(Latitude - 0.5, Latitude + 0.5)) &
                (df["Longitude"].between(Longitude - 0.5, Longitude + 0.5))
            )
            if area_mask.any():
                area_avg = df.loc[area_mask, "MedHouseVal"].mean() * 100000
                diff_pct = (pred - area_avg) / area_avg * 100
                if diff_pct >= 0:
                    insight = f"🏗️ Harga ini {diff_pct:.1f}% lebih tinggi dari rata-rata area sekitar (${area_avg:,.0f})."
                else:
                    insight = f"📉 Harga ini {abs(diff_pct):.1f}% lebih rendah dari rata-rata area sekitar (${area_avg:,.0f})."
            else:
                insight = "ℹ️ Data area sekitarmu terbatas — gunakan lokasi lain untuk perbandingan lokal."

            st.markdown(f"<div class='card'><p>{insight}</p></div>", unsafe_allow_html=True)

    with col_right:
        # Tampilkan gambar rumah sesuai kategori harga (ukuran tetap agar tidak membesar)
        default_img = "https://i.imgur.com/8Q1Xb6o.png"  # fallback
        if 'last_pred' in st.session_state:
            last_pred_for_img = st.session_state['last_pred']
        else:
            # use currently computed (if button just pressed)
            last_pred_for_img = None

        # Determine image based on last_pred if available; else use sample heuristic
        img_url = default_img
        if 'last_pred' in st.session_state:
            p = st.session_state['last_pred']
            if p < 200000:
                img_url = "https://i.imgur.com/jFZsokO.png"   # simple house
            elif p < 500000:
                img_url = "https://i.imgur.com/8A3lJkz.png"   # modern suburban
            else:
                img_url = "https://i.imgur.com/T6A2b5V.png"   # luxury
        else:
            img_url = default_img

        # image fixed width (tidak ikut membesar)
        st.image(img_url, caption="Contoh rumah serupa", width=320)

        st.markdown("<div class='small-muted' style='margin-top:6px'>Gambar rumah di atas dipilih berdasarkan nilai estimasi.</div>", unsafe_allow_html=True)

# ---------------------------
# TAB: Evaluasi Model
# ---------------------------
with tab_eval:
    st.markdown("### ⚙️ Evaluasi Model")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Mean Squared Error", f"{mse:.4f}")
        st.metric("R² Score", f"{r2:.4f}")
    with c2:
        coef_df = pd.DataFrame({"Feature": df[feature_names].columns, "Koefisien": model.coef_.round(4)})
        st.dataframe(coef_df.sort_values(by="Koefisien", ascending=False), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Interpretasi singkat")
    st.markdown("- Koefisien positif: atribut menaikkan prediksi harga.")
    st.markdown("- Koefisien negatif: atribut menurunkan prediksi harga.")
    st.markdown("- Perhatikan keterbatasan: model linear sederhana, bisa ditingkatkan dengan tree-based models & SHAP untuk interpretasi.")

# ---------------------------
# TAB: Visualisasi & Map (responsif)
# ---------------------------
with tab_vis:
    st.markdown("### 📈 Visualisasi Data (responsif) & Peta Interaktif")
    left_col, right_col = st.columns([2,1])

    # Left: Charts (menggunakan use_container_width agar responsif)
    with left_col:
        st.markdown("**Hubungan Pendapatan vs Harga Rumah**")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.scatterplot(x=df["MedInc"], y=df["MedHouseVal"], alpha=0.4, color="#58A6FF", ax=ax)
        ax.set_xlabel("Pendapatan Median (x10.000 USD)")
        ax.set_ylabel("Harga Rumah Median (x100.000 USD)")
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Prediksi vs Aktual (Test set)**")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="#00FFAA", ax=ax2)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        ax2.set_xlabel("Harga Aktual (x100.000 USD)")
        ax2.set_ylabel("Harga Prediksi")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2, use_container_width=True)

    # Right: Map & last prediction summary
    with right_col:
        st.markdown("**Lokasi & Ringkasan Prediksi Terakhir**")
        if 'last_pred' in st.session_state:
            lp = st.session_state['last_pred']
            lat = st.session_state['last_lat']
            lon = st.session_state['last_lon']
            st.markdown(f"<div class='card'><b>Terakhir:</b><br>${lp:,.0f}<br><span class='small-muted'>{lat:.3f}, {lon:.3f}</span></div>", unsafe_allow_html=True)

            # prepare map data with color bucket
            if lp < 200000:
                color = [50, 200, 100]   # hijau
            elif lp < 500000:
                color = [255, 200, 50]   # kuning
            else:
                color = [255, 80, 90]    # merah

            map_data = pd.DataFrame({
                "lat": [lat],
                "lon": [lon],
                "price": [lp],
                "color": [color]
            })

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_data,
                get_position='[lon, lat]',
                get_fill_color='color',
                get_radius=4000,
                pickable=True,
                opacity=0.9
            )

            view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=8)
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"html": "<b>Estimated:</b> ${price}", "style": {"color": "white"}})
            st.pydeck_chart(deck, use_container_width=True, height=320)
        else:
            st.markdown("<div class='card'><i>Belum ada prediksi. Tekan 'Prediksi Sekarang' di tab Prediksi.</i></div>", unsafe_allow_html=True)
            # show general map center California
            view_state = pdk.ViewState(latitude=36.5, longitude=-119.5, zoom=5)
            deck = pdk.Deck(initial_view_state=view_state, map_style='dark')
            st.pydeck_chart(deck, use_container_width=True, height=320)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color: #9aa3b2; font-size:12px'>© 2025 California Housing Predictor — polished with Streamlit</div>", unsafe_allow_html=True)
