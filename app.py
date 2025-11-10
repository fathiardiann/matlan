# =============================================
# 🏠 Prediksi Harga Rumah California — Versi Simplified (1 Tab)
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------
# Konfigurasi Halaman
# ---------------------------
st.set_page_config(
    page_title="Prediksi Harga Rumah California",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Styling CSS
# ---------------------------
st.markdown("""
<style>
    body {background-color: #0E1117; color: #E6E6E6;}
    .card {background-color: #161B22; border-radius: 12px; padding: 16px; margin-bottom:10px;}
    .small-muted {color: #9aa3b2; font-size:12px;}
    .stButton>button {
        background: linear-gradient(90deg,#0078FF,#0057CC);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {background: linear-gradient(90deg,#3399FF,#0078FF); transform: scale(1.03);}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load data & model
# ---------------------------
@st.cache_data
def load_model():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df[data.feature_names]
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, df

model, df = load_model()

# ---------------------------
# Header
# ---------------------------
st.markdown("<h1 style='text-align:center'>🏠 Prediksi Harga Rumah California</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: #9aa3b2'>Model regresi linier sederhana dengan tampilan modern.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Sidebar Input
# ---------------------------
st.sidebar.header("Masukkan Data Rumah")

MedInc = st.sidebar.slider("Pendapatan Median (x10.000 USD)", 0.0, 20.0, 8.3, step=0.1)
HouseAge = st.sidebar.slider("Usia Rumah (tahun)", 1, 50, 20)
AveRooms = st.sidebar.slider("Rata-rata Jumlah Ruangan", 1.0, 10.0, 6.5, step=0.1)
AveBedrms = st.sidebar.slider("Rata-rata Jumlah Kamar Tidur", 0.5, 5.0, 1.0, step=0.1)
Population = st.sidebar.number_input("Populasi", min_value=100, max_value=100000, value=1200, step=100)
AveOccup = st.sidebar.slider("Rata-rata Penghuni per Rumah", 1.0, 15.0, 2.8, step=0.1)
Latitude = st.sidebar.slider("Latitude", 30.0, 42.0, 37.88, step=0.01)
Longitude = st.sidebar.slider("Longitude", -128.0, -112.0, -122.25, step=0.01)

# ---------------------------
# Fungsi Prediksi
# ---------------------------
def predict_price():
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
    pred = model.predict(sample)[0] * 100000
    lower, upper = pred * 0.95, pred * 1.05
    return pred, lower, upper

# ---------------------------
# Halaman Prediksi
# ---------------------------
col_left, col_right = st.columns([2,1])

with col_left:
    st.markdown("### 🔮 Estimasi Harga Rumah")
    if st.button("Prediksi Sekarang"):
        with st.spinner("🔍 Menghitung estimasi..."):
            time.sleep(0.8)
            pred, lower, upper = predict_price()
            st.session_state["pred"] = pred

        st.success("✅ Prediksi selesai!")
        st.markdown(f"""
        <div class="card">
            <h3>🏡 Estimasi Harga Rumah</h3>
            <p style='font-size:28px; color:#00FFAA; margin:6px 0'><b>${pred:,.0f}</b></p>
            <p class='small-muted'>Rentang estimasi: ${lower:,.0f} – ${upper:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Insight area lokal
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
    # Tentukan gambar berdasarkan prediksi terakhir
    if "pred" in st.session_state:
        p = st.session_state["pred"]
    else:
        p = None

    # Gambar default (belum prediksi)
    question_img_top = "https://raw.githubusercontent.com/fathiardiann/matlan/refs/heads/main/retro-block-exclamation-31200.jpg"
    question_img_bottom = "https://raw.githubusercontent.com/fathiardiann/matlan/refs/heads/main/retro-block-exclamation-31200.jpg"

    # Ganti URL di bawah dengan gambar kamu sendiri dari GitHub nanti
    low_img_top = "https://photos.zillowstatic.com/fp/0731b4b39b36cfc846d8e0fc3bfebb9f-cc_ft_768.webp"
    low_img_bottom = "https://photos.zillowstatic.com/fp/d16516c438eccf3e711a54b40fccb59e-uncropped_scaled_within_1536_1152.webp"
    mid30_top = "https://photos.zillowstatic.com/fp/cd54bbe7b3622f9ce004ab68e8fb2daa-cc_ft_768.webp"
    mid30_bottom = "https://photos.zillowstatic.com/fp/aec6d2aea8e0a8595706875ebda0e314-uncropped_scaled_within_1536_1152.webp"
    mid_img_top = "https://photos.zillowstatic.com/fp/0fa4dfae01f5cf06697212fcce9628e6-cc_ft_768.webp"
    mid_img_bottom = "https://photos.zillowstatic.com/fp/8ff43a4c45c28b2a2f350cefa91bb669-cc_ft_384.webp"
    upper_img_top = "https://photos.zillowstatic.com/fp/a2081284f721136cdc3469fe474f369c-sc_1536_1024.webp"
    upper_img_bottom = "https://photos.zillowstatic.com/fp/81aaf55831b4eba303f8ccfd84fa64fc-sc_1536_1024.webp"
    luxury_img_top = "https://photos.zillowstatic.com/fp/9fc42898768a6ba3e122e74b14b45abf-sc_1536_1024.webp"
    luxury_img_bottom = "https://photos.zillowstatic.com/fp/c79963958a7a845fc082b727f582e533-sc_1536_1024.webp"
    under80k1 = "https://photos.zillowstatic.com/fp/f5a57a7af4b0b60d5629cad3fca8d580-cc_ft_768.webp"
    under80k2 = "https://photos.zillowstatic.com/fp/85de92383c5bd0a8ad28c243481cf9c5-cc_ft_768.webp"
    # Pilih gambar berdasarkan status
    if p is None:
        img3, img4 = question_img_top, question_img_bottom
    elif p < 200000:
        img1, img2 = low_img_top, low_img_bottom
    elif p < 350000:
        img1, img2 = mid30_top, mid30_bottom
    elif p < 500000:
        img1, img2 = mid_img_top, mid_img_bottom
    elif p < 800000:
        img1, img2 = under80k1, under80k2
    elif p < 1000000:
        img1, img2 = upper_img_top, upper_img_bottom
    else:
        img1, img2 = luxury_img_top, luxury_img_bottom
    # Tampilkan dua gambar
    st.image(img1, width=420)
    st.image(img2, width=420)
    st.image(img3, width=220)
    st.image(img4, width=220)

    # Keterangan
    if p is None:
        st.markdown("<div class='small-muted' style='margin-top:6px'>Tekan tombol <b>Prediksi Sekarang</b> untuk melihat hasil dan gambar rumah.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-muted' style='margin-top:6px'>Gambar rumah di atas sesuai kategori harga.</div>", unsafe_allow_html=True)

 

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:#9aa3b2; font-size:12px'>© 2025 California Housing Predictor</div>", unsafe_allow_html=True)









