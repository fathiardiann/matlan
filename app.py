# =============================================
# 🏠 Prediksi Harga Rumah California — Versi Final Polished
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
    h1, h2, h3 {font-family: 'Poppins', sans-serif;}
    body, p, div {font-family: 'Inter', sans-serif;}
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
# Load data & model (cached)
# ---------------------------
@st.cache_resource
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
st.markdown("<p style='text-align:center; color: #9aa3b2'>Aplikasi prediksi harga rumah berbasis data California Housing (Scikit-learn).</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Sidebar Input
# ---------------------------
st.sidebar.header("🧩 Masukkan Data Rumah")

st.sidebar.markdown("**💰 Pendapatan & Kondisi Rumah**")
MedInc = st.sidebar.slider("Pendapatan Median (x10.000 USD)", 0.0, 25.0, 8.3, step=0.1)
HouseAge = st.sidebar.slider("Usia Rumah (tahun)", 1, 80, 20)
AveRooms = st.sidebar.slider("Rata-rata Jumlah Ruangan", 1.0, 15.0, 6.5, step=0.1)
AveBedrms = st.sidebar.slider("Rata-rata Jumlah Kamar Tidur", 0.5, 8.0, 1.0, step=0.1)
Population = st.sidebar.number_input("Populasi Area", min_value=50, max_value=200000, value=1200, step=100)
AveOccup = st.sidebar.slider("Rata-rata Penghuni per Rumah", 0.5, 20.0, 2.8, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**📍 Lokasi (Koordinat)**")
Latitude = st.sidebar.slider("Latitude", 30.0, 45.0, 37.88, step=0.01)
Longitude = st.sidebar.slider("Longitude", -128.0, -112.0, -122.25, step=0.01)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Input"):
    st.experimental_rerun()

st.sidebar.info("""
**Tentang Aplikasi**
- Dataset: *California Housing* (Scikit-learn)
- Model: Linear Regression
- Developer: [Nama Kamu]
""")

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
col_left, col_right = st.columns([1.8, 1.2])

with col_left:
    st.markdown("### 🔮 Estimasi Harga Rumah")
    st.markdown("""
    <small style='color:#9aa3b2'>
    Masukkan parameter rumah di sidebar lalu tekan tombol <b>Prediksi Sekarang</b> untuk melihat estimasi harga.
    </small>
    """, unsafe_allow_html=True)

    if st.button("Prediksi Sekarang"):
        with st.spinner("🔍 Menghitung estimasi..."):
            time.sleep(0.8)
            pred, lower, upper = predict_price()
            st.session_state["pred"] = pred

        st.success("✅ Prediksi selesai!")
        st.balloons()

        st.markdown(f"""
        <div class="card">
            <h3>🏡 Estimasi Harga Rumah</h3>
            <p style='font-size:32px; color:#00FFAA; margin:6px 0'><b>${pred:,.0f}</b></p>
            <p class='small-muted'>≈ {pred/1000:,.1f} ribu USD</p>
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

        # Narasi tambahan
        if pred < 200000:
            comment = "🏘️ Termasuk kategori rumah sederhana, biasanya di area suburban atau rural."
        elif pred < 500000:
            comment = "🏠 Rumah keluarga menengah di area perkotaan kecil."
        elif pred < 1000000:
            comment = "🏡 Properti bernilai tinggi di area metropolitan populer."
        else:
            comment = "💎 Rumah mewah! Biasanya dekat pantai atau pusat kota besar."
        st.info(comment)

with col_right:
    if "pred" in st.session_state:
        p = st.session_state["pred"]
    else:
        p = None

    # Gambar default (belum prediksi)
    question_img_top = "https://upload.wikimedia.org/wikipedia/commons/9/99/Question_mark_black.png"
    question_img_bottom = "https://upload.wikimedia.org/wikipedia/commons/9/99/Question_mark_black.png"

    # Ganti ini nanti dengan URL GitHub kamu
    low_img_top = "https://placehold.co/420x250/333/FFF?text=Low+House+1"
    low_img_bottom = "https://placehold.co/420x250/333/FFF?text=Low+House+2"
    mid_img_top = "https://placehold.co/420x250/555/FFF?text=Mid+House+1"
    mid_img_bottom = "https://placehold.co/420x250/555/FFF?text=Mid+House+2"
    upper_img_top = "https://placehold.co/420x250/777/FFF?text=Upper+House+1"
    upper_img_bottom = "https://placehold.co/420x250/777/FFF?text=Upper+House+2"
    luxury_img_top = "https://placehold.co/420x250/999/FFF?text=Luxury+House+1"
    luxury_img_bottom = "https://placehold.co/420x250/999/FFF?text=Luxury+House+2"

    # Pilih gambar berdasarkan status
   if p is None:
        img1, img2 = question_img_top, question_img_bottom
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

    st.image(img1, width=420)
    st.image(img2, width=420)

    if p is None:
        st.markdown("<div class='small-muted' style='margin-top:6px'>Tekan tombol <b>Prediksi Sekarang</b> untuk melihat hasil dan gambar rumah.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-muted' style='margin-top:6px'>Gambar rumah di atas sesuai kategori harga.</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:#9aa3b2; font-size:12px'>© 2025 California Housing Predictor | Dibuat dengan ❤️ menggunakan Streamlit</div>", unsafe_allow_html=True)
