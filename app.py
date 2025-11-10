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
    .card {
        background-color: #161B22;
        border-radius: 12px;
        padding: 16px;
        margin-bottom:10px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 16px rgba(0, 200, 255, 0.15);
    }
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
    .stButton>button:hover {
        background: linear-gradient(90deg,#3399FF,#0078FF);
        transform: scale(1.03);
    }
    /* Geser gambar tanda seru supaya sejajar dengan gambar rumah */
    img[src="https://raw.githubusercontent.com/fathiardiann/matlan/refs/heads/main/retro-block-exclamation-31200.jpg"] {
        display: block;
        margin-left: auto;
        margin-right: 0;
        transform: translateX(60px); /* geser kanan, atur sesuai selera */
    }

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
st.markdown("""
<h1 style='text-align:center; line-height:1.2;'>
🏠 Prediksi Harga Rumah California<br>
<span>   Menggunakan Linear Regression</span>
</h1>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color: #9aa3b2'>Model regresi linier untuk memprediksi harga rumah berdasarkan faktor ekonomi dan geografis.</p>", unsafe_allow_html=True)

st.markdown("""
<style>
@font-face {
    font-family: 'JJK';
    src: url('https://raw.githubusercontent.com/fathiardiann/matlan/refs/heads/main/fonts/jjk.ttf') format('truetype');
}
.kelompok5 {
    text-align: center;
    font-family: 'JJK', sans-serif;
    font-size: 30px;
    color: #9aa3b2;
    margin-top: -10px;
    letter-spacing: 1px;
    
}
</style>
""", unsafe_allow_html=True)
st.markdown("<div class='kelompok5'>Kelompok 5</div>", unsafe_allow_html=True)


st.markdown("---")

# ---------------------------
# Sidebar Input
# ---------------------------
st.sidebar.header("Masukkan Data Rumah")

MedInc = st.sidebar.slider("Pendapatan Median (x10.000 USD)", 0.0, 20.0, 8.3, step=0.1)
HouseAge = st.sidebar.slider("Usia Rumah (tahun)", 1, 50, 20)
AveRooms = st.sidebar.slider("Rata-rata Jumlah Ruangan", 1.0, 30.0, 6.5, step=0.1)
AveBedrms = st.sidebar.slider("Rata-rata Jumlah Kamar Tidur", 0.5, 20.0, 1.0, step=0.1)
Population = st.sidebar.number_input("Populasi", min_value=100, max_value=100000, value=1200, step=100)
AveOccup = st.sidebar.slider("Rata-rata Penghuni per Rumah", 1.0, 15.0, 2.8, step=0.1)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.88, step=0.01)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -122.25, step=0.01)

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
    st.markdown("### 🏡 Estimasi Harga Rumah")

    # Tombol prediksi — simpan juga lower & upper ke session state agar tersedia di area lain
    if st.button("Prediksi Sekarang"):
        with st.spinner("🔍 Menghitung estimasi..."):
            time.sleep(0.8)
            pred, lower, upper = predict_price()
            st.session_state["pred"] = pred
            st.session_state["lower"] = lower
            st.session_state["upper"] = upper

        st.success("✅ Prediksi selesai!")

    # Ambil prediksi dari session_state (bisa None jika belum ada)
    p = st.session_state.get("pred", None)
    lower = st.session_state.get("lower", None)
    upper = st.session_state.get("upper", None)

    # Tampilkan kartu hasil hanya jika sudah ada prediksi
    if p is not None:
        USD_TO_IDR = 16000  # bisa ubah kurs sesuai kebutuhan
        rupiah = p * USD_TO_IDR

        st.markdown(f"""
        <div class="card">
            <h3> 💰 Estimasi Harga Rumah</h3>
            <p style='font-size:28px; color:#00FFAA; margin:6px 0'><b>${p:,.0f}</b></p>
            <p style='font-size:20px; color:#FFD700; margin:-5px 0 8px 0'><b>≈ Rp{rupiah:,.0f}</b></p>
            <p class='small-muted'>Rentang estimasi: ${lower:,.0f} – ${upper:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)


        # ================================
        # Deskripsi kategori harga
        # ================================
        if p < 200000:
            desc = (
                "Harga rumah tergolong <b>rendah</b>, kemungkinan berada di area pedesaan "
                "atau pinggiran kota dengan akses terbatas ke pusat ekonomi utama."
            )
        elif p < 350000:
            desc = (
                "Rumah ini berada di kategori <b>menengah ke bawah</b>, umumnya di wilayah suburban "
                "dengan fasilitas standar dan lingkungan yang tenang."
            )
        elif p < 500000:
            desc = (
                "Termasuk kategori <b>menengah</b> — lokasi kemungkinan di area berkembang "
                "dengan akses cukup baik ke fasilitas umum dan transportasi."
            )
        elif p < 800000:
            desc = (
                "Harga rumah ini <b>di atas rata-rata</b>, biasanya berlokasi di kawasan perkotaan "
                "dengan lingkungan yang strategis dan permintaan tinggi."
            )
        elif p < 1000000:
            desc = (
                "Termasuk kategori <b>premium</b> — rumah di kawasan populer dengan fasilitas lengkap "
                "dan nilai investasi yang tinggi."
            )
        else:
            desc = (
                "<b>Rumah mewah</b> — berada di area eksklusif dengan fasilitas terbaik "
                "dan nilai pasar yang stabil. Cocok untuk investasi jangka panjang."
            )

        # ================================
        # Tampilkan deskripsi dalam card
        # ================================
        st.markdown(f"""
        <div class="card" style='margin-top:10px'>
            <p style='font-size:15px; color:#9aa3b2; text-align:justify;'>
            {desc}
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Pesan ringan sebelum prediksi dijalankan
        st.markdown(
            "<div class='small-muted'>Tekan tombol <b>Prediksi Sekarang</b> untuk melihat estimasi harga dan insight.</div>",
            unsafe_allow_html=True
        )


with col_right:
    # Tentukan gambar berdasarkan prediksi terakhir
    p = st.session_state.get("pred", None)

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

    if p is None:
        st.image(img3, width=220)
        st.image(img4, width=220)
    else:
        st.image(img1, width=420)
        st.image(img2, width=420)

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






