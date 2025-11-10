# =============================================
# 🏠 PREDIKSI HARGA RUMAH (CALIFORNIA HOUSING)
# Versi Kartu Simetris & Terpusat (Modern Clean)
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠", layout="wide")

# --- CSS Custom untuk layout tengah dan kartu
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.main {
    background-color: #0E1117;
    padding-top: 30px;
}
.center-container {
    max-width: 1000px;
    margin: 0 auto;
}
.card {
    background: #161B22;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.05);
    text-align: center;
    width: 100%;
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: #00FFAA;
}
.metric-label {
    font-size: 16px;
    color: #aaa;
}
.section {
    margin-top: 20px;
    background: #161B22;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 0 10px rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# --- Judul
st.markdown("<div class='center-container'><h1 style='text-align:center;'>🏠 Prediksi Harga Rumah California</h1></div>", unsafe_allow_html=True)

# --- Model
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

# --- Sidebar Input
st.sidebar.header("Masukkan Data Rumah")
MedInc = st.sidebar.slider("Pendapatan Median (x10.000 USD)", 0.0, 15.0, 8.3)
HouseAge = st.sidebar.slider("Usia Rumah (tahun)", 1, 50, 20)
AveRooms = st.sidebar.slider("Rata-rata Jumlah Ruangan", 1.0, 10.0, 6.5)
AveBedrms = st.sidebar.slider("Rata-rata Jumlah Kamar Tidur", 0.5, 5.0, 1.0)
Population = st.sidebar.number_input("Populasi", 100, 5000, 1200)
AveOccup = st.sidebar.slider("Rata-rata Penghuni per Rumah", 1.0, 10.0, 2.8)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.88)
Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -122.25)

# --- Prediksi
sample = pd.DataFrame([{
    'MedInc': MedInc, 'HouseAge': HouseAge, 'AveRooms': AveRooms,
    'AveBedrms': AveBedrms, 'Population': Population, 'AveOccup': AveOccup,
    'Latitude': Latitude, 'Longitude': Longitude
}])
pred = model.predict(sample)[0] * 100000

# --- Kontainer Tengah
st.markdown("<div class='center-container'>", unsafe_allow_html=True)

# --- Estimasi Harga Rumah (Baris 1)
st.markdown("<div class='card'><h3>🏡 Estimasi Harga Rumah:</h3>", unsafe_allow_html=True)
st.markdown(f"<div class='metric-value'>${pred:,.0f}</div></div>", unsafe_allow_html=True)

# --- Baris 2 (MSE & R2)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>📉 Mean Squared Error</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{mse:.4f}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label'>📈 R² Score</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{r2:.4f}</div></div>", unsafe_allow_html=True)

# --- Baris 3: Koefisien Model
st.markdown("<div class='section'><h3 style='text-align:center;'>⚙️ Koefisien Model</h3>", unsafe_allow_html=True)
coef_df = pd.DataFrame({"Feature": X.columns, "Koefisien": model.coef_.round(4)})
coef_df = coef_df.sort_values(by="Koefisien", ascending=False)
st.dataframe(coef_df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Baris 4: Visualisasi
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>📊 Visualisasi</h3>", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    fig1, ax1 = plt.subplots(figsize=(5,4))
    sns.scatterplot(x=df["MedInc"], y=df["MedHouseVal"], alpha=0.4, color="#00C2FF", ax=ax1)
    ax1.set_xlabel("Pendapatan Median (x10.000 USD)")
    ax1.set_ylabel("Harga Rumah Median (x100.000 USD)")
    st.pyplot(fig1)

with col4:
    fig2, ax2 = plt.subplots(figsize=(5,4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#00FFAA", ax=ax2)
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    ax2.set_xlabel("Harga Aktual (x100.000 USD)")
    ax2.set_ylabel("Harga Prediksi")
    st.pyplot(fig2)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
