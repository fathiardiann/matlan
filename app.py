# =============================================
# 🏠 PREDIKSI HARGA RUMAH (CALIFORNIA HOUSING)
# Versi Streamlit dengan Tampilan Modern
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

# --- Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling CSS Custom
st.markdown("""
<style>
    body {background-color: #0E1117;}
    .main {
        background-color: #0E1117;
        color: white;
        font-family: 'Arial';
    }
    h1, h2, h3 {
        color: #58a6ff;
        text-align: center;
    }
    .stButton>button {
        background-color: #0078FF;
        color: white;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0057CC;
        color: #FFF;
    }
    .card {
        background-color: #161B22;
        padding: 1.5em;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Judul Halaman
st.markdown("<h1>🏠 Prediksi Harga Rumah California</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Model Linear Regression menggunakan dataset California Housing.</p>", unsafe_allow_html=True)

# --- Load Data dan Model
data = fetch_california_housing(as_frame=True)
df = data.frame
X = df[data.feature_names]
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

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

# --- Tombol Prediksi
st.markdown("### 🔮 Hasil Prediksi")

if st.button("Prediksi Sekarang"):
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

    st.markdown(f"""
    <div class='card'>
        <h3>🏡 Estimasi Harga Rumah:</h3>
        <p style='font-size:22px; color:#00FFAA;'><b>${pred:,.0f}</b></p>
    </div>
    """, unsafe_allow_html=True)

# --- Evaluasi Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📊 Evaluasi Model")
    st.metric(label="Mean Squared Error", value=f"{mse:.4f}")
    st.metric(label="R² Score", value=f"{r2:.4f}")

with col2:
    st.markdown("### ⚙️ Koefisien Model")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Koefisien": model.coef_.round(4)
    }).sort_values(by="Koefisien", ascending=False)
    st.dataframe(coef_df, use_container_width=True)

# --- Visualisasi
st.markdown("### 📈 Hubungan Pendapatan vs Harga Rumah")
fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(x=df["MedInc"], y=df["MedHouseVal"], alpha=0.4, color="#58a6ff", ax=ax)
ax.set_xlabel("Pendapatan Median (x10.000 USD)")
ax.set_ylabel("Harga Rumah Median (x100.000 USD)")
st.pyplot(fig)

st.markdown("### 📉 Prediksi vs Aktual")
fig2, ax2 = plt.subplots(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="#00FFAA", ax=ax2)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax2.set_xlabel("Harga Aktual (x100.000 USD)")
ax2.set_ylabel("Harga Prediksi")
st.pyplot(fig2)
