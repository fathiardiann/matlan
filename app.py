# =============================================
# 🏠 PREDIKSI HARGA RUMAH (CALIFORNIA HOUSING)
# Versi Streamlit — UI/UX Senior
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

# =====================================================
# 🔧 Konfigurasi Halaman
# =====================================================
st.set_page_config(
    page_title="Prediksi Harga Rumah California",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 🎨 Styling Custom CSS
# =====================================================
st.markdown("""
<style>
    body {
        background-color: #0E1117;
        color: #E6E6E6;
    }
    .main {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #58A6FF;
        text-align: center;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(90deg, #0078FF, #0057CC);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3399FF, #0078FF);
        transform: scale(1.03);
    }
    .card {
        background-color: #161B22;
        border-radius: 15px;
        padding: 1.5em;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.4);
        text-align: center;
        margin-bottom: 1.2em;
    }
    hr {
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 📦 Load Data dan Latih Model
# =====================================================
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    return df, data.feature_names

df, feature_names = load_data()

X = df[feature_names]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# =====================================================
# 🧭 Header
# =====================================================
st.markdown("<h1>🏠 Prediksi Harga Rumah California</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Model regresi linier dengan visualisasi dan analisis performa.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# =====================================================
# 🧩 Tabs Layout (UX Senior)
# =====================================================
tab_pred, tab_eval, tab_vis = st.tabs(["🔮 Prediksi", "📊 Evaluasi Model", "📈 Visualisasi Data"])

# =====================================================
# 🔮 TAB 1 — Prediksi Harga
# =====================================================
with tab_pred:
    st.sidebar.header("Masukkan Data Rumah")
    MedInc = st.sidebar.slider("Pendapatan Median (x10.000 USD)", 0.0, 15.0, 8.3)
    HouseAge = st.sidebar.slider("Usia Rumah (tahun)", 1, 50, 20)
    AveRooms = st.sidebar.slider("Rata-rata Jumlah Ruangan", 1.0, 10.0, 6.5)
    AveBedrms = st.sidebar.slider("Rata-rata Jumlah Kamar Tidur", 0.5, 5.0, 1.0)
    Population = st.sidebar.number_input("Populasi", 100, 5000, 1200)
    AveOccup = st.sidebar.slider("Rata-rata Penghuni per Rumah", 1.0, 10.0, 2.8)
    Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.88)
    Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -122.25)

    st.markdown("### 🔮 Estimasi Harga Rumah")

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
        lower, upper = pred * 0.95, pred * 1.05

        st.markdown(f"""
        <div class='card'>
            <h3>🏡 Estimasi Harga Rumah</h3>
            <p style='font-size:28px; color:#00FFAA;'><b>${pred:,.0f}</b></p>
            <p>Rentang estimasi: ${lower:,.0f} – ${upper:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("💡 Gunakan sidebar untuk menyesuaikan parameter rumah dan tekan tombol *Prediksi Sekarang*.")

# =====================================================
# 📊 TAB 2 — Evaluasi Model
# =====================================================
with tab_eval:
    st.markdown("### ⚙️ Evaluasi Model")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Squared Error", f"{mse:.4f}")
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Koefisien": model.coef_.round(4)
        }).sort_values(by="Koefisien", ascending=False)
        st.dataframe(coef_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Interpretasi:")
    st.markdown("- **R² Score** menunjukkan seberapa baik model menjelaskan variasi data (mendekati 1 berarti bagus).")
    st.markdown("- **Koefisien positif** artinya fitur tersebut meningkatkan harga rumah, sedangkan negatif menurunkan.")

# =====================================================
# 📈 TAB 3 — Visualisasi Data
# =====================================================
with tab_vis:
    st.markdown("### 💰 Hubungan Pendapatan vs Harga Rumah")
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(x=df["MedInc"], y=df["MedHouseVal"], alpha=0.4, color="#58A6FF", ax=ax)
    ax.set_xlabel("Pendapatan Median (x10.000 USD)")
    ax.set_ylabel("Harga Rumah Median (x100.000 USD)")
    ax.grid(alpha=0.2)
    st.pyplot(fig)

    st.markdown("### 🎯 Prediksi vs Aktual")
    fig2, ax2 = plt.subplots(figsize=(6,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="#00FFAA", ax=ax2)
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    ax2.set_xlabel("Harga Aktual (x100.000 USD)")
    ax2.set_ylabel("Harga Prediksi")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)

    st.caption("Model linear cenderung memiliki kesalahan kecil pada kisaran tengah, namun mulai menyimpang di harga ekstrem.")

# =====================================================
# 🧠 Footer
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:13px; color:gray;'>© 2025 California Housing Predictor | Dibuat dengan Streamlit</p>", unsafe_allow_html=True)
