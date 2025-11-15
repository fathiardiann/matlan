# 🏠 Prediksi Harga Rumah di California  
### _Menggunakan Linear Regression (Regresi Linear)_

> 📊 Proyek Data Science untuk memprediksi harga rumah berdasarkan faktor ekonomi, sosial, dan geografis menggunakan model **Regresi Linear** dari Scikit-learn.

---

## 🌟 Deskripsi Proyek  
Harga rumah di California sangat bervariasi antar wilayah — dipengaruhi oleh pendapatan, usia rumah, populasi, dan lokasi geografis.  
Proyek ini bertujuan untuk:
- Menganalisis hubungan antar fitur wilayah terhadap harga rumah.  
- Membangun model **Linear Regression** untuk melakukan prediksi.  
- Mengukur akurasi hasil prediksi dibandingkan data asli.  
- Mengidentifikasi fitur yang paling berpengaruh terhadap harga rumah.  

---

## 👥 Anggota Kelompok  
📘 **Mata Kuliah:** Matematika Lanjut  
📚 **Kelas:** 2IA18  
>> zathi
---

## 🧩 Dataset  
**📂 Sumber:** `sklearn.datasets.fetch_california_housing`  
Dataset ini berasal dari Sensus AS tahun 1990 dan sering digunakan sebagai benchmark model regresi.

| Detail | Keterangan |
|--------|-------------|
| Jumlah Data | ± 20.640 baris |
| Jumlah Fitur | 9 kolom |
| Target Prediksi | `MedHouseVal` (Median Harga Rumah) |
| Asal Dataset | U.S. Census (1990) |

**Fitur yang Digunakan:**
- `MedInc` → Pendapatan rata-rata penduduk  
- `HouseAge` → Usia rata-rata rumah  
- `AveRooms` → Rata-rata jumlah ruangan  
- `AveBedrms` → Rata-rata jumlah kamar tidur  
- `Population` → Jumlah populasi  
- `AveOccup` → Rata-rata penghuni per rumah  
- `Latitude`, `Longitude` → Koordinat geografis  

---

## ⚙️ Teknologi & Library  
| Library | Fungsi |
|----------|--------|
| `pandas` | Mengolah dan menganalisis data |
| `numpy` | Operasi numerik dan array |
| `matplotlib.pyplot` | Visualisasi data |
| `seaborn` | Visualisasi data yang menarik |
| `sklearn.datasets` | Mengambil dataset California |
| `sklearn.model_selection` | Membagi data menjadi train/test |
| `sklearn.linear_model` | Membuat model Linear Regression |
| `sklearn.metrics` | Mengevaluasi performa model |

---

## 🔍 Alur Proses  
1️⃣ **Data Splitting**  
Dataset dibagi menjadi **80% data latih (train)** dan **20% data uji (test)**.  

2️⃣ **Training**  
Model *Linear Regression* dilatih menggunakan fitur-fitur yang tersedia untuk mempelajari hubungan antar variabel.  

3️⃣ **Testing**  
Model digunakan untuk memprediksi harga rumah berdasarkan data baru.  

4️⃣ **Evaluasi**  
Hasil prediksi dibandingkan dengan harga aktual menggunakan metrik seperti:  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- R² Score  

---

## 🧮 Konsep Regresi Linear  
Persamaan umum regresi linear:

\[
Y = β₀ + β₁X₁ + β₂X₂ + … + βₙXₙ + ε
\]

**Keterangan:**
- `Y` → Harga median rumah  
- `X₁...Xₙ` → Fitur seperti pendapatan, usia rumah, populasi, dll  
- `β₀` → Intersep (nilai awal)  
- `β₁...βₙ` → Koefisien tiap fitur  
- `ε` → Error (selisih hasil prediksi dan aktual)

---

## 📊 Hasil Analisis  
✅ Model mampu memprediksi harga rumah dengan **akurasi yang baik**.  
✅ Faktor paling berpengaruh terhadap harga rumah adalah:  
   - **Pendapatan rata-rata (`MedInc`)**  
   - **Lokasi geografis (`Latitude`, `Longitude`)**  
✅ Model membantu memahami pola ekonomi & perumahan secara **cepat dan berbasis data**.

---

## 🧾 Contoh Kode  
```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Split data
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
