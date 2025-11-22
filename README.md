# Sistem Prediksi Kuota Subsidi BBM (AI-Based)

Proyek ini adalah sistem berbasis Python untuk memprediksi kebutuhan kuota subsidi bahan bakar minyak (BBM) per kendaraan. Sistem ini menggunakan tiga metode Machine Learning (ARIMA, LSTM, dan Prophet) untuk menghasilkan rekomendasi kuota masa depan berdasarkan data historis transaksi.

Output dari sistem ini berupa File CSV (untuk laporan Harian, Bulanan, Tahunan) dan Grafik PNG (visualisasi tren prediksi per kendaraan).

## 🚀 Fitur Utama

**Multi-Model Prediction:** Menggunakan 3 algoritma berbeda untuk perbandingan:
- **ARIMA:** Statistik klasik untuk time series.
- **LSTM (Deep Learning):** Jaringan saraf tiruan untuk menangkap pola kompleks.
- **Facebook Prophet:** Kuat menangani tren musiman dan data yang bolong.

**Comprehensive Reporting:**
- Menyimpan hasil dalam format CSV yang mudah diolah.
- Mencakup kolom: Metode, Plat Nomor, Tipe (Harian/Bulanan/Tahunan), dan Kuota.

**Automated Visualization:** Membuat grafik prediksi (PNG) secara otomatis untuk setiap kendaraan.

**Data Aggregation:** Mengolah data transaksi mentah menjadi ringkasan harian sebelum diprediksi.

## 📦 Persyaratan Sistem

- Python 3.8 atau lebih baru.
- Library yang dibutuhkan (tercantum di `requirements.txt`).

## 🛠️ Instalasi

1. Clone atau Download repositori ini.
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
