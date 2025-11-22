import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Enum, Float, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
import enum
from datetime import timedelta
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Library Machine Learning
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ==========================================
# 1. KONFIGURASI & STRUKTUR DATA
# ==========================================
Base = declarative_base()

class VehicleTypeEnum(str, enum.Enum):
    MOTOR = "MOTOR"
    MOBIL = "MOBIL"
    ANGKUTAN_UMUM = "ANGKUTAN_UMUM"
    LAIN = "LAIN"

class Vehicle(Base):
    __tablename__ = "vehicles"
    id = Column(Integer, primary_key=True, index=True)
    plat_nomor = Column(String(50), unique=True, nullable=False, index=True)
    nama_pemilik = Column(String(255), nullable=False)
    jenis_kendaraan = Column(String(50), default=VehicleTypeEnum.LAIN.value)
    kuota_bulanan_liter = Column(Integer, default=60)

class FuelTransaction(Base):
    __tablename__ = "fuel_transactions"
    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True)
    waktu_transaksi = Column(DateTime(timezone=True), nullable=False)
    jenis_bbm = Column(String, nullable=False)
    volume_liter = Column(Float, nullable=False)

# ==========================================
# 2. FUNGSI LOAD & PREPROCESSING DATA
# ==========================================

def load_vehicles_map(filepath):
    try:
        df = pd.read_csv(filepath)
        return pd.Series(df.plat_nomor.values, index=df.id).to_dict()
    except Exception as e:
        print(f"[!] Warning: Tidak bisa load {filepath} ({e}). Menggunakan ID sebagai Plat.")
        return {}

def load_data_from_csv(filepath):
    print(f"[-] Memuat data transaksi dari {filepath}...")
    df = pd.read_csv(filepath)
    df['waktu_transaksi'] = pd.to_datetime(df['waktu_transaksi'])
    
    df['date'] = df['waktu_transaksi'].dt.date
    daily_data = df.groupby(['vehicle_id', 'date'])['volume_liter'].sum().reset_index()
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    
    return daily_data

def prepare_vehicle_series(df, vehicle_id):
    v_df = df[df['vehicle_id'] == vehicle_id].copy()
    v_df.set_index('date', inplace=True)
    idx = pd.date_range(start=v_df.index.min(), end=v_df.index.max(), freq='D')
    v_df = v_df.reindex(idx, fill_value=0)
    return v_df['volume_liter']

# ==========================================
# 3. METODE PREDIKSI (ARIMA, LSTM, PROPHET)
# ==========================================

def predict_arima(series, days_to_predict=365):
    train = series.values
    try:
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=days_to_predict)
        return np.maximum(forecast, 0)
    except Exception:
        return np.zeros(days_to_predict)

def create_lstm_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def predict_lstm(series, days_to_predict=365):
    try:
        data = series.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        look_back = 30
        if len(data) < look_back + 2:
            return np.zeros(days_to_predict)

        X_train, y_train = create_lstm_dataset(data_scaled, look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        
        model = Sequential()
        model.add(LSTM(50, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', verbose=0)
        model.fit(X_train, y_train, epochs=5, batch_size=1, verbose=0)
        
        predictions = []
        curr_input = data_scaled[-look_back:].reshape(1, 1, look_back)
        
        for _ in range(days_to_predict):
            pred = model.predict(curr_input, verbose=0)
            predictions.append(pred[0, 0])
            new_input = np.append(curr_input[0,0,1:], pred[0,0])
            curr_input = new_input.reshape(1, 1, look_back)
            
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return np.maximum(predictions.flatten(), 0)
    except Exception:
        return np.zeros(days_to_predict)

def predict_prophet(series, days_to_predict=365):
    try:
        df_prophet = series.reset_index()
        df_prophet.columns = ['ds', 'y']
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=days_to_predict)
        forecast = model.predict(future)
        return np.maximum(forecast.tail(days_to_predict)['yhat'].values, 0)
    except Exception:
        return np.zeros(days_to_predict)

# ==========================================
# 4. FUNGSI SAVE CSV & GENERATE PNG (UPDATED)
# ==========================================

def generate_plot(x_values, y_values, plat_nomor, method_name, period_type, output_folder):
    """
    Membuat plot Harian (Line) atau Bulanan/Tahunan (Bar).
    """
    plt.figure(figsize=(12, 6))
    
    safe_plat = str(plat_nomor).replace(" ", "_")
    filename = f"{output_folder}/{method_name}_{safe_plat}_{period_type}.png"
    
    if period_type == "Harian":
        # Plot Line untuk Harian (karena datanya banyak)
        plt.plot(x_values, y_values, label=f'Prediksi {method_name}', color='#1f77b4', linewidth=1)
        plt.title(f'Prediksi Harian BBM - {plat_nomor} ({method_name})')
        
        # Format tanggal sumbu X agar tidak menumpuk
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
    else:
        # Plot Bar untuk Bulanan dan Tahunan (lebih enak dilihat)
        # Konversi index periode ke string agar bisa di-plot sebagai kategori
        x_labels = x_values.astype(str)
        plt.bar(x_labels, y_values, color='#ff7f0e', alpha=0.7, label=f'Total {method_name}')
        plt.title(f'Total Kuota BBM {period_type} - {plat_nomor} ({method_name})')
        
        # Tambahkan label nilai di atas batang
        for i, v in enumerate(y_values):
            plt.text(i, v + (0.01 * max(y_values)), f"{v:.1f}", ha='center', fontsize=8)
        
        plt.xticks(rotation=45, ha='right')

    plt.xlabel('Waktu')
    plt.ylabel('Liter')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(filename)
    plt.close()

def save_results_to_csv_and_png(results, method_name, vehicle_map):
    # Buat folder khusus per metode
    output_folder = f"hasil_prediksi_images/{method_name}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_records = []
    print(f"[-] Memproses Output (CSV & PNG) untuk metode {method_name}...")

    for vid, data_prediksi in results.items():
        plat = vehicle_map.get(vid, f"UNKNOWN-{vid}")
        
        # 1. Siapkan DataFrame Harian
        dates = pd.date_range(start="2026-01-01", periods=len(data_prediksi), freq='D')
        df_daily = pd.DataFrame({'tanggal': dates, 'liter': data_prediksi})
        df_daily.set_index('tanggal', inplace=True) # Set index ke tanggal untuk resampling

        # --- A. PROSES HARIAN ---
        # Simpan ke List
        for date, row in df_daily.iterrows():
            all_records.append({
                'vehicle_id': vid, 'plat_nomor': plat, 'metode': method_name,
                'type': 'Harian', 'waktu': date.strftime('%Y-%m-%d'),
                'kuota_liter': round(row['liter'], 2)
            })
        # Generate PNG Harian
        generate_plot(df_daily.index, df_daily['liter'], plat, method_name, "Harian", output_folder)

        # --- B. PROSES BULANAN (Aggregasi) ---
        df_monthly = df_daily.resample('M')['liter'].sum()
        # Simpan ke List
        for period, val in df_monthly.items():
            all_records.append({
                'vehicle_id': vid, 'plat_nomor': plat, 'metode': method_name,
                'type': 'Bulanan', 'waktu': period.strftime('%Y-%m'),
                'kuota_liter': round(val, 2)
            })
        # Generate PNG Bulanan
        generate_plot(df_monthly.index, df_monthly.values, plat, method_name, "Bulanan", output_folder)

        # --- C. PROSES TAHUNAN (Aggregasi) ---
        df_yearly = df_daily.resample('Y')['liter'].sum()
        # Simpan ke List
        for period, val in df_yearly.items():
            all_records.append({
                'vehicle_id': vid, 'plat_nomor': plat, 'metode': method_name,
                'type': 'Tahunan', 'waktu': period.strftime('%Y'),
                'kuota_liter': round(val, 2)
            })
        # Generate PNG Tahunan
        generate_plot(df_yearly.index, df_yearly.values, plat, method_name, "Tahunan", output_folder)

    # Simpan semua ke CSV
    df_final = pd.DataFrame(all_records)
    csv_filename = f"rekomendasi_{method_name}.csv"
    df_final.to_csv(csv_filename, index=False)
    print(f"[+] Selesai: {csv_filename} & Gambar tersimpan di {output_folder}/")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    trx_file = 'data/fuel_transactions.csv' 
    veh_file = 'data/vehicles.csv'
    
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Folder 'data' dibuat. Silakan masukkan file CSV Anda di sana.")
        return

    try:
        df = load_data_from_csv(trx_file)
        vehicle_map = load_vehicles_map(veh_file)
    except FileNotFoundError:
        print(f"File {trx_file} tidak ditemukan.")
        return

    unique_vehicles = df['vehicle_id'].unique()
    print(f"Ditemukan {len(unique_vehicles)} kendaraan unik.")
    
    results_arima = {}
    results_lstm = {}
    results_prophet = {}
    
    print("[-] Memulai proses prediksi...")
    for vid in unique_vehicles:
        series = prepare_vehicle_series(df, vid)
        
        # Lakukan Prediksi (Setahun kedepan)
        results_arima[vid] = predict_arima(series, 365)
        results_lstm[vid] = predict_lstm(series, 365)
        results_prophet[vid] = predict_prophet(series, 365)
    
    # Simpan CSV dan Generate PNG (Harian, Bulanan, Tahunan)
    save_results_to_csv_and_png(results_arima, "ARIMA", vehicle_map)
    save_results_to_csv_and_png(results_lstm, "LSTM", vehicle_map)
    save_results_to_csv_and_png(results_prophet, "Prophet", vehicle_map)
    
    print("\n[SELESAI] Semua proses berhasil.")

if __name__ == "__main__":
    main()