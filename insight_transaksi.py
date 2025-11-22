import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================
# 1. Load Data
# ================================
df = pd.read_csv("data/fuel_transactions.csv", parse_dates=["waktu_transaksi"])

df.columns = df.columns.str.lower()

print("==== SAMPLE DATA ====")
print(df.head(), "\n")


# ================================
# 2. Insight Dasar
# ================================
print("==== INFO DATA ====")
print(df.info(), "\n")

print("==== STATISTIK VOLUME ====")
print(df["volume_liter"].describe(), "\n")

print("Total transaksi:", len(df))
print("Total liter terpakai:", df["volume_liter"].sum())
print("Rata-rata liter per transaksi:", df["volume_liter"].mean(), "\n")


# ================================
# 3. Insight Per Kendaraan
# ================================
insight_vehicle = df.groupby("vehicle_id")["volume_liter"].agg(
    total_liter="sum",
    mean_liter="mean",
    max_liter="max",
    min_liter="min",
    total_transaksi="count"
).reset_index()

print("==== INSIGHT PER KENDARAAN ====")
print(insight_vehicle, "\n")


# ================================
# 3A. Grafik Total Liter per Kendaraan
# ================================
plt.figure(figsize=(12,6))
plt.bar(insight_vehicle["vehicle_id"].astype(str), insight_vehicle["total_liter"])
plt.title("Total Liter BBM per Kendaraan")
plt.xlabel("Vehicle ID")
plt.ylabel("Total Liter")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ================================
# 3B. Grafik Rata-rata Liter per Transaksi
# ================================
plt.figure(figsize=(12,6))
plt.bar(insight_vehicle["vehicle_id"].astype(str), insight_vehicle["mean_liter"])
plt.title("Rata-rata Liter per Transaksi per Kendaraan")
plt.xlabel("Vehicle ID")
plt.ylabel("Rata-rata Liter")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ================================
# 4. Volume BBM per Bulan
# ================================
df["bulan"] = df["waktu_transaksi"].dt.to_period("M")
monthly_usage = df.groupby("bulan")["volume_liter"].sum().reset_index()

print("==== PEMAKAIAN PER BULAN ====")
print(monthly_usage, "\n")

# Grafik bulanan
plt.figure(figsize=(12,6))
plt.plot(monthly_usage["bulan"].astype(str), monthly_usage["volume_liter"], marker="o")
plt.title("Total Pemakaian BBM per Bulan")
plt.xlabel("Bulan")
plt.ylabel("Liter")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# ================================
# 5. Frekuensi Pengisian per Hari
# ================================
df["tanggal"] = df["waktu_transaksi"].dt.date
daily_freq = df.groupby("tanggal").size().reset_index(name="jumlah_transaksi")

print("==== FREKUENSI PENGISIAN PER HARI ====")
print(daily_freq, "\n")

# Grafik frekuensi harian
plt.figure(figsize=(12,6))
plt.plot(daily_freq["tanggal"], daily_freq["jumlah_transaksi"], marker=".")
plt.title("Frekuensi Pengisian BBM per Hari")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Transaksi")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# ================================
# 6. Insight Jenis BBM
# ================================
bbm_usage = df.groupby("jenis_bbm")["volume_liter"].sum().reset_index()

print("==== PEMAKAIAN PER JENIS BBM ====")
print(bbm_usage, "\n")

# Grafik jenis BBM
plt.figure(figsize=(8,6))
plt.bar(bbm_usage["jenis_bbm"], bbm_usage["volume_liter"])
plt.title("Pemakaian Total per Jenis BBM")
plt.xlabel("Jenis BBM")
plt.ylabel("Liter")
plt.tight_layout()
plt.show()


# ================================
# 7. Outlier Detection
# ================================
q1 = df["volume_liter"].quantile(0.25)
q3 = df["volume_liter"].quantile(0.75)
iqr = q3 - q1

outliers = df[df["volume_liter"] > q3 + 1.5 * iqr]

print("==== OUTLIER VOLUME ====")
print(outliers, "\n")


# ================================
# 8. Kendaraan dengan Pemakaian Tertinggi & Terendah
# ================================
top_vehicle = insight_vehicle.sort_values("total_liter", ascending=False).head(1)
low_vehicle = insight_vehicle.sort_values("total_liter", ascending=True).head(1)

print("Kendaraan pemakaian tertinggi:")
print(top_vehicle, "\n")

print("Kendaraan pemakaian terendah:")
print(low_vehicle, "\n")
