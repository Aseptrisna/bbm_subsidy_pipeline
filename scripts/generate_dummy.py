import csv
import numpy as np
from datetime import datetime, timedelta
import random

# ================================
# CONFIG
# ================================
NUM_MONTHS = 12
NUM_VEHICLES = 4

VEHICLES = [
    {"id": 1, "plat_nomor": "BE-1001-XX", "nama_pemilik": "Asep", "jenis_kendaraan": "Mobil", "kuota_bulanan_liter": 100},
    {"id": 2, "plat_nomor": "BE-1002-XX", "nama_pemilik": "Budi", "jenis_kendaraan": "Mobil", "kuota_bulanan_liter": 120},
    {"id": 3, "plat_nomor": "BE-1003-XX", "nama_pemilik": "Cici", "jenis_kendaraan": "Motor", "kuota_bulanan_liter": 30},
    {"id": 4, "plat_nomor": "BE-1004-XX", "nama_pemilik": "Dedi", "jenis_kendaraan": "Mobil", "kuota_bulanan_liter": 500},
]

STATIONS = [
    {"id": i, "kode_spbu": f"SPBU-{i:04d}", "nama_spbu": f"SPBU Nomor {i}", "kota": "Bandar Lampung"}
    for i in range(1, 11)
]

start_date = datetime(2025, 1, 1)
end_date = datetime(2026, 1, 1)


# ================================
# HELPER
# ================================
def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def random_time_of_day():
    p = np.random.rand()
    if p < 0.25:   # 06.00–09.00
        hour = np.random.randint(6, 10)
    elif p < 0.60: # 11.00–14.00
        hour = np.random.randint(11, 15)
    elif p < 0.90: # 16.00–19.00
        hour = np.random.randint(16, 20)
    else:          # 20.00–23.00
        hour = np.random.randint(20, 24)

    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return hour, minute, second


def realistic_volume(vehicle, jenis_bbm):
    if vehicle["jenis_kendaraan"].lower() == "motor":
        return round(np.random.uniform(1.5, 4.5), 2)

    if jenis_bbm == "PERTALITE":
        return round(np.random.uniform(15, 40), 2)

    if jenis_bbm == "SOLAR":
        return round(np.random.uniform(20, 70), 2)

    return round(np.random.uniform(5, 30), 2)


def pick_station():
    p = np.random.rand()
    if p < 0.40:  # SPBU favorit
        return np.random.choice([1, 2, 3])
    return np.random.randint(4, 11)


# ================================
# EXPORT VEHICLES
# ================================
def export_vehicles():
    with open("./data/vehicles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "plat_nomor", "nama_pemilik", "jenis_kendaraan", "kuota_bulanan_liter"])

        for v in VEHICLES:
            writer.writerow([
                v["id"], v["plat_nomor"], v["nama_pemilik"],
                v["jenis_kendaraan"], v["kuota_bulanan_liter"]
            ])


# ================================
# EXPORT STATIONS
# ================================
def export_stations():
    with open("./data/stations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "kode_spbu", "nama_spbu", "kota"])

        for s in STATIONS:
            writer.writerow([s["id"], s["kode_spbu"], s["nama_spbu"], s["kota"]])


# ================================
# REALISTIC TRANSACTIONS
# ================================
def export_transactions():
    with open("./data/fuel_transactions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "vehicle_id", "station_id", "waktu_transaksi", "jenis_bbm", "volume_liter"])

        tx_id = 1

        for veh in VEHICLES:

            # Pengisian tergantung kebiasaan kendaraan
            if veh["id"] == 4: # kuota besar
                freq_per_month = np.random.randint(12, 20)
            elif veh["jenis_kendaraan"] == "Motor":
                freq_per_month = np.random.randint(4, 8)
            else:
                freq_per_month = np.random.randint(6, 12)

            total_tx = freq_per_month * NUM_MONTHS

            # Tentukan jenis BBM tetap
            jenis_bbm = "PERTALITE" if veh["jenis_kendaraan"] == "Motor" else np.random.choice(["PERTALITE", "SOLAR"])

            for _ in range(total_tx):

                # tanggal acak per bulan
                day_offset = np.random.randint(0, (end_date - start_date).days)
                date = start_date + timedelta(days=day_offset)

                # weekend lebih sepi
                if date.weekday() >= 5 and random.random() < 0.20:
                    continue

                # random jam
                h, m, s = random_time_of_day()
                date = date.replace(hour=h, minute=m, second=s)

                volume = realistic_volume(veh, jenis_bbm)
                station = pick_station()

                writer.writerow([
                    tx_id,
                    veh["id"],
                    station,
                    iso(date),
                    jenis_bbm,
                    volume
                ])
                tx_id += 1

    print("Total transaksi:", tx_id - 1)


# ================================
# RUN
# ================================
if __name__ == "__main__":
    print("Generating realistic dummy CSV...")
    export_vehicles()
    export_stations()
    export_transactions()
    print("Selesai!")
