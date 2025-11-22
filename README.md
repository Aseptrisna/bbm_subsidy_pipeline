# BBM Subsidy Recommendation Pipeline
Project: Rekomendasi Kuota Subsidi BBM per Kendaraan (ARIMA, LSTM, Prophet)

## Cara cepat menjalankan (demo)
1. Buat virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # linux / mac
   .venv\Scripts\activate    # windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Generate dummy data dan simpan ke database Postgres/Supabase:
   ```bash
   python scripts/generate_dummy.py 
   ```
4. Jalankan pipeline:
   ```bash
   python run_pipeline.py --method all --db "postgresql+asyncpg://postgres:BBMSubsidi2025@aws-1-ap-southeast-2.pooler.supabase.com:5432/postgres"
   ```
   ```python run_pipeline.py --method arima

5. Hasil rekomendasi akan disimpan di tabel `recommendations_<method>` dan file `rekomendasi_<method>.xlsx`.

## Catatan
- Gunakan environment variable `DATABASE_URL` untuk kredensial DB jika ingin lebih aman.
- Untuk LSTM: TensorFlow akan menggunakan GPU jika tersedia.
- Prophet dan pmdarima kadang perlu build tools; gunakan wheel jika instalasi gagal.
