# db_utils.py
import os
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from loguru import logger

DATABASE_URL = os.getenv("DATABASE_URL") or (
    "postgresql+asyncpg://postgres.erggmiacluaaeiiuutvp:BBMSubsidi2025"
    "@aws-1-ap-southeast-2.pooler.supabase.com:5432/postgres"
)

engine: AsyncEngine = create_async_engine(DATABASE_URL, future=True)


async def create_tables():
    """Create base tables and ENUM safely (asyncpg-compatible)."""
    async with engine.begin() as conn:
        # 1. Create vehicles table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS vehicles (
                id SERIAL PRIMARY KEY,
                plat_nomor TEXT UNIQUE,
                nama_pemilik TEXT,
                jenis_kendaraan TEXT,
                kuota_bulanan_liter FLOAT
            );
        """))

        # 2. Create enum type ONLY if doesn't exist
        await conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_type WHERE typname = 'bbm_enum'
                ) THEN
                    CREATE TYPE bbm_enum AS ENUM ('Pertalite', 'Solar');
                END IF;
            END
            $$;
        """))

        # 3. Create fuel_transactions table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fuel_transactions (
                id SERIAL PRIMARY KEY,
                vehicle_id INTEGER REFERENCES vehicles(id),
                station_id INTEGER,
                waktu_transaksi TIMESTAMP WITH TIME ZONE,
                jenis_bbm bbm_enum,
                volume_liter FLOAT
            );
        """))

    logger.info("All tables ensured.")


async def execute_many(query: str, params: list):
    async with engine.begin() as conn:
        await conn.execute(text(query), params)


async def fetch_df(query: str, params: dict = None):
    import pandas as pd
    async with engine.connect() as conn:
        result = await conn.execute(text(query), params or {})
        rows = result.fetchall()
        cols = result.keys()
    return pd.DataFrame(rows, columns=cols)


async def insert_dataframe(table: str, df):
    if df.empty:
        return

    cols = list(df.columns)
    placeholders = ", ".join([f":{c}" for c in cols])
    colnames = ", ".join(cols)
    query = f"INSERT INTO {table} ({colnames}) VALUES ({placeholders})"
    params = df.to_dict(orient='records')

    await execute_many(query, params)


async def save_recommendations_table(method: str):
    table_name = f"recommendations_{method}"

    async with engine.begin() as conn:
        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                vehicle_id INTEGER,
                method TEXT,
                period_type TEXT,
                period_start TIMESTAMP WITH TIME ZONE,
                period_end TIMESTAMP WITH TIME ZONE,
                recommended_liter FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
            );
        """))


async def insert_recommendations(method: str, df):
    await save_recommendations_table(method)
    await insert_dataframe(f"recommendations_{method}", df)
    logger.info(f"Inserted {len(df)} rows into recommendations_{method}.")
