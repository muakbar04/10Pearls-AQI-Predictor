"""
fetch_features.py
Fetches REAL historical data for Karachi using Open-Meteo (for training history)
and Open-Meteo (for live dashboard updates).
PUSHES DATA TO HOPSWORKS FEATURE STORE (Serverless).
"""

import argparse, os, json, requests
import pandas as pd
import numpy as np
from datetime import datetime
import hopsworks

# KARACHI COORDINATES
LAT = 24.8607
LON = 67.0011

def fetch_historical_history(start_date, end_date):
    """
    Fetches hourly data from Open-Meteo for training.
    """
    print(f"Fetching historical data from {start_date} to {end_date}...")
    
    # 1. Fetch Weather History
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto"
    }
    
    # 2. Fetch Air Quality History
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm10,pm2_5,nitrogen_dioxide,ozone",
        "timezone": "auto"
    }

    try:
        r_w = requests.get(weather_url, params=weather_params)
        r_w.raise_for_status()
        w_data = r_w.json()
        
        r_aq = requests.get(aq_url, params=aq_params)
        r_aq.raise_for_status()
        aq_data = r_aq.json()

        df_w = pd.DataFrame({
            'timestamp': pd.to_datetime(w_data['hourly']['time']),
            'temp': w_data['hourly']['temperature_2m'],
            'humidity': w_data['hourly']['relative_humidity_2m'],
            'wind_speed': w_data['hourly']['wind_speed_10m']
        })

        df_aq = pd.DataFrame({
            'timestamp': pd.to_datetime(aq_data['hourly']['time']),
            'pm25': aq_data['hourly']['pm2_5'],
            'pm10': aq_data['hourly']['pm10'],
            'no2': aq_data['hourly']['nitrogen_dioxide'],
            'o3': aq_data['hourly']['ozone']
        })

        df = pd.merge(df_w, df_aq, on='timestamp', how='inner')
        return df

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def fetch_live_open_meteo():
    """
    Fetches the CURRENT reading from Open-Meteo (Unified Weather + Air Quality).
    Replaces AQICN to ensure data consistency with training.
    """
    # 1. Weather API
    w_url = "https://api.open-meteo.com/v1/forecast"
    w_params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto"
    }
    
    # 2. Air Quality API
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "pm10,pm2_5,nitrogen_dioxide,ozone",
        "timezone": "auto"
    }

    try:
        r_w = requests.get(w_url, params=w_params)
        w_data = r_w.json()
        
        r_aq = requests.get(aq_url, params=aq_params)
        aq_data = r_aq.json()
        
        if 'current' not in w_data or 'current' not in aq_data:
            print("⚠️ Open-Meteo returned incomplete data.")
            return None

        w_curr = w_data['current']
        aq_curr = aq_data['current']

        return {
            'timestamp': datetime.now(),
            # Weather
            'temp': w_curr.get('temperature_2m'),
            'humidity': w_curr.get('relative_humidity_2m'),
            'wind_speed': w_curr.get('wind_speed_10m'),
            # Pollution
            'pm25': aq_curr.get('pm2_5'),
            'pm10': aq_curr.get('pm10'),
            'no2': aq_curr.get('nitrogen_dioxide'),
            'o3': aq_curr.get('ozone')
        }

    except Exception as e:
        print(f"Open-Meteo Live Fetch Error: {e}")
        return None

def compute_aqi_from_pm25(pm25):
    if pd.isna(pm25): return 0
    if pm25 <= 12: return 50 * (pm25 / 12)
    elif pm25 <= 35.4: return 50 + (50 * (pm25 - 12) / (35.4 - 12))
    elif pm25 <= 55.4: return 100 + (50 * (pm25 - 35.4) / (55.4 - 35.4))
    else: return 150 + (100 * (pm25 - 55.4) / 100.0)

def add_derived_features(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Time features
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    
    # Calculate AQI target
    df['aqi_pm25'] = df['pm25'].apply(compute_aqi_from_pm25)

    # Derived Lag/Rolling features
    df['pm25_change'] = df['pm25'].diff().fillna(0)
    df['aqi_change_rate'] = df['aqi_pm25'].pct_change().fillna(0)
    df['pm25_3h_mean'] = df['pm25'].rolling(3, min_periods=1).mean()
    df['pm25_24h_mean'] = df['pm25'].rolling(24, min_periods=1).mean()
    
    return df.dropna()

def push_to_hopsworks(df):
    """
    Connects to Hopsworks and inserts the dataframe into the Feature Store.
    """
    if df.empty:
        print("DataFrame is empty. Skipping upload.")
        return

    # 1. Login
    print("Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # 2. Get or Create Feature Group
    print("Getting/Creating Feature Group 'karachi_aqi'...")
    aqi_fg = fs.get_or_create_feature_group(
        name="karachi_aqi",
        version=1,
        primary_key=["timestamp"], 
        event_time="timestamp",
        description="Hourly Air Quality & Weather Data for Karachi"
    )

    # 3. Insert Data
    print(f"Uploading {len(df)} rows to Feature Store...")
    # 'wait_for_job=True' is CRITICAL for preventing connection timeouts on local wifi
    aqi_fg.insert(df, write_options={"wait_for_job" : True})
    print("✅ Upload successful!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='Karachi')
    parser.add_argument('--start', help='YYYY-MM-DD')
    parser.add_argument('--end', help='YYYY-MM-DD')
    parser.add_argument('--api_keys', default='{}') # Kept for backward compat, though unused now
    args = parser.parse_args()
    
    df = pd.DataFrame()

    # 1. FETCH MODE: If dates provided, fetch HISTORY for training
    if args.start and args.end:
        df = fetch_historical_history(args.start, args.end)
        if not df.empty:
            df = add_derived_features(df)
            print("History fetched successfully.")
        else:
            print("Failed to fetch historical data.")

    # 2. LIVE MODE: Fetch CURRENT reading from Open-Meteo
    else:
        print("Fetching live data from Open-Meteo...")
        live_data = fetch_live_open_meteo()
        if live_data:
            df = pd.DataFrame([live_data])
            df = add_derived_features(df)
            print(f"Live data fetched: PM2.5={df['pm25'].values[0]} | Temp={df['temp'].values[0]}°C")
        else:
            print("Failed to fetch live data.")

    # 3. PUSH TO HOPSWORKS
    if not df.empty:
        # --- FIX: Surgical Type Alignment (Crucial for Hopsworks Schema) ---
        
        # 1. Decimals (Double/Float64)
        float_cols = ['temp', 'pm25', 'pm10', 'no2', 'o3', 'wind_speed', 
                      'pm25_change', 'aqi_change_rate', 'pm25_3h_mean', 'pm25_24h_mean']
        
        # 2. 64-bit Integers (BigInt) - Humidity
        bigint_cols = ['humidity']

        # 3. 32-bit Integers (Int) - Time features
        int_cols = ['year', 'month', 'day', 'hour', 'weekday']

        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
        
        for col in bigint_cols:
            if col in df.columns:
                df[col] = df[col].round().astype(np.int64)

        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].round().astype(np.int32)
                
        push_to_hopsworks(df)
    else:
        print("No data to upload.")

if __name__ == '__main__':
    main()