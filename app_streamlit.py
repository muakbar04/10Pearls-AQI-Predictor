import os
import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import pytz
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hopsworks

st.set_page_config(layout="wide", page_title="Karachi AQI Forecast (Serverless)")

# 1. CONNECT TO HOPSWORKS (Cached)
@st.cache_resource
def get_hopsworks_resources():
    """Connects to Hopsworks and returns Feature Store & Model Registry"""
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        return fs, mr
    except Exception as e:
        st.error(f"Could not connect to Hopsworks: {e}")
        return None, None

# 2. FETCH DATA (Live from Cloud)
@st.cache_data(ttl=300)
def load_batch_data(_fs):
    """Fetches input features from Feature Store"""
    try:
        # Get the Feature Group
        fg = _fs.get_feature_group(name="karachi_aqi", version=1)
        
        # Select all columns and read into Pandas
        df = fg.select_all().read()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data from Hopsworks: {e}")
        return None

# 3. LOAD MODEL
@st.cache_resource
def load_serverless_model(_mr):
    """Downloads the BEST model and robustly finds artifacts"""
    try:
        # Get best model metadata
        model_meta = _mr.get_best_model("karachi_aqi_xgboost", metric="rmse", direction="min")
        
        # Download artifacts
        download_path = model_meta.download()
        
        # Hopsworks might nest files or flatten them. We search recursively.
        model_file = None
        scaler_file = None
        meta_file = None
        
        for root, dirs, files in os.walk(download_path):
            for file in files:
                if file == "best_model.pkl":
                    model_file = os.path.join(root, file)
                elif file == "scaler.pkl":
                    scaler_file = os.path.join(root, file)
                elif file == "model_meta.json":
                    meta_file = os.path.join(root, file)
        
        if not (model_file and scaler_file and meta_file):
            st.error(f"Missing artifacts in download! Found: {os.listdir(download_path)}")
            return None, None, None

        # Load artifacts
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            
        return model, scaler, meta

    except Exception as e:
        st.error(f"Error fetching model from Registry: {e}")
        return None, None, None

# 4. PREPROCESSING HELPER
def add_physics_features(df):
    df = df.copy()
    # Cyclical Time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # Dynamics
    df['pm25_6h_std'] = df['pm25'].rolling(window=6).std()
    df['pm25_trend_12h'] = df['pm25'] - df['pm25'].shift(12)
    
    return df.dropna().reset_index(drop=True)

# UI HELPER
def get_aqi_color(aqi):
    if aqi <= 50: return "green", "Good"
    if aqi <= 100: return "#FDD835", "Moderate"
    if aqi <= 150: return "orange", "Unhealthy for Sensitive Groups"
    if aqi <= 200: return "red", "Unhealthy"
    if aqi <= 300: return "purple", "Very Unhealthy"
    return "#7e0023", "Hazardous"

# MAIN APP LOGIC
st.title("🌬️ Karachi AQI Forecast")
st.markdown("Powered by **Hopsworks Feature Store** & **XGBoost**")
st.markdown("---")

# Load Resources
with st.spinner("Connecting to Serverless Cloud..."):
    fs, mr = get_hopsworks_resources()

if fs and mr:
    df_raw = load_batch_data(fs)
    model, scaler, meta = load_serverless_model(mr)

    if df_raw is not None and model is not None:
        # Preprocess Data (Add Physics Features)
        df = add_physics_features(df_raw)

        # STEP 1: LOGIC (Filter in UTC)
        now_utc = datetime.now(pytz.utc)

        # Ensure timestamp is UTC-aware
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

        # Filter: Keep only past/present rows
        past_df = df[df['timestamp'] <= now_utc]

        # STEP 2: DISPLAY (Convert to Karachi Time)
        pk_tz = pytz.timezone('Asia/Karachi')
        
        # Convert the main dataframe
        df['timestamp'] = df['timestamp'].dt.tz_convert(pk_tz)
        
        # Convert the filtered dataframe
        if not past_df.empty:
            past_df = past_df.copy()
            past_df['timestamp'] = past_df['timestamp'].dt.tz_convert(pk_tz)
            
            latest = past_df.iloc[-1]
            prev = past_df.iloc[-2] if len(past_df) > 1 else latest
        else:
            # Fallback
            latest = df.iloc[-1]
            prev = df.iloc[-2]

        # TOP ROW: METRICS
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current PM2.5", f"{latest['pm25']:.1f}", f"{latest['pm25'] - prev['pm25']:.1f}")
        with col2:
            aqi_val = latest['aqi_pm25']
            color, label = get_aqi_color(aqi_val)
            st.markdown(f"**Current AQI**")
            st.markdown(f"<h2 style='color:{color};'>{int(aqi_val)} - {label}</h2>", unsafe_allow_html=True)
        with col3:
            st.metric("Temp", f"{latest['temp']}°C")
        with col4:
            st.metric("Wind", f"{latest['wind_speed']} km/h")

        # CHARTS
        tab1, tab2 = st.tabs(["📈 History", "🤖 72h Forecast"])

        with tab1:
            # Plot last 7 days (now in Karachi Time) using FILTERED data
            fig = px.line(past_df.tail(168), x='timestamp', y=['pm25', 'aqi_pm25'], 
                          title="Last 7 Days History (Karachi Time)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # INFERENCE LOGIC 
            input_width = meta.get('input_width', 24)
            feature_cols = meta.get('feature_cols', [])
            
            if len(past_df) < input_width:
                st.error(f"Need at least {input_width} hours of data history.")
            else:
                # 1. Prepare Input Vector
                input_df = past_df.iloc[-input_width:]
                
                try:
                    X_input = input_df[feature_cols].values.flatten().reshape(1, -1)
                    
                    # 2. Scale & Predict
                    X_scaled = scaler.transform(X_input)
                    pred_vector = model.predict(X_scaled)
                    
                    if pred_vector.ndim > 1: pred_vector = pred_vector[0]
                    
                    # 3. Create Future Timeline
                    future_dates = [latest['timestamp'] + timedelta(hours=i+1) for i in range(len(pred_vector))]
                    
                    # 4. Plot
                    fig_fc = go.Figure()
                    
                    past = past_df.tail(48)
                    fig_fc.add_trace(go.Scatter(x=past['timestamp'], y=past['aqi_pm25'], name="Past", line=dict(color='cyan')))
                    
                    # Future Forecast
                    x_fc = [past['timestamp'].iloc[-1]] + future_dates
                    y_fc = [past['aqi_pm25'].iloc[-1]] + list(pred_vector)
                    
                    fig_fc.add_trace(go.Scatter(x=x_fc, y=y_fc, name="Forecast", line=dict(dash='dash', color='orange', width=3)))
                    
                    fig_fc.update_layout(title="AI Forecast Trajectory (Karachi Time)", template="plotly_dark", hovermode="x unified")
                    st.plotly_chart(fig_fc, use_container_width=True)
                    
                    peak_aqi = max(pred_vector)
                    peak_date = future_dates[np.argmax(pred_vector)]
                    if peak_aqi >= 150:
                        st.error(f"🚨 **ALERT:** Hazardous AQI Peak of **{int(peak_aqi)}** forecasted on {peak_date.strftime('%A at %I:%M %p')}! Please take precautions.")
                    elif peak_aqi >= 100:
                        st.warning(f"⚠️ **Warning:** AQI expected to reach **{int(peak_aqi)}** (Unhealthy for Sensitive Groups) on {peak_date.strftime('%A at %I:%M %p')}.")
                    else:
                        st.success(f"✅ **Forecast Peak:** {int(peak_aqi)} AQI on {peak_date.strftime('%A at %I:%M %p')}. Air quality looks good!")
                        
                    st.markdown("---")
                    st.markdown("### 📅 Daily Average Forecast")
                    
                    # 1. Create a temp dataframe to easily group the 72 hours by calendar day
                    df_future = pd.DataFrame({
                        'Date': [d.date() for d in future_dates],
                        'AQI': pred_vector
                    })
                    
                    # 2. Group by the Date and calculate the mean AQI for each day
                    daily_avg = df_future.groupby('Date')['AQI'].mean().reset_index()
                    
                    # 3. Display them nicely in side-by-side columns
                    avg_cols = st.columns(len(daily_avg))
                    for i, row in daily_avg.iterrows():
                        with avg_cols[i]:
                            day_str = row['Date'].strftime('%a, %b %d') # e.g., "Mon, Oct 28"
                            avg_val = int(row['AQI'])
                            color, label = get_aqi_color(avg_val) # Reuse your color helper!
                            
                            st.markdown(f"**{day_str}**")
                            st.markdown(f"<h3 style='color:{color};'>{avg_val}</h3>", unsafe_allow_html=True)
                            st.caption(label)
                    
                except KeyError as e:
                    st.error(f"Feature Mismatch! {e}")
                    st.write("Available columns:", df.columns.tolist())

        # Sidebar Meta
        st.sidebar.success(f"Model Loaded: {meta.get('last_trained')}")
        if 'metrics' in meta:
             st.sidebar.info(f"Training RMSE: {meta['metrics']['rmse']:.2f}")

        # View Raw Data
        with st.expander("View Feature Store Data"):
            st.dataframe(df.tail(50), use_container_width=True)

    else:
        st.warning("Waiting for data/model to load...")

# REFRESH BUTTON
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.toast("Fetching new data from source...")
    import subprocess
    subprocess.run("python fetch_features.py", shell=True) 
    st.cache_data.clear()
    st.rerun()