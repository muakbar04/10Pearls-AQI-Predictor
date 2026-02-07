"""
train_model.py
SERVERLESS EDITION (Fixed):
 - Uses ephemeral /tmp storage.
 - BUNDLES all artifacts (model, scaler, meta) into one upload.
 - Auto-increments model version.
"""

import os
import json
import argparse
import tempfile
import joblib
import numpy as np
import pandas as pd
import hopsworks

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor 
from xgboost import XGBRegressor
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# --- FEATURE ENGINEERING ---
def add_physics_features(df):
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['pm25_6h_std'] = df['pm25'].rolling(window=6).std()
    df['pm25_trend_12h'] = df['pm25'] - df['pm25'].shift(12)
    return df.dropna().reset_index(drop=True)

def create_sequences(df, input_width=24, label_width=72, target_col='aqi_pm25'):
    base_features = ['temp','humidity','wind_speed','pm25','pm10','no2','o3',
                     'weekday','pm25_change','aqi_change_rate',
                     'pm25_3h_mean','pm25_24h_mean',
                     'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                     'pm25_6h_std', 'pm25_trend_12h']
    
    feature_cols = [c for c in base_features if c in df.columns]
    
    X, y = [], []
    total_len = len(df)
    
    for i in range(total_len - input_width - label_width):
        window = df.iloc[i : i+input_width][feature_cols].values
        X.append(window.flatten())
        target = df.iloc[i+input_width : i+input_width+label_width][target_col].values
        y.append(target)
        
    return np.array(X), np.array(y), feature_cols

def evaluate_model(y_true, y_pred, name):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    print(f"\n[{name}] RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    return {"rmse": rmse, "mae": mae}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=72)
    parser.add_argument('--input_width', type=int, default=24)
    args = parser.parse_args()

    # 1. CONNECT & FETCH
    print("Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    try:
        aqi_fg = fs.get_feature_group(name="karachi_aqi", version=1)
        print("Fetching data from Feature Store...")
        df = aqi_fg.select_all().read()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"❌ Error fetching from Hopsworks: {e}")
        return

    # 2. PREPARE DATA
    print("Engineering features...")
    df = add_physics_features(df)
    X, y, feature_cols = create_sequences(df, args.input_width, args.horizon)
    
    if len(X) == 0: 
        print("Insufficient data.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 3. TRAIN
    print("\nTraining XGBoost...")
    xgb_base = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror',
        n_jobs=-1, random_state=42
    )
    xgb = MultiOutputRegressor(xgb_base)
    xgb.fit(X_train_s, y_train)
    metrics = evaluate_model(y_test, xgb.predict(X_test_s), "XGBoost")

    # 4. SERVERLESS UPLOAD (The Fix)
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"\nCreated ephemeral workspace: {tmp_dir}")
        
        # A. Save Artifacts to Temp Directory
        model_path = os.path.join(tmp_dir, "best_model.pkl") 
        scaler_path = os.path.join(tmp_dir, "scaler.pkl")
        meta_path = os.path.join(tmp_dir, "model_meta.json")

        print("Serializing artifacts...")
        joblib.dump(xgb, model_path)
        joblib.dump(scaler, scaler_path)
        
        meta = {
            "metrics": metrics,
            "feature_cols": feature_cols,
            "input_width": args.input_width,
            "horizon": args.horizon,
            "last_trained": str(pd.Timestamp.now())
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # B. Register in Hopsworks
        print("Registering in Model Registry...")
        mr = project.get_model_registry()
        
        input_schema = Schema(X_train_s)
        output_schema = Schema(y_train)
        model_schema = ModelSchema(input_schema, output_schema)

        aqi_model = mr.python.create_model(
            name="karachi_aqi_xgboost", 
            metrics=metrics,
            model_schema=model_schema,
            input_example=X_train_s[:1],
            description="Serverless XGBoost 72h Forecast"
        )
        
        # THE KEY FIX: Pass the DIRECTORY path, not individual files
        # This uploads everything inside tmp_dir as a single Model Version
        print(f"Uploading artifacts from {tmp_dir}...")
        aqi_model.save(tmp_dir)
        
        print("✅ Upload Complete. Model Version Auto-Incremented.")

if __name__ == "__main__":
    main()