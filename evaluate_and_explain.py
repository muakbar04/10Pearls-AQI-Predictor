import os, joblib, json
import pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

def plot_trajectories(y_true, y_pred, indices, out_path):
    """Plots actual vs predicted lines for specific test samples."""
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        if idx >= len(y_true): continue
        plt.subplot(1, 3, i+1)
        plt.plot(y_true[idx], label='Actual', marker='.', alpha=0.7)
        plt.plot(y_pred[idx], label='Predicted', linestyle='--', linewidth=2)
        plt.title(f"Test Sample #{idx}")
        plt.xlabel("Hours Ahead")
        plt.ylabel("AQI")
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved trajectory plots to {out_path}")

def main():
    # 1. Load Data
    df = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['timestamp']).dropna().reset_index(drop=True)
    
    # Load Meta to get feature config
    with open(os.path.join(MODEL_DIR, 'model_meta.json'), 'r') as f:
        meta = json.load(f)
    
    feature_cols = meta['feature_cols']
    horizon = meta['horizon']
    
    # Re-create sequences
    X = []
    y = []
    for i in range(len(df)-horizon):
        X.append(df.iloc[i][feature_cols].values.astype(float))
        y.append(df.iloc[i+1 : i+1+horizon]['aqi_pm25'].values.astype(float))
    X = np.array(X); y = np.array(y)
    
    # 2. Load Model & Scaler
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    X_scaled = scaler.transform(X)
    
    if meta['best_model'] == 'mlp':
        from tensorflow import keras
        model = keras.models.load_model(os.path.join(MODEL_DIR, 'best_model.keras'))
        y_pred = model.predict(X_scaled, verbose=0)
    else:
        model = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
        y_pred = model.predict(X_scaled)
        
    # 3. Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    print(f"Evaluation on full dataset -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    # 4. Visualization (Plot 3 random samples)
    indices = [0, len(y)//2, len(y)-10] # Beginning, Middle, End
    plot_trajectories(y, y_pred, indices, os.path.join(MODEL_DIR, 'forecast_trajectories.png'))
    
    # Save Report
    report = {
        "overall_rmse": float(rmse),
        "overall_mae": float(mae),
        "feature_cols": feature_cols
    }
    with open(os.path.join(MODEL_DIR, 'explain_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    main()