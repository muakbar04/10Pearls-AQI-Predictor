# 🌬️ Pearls AQI Predictor: Serverless Air Quality Forecasting

**Pearls AQI Predictor** is a 100% serverless, end-to-end machine learning pipeline that forecasts the PM2.5-based Air Quality Index (AQI) for Karachi, Pakistan, up to 72 hours in advance.

This project automates the entire ML lifecycle—from real-time data ingestion and dynamic feature engineering to daily model retraining and interactive visualization—without the need to provision or manage dedicated servers.

### 🚀 Live Demo

**Access the interactive dashboard here:** [Pearls AQI Predictor]()

---

## 🧠 Key Features

* **Automated Data Pipelines:** Continuous hourly fetching of historical and live weather/pollutant data via the Open-Meteo API.
* **Serverless MLOps:** Utilizes Hopsworks Feature Store and Model Registry as the single source of truth for features and model artifacts.
* **High-Performance Forecasting:** Powered by a heavily tuned XGBoost model (wrapped in a `MultiOutputRegressor`) handling non-linear meteorological relationships and sudden pollution spikes (Achieved an RMSE of 26.89).
* **Explainable AI (XAI):** Real-time integration of SHAP (SHapley Additive exPlanations) to provide transparency into which features drive the immediate next hour's prediction.
* **Hazard Alerting System:** Automatically warns users if forecasted AQI crosses into "Unhealthy" or "Hazardous" thresholds.

---

## 🏗️ System Architecture & Tech Stack

The system operates on a decoupled, serverless architecture optimized for scalability and low maintenance.

* **Data Source:** Open-Meteo API (Weather & Air Quality)
* **Data Engineering:** Pandas, Numpy
* **Feature Store & Model Registry:** Hopsworks
* **Machine Learning:** Scikit-Learn, XGBoost, SHAP
* **CI/CD Orchestration:** GitHub Actions
* **Frontend Deployment:** Streamlit Cloud, Plotly Express

---

## 🛠️ Repository Structure

* `fetch_features.py`: The data engineering pipeline. Fetches live/historical data, computes cyclical time encodings and rolling averages, and pushes updates to the Hopsworks Feature Store.
* `train_model.py`: The training pipeline. Pulls the latest features, generates 24-hour lookback sequences, trains the XGBoost model for a 72-hour horizon, and registers the artifacts.
* `app_streamlit.py`: The frontend dashboard. Pulls live data and the latest model artifacts to generate interactive forecasts and SHAP explainability charts.
* `.github/workflows/`: Contains the CI/CD YAML files:
* `hourly_features.yml`: Runs `fetch_features.py` every hour.
* `daily_training.yml`: Runs `train_model.py` every night at midnight UTC.



---

## 💡 Engineering Challenges Overcome

Building this production-ready pipeline involved solving several critical system-design and data-science challenges:

1. **Timezone Synchronization & Data Leakage:** Reconciled UTC-based cloud data with `Asia/Karachi` local timeframes by strictly enforcing UTC localization during Hopsworks ingestion and selectively applying visual conversions at the Streamlit frontend.
2. **Serverless Cold Starts:** Mitigated initial load latency on Streamlit Cloud by aggressively caching the Hopsworks client connection, model artifacts, and batch data using `@st.cache_resource` and `@st.cache_data`.
3. **Dependency Hell Resolution:** Avoided silent C-extension clashes and build failures in the cloud by heavily pruning deep-learning libraries (dropping TF/PyTorch in favor of XGBoost) and utilizing strict version pinning.
4. **Auto-Regressive Responsiveness:** Extensively validated lag features to ensure the 72-hour forecast trajectory dynamically corrected itself during abrupt, real-world PM2.5 spikes rather than defaulting to flat historical averages.

---

## 💻 Quickstart: Running Locally

To run this project on your local machine:

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

```

**2. Install dependencies**

```bash
pip install -r requirements.txt

```

**3. Set up Hopsworks API Key**
You will need a free Hopsworks account. Create an API key and set it as an environment variable:

* **Windows:** `set HOPSWORKS_API_KEY="your_api_key_here"`
* **Mac/Linux:** `export HOPSWORKS_API_KEY="your_api_key_here"`

**4. Run the application**

```bash
streamlit run app_streamlit.py

```
