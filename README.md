# Football Injury Prediction Model

ML injury risk prediction pipeline using **feature engineering**, **XGBoost**, and **SHAP explainability**.

## Features

- **Feature Engineering**
  - **ACWR** (Acute:Chronic Workload Ratio): 7-day acute / 28-day chronic load
  - **Monotony Index**: mean weekly load / std weekly load
  - **Joint angles**: proxy from 2D track/pose data
  - **Fatigue**: exponential decay cumulative load
  - **Sliding window**: rolling mean (3d, 7d, 14d)

- **Model**: XGBoost (tabular biometric + wellness → binary injury risk)

- **Explainability**: SHAP top-3 risk factors per flagged player (why each was flagged)

## Datasets

1. [Multimodal Sports Injury Dataset](https://www.kaggle.com/datasets/anjalibhegam/multimodal-sports-injury-dataset)
2. [NFL Playing Surface Analytics](https://www.kaggle.com/competitions/nfl-playing-surface-analytics/data)

## Setup

```bash
# Create environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### Kaggle API (for downloads)

1. Create a Kaggle account and API key
2. Place `kaggle.json` in `~/.kaggle/` (or `C:\Users\<user>\.kaggle\` on Windows)

## Usage

```bash
# Run on demo synthetic data (no download)
python run_pipeline.py

# Download datasets and run
python run_pipeline.py --download

# Demo mode (forces synthetic data)
python run_pipeline.py --demo
```

## Fullstack Integration (Frontend + Backend)

This repo now includes a backend API (FastAPI) that wraps the ML prediction pipeline and serves squad risk data to the React frontend.

### 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2) Start backend API

```bash
uvicorn backend.main:app --reload --port 8000
```

Or from the project root:

```bash
npm run start:backend
```

API endpoints:
- `GET /api/health`
- `GET /api/squad?count=6`
- `POST /api/predict/squad`

### 3) Start frontend

```bash
npm install
npm start
```

Frontend notes:
- `npm start` proxies `/api/*` requests to `http://localhost:8000` in development
- `REACT_APP_API_BASE_URL` is optional if the backend runs on a different host/port

## Output

- Classification report and ROC-AUC
- SHAP top-3 risk factors for each flagged player
- Confusion matrix

## Project Structure

```
AI Injury Prediction/
├── data_loaders.py      # Loaders for NFL & Multimodal datasets
├── feature_engineering.py # ACWR, monotony, fatigue, joint angles, sliding window
├── model.py             # XGBoost + SHAP pipeline
├── run_pipeline.py      # Entry script
├── requirements.txt
└── README.md
```
