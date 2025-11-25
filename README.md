# EMIPredict AI — Intelligent Financial Risk Assessment Platform

This repository is a scaffold for the EMIPredict AI project: a platform for EMI eligibility classification and maximum EMI regression with MLflow tracking and a Streamlit UI.

What's included (minimal scaffold):

- `src/data/generate_sample.py` — create a small synthetic dataset for local development (1000 rows)
- `src/preprocessing.py` — data loading, cleaning, splitting utilities
- `src/feature_engineering.py` — feature derivation and encoding
- `src/train.py` — trains classification and regression models, logs runs to MLflow
- `src/app.py` — Streamlit app skeleton for predictions and simple CRUD on CSV dataset
- `requirements.txt` — Python dependencies

Quickstart (Windows PowerShell):

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Generate sample data (1000 rows):

```powershell
python src/data/generate_sample.py --out data/sample_emi.csv --n 1000
```

3. Train models (this will create an MLflow `mlruns/` folder):

```powershell
python src/train.py --data data/sample_emi.csv --output models --mlflow_uri ./mlruns
```

4. Run the Streamlit app:

```powershell
streamlit run src/app.py
```

Notes:
- This is a development scaffold. Replace `data/sample_emi.csv` with the full dataset (400k) when available and adjust compute resources.
- MLflow is configured for local file-based tracking (default `mlruns/`). For production, point MLflow to a tracking server and artifact store.

Next steps:
- Extend models (hyperparameter tuning, cross-validation)
- Add full EDA notebooks and visualization pages
- Implement a CI/CD pipeline for Streamlit Cloud deployment

License: Add your license here.
