# EMIPredict — EMI eligibility & max-EMI estimator

EMIPredict is an end-to-end, reproducible ML scaffold for predicting EMI loan eligibility (classification) and estimating a customer's maximum monthly EMI (regression). It includes a synthetic data generator, preprocessing and feature-engineering modules, model training with MLflow experiment tracking, and a Streamlit web UI that loads the exact preprocessing artifacts used during training so predictions are consistent.

This repository is intended as a development scaffold you can scale to your full dataset, add production-grade MLflow tracking, and deploy the Streamlit app.

Key files

- `src/data/generate_sample.py` — synthetic dataset generator for quick experiments
- `src/preprocessing.py` — data loading, cleaning, and split utilities
- `src/feature_engineering.py` — derived financial features and encoding helpers
- `src/train.py` — training scripts, MLflow logging, and artifact persistence
- `src/app.py` — Streamlit app that loads saved artifacts and serves predictions
- `requirements.txt` — Python dependencies

Features

- Synthetic dataset generator for local experimentation
- Derived financial ratio features (debt-to-income, affordability, etc.)
- Classification (EMI eligibility) and regression (max monthly EMI) models
- Baselines (Logistic Regression, RandomForest) and XGBoost with quick randomized search
- MLflow experiment tracking, model signatures, and saved preprocessing artifacts
- Streamlit app for interactive predictions using the same preprocessing pipeline

Quickstart (Windows PowerShell)

1. Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. Generate a small sample dataset (1000 rows)

```powershell
python src/data/generate_sample.py --output data/sample_emi.csv --n 1000
```

3. Train models (artifacts go to `./models` and MLflow writes to `./mlruns` by default)

```powershell
python src/train.py --data data/sample_emi.csv --output models --mlflow_uri ./mlruns
```

4. Run the Streamlit app

```powershell
streamlit run src/app.py
```

Notes

- Replace the sample dataset with your full dataset (400k rows) before production training. Adjust hyperparameters and compute resources accordingly.
- MLflow runs are stored locally in `mlruns/` by default. For collaboration or production use, configure a remote MLflow tracking server and artifact store.

Next steps / roadmap

- Expand hyperparameter search and cross-validation for production models
- Add EDA notebooks and automated model evaluation reports
- Wire MLflow Model Registry or a remote tracking server for team workflows
- Deploy the Streamlit app (Streamlit Cloud, Docker, or a cloud VM) and configure CI/CD

License

Add an open-source license (e.g., MIT) if you want this repo to be publicly reusable. I can add a LICENSE file on request.

If you want, I can commit this README update and push it to the repository now.
