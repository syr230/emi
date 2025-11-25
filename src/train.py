"""Training script: trains simple classification and regression models and logs to MLflow.

This is a minimal example — expand for hyperparameter tuning and CV.
"""
import argparse
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import load_data, basic_clean, split_data
from feature_engineering import add_financial_ratios, simple_encode


def train_and_log_classification(X_train, y_train, X_val, y_val, run_name, model_dir):
    mlflow.set_experiment('emipredict_classification')
    with mlflow.start_run(run_name=run_name):
        # Build pipeline (scaler + logistic) so the saved model accepts raw DataFrame inputs
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        pipeline = Pipeline([('scaler', scaler), ('clf', clf)])
        pipeline.fit(X_train, y_train)

        # predictions on validation
        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average='weighted', zero_division=0)
        rec = recall_score(y_val, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_val, preds, average='weighted', zero_division=0)
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall', rec)
        mlflow.log_metric('f1', f1)

        # save pipeline and log with signature & input example
        (Path(model_dir)).mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, Path(model_dir) / 'logistic_pipeline.joblib')
        try:
            input_example = X_train.head(1)
            preds_example = pipeline.predict(input_example)
            signature = infer_signature(input_example, preds_example)
            # Log with a name that matches the saved joblib artifact
            mlflow.sklearn.log_model(pipeline, 'logistic_pipeline', signature=signature, input_example=input_example)
        except Exception:
            mlflow.sklearn.log_model(pipeline, 'logistic_pipeline')


def train_and_log_regression(X_train, y_train, X_val, y_val, run_name, model_dir):
    mlflow.set_experiment('emipredict_regression')
    with mlflow.start_run(run_name=run_name):
        reg = RandomForestRegressor(n_estimators=50, random_state=42)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_val)
        # compute RMSE manually for compatibility across sklearn versions
        rmse = mean_squared_error(y_val, preds) ** 0.5
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2)
        # log model with signature
        try:
            input_example = X_train.head(1)
            preds_example = reg.predict(input_example)
            signature = infer_signature(input_example, preds_example)
            mlflow.sklearn.log_model(reg, 'rf_regressor', signature=signature, input_example=input_example)
        except Exception:
            mlflow.sklearn.log_model(reg, 'rf_regressor')
        joblib.dump(reg, Path(model_dir) / 'rf_regressor.joblib')


def train_xgboost_classification(X_train, y_train, X_val, y_val, run_name, model_dir):
    mlflow.set_experiment('emipredict_classification')
    with mlflow.start_run(run_name=run_name):
        # Encode string class labels to integers for XGBoost
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)

        # small randomized search for XGBoost classifier
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0]
        }
        rnd = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=6, scoring='f1_weighted', cv=3, n_jobs=-1, random_state=42)
        rnd.fit(X_train, y_train_enc)
        best = rnd.best_estimator_
        preds = best.predict(X_val)
        # preds will be integer labels; convert back to original if needed
        try:
            preds_labels = le.inverse_transform(preds)
        except Exception:
            preds_labels = preds

        acc = accuracy_score(y_val, preds_labels)
        prec = precision_score(y_val, preds_labels, average='weighted', zero_division=0)
        rec = recall_score(y_val, preds_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_val, preds_labels, average='weighted', zero_division=0)
        mlflow.log_params(rnd.best_params_)
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('precision', prec)
        mlflow.log_metric('recall', rec)
        mlflow.log_metric('f1', f1)
        # save model and label encoder
        joblib.dump(le, Path(model_dir) / 'label_encoder.joblib')
        joblib.dump(best, Path(model_dir) / 'xgb_classifier.joblib')
        # log model with signature
        try:
            input_example = X_train.head(1)
            preds_example = best.predict(input_example)
            # convert integer preds back to label strings for signature
            try:
                preds_example_labels = le.inverse_transform(preds_example)
            except Exception:
                preds_example_labels = preds_example
            signature = infer_signature(input_example, preds_example_labels)
            mlflow.sklearn.log_model(best, 'xgb_classifier', signature=signature, input_example=input_example)
        except Exception:
            mlflow.sklearn.log_model(best, 'xgb_classifier')
        mlflow.log_artifact(str(Path(model_dir) / 'label_encoder.joblib'), artifact_path='preprocessing')


def train_xgboost_regression(X_train, y_train, X_val, y_val, run_name, model_dir):
    mlflow.set_experiment('emipredict_regression')
    with mlflow.start_run(run_name=run_name):
        xgb = XGBRegressor(objective='reg:squarederror', verbosity=0)
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0]
        }
        rnd = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=6, scoring='neg_mean_squared_error', cv=3, n_jobs=-1, random_state=42)
        rnd.fit(X_train, y_train)
        best = rnd.best_estimator_
        preds = best.predict(X_val)
        rmse = mean_squared_error(y_val, preds) ** 0.5
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        mlflow.log_params(rnd.best_params_)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2)
        joblib.dump(best, Path(model_dir) / 'xgb_regressor.joblib')
        try:
            input_example = X_train.head(1)
            preds_example = best.predict(input_example)
            signature = infer_signature(input_example, preds_example)
            mlflow.sklearn.log_model(best, 'xgb_regressor', signature=signature, input_example=input_example)
        except Exception:
            mlflow.sklearn.log_model(best, 'xgb_regressor')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='models')
    parser.add_argument('--mlflow_uri', default='mlruns')
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    df = basic_clean(df)
    train, val, test = split_data(df)

    # Feature engineering
    train = add_financial_ratios(train)
    val = add_financial_ratios(val)

    # Simple encoding (drop target columns first)
    y_train_clf = train['emi_eligibility']
    y_val_clf = val['emi_eligibility']
    X_train = train.drop(columns=['emi_eligibility', 'max_monthly_emi'])
    X_val = val.drop(columns=['emi_eligibility', 'max_monthly_emi'])

    # Fit encoder on training set and transform validation using the same encoder
    X_train_enc, enc = simple_encode(X_train)
    X_val_enc, _ = simple_encode(X_val, categorical_cols=X_train.select_dtypes(include=['object','category']).columns.tolist(), encoder=enc)

    # persist encoder and feature column list so the app can use identical transforms
    try:
        joblib.dump(enc, Path(args.output) / 'onehot_encoder.joblib')
        joblib.dump(list(X_train_enc.columns), Path(args.output) / 'feature_columns.joblib')
        # also log as artifacts
        mlflow.log_artifact(str(Path(args.output) / 'onehot_encoder.joblib'), artifact_path='preprocessing')
        mlflow.log_artifact(str(Path(args.output) / 'feature_columns.joblib'), artifact_path='preprocessing')
    except Exception:
        pass

    # Ensure column alignment
    common_cols = X_train_enc.columns.intersection(X_val_enc.columns)
    X_train_enc = X_train_enc[common_cols]
    X_val_enc = X_val_enc[common_cols]

    # ensure no active MLflow run before starting
    try:
        mlflow.end_run()
    except Exception:
        pass
    train_and_log_classification(X_train_enc, y_train_clf, X_val_enc, y_val_clf, 'logistic_vs_rf', args.output)
    # ensure any active MLflow runs are closed before starting regression runs
    try:
        mlflow.end_run()
    except Exception:
        pass

    # Regression targets
    y_train_reg = train['max_monthly_emi']
    y_val_reg = val['max_monthly_emi']
    X_train_reg = X_train_enc
    X_val_reg = X_val_enc

    train_and_log_regression(X_train_reg, y_train_reg, X_val_reg, y_val_reg, 'rf_regression', args.output)

    # ensure closed before XGBoost runs
    try:
        mlflow.end_run()
    except Exception:
        pass

    # XGBoost searches (fast settings)
    train_xgboost_classification(X_train_enc, y_train_clf, X_val_enc, y_val_clf, 'xgb_class_search', args.output)
    try:
        mlflow.end_run()
    except Exception:
        pass
    train_xgboost_regression(X_train_reg, y_train_reg, X_val_reg, y_val_reg, 'xgb_reg_search', args.output)

    print('Training complete. Models saved to', args.output)


if __name__ == '__main__':
    main()
