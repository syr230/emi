"""Streamlit app skeleton for EMIPredict AI.

Run with: streamlit run src/app.py
"""
try:
    import streamlit as st
    import pandas as pd
    from pathlib import Path
    import joblib
    from feature_engineering import add_financial_ratios, simple_encode

except Exception:
    # Minimal fallback stub for 'st' so the module can be parsed/executed
    class _StubSidebar:
        def selectbox(self, label, options, index=0, **kwargs):
            return options[index] if options else None

    class _Stub:
        def __init__(self):
            self.sidebar = _StubSidebar()

        def cache_data(self, func=None, **kwargs):
            # Support both @st.cache_data and @st.cache_data(...)
            if callable(func):
                return func
            def decorator(f):
                return f
            return decorator

        def set_page_config(self, *args, **kwargs):
            return None

        def title(self, *args, **kwargs):
            return None

        def header(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def form(self, *args, **kwargs):
            class _FormCtx:
                def __enter__(self_inner):
                    return None
                def __exit__(self_inner, exc_type, exc, tb):
                    return False
            return _FormCtx()

        def number_input(self, label, min_value=None, max_value=None, value=0, **kwargs):
            return value

        def selectbox(self, label, options, index=0, **kwargs):
            return options[index] if options else None

        def form_submit_button(self, *args, **kwargs):
            return False

        def write(self, *args, **kwargs):
            return None

        def dataframe(self, *args, **kwargs):
            return None

        def file_uploader(self, *args, **kwargs):
            return None

        def success(self, *args, **kwargs):
            return None

        def button(self, *args, **kwargs):
            return False

    st = _Stub()

import pandas as pd
from pathlib import Path
import joblib

# Artifact paths and loaders (defined outside try/except so available with stub)
MODEL_DIR = Path('models')
LOGISTIC_PIPELINE_PATH = MODEL_DIR / 'logistic_pipeline.joblib'
XGB_CLASS_PATH = MODEL_DIR / 'xgb_classifier.joblib'
XGB_REG_PATH = MODEL_DIR / 'xgb_regressor.joblib'
RF_REG_PATH = MODEL_DIR / 'rf_regressor.joblib'
ONEHOT_PATH = MODEL_DIR / 'onehot_encoder.joblib'
FEATURE_COLUMNS_PATH = MODEL_DIR / 'feature_columns.joblib'
LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.joblib'


def load_artifact(path):
    try:
        if Path(path).exists():
            return joblib.load(path)
    except Exception:
        return None
    return None

# load artifacts (may be None if not present)
LOGISTIC_PIPELINE = load_artifact(LOGISTIC_PIPELINE_PATH)
XGB_CLASS = load_artifact(XGB_CLASS_PATH)
XGB_REG = load_artifact(XGB_REG_PATH)
RF_REG = load_artifact(RF_REG_PATH)
ONEHOT = load_artifact(ONEHOT_PATH)
FEATURE_COLUMNS = load_artifact(FEATURE_COLUMNS_PATH)
LABEL_ENCODER = load_artifact(LABEL_ENCODER_PATH)

DATA_PATH = Path('data/sample_emi.csv')
MODEL_DIR = Path('models')


def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


@st.cache_data
def load_model(path):
    if Path(path).exists():
        return joblib.load(path)
    return None


def main():
    st.set_page_config(page_title='EMIPredict AI')
    st.title('EMIPredict AI — EMI Eligibility & Amount Prediction')

    menu = st.sidebar.selectbox('Menu', ['Predict', 'Data', 'Model'])

    if menu == 'Predict':
        st.header('Predict EMI Eligibility / Max EMI')
        df = load_data()
        if df.empty:
            st.warning('No dataset found. Run the data generator to create a sample dataset.')
        else:
            st.info('Using sample data for field defaults')
        with st.form('input_form'):
            # Demographics
            age = st.number_input('Age', min_value=18, max_value=100, value=30)
            gender = st.selectbox('Gender', ['Male', 'Female'])
            marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
            education = st.selectbox('Education', ['High School', 'Graduate', 'Post Graduate', 'Professional'])
            # Employment
            monthly_salary = st.number_input('Monthly Salary', min_value=0, value=30000)
            employment_type = st.selectbox('Employment Type', ['Private', 'Government', 'Self-employed'])
            years_of_employment = st.number_input('Years of Employment', min_value=0, value=2)
            company_type = st.selectbox('Company Type', ['Small', 'Medium', 'Large'])
            # Housing & family
            house_type = st.selectbox('House Type', ['Rented', 'Own', 'Family'])
            monthly_rent = st.number_input('Monthly Rent', min_value=0, value=0)
            family_size = st.number_input('Family Size', min_value=1, value=3)
            dependents = st.number_input('Dependents', min_value=0, value=0)
            # Expenses
            school_fees = st.number_input('School Fees', min_value=0, value=0)
            college_fees = st.number_input('College Fees', min_value=0, value=0)
            travel_expenses = st.number_input('Travel Expenses', min_value=0, value=1000)
            groceries_utilities = st.number_input('Groceries & Utilities', min_value=0, value=5000)
            other_monthly_expenses = st.number_input('Other Monthly Expenses', min_value=0, value=1000)
            # Financials
            existing_loans = st.number_input('Existing Loans (count)', min_value=0, value=0)
            current_emi_amount = st.number_input('Current EMI Amount', min_value=0, value=0)
            credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
            bank_balance = st.number_input('Bank Balance', min_value=0, value=50000)
            emergency_fund = st.number_input('Emergency Fund', min_value=0, value=20000)
            # Loan request
            emi_scenario = st.selectbox('EMI Scenario', ['E-commerce','Home Appliances','Vehicle','Personal Loan','Education'])
            requested_amount = st.number_input('Requested Amount', min_value=0, value=50000)
            requested_tenure = st.number_input('Requested Tenure (months)', min_value=1, value=12)

            submitted = st.form_submit_button('Predict')

        if submitted:
            # Assemble row
            row = {
                'age': age,
                'gender': gender,
                'marital_status': marital_status,
                'education': education,
                'monthly_salary': monthly_salary,
                'employment_type': employment_type,
                'years_of_employment': years_of_employment,
                'company_type': company_type,
                'house_type': house_type,
                'monthly_rent': monthly_rent,
                'family_size': family_size,
                'dependents': dependents,
                'school_fees': school_fees,
                'college_fees': college_fees,
                'travel_expenses': travel_expenses,
                'groceries_utilities': groceries_utilities,
                'other_monthly_expenses': other_monthly_expenses,
                'existing_loans': existing_loans,
                'current_emi_amount': current_emi_amount,
                'credit_score': credit_score,
                'bank_balance': bank_balance,
                'emergency_fund': emergency_fund,
                'emi_scenario': emi_scenario,
                'requested_amount': requested_amount,
                'requested_tenure': requested_tenure,
            }
            input_df = pd.DataFrame([row])
            # feature engineering
            xf = add_financial_ratios(input_df)

            # encode using saved onehot encoder if available
            if ONEHOT is not None:
                xf_enc, _ = simple_encode(xf, encoder=ONEHOT)
            else:
                xf_enc, _ = simple_encode(xf)

            # align to training feature columns if available
            if FEATURE_COLUMNS is not None:
                for c in FEATURE_COLUMNS:
                    if c not in xf_enc.columns:
                        xf_enc[c] = 0
                xf_enc = xf_enc[FEATURE_COLUMNS]

            st.subheader('Predictions')
            # Classification: prefer XGBoost classifier, fallback to logistic pipeline
            if XGB_CLASS is not None:
                try:
                    pred = XGB_CLASS.predict(xf_enc)[0]
                    # decode label if encoder present
                    if LABEL_ENCODER is not None:
                        try:
                            pred_label = LABEL_ENCODER.inverse_transform([pred])[0]
                        except Exception:
                            pred_label = pred
                    else:
                        pred_label = pred
                    st.write('XGBoost Classifier Prediction:', pred_label)
                    if hasattr(XGB_CLASS, 'predict_proba'):
                        probs = XGB_CLASS.predict_proba(xf_enc)[0]
                        st.write('Probabilities:', probs)
                except Exception as e:
                    st.error('Error running XGBoost classifier: ' + str(e))
            elif LOGISTIC_PIPELINE is not None:
                try:
                    pred = LOGISTIC_PIPELINE.predict(xf_enc)[0]
                    st.write('Logistic Pipeline Prediction:', pred)
                except Exception as e:
                    st.error('Error running logistic pipeline: ' + str(e))
            else:
                st.info('No classification model available. Run training to create models in `models/`.')

            # Regression: prefer XGBoost regressor, then RF
            if XGB_REG is not None:
                try:
                    reg_pred = XGB_REG.predict(xf_enc)[0]
                    st.write('XGBoost Regresser predicted max_monthly_emi:', float(reg_pred))
                except Exception as e:
                    st.error('Error running XGBoost regressor: ' + str(e))
            elif RF_REG is not None:
                try:
                    reg_pred = RF_REG.predict(xf_enc)[0]
                    st.write('RandomForest predicted max_monthly_emi:', float(reg_pred))
                except Exception as e:
                    st.error('Error running RF regressor: ' + str(e))
            else:
                st.info('No regression model available. Run training to create models in `models/`.')

    elif menu == 'Data':
        st.header('Dataset (CRUD)')
        df = load_data()
        st.write('Rows:', 0 if df.empty else len(df))
        st.dataframe(df.head(50))
        # Simple upload to replace dataset
        uploaded = st.file_uploader('Upload CSV to replace dataset')
        if uploaded is not None:
            out = Path('data')
            out.mkdir(parents=True, exist_ok=True)
            df2 = pd.read_csv(uploaded)
            df2.to_csv(out / 'sample_emi.csv', index=False)
            st.success('Dataset replaced. Refresh to load new data.')

    else:
        st.header('Model')
        st.write('Models saved to `models/`')
        if MODEL_DIR.exists():
            files = list(MODEL_DIR.glob('*.joblib'))
            if files:
                for f in files:
                    st.write(f.name)
                    if st.button(f'Load {f.name}'):
                        m = load_model(f)
                        st.write('Loaded model:', type(m))
            else:
                st.info('No models found. Run `python src/train.py` to train and save models to `models/`.')
        else:
            st.info('No models directory found. Run training to produce models.')


if __name__ == '__main__':
    main()
