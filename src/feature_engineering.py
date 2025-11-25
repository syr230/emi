"""Feature engineering helpers: compute ratios and encode simple categorical variables."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, Any, Optional


def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Debt-to-income: current EMI + existing loans burden / salary
    df['debt_to_income'] = (df['current_emi_amount'] + df['other_monthly_expenses']) / df['monthly_salary'].replace(0, np.nan)
    df['expense_to_income'] = (df['monthly_rent'] + df['travel_expenses'] + df['groceries_utilities'] + df['other_monthly_expenses']) / df['monthly_salary'].replace(0, np.nan)
    df['affordability_ratio'] = (df['monthly_salary'] - (df['monthly_rent'] + df['travel_expenses'] + df['groceries_utilities'] + df['other_monthly_expenses'] + df['current_emi_amount'])) / df['monthly_salary'].replace(0, np.nan)
    df['dependents_ratio'] = df['dependents'] / df['family_size'].replace(0, np.nan)
    # Fill inf/nan with zeros for downstream models
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def simple_encode(df: pd.DataFrame, categorical_cols=None, encoder: Optional[Any] = None) -> Tuple[pd.DataFrame, Any]:
    """One-hot encode categorical columns.

    If `encoder` is None, the function fits a OneHotEncoder on `categorical_cols` and returns
    (encoded_df, encoder). If `encoder` is provided, it is used to transform the dataframe and
    returned unchanged.
    """
    df = df.copy()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # sklearn >=1.2 uses 'sparse_output' instead of 'sparse'
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        if len(categorical_cols) == 0:
            return df, encoder
        encoded = encoder.fit_transform(df[categorical_cols])
    else:
        if len(categorical_cols) == 0:
            return df, encoder
        encoded = encoder.transform(df[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)
    return df, encoder


if __name__ == '__main__':
    print('feature_engineering module - functions only')
