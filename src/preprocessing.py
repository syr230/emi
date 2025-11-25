"""Data loading and preprocessing utilities for EMIPredict AI.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    df = pd.read_csv(p)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop exact duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # Basic null handling: fill numeric with median, categorical with mode
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def split_data(df: pd.DataFrame, target_class='emi_eligibility', target_reg='max_monthly_emi', test_size=0.2, val_size=0.1, random_state=42):
    # First split out test
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    # split train_val into train and val
    val_rel = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_rel, random_state=random_state)
    Xy_train = train.copy()
    Xy_val = val.copy()
    Xy_test = test.copy()
    return Xy_train, Xy_val, Xy_test


if __name__ == '__main__':
    print('preprocessing module - utility functions only')
