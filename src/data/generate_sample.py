"""Generate a small synthetic EMI dataset matching the project schema.

Usage:
    python src/data/generate_sample.py --out data/sample_emi.csv --n 1000
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate(n=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    ages = rng.integers(25, 61, size=n)
    genders = rng.choice(['Male', 'Female'], size=n)
    marital = rng.choice(['Single', 'Married'], size=n, p=[0.4, 0.6])
    education = rng.choice(['High School', 'Graduate', 'Post Graduate', 'Professional'], size=n, p=[0.2,0.5,0.25,0.05])
    monthly_salary = rng.integers(15000, 200001, size=n)
    employment_type = rng.choice(['Private', 'Government', 'Self-employed'], size=n, p=[0.7,0.15,0.15])
    years_of_employment = rng.integers(0, 41, size=n)
    company_type = rng.choice(['Small', 'Medium', 'Large'], size=n, p=[0.4,0.4,0.2])
    house_type = rng.choice(['Rented', 'Own', 'Family'], size=n, p=[0.3,0.5,0.2])
    monthly_rent = (house_type == 'Rented') * rng.integers(5000, 50001, size=n)
    family_size = rng.integers(1, 7, size=n)
    dependents = rng.integers(0, 4, size=n)
    school_fees = rng.integers(0, 20001, size=n)
    college_fees = rng.integers(0, 30001, size=n)
    travel_expenses = rng.integers(500, 10001, size=n)
    groceries_utilities = rng.integers(2000, 30001, size=n)
    other_monthly_expenses = rng.integers(0, 15001, size=n)
    existing_loans = rng.integers(0, 3, size=n)
    current_emi_amount = existing_loans * rng.integers(1000, 15001, size=n)
    credit_score = rng.integers(300, 851, size=n)
    bank_balance = rng.integers(0, 500001, size=n)
    emergency_fund = rng.integers(0, 200001, size=n)
    emi_scenario = rng.choice(['E-commerce','Home Appliances','Vehicle','Personal Loan','Education'], size=n)
    # requested amount and tenure range roughly by scenario
    requested_amount = []
    requested_tenure = []
    for s in emi_scenario:
        if s == 'E-commerce':
            requested_amount.append(rng.integers(10000, 200001))
            requested_tenure.append(rng.integers(3, 25))
        elif s == 'Home Appliances':
            requested_amount.append(rng.integers(20000, 300001))
            requested_tenure.append(rng.integers(6, 37))
        elif s == 'Vehicle':
            requested_amount.append(rng.integers(80000, 1500001))
            requested_tenure.append(rng.integers(12, 85))
        elif s == 'Personal Loan':
            requested_amount.append(rng.integers(50000, 1000001))
            requested_tenure.append(rng.integers(12, 61))
        else:
            requested_amount.append(rng.integers(50000, 500001))
            requested_tenure.append(rng.integers(6, 49))

    requested_amount = np.array(requested_amount)
    requested_tenure = np.array(requested_tenure)

    # Simple rule-based target generation for demo purposes
    disposable_income = monthly_salary - (monthly_rent + travel_expenses + groceries_utilities + other_monthly_expenses + current_emi_amount)
    affordability = disposable_income - (0.2 * monthly_salary)
    max_monthly_emi = np.clip((affordability * 0.7).astype(int), 500, 50000)

    # classification: Eligible / High_Risk / Not_Eligible
    emi_eligibility = []
    for i in range(n):
        if credit_score[i] > 700 and max_monthly_emi[i] >= (requested_amount[i] / max(1, requested_tenure[i])):
            emi_eligibility.append('Eligible')
        elif credit_score[i] > 600 and max_monthly_emi[i] * 0.8 >= (requested_amount[i] / max(1, requested_tenure[i])):
            emi_eligibility.append('High_Risk')
        else:
            emi_eligibility.append('Not_Eligible')

    df = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'marital_status': marital,
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
        'max_monthly_emi': max_monthly_emi,
        'emi_eligibility': emi_eligibility,
    })

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/sample_emi.csv')
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate(n=args.n, random_state=args.seed)
    df.to_csv(path, index=False)
    print(f"Saved sample data to {path} ({len(df)} rows)")


if __name__ == '__main__':
    main()
