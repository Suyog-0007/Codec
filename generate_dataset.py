import pandas as pd
import numpy as np

np.random.seed(42)
n = 7043

# Simulate IBM Telco-style dataset
gender = np.random.choice(['Male', 'Female'], n)
senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n)
dependents = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
tenure = np.random.randint(0, 72, n)
phone_service = np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1])
multiple_lines = np.where(phone_service == 'No', 'No phone service',
                  np.random.choice(['Yes', 'No'], n))
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
online_security = np.where(internet_service == 'No', 'No internet service',
                   np.random.choice(['Yes', 'No'], n))
online_backup = np.where(internet_service == 'No', 'No internet service',
                  np.random.choice(['Yes', 'No'], n))
device_protection = np.where(internet_service == 'No', 'No internet service',
                     np.random.choice(['Yes', 'No'], n))
tech_support = np.where(internet_service == 'No', 'No internet service',
                np.random.choice(['Yes', 'No'], n))
streaming_tv = np.where(internet_service == 'No', 'No internet service',
               np.random.choice(['Yes', 'No'], n))
streaming_movies = np.where(internet_service == 'No', 'No internet service',
                   np.random.choice(['Yes', 'No'], n))
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24])
paperless_billing = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
payment_method = np.random.choice(
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n)

monthly_charges = np.round(np.random.uniform(18, 118, n), 2)
total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 5, n), 2)
total_charges = np.clip(total_charges, 0, None)

# Churn logic (rule-based to mimic real patterns)
churn_prob = (
    0.05
    + 0.15 * (contract == 'Month-to-month')
    + 0.10 * (internet_service == 'Fiber optic')
    - 0.10 * (tenure > 30)
    + 0.08 * (senior == 1)
    - 0.05 * (online_security == 'Yes')
    + 0.05 * (payment_method == 'Electronic check')
    - 0.05 * (partner == 'Yes')
)
churn_prob = np.clip(churn_prob, 0.02, 0.95)
churn = np.random.binomial(1, churn_prob, n)
churn_label = np.where(churn == 1, 'Yes', 'No')

df = pd.DataFrame({
    'customerID': [f'CUST-{i:05d}' for i in range(n)],
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn_label
})

# Introduce some missing values to TotalCharges (realistic)
missing_idx = np.random.choice(df.index, size=11, replace=False)
df.loc[missing_idx, 'TotalCharges'] = np.nan

df.to_csv('telco_churn.csv', index=False)
print(f"Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
