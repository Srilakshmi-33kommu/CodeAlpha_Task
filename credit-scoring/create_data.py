import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'Income': np.random.normal(50000, 15000, n_samples).astype(int),
    'Debt': np.random.normal(20000, 5000, n_samples).astype(int),
    'Age': np.random.randint(18, 70, n_samples),
    'PaymentHistory': np.random.uniform(0.7, 1.0, n_samples),
    'CreditLimit': np.random.normal(30000, 10000, n_samples).astype(int),
    'CreditUsed': np.random.normal(12000, 5000, n_samples).astype(int),
    'EmploymentType': np.random.choice(['FullTime', 'PartTime', 'SelfEmployed', 'Unemployed'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
    'EducationLevel': np.random.choice(['HighSchool', 'Bachelors', 'Masters', 'PhD'], n_samples, p=[0.3, 0.5, 0.15, 0.05]),
}

df = pd.DataFrame(data)

# Create target variable (Creditworthiness)
df['Creditworthiness'] = (
    (df['Income'] > 45000) & 
    (df['Debt'] < 30000) & 
    (df['PaymentHistory'] > 0.9) & 
    (df['CreditUsed'] < df['CreditLimit'] * 0.8)
).astype(int)

# Save to CSV
df.to_csv('credit_data.csv', index=False)
print("Sample credit_data.csv created with 1000 records!")