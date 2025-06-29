# Creditworthiness Prediction System
# Objective: Predict creditworthiness using financial data
# Approach: Classification with Logistic Regression, Decision Trees, and Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, auc, classification_report)
from imblearn.over_sampling import SMOTE
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Step 1: Create Synthetic Dataset ---
def create_credit_data(n_samples=1000):
    """Generate synthetic credit data with realistic financial features"""
    np.random.seed(42)
    
    data = {
        'Income': np.random.normal(50000, 15000, n_samples).astype(int),
        'Debt': np.random.normal(20000, 5000, n_samples).astype(int),
        'Age': np.random.randint(18, 70, n_samples),
        'PaymentHistory': np.random.uniform(0.7, 1.0, n_samples),
        'CreditLimit': np.random.normal(30000, 10000, n_samples).astype(int),
        'CreditUsed': np.random.normal(12000, 5000, n_samples).astype(int),
        'EmploymentType': np.random.choice(
            ['FullTime', 'PartTime', 'SelfEmployed', 'Unemployed'], 
            n_samples, 
            p=[0.6, 0.2, 0.15, 0.05]
        ),
        'EducationLevel': np.random.choice(
            ['HighSchool', 'Bachelors', 'Masters', 'PhD'], 
            n_samples, 
            p=[0.3, 0.5, 0.15, 0.05]
        ),
        'LoanHistory': np.random.randint(0, 10, n_samples),
        'CreditInquiries': np.random.randint(0, 8, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (Creditworthiness)
    df['Creditworthiness'] = (
        (df['Income'] > 45000) & 
        (df['Debt'] < 30000) & 
        (df['PaymentHistory'] > 0.9) & 
        (df['CreditUsed'] < df['CreditLimit'] * 0.8) &
        (df['LoanHistory'] > 3) &
        (df['CreditInquiries'] < 4)
    ).astype(int)
    
    return df

# Generate synthetic data
print("Creating synthetic credit dataset...")
data = create_credit_data(1500)

# --- Step 2: Feature Engineering ---
print("\nPerforming feature engineering...")
data['Debt_to_Income'] = data['Debt'] / (data['Income'] + 1e-6)
data['Payment_Missed_Rate'] = 1 - data['PaymentHistory']
data['Credit_Utilization'] = data['CreditUsed'] / (data['CreditLimit'] + 1e-6)
data['Available_Credit'] = data['CreditLimit'] - data['CreditUsed']
data['Loan_to_Income'] = data['Debt'] / (data['Income'] + 1e-6)

# --- Step 3: Data Exploration ---
print("\nDataset Overview:")
print(f"Shape: {data.shape}")
print(f"\nCreditworthiness Distribution:\n{data['Creditworthiness'].value_counts(normalize=True)}")
print("\nData Types:\n", data.dtypes)
print("\nMissing Values:\n", data.isnull().sum())

# --- Step 4: Prepare Data ---
X = data.drop('Creditworthiness', axis=1)
y = data['Creditworthiness']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 5: Preprocessing Pipeline ---
numeric_features = ['Income', 'Debt', 'Age', 'CreditLimit', 'CreditUsed', 
                    'LoanHistory', 'CreditInquiries', 'Debt_to_Income', 
                    'Payment_Missed_Rate', 'Credit_Utilization', 
                    'Available_Credit', 'Loan_to_Income']

categorical_features = ['EmploymentType', 'EducationLevel']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Handle class imbalance using SMOTE
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)

# --- Step 6: Model Training ---
print("\nTraining models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_res, y_train_res)
    
    # Preprocess test data
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_preprocessed)
    y_proba = model.predict_proba(X_test_preprocessed)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC AUC': roc_auc,
        'Model': model
    }
    
    # Print classification report
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Creditworthy', 'Creditworthy'],
                yticklabels=['Not Creditworthy', 'Creditworthy'])
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

# --- Step 7: Model Comparison ---
print("\nModel Performance Comparison:")
results_df = pd.DataFrame(results).T.drop('Model', axis=1)
print(results_df)

# Plot ROC curves
plt.figure(figsize=(10, 8))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['Model'].predict_proba(X_test_preprocessed)[:, 1])
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {res["ROC AUC"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Step 8: Model Optimization ---
print("\nOptimizing Random Forest with Grid Search...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate optimized model
y_pred_best = best_rf.predict(X_test_preprocessed)
y_proba_best = best_rf.predict_proba(X_test_preprocessed)[:, 1]

print("\nOptimized Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best))

print("Optimized ROC AUC:", roc_auc_score(y_test, y_proba_best))

# --- Step 9: Feature Importance ---
feature_importances = best_rf.feature_importances_
features = numeric_features + list(preprocessor.named_transformers_['cat']
                                  .named_steps['onehot']
                                  .get_feature_names_out(categorical_features))

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Top 15 Important Features')
plt.tight_layout()
plt.show()

# --- Step 10: Business Interpretation ---
print("\nKey Business Insights:")
print("1. Top predictors of creditworthiness:")
print(importance_df.head(5))
print("\n2. Model Recommendations:")
print("   - For high precision (minimizing false approvals): Focus on Precision metric")
print("   - For identifying most creditworthy applicants: Focus on Recall metric")
print("   - Best overall model: Random Forest (highest ROC-AUC)")

print("\nCreditworthiness prediction completed successfully!")