# Customer Churn Prediction Model

A machine learning pipeline that predicts customer churn for a telecommunications company using Random Forest classification. The model processes customer data, performs feature engineering, and identifies key factors driving churn.

## Overview

This project helps identify customers likely to churn (cancel their service) based on their demographic information, service usage, and billing patterns. The pipeline includes:

- Data cleaning and preprocessing
- Binary, categorical, and numerical feature encoding
- Feature selection via correlation analysis and importance ranking
- Random Forest classification with hyperparameter tuning
- Comprehensive model evaluation

## Installation

```bash
cd model-predict-churn
pip install -r requirements.txt
```

## Usage

### Basic Usage

Ensure your dataset (`churn.csv`) is in the project directory and run:

```bash
python train-test-churn-prediction.py
```

### Expected Data Format

Your CSV file should contain these columns:

- `customerID`: Unique customer identifier
- `gender`: Customer gender ('Male' or 'Female')
- `SeniorCitizen`: Whether customer is 65+ (0 or 1)
- `Partner`: Has a partner ('Yes' or 'No')
- `Dependents`: Has dependents ('Yes' or 'No')
- `tenure`: Months as customer
- `PhoneService`: Has phone service ('Yes' or 'No')
- `MultipleLines`: Has multiple lines ('Yes', 'No', 'No phone service')
- `InternetService`: Type of internet ('DSL', 'Fiber optic', 'No')
- `OnlineSecurity`: Has online security ('Yes', 'No', 'No internet service')
- `OnlineBackup`: Has online backup ('Yes', 'No', 'No internet service')
- `DeviceProtection`: Has device protection ('Yes', 'No', 'No internet service')
- `TechSupport`: Has tech support ('Yes', 'No', 'No internet service')
- `StreamingTV`: Has streaming TV ('Yes', 'No', 'No internet service')
- `StreamingMovies`: Has streaming movies ('Yes', 'No', 'No internet service')
- `Contract`: Contract type ('Month-to-month', 'One year', 'Two year')
- `PaperlessBilling`: Uses paperless billing ('Yes' or 'No')
- `PaymentMethod`: Payment method ('Electronic check', 'Mailed check', 'Bank transfer', 'Credit card')
- `MonthlyCharges`: Monthly charges in dollars
- `Churn`: Target variable ('Yes' or 'No')

## How It Works

### 1. Data Cleaning
- Standardizes column names
- Converts service columns to consistent Yes/No format
- Removes duplicates and null values
- Filters outliers

### 2. Feature Engineering
- Calculates `TotalCharges` from `MonthlyCharges` × `Tenure`
- Binary encodes Yes/No features (17 features)
- One-hot encodes categorical features (InternetService, Contract, PaymentMethod)
- Standardizes numerical features (Tenure, MonthlyCharges, TotalCharges)

### 3. Feature Selection
- **Correlation Analysis**: Removes features with correlation >0.7
- **Importance Ranking**: Selects top features based on Random Forest importance scores
- Reduces dimensionality while retaining predictive power

### 4. Model Training
- Random Forest Classifier (100 trees, max_depth=7)
- 70/30 train-test split with stratification
- Optional hyperparameter tuning via GridSearchCV

### 5. Evaluation
- **Accuracy**: Overall prediction correctness
- **Precision**: Of predicted churners, how many actually churned
- **Recall**: Of actual churners, how many were caught
- **F1 Score**: Balance between precision and recall
- **ROC AUC**: Model's ability to distinguish classes

## Output Files

The script generates three files in the `output/` directory:

1. `churn_model_YYYY-MM-DD.pkl` - Trained Random Forest model
2. `df_selected_features_YYYY-MM-DD.csv` - Sample of processed features
3. `selected_features_YYYY-MM-DD.txt` - List of selected feature names

### Using the Saved Model

```python
import pickle
import pandas as pd

# Load the model
with open('output/churn_model_2024-11-01.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your data (must match training feature set)
customer_data = pd.read_csv('new_customers.csv')

# Make predictions
predictions = model.predict(customer_data)
# Returns: 1 for churn, 0 for no churn
```

## Configuration

Edit these variables in the script:

```python
# Hyperparameter optimization
optimize_features = False  # Set True for GridSearchCV

# Feature importance threshold
threshold = 0.01  # Minimum importance to include feature

# Train-test split
test_size = 0.3  # 30% for testing

# Model parameters
RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=7,       # Maximum tree depth
    random_state=42    # Reproducibility seed
)
```

## Key Findings

Based on typical telecom churn data, the most important features are usually:

1. **Contract Type**: Month-to-month contracts have higher churn
2. **Tenure**: Newer customers churn more frequently
3. **Monthly Charges**: Higher charges correlate with churn
4. **Internet Service**: Fiber optic users may churn more
5. **Tech Support**: Lack of tech support increases churn risk

## Model Performance

Expected performance metrics:
- Accuracy: 78-82%
- Precision: 65-70%
- Recall: 55-65%
- F1 Score: 60-67%
- ROC AUC: 75-80%

Note: Churn prediction is inherently challenging due to class imbalance (typically 20-30% churn rate).

## Troubleshooting

**"churn.csv not found"**
- Ensure the CSV file is in the same directory as the script

**Poor model performance**
- Check class balance (churn rate should be 15-35%)
- Verify all required columns are present
- Try enabling hyperparameter optimization (`optimize_features = True`)

**Too many/few features selected**
- Adjust the `threshold` value (lower = more features, higher = fewer features)

## License

This project is provided for educational and commercial use.

