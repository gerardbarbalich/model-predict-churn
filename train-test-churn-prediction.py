"""
This script performs churn prediction analysis for a Telco Company. 
It includes data cleaning, feature engineering, model training using 
RandomForestClassifier, and model evaluation.

Expected Inputs:
- CSV file ('churn_schibsted.csv') containing customer churn data

Expected Output:
- Pickle file ('churn_model.pkl') storing the trained churn prediction model
- Evaluation metrics printed to console: Accuracy, Precision, Recall, F1 Score, ROC AUC Score

Dependencies:
- python 3.11.0, pandas 2.0.3, numpy 1.25, scikit-learn 1.3 , matplotlib, seaborn
"""
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, List, Any


# Initialize an empty dictionary to store data
data: Dict[str, Any] = {
    'df': pd.DataFrame,
    'df_processed': pd.DataFrame,
    'df_binary': pd.DataFrame,
    'df_scaled': pd.DataFrame,
    'df_X_selected_features': pd.DataFrame, 
    'ls_binary_cols': list,
    'ls_cat_cols': list,
    'ls_numeric_cols': list,
    'ls_total_services': list,
}

# Binary feature columns (Yes/No encoded as 1/0)
data['ls_binary_cols'] = [
    'IsFemale', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'OnlineSecurity', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'PaperlessBilling', 'Churn'
]

# Categorical feature columns (will be one-hot encoded)
data['ls_cat_cols'] = [
    'InternetService', 
    'Contract', 
    'PaymentMethod'
]

# Continuous numeric feature columns (will be scaled)
data['ls_numeric_cols'] = [
    'Tenure', 
    'MonthlyCharges', 
    'TotalCharges'
]

# List of all service features (for potential feature engineering)
data['ls_total_services'] = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies'
]


def scale_features(
    df: pd.DataFrame, list_of_columns: list) -> (pd.DataFrame, StandardScaler):
    """
    Standardize numerical features to have zero mean and unit variance.
    
    Args:
        df: DataFrame containing features to scale
        list_of_columns: List of column names to apply scaling to
    
    Returns:
        Tuple of (scaled DataFrame, fitted StandardScaler)
    """
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[list_of_columns])
    scaled_and_labelled_df = pd.DataFrame(scaled_df, columns=list_of_columns, index=df.index)
    return scaled_and_labelled_df, scaler


def scale_and_assess_feature_correlation(
    df: pd.DataFrame, list_of_columns: list) -> (pd.DataFrame, StandardScaler):
    """
    Scale features and visualize their correlations.
    
    This helps identify highly correlated features that may cause multicollinearity.
    
    Args:
        df: DataFrame containing features
        list_of_columns: Columns to scale and analyze
    
    Returns:
        Tuple of (scaled DataFrame, fitted StandardScaler)
    """
    scaled_and_labelled_df, scaler = scale_features(df, list_of_columns)
    correlation_matrix = scaled_and_labelled_df.corr()
    
    # Create correlation heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    
    return scaled_and_labelled_df, scaler


def find_correlation(corr, cutoff=0.7, exact=None):
    """
    This function is the Python implementation of the R function 
    `findCorrelation()`. It searches through a correlation matrix and returns a list of column names 
    to remove to reduce pairwise correlations.
    
    For the documentation of the R function, see 
    https://www.rdocumentation.org/packages/caret/topics/findCorrelation
    and for the source code of `findCorrelation()`, see
    https://github.com/topepo/caret/blob/master/pkg/caret/R/findCorrelation.R
    
    Parameters:
    -----------
    corr: pandas dataframe.
        A correlation matrix as a pandas dataframe.
    cutoff: float, default: 0.9.
        A numeric value for the pairwise absolute correlation cutoff
    exact: bool, default: None
        A boolean value that determines whether the average correlations be 
        recomputed at each step
    """
    
    def _findCorrelation_fast(corr, avg, cutoff):
        """Fast method: doesn't recompute averages at each step"""
        # Find all feature pairs with correlation above cutoff (upper triangle only)
        combsAboveCutoff = corr.where(lambda x: (np.tril(x)==0) & (x > cutoff)).stack().index

        rowsToCheck = combsAboveCutoff.get_level_values(0)
        colsToCheck = combsAboveCutoff.get_level_values(1)

        # For each pair, mark the feature with higher average correlation for deletion
        msk = avg[colsToCheck] > avg[rowsToCheck].values
        deletecol = pd.unique(np.r_[colsToCheck[msk], rowsToCheck[~msk]]).tolist()

        return deletecol


    def _findCorrelation_exact(corr, avg, cutoff):
        """Exact method: recomputes averages after each removal"""
        # Sort by average correlation (highest first)
        x = corr.loc[(*[avg.sort_values(ascending=False).index]*2,)]

        # Convert integer dtypes to float for NaN support
        if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            x = x.astype(float)

        # Set diagonal to NaN (perfect self-correlation not relevant)
        x.values[(*[np.arange(len(x))]*2,)] = np.nan

        deletecol = []
        # Check each pair of features
        for ix, i in enumerate(x.columns[:-1]):
            for j in x.columns[ix+1:]:
                if x.loc[i, j] > cutoff:
                    # Remove the feature with higher mean correlation
                    if x[i].mean() > x[j].mean():
                        deletecol.append(i)
                        x.loc[i] = x[i] = np.nan
                    else:
                        deletecol.append(j)
                        x.loc[j] = x[j] = np.nan
        return deletecol

    
    if not np.allclose(corr, corr.T) or any(corr.columns!=corr.index):
        raise ValueError("correlation matrix is not symmetric.")
        
    acorr = corr.abs()
    avg = acorr.mean()
        
    if exact or exact is None and corr.shape[1]<100:
        return _findCorrelation_exact(acorr, avg, cutoff)
    else:
        return _findCorrelation_fast(acorr, avg, cutoff)


def assess_correlation_trim_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze feature correlations and remove highly correlated features.
    
    This reduces multicollinearity by identifying and dropping features that
    are highly correlated with others. This improves model interpretability
    and can prevent overfitting.
    
    Args:
        df: DataFrame with features to analyze
    
    Returns:
        DataFrame with highly correlated features removed
    """
    # Remove any duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Calculate and visualize correlations
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    
    # Identify columns to drop
    to_drop = find_correlation(correlation_matrix)
    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")

    return df.drop(columns=to_drop)



# ============================================================================
# SECTION 1: Import, Examine, and Clean Data
# ============================================================================

# Load the customer churn dataset
data['df'] = pd.read_csv('churn.csv')
print("Dataset shape:", data['df'].shape)
print("\nFirst few rows:")
print(data['df'].head())

# Standardize column names for consistency
data['df'] = data['df'].rename(columns={
    'customerID': 'CustomerID',
    'gender': 'IsFemale',
    'tenure': 'Tenure'
})

# Set CustomerID as index
data['df'].set_index('CustomerID', inplace=True, drop=True)

# Standardize service feature values
# Convert 'No phone service' and 'No internet service' to simple 'No'
cols_to_replace = [
    'MultipleLines', 'OnlineSecurity', 'StreamingTV', 'StreamingMovies', 'TechSupport'
]
for col in cols_to_replace:
    data['df'][col] = data['df'][col].replace({
        '': np.nan,
        ' ': 0,
        'No phone service': 'No',
        'No internet service': 'No'  
    })

# Remove duplicate rows and rows with missing values
initial_rows = len(data['df'])
data['df'].drop_duplicates(inplace=True)
data['df'].dropna(inplace=True)
print(f"\nRemoved {initial_rows - len(data['df'])} rows (duplicates and nulls)")


# ============================================================================
# SECTION 2: Feature Engineering and Encoding
# ============================================================================

# Standardize gender encoding to Yes/No
data['df']['IsFemale'] = data['df']['IsFemale'].replace({'Female': 'Yes', 'Male': 'No'})

# Convert SeniorCitizen from 1/0 to Yes/No for consistency
data['df']['SeniorCitizen'] = data['df']['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})

# Calculate total charges from monthly charges and tenure
data['df']['TotalCharges'] = data['df']['MonthlyCharges'] * data['df']['Tenure']

# Remove outliers using reasonable business thresholds
data['df'] = data['df'].loc[data['df']['Tenure'].between(0, 100)]
data['df'] = data['df'].loc[data['df']['TotalCharges'].between(0, 10000)]
print(f"After outlier removal: {len(data['df'])} rows")

# 2.1: Binary Encoding (Yes/No → 1/0)
data['df_binary'] = data['df'][data['ls_binary_cols']].replace({'Yes': 1, 'No': 0})
print(f"\nBinary encoded features: {len(data['ls_binary_cols'])} columns")

# 2.2: One-Hot Encoding for categorical features
encoder = OneHotEncoder(sparse=False, dtype=int)
encoded_data = encoder.fit_transform(data['df'][data['ls_cat_cols']])
data['df_encoded'] = pd.DataFrame(
    encoded_data, 
    columns=encoder.get_feature_names_out(data['ls_cat_cols'])
)
# Preserve the original index
data['df_encoded'].index = data['df'][data['ls_cat_cols']].index
print(f"One-hot encoded features: {len(data['df_encoded'].columns)} columns")

# 2.3: Standardize numerical features
data['df_scaled'], data['scaler'] = scale_and_assess_feature_correlation(
    data['df'], 
    data['ls_numeric_cols']
)
print(f"Scaled numeric features: {len(data['ls_numeric_cols'])} columns") 


# ============================================================================
# SECTION 3: Feature Selection via Correlation Analysis
# ============================================================================

# Remove highly correlated features from each feature set
print("\nTrimming binary features...")
data['df_binary_trimmed'] = assess_correlation_trim_columns(data['df_binary'])

print("\nTrimming scaled features...")
data['df_scaled_trimmed'] = assess_correlation_trim_columns(data['df_scaled'])

print("\nTrimming encoded features...")
data['df_encoded_trimmed'] = assess_correlation_trim_columns(data['df_encoded'])

# Combine all processed feature sets
data['df_processed'] = pd.concat(
    [data['df_binary_trimmed'], 
     data['df_scaled_trimmed'],
     data['df_encoded_trimmed']], 
    axis=1
).dropna(axis=0)

print(f"\nFinal processed dataset: {data['df_processed'].shape}")

# Check class balance (important for churn prediction)
print('\n' + '='*60)
print('CLASS BALANCE CHECK')
print('='*60)
print('Churn counts:')
print(data['df_processed']['Churn'].value_counts())
print(f"Churn rate: {data['df_processed']['Churn'].mean():.1%}")

# Verify no missing data remains
missing_values = data['df_processed'].isnull().sum()
if missing_values.sum() > 0:
    print('\nWarning: Missing values found:')
    print(missing_values[missing_values > 0])
else:
    print('\nNo missing values ✓')


# ============================================================================
# SECTION 4: Model Training and Hyperparameter Tuning
# ============================================================================

# Separate features and target variable
X = data['df_processed'].drop(['Churn'], axis=1)
y = data['df_processed']['Churn']

# Split into training and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y  # Maintain class balance in splits
)

print('\n' + '='*60)
print('MODEL TRAINING')
print('='*60)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# 4.1: Train the model (with optional hyperparameter optimization)
optimize_features = False  # Set to True to run grid search

if optimize_features:
    print("\nRunning hyperparameter optimization (this may take a while)...")
    params = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
    clf = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        params, 
        cv=5,
        scoring='roc_auc'
    )
    clf.fit(X_train, y_train)
    print(f"Best parameters: {clf.best_params_}")
    print(f"Best cross-validation score: {clf.best_score_:.3f}")
else:
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(random_state=42, max_depth=7, n_estimators=100)
    clf.fit(X_train, y_train)
    print("Training complete")


# 4.2: Feature Selection Based on Importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Extract feature importances from the trained model
feature_importance = clf.feature_importances_

# Create a sorted DataFrame of feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': feature_importance
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance_df.head(10).to_string(index=False))

# Select features above importance threshold
threshold = 0.01  # Features contributing at least 1%
selected_features = feature_importance_df[
    feature_importance_df['Importance'] > threshold
]['Feature'].tolist()

print(f"\nSelected {len(selected_features)} features above {threshold} importance threshold")

# Create dataset with only selected features
data['df_X_selected_features'] = X[selected_features]

# Re-split data with selected features
X_train, X_test, y_train, y_test = train_test_split(
    data['df_X_selected_features'], 
    y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)

# Retrain model on selected features
print("\nRetraining model with selected features...")
clf = RandomForestClassifier(random_state=42, max_depth=7, n_estimators=100)
clf.fit(X_train, y_train)
print("Retraining complete")


# ============================================================================
# SECTION 5: Model Evaluation
# ============================================================================

# Generate predictions on test set
y_pred = clf.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test, y_pred)

# Display comprehensive results
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)

print(f"\nAccuracy: {accuracy:.3f}")
print(f"  → {accuracy*100:.1f}% of all predictions were correct")

print(f"\nPrecision: {precision:.3f}")  
print(f"  → Of customers predicted to churn, {precision*100:.1f}% actually churned")

print(f"\nRecall (Sensitivity): {recall:.3f}")
print(f"  → Of all customers who churned, {recall*100:.1f}% were correctly identified")

print(f"\nF1 Score: {f1:.3f}")
print(f"  → Harmonic mean of precision and recall")

print(f"\nROC AUC Score: {roc_auc:.3f}")
print(f"  → Model's ability to distinguish between classes (0.5=random, 1.0=perfect)")


# ============================================================================
# SECTION 6: Export Model and Feature Information
# ============================================================================

# Create output directory if it doesn't exist
import os
os.makedirs('output', exist_ok=True)

# Generate filename with current date
current_date = str(date.today())

# Save the trained model
model_filename = f'output/churn_model_{current_date}.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(clf, file)
print(f"\n{'='*60}")
print(f"Model saved to: {model_filename}")

# Save sample of selected features for reference
features_filename = f'output/df_selected_features_{current_date}.csv'
data['df_X_selected_features'].head(5).to_csv(features_filename)
print(f"Feature sample saved to: {features_filename}")

# Save the list of selected feature names
feature_list_filename = f'output/selected_features_{current_date}.txt'
with open(feature_list_filename, 'w') as f:
    f.write('\n'.join(selected_features))
print(f"Feature list saved to: {feature_list_filename}")
print(f"{'='*60}")

