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

data['ls_binary_cols'] = [
    'IsFemale', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'OnlineSecurity', 'TechSupport', 'StreamingTV', 
    'StreamingMovies', 'TechSupport' ,'StreamingTV', 'StreamingMovies', 
    'PaperlessBilling', 'Churn'
]

data['ls_cat_cols'] = [
    'InternetService', 
    'Contract', 
    'PaymentMethod', 
    # 'City'
    ]

data['ls_numeric_cols'] = [
    'Tenure', 
    'MonthlyCharges', 
    'TotalCharges']

data['ls_total_services'] = [
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
    'StreamingMovies']


def scale_features(
    df: pd.DataFrame, list_of_columns: list) -> (pd.DataFrame, StandardScaler):
    """Scales the features"""
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df[list_of_columns])
    scaled_and_labelled_df = pd.DataFrame(scaled_df, columns=list_of_columns, index=df.index)
    return scaled_and_labelled_df, scaler


def scale_and_assess_feature_correlation(
    df: pd.DataFrame, list_of_columns: list) -> (pd.DataFrame, StandardScaler):
    """Once scaled, this assesses the correlation of the features"""
    scaled_and_labelled_df, scaler = scale_features(df, list_of_columns)
    correlation_matrix = scaled_and_labelled_df.corr()
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

        combsAboveCutoff = corr.where(lambda x: (np.tril(x)==0) & (x > cutoff)).stack().index

        rowsToCheck = combsAboveCutoff.get_level_values(0)
        colsToCheck = combsAboveCutoff.get_level_values(1)

        msk = avg[colsToCheck] > avg[rowsToCheck].values
        deletecol = pd.unique(np.r_[colsToCheck[msk], rowsToCheck[~msk]]).tolist()

        return deletecol


    def _findCorrelation_exact(corr, avg, cutoff):

        x = corr.loc[(*[avg.sort_values(ascending=False).index]*2,)]

        if (x.dtypes.values[:, None] == ['int64', 'int32', 'int16', 'int8']).any():
            x = x.astype(float)

        x.values[(*[np.arange(len(x))]*2,)] = np.nan

        deletecol = []
        for ix, i in enumerate(x.columns[:-1]):
            for j in x.columns[ix+1:]:
                if x.loc[i, j] > cutoff:
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


def assess_correlation_trim_columns(
    df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in df, asseses correlation, drops unneeded columns 
    """
    df = df.loc[:, ~df.columns.duplicated()]
    
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    
    to_drop = find_correlation(correlation_matrix)

    return df.drop(columns=to_drop)



# Section 1: Import, examine, and clean the data
data['df'] = pd.read_csv('churn.csv')
print(data['df'].head())

# Cleaning
data['df'] = data['df'].rename(columns={
    'customerID': 'CustomerID',
    'gender': 'IsFemale',
    'tenure': 'Tenure'
})

data['df'].set_index('CustomerID', inplace=True, drop=True)

cols_to_replace = [
    'MultipleLines', 'OnlineSecurity', 'StreamingTV', 'StreamingMovies','TechSupport'
]
for col in cols_to_replace:
    data['df'][col] = data['df'][col].replace({
        '': np.nan,
        ' ': 0,
        # ' internet service': '',
        'No phone service': 'No',
        'No internet service': 'No'  
    })

# Drop dupes and nulls
data['df'].drop_duplicates(inplace=True)
data['df'].dropna(inplace=True)


# Section 2: Encode and scale data

data['df']['IsFemale'] = data['df']['IsFemale'].replace({'Female':'Yes', 'Male':'No'})
data['df']['SeniorCitizen'] = data['df']['SeniorCitizen'].replace({1:'Yes', 0:'No'})
data['df']['TotalCharges'] = data['df']['MonthlyCharges'] * data['df']['Tenure']

# Remove outliers
data['df'] = data['df'].loc[data['df']['Tenure'].between(0, 100)]
data['df'] = data['df'].loc[data['df']['TotalCharges'].between(0, 10000)]

# 2.1 Binary encode data
data['df_binary']= data['df'][data['ls_binary_cols']].replace({'Yes': 1, 'No': 0})


# 2,2 One hot encode data
encoder = OneHotEncoder(sparse=False, dtype=int)
encoded_data = encoder.fit_transform(data['df'][data['ls_cat_cols']])
data['df_encoded'] = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(data['ls_cat_cols']))
data['df_encoded'].index = data['df'][data['ls_cat_cols']].index # Set the index from data['df'][data['ls_cat_cols']]

# 2.3 Scale data
data['df_scaled'], data['scaler'] = scale_and_assess_feature_correlation(data['df'], data['ls_numeric_cols']) # none of these are over 0.8 correlation 


# Section 3: Assess features, trim, and concat all dataframes
data['df_binary_trimmed'] = assess_correlation_trim_columns(data['df_binary'])
data['df_scaled_trimmed'] = assess_correlation_trim_columns(data['df_scaled'])
data['df_encoded_trimmed'] = assess_correlation_trim_columns(data['df_encoded'])

# Concat all of the dfs
data['df_processed'] = pd.concat(
    [data['df_binary_trimmed'], 
     data['df_scaled_trimmed'],
     data['df_encoded_trimmed']], axis=1).dropna(axis=0)


# Check for imbalanced data
print('Churn counts:\n', data['df_processed']['Churn'].value_counts())

# Check for missing data
print('Missing values:\n', data['df_processed'].isnull().sum())


# Section 4: Train and evaluate the model
# Split the data into training and testing sets
X = data['df_processed'].drop(['Churn'], axis=1)
y = data['df_processed']['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 4.1: Train the model and optimize it 
optimize_features = False
if optimize_features:
    params = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
    clf = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5)
    clf.fit(X_train, y_train)
    print(f"Best parameters: {clf.best_params_}")
else:
    clf = RandomForestClassifier(random_state=42, max_depth=7, n_estimators=100)
    clf.fit(X_train, y_train)


# 4.2: Select top features
# Calculate feature importance
feature_importance = clf.feature_importances_

# Create a DataFrame to associate features with their importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Select top 'n' important features or set a threshold
threshold = 0.1  # Example threshold value (you can adjust this)
selected_features = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

# Retain only the selected features in your dataset
data['df_X_selected_features'] = X[selected_features]
data['df_X_selected_features']

X_train, X_test, y_train, y_test = train_test_split(data['df_X_selected_features'], y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42, max_depth=7, n_estimators=100)
clf.fit(X_train, y_train)


# 4.3: Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test, y_pred)

# Display results
print(f"Accuracy: {accuracy}") # Accuracy: The proportion of correctly predicted outcomes among the total predictions made. 
print(f"Precision: {precision}") # Precision: The ratio of correctly predicted positive observations to the total predicted positive observations.
print(f"Recall: {recall}") # Recall (Sensitivity): The ratio of correctly predicted positive observations to the all observations in the actual class.
print(f"F1 Score: {f1}") # F1 Score: The weighted average of Precision and Recall. It's a good balance between Precision and Recall, especially when there's an uneven class distribution.
print(f"ROC AUC Score: {roc_auc}") # ROC AUC Score: The area under the Receiver Operating Characteristic curve. It measures the ability of the model to distinguish between classes, particularly useful for imbalanced datasets.


# Section 5: Output the model and df head of selected features as a pickle file
date = str(date.today())

with open(f'output/churn_model_{str(date)}.pkl', 'wb') as file:
    pickle.dump(clf, file)

data['ls_X_selected_features'].head(5).to_csv(f'output/df_selected_features{date}.csv')

