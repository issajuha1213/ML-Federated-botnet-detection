import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
%matplotlib inline
# Scikit-learn for Machine Learning Models & Metrics
from sklearn.model_selection import train_test_split, KFold, GridSearchCV  # splitting datasets, cross-validation, and hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier  # the Random Forest classification model
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score)  # various evaluation metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder  # data scaling and encoding
from sklearn.inspection import permutation_importance  # feature importancer tuning?
from sklearn.feature_selection import SelectFromModel 
from tensorflow import keras  
from tensorflow.keras.models import Sequential  # linear stack of layers in neural networks
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, Flatten, BatchNormalization, Activation  # neural network layers
from tensorflow.keras.regularizers import l1_l2  # L1 and L2 regularization
from tensorflow.keras.optimizers import Adam  # Optimizer for training neural networks
#Import Dataset
dataset=pd.read_csv('combined9.csv')
d1= pd.read_csv('Device 1.csv')
d2= pd.read_csv('Device 2.csv')
d3= pd.read_csv('Device 3.csv')
d4= pd.read_csv('Device 4.csv')
d5= pd.read_csv('Device 5.csv')
d6= pd.read_csv('Device 6.csv')
d7= pd.read_csv('Device 7.csv')
d8= pd.read_csv('Device 8.csv')
d9= pd.read_csv('Device 9.csv')
# Brief description for the dataset
print("Dataset Shape:", dataset.shape) # Dataset.shape is a function that returns a tuple (number of rows, number of columns)
dataset.info()   # Datatype
dataset.describe()
dataset.columns
dataset.head()
def analyze_and_clean_data(datasets):
    for idx in range(9):
        print(f"Analysis for Dataset {idx+1}")
        # Calculate missing values for each column in the dataset
        missing_values = dataset.isna().sum()
        # Calculate infinite values for each column in the dataset
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns
        infinite_values = np.isinf(dataset[numeric_cols]).sum()

        # Calculate percentages
        total_values = len(dataset)
        missing_percentages = missing_values / total_values
        infinite_percentages = infinite_values / total_values

        # Set thresholds
        MISSING_VALUES_COLUMN_THRESHOLD = 0.5 
        INFINITE_VALUES_THRESHOLD = 0.1        
        MISSING_VALUES_ROW_THRESHOLD = 0.5
        # Determine columns to drop
        columns_to_drop = missing_percentages[(missing_percentages > MISSING_VALUES_COLUMN_THRESHOLD) | (infinite_percentages > INFINITE_VALUES_THRESHOLD)].index

        # Dataframe of two columns (Missing & Infinite Percentages) and replace any missing value with zero
        columns_to_drop_dataset = pd.DataFrame({'Missing Percentage': missing_percentages[columns_to_drop], 'Infinite Percentage': infinite_percentages[columns_to_drop]}).fillna(0)

        # Identify columns for imputation
        moderate_missing_cols = missing_percentages[(missing_percentages > 0) & (missing_percentages <= MISSING_VALUES_COLUMN_THRESHOLD)]
        moderate_infinite_cols = infinite_percentages[(infinite_percentages > 0) & (infinite_percentages <= INFINITE_VALUES_THRESHOLD)]

        # Combine the columns using union of indices
        columns_for_imputation_set = set(moderate_missing_cols.index).union(set(moderate_infinite_cols.index))
        # Convert the set to a list for DataFrame creation
        columns_for_imputation_list = list(columns_for_imputation_set)

        # Create DataFrame for columns to impute
        columns_for_imputation_dataset = pd.DataFrame(index=columns_for_imputation_list)
        columns_for_imputation_dataset['Missing Percentage'] = missing_percentages[columns_for_imputation_dataset.index]
        columns_for_imputation_dataset['Infinite Percentage'] = infinite_percentages[columns_for_imputation_dataset.index]
        columns_for_imputation_dataset.fillna(0, inplace=True)  # Fill NaN with 0 for clarity

        # Analyze rows for missing values
        rows_to_drop_dataset = pd.DataFrame(dataset.isna().sum(axis=1) / dataset.shape[1], columns=['Missing Values Proportion'])
        rows_to_drop_dataset = rows_to_drop_dataset[rows_to_drop_dataset['Missing Values Proportion'] > MISSING_VALUES_ROW_THRESHOLD]

        # Print results
        print(f"\nColumns with High Missing/Infinite Values (To Drop): {len(columns_to_drop_dataset)}")
        if not(columns_to_drop_dataset.empty):
            print(columns_to_drop_dataset)

        print(f"\nColumns with Moderate Missing/Infinite Values (Consider Imputation): {len(columns_for_imputation_dataset)}")
        if not(columns_for_imputation_dataset.empty):
            print(columns_for_imputation_dataset)

        print(f"\nTotal Number of Rows to Drop: {len(rows_to_drop_dataset)}")
        if not(rows_to_drop_dataset.empty):
            print(rows_to_drop_dataset)
        
        # Detailed count for missing and infinite values
        missing_values_count = missing_values[missing_values > 0]
        infinite_values_count = infinite_values[infinite_values > 0]
        missing_values_dataset = pd.DataFrame(missing_values_count, columns=['Missing Values Count'])
        infinite_values_dataset = pd.DataFrame(infinite_values_count, columns=['Infinite Values Count'])

        if not missing_values_dataset.empty:
            print("\nMissing Values Count Before Imputation:\n", missing_values_dataset)
        else:
            print("\nNo missing values detected.")

        if not infinite_values_dataset.empty:
            print("\nInfinite Values Count in Numeric Columns Before Imputation:\n", infinite_values_dataset)
        else:
            print("\nNo infinite values detected.")
        print("\n" + "="*40 + "\n")

# Assuming datasets is a list of dataframes
datasets = [d1, d2, d3, d4, d5, d6, d7, d8, d9]
analyze_and_clean_data(datasets)
# Define a function for standardization
def standardize_dataframe(df):
    scaler = StandardScaler()  # Create a StandardScaler object
    
    # Extract all columns except the last one for standardization
    columns_to_standardize = df.columns[:-1]
    
    # Apply standardization to the specified columns
    df_standardized = df.copy()
    df_standardized[columns_to_standardize] = scaler.fit_transform(df_standardized[columns_to_standardize])

    return df_standardized
# Define a function for normalization
def normalize_dataframe(df):
    scaler = MinMaxScaler()  # Create a MinMaxScaler object
    
    # Extract all columns except the last one for normalization
    columns_to_normalize = df.columns[:-1]
    
    # Apply normalization to the specified columns
    df_normalized = df.copy()
    df_normalized[columns_to_normalize] = scaler.fit_transform(df_normalized[columns_to_normalize])
    
    return df_normalized
def min_max_scale_dataframe(df):
    scaler = MinMaxScaler()  
    df_scaled = df.copy()
    
    # Extract all columns except the last one for scaling
    columns_to_scale = df.columns[:-1]
    
    # Apply Min-Max scaling to the specified columns
    df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
    
    return df_scaled
datasets1 = [d1, d2, d3, d4, d5, d6, d7, d8, d9]  # List your datasets here
datasets = []
for df in datasets1:
    df = standardize_dataframe(df)
    df = normalize_dataframe(df)
    df = min_max_scale_dataframe(df)
    datasets.append(df)
# Assuming dataset contains your entire dataset, where the last column is the target variable
X = dataset.iloc[:, :-1]  # Extract features (all columns except the last one)
y = dataset.iloc[:, -1]   # Extract target variable (last column)

# Replace RandomForestClassifier with any other model if desired
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Specify a threshold for feature importance
threshold = 0.01

# Create a feature selector based on importance scores
feature_selector = SelectFromModel(clf, threshold=threshold, prefit=True)

# Transform the dataset to select important features
X_selected = feature_selector.transform(X)

# Get the indices of selected features
selected_indices = feature_selector.get_support(indices=True)

# Get the names of all features
feature_names = X.columns

# Get the names of selected features
selected_feature_names = np.array(feature_names)[selected_indices]

# Print the names of selected features
print("Names of selected features:", selected_feature_names)
