import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6],
    'Name': ['Ali', 'Sara', 'John', 'Ali', np.nan, 'Sara'],
    'Age': [25, 30, np.nan, 25, 40, 'thirty'],
    'Salary': [50000, 60000, 55000, None, 70000, 60000],
    'Department': ['HR', 'Finance', 'IT', 'HR', 'Finance', None],
    'Joining_Date': ['2020-01-01', '2019-05-10', 'not available', '2020-01-01', '2021-03-15', '2019-05-10']
})

print(df.isnull().sum())

df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Age'] = df['Age'].fillna(df['Age'].median())

print("Before:", df.shape)

df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

le = LabelEncoder()
df['Dept_Label'] = le.fit_transform(df['Department'])

#one hot shot
df = pd.get_dummies(df, columns=['Department'], drop_first=True)
print(df.head())
# Replace categorical values with numeric ranks
df['Quality_Encoded'] = df['Quality'].map(quality_mapping)

print("\nAfter Ordinal Encoding:")
print(df)

#Scaling
scaler_minmax = MinMaxScaler()
encoded_df['Salary_MinMax'] = scaler_minmax.fit_transform(encoded_df[['Salary']])
scaler_std = StandardScaler()
encoded_df['Age_Standard'] = scaler_std.fit_transform(encoded_df[['Age']])

X = encoded_df[['Age', 'Salary_MinMax', 'Age_Standard']]
y = np.random.choice([0, 1], size=len(encoded_df))  # dummy target

#feature selection
best_features = SelectKBest(score_func=chi2, k=2)
fit = best_features.fit(X.abs(), y)
print("Scores:", fit.scores_)

#imbalanced data
# Example dataset (90% class 0, 10% class 1)
X = pd.DataFrame({
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100)
})

y = np.array([0]*90 + [1]*10)  # 90 zeros and 10 ones

print("Before balancing:", Counter(y))
# Create object
ros = RandomOverSampler(random_state=42)

# Fit and resample
X_over, y_over = ros.fit_resample(X, y)

print("After RandomOverSampler:", Counter(y_over))
# Create object
rus = RandomUnderSampler(random_state=42)

# Fit and resample
X_under, y_under = rus.fit_resample(X, y)

print("After RandomUnderSampler:", Counter(y_under))
