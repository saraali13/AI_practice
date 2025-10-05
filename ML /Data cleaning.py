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
from sklearn.impute import SimpeImputer
from spicy.stats import zscore

df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6],
    'Name': ['Ali', 'Sara', 'John', 'Ali', np.nan, 'Sara'],
    'Age': [25, 30, np.nan, 25, 40, 'thirty'],
    'Salary': [50000, 60000, 55000, None, 70000, 60000],
    'Department': ['HR', 'Finance', 'IT', 'HR', 'Finance', None],
    'Joining_Date': ['2020-01-01', '2019-05-10', 'not available', '2020-01-01', '2021-03-15', '2019-05-10']
})

#handling missing vals

print(df.isnull().sum())

df=df.dropna()#row
df=df.dropna(axis=1)#col

df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])
df['Age'] = df['Age'].ffill(inplace=True)
df['Age'] = df['Age'].bfill(inplace=True)
df['Age'] = df['Age'].fillna(df['Age'].median())

imp_mean=SimpeImputer(strategy="mean")
df[["Age","Score"]]=imp_mean.fit_transform(df[["Age","Score"]])

#handling dup vals

print("Before:", df.shape)
dup=df.duplicated()
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)
df=df.drop_duplicates(subset=["Age","Score"]) #drop dups based on these cols only

#Feature Encoding

le = LabelEncoder()
df['Dept_Label'] = le.fit_transform(df['Department'])#like low<med<high

#one hot shot
df = pd.get_dummies(df, columns=['Department'], drop_first=True) #no order category Like cities
print(df.head())

#Ordinal encoding
quality_mapping={"hello":1,"HI":2,"Hey":3}
df['Quality_Encoded'] = df['Quality'].map(quality_mapping)

print("\nAfter Ordinal Encoding:")
print(df)

#Feature Scaling

scaler_minmax = MinMaxScaler()# ragne (0,1), Non -ve
encoded_df['Salary_MinMax'] = scaler_minmax.fit_transform(encoded_df[['Salary']])

scaler_std = StandardScaler()#transform data to have mean=0 and std dev=1
encoded_df['Age_Standard'] = scaler_std.fit_transform(encoded_df[['Age']])

#feature selection

best_features = SelectKBest(score_func=chi2, k=2)
fit = best_features.fit(X.abs(), y)
print("Scores:", fit.scores_)

#imbalanced data

ros = RandomOverSampler(random_state=42)
X_over, y_over = ros.fit_resample(X, y)

rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X, y)

#Outliers detection
df["col4"]=zscore(df["c4"])
outliers=df[df["col4"].abs()>3]
print(outliers)

#else use boxplot, scatterplot, historgram

#outlier handling

#drop outliers
df=df[(df["c1"]>=lower_bound) & (df["c1"]>=upper_bound)] 

#clip data extremes
lower=df["c1"].quantile(0.05)
upper=df["c1"].quantile(0.95)
df["c2"]=df["c1"].clop(lower,upper) 

#Log transform
df["log_c1"]=np.log(df["c1"])

