import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

data1 = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5, 6],
    'Name': ['Ali', 'Sara', 'John', 'Ali', np.nan, 'Sara'],
    'Age': [25, 30, np.nan, 25, 40, 'thirty'],
    'Salary': [50000, 60000, 55000, None, 70000, 60000],
    'Department': ['HR', 'Finance', 'IT', 'HR', 'Finance', None],
    'Joining_Date': ['2020-01-01', '2019-05-10', 'not available', '2020-01-01', '2021-03-15', '2019-05-10']
})

data2 = pd.DataFrame({
    'ID': [7, 8, 9],
    'Name': ['Ahsan', 'Mina', 'Ali'],
    'Age': [28, 26, 32],
    'Salary': [62000, 58000, 65000],
    'Department': ['IT', 'HR', 'Finance'],
    'Joining_Date': ['2022-04-20', '2023-06-10', '2021-08-05']
})

combined_append = data1.append(data2, ignore_index=True)
print("Append:\n", combined_append)
combined_concat = pd.concat([data1, data2], ignore_index=True)
print("Concat:\n", combined_concat)
dept_info = pd.DataFrame({
    'Department': ['HR', 'Finance', 'IT'],
    'Manager': ['Asma', 'Bilal', 'Kiran']
})

merged_df = pd.merge(combined_concat, dept_info, on='Department', how='left')
print("Merge:\n", merged_df.head())

df.tail()
df.columns#col names
df.count()#non null vals for each cols
df.drop("C1",axis=1)#remove col
df.insert(0,"c2",COL2)#adding a col
r4={"c1":2,"c2":4}
df.append(r4,ignore_index=True)
df.reset_index(drop=False,inplace=True)
df.set_index("c1") #index=c1
df.sort_values("c2")# data is sorted acc to c2
df["c2"][`df["c2"]==0]="male"# change values in a col male whereever it is 0
df["c1"] # or df.pop("c1")
df[["c1","c2"]]
