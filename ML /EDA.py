import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
df = pd.read_csv(url, header=None)

# Assign column names
columns = ["ID", "Diagnosis"] + [f"feature_{i}" for i in range(30)]
df.columns = columns

# Drop ID column
df.drop('ID', axis=1, inplace=True)

# Encode Diagnosis: M=1, B=0
from sklearn.preprocessing import LabelEncoder
df['Diagnosis'] = LabelEncoder().fit_transform(df['Diagnosis'])

# -------------------------
# 1. Basic Information, Basic sturucture of dataset
# -------------------------
print("Shape of dataset:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print(df.info())
print("\nClass Distribution:\n", df['Diagnosis'].value_counts())
print("\nSummary Statistics:\n", df.describe())
#Summary Stats-> Mean, Std, Min, Max

# -------------------------
# 2. Check for Missing Values, data quality
# -------------------------
print("\nMissing Values:\n", df.isnull().sum())
print("Duplicated values: ",df.duplicated().sum())

# -------------------------
# 3. Correlation Matrix, Identifies multicollinearity
# -------------------------
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# -------------------------
# 4. Target Class Distribution,Histograms
# -------------------------
sns.countplot(data=df, x='Diagnosis')
plt.title("Distribution of Diagnosis (0 = Benign, 1 = Malignant)")
plt.show()
print(df["Diagnosis"].value_counts(normalize=True))

# -------------------------
# 5. Distribution of Features
# -------------------------
import numpy as np

# Sample 6 features for plotting
sample_features = np.random.choice(df.columns[1:], 6, replace=False)
df[sample_features].hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.show()

# -------------------------
# 6. Boxplots by Diagnosis
# -------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Diagnosis', y='feature_0')  # Example for one feature
plt.title("Boxplot of Feature_0 by Diagnosis")
plt.show()
