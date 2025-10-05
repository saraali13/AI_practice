import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from math import log2

# Load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split small sample
train = df.sample(10, random_state=1)  # tiny sample
print(train)

# Function to calculate entropy
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))

# Example: Calculate entropy of target
print("\nEntropy of dataset:", entropy(train['target']))

# Example: Calculate Information Gain for one feature
def info_gain(data, feature, target):
    total_entropy = entropy(data[target])
    vals, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data[data[feature]==vals[i]][target]) for i in range(len(vals))])
    return total_entropy - weighted_entropy

# Just test on 1 feature
print("Information Gain for feature 0 (first column):", info_gain(train, iris.feature_names[0], 'target'))
