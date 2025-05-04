import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Iris.csv")

# Use only feature columns (ignore Id and Species)
x = df.iloc[:, 1:5].values  # SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm

# Standardize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(x)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 classes in Iris
kmeans.fit(data_scaled)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the clustering result (first 2 features)
plt.figure(figsize=(8, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis', label='Data Points')
plt.scatter(centers[:, 0], centers[:, 1], marker="*", s=200, color='red', label='Centroids')
plt.xlabel("Standardized Sepal Length")
plt.ylabel("Standardized Sepal Width")
plt.title("KMeans Clustering on Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
