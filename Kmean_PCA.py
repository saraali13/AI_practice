from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3)
model=kmeans.fit(X_scaled)
pred=model.predict(x_test)

labels = kmeans.labels_

inertias = []
for k in range(1,10):
    km = KMeans(n_cluster=k,init="k-means++", max_iter=300,n_init=10,random_state=0)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(range(1,11),inertias)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca.get_covariance()
exp_variance=pca.explained_varience_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.plot(range(1, len(features)+1), cumulative_variance, marker='o', linestyle='--')

n_components = np.argmax(cumulative_variance >= 0.9) + 1
pca_final = PCA(n_components=n_components)
X_reduced = pca_final.fit_transform(X_scaled)

#between 1st 2 pricipal comp
sns.scatterplot(
    x='PC1',
    y='PC2',
    data=df_pca
)
