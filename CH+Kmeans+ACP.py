from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

def calinski_harabasz_score(X, labels):
    n_samples, n_features = X.shape
    n_clusters = len(np.unique(labels))

    centroids = np.zeros((n_clusters, n_features))
    for k in range(n_clusters):
        centroids[k] = np.mean(X[labels == k], axis=0)

    overall_centroid = np.mean(X, axis=0)

    between_dispersion = 0
    within_dispersion = 0

    for k in range(n_clusters):
        between_dispersion += np.sum((labels == k) * np.linalg.norm(centroids[k] - overall_centroid)**2)

        within_dispersion += np.sum((labels == k) * np.linalg.norm(X[labels == k] - centroids[k])**2)

    between_dispersion /= (n_clusters - 1)
    within_dispersion /= (n_samples - n_clusters)

    ch_score = between_dispersion / within_dispersion * (n_samples - n_clusters) / (n_clusters - 1)

    return ch_score

def find_optimal_k(X, k_min, k_max):
    ch_scores = []
    kmeans = KMeans(n_clusters=k_min, random_state=0).fit(X)
    labels = kmeans.labels_
    maxk = k_min
    maxScore = calinski_harabasz_score(X, labels)
    ch_scores.append(maxScore)
    for k in range(k_min+1, k_max+1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = kmeans.labels_
        ch_score = calinski_harabasz_score(X, labels)
        ch_scores.append(ch_score)
        if ch_score > maxScore:
            maxScore = ch_score
            maxk = k
    optimal_k = maxk
    return optimal_k, ch_scores

start_time = time.time()
# Generate some sample data
iris = load_iris()
X = iris.data[:, :4]

# Normalize the data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Perform PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)
import time

start_time = time.time()
# Determine the optimal number of clusters using the Calinski-Harabasz index
k_min = 2
k_max = 10
optimal_k,ch_scores = find_optimal_k(X_pca, k_min, k_max)

# Fit the k-means algorithm with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(X_norm)

# Print the Calinski-Harabasz index score for the optimal clustering solution
labels = kmeans.labels_
ch_score = calinski_harabasz_score(X_norm, labels)
print(f"The optimal number of clusters is {optimal_k}")
print(f"The Calinski-Harabasz index score for the optimal clustering solution is {ch_score}")

end_time = time.time()

execution_time = end_time - start_time

print(f"Execution time: {execution_time} seconds")

# plot the clusters
plt.scatter(X_norm[:, 0], X_norm[:, 1], c=labels)
plt.title(f"KMeans Clustering with {optimal_k} Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Plot the Calinski-Harabasz index scores for different numbers of clusters
plt.plot(range(k_min, k_max+1), ch_scores)
plt.xlabel("Number of clusters")
plt.ylabel("Calinski-Harabasz Index")
plt.show()