import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd 

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
    if(n_clusters == 1):
        return 0

    between_dispersion /= (n_clusters - 1)
    if within_dispersion == 0:
        ch_score = 0
    else:
        within_dispersion /= (n_samples - n_clusters)
        ch_score = between_dispersion / within_dispersion * (n_samples - n_clusters) / (n_clusters - 1)
    return ch_score

def agglomerative_clustering(X):
    # Placer chaque objet dans son propre cluster:
    clusters = [[i] for i in range(len(X))]
    centers = X.copy()
    ch_scores = {}
    
    # Tant que il y a des objets a agglomerer
    while len(clusters) > 1:
        print(len(clusters))
        print(clusters)
        # Calculer une liste des distances entre les clusters et la trier dans l’ordre croissant
        distances = []
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                c1 = clusters[i]
                c2 = clusters[j]
                center1 = np.mean(centers[c1], axis=0)
                center2 = np.mean(centers[c2], axis=0)
                distance = np.linalg.norm(center1 - center2)
                distances.append((i, j, distance))
        
        # Agglomerer les objets ayant la distance minimale.
        i, j, _ = sorted(distances, key=lambda x: x[2])[0]
        c1 = clusters[i]
        c2 = clusters[j]
        
        # Calculer le centre du nouveau cluster
        merged_cluster = c1 + c2
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(merged_cluster)
        center = np.mean(centers[merged_cluster], axis=0)
        centers[merged_cluster] = center
        # Calculate calinski_harabasz_score for each cluster
        labels = np.zeros(len(X))
        for i, c in enumerate(clusters):
            for j in c:
                labels[j] = i
        ch_score = calinski_harabasz_score(X, labels)
        ch_scores[len(clusters)]=ch_score
        print(ch_score)


    # Find the number of clusters with the highest calinski_harabasz_score
    best_n_clusters = max(ch_scores, key=ch_scores.get)
    # visulaize the calinski_harabasz_score for each number of clusters
    plt.plot(list(ch_scores.keys()), list(ch_scores.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("Calinski-Harabasz Score")
    plt.show()
    return best_n_clusters

# load a dataset 
#df_segmentation = pd.read_csv('cereal.csv', index_col = 0)
# Select only columns with numeric data types
#df_numeric = df_segmentation.select_dtypes(include=['float64', 'int64'])
# we will have only two columns
#df_numeric = df_numeric.iloc[:, 5:7]
# Normalize the data
scaler = StandardScaler()
#X= scaler.fit_transform(df_numeric)
X = np.random.rand(50, 2)
print(X)
X= scaler.fit_transform(X)
clusters = agglomerative_clustering(X)
kmeans = KMeans(clusters, random_state=42).fit(X)
labels = kmeans.labels_

# plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title(f"KMeans Clustering with {clusters} Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage

# linkage method : ward, single, complete, average, weighted
linkage_matrix = linkage(X, method='ward')

# plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
