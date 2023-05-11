# Find-the-number-of-clusters-using-Calinski-Harabasz-idex
This repository contains two Python scripts for clustering analysis using different methods. 
Both scripts use the Calinski-Harabasz index to find the best number of clusters to use in k-means.
## The Calinski-Harabasz index:
The Calinski-Harabasz index is a measure of cluster separation that compares the ratio of between-cluster dispersion and within-cluster dispersion. 
A higher index value indicates better clustering.
### The first script (CH+Kmeans+ACP.py) 
The script use PCA to reduce the dimensionality of the data. 
It then determines the optimal number of clusters using the Calinski-Harabasz index and fits the K-means algorithm with the optimal number of clusters.
### The second script (CAH+CH.py) 
Uses agglomerative hierarchical clustering (AHC) to find the best number of clusters to use in k-means using the Calinski-Harabasz index score for each clustering solution. 
## Prerequisites
The scripts are written in Python 3 and require the following packages: numpy, matplotlib, scikit-learn, and pandas. 
