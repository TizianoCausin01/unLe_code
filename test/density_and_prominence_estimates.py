# %%
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir(
    "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/unLe_code"
)
from src import unLe_package
from importlib import reload
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch

# %% REAL DATA
data_dir = "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/vectorial_repr.csv"
data = unLe_package.csv2np(data_dir)

# nans treatment
clean_data = unLe_package.nan_treatment(data, 500, 1, "median")
clean_dataT = (
    clean_data.T
)  # transpose the datasuch that my points are row vectors. Rows: points, columns: features

# %% GPT GENERATED CLUSTER
# Set random seed for reproducibility
np.random.seed(42)
# Generate data for two clusters
N_test = 300
std = 0.1
# Cluster 1: 100 points centered around (2, 2)
cluster_1 = std * np.random.randn(N_test, 2) + np.array([2, 2])
# Cluster 2: 100 points centered around (8, 8)
cluster_2 = std * np.random.randn(N_test, 2) + np.array([8, 8])
# Cluster 3: 100 points centered around (4, 4)
cluster_3 = std * np.random.randn(N_test, 2) + np.array([4, 4])
# Combine both clusters to form the dataset
data = np.vstack([cluster_1, cluster_2, cluster_3])
clean_dataT = data
# Plot the dataset to visualize the two clusters
plt.scatter(
    data[:, 0],
    data[:, 1],
    c=["blue"] * N_test + ["red"] * N_test + ["green"] * N_test,
    label="Data Points",
)
plt.title("Dataset with Two Clear Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# %% FIRST STEP OF DENSITY-PEAKS CLUSTERING
epsilon = 0.1  # threshold of acceptance
distance_vec = pdist(clean_dataT, metric="euclidean")
distance_matrix = squareform(distance_vec)  # euclidean distance matrix
dc = unLe_package.get_cutoff(distance_matrix, 0.02, epsilon)  # final result
density_estimates = np.sum(distance_matrix <= dc, axis=1)
sort_idx = np.argsort(density_estimates)[::-1]  # gives the indeces in decreasing order
sorted_distance_matrix = distance_matrix[sort_idx][:, sort_idx]
sorted_density_estimates = density_estimates[sort_idx]
prominence = unLe_package.compute_prominence(sorted_distance_matrix)
plt.scatter(sorted_density_estimates, prominence)
plt.xlabel("density")
plt.ylabel("prominence")
plt.show()

# %%
# Linkage method can be 'single', 'complete', 'average', or 'ward' (Ward minimizes variance)
linkage_matrix = sch.linkage(distance_matrix, method="ward")
# Step 3: Plot the dendrogram to visualize the hierarchical clustering
sch.dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()
