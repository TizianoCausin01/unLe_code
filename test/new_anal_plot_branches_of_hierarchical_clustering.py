# STEPS OF PCA
# %% load packages
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import eigh
import os
from src import unLe_package
from importlib import reload
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch

os.chdir(
    "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/unLe_code"
)
# %%
reload(unLe_package)
# %%
data_dir = "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/vectorial_repr.csv"
data = unLe_package.csv2np(data_dir)

# nans treatment
clean_data = unLe_package.nan_treatment(data, 500, 1, "median")
clean_dataT = (
    clean_data.T
)  # transpose the datasuch that my points are row vectors. Rows: points, columns: features
N, D = clean_dataT.shape
# %%
# compute covariance matrix
cov_mat = unLe_package.compute_cov_mat(clean_dataT, 0)

# %% extracts eigenvalues and eigenvectors
eval, evec = eigh(cov_mat)

# %% plot top components
plt.scatter(np.arange(D), eval)
plt.show()
# %% look for the rank
rank_cov = np.linalg.matrix_rank(cov_mat)
print(rank_cov)

# %%
var_explained = unLe_package.variance_explained(eval, 2)
print(var_explained)
# %%
unLe_package.plot_PC1_PC2(clean_dataT, evec)
# %%
evals, evec = unLe_package.pca_wrapper(
    clean_dataT, "all"
)  # wraps up all the steps above in a function

# %%
distance_vec = pdist(clean_dataT, metric="euclidean")
distance_matrix = squareform(distance_vec)  # euclidean distance matrix

# Linkage method can be 'single', 'complete', 'average', or 'ward' (Ward minimizes variance)
linkage_matrix = sch.linkage(distance_vec, method="ward")
# Step 3: Plot the dendrogram to visualize the hierarchical clustering
sch.dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()
# %%
cluster_labels = sch.fcluster(linkage_matrix, t=3, criterion="maxclust")
# %%
data1 = clean_dataT[cluster_labels == 1, :]
data2 = clean_dataT[cluster_labels == 2, :]
data3 = clean_dataT[cluster_labels == 3, :]

# %%
PC1_1 = data1.dot(evec[:, -1])
PC2_1 = data1.dot(evec[:, -2])
PC1_2 = data2.dot(evec[:, -1])
PC2_2 = data2.dot(evec[:, -2])
PC1_3 = data3.dot(evec[:, -1])
PC2_3 = data3.dot(evec[:, -2])

plt.scatter(PC1_1, PC2_1, label="Data 1", color="blue")  # First category in blue
plt.scatter(PC1_2, PC2_2, label="Data 2", color="red")  # Second category in red
plt.scatter(PC1_3, PC2_3, label="Data 3", color="green")  # Second category in red
plt.xlabel("PC1")
plt.ylabel("PC2")
# plt.title(f"{title} PC1-PC2")
# box_str = f"explained variance: {round(explained_variance,2)}"
# plt.text(
#     0.95,
#     0.95,
#     box_str,
#     transform=plt.gca().transAxes,
#     fontsize=9,
#     bbox=dict(facecolor="red", alpha=0.5),
#     ha="right",
#     va="top",
# )

# plt.savefig(f"{path2save}/{title}_PC1-PC2.png")
plt.legend()
plt.show()


# %%
