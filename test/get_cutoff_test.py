# %%
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir(
    "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/unLe_code"
)
from src import unLe_package
from importlib import reload
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

# %%
data_dir = "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/vectorial_repr.csv"
data = unLe_package.csv2np(data_dir)
# %%
# nans treatment
clean_data = unLe_package.nan_treatment(data, 100, 1, "median")
clean_dataT = (
    clean_data.T
)  # transpose the datasuch that my points are row vectors. Rows: points, columns: features

# # %% try with another apporach

# # As a rule of thumb, one can choose dc so that the average
# # number of neighbors is around 1 to 2% of the total number
# # of points in the data set.
# N, D = clean_dataT.shape  # N = num of datapts, D = dimensionality
# percent = 0.015  # 1-2% of the total number of points
# avg_NN = round(N * percent)  # average number of neighbors to select

# # Fit NearestNeighbors model to find k-th nearest neighbors
# knn = NearestNeighbors(n_neighbors=avg_NN, algorithm="ball_tree")
# knn.fit(clean_dataT)
# # Find distances to the k nearest neighbors
# distances, _ = knn.kneighbors(clean_dataT)
# # Compute the average distance to the k-th nearest neighbor
# dc = np.median(distances[:, -1])  # initial guess for the distance cutoff

# %%
epsilon = 0.1  # threshold of acceptance
distance_vec = pdist(clean_dataT, metric="euclidean")
distance_matrix = squareform(distance_vec)  # euclidean distance matrix
dc = unLe_package.get_cutoff(distance_matrix, 0.015, epsilon)  # final result
