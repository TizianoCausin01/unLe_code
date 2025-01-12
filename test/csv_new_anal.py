# %% load packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import os
from src import unLe_package
from importlib import reload
import itertools

os.chdir(
    "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/unLe_code"
)
import pandas as pd

# %% Read the CSV file into a DataFrame
data_dir = "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/vectorial_repr.csv"
data = pd.read_csv(data_dir)
data_npz_dir = (
    "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/cache_array.npz"
)
data_npz = np.load(data_npz_dir)  # loads the other dataset
combinations = list(
    itertools.combinations(data_npz.files, 2)
)  # such that all the categories will be called properly


# %%
def pca_wrapper_comb_csv(data1: np.array, data2: np.array, title: str, path2save=None):
    data_tot = np.hstack(
        (data1, data2)
    )  # data are still neurons x pics, so we stack them horizontally
    data_tot_clean = unLe_package.nan_treatment(
        data_tot, 10, 1, "median"
    )  # removes neurons (rows) with too many nans
    data_tot_clean_T = data_tot_clean.T  # finally transposes the data-matrix
    N, D = data_tot_clean_T.shape
    # compute covariance matrix
    cov_mat = unLe_package.compute_cov_mat(data_tot_clean_T, 0)
    # extracts eigenvalues and eigenvectors
    evals, evec = eigh(cov_mat)
    # look for the rank
    rank_cov = np.linalg.matrix_rank(cov_mat)
    print(rank_cov)
    # computes the explained variance
    var_explained = unLe_package.variance_explained(evals, 2)
    print(f"variance explained by the first 2 components: {var_explained}")
    # plots the top components
    unLe_package.plot_PC1_PC2_comb(
        data_tot_clean_T[0:16, :],
        data_tot_clean_T[16:32, :],
        evec,
        title,
        var_explained,
        path2save,
    )
    return evals, evec


# %%
path2save = "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/assignment_figures/combinations_of_cats_csv"
for combo in combinations:
    data1_filt = data.filter(like=combo[0])
    data1 = data1_filt.to_numpy()
    # data1_clean = unLe_package.nan_treatment(data1, 3, 1, "median")
    data2_filt = data.filter(like=combo[1])
    data2 = data2_filt.to_numpy()
    pca_wrapper_comb_csv(
        data1, data2, f"{combo[0]} vs {combo[1]}", path2save
    )  # here the data are still neurons x pics

# %%
data1_filt = data.filter(like=combo[0])
data2_filt = data.filter(like=combo[1])
print(data1_filt.shape, data2_filt.shape)
data_tot = np.hstack((data1, data2))
print(data_tot.shape)
data_tot_clean = unLe_package.nan_treatment(data_tot, 10, 1, "median")
print(data_tot_clean.shape)
print(np.any(np.isnan(data_tot_clean)))
# %%
reload(unLe_package)
# %%
