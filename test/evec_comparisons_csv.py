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

# %%
data_dir = "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/vectorial_repr.csv"
data_df = pd.read_csv(data_dir)
data_npz_dir = (
    "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/cache_array.npz"
)
data_npz = np.load(data_npz_dir)  # loads the other dataset
combinations = list(
    itertools.combinations(data_npz.files, 2)
)  # such that all the categories will be called properly


# %%
def pca_quick_wrapper(data: np.array):
    N, D = data.shape
    # compute covariance matrix
    cov_mat = unLe_package.compute_cov_mat(data, 0)
    # extracts eigenvalues and eigenvectors
    evals, evec = eigh(cov_mat)
    return evec[:, -1]


# %%


# %%
def get_indeces(data, cat):
    matching_columns = data.filter(like=cat).columns
    column_indices = [data.columns.get_loc(col) for col in matching_columns]
    return column_indices


# %%

data_clean_np = (unLe_package.nan_treatment(data_df.to_numpy(), 100, 1, "median")).T
# %%
evecs_list = None
for cat in data_npz.files:  # iterates across all categories
    idx = get_indeces(data_df, cat)
    current_cat = data_clean_np[idx, :]
    evec = pca_quick_wrapper(current_cat)
    if evecs_list is None:
        evecs_list = evec
    else:
        evecs_list = np.vstack((evecs_list, evec))

evecs_list = evecs_list.T  # to have them all as column vectors

# %%
for i in range(evecs_list.shape[1]):
    plt.plot(evecs_list[:, i])
plt.show()
# %%

# %%
