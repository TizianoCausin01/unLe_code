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
# %%
reload(unLe_package)
# %%
data_dir = (
    "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/cache_array.npz"
)
data = np.load(data_dir)

# %%
# 1st dimension: neurons; 2nd dimension: type of stim; 3rd dimension: repetition; 4th: time bins
arr = np.copy(data["butternut_squash"])
n_neu, n_types, n_trials, n_tbins = arr.shape


# %%
def tensor_reshaping_traj(data: np.array) -> np.array:
    mean_arr = np.mean(
        data, axis=2
    )  # computes the average across the trials dimension (so it becomes a 3D array)
    flattened_arr = np.mean(
        mean_arr, axis=1
    )  # averages also across different examples of the same category
    return flattened_arr


# %%

data_tot = None
for categ in data.files:
    data_clean = (
        tensor_reshaping_traj(data[categ])
    ).T  # the result is a 2d array (time x neurons)
    if data_tot is None:
        data_tot = data_clean
    else:
        data_tot = np.vstack((data_tot, data_clean))
print(data_tot.shape)


# %%
def pca_quick_wrapper(data: np.array):
    # compute covariance matrix
    cov_mat = unLe_package.compute_cov_mat(data, 0)
    # extracts eigenvalues and eigenvectors
    evals, evec = eigh(cov_mat)
    return evec


# %%
evec = pca_quick_wrapper(
    data_tot
)  # computes the eigenvectors on the total data (categories, time)
# %%
for categ in data.files:
    data_clean = (tensor_reshaping_traj(data[categ])).T
    compon1 = data_clean @ evec[:, -1]
    compon2 = data_clean @ evec[:, -2]
    plt.plot(compon1, compon2)
plt.show


# %%
