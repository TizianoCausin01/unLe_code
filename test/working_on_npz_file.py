# %% load packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import os
from src import unLe_package
from importlib import reload

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
print(data.files)
print(data["butternut_squash"][0, 1, :, :])
# %%
# 1st dimension: neurons; 2nd dimension: type of stim; 3rd dimension: repetition; 4th: time bins
arr = np.copy(data["butternut_squash"])
n_neu, n_types, n_trials, n_tbins = arr.shape

# %%
nan_mask = np.all(
    arr == 0, axis=3
)  # finds the elements that along the 3rd axis are all 0s (because not recorded)
nan_arr = np.copy(arr)
nan_arr[nan_mask, :] = np.nan  # marks all the responses with all 0s with all nans
mean_arr = np.nanmean(
    nan_arr, axis=2
)  # computes the average (excluding the nans) across the trials dimension (so it becomes a 3D array)
flattened_arr = mean_arr.reshape(
    -1, arr.shape[3]
)  # reshapes the 3D array into a 2D, neurons and types of images (grouped by neurons) on the rows and time bins in the columns
clean_arr = flattened_arr[
    ~np.all(np.isnan(flattened_arr), axis=1), :
]  # it is preserving only the rows without nans
# %%
unLe_package.tensor_reshaping(data["Madagascar_cat"])
# %%
