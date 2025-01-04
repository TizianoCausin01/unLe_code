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

path2save = "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/assignment_figures/pca_categories_plots"
for categ in data.files:
    data_clean = unLe_package.tensor_reshaping(data[categ])
    res = unLe_package.pca_wrapper(data_clean, categ, path2save)
