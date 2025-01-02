# STEPS OF PCA
# %% load packages
import matplotlib.pyplot as plt
import math
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
var_explained = unLe_package.variance_explained(eval, 10)
print(var_explained)

# %%
