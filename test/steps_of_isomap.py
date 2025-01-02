# %%
# imports packages
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import eigh
from numba import jit
import os
from scipy.spatial.distance import pdist, squareform, euclidean
from src import unLe_package
from importlib import reload

os.chdir(
    "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/unLe_code"
)


plt.ion()

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
dc = 70
G_init = unLe_package.war_floyd_init(clean_dataT, dc)

# %%
G = unLe_package.war_floyd(G_init)

# %%
np.all(np.equal(G_init, G))
np.any(np.isinf(G))
# %%
print("G id:", id(G))
print("G_init id:", id(G_init))
# %%
print(G)
# %% double centering
G_dcnt = unLe_package.double_centering(G, 1e-5)

# %%
eval_dG, evec = eigh(G_dcnt)
plt.scatter(np.arange(N), eval_dG)
plt.show()
print(np.linalg.matrix_rank(G_dcnt))

# %%
unLe_package.variance_explained(eval_dG, 600)
