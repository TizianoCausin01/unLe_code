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
expl_var = unLe_package.variance_explained(eval_dG, 2)


# %%
def plot_isomap_PC1_PC2(
    data: np.array,
    evec: np.array,
    eval: np.array,
    title: str,
    explained_variance: float,
    path2save=None,
):
    PC1 = np.sqrt(eval[-1]) * evec[:, -1]
    PC2 = np.sqrt(eval[-2]) * evec[:, -2]
    plt.scatter(PC1, PC2)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{title} PC1-PC2")
    box_str = f"explained variance: {round(explained_variance,2)}"
    plt.text(
        0.95,
        0.95,
        box_str,
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(facecolor="red", alpha=0.5),
        ha="right",
        va="top",
    )
    if path2save is not None:
        plt.savefig(f"{path2save}/{title}_isomap_PC1-PC2.png")
    plt.show()


# %%
unLe_package.plot_isomap_PC1_PC2(clean_dataT, evec, eval_dG, "all", expl_var)


# %%
def isomap_wrapper(data: np.array, dc, title: str, path2save: str):
    G_init = unLe_package.war_floyd_init(clean_dataT, dc)
    G = unLe_package.war_floyd(G_init)
    G_dcnt = unLe_package.double_centering(G, 1e-5)
    eval_dG, evec = eigh(G_dcnt)
    plt.scatter(np.arange(N), eval_dG)
    plt.title("isomap eigenvalues spectrum")
    plt.savefig(f"{path2save}/{title}_isomap_spectrum.png")
    plt.show()
    # computes the explained variance
    var_explained = unLe_package.variance_explained(eval_dG, 2)


# %%
