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
print(data.files)
print(data["butternut_squash"][0, 1, :, :])
# %%
# 1st dimension: neurons; 2nd dimension: type of stim; 3rd dimension: repetition; 4th: time bins
arr = np.copy(data["butternut_squash"])
n_neu, n_types, n_trials, n_tbins = arr.shape


# %%
path2save = "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/assignment_figures/with_zeros_pca_categories_plots"
for categ in data.files:
    data_clean = unLe_package.tensor_reshaping(data[categ])
    res = unLe_package.pca_wrapper(data_clean, categ, path2save)

# %%


# %%
def plot_PC1_PC2_comb(
    data1: np.array,
    data2: np.array,
    evec: np.array,
    title: str,
    explained_variance: float,
    path2save=None,
):
    PC1_1 = data1.dot(evec[:, -1])
    PC1_2 = data2.dot(evec[:, -1])
    PC2_1 = data1.dot(evec[:, -2])
    PC2_2 = data2.dot(evec[:, -2])
    plt.scatter(PC1_1, PC2_1)
    plt.scatter(PC1_2, PC2_2)
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
        plt.savefig(f"{path2save}/{title}_PC1-PC2.png")
    plt.show()


# %%
def pca_wrapper_comb(data1: np.array, data2: np.array, title: str, path2save=None):
    data_tot = np.vstack((data1, data2))
    N, D = data_tot.shape
    # compute covariance matrix
    cov_mat = unLe_package.compute_cov_mat(data_tot, 0)
    # extracts eigenvalues and eigenvectors
    evals, evec = eigh(cov_mat)
    # plot top components
    plt.scatter(np.arange(D), evals)
    if path2save is not None:
        plt.title(f"{title} eigenvalues spectrum")
        plt.savefig(f"{path2save}/{title}_spectrum.png")
    plt.show()
    # look for the rank
    rank_cov = np.linalg.matrix_rank(cov_mat)
    print(rank_cov)
    # computes the explained variance
    var_explained = unLe_package.variance_explained(evals, 2)
    print(f"variance explained by the first 2 components: {var_explained}")
    # plots the top components
    plot_PC1_PC2_comb(data1, data2, evec, title, var_explained, path2save)
    return evals, evec


# %%

path2save = "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/assignment_figures/combinations_of_cats"
data_clean1 = unLe_package.tensor_reshaping(data["butternut_squash"])
data_clean2 = unLe_package.tensor_reshaping(data["American_egret"])
# %%
pca_wrapper_comb(data_clean1, data_clean2, "aaa")
# %%
combinations = list(itertools.combinations(data.files, 2))
# %%
data.files
# %%
print(combinations)

data[combinations[1][1]]
# %%
path2save = "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/assignment_figures/combinations_of_cats"
for i in range(len(combinations)):
    couple = combinations[i]
    data_clean1 = unLe_package.tensor_reshaping(data[couple[0]])
    data_clean2 = unLe_package.tensor_reshaping(data[couple[1]])
    pca_wrapper_comb(data_clean1, data_clean2, f"{couple[0]} vs {couple[1]}", path2save)
# %%
range(len(combinations))

# %%
