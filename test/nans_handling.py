# %%
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir(
    "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/unLe_code"
)
from src import unLe_package

# %%
data_dir = "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/vectorial_repr.csv"
data = unLe_package.csv2np(data_dir)

# %%
# nans treatment
nans_cutoff = 100


def nan_treatment(data: np.array, nans_cutoff: int, axis: int, type) -> np.array:
    nans_x_neu = np.sum(np.isnan(data), axis=axis)  # sum over colums
    cleaner_data = data[
        nans_x_neu <= nans_cutoff, :
    ]  # keeps only the neurons with nans <= nans_cutoff
    print(data.shape)  # old shape
    print(cleaner_data.shape)  # inspects the new shape of the array
    clean_data = unLe_package.nan_imputation(
        cleaner_data, type, 1
    )  # substitutes the remaining nans with the median firing rate
    np.any(np.isnan(clean_data))  # checks that no nan is left
    return clean_data


# %%
nan_treatment(data, nans_cutoff, 1, "median")
# %%
nans_x_neu = np.sum(np.isnan(data), axis=1)  # sum over colums
cleaner_data = data[
    nans_x_neu <= nans_cutoff, :
]  # keeps only the neurons with nans <= nans_cutoff
print(data.shape)  # old shape
print(cleaner_data.shape)  # inspects the new shape of the array
clean_data = unLe_package.nan_imputation(
    cleaner_data, "median", 1
)  # substitutes the remaining nans with the median firing rate
np.any(np.isnan(clean_data))  # checks that no nan is left
# %%
# density estimation as in paper
# clustering as in paper
