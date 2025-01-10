# unLe package
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import eigh
from numba import jit
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from numba import jit

## EXPORTED FUNCTIONS
__all__ = [
    "compute_cov_mat",
    "cnt_mat",
    "double_centering",
    "war_floyd_init",
    "war_floyd",
    "ys_func_of_nans",
    "nan_treatment",
    "nan_imputation",
    "get_cutoff",
    "compute_prominence",
]

# ===================================
# GENERAL USEFUL FUNCTIONS
# ===================================
"""
pca_wrapper
Function to wrap up all the PCA process. First computes the covariance matrix, 
then extracts the eigenvalues and the associated eigenvectors and finally plots
the spectrum and the data in two components (PC1-PC2).
INPUT:
- data: np.array -> (NxD) Data matrix, requires points in the rows and features in the columns
- title: str -> for the title of the plots
- path2save: str, default=None -> if it's an argument, it should be the folder where you will store the images of the plots
OUTPUT
- evals: np.array, evec: np.array -> the eigenvalues (Dx1) and associated eigenvectors 
(DxD, evecs are column vectors) respectively, sorted in increasing order of eigenvalues
"""


def pca_wrapper(data: np.array, title: str, path2save=None):
    N, D = data.shape
    # compute covariance matrix
    cov_mat = compute_cov_mat(data, 0)
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
    var_explained = variance_explained(evals, 2)
    print(f"variance explained by the first 2 components: {var_explained}")
    # plots the top components
    plot_PC1_PC2(data, evec, title, var_explained, path2save)
    return evals, evec


# EOF

"""
pca_wrapper
Function to wrap up all the PCA process when you have to compare two categories in the shared
space of their joint PCs. First computes the covariance matrix, then extracts the eigenvalues 
and the associated eigenvectors and finally plots the spectrum and the data in two components (PC1-PC2).
INPUT:
- data1: np.array -> (NxD) Data matrix of the first category, requires points in the rows and features in the columns
- data2: np.array -> (NxD) Data matrix of the second category, requires points in the rows and features in the columns
- title: str -> for the title of the plots
OUTPUT
- evals: np.array, evec: np.array -> the eigenvalues (Dx1) and associated eigenvectors 
(DxD, evecs are column vectors) respectively, sorted in increasing order of eigenvalues
"""


def pca_wrapper_comb(data1: np.array, data2: np.array, title: str, path2save=None):
    data_tot = np.vstack((data1, data2))
    N, D = data_tot.shape
    # compute covariance matrix
    cov_mat = compute_cov_mat(data_tot, 0)
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
    var_explained = variance_explained(evals, 2)
    print(f"variance explained by the first 2 components: {var_explained}")
    # plots the top components
    plot_PC1_PC2_comb(data1, data2, evec, title, var_explained, path2save)
    return evals, evec


"""
isomap_wrapper
Function to wrap up all the isomap process. First computes the Gram matrix with Warshall-Floyd, 
then extracts the eigenvalues and the associated eigenvectors and finally plots
the spectrum and the data in two components (PC1-PC2).
INPUT:
- data: np.array -> (NxD) Data matrix, requires points in the rows and features in the columns
- dc -> distance cutoff in the warshall-floyd initialization
- title: str -> for the title of the plots
OUTPUT
- evals: np.array, evec: np.array -> the eigenvalues (Dx1) and associated eigenvectors 
(DxD, evecs are column vectors) respectively, sorted in increasing order of eigenvalues
"""


def isomap_wrapper(data: np.array, dc, title: str, path2save=None):
    G_init = war_floyd_init(
        data, dc
    )  # initializes the distance matrix for the warshall-floyd algorithm
    G = war_floyd(G_init)
    G_dcnt = double_centering(G, 1e-5)
    evals, evec = eigh(G_dcnt)
    N = len(G)
    plt.scatter(np.arange(N), evals)
    if path2save is not None:
        plt.title("isomap eigenvalues spectrum")
        plt.savefig(f"{path2save}/{title}_isomap_spectrum.png")
    plt.show()
    # computes the explained variance
    var_explained = variance_explained(evals, 2)
    plot_isomap_PC1_PC2(data, evec, evals, title, var_explained, path2save)
    return evals


"""
compute_cov_mat
Computes the covariance matrix of "data", after having centered it. Depending on whether the datapoints are the rows or the
columns it is (1/N) * data.T @ data (for rows, axis=0) or (1/N) * data @ data.T (for cols, axis=1)
Input:
- data: np.array -> NxD or DxN data matrix
- axis: int -> 0 (NxD) or 1 (DxN), indicates where the datapoints are

Output:
- cov_mat: np.array -> DxD covariance matrix

"""


def compute_cov_mat(data: np.array, axis: int) -> np.array:
    data_cnt = center_mat(
        data, axis
    )  # centers the data matrix (i.e. subtracts the average across the dimension in which the datapoints are)
    if axis == 0:
        N, _ = data_cnt.shape
        cov_mat = (1 / N) * data_cnt.T @ data_cnt
    elif axis == 1:
        _, N = data_cnt.shape
        cov_mat = (1 / N) * data_cnt @ data_cnt.T
    else:  # if the axis is neither 0 nor 1
        raise ValueError("The axis provided is neither rows nor columns")
    # end if
    return cov_mat


# EOF

"""
center_mat
Centers the data matrix, i.e. subtracts to each element the average across datapoints of 
each feature. 
Input:
- data: np.array -> the data matrix 
- axis: int -> the axis with datapoints along which is performed the average

Output:
- data_cnt: np.array -> the center data matrix
"""


def center_mat(data: np.array, axis: int) -> np.array:
    mean_data = np.mean(data, axis, keepdims=True)
    data_cnt = data - mean_data
    return data_cnt


"""
double_centering
Used for MDS. It makes both the sum over the rows and over the columns  = 0 .
The formula is G_dcnt = G - G_row_avg - G_col_avg + G_all_avg
INPUT :
- G -> a Gram matrix. It's symmetric and comes from the dot prod of my trials. 
    Its size is (#trials x #trials)

OUTPUT :
- G_dcnt -> the double-centered Gram-matrix


"""


def double_centering(G: np.ndarray, epsilon: float) -> np.ndarray:
    N = len(G)
    G_dcnt = (
        -0.5 * (np.eye(N) - 1 / N * np.ones(N)).T @ G @ (np.eye(N) - 1 / N * np.ones(N))
    )
    # to check if it's really centered
    control_double_centering(G_dcnt, epsilon)
    return G_dcnt


# EOF


"""
control_double_centering
Controls if G_dcnt is correctly double-centered up to a certain threshold epsilon.
INPUT:
- G_dcnt: np.ndarray -> double-centered Gram matrix
- epsilon: float -> threshold of acceptance

OUTPUT:
none 

"""


def control_double_centering(G_dcnt: np.ndarray, epsilon: float):
    if any(np.abs(np.sum(G_dcnt, axis=0)) > epsilon) or any(
        np.abs(np.sum(G_dcnt, axis=1)) > epsilon
    ):
        raise ValueError("the matrix isn't double-centered")


# EOF

"""
variance_explained
Computes the proportion of variance explained by eigenvectors associated to the largest d eigenvalues, s.t. in PCA 
A.T @ x_i = y_i A is the projection matrix (dxD), x_i (€ R^D) is the initial data point, 
y_i (€ R^d) is the dimensionality reduced datapoint.
Input:
- eval: np.array -> the sorted evals in increasing order
- d: int -> the reduced dimensionality

Output: 
- chi_d: float -> the proportion of variance explained by the top d evals
"""


def variance_explained(eval: np.array, d: int) -> float:
    D = len(eval)
    chi_d = np.sum(eval[-d:]) / np.sum(eval)
    return chi_d


# EOF

"""
war_floyd_init
Initializes the Gram matrix for further warshall-floyd. It first computes the distance matrix.
Then, it substitues with inifinite all the entries that are above a distance_cutoff.
pdist assumes my data matrix is (NxD), datapoints along the rows
Input:
- data: np.array -> NxD data matrix
- distance_cutoff: float -> distance above which we consider our data not connected anymore

Output:
- G: np.array -> NxN Gram matrix
"""


def war_floyd_init(data: np.array, distance_cutoff: float) -> np.array:
    dist = pdist(data, metric="euclidean")
    dist_mat = squareform(dist)
    G = np.copy(dist_mat)
    G[G >= distance_cutoff] = float("inf")
    return G


# EOF

"""
war_floyd
Warshall-Floyd algorithm, given an already initialized Gram matrix. It is a way to establish the 
distances between points by following the embedding manifold profile. If a distance is smaller by 
passing through another point, you substitute it. At the end, the Gram matrix has in its entries 
the shortest distances between the points. war-floyd_init is not embedded in this function because 
otherwise I wouldn't be able to use jit.
Input:
- G: np.array -> the Gram matrix already initialized

Output:
- G: np.array -> the updated Gram matrix 
"""


@jit(fastmath=True, nopython=True)
def war_floyd(G: np.array) -> np.array:
    N = len(G)
    G_new = np.copy(G)
    # algorithm implementation
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if G_new[i, j] > (
                    G_new[i, k] + G_new[k, j]
                ):  # if the distance between i and j is shorted when passing through k
                    G_new[i, j] = G_new[i, k] + G_new[k, j]  # update the entry
                # end if
            # end for j
    # end for i
    # end for k
    if np.any(np.isinf(G)):
        raise ValueError(
            "The distance cutoff wasn't high enough to connect all the points of the graph"
        )
    return G_new


# EOF

# ===================================
# ===================================
# CSV FILE PART - (DENSITY-PEAKS CLUSTERING & PCA)
# ===================================
# ===================================

"""
csv2np
Converts the csv file into a numpy one
Input:
- path2data: str -> path to where the data are stored

Output:
- np_array: np.array -> same data converted into a numpy array
"""


def csv2np(path2data: str):
    data_frame = pd.read_csv(path2data)
    np_array = data_frame.to_numpy()  # Convert the DataFrame to a NumPy array
    return np_array


# EOF


# ===================================
# NANs HANDLING
# ===================================

"""
ys_func_of_nans
Useful to visually inspect how much of one data dimension (y) contains nans as function
of the number of nans along another dimension (x). E.g. how many neurons (rows) are there with <= nans_num along xaxis (i.e. images, columns) 
Input :
- data: np.array -> it is the table with data
- xaxis: int -> it can be 0,1,... depending on the axis you want to use as x. 
It is the dimension over which you sum the nans in the first place
-yaxis: int -> it can be 0,1,... depending on the axis you want to use as y. 
It is the dimension you take into account to see how many usable data there are

Output :
- nans_num: np.array -> 1:1:dimensions[xaxis] a monotonically increasing function with step=1 
for further plotting
- y: np.array -> y.shape = dimensions[yaxis] a monotonically increasing function of elements 
with <= nans_num. It can be used for further plotting.
"""


def ys_func_of_nans(data: np.array, xaxis: int, yaxis: int):
    dimensions = data.shape
    nans_per_dim = np.sum(
        np.isnan(data), axis=xaxis
    )  # it is summing over the xaxis to obtain a vec s.t. len(vec) = dimensions[yaxis]
    nans_num = np.arange(1, dimensions[xaxis] + 1)
    y = np.array([np.sum(nans_per_dim <= x) for x in nans_num])
    return nans_num, y


# EOF


"""
nan_treatment
It keeps only the neurons with nans <= nans_cutoff and substitutes the
remaining nans with the mean or median firing rate.
Input:
- data: np.array -> dataset with nans
- nans_cutoff: int -> maximum number of nans you want to keep per element
- axis: int -> axis along which you perform the statistical operations
- type: str -> type of substitution you want to do (either 'mean' or 'median')

Output:
- clean_data: np.array -> dataset with nans that have been removed or substituted
"""


def nan_treatment(data: np.array, nans_cutoff: int, axis: int, type) -> np.array:
    nans_x_neu = np.sum(np.isnan(data), axis=axis)  # sum over colums
    cleaner_data = data[
        nans_x_neu <= nans_cutoff, :
    ]  # keeps only the neurons with nans <= nans_cutoff
    clean_data = nan_imputation(
        cleaner_data, type, 1
    )  # substitutes the remaining nans with the median firing rate
    np.any(np.isnan(clean_data))  # checks that no nan is left
    return clean_data


# EOF

"""
nan_imputation
Changes the nans in a dataset with the medians or means of the remaining data of the same type.
E.g. substitutes the nans in the response of a neuron with the average of other neurons' responses 
to the same image.
Input:
- data: np.array -> dataset with nans 
- type: str -> type of substitution you want to do
- axis: int -> axis along which you perform the statistical operations

Output:
- data: np.array -> dataset with nans that have been substituted
"""


def nan_imputation(data: np.array, type: str, axis: int):
    if type == "mean":
        substitutes = np.nanmean(data, axis=axis, keepdims=True)
    elif type == "median":
        substitutes = np.nanmedian(data, axis=axis, keepdims=True)
    else:
        raise ValueError("the type passed is neither 'mean' nor 'median'")
    # end if
    nan_positions = np.isnan(data)
    substitutes_array = np.broadcast_to(
        substitutes, data.shape
    )  # creating an array of the same shape of data with the vector of means
    clean_data = np.copy(data)  # copies the dataset to avoid modifying it
    clean_data[nan_positions] = substitutes_array[
        nan_positions
    ]  # substituting the actural values
    return clean_data
    # EOF


"""
plot_PC1_PC2
Plots the first two components from PCA. 
The low-dimensional data representations are given by the dot product of the datapoint with the PC 
(i.e. the eigenvector corresponding to the top ith eigenvalue).
Input:
- data: np.array -> (NxD) dataset
- evec: np.array -> (DxD) eigenvector matrix, each column is an eigenvector, sorted in increasing order according to the associated eigenvalue
- title: str -> the title of the plot
- explained_variance: float -> % of explained variance by the first two components
- path2save: str -> optional argument to save the plot

Output:
none
"""


def plot_PC1_PC2(
    data: np.array,
    evec: np.array,
    title: str,
    explained_variance: float,
    path2save=None,
):
    PC1 = data.dot(evec[:, -1])
    PC2 = data.dot(evec[:, -2])
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
        plt.savefig(f"{path2save}/{title}_PC1-PC2.png")
    plt.show()


"""
plot_isomap_PC1_PC2
Plots the first two components from isomap. 
The ith component of low-dimensional data representations are given sqrt(eval[-i])*evec[-i] as in p. 4 of Tenenbaum 2000.
(the -i idx is because our eigenvalues and eigenvectors are sorted in increasing order)
Input:
- data: np.array -> (NxD) dataset
- evec: np.array -> (DxD) eigenvector matrix, each column is an eigenvector, sorted in increasing order according to the associated eigenvalue
- eval
- title: str -> the title of the plot
- explained_variance: float -> % of explained variance by the first two components
- path2save: str -> optional argument to save the plot

Output:
none
"""


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


"""
plot_PC1_PC2_comb
Plots the first two components from PCA coming from two categories two see how 
much they are overlapping/discriminable.
The low-dimensional data representations are given by the dot product of the datapoint with the PCs 
(i.e. the eigenvector corresponding to the top ith eigenvalue).

Input:
- data1: np.array -> (NxD) dataset of the first category
- data2: np.array -> (NxD) dataset of the second category
- evec: np.array -> (DxD) eigenvector matrix, each column is an eigenvector, sorted in increasing order according to the associated eigenvalue
- title: str -> the title of the plot, should be somthing like f"{name1} & {name2}"
- explained_variance: float -> % of explained variance by the first two components
- path2save: str -> optional argument to save the plot

Output:
none
"""


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
    plt.scatter(PC1_2, PC2_2)  # plots the second category in a hold on way
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


# ===================================
# DENSITY ESTIMATION
# ===================================

"""
get_cutoff
Inspired by "As a rule of thumb, one can choose dc so that the 
average number of neighbors is around 1 to 2% of the total number 
of points in the data set." (Rodriguez & Laio 2014, p. 3 (1494))
It aims at finding the right cutoff distance for further estimation
such that on average each point will have % neighbours within the
hypersphere of radius = distance_cutoff. It uses bisection search to 
find the correct cutoff distance.
Input:
- distance_matrix: np.array -> an NxN matrix with the 
pairwise distances of the points
- percent: float -> the percentage of points you want in 
your neighbourhood (already divided by 100)
- epsilon: float -> the degree of accuracy you want for your guess

Output:
- distance_cutoff: float -> the obtained cutoff such that we'll 
have on average the required percentage of datapoints within that range
"""


def get_cutoff(distance_matrix: np.array, percent: float, epsilon: float) -> float:
    N = len(distance_matrix)  # N = num of datapts
    target = (
        round(N * percent) + 1
    )  # average number of neighbors to select, +1 because on diagonal of the distance matrix I have the point compared to itself
    high = np.max(distance_matrix)  # sets the highest distance
    low = 0  # sets the lowest possible distance
    distance_cutoff = (high + low) / 2  # sets the initial guess
    res = (
        np.sum(distance_matrix <= distance_cutoff) / N
    )  # how many points are within the distance cutoff on average
    num_trials = 0  # counter
    while abs(res - target) > epsilon:
        print(num_trials, distance_cutoff)
        num_trials += 1  # counter update
        if res > target:  # if we overshoot, the new ceiling will be our previous guess
            high = distance_cutoff
        else:
            low = distance_cutoff  # if we undershoot, the new floor will be our previous guess
        distance_cutoff = (high + low) / 2  # updates the guess
        res = np.sum(distance_matrix <= distance_cutoff) / N
    # end while
    print(f"cutoff = {round(distance_cutoff, 3)}, found in {num_trials} iterations")
    return distance_cutoff


# EOF

"""
compute_prominence
Prominence is defined as the distance from the closest point with higher density.
Input:
- sorted_distance_matrix: np.array -> NxN distance matrix sorted in descending order of density

Output:
- prominence: np.array -> Nx1 vector of prominences sorted in descending order of density
"""


def compute_prominence(sorted_distance_matrix: np.array) -> np.array:
    N = len(sorted_distance_matrix)
    prominence = np.zeros(N)  # initialization prominence matrix
    for i in range(N):
        if i == 0:
            prominence[i] = np.max(
                sorted_distance_matrix
            )  # by convention the highest density point is assigned to the highest possible distance
        else:
            prominence[i] = np.min(
                sorted_distance_matrix[0:i, i]
            )  # takes the minimum distance from the point i (column) among the points with higher density (0:i-1 rows)
        # end if
    # end for
    return prominence


# EOF


# ===================================
# ===================================
# NPZ FILE PART - PCA
# ===================================
# ===================================

"""
tensor_reshaping
It takes a field from the .npz file, takes off the non-existent trials, averages across 
trials of the same image and reshapes the tensor such that it is 2D. 1st dimension neurons 
and types (grouped consecutively by neurons), 2nd dimension time bins. By type I mean different
instances of the same object (e.g. a dog running/ eating/ jumping)
INPUT:
- data: np.array -> (num_neurons x num_types x num_trials x time_bins)

OUTPUT:
- clean_arr: np.array -> (num_neurons*num_types x time_bins)
"""


def tensor_reshaping(data: np.array) -> np.array:
    # nan_mask = np.all(
    #     data == 0, axis=3
    # )  # finds the elements that along the 3rd axis are all 0s (because not recorded)
    # nan_arr = np.copy(data)  # array with nans
    # nan_arr[nan_mask, :] = np.nan  # marks all the responses with all 0s with all nans
    mean_arr = np.mean(
        data, axis=2
    )  # computes the average across the trials dimension (so it becomes a 3D array)
    flattened_arr = mean_arr.reshape(
        -1, data.shape[3]
    )  # reshapes the 3D array into a 2D, neurons and types of images (grouped by neurons) on the rows and time bins in the columns
    clean_arr = flattened_arr[
        ~np.all(np.isnan(flattened_arr), axis=1), :
    ]  # it is preserving only the rows without nans
    if np.any(np.isnan(clean_arr)):
        raise ValueError("NaNs still present")
    return clean_arr


# EOF
