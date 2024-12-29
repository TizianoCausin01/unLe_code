# unLe package
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import eigh
from numba import jit
import pandas as pd

## EXPORTED FUNCTIONS
__all__ = ["double_centering", "ys_func_of_nans"]


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

    G_col_avg = np.mean(
        G, axis=0, keepdims=True
    )  # averages along the columns and makes it a row vector to do the columnwise subtracttion
    G_row_avg = np.mean(
        G, axis=1, keepdims=True
    )  # averages along the rows and makes it a row vector to do the rowwise subtracttion
    G_all_avg = np.mean(G)
    G_dcnt = G - G_row_avg - G_col_avg + G_all_avg
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


# density-peaks clustering

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
