# unLe package
print("aa")
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import eigh
from numba import jit

## EXPORTED FUNCTIONS
__all__ = ["double_centering"]

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
