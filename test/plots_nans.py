# %%
import os
import matplotlib.pyplot as plt
import numpy as np

os.chdir(
    "/Users/tizianocausin/Desktop/backUp20240609/sissa/unsupervised_learning/unLe_code"
)
from src import unLe_package

# %%
# downloading data
data_pop = "/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/unLe_data/vectorial_repr.csv"
data = unLe_package.csv2np(data_pop)
# %%
# axis=0 -> neurons, axis=1 -> images
# how many images have <= nan neurons?
x_img, y_img = unLe_package.ys_func_of_nans(data, 0, 1)
plt.plot(x_img, y_img)
plt.title("# of img with <= than x nans")
plt.xlabel("# of nan neurons")
plt.ylabel("# of images")
plt.show()

# how many neurons have <= nan images?
x_neu, y_neu = unLe_package.ys_func_of_nans(data, 1, 0)
plt.plot(x_neu, y_neu)
plt.title("# of neurons with <= than x nans")
plt.xlabel("# of images")
plt.ylabel("# of neurons")
plt.show()
