#%%
# imports packages 
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.linalg import eigh
from numba import jit
plt.ion()
#%%
# creates a manifold in D = 10 and d = 2
N = 1000
D = 10
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 20) 
x,y = np.meshgrid(x,y)

# Compute z based on the sine surface function
z = np.sin(np.sqrt(x**2 + y**2))
#%%
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(x,y,z)
plt.show()
#%%
print(x.shape, y.shape, z.shape)
# Combine into a (100, 3) matrix of 3D vectors
X  = np.vstack((x.flatten(), y.flatten(), z.flatten(), np.zeros((D - 3,N))))
C = X @ X.T 
print(C.shape)
eval, evec = eigh(C, eigvals=(0,9))
print("rank(C) =", np.linalg.matrix_rank(C))

#%% defines euclidean dist
@jit(fastmath=True, nopython=True)
def euclidean_distance(x1, x2):
    """ computes the euclidean distance between two N-dimensional vectors """
    diff_vec = (x1 - x2)**2
    dist = math.sqrt(np.sum(diff_vec))
    return dist
# EOF

#%% warshall-floyd algorithm
# G initialization
@jit(fastmath=True, nopython=True)
def war_floyd(N, dc, X):
    G = np.zeros((N,N))
    for i in np.arange(N): # rows
      for j in np.arange(N): # cols
          dist = euclidean_distance(X[:, i], X[:, j])
          if dist < dc:
            G[i, j] = dist
          else:
            G[i, j] = float('inf')
          #end if
    #end for cols
    #end for rows
    if not(np.array_equal(G,G.T)):
      ValueError("the Gram matrix is not symmetric")

    # algorithm implementation

    for k in range(N):
      for i in range(N):
        for j in range(N):
          if G[i, j] > G[i, k] + G[k, j]:
            G[i, j] = G[i, k] + G[k, j]
          #end if 
        #end for j
    #end for i
    #end for k
    return G
#%%
dc = .5 # distance cutoff
G = war_floyd(N, dc, X)

#%% double centering
G_col = np.mean(G, axis=0, keepdims=True) # averages along the columns and makes it a row vector to do the columnwise subtracttion 
G_row = np.mean(G, axis=1, keepdims=True) # averages along the rows and makes it a row vector to do the rowwise subtracttion
G_all = np.mean(G)
G_cnt = G - G_row - G_col + G_all 
#%%
G_dcnt = -1/2 * (np.eye(N) - 1/N * np.ones(1000)).T * G *(np.eye(N) - 1/N * np.ones(N))
#%%
eval_G, evec = eigh(G_cnt)
plt.scatter(np.arange(1000),eval_G)
plt.show()
np.linalg.matrix_rank(G_cnt)
print(eval_G[990:999])
#%%
eval_dG, evec = eigh(G_dcnt)
plt.scatter(np.arange(1000),eval_dG)
plt.show()
np.linalg.matrix_rank(G_dcnt)
print(eval_G[990:999])

# %%
