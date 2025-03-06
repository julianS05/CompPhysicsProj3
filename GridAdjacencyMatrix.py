from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import diags
import matplotlib.animation as animation
from scipy.sparse import csr_matrix


def create_adjacency_lattice(rows, cols, periodic=False):
    mat = np.zeros((rows*cols, rows*cols))
    # defining the interior matrices
    int_mat1 = diags([np.ones(cols-1), np.ones(cols-1)],[-1,1]).toarray()
    int_mat2 = diags([np.ones(cols-1),np.ones(cols),np.ones(cols-1)], [-1,0,1]).toarray()


    # pasting the off-diagonal interior matrices
    for i in range(0, rows*cols-cols, cols):
        mat[i+cols:(i+2*cols),i:(i+cols)]=int_mat2
        mat[i:(i+cols),i+cols:(i+2*cols)]=int_mat2

    # pasting the diagonal interior matrices
    for i in range(0,rows*cols, cols):
        mat[i:(i+cols),i:(i+cols)]=int_mat1

    return csr_matrix(mat)
    # return mat