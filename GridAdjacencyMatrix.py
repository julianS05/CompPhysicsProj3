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

        mat[i, i + cols - 1] = 1
        mat[i + cols - 1, i] = 1
    
    for i in range(cols):
        mat[i, (rows-1)*cols + i] = 1  # Connect first row to last row in the same column
        mat[(rows-1)*cols + i, i] = 1

    # return csr_matrix(mat)
    return mat

# plt.matshow(create_adjacency_lattice(5, 5, False))
# plt.show()

def periodic_adj_lattice(rows, cols):
    total_nodes = rows * cols
    adj_mat = np.zeros((total_nodes, total_nodes), dtype=int)

    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j

            adj_mat[node_id, ((i - 1)%rows) * cols + j] = 1
            adj_mat[node_id, ((i + 1)%rows) * cols + j] = 1
            adj_mat[node_id, i * cols + ((j - 1)%cols)] = 1
            adj_mat[node_id, i * cols + ((j + 1)%cols)] = 1
            adj_mat[node_id, ((i - 1)%rows) * cols + ((j - 1)%cols)] = 1
            adj_mat[node_id, ((i - 1)%rows) * cols + ((j + 1)%cols)] = 1
            adj_mat[node_id, ((i + 1)%rows) * cols + ((j + 1)%cols)] = 1
            adj_mat[node_id, ((i + 1)%rows) * cols + ((j - 1)%cols)] = 1
    
    return csr_matrix(adj_mat)
    # return adj_mat