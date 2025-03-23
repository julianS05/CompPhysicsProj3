import numpy as np
from numba import jit
import itertools
from scipy.sparse import csr_matrix

def periodic_2D_lattice(rows, cols, csr=False):
    '''
    Generates an adjacency matrix for a 2D periodic lattice.

    Parameters:
        - rows (int): Number of rows in the lattice.
        - cols (int): Number of columns in the lattice.
        - csr (bool): Whether to return a scipy csr sparse matrix or not.

    Returns:
        - np.ndarray: A (rows * cols) x (rows * cols) adjacency matrix representing 
                the periodic 2D lattice, where an entry of 1 indicates an edge 
                between two nodes.
    '''
    total_nodes = rows * cols
    adj_mat = np.zeros((total_nodes, total_nodes), dtype=int)

    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j

            adj_mat[node_id, ((i - 1)%rows) * cols + j] = 1
            adj_mat[node_id, ((i + 1)%rows) * cols + j] = 1
            adj_mat[node_id, i * cols + ((j - 1)%cols)] = 1
            adj_mat[node_id, i * cols + ((j + 1)%cols)] = 1
    
    if csr: adj_mat = csr_matrix(adj_mat)

    return adj_mat

def periodic_3D_lattice(L, csr=False):
    '''
    Generates an adjacency matrix for a 3D periodic lattice.

    Parameters:
        - L (int): Side length of the LxLxL lattice.
        - csr (bool): Whether to return a scipy csr sparse matrix or not.


    Returns:
        - np.ndarray: A (L^3) x (L^3) adjacency matrix representing 
                the periodic 3D lattice, where an entry of 1 indicates an edge 
                between two nodes.
    '''
    total_nodes = np.power(L, 3)
    adj_mat = np.zeros((total_nodes, total_nodes), dtype=np.int64)

    for i in range(L):
        for j in range(L):
            for k in range(L):
                node_id = np.ravel_multi_index((i,j,k), (L,L,L))

                adj_mat[node_id, np.ravel_multi_index(((i-1)%L,j,k), (L,L,L))] = 1
                adj_mat[node_id, np.ravel_multi_index(((i+1)%L,j,k), (L,L,L))] = 1
                adj_mat[node_id, np.ravel_multi_index((i,(j-1)%L,k), (L,L,L))] = 1
                adj_mat[node_id, np.ravel_multi_index((i,(j+1)%L,k), (L,L,L))] = 1
                adj_mat[node_id, np.ravel_multi_index((i,j,(k-1)%L), (L,L,L))] = 1
                adj_mat[node_id, np.ravel_multi_index((i,j,(k+1)%L), (L,L,L))] = 1
    
    if csr: adj_mat = csr_matrix(adj_mat)
    return adj_mat

def periodic_nD_lattice(L, n, csr=False):
    '''
    Generates an adjacency matrix for an n-dimensional periodic lattice.

    Parameters:
        - L (int): Side length of the n-dimensional lattice.
        - n (int): Dimension of the lattice.
        - csr (bool): Whether to return a scipy csr sparse matrix or not.

    Returns:
        - np.ndarray: A (L^n) x (L^n) adjacency matrix representing 
                the periodic n-dimensional lattice, where an entry of 1 
                indicates an edge between two nodes.
    '''
    total_nodes = np.power(L, n)
    mat_shape = tuple([L for i in range(n)])
    adj_mat = np.zeros((total_nodes, total_nodes), dtype=np.int64)
    # rather than nesting n for-loops, precalculate all the different combinations
    # of the n nested for loop indices
    combs = np.array(list(itertools.product(*[range(L) for i in range(n)])))

    for i in range(len(combs)):
        node_id = np.ravel_multi_index(tuple(combs[i]), mat_shape)
        for j in range(len(combs[i])):
            curr_comb = combs[i].copy()
            curr_comb[j] = (curr_comb[j]-1)%L
            adj_mat[node_id, np.ravel_multi_index(tuple(curr_comb), mat_shape)] = 1

            curr_comb = combs[i].copy()
            curr_comb[j] = (curr_comb[j]+1)%L
            adj_mat[node_id, np.ravel_multi_index(tuple(curr_comb), mat_shape)] = 1
    
    if csr: adj_mat = csr_matrix(adj_mat)
    return adj_mat
