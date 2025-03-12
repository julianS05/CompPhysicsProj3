import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice, periodic_adj_lattice
from numba import jit
import itertools
from matplotlib import pyplot as plt

@jit(nopython=True)
def calc_energy_normal(spins, adj_mat):
    H = 0
    for i in range(len(spins)):
        row = adj_mat[i]
        s = 0
        for r in range(len(row)):
            s += row[r] * spins[i] * spins[r]
        H += s
    return -H

rows, cols = 4, 4
lattice = np.arange(0, rows*cols, 1).reshape((rows, cols))
# spin_combs = [list(i) for i in itertools.product([-1, 1], repeat=rows*cols)]

# For non-periodic boundary conditions
# adj_mat = create_adjacency_lattice(rows, cols, False)

# For periodic boundary conditions
adj_mat = periodic_adj_lattice(rows, cols)


spins = np.random.choice([-1,1], rows*cols)

# First initialize array to keep track of the number of -1's around each point
num_neg_one_neighbors = np.zeros(rows*cols)


for spin_num in range(len(spins)):
    neighbors = adj_mat[spin_num]
    spots = np.nonzero(neighbors)
    num_negone = 0

    for ind in spots[0]:
        if spins[ind] == -1:
            num_negone += 1
    num_neg_one_neighbors[spin_num] = num_negone

print(num_neg_one_neighbors)

# if self is -1, multiply numbers by -1, if self is 1, multiply by 1
lookup = [4, 2, 0, -2, -4]

E = 0
for i in range(len(num_neg_one_neighbors)):
    E += spins[i] * lookup[int(num_neg_one_neighbors[i])]

print(-E)

print(calc_energy_normal(spins, adj_mat))
# Visualize lattice neighbors
# colors = np.zeros(rows*cols)
# site = 6
# row = adj_mat[site]
# for i in range(len(row)):
#     if row[i] == 1:
#         colors[i] = 1

# plt.imshow(np.reshape(colors, (rows,cols), 'C'), cmap='binary')
# plt.show()