import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice, periodic_adj_lattice
from numba import jit
import itertools

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
spin_combs = [list(i) for i in itertools.product([-1, 1], repeat=rows*cols)]

# For non-periodic boundary conditions
# adj_mat = create_adjacency_lattice(rows, cols, False)

# For periodic boundary conditions
adj_mat = periodic_adj_lattice(rows, cols)

ones_energy = {}

for comb in spin_combs:
    num_1s = comb.count(1)
    num_negs = comb.count(-1)
    sumones = num_1s
    H = calc_energy_normal(comb, adj_mat)
    energies = ones_energy.get(sumones)
    if energies is None:
        ones_energy[sumones] = [H]
    else:
        energies.append(H)

for key in ones_energy:
    print(key, set(ones_energy[key]))