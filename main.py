import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IsingVisuals import Anim2DGridIsing, show_snapshots
from numba import jit



# algorithm for lookup table on the change in energy for number of different spins
# anti ferromagnetic but changing sign of J
# conserved ising
# plot equilibration time scale tau


# INITIALIZE VARIABLES
# ------------------------------------------------------------------------------

# grid size
ROWS = 10
COLS = 10

# steps to simulate (might change to keep going until equilibrium)
steps = 1000000

# randomly initializing spins for the grid
lattice_elements = np.random.choice([-1,1], ROWS*COLS)

# arrays to keep track of previous states
spin_history = np.zeros((steps,ROWS*COLS))
mag_history = np.zeros(steps)

# making adjacency matrix for 2D grid
adj_mat = create_adjacency_lattice(ROWS, COLS)

# DEFINING USEFUL FUNCTIONS
# ------------------------------------------------------------------------------

# used to calculate the total energy of the lattice
# @jit(nopython=True)
# def calc_energy_normal(spins, adj_mat):
#     H = 0
#     for i in range(len(spins)):
#         row = adj_mat[i]
#         s = 0
#         for r in range(len(row)):
#             s += row[r] * spins[i] * spins[r]
#         H += s
#     return -H

# MORE EFFICIENT CALC_ENERGY FUNCTION USING SPARSE MATRIX
###############################
# Cool note: using sparse matricies doesn't add too much efficiency for small grids
# but adds HUGE efficiency for large grids since large grids don't have more 1's per row 
# than small ones, only more rows. This fact will probably change as the dimension increases
###############################
@jit(nopython=True)
def calc_energy_sparse(spins, adj_data, adj_indices, adj_indptr):
    H = 0
    for i in range(len(spins)):
        start_row = adj_indptr[i]
        end_row = adj_indptr[i+1]

        for ind in range(start_row, end_row):
            other = adj_indices[ind]
            val = adj_data[ind]
            H += val * spins[i] * spins[other]
    return -H

# used to calculate the magnetization of the lattice
@jit(nopython=True)
def calc_magnetization(spins):
    return np.sum(spins) / len(spins)

# used to calculate the average magnization
@jit(nopython=True)
def calc_avg_mag(mag_hist):
    return np.var(mag_hist)

# MAIN LOOP
# ------------------------------------------------------------------------------

# calculate the initial lattice energy
curr_E = calc_energy_sparse(lattice_elements, adj_mat.data, adj_mat.indices, adj_mat.indptr)

backwards_theshold = int(0.1*steps) # how many data points in the past you want to use to calculate the average magnetization

avg_mag = np.zeros(steps) 

# inverse temperature
beta = 0.5

# KAWASAKI DYNAMICS LOOP
for i in range(steps):
    # Randomly pick two different lattice points
    rand_ind1, rand_ind2 = np.random.choice(len(lattice_elements), 2, replace=False)

    # Ensure the spins are different to maintain conservation
    if lattice_elements[rand_ind1] == lattice_elements[rand_ind2]:
        spin_history[i] = lattice_elements
        mag_history[i] = calc_magnetization(lattice_elements)
        continue

    # Swap the spins to propose a move
    lattice_elements[rand_ind1], lattice_elements[rand_ind2] = lattice_elements[rand_ind2], lattice_elements[rand_ind1]

    # Calculate new energy and change in energy
    new_energy = calc_energy_sparse(lattice_elements, adj_mat.data, adj_mat.indices, adj_mat.indptr)
    dE = new_energy - curr_E

    # Accept or reject the move using Metropolis criteria
    if dE < 0 or np.random.rand() < np.exp(-dE * beta):
        curr_E = new_energy  # Accept move and update energy
    else:
        # Revert the swap if the move is rejected
        lattice_elements[rand_ind1], lattice_elements[rand_ind2] = lattice_elements[rand_ind2], lattice_elements[rand_ind1]

    # Store values for visualization and tracking
    spin_history[i] = lattice_elements
    mag_history[i] = calc_magnetization(lattice_elements)

# Visualization
# show_snapshots(spin_history, ROWS, COLS, [0, int(steps/5)-1, 2*int(steps/5)-1, 3*int(steps/5)-1, 4*int(steps/5)-1, steps-1], 2, 3)
show_snapshots(spin_history, ROWS, COLS, [0, int(steps/10000)-1, int(steps/1000)-1, int(steps/100)-1, int(steps/10)-1, steps-1], 2, 3)

plt.plot(np.arange(0, steps, 1), mag_history)
plt.show()

# Check neighbors over time


# plt.plot(np.arange(0, steps, 1), diff_mag)
# plt.show()

# anim = Anim2DGridIsing(ROWS, COLS)
# anim.animate(spin_history).save("test.mp4")
