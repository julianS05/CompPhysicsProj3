import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IsingVisuals import Anim2DGridIsing, show_snapshots
from numba import jit


# INITIALIZE VARIABLES
# ------------------------------------------------------------------------------

# grid size
ROWS = 50
COLS = 50

# steps to simulate (might change to keep going until equilibrium)
steps = 150000

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

# METROPOLIS LOOP
for i in range(steps):
    # pick random lattice point
    rand_ind = np.random.randint(0, len(lattice_elements))
    # Flip spin
    lattice_elements[rand_ind] = -lattice_elements[rand_ind]
    # calculate new energy and change in energy
    new_energy = calc_energy_sparse(lattice_elements, adj_mat.data, adj_mat.indices, adj_mat.indptr)
    dE = new_energy-curr_E

    if dE < 0 or np.random.rand() < np.exp(-dE*beta):
        # accept the move and update current energy
        curr_E = new_energy
    else:
        # flip spin back and keep current energy the same
        lattice_elements[rand_ind] = -lattice_elements[rand_ind]

    # store values
    spin_history[i] = lattice_elements
    mag_history[i] = calc_magnetization(lattice_elements)

    # if i >= backwards_theshold:
    #     avg_mag[i] = calc_avg_mag(mag_history[i-backwards_theshold:i])

# print(mag_history)

show_snapshots(spin_history, ROWS, COLS, [0, int(steps/5)-1, 2*int(steps/5)-1, 3*int(steps/5)-1, 4*int(steps/5)-1, steps-1], 2, 3)

plt.plot(np.arange(0, steps, 1), mag_history)
plt.show()

# plt.plot(np.arange(0, steps, 1), diff_mag)
# plt.show()

# anim = Anim2DGridIsing(ROWS, COLS)
# anim.animate(spin_history).save("test.mp4")
