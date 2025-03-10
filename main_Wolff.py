import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IsingVisuals import Anim2DGridIsing, show_snapshots
from numba import jit

# INITIALIZE VARIABLES
# ------------------------------------------------------------------------------

# grid size
ROWS = 10
COLS = 10

# steps to simulate (might change to keep going until equilibrium)
steps = 3

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

def wolff_update(lattice, adj_indices, adj_indptr):
    # pick a random seed
    seed = np.random.randint(0,ROWS*COLS)
    spin_seed = lattice[seed]

    # bond addition probability
    # p_add = 1 - np.exp(-2 * beta * J)
    p_add = 1 - np.exp(-2 * 0.5)
    
    # Boolean cluster to check if the lattice is in the neighbor cluster
    in_cluster = np.zeros(ROWS*COLS, dtype=bool)
    in_cluster[seed] = True
    
    # Initialize the stack with the seed ind
    stack = np.array([seed])
    
    while stack.size > 0:
        new_stack = []
        for ind in stack:
            # get neighbor indices using the sparse matrix
            start = adj_indptr[ind]
            end = adj_indptr[ind + 1]
            neighbors = adj_indices[start:end]
            for i in neighbors:
                if not in_cluster[i] and lattice[i] == spin_seed:
                    # add neighbor with probability p_add
                    if np.random.rand() < p_add:
                        in_cluster[i] = True
                        new_stack.append(i)
        # update stack
        stack = np.array(new_stack)
    
    # flip all spins in the cluster
    cluster_indices = np.where(in_cluster)[0]
    lattice[cluster_indices] *= -1
    return lattice

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

# --- WOLFF LOOP ---
# Reinitialize the lattice for the Wolff simulation.
lattice_elements_wolff = np.random.choice([-1, 1], ROWS * COLS)
wolff_spin_history = np.zeros((steps, ROWS * COLS))
wolff_mag_history = np.zeros(steps)

for i in range(steps):
    lattice_elements_wolff = wolff_update(lattice_elements_wolff, adj_mat.indices, adj_mat.indptr)
    wolff_spin_history[i] = lattice_elements_wolff
    wolff_mag_history[i] = calc_magnetization(lattice_elements_wolff)

show_snapshots(wolff_spin_history, ROWS, COLS,[0, int(steps/5)-1, 2*int(steps/5)-1, 3*int(steps/5)-1, 4*int(steps/5)-1, steps-1], 2, 3)

plt.figure()
plt.plot(np.arange(steps), wolff_mag_history)
plt.title("Wolff Magnetization")
plt.show()


# print(mag_history)

show_snapshots(spin_history, ROWS, COLS, [0, int(steps/5)-1, 2*int(steps/5)-1, 3*int(steps/5)-1, 4*int(steps/5)-1, steps-1], 2, 3)

plt.plot(np.arange(0, steps, 1), mag_history)
plt.show()

# plt.plot(np.arange(0, steps, 1), diff_mag)
# plt.show()

# anim = Anim2DGridIsing(ROWS, COLS)
# anim.animate(spin_history).save("test.mp4")
