import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from IsingAnim import Anim2DGridIsing

# INITIALIZE VARIABLES
# ------------------------------------------------------------------------------

# grid size
ROWS = 20
COLS = 20

# steps to simulate (might change to keep going until equilibrium)
steps = 1000

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
def calc_energy(spins, adj_mat):
    H = 0
    for i in range(len(spins)):
        row = adj_mat[i]
        s = 0
        for r in range(len(row)):
            s += row[r] * spins[i] * spins[r]
        H += s
    return -H

# used to calculate the magnetization of the lattice
def calc_magnetization(spins):
    return sum(spins) / len(spins)

# used to calculate the average magnization
def calc_avg_mag(mag_hist):
    return np.mean(mag_hist)

# MAIN LOOP
# ------------------------------------------------------------------------------

# calculate the initial lattice energy
curr_E = calc_energy(lattice_elements, adj_mat)

backwards_theshold = 100 # how many data points in the past you want to use to calculate the average magnetization

avg_mag = np.zeros(steps) 

# inverse temperature
beta = 0.4

# METROPOLIS LOOP
for i in range(steps):
    # pick random lattice point
    rand_ind = np.random.randint(0, len(lattice_elements))
    # Flip spin
    lattice_elements[rand_ind] = -lattice_elements[rand_ind]
    # calculate new energy and change in energy
    new_energy = calc_energy(lattice_elements, adj_mat)
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

plt.plot(np.arange(0, steps, 1), mag_history)
# plt.plot(np.arange(0, steps, 1), avg_mag)

anim = Anim2DGridIsing(ROWS, COLS)
anim.animate(spin_history).save("test.mp4")

plt.show()