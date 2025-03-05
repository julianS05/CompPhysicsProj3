import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# INITIALIZE GRID
ROWS = 4
COLS = 4

steps = 100

adj_mat = create_adjacency_lattice(ROWS, COLS)

lattice_elements = np.random.choice([-1,1], ROWS*COLS)
spin_history = np.zeros((steps,16))
mag_history = np.zeros(steps)

def calc_energy(spins, adj_mat):
    H = 0
    for i in range(len(spins)):
        row = adj_mat[i]
        s = 0
        for r in range(len(row)):
            s += row[r] * spins[i] * spins[r]
        H += s
    return -H

def calc_magnetization(spins):
    return sum(spins) / len(spins)

curr_E = calc_energy(lattice_elements, adj_mat)

# METROPOLIS LOOP
for i in range(len(lattice_elements)):
    # Flip spin
    lattice_elements[i] = -lattice_elements[i]
    new_energy = calc_energy(lattice_elements, adj_mat)
    dE = new_energy-curr_E

    if dE >= 0:# or np.random.rand() > np.exp(-dE*beta):# or rand() > exp(-dE/T): # add thermal probability later
        # flip spin back
        lattice_elements[i] = -lattice_elements[i]
        print("flip back")
    spin_history[i] = lattice_elements
    mag_history[i] = calc_magnetization(lattice_elements)

print(mag_history)

fig = plt.figure()

ims = []
for i in range(len(spin_history)):
    im = plt.imshow(np.reshape(spin_history[i], (4,4), 'C'), cmap='binary', animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

ani.save('test.mp4')