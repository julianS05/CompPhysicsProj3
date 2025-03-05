import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# INITIALIZE GRID
ROWS = 10
COLS = 10

steps = 5000

adj_mat = create_adjacency_lattice(ROWS, COLS)

lattice_elements = np.random.choice([-1,1], ROWS*COLS)
spin_history = np.zeros((steps,ROWS*COLS))
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

def calc_avg_mag(mag_hist):
    return np.mean(mag_hist)

curr_E = calc_energy(lattice_elements, adj_mat)

backwards_theshold = 100 # how many data points in the past you want to use to calculate the average magnetization

avg_mag = np.zeros(steps) 

beta = 0.8

# METROPOLIS LOOP
for i in range(steps):
    # pick random lattice point
    rand_ind = np.random.randint(0, len(lattice_elements))
    # Flip spin
    lattice_elements[rand_ind] = -lattice_elements[rand_ind]
    new_energy = calc_energy(lattice_elements, adj_mat)
    dE = new_energy-curr_E

    if dE < 0 or np.random.rand() < np.exp(-dE*beta):
        print("accept")
        curr_E = new_energy
    else:
        # flip spin back
        print("flip back")
        lattice_elements[rand_ind] = -lattice_elements[rand_ind]

    spin_history[i] = lattice_elements
    mag_history[i] = calc_magnetization(lattice_elements)
    # print(mag_history)
    if i >= backwards_theshold:
        avg_mag[i] = calc_avg_mag(mag_history[i-backwards_theshold:i])




# fig = plt.figure()

# ims = []
# for i in range(len(spin_history)):
#     im = plt.imshow(np.reshape(spin_history[i], (ROWS,COLS), 'C'), cmap='binary', animated=True)
#     ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True,
#                                 repeat_delay=1000)

# ani.save('test.mp4')


plt.plot(np.arange(0, steps, 1), mag_history)
plt.plot(np.arange(0, steps, 1), avg_mag)
plt.show()