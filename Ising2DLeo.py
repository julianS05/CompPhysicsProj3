import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice
from matplotlib import pyplot as plt
import matplotlib.animation as animation

def isingModel(M, N, J, T): #M, N deterimines the size of the grid, J is the thermal coefficient, T is temp
    # INITIALIZE GRID
    ROWS = N
    COLS = M
    t = int(1000 * N)

    adj_mat = create_adjacency_lattice(ROWS, COLS)

    lattice_elements = np.random.choice([-1,1], ROWS*COLS)
    spin_history = np.zeros((t, ROWS*COLS))
    mag_history = np.zeros(t)

    def calc_energy(spins, adj_mat):
        H = 0
        for i in range(len(spins)):
            row = adj_mat[i]
            s = 0
            for r in range(len(row)):
                s += J * row[r] * spins[i] * spins[r]
            H += s
        return -H

    def calc_magnetization(spins):
        return sum(spins) / len(spins)

    curr_E = calc_energy(lattice_elements, adj_mat)

    # METROPOLIS LOOP
    for i in range(t):
        # Flip spin
        rand_ind = np.random.randint(0, len(lattice_elements))
        lattice_elements[rand_ind] = -lattice_elements[rand_ind]
        new_energy = calc_energy(lattice_elements, adj_mat)
        dE = new_energy-curr_E

        if dE >= 0: #does not change the current energy if it doesn't flip back
            p = np.exp(-dE / T)
            if np.random.rand() <= p: # add thermal probability later
                # flip spin back
                lattice_elements[rand_ind] = -lattice_elements[rand_ind]
                print("flip back")
        spin_history[i] = lattice_elements
        mag_history[i] = calc_magnetization(lattice_elements)
        curr_E = new_energy

    print(mag_history)

    fig = plt.figure()

    ims = []
    for i in range(len(spin_history)):
        im = plt.imshow(np.reshape(spin_history[i], (ROWS,COLS), 'C'), cmap='binary', animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True,
                                    repeat_delay=1000)

    ani.save('test.mp4')

isingModel(10, 10, 0.2, 300)