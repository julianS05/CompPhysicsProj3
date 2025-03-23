import numpy as np
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import multiprocessing as mp

from Ising_Classes import Ising_2D_Lattice_Lookup_Table, Ising_2D_Lattice_Unoptimized, Ising_ND_Lattice
from IsingAnim import show_snapshots


L = 25
steps = 2000000
T = np.arange(1, 5, 0.02)
init_state = np.random.choice([0,1], L*L)
Ising = Ising_2D_Lattice_Lookup_Table(L, L)

mag_abs = np.zeros(len(T))
for i in range(len(T)):
    state_history, mag_hist = Ising.simulate(steps, T[i], init_state)
    mag_abs[i] = abs(np.mean(mag_hist[-500000:-1]))

    del state_history
    del mag_hist
    gc.collect()

    print(f"done {i}")

plt.plot(T, mag_abs, 'o', color="RoyalBlue")
plt.xlabel("Temperature")
plt.ylabel("abs(Average Magnetization)")
plt.title("Avg. Mag vs Temperature")
plt.show()
# show_snapshots(state_history, L, L, [0, int(steps/5)-1, 2*int(steps/5)-1, 3*int(steps/5)-1, 4*int(steps/5)-1, steps-1], 2, 3)


# MAGNETIZATION VS STEP
L = 25
steps = 3000000
points = 16
temps = np.linspace(1, 8, points)
leg = [f"T={round(t,1)}" for t in temps]
colors = plt.cm.inferno(np.linspace(0,1,points))

init_state = np.random.choice([0,1], L*L)
Ising = Ising_2D_Lattice_Lookup_Table(L, L)

mags = np.zeros((points, L*L))
for i in range(len(temps)):
    state_hist, mag_hist = Ising.simulate(steps, round(temps[i],1), init_state)
    
    plt.plot(np.arange(0, steps, 1), mag_hist, color=colors[i])

    del state_hist
    del mag_hist
    gc.collect()

    print(f"done {i}")


plt.legend(leg, loc="upper right")
plt.title("Magnetization vs Steps for Different Temps (25x25 lattice)")
plt.xlabel("Steps")
plt.ylabel("Avg. Mag.")
plt.show()
