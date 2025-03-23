import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import seaborn as sns

from Ising_Classes import Ising_2D_Lattice_Lookup_Table, Ising_2D_Lattice_Unoptimized, Ising_ND_Lattice
from IsingAnim import show_snapshots

colors = sns.color_palette("flare")

##################### SIMULATION TIME VS STEPS COMPARISON ######################

def sim_time_v_steps():
    L = 30
    T = 1.3
    points = 20

    steps = np.linspace(1000, 1000000, points, dtype=np.int64)
    new_times = np.zeros(len(steps))
    old_times = np.zeros(len(steps))

    for i in range(len(steps)):
        iters = steps[i]

        Lookup_Ising = Ising_2D_Lattice_Lookup_Table(L, L)
        Unoptim_Ising = Ising_2D_Lattice_Unoptimized(L, L)

        t1 = perf_counter()
        Lookup_Ising.simulate(iters, T)
        t2 = perf_counter()
        Unoptim_Ising.simulate(iters, T)
        t3 = perf_counter()

        new_times[i] = t2 - t1
        old_times[i] = t3 - t2

        print(f"Done {i+1}")

    plt.plot(steps, new_times, color=colors[1], marker="s")
    plt.plot(steps, old_times, color=colors[4], marker="s")
    plt.legend(["Lookup Table Version", "Non-Lookup Table Version"])
    plt.xlabel("Steps")
    plt.ylabel("Time (s)")
    plt.title("Simulation Time vs Steps (30x30 Lattice)")
    plt.savefig("SimTimeVsStep.png", dpi=300)
    plt.show()

# sim_time_v_steps()

##################### SIMULATION TIME VS SIZE COMPARISON #######################

def sim_time_v_sizes():
    steps = 500000
    points = 25
    T = 1.3

    sizes = np.linspace(4, 60, 25, dtype=np.int64)
    new_times = np.zeros(len(sizes))
    old_times = np.zeros(len(sizes))

    for i in range(len(sizes)):
        L = sizes[i]

        Lookup_Ising = Ising_2D_Lattice_Lookup_Table(L, L)
        Unoptim_Ising = Ising_2D_Lattice_Unoptimized(L, L)

        t1 = perf_counter()
        Lookup_Ising.simulate(steps, T)
        t2 = perf_counter()
        Unoptim_Ising.simulate(steps, T)
        t3 = perf_counter()

        new_times[i] = t2 - t1
        old_times[i] = t3 - t2

        print(f"Done {i+1}")

    plt.plot(sizes, new_times, color=colors[1], marker="s")
    plt.plot(sizes, old_times, color=colors[4], marker="s")
    plt.legend(["Lookup Table Version", "Non-Lookup Table Version"])
    plt.xlabel("Lattice Side Length")
    plt.ylabel("Time (s)")
    plt.title("Simulation Time vs Lattice Size (500,000 steps)")
    plt.savefig("SimTimeVsSize.png", dpi=300)
    plt.show()

# sim_time_v_sizes()
