import numpy as np
import gc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import multiprocessing as mp

from Ising_Classes import Ising_2D_Lattice_Lookup_Table, Ising_2D_Lattice_Unoptimized, Ising_ND_Lattice
from IsingAnim import show_snapshots

# MAGNETIZATION VS TEMP
def mag_v_temp(lattice_elems, sim_elems, temp_range):
    L, dim = lattice_elems
    steps, points = sim_elems
    T_start, T_end = temp_range

    temps = np.linspace(T_start, T_end, points)
    mag_abs = np.zeros(points)

    for i in range(len(temps)):
        I = Ising_ND_Lattice(L, dim)
        state_hist, mag_hist = I.simulate(steps, temps[i])
        mag_abs[i] = abs(np.mean(mag_hist[-steps//4:-1]))

        del state_hist
        del mag_hist
        gc.collect()

        print(f"Done {i+1}")
    
    return mag_abs

def run_sim(dim):
    L = 5
    steps = 3000000
    points = 30
    T_start = 0.1
    T_end = 12
    mag_abs = mag_v_temp((L, dim), (steps, points), (T_start, T_end))
    temps = np.linspace(T_start, T_end, points)
    print(f"Finished {dim}D simulation")
    return temps, mag_abs

if __name__ == "__main__":
    mp.set_start_method("fork")

    L = 5
    steps = 3000000
    dims = np.arange(1, 6, 1, dtype=np.int64)
    colors = plt.cm.viridis(np.linspace(0,1,6))

    with mp.Pool(processes=len(dims)) as pool:
        # use multiprocessing to run each simulation concurrently
        res = pool.map(run_sim, dims)

    temps, data = zip(*res)
    temps = np.array(temps)
    data = np.array(data)

    for i in range(len(data)):
        plt.plot(temps[i], data[i], 'o', color=colors[i], label=f"{dims[i]}D")

    np.savetxt("mag_temp_final.csv", data, delimiter=",")
    np.savetxt("x_axis_final.csv", temps, delimiter=",")