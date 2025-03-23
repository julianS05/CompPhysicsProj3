import numpy as np
import matplotlib.pyplot as plt

dims = np.arange(1, 7, 1, dtype=np.int64)
colors = plt.cm.viridis(np.linspace(0,1,len(dims)))
temps = np.loadtxt('x_axis_final.csv', delimiter=',')
data = np.loadtxt('mag_temp_final.csv', delimiter=',')

L = 5
steps = 3000000

plt.figure(figsize=(8, 6))

for i in range(len(data)):
    plt.plot(temps[i], data[i], 'o', color=colors[i], label=f"{dims[i]}D")
    plt.plot(temps[i], data[i], '-', color=colors[i], alpha=0.3)

plt.xlim(0, 14)
plt.tick_params(axis='both', direction='in')
plt.grid()
plt.xlabel("Temperature")
plt.ylabel("abs(Avgerage Magnetization)")
# plt.xticks(np.arange(0, 15, 1))
plt.title(f"Avg. Mag. vs Temp. For {L}x{L} {dims[0]}-D to {dims[-1]}-D Lattices ({steps} steps)")
plt.legend(loc="upper right", fancybox=False, framealpha=1, ncol=2)
plt.figtext(0.12, 0.008, '$^*$Outlier points where simulation "gets stuck" were removed.')

# plt.axvspan(1.7, 3, alpha=0.2, color=colors[1], lw=0)
# plt.axvspan(3.5, 5.3, alpha=0.2, color=colors[2], lw=0)
# plt.axvspan(5.4, 7.35, alpha=0.2, color=colors[3], lw=0)
# plt.axvspan(7.3, 9.3, alpha=0.2, color=colors[4], lw=0)
# plt.axvspan(9.4, 12, alpha=0.2, color=colors[5], lw=0)
plt.savefig(f"Mag_vs_temp_mult_dims.png", dpi=300)
plt.show()