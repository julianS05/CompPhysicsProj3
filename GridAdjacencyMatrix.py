from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import diags
import matplotlib.animation as animation


def create_adjacency_lattice(rows, cols, periodic=False):
    mat = np.zeros((rows*cols, rows*cols))
    # defining the interior matrices
    int_mat1 = diags([np.ones(cols-1), np.ones(cols-1)],[-1,1]).toarray()
    int_mat2 = diags([np.ones(cols-1),np.ones(cols),np.ones(cols-1)], [-1,0,1]).toarray()
    print(int_mat2)


    # pasting the off-diagonal interior matrices
    for i in range(0, rows*cols-cols, cols):
        mat[i+cols:(i+2*cols),i:(i+cols)]=int_mat2
        mat[i:(i+cols),i+cols:(i+2*cols)]=int_mat2

    # pasting the diagonal interior matrices
    for i in range(0,rows*cols, cols):
        mat[i:(i+cols),i:(i+cols)]=int_mat1

    return mat




create_adjacency_lattice(3,3)
# # print(calc_energy(spins, adj_mat))

# beta = 0.5 # beta = 1/T


# spins = np.random.choice([-1,1], 16)

# state_history = [spins]
# magnetization_history = [calc_magnetization(spins)]

# # randomize which spins to change later
# curr_E = calc_energy(spins, adj_mat)
# for i in range(len(spins)):
#     spins[i]=-spins[i]
#     new_energy = calc_energy(spins, adj_mat)
#     dE = new_energy-curr_E
#     if dE >= 0:# or np.random.rand() > np.exp(-dE*beta):# or rand() > exp(-dE/T): # add thermal probability later
#         spins[i] = -spins[i]
    
#     state_history.append(spins)
#     print(state_history[-1], i)
#     print()
#     # if i > 0: print(np.subtract(spins, state_history[i-1]))
#     # magnetization_history.append(calc_magnetization(spins))

# state_history = np.array(state_history)

# # plot


# print(state_history)

# # fig = plt.figure()

# # ims = []
# # for i in range(len(state_history)):
# #     im = plt.imshow(np.reshape(state_history[i], (4,4), 'C'), cmap='binary', animated=True)
# #     ims.append([im])

# # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
# #                                 repeat_delay=1000)

# # ani.save('test.mp4')

# # plt.show()