import numpy as np
from GridAdjacencyMatrix import create_adjacency_lattice, periodic_adj_lattice
from numba import jit
# from numba.experimental import jitclass

@jit(nopython=True)
def _calc_energy(spins, adj_data, adj_indices, adj_indptr):
    H = 0
    for i in range(len(spins)):
        start_row = adj_indptr[i]
        end_row = adj_indptr[i+1]

        for ind in range(start_row, end_row):
            other = adj_indices[ind]
            val = adj_data[ind]
            H += val * spins[i] * spins[other]
    return -H

class GridIsing2D:
    def __init__(self, rows, cols):
        self.ROWS = rows
        self.COLS = cols
        self.adjacency_mat = periodic_adj_lattice(self.ROWS, self.COLS)
    
    # @jit(nopython=True)
    def calc_energy(self, state):
        return _calc_energy(state, self.adjacency_mat.data, self.adjacency_mat.indices, self.adjacency_mat.indptr)
    
    # def move(self, state, beta):
    #     curr_E = self.calc_energy(state)
    #     # pick random lattice point
    #     rand_ind = np.random.randint(0, len(state))
    #     # Flip spin
    #     state[rand_ind] = -state[rand_ind]
    #     # calculate new energy and change in energy
    #     new_E = self.calc_energy(state)
    #     dE = new_E - curr_E

    #     if dE < 0 or np.random.rand() < np.exp(-dE*beta):
    #         # accept the move and update current energy
    #         curr_E = new_E
    #     else:
    #         # flip spin back and keep current energy the same
    #         state[rand_ind] = -state[rand_ind]

    # @jit(nopython=True)
    def calc_magnetization(self, state):
        return np.sum(state) / len(state)

    def simulate(self, steps, temp, starting_state=None):
        lattice_state = None
        if starting_state is  None:
            lattice_state = np.random.choice([-1,1], self.ROWS*self.COLS)
        else:
            lattice_state = starting_state
        beta = 1 / temp

        state_history = np.zeros((steps,self.ROWS*self.COLS))
        magnetization_history = np.zeros(steps)

        np.random.seed(None) # Remove the random seed!
        curr_E = self.calc_energy(lattice_state)
        for step in range(steps):
            # pick random lattice point
            rand_ind = np.random.randint(0, len(lattice_state))
            # Flip spin
            lattice_state[rand_ind] = -lattice_state[rand_ind]
            # calculate new energy and change in energy
            new_E = self.calc_energy(lattice_state)
            dE = new_E - curr_E

            if dE < 0 or np.random.rand() < np.exp(-dE*beta):
                # accept the move and update current energy
                curr_E = new_E
            else:
                # flip spin back and keep current energy the same
                lattice_state[rand_ind] = -lattice_state[rand_ind]
            state_history[step] = lattice_state
            magnetization_history[step] = self.calc_magnetization(lattice_state)
        
        return state_history, magnetization_history

# class GridIsing2D_optim:
#     def __init__(self, rows, cols):
#         self.ROWS = rows
#         self.COLS = cols
#         self.adjacency_mat = periodic_adj_lattice(self.ROWS, self.COLS)
#         self.energy_lookup = [4, 2, 0, -2, -4]
    
#     def get_down_spin_neighbors(self):
#         down_spin_neighbors = np.zeros(self.ROWS*self.COLS, dtype=np.int8)
#         for spin_num in range(len(self.spins)):
#             neighbors = self.adjacency_mat[spin_num]
#             neighbor_spins = self.spins[np.nonzero(neighbors)]
#             down_spin_neighbors[spin_num] = np.sum(neighbor_spins)
        
#         return down_spin_neighbors

#     def initialize_lattice(self, seed=None, starting_state=None):
#         np.random.seed(seed)

#         self.spins = []
#         if starting_state is  None:
#             self.spins = np.random.choice([-1,1], self.ROWS*self.COLS)
#         else:
#             self.spins = starting_state
        
#         self.num_down_spin_neighbors = self.get_down_spin_neighbors()
#         # print(self.num_down_spin_neighbors)

#     # @jit(nopython=True)
#     def calc_energy(self):
#         H = 0
#         for i in range(len(self.num_down_spin_neighbors)):
#             H += self.spins[i] * self.energy_lookup[self.num_down_spin_neighbors[i]]
        
#         return -H

#     # @jit(nopython=True)
#     def calc_magnetization(self, state):
#         return np.sum(state) / len(state)
    
#     def calc_avg_mag(self, mags):
#         return np.mean(mags)

#     def simulate(self, steps, temp):
#         beta = 1 / temp
#         state_history = np.zeros((steps,self.ROWS*self.COLS))
#         magnetization_history = np.zeros(steps)

#         curr_E = self.calc_energy()
#         for step in range(steps):
#             # pick random lattice point
#             rand_ind = np.random.randint(0, len(self.spins))
#             # Flip spin
#             self.spins[rand_ind] = -self.spins[rand_ind]
#             # calculate new energy and change in energy
#             new_E = self.calc_energy()
#             dE = new_E - curr_E

#             if dE < 0 or np.random.rand() < np.exp(-dE*beta):
#                 # accept the move and update current energy
#                 curr_E = new_E
#                 # update the number of down spins of each neighbor
#                 np.add.at(self.num_down_spin_neighbors, np.nonzero(self.adjacency_mat[rand_ind]), self.spins[rand_ind])
#             else:
#                 # flip spin back and keep current energy the same
#                 self.spins[rand_ind] = -self.spins[rand_ind]

#             state_history[step] = self.spins
#             magnetization_history[step] = self.calc_magnetization(self.spins)
        
#         return state_history, magnetization_history