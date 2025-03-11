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
        # H = 0
        # for i in range(len(state)):
        #     start_row = self.adjacency_mat.indptr[i]
        #     end_row = self.adjacency_mat.indptr[i+1]

        #     for ind in range(start_row, end_row):
        #         other = self.adjacency_mat.indices[ind]
        #         val = self.adjacency_mat.data[ind]
        #         H += val * state[i] * state[other]
        # return -H
    
    def move(self, state, beta):
        curr_E = self.calc_energy(state)
        # pick random lattice point
        rand_ind = np.random.randint(0, len(state))
        # Flip spin
        state[rand_ind] = -state[rand_ind]
        # calculate new energy and change in energy
        new_E = self.calc_energy(state)
        dE = new_E - curr_E

        if dE < 0 or np.random.rand() < np.exp(-dE*beta):
            # accept the move and update current energy
            curr_E = new_E
        else:
            # flip spin back and keep current energy the same
            state[rand_ind] = -state[rand_ind]

    # @jit(nopython=True)
    def calc_magnetization(self, state):
        return np.sum(state) / len(state)

    def simulate(self, steps, temp, starting_state=None):
        if starting_state is  None:
            lattice_state = np.random.choice([-1,1], self.ROWS*self.COLS)
        else:
            lattice_state = starting_state
        beta = 1 / temp

        state_history = np.zeros((steps,self.ROWS*self.COLS))
        magnetization_history = np.zeros(steps)

        np.random.seed() # Remove the random seed!

        for step in range(steps):
            self.move(lattice_state, beta)
            state_history[step] = lattice_state
            magnetization_history[step] = self.calc_magnetization(lattice_state)
        
        return state_history, magnetization_history


# test = GridIsing2D(3, 3)
# test.simulate(4, 8)