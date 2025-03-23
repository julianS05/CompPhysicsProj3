import numpy as np
from numba import jit
import gc

# import functions to create adjacency matrices
from Adjacency_Matrices import periodic_2D_lattice, periodic_3D_lattice, periodic_nD_lattice

################### NUMBA FUNCTIONS FOR LOOKUP TABLE CLASSES ###################
@jit(nopython=True)
def _calc_energy(spins, neighbor_values, lookup):
    '''
    JIT function to calculate the energy of a 2D lattice.

    Parameters:
        - Spins (np.ndarray): List of the spin values at each lattice site.
        - Neighbor_values (np.ndarray): List of the number of down spin neighbors for each lattice site.
        - Lookup (np.ndarray): Energy values for the lookup table.
    
    Returns:
        - float: The total energy of the configuration.
    '''
    H = 0
    for i in range(len(spins)):
        H += lookup[spins[i]][neighbor_values[i]]
    
    return -H

@jit(nopython=True)
def _update_neighbor_spin_nums(neighbor_indices, neighbor_values, point, change):
    '''
    JIT function to update the number of down spin neighbors for all neighbors of a certain lattice site.

    Parameters:
        - Neighbor_indices (np.ndarray): List of the indices of the neighbors.
        - Neighbor_values (np.ndarray): List of the number of down spin neighbors for each lattice site.
        - Point (int): index of the point who's neighbors to update.
        - Change (int): Amount to change the values by
    
    Returns nothing, but changes neighbor_values
    '''
    neighbors = neighbor_indices[point]
    neighbor_values[neighbors] += change

@jit(nopython=True)
def _calc_mag(spins):
    '''
    JIT function to calculate the average magnetization of a given configuration.

    Parameters:
        - Neighbor_indices (np.ndarray): List of the indices of the neighbors.
        - Neighbor_values (np.ndarray): List of the number of down spin neighbors for each lattice site.
        - Point (int): index of the point who's neighbors to update.
        - Change (int): Amount to change the values by.
    
    Returns:
        - float: The average magnetization of the given configuration.
    '''
    # -1's are 1's and 1's are 0's
    # So sum of spins is (len(spins) - sum(spins)) - sum(spins)
    sum_ones = sum(spins)
    l = spins.shape[0]
    return (l - 2 * sum_ones) / l

@jit(nopython=True)
def _mem_optim_get_neighbor_indices(L):
    '''
    Generates a memory-optimized lookup table of neighbor indices for a periodic 2D lattice.

    Parameters:
        - L (int): The size of the lattice (L x L grid).

    Returns:
        - np.ndarray: A (L*L, 4) array where each row contains indices of the four neighboring spins 
                for a given site in the periodic lattice.
    '''
    neighbors = np.zeros((L*L, 4), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            neighbors[i*L+j] = np.array([((i - 1)%L) * L + j, ((i + 1)%L) * L + j, i * L + ((j - 1)%L), i * L + ((j + 1)%L)])
    return neighbors

@jit(nopython=True)
def _mem_optim_get_down_neighbors(spins, neighbor_indices):
    '''
    Computes the number of down-spin neighbors for each spin.

    Parameters:
        - spins (np.ndarray): A 1D array representing the spin configuration (1 for down, 0 for up).
        - neighbor_indices (np.ndarray): A 2D array where each row contains the indices of neighboring spins.

    Returns:
        - np.ndarray: A 1D array where each element represents the number of down-spin neighbors for each site.
    '''
    neighbors = np.zeros(len(spins), dtype=np.int64)
    for i in range(len(neighbors)):
        neighbors[i] = sum(spins[neighbor_indices[i]])
    return neighbors

def _create_NDim_lookup(dim):
    lookup = np.zeros((2, 2*dim+1), dtype=np.int64)
    for count, i in enumerate(range(2*dim, -2*dim-1, -2)):
        lookup[0][count] = i
        lookup[1][count] = -i
    return lookup

################################################################################
########################## LOOKUP TABLE ISING MODEL ############################
################################################################################
###---------------- Optimized for simulation speed by using a ---------------###
##--------------------- lookup table to calculate energy ---------------------##

class Ising_2D_Lattice_Lookup_Table:
    '''
    A class representing a 2D Ising model on a periodic lattice with a lookup table 
    for quick energy calculations. Uses the value of 0 for spin up and 1 for spin
    down to facilitate fast calculations.
    '''
    def __init__(self, rows, cols):
        '''
        Initializes the Ising model with a periodic 2D lattice.

        Parameters:
            - rows (int): Number of rows in the lattice.
            - cols (int): Number of columns in the lattice.
        '''
        self.ROWS = rows
        self.COLS = cols
        self.adj_mat = periodic_2D_lattice(self.ROWS, self.COLS)
        self.LOOKUP_TABLE = np.array([[4, 2, 0, -2, -4], [-4, -2, 0, 2, 4]])
    
    def simulate(self, steps, temp, initial_state=None):
        '''
        Simulates the Ising model using the Metropolis algorithm.

        Parameters:
            - steps (int): Number of simulation steps.
            - temp (float): Temperature of the system.
            - initial_state (np.ndarray, optional): Initial spin configuration (1 for up, 0 for down). 
                If None, a random initialization is used.

        Returns:
            - tuple[np.ndarray, np.ndarray]: 
                - state_history (np.ndarray): Spin configurations over time.
                - magnetization_history (np.ndarray): Magnetization over time.
        '''
        np.random.seed(None)
        # Initialize the needed arrays
        self.spins = np.random.choice([0,1], self.ROWS*self.COLS) if (isinstance(initial_state, np.ndarray) or initial_state is None) else initial_state
        self.neighbor_indices = self.init_neighbor_indices()
        self.neighbor_down_spins = self.get_down_spin_neighbors()

        state_history = np.zeros((steps,self.ROWS*self.COLS))
        magnetization_history = np.zeros(steps)

        # Initialize simulation variables
        beta = 1 / temp
        E_old = self.calc_energy()

        # main loop
        for step in range(steps):
            # get random array index
            rand_ind = np.random.randint(0, len(self.spins))

            before_spin = self.spins[rand_ind]

            # increase number of down spins if previous spin was up and is now switched down
            change = 1 if before_spin == 0 else -1
            # update neighbors and calculate new energy
            self.update_neighbor_spin_nums(rand_ind, change)
            E_new = self.calc_energy()
            dE = E_new - E_old

            if dE < 0 or np.random.rand() < np.exp(-dE*beta):
                # accept the move and actually change spin
                E_old = E_new
                self.spins[rand_ind] = 1 if before_spin == 0 else 0
            else:
                # revert the number of down spin neighbors
                self.update_neighbor_spin_nums(rand_ind, -change)

            # state_history[step] = self.spins
            magnetization_history[step] = self.calc_mag(self.spins)
        
        return state_history, magnetization_history

    def init_neighbor_indices(self):
        '''
        Initializes an array of neighbor indices for each lattice site.

        Returns:
            - np.ndarray: A 2D array where each row contains indices of neighboring spins.
        '''
        neighbors = []
        for row in range(len(self.adj_mat)):
            nonzeros = np.nonzero(self.adj_mat[row])
            neighbors.append(nonzeros[0])
        
        return np.array(neighbors)
    
    def get_down_spin_neighbors(self):
        '''
        Computes the number of down-spin neighbors for each spin in the lattice.

        Returns:
            - np.ndarray: A 1D array where each element represents the number of down-spin neighbors.
        '''
        neighbors = np.zeros(len(self.spins), dtype=np.int64)
        for i in range(len(self.spins)):
            neighbors[i] = np.dot(self.adj_mat[i], self.spins)
        
        return neighbors

    def calc_energy(self):
        '''
        Wrapper for JIT function that computes the total energy of the current spin configuration.

        Returns:
            - float: The total energy of the system.
        '''
        return _calc_energy(self.spins, self.neighbor_down_spins, self.LOOKUP_TABLE)

    def update_neighbor_spin_nums(self, rand_ind, change):
        '''
        Wrapper for JIT function that updates the number of down-spin neighbors when a spin flips.

        Parameters:
            - rand_ind (int): Index of the spin that flipped.
            - change (int): +1 if the spin flipped from up to down, -1 if flipped from down to up.
        '''
        _update_neighbor_spin_nums(self.neighbor_indices, self.neighbor_down_spins, rand_ind, change)
    
    def calc_mag(self, spins):
        '''
        Wrapper for JIT function that computes the magnetization of the given spin configuration.

        Parameters:
            - spins (np.ndarray): The current spin configuration.

        Returns:
            - float: The magnetization of the system.
        '''
        return _calc_mag(spins)

#------------------------------------------------------------------------------#

################################################################################
######################## SPACE EFFICIENT ISING MODEL ###########################
################################################################################
###-------------- Optimized for large 2D lattice sizes by only --------------###
##----------------- storing the last state and keeping track -----------------##
#--------------- track of neighbors without an adjacency matrix ---------------#

class Ising_2D_Large_Lattice:
    '''
    A memory-efficient implementation of the 2D Ising model optimized for large lattice sizes.
    It avoids storing a full adjacency matrix by only tracking neighbor indices.
    '''
    def __init__(self, L):
        '''
        Initializes the Ising model for a large periodic 2D lattice.

        Parameters:
        L (int): The size of the lattice (L x L grid).
        '''
        self.ROWS = L
        self.COLS = L
        self.LOOKUP_TABLE = np.array([[4, 2, 0, -2, -4], [-4, -2, 0, 2, 4]])
    
    def simulate(self, steps, temp, initial_state=None):
        '''
        Simulates the Ising model using the Metropolis algorithm.

        Parameters:
            - steps (int): Number of simulation steps.
            - temp (float): Temperature of the system.
            - initial_state (np.ndarray, optional): Initial spin configuration (0 for up, 1 for down). 
                    If None, a random initialization is used.

        Returns:
            - tuple[np.ndarray, np.ndarray]: 
                - Final spin configuration (np.ndarray).
                - Magnetization history over time (np.ndarray).
        '''
        np.random.seed(None)
        # Initialize the needed arrays
        self.spins = np.random.choice([0,1], self.ROWS*self.COLS) if (isinstance(initial_state, np.ndarray) or initial_state is None) else initial_state
        # calculate neighbor indices and values without adjacency matrix
        self.neighbor_indices = _mem_optim_get_neighbor_indices(self.ROWS)
        self.neighbor_down_spins = _mem_optim_get_down_neighbors(self.spins, self.neighbor_indices)

        magnetization_history = np.zeros(steps)

        # Initialize simulation variables
        beta = 1 / temp
        E_old = self.calc_energy()

        # main loop
        for step in range(steps):
            # get random array index
            rand_ind = np.random.randint(0, len(self.spins))

            before_spin = self.spins[rand_ind]

            # increase number of down spins if previous spin was up and is now switched down
            change = 1 if before_spin == 0 else -1
            # update neighbors and calculate new energy
            self.update_neighbor_spin_nums(rand_ind, change)
            E_new = self.calc_energy()
            dE = E_new - E_old

            if dE < 0 or np.random.rand() < np.exp(-dE*beta):
                # accept the move and actually change spin
                E_old = E_new
                self.spins[rand_ind] = 1 if before_spin == 0 else 0
            else:
                # revert the number of down spin neighbors
                self.update_neighbor_spin_nums(rand_ind, -change)

            magnetization_history[step] = self.calc_mag(self.spins)
        
        return self.spins, magnetization_history

    def calc_energy(self):
        '''
        Wrapper for JIT function that computes the total energy of the current spin configuration.

        Returns:
            - float: The total energy of the system.
        '''
        return _calc_energy(self.spins, self.neighbor_down_spins, self.LOOKUP_TABLE)

    def update_neighbor_spin_nums(self, rand_ind, change):
        '''
        Wrapper for JIT function that updates the number of down-spin neighbors when a spin flips.

        Parameters:
            - rand_ind (int): Index of the spin that flipped.
            - change (int): +1 if the spin flipped from up to down, -1 if flipped from down to up.
        '''
        _update_neighbor_spin_nums(self.neighbor_indices, self.neighbor_down_spins, rand_ind, change)
    
    def calc_mag(self, spins):
        '''
        Wrapper for JIT function that computes the magnetization of the given spin configuration.

        Parameters:
            - spins (np.ndarray): The current spin configuration.

        Returns:
            - float: The magnetization of the system.
        '''
        return _calc_mag(spins)


################################################################################
########################### N-D SQUARE ISING MODEL #############################
################################################################################
###---------------- Made to simulate a square lattice in any ----------------###
##------------------------ dimension using Metropolis ------------------------##

class Ising_ND_Lattice:
    def __init__(self, side_length, dimension):
        self.L = side_length
        self.dim = dimension
        self.adj_mat = periodic_nD_lattice(self.L, self.dim)
        self.LOOKUP_TABLE = _create_NDim_lookup(self.dim)
        self.arr_size = np.power(self.L, self.dim)
    
    def simulate(self, steps, temp, initial_state=None):
        '''
        Simulates the Ising model using the Metropolis algorithm.

        Parameters:
            - steps (int): Number of simulation steps.
            - temp (float): Temperature of the system.
            - initial_state (np.ndarray, optional): Initial spin configuration (1 for up, 0 for down). 
                If None, a random initialization is used.

        Returns:
            - tuple[np.ndarray, np.ndarray]: 
                - state_history (np.ndarray): Spin configurations over time.
                - magnetization_history (np.ndarray): Magnetization over time.
        '''
        np.random.seed(None)
        # Initialize the needed arrays
        self.spins = np.random.choice([0,1], self.arr_size) if (isinstance(initial_state, np.ndarray) or initial_state is None) else initial_state
        self.neighbor_indices = self.init_neighbor_indices()
        self.neighbor_down_spins = self.get_down_spin_neighbors()

        del self.adj_mat
        gc.collect()

        state_history = np.zeros((steps,self.arr_size))
        magnetization_history = np.zeros(steps)

        # Initialize simulation variables
        beta = 1 / temp
        E_old = self.calc_energy_NDim()

        # main loop
        for step in range(steps):
            # get random array index
            rand_ind = np.random.randint(0, len(self.spins))

            before_spin = self.spins[rand_ind]

            # increase number of down spins if previous spin was up and is now switched down
            change = 1 if before_spin == 0 else -1
            # update neighbors and calculate new energy
            self.update_neighbor_spin_nums(rand_ind, change)
            E_new = self.calc_energy_NDim()
            dE = E_new - E_old

            if dE < 0 or np.random.rand() < np.exp(-dE*beta):
                # accept the move and actually change spin
                E_old = E_new
                self.spins[rand_ind] = 1 if before_spin == 0 else 0
            else:
                # revert the number of down spin neighbors
                self.update_neighbor_spin_nums(rand_ind, -change)

            # state_history[step] = self.spins
            magnetization_history[step] = self.calc_mag(self.spins)
        
        return state_history, magnetization_history
    
    def init_neighbor_indices(self):
        '''
        Initializes an array of neighbor indices for each lattice site.

        Returns:
            - np.ndarray: A 2D array where each row contains indices of neighboring spins.
        '''
        neighbors = []
        for row in range(len(self.adj_mat)):
            nonzeros = np.nonzero(self.adj_mat[row])
            neighbors.append(nonzeros[0])
        
        return np.array(neighbors)
    
    def get_down_spin_neighbors(self):
        '''
        Computes the number of down-spin neighbors for each spin in the lattice.

        Returns:
            - np.ndarray: A 1D array where each element represents the number of down-spin neighbors.
        '''
        neighbors = np.zeros(len(self.spins), dtype=np.int64)
        for i in range(len(self.spins)):
            neighbors[i] = np.dot(self.adj_mat[i], self.spins)
        
        return neighbors

    def update_neighbor_spin_nums(self, rand_ind, change):
        '''
        Wrapper for JIT function that updates the number of down-spin neighbors when a spin flips.

        Parameters:
            - rand_ind (int): Index of the spin that flipped.
            - change (int): +1 if the spin flipped from up to down, -1 if flipped from down to up.
        '''
        _update_neighbor_spin_nums(self.neighbor_indices, self.neighbor_down_spins, rand_ind, change)
    
    def calc_energy_NDim(self):
        '''
        Wrapper for JIT function that computes the total energy of the current spin configuration.

        Returns:
            - float: The total energy of the system.
        '''
        return _calc_energy(self.spins, self.neighbor_down_spins, self.LOOKUP_TABLE)

    def calc_mag(self, spins):
        '''
        Wrapper for JIT function that computes the magnetization of the given spin configuration.

        Parameters:
            - spins (np.ndarray): The current spin configuration.

        Returns:
            - float: The magnetization of the system.
        '''
        return _calc_mag(spins)

#------------------------------------------------------------------------------#

################# NUMBA FUNCTIONS FOR NON-LOOKUP TABLE CLASSES #################

@jit(nopython=True)
def _calc_energy_normal(spins, adj_data, adj_indices, adj_indptr):
    '''
    Non JIT function that computes the total energy of a spin configuration using a sparse adjacency matrix representation.

    Parameters:
        - spins (np.ndarray): A 1D array representing the spin configuration (1 for up, -1 for down).
        - adj_data (np.ndarray): A 1D array containing the nonzero values of the adjacency matrix.
        - adj_indices (np.ndarray): A 1D array containing the column indices corresponding to adj_data.
        - adj_indptr (np.ndarray): A 1D array marking the start and end indices of each row in adj_data and adj_indices.

    Returns:
        - float: The total energy of the system.
    '''
    H = 0
    for i in range(len(spins)):
        start_row = adj_indptr[i]
        end_row = adj_indptr[i+1]

        for ind in range(start_row, end_row):
            other = adj_indices[ind]
            val = adj_data[ind]
            H += val * spins[i] * spins[other]
    return -H

@jit(nopython=True)
def _calc_magnetization_normal(state):
    '''
    Non JIT function to calculate the average magnetization of a given configuration.

    Parameters:
        - State (np.ndarray): Current spin state to calculate magnetization of.
    
    Returns:
        - float: The average magnetization of the given configuration.
    '''
    return np.sum(state) / len(state)

################################################################################
############################ BASIC 2D ISING MODEL ##############################
################################################################################
###----------------- First implementation w/o lookup table ------------------###
##---------------------- using the Metropolis algorithm ----------------------##

class Ising_2D_Lattice_Unoptimized:
    '''
    A class implementing the 2D Ising model on a periodic lattice using a sparse
    adjacency matrix representation. Does not use a lookup table.
    '''
    def __init__(self, rows, cols):
        '''
        Initializes the Ising model for a 2D periodic lattice.

        Parameters:
            - rows (int): The number of rows in the lattice.
            - cols (int): The number of columns in the lattice.
        '''
        self.ROWS = rows
        self.COLS = cols
        self.adjacency_mat = periodic_2D_lattice(self.ROWS, self.COLS, True)
    
    def calc_energy(self, state):
        '''
        Wrapper for JIT function that computes the total energy of the given spin configuration.

        Parameters:
            - state (np.ndarray): A 1D array representing the spin configuration (-1 for down, 1 for up).

        Returns:
            - float: The total energy of the system.
        '''
        return _calc_energy_normal(state, self.adjacency_mat.data, self.adjacency_mat.indices, self.adjacency_mat.indptr)

    def simulate(self, steps, temp, starting_state=None):
        '''
        Runs the Metropolis-Hastings algorithm to simulate the Ising model over a given number of steps.

        Parameters:
            - steps (int): Number of simulation steps.
            - temp (float): Temperature of the system.
            - starting_state (np.ndarray, optional): Initial spin configuration (-1 for down, 1 for up). 
                    If None, a random initialization is used.

        Returns:
            - tuple[np.ndarray, np.ndarray]: 
                - State history over time (2D array of shape (steps, ROWS * COLS)).
                - Magnetization history over time (1D array of shape (steps)).
        '''
        lattice_state = np.random.choice([-1,1], self.ROWS*self.COLS) if (isinstance(starting_state, np.ndarray) or starting_state is None) else starting_state
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
    
    def calc_magnetization(self, state):
        '''
        Wrrapper for JIT function that computes the magnetization of the given spin configuration.

        Parameters:
            - state (np.ndarray): The current spin configuration.

        Returns:
            - float: The magnetization of the system, defined as the sum of all spin values.
        '''
        return _calc_magnetization_normal(state)
