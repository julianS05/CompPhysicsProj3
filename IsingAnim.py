from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Anim2DGridIsing:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure()
    
    def animate(self, state_history):
        ims = []
        for i in range(len(state_history)):
            im = plt.imshow(np.reshape(state_history[i], (self.rows,self.cols), 'C'), cmap='binary', animated=True)
            ims.append([im])

        return animation.ArtistAnimation(self.fig, ims, interval=30, blit=True, repeat_delay=1000)