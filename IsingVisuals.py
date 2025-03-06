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

def show_snapshots(state_history, grid_rows, grid_cols, indices, img_rows, img_cols):
    fig = plt.figure()
    for i in range(len(indices)):
        plt.subplot(img_rows, img_cols, i+1)
        plt.imshow(np.reshape(state_history[indices[i]], (grid_rows,grid_cols), 'C'), cmap='binary')
        plt.axis('off')  # Hide the axis labels
        plt.title(f"Step {indices[i]+1}") 
    
    fig.tight_layout()
    plt.show()
