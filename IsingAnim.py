from matplotlib import pyplot as plt
import matplotlib.animation as animation

class Anim2DGridIsing:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure()
    
    def animate(self, state_history):
        ims = []
        for i in range(len(spin_history)):
            im = plt.imshow(np.reshape(spin_history[i], (self.rows,self.cols), 'C'), cmap='binary', animated=True)
            ims.append([im])

        return animation.ArtistAnimation(self.fig, ims, interval=30, blit=True, repeat_delay=1000)