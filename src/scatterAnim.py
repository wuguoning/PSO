#========================================================
"""
Scatter Plot Animation

Author:    Gordon Woo
Email:      wuguoning@gmail.com
Department: China University of Petroleum at Beijing
Date:       2021-01-04
"""
#========================================================
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints, datain, n3, func, bounds):
        """
        Parameters:
            numpoints: number of points
            datain:    input data
            n3:        number of iteration
            fun:       optimization function
        """
        self.numpoints = numpoints
        self.datain = datain
        self.n3 = n3
        self.func = func
        self.bounds = bounds
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(16,9))
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=360,
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        self.ax.axis(self.bounds)
        self.ax.plot(0,0,'r*',markersize=20)
        # plot background
        x = np.arange(self.bounds[0], self.bounds[1], 0.01)
        y = np.arange(self.bounds[2], self.bounds[3], 0.01)
        X,Y = np.meshgrid(x,y)
        Z = self.func(X, Y)
        cs = self.ax.contourf(X, Y, Z, cmap=plt.cm.jet)
        self.ax.contour(cs,colors='k')

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""

        s, c = np.random.random((self.numpoints, 2)).T
        i=0
        while True:
            i = i%self.n3
            xy = self.datain[i]
            i = i+1
            yield np.c_[xy[0,:], xy[1,:], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


