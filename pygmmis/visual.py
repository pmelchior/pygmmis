import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_ellipse(ax, mu, covariance, color, linewidth=2, alpha=0.5):
    var, U = np.linalg.eig(covariance)
    angle = 180. / np.pi * np.arccos(np.abs(U[0, 0]))
    e = Ellipse(mu, 2 * np.sqrt(5.991 * var[0]),
                2 * np.sqrt(5.991 * var[1]),
                angle=angle)
    e.set_alpha(alpha)
    e.set_linewidth(linewidth)
    e.set_edgecolor(color)
    e.set_facecolor(color)
    e.set_fill(False)
    ax.add_artist(e)
    return e


def plot_centre(ax, mu, color):
    return ax.scatter(*mu.T, s=10, alpha=0.5, color=color)

def plot_direction(ax, old, new, **kwargs):
    return ax.arrow(*old, *new-old, **kwargs)


class GMMTracker(object):
    def __init__(self, backend, data):
        assert backend.mu.shape[-1] == 2
        self.backend = backend
        self.data = data
        self.artists = []
        self.n = None
        self.fig, self.axes = None, None
        self.n_components = self.backend.mu.shape[1]
        self.ndim = self.backend.mu.shape[2]

    def figure(self):
        a = np.sqrt(self.n_components)
        shape = [int(np.floor(a)), int(np.ceil(a))]
        if np.prod(shape) < self.n_components:
            shape[0] += 1
        self.fig = plt.figure()
        self.axes = []
        ax = None
        for i in range(self.n_components):
            ax = self.fig.add_subplot(shape[0], shape[1], i+1, sharex=ax, sharey=ax)
            ax.scatter(*self.data.T, s=1, alpha=0.4)
            ax.set_title(i)
            self.axes.append(ax)


    def plot(self, n, clear=True, color='k'):
        if clear:
            self.clear()
        if n < 0:
            n = len(self.backend) + n
        self.n = n
        if self.axes is None:
            self.figure()
        for i, ax in enumerate(self.axes):
            e = plot_ellipse(ax, self.backend.mu[n][i], self.backend.V[n][i], color)
            c = plot_centre(ax, self.backend.mu[n][i], color)
            self.artists.append(e)
            self.artists.append(c)
            if n < len(self.backend.mu)-1:
                direction = plot_direction(ax, self.backend.mu[n][i], self.backend.mu[n+1][i], color=color, label='EMstep')
                self.artists.append(direction)


    def clear(self):
        for a in self.artists:
            a.remove()
        self.artists = []

    def next(self):
        if self.n == len(self.backend)-1:
            raise IndexError("No more iterations left")
        if self.n is None:
            self.n = 0
            self.plot(0)
        self.n += 1
        self.plot(self.n)

    def previous(self):
        if (self.n is None) or (self.n == 0):
            raise IndexError("You are at the start")
        self.n -= 1
        self.plot(self.n)

    def __len__(self):
        return len(self.backend)

    def plot_trace(self, start=0, stop=-1, step=1):
        self.clear()
        if stop < 0:
            stop += len(self)
        cmap = matplotlib.cm.get_cmap('viridis')
        from matplotlib.colors import Normalize
        mappable = matplotlib.cm.ScalarMappable(norm=Normalize(0, stop), cmap=cmap)
        mappable.set_array(np.arange(0, stop))
        ranged = range(start, stop, step)
        for i in ranged:
            color = cmap((i - ranged.start) / (ranged.stop - ranged.start))
            self.plot(i, clear=False, color=color)
