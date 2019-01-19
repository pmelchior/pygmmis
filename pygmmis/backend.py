import warnings
from collections import OrderedDict, namedtuple
from itertools import cycle
import numpy as np


__all__ = ['Backend']

class BackendBase(object):
    def __init__(self, name):
        self.name = name
        self.varnames = None

        self.iteration = 0
        self.arrays = {}
        self._length = 0


    def setup(self, length, **varvalues):
        self.varnames = list(varvalues.keys())
        for k, v in varvalues.items():
            v = np.asarray(v)
            self.arrays[k] = np.zeros((length, ) + v.shape, dtype=v.dtype)
        self._length = length


    def grow(self, length):
        space_left = self._length - self.iteration
        space_needed = length - space_left
        for k, array in self.arrays.items():
            self.arrays[k] = np.append(array, np.zeros((space_needed,) + array.shape[1:], dtype=array.dtype), axis=0)
        self._length += space_needed


    def save(self, **varvalues):
        try:
            for k, v in varvalues.items():
                self.arrays[k][self.iteration] = np.asarray(v)
            self.iteration += 1
        except IndexError as e:
            self.grow(1)
            self.save(**varvalues)
            warnings.warn("Backend is full, now expanding backend 1 step at a time. "
                        "This is really inefficient. Use backend.grow(x) to make more space")


    def __getitem__(self, item):
        if isinstance(item, int):
            return {v: self.get_values(v, item) for v in self.varnames}

        if isinstance(item, str):
            return self.get_values(item)

        if isinstance(item, tuple):
            return self.get_values(item[0], item[1:])


    def get_values(self, varname, index=None):
        variable = self.arrays[varname]
        if index is None:
            index = slice(0, self.iteration)
        if isinstance(index, slice):
            if index.stop > self.iteration:
                raise IndexError("Cannot get slices {} above current iteration {}".format(index, self.iteration))
        if isinstance(index, tuple):
            if index[0] < 0:
                index = (self.iteration + index[0], ) + index[1:]
        elif isinstance(index, int):
            if index < 0:
                index += self.iteration
        return variable[index]


    def __len__(self):
        return self.iteration

    def __repr__(self):
        return "<XDGMM Backend - {} iterations ({})>".format(self.iteration, self.varnames)

    def __getattr__(self, item):
        return self.arrays[item]

Event = namedtuple('Event', ['event', 'origin', 'origin_iteration', 'into', 'into_iteration'])


class MultiBackend(object):
    def __init__(self, name, backend_type):
        self.name = name
        self.backend_type = backend_type
        self.varnames = None
        self.store = OrderedDict(master=backend_type('master'))
        self.events = []
        self.current_name = 'master'

    def print_events(self):
        for event in self.events:
            print("{}[{}] -{}-> {}[{}]".format(event.origin, event.origin_iteration, event.event, event.into, event.into_iteration))

    def draw_tree(self):
        import matplotlib.pyplot as plt
        import networkx as nx
        ypos = {k: -i for i, k in enumerate(self.store.keys())}

        G = nx.MultiDiGraph()
        G.add_node('master[0]', pos=[0, 0])
        for event in self.events:
            if event.event != 'switch':
                old = "{}[{}]".format(event.origin, event.origin_iteration)
                new = "{}[{}]".format(event.into, event.into_iteration)

                if old not in G:
                    chain = [int(i.split('[')[1][:-1]) for i in G.nodes.keys() if event.origin in i]
                    previous = "{}[{}]".format(event.origin, max(chain))
                    G.add_edge(previous, old)
                    xpos = G.nodes[previous]['pos'][0] + 1
                    G.nodes[old]['pos'] = [xpos, ypos[event.origin]]
                G.add_edge(old, new)
                xpos = G.nodes[old]['pos'][0] + 1
                G.nodes[new]['pos'] = [xpos, ypos[event.into]]
            else:
                chain = [int(i.split('[')[1][:-1]) for i in G.nodes.keys() if event.into in i]
                previous = "{}[{}]".format(event.into, max(chain))
                new = "{}[{}]".format(event.into, event.into_iteration)
                G.add_edge(previous, new)
                xpos = G.nodes[previous]['pos'][0] + 1
                G.nodes[new]['pos'] = [xpos, ypos[event.into]]

        pos = {k: v['pos'] for k, v in G.nodes.items()}
        nx.draw(G, pos, with_labels=True)

    @property
    def current(self):
        return self.store[self.current_name]

    @property
    def master(self):
        return self.store['master']

    def grow(self, length):
        return self.current.grow(length)

    def get_values(self, varname, index=None):
        return self.current.get_values(varname, index)

    def __getattr__(self, item):
        return self.current.__getattr__(item)

    def __getitem__(self, item):
        return self.current.__getitem__(item)

    def setup(self, length, **varvalues):
        self.master.setup(length, **varvalues)
        self.varnames = self.master.varnames

    def save(self, **varvalues):
        self.current.save(**varvalues)

    def branch_chain(self, length, name):
        new = self.backend_type(name)
        new.setup(length, **self.master[0])
        self.store[name] = new
        self.events.append(Event('branch', self.current_name, self.store[self.current_name].iteration, name, 0))
        self.switch_chain(name)

    def merge_chain(self, into):
        self.store[into].grow(len(self.current))
        new_length = len(self.store[into])
        for i in range(len(self.current)):
            self.store[into].save(**self.current[i])
        self.events.append(Event('merge', self.current_name, self.store[self.current_name].iteration, into, new_length))
        self.switch_chain(into)

    def switch_chain(self, name):
        self.events.append(Event('switch', self.current_name, self.store[self.current_name].iteration, name, self.store[name].iteration))
        self.current_name = name

    @property
    def iteration(self):
        return len(self.current)

    def __len__(self):
        return max([len(s) for s in self.store.values()])

    def __repr__(self):
        return "<GMMBackend - {} iterations - {} chains (current={}) ({})>".format(self.iteration, len(self.store), self.current_name, self.varnames)


def Backend(name, mode='npy'):
    return MultiBackend(name, BackendBase)


if __name__ == '__main__':
    # backend = Backend('backend', ['a', 'b', 'c'])
    #
    a = np.ones(3)
    b = np.eye(5)
    c = 1
    #
    # backend.setup(1, a=a, b=b, c=c)
    # backend.save(a=a, b=b, c=c)

    store = Backend('backend')
    store.setup(2, a=a, b=b, c=c)
    store.save(a=a, b=b, c=c)
    store.save(a=a, b=b, c=c)

    store.branch_chain(2, 'event-1')
    store.save(a=a, b=b, c=c)
    store.save(a=a*2, b=b*2, c=c*2)
    store.merge_chain('master')

    store.branch_chain(1, 'event-2')
    store.save(a=a, b=b, c=c)

    store.switch_chain('master')
    store.grow(1)
    store.save(a=a, b=b, c=c)

    import matplotlib.pyplot as plt
    store.draw_tree()
    store.print_events()
    plt.show()