#!/bin/env python

import iemgmm, icgmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
from functools import partial

# set up RNG
seed = 42
from numpy.random import RandomState
rng = RandomState(seed)

def plotResults(data, sel, gmm, patch=None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal')

    # plot inner and outer points
    ax.plot(data[:,0][sel], data[:,1][sel], 'bo', mec='None')
    ax.plot(data[:,0][sel==False], data[:,1][sel==False], 'o', mfc='None', mec='b')

    # prediction
    B = 100
    x,y = np.meshgrid(np.linspace(-5,15,B), np.linspace(-5,15,B))
    coords = np.dstack((x.flatten(), y.flatten()))[0]

    # compute sum_k(p_k(x)) for all x
    logL_i = gmm.logL(coords)
    # for better visibility use arcshinh stretch
    p = np.arcsinh(np.exp(logL_i.reshape((B,B)))/1e-4)
    cs = ax.contourf(p, 10, extent=(-5,15,-5,15), cmap=plt.cm.Greys)
    for c in cs.collections:
        c.set_edgecolor(c.get_facecolor())

    # plot boundary
    if patch is not None:
        import copy
        if hasattr(patch, '__iter__'):
            for p in patch:
                ax.add_artist(copy.copy(p))
        else:
            ax.add_artist(copy.copy(patch))

    # add complete data logL to plot
    logL = gmm.logL(data).mean()
    ax.text(0.05, 0.95, '$\log{\mathcal{L}} = %.3f$' % logL, ha='left', va='top', transform=ax.transAxes)
    
    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def getBox(coords):
    box_limits = np.array([[0,0],[10,10]])
    return (coords[:,0] > box_limits[0,0]) & (coords[:,0] < box_limits[1,0]) & (coords[:,1] > box_limits[0,1]) & (coords[:,1] < box_limits[1,1])

def getHole(coords):
    x,y,r = 6.5, 6., 2
    return ((coords[:,0] - x)**2 + (coords[:,1] - y)**2 > r**2)

def getBoxWithHole(coords):
    return getBox(coords)*getHole(coords)

def getHalfDensity(coords, rng=np.random):
    mask = np.ones(coords.shape[0], dtype='bool')
    mask[rng.rand(coords.shape[0]) < 0.5] = 0
    return mask

def getTaperedDensity(coords, rng=np.random):
    mask = np.ones(coords.shape[0], dtype='bool')
    mask[rng.rand(coords.shape[0]) < coords[:,0]/7] = 0
    return mask

def getCut(coords):
    return (coords[:,0] < 5)

# draw N points from 3-component GMM
N = 400
D = 2
gmm = icgmm.ICGMM(K=3, D=2, rng=rng)
gmm.amp[:] = np.array([ 0.36060026,  0.27986906,  0.206774])
gmm.amp /= gmm.amp.sum()
gmm.mean[:,:] = np.array([[ 0.08016886,  0.21300697],
                          [ 0.70306351,  0.6709532 ],
                          [ 0.01087670,  0.852077]])*10
gmm.covar[:,:,:] = np.array([[[ 0.08530014, -0.00314178],
                              [-0.00314178,  0.00541106]],
                             [[ 0.03053402, 0.0125736],
                              [0.0125736,  0.01075791]],
                             [[ 0.00258605,  0.00409287],
                             [ 0.00409287,  0.01065186]]])*100

orig = gmm.draw(N)

K = 3
R = 10

# limit data to within the box
#cb = getBoxWithHole
#ps = [patches.Rectangle([0,0], 10, 10, fc="none", ec='b', ls='dotted'),
#      patches.Circle([6.5, 6.], radius=2, fc="none", ec='b', ls='dotted')]
cb = getBox
ps = patches.Rectangle([0,0], 10, 10, fc="none", ec='b', ls='dotted')
#cb = getHole
#ps = patches.Circle([6.5, 6.], radius=2, fc="none", ec='b', ls='dotted')
#cb = partial(getTaperedDensity, rng=rng)
#ps = None
#cb = getCut
#from matplotlib.path import Path
#path = Path([(5,-5), (5,15)], [Path.MOVETO, Path.LINETO])
#ps = patches.PathPatch(path, fc="none", ec='b', ls='dotted')

sel = cb(orig)
data = orig[sel]

"""
new_gmm = icgmm.ICGMM(K=K, data=data, cutoff=10, w=0.1, rng=rng, verbose=False)
plotResults(orig, sel, new_gmm, patch=ps)

new_gmm = icgmm.ICGMM(K=K, data=data, cutoff=10, w=0.1, sel_callback=cb, n_missing=(sel==False).sum())#, verbose=True)
plotResults(orig, sel, new_gmm, patch=ps)

new_gmm = icgmm.ICGMM(K=K, data=data, cutoff=10, w=0.1, sel_callback=cb, n_missing=None, rng=rng, verbose=False)
plotResults(orig, sel, new_gmm, patch=ps)
"""

# 1) GMM with imputation
imp = iemgmm.GMM(K=K*R, D=D)
ll = np.empty(R)
"""
start = datetime.datetime.now()
rng = RandomState(seed)
for r in xrange(R):
    imp_ = iemgmm.GMM(K=K, data=data, w=0.1, n_missing=(sel==False).sum(), sel_callback=cb, rng=rng, verbose=False)
    ll[r] = imp_.logL(data).mean()
    imp.amp[r*K:(r+1)*K] = imp_.amp
    imp.mean[r*K:(r+1)*K,:] = imp_.mean
    imp.covar[r*K:(r+1)*K,:,:] = imp_.covar
imp.amp /= imp.amp.sum()
print "execution time %ds" % (datetime.datetime.now() - start).seconds
plotResults(orig, sel, imp, patch=ps)

for r in xrange(R):
    imp.amp[r*K:(r+1)*K] *= np.exp(ll[r])
imp.amp /= imp.amp.sum()
plotResults(orig, sel, imp, patch=ps)


# 2) ICGMM with imputation
start = datetime.datetime.now()
rng = RandomState(seed)
for r in xrange(R):
    imp_ = icgmm.ICGMM(K=K, data=data, w=0.1, cutoff=10, sel_callback=cb, n_missing=(sel==False).sum(), rng=rng, verbose=False)
    ll[r] = imp_.logL(data).mean()
    imp.amp[r*K:(r+1)*K] = imp_.amp
    imp.mean[r*K:(r+1)*K,:] = imp_.mean
    imp.covar[r*K:(r+1)*K,:,:] = imp_.covar
imp.amp /= imp.amp.sum()
print "execution time %ds" % (datetime.datetime.now() - start).seconds
plotResults(orig, sel, imp, patch=ps)

for r in xrange(R):
    imp.amp[r*K:(r+1)*K] *= np.exp(ll[r])
imp.amp /= imp.amp.sum()
plotResults(orig, sel, imp, patch=ps)
"""

# 3) ICGMM with imputation but unknown n_missing
start = datetime.datetime.now()
rng = RandomState(seed)
for r in xrange(R):
    imp_ = icgmm.ICGMM(K=K, data=data, w=0.1, cutoff=10, sel_callback=cb, n_missing=None, rng=rng, verbose=False)
    ll[r] = imp_.logL(data).mean()
    imp.amp[r*K:(r+1)*K] = imp_.amp
    imp.mean[r*K:(r+1)*K,:] = imp_.mean
    imp.covar[r*K:(r+1)*K,:,:] = imp_.covar
imp.amp /= imp.amp.sum()
print "execution time %ds" % (datetime.datetime.now() - start).seconds
plotResults(orig, sel, imp, patch=ps)

for r in xrange(R):
    imp.amp[r*K:(r+1)*K] *= np.exp(ll[r])
imp.amp /= imp.amp.sum()
plotResults(orig, sel, imp, patch=ps)


