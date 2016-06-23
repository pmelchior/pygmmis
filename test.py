#!/bin/env python

import pygmmi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import datetime
from functools import partial
from sklearn.neighbors import KDTree

def plotResults(orig, data, gmm, l, patch=None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal')

    # plot inner and outer points
    ax.plot(orig[:,0], orig[:,1], 'o', mfc='r', mec='None')
    ax.plot(data[:,0], data[:,1], 'o', mfc='b', mec='None')

    # prediction
    B = 100
    x,y = np.meshgrid(np.linspace(-5,15,B), np.linspace(-5,15,B))
    coords = np.dstack((x.flatten(), y.flatten()))[0]

    # weight with logL of each run
    gmm_ = gmm
    gmm_.amp = (np.split(gmm_.amp, l.size) * l[:,None]).flatten()
    gmm_.amp /= gmm_.amp.sum()

    # compute sum_k(p_k(x)) for all x
    p = gmm_(coords).reshape((B,B))
    # for better visibility use arcshinh stretch
    p = np.arcsinh(p/1e-4)
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
    logL = gmm_(orig, as_log=True).mean()
    ax.text(0.05, 0.95, '$\log{\mathcal{L}} = %.3f$' % logL, ha='left', va='top', transform=ax.transAxes)

    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def plotDifferences(orig, data, gmm, R, l, patch=None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal')

    # plot inner and outer points
    ax.plot(orig[:,0], orig[:,1], 'o', mfc='r', mec='None')
    ax.plot(data[:,0], data[:,1], 'o', mfc='b', mec='None')

    # prediction
    B = 100
    x,y = np.meshgrid(np.linspace(-5,15,B), np.linspace(-5,15,B))
    coords = np.dstack((x.flatten(), y.flatten()))[0]

    # weight with logL of each run
    gmm_ = gmm
    gmm_.amp = (np.split(gmm_.amp, l.size) * l[:,None]).flatten()
    gmm_.amp /= gmm_.amp.sum()

    # compute sum_k(p_k(x)) for all x
    pw = gmm_(coords).reshape((B,B))

    # use each run and compute weighted std
    K = gmm.K / R
    p = np.empty((R,B,B))
    for r in xrange(R):
        comps = np.arange(r*K, (r+1)*K)
        gmm_ = pygmmi.GMM(K=K, D=gmm.D)
        gmm_.amp[:] = gmm.amp[comps]
        gmm_.amp /= gmm_.amp.sum()
        gmm_.mean[:,:] = gmm.mean[comps,:]
        gmm_.covar[:,:,:] = gmm.covar[comps,:,:]

        # compute sum_k(p_k(x)) for all x
        p[r,:,:] = gmm_(coords).reshape((B,B))

    p = ((p-pw[None,:,:])**2 * l[:,None, None]).sum(axis=0)
    V1 = l.sum()
    V2 = (l**2).sum()
    p /= (V1 - V2/V1)

    p = np.arcsinh(np.sqrt(p)/1e-4)
    cs = ax.contourf(p, 10, extent=(-5,15,-5,15), cmap=plt.cm.Greys, vmin=np.arcsinh(pw/1e-4).min(), vmax=np.arcsinh(pw/1e-4).max())
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

    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def plotCoverage(orig, data, gmm, patch=None, sel_callback=None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal')

    # plot inner and outer points
    ax.plot(orig[:,0], orig[:,1], 'o', mfc='r', mec='None')
    ax.plot(data[:,0], data[:,1], 'o', mfc='b', mec='None')

    # prediction
    B = 100
    x,y = np.meshgrid(np.linspace(-5,15,B), np.linspace(-5,15,B))
    coords = np.dstack((x.flatten(), y.flatten()))[0]

    # compute sum_k(p_k(x)) for all x
    coverage = getCoverage(gmm, coords, sel_callback=sel_callback).reshape((B,B))
    cs = ax.contourf(coverage, 10, extent=(-5,15,-5,15), cmap=plt.cm.Greys)
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

    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def getCoverage(gmm, coords, sel_callback=None, repeat=5, rotate=True):
    # create a new gmm with randomly drawn components at each point in coords:
    # estimate how this gmm can cover the volume spanned by coords
    if sel_callback is None:
        return np.ones(len(coords))
    else:
        coverage = np.zeros(len(coords))
        from sklearn.neighbors import KDTree
        for r in xrange(repeat):
            sel = sel_callback(coords)
            inv_sel = sel == False
            coverage[sel] += 1./repeat

            gmm_ = pygmmi.GMM(K=gmm.K, D=gmm.D)
            gmm_.amp = np.random.rand(K)
            gmm_.amp /= gmm_.amp.sum()
            gmm_.covar = gmm.covar

            if rotate:
                # use random rotations for each component covariance
                # from http://www.mathworks.com/matlabcentral/newsreader/view_thread/298500
                # since we don't care about parity flips we don't have to check
                # the determinant of R (and hence don't need R)
                for k in xrange(gmm_.K):
                    Q,_ = np.linalg.qr(np.random.normal(size=(gmm.D, gmm.D)), mode='complete')
                    gmm_.covar[k] = np.dot(Q, np.dot(gmm_.covar[k], Q.T))

            inside = coords[sel]
            outside = coords[inv_sel]
            outside_cov = coverage[inv_sel]
            tree = KDTree(inside)
            closest_inside = tree.query(outside, k=1, return_distance=False).flatten()
            unique_closest = np.unique(closest_inside)
            for c in unique_closest:
                gmm_.mean[:] = inside[c]
                closest_to_c = (closest_inside == c)
                outside_cov[closest_to_c] += gmm_(outside[closest_to_c]) / gmm_(inside[c]) / repeat
            coverage[inv_sel] = outside_cov
    return coverage

def getBox(coords, gmm=None):
    box_limits = np.array([[0,0],[10,10]])
    return (coords[:,0] > box_limits[0,0]) & (coords[:,0] < box_limits[1,0]) & (coords[:,1] > box_limits[0,1]) & (coords[:,1] < box_limits[1,1])

def getHole(coords, gmm=None):
    x,y,r = 6.5, 6., 2
    return ((coords[:,0] - x)**2 + (coords[:,1] - y)**2 > r**2)

def getBoxWithHole(coords, gmm=None):
    return getBox(coords)*getHole(coords)

def getHalfDensity(coords, gmm=None, rng=np.random):
    mask = np.ones(coords.shape[0], dtype='bool')
    mask[rng.rand(coords.shape[0]) < 0.5] = 0
    return mask

def getTaperedDensity(coords, gmm=None, rng=np.random):
    mask = np.ones(coords.shape[0], dtype='bool')
    mask[rng.rand(coords.shape[0]) < coords[:,0]/8] = 0
    return mask

def getCut(coords, gmm=None):
    return (coords[:,0] < 7)

def getOver(coords, gmm):
    p_x = gmm(coords)
    return p_x > 0.01

def getUnder(coords, gmm):
    p_x = gmm(coords)
    return p_x < 0.025

def getSelection(type="hole", rng=np.random):
    if type == "hole":
        cb = getHole
        ps = patches.Circle([6.5, 6.], radius=2, fc="none", ec='b', ls='dotted')
    if type == "box":
        cb = getBox
        ps = patches.Rectangle([0,0], 10, 10, fc="none", ec='b', ls='dotted')
    if type == "boxWithHole":
        cb = getBoxWithHole
        ps = [patches.Circle([6.5, 6.], radius=2, fc="none", ec='b', ls='dotted'),
            patches.Rectangle([0,0], 10, 10, fc="none", ec='b', ls='dotted')]
    if type == "cut":
        cb = getCut
        ps = lines.Line2D([7, 7],[-5, 15], ls='dotted', color='b')
    if type == "tapered":
        cb = partial(getTaperedDensity, rng=rng)
        ps = lines.Line2D([8, 8],[-5, 15], ls='dotted', color='b')
    if type == "under":
        cb = getUnder
        ps = None
    if type == "over":
        cb = getOver
        ps = None
    return cb, ps

if __name__ == '__main__':

    # set up test
    seed = 8422#np.random.randint(1, 10000)
    from numpy.random import RandomState
    rng = RandomState(seed)
    pygmmi.VERBOSITY = 2
    w = 0.1
    cutoff = 3

    # draw N points from 3-component GMM
    N = 400
    D = 2
    gmm = pygmmi.GMM(K=3, D=2)
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

    orig = gmm.draw(N, rng=rng)

    # get observational selection function
    cb, ps = getSelection("cut", rng=rng)

    # add isotropic errors on data
    disp = 0.8
    noisy = orig + rng.normal(0, scale=disp, size=(len(orig), D))
    # apply selection
    sel = cb(noisy, gmm)
    data = pygmmi.createShared(noisy[sel])
    covar = disp**2 * np.eye(D)
    tree = KDTree(data, leaf_size=20)

    # plot data vs true model
    plotResults(orig, data, gmm, np.ones(1), patch=ps)

    # repeated runs: store results and logL
    K = 3
    R = 10
    gmm_ = pygmmi.GMM(K=K, D=D)
    avg = pygmmi.GMM(K=K*R, D=D)
    l = np.empty(R)

    # 1) run without imputation, ignoring errors
    start = datetime.datetime.now()
    rng = RandomState(seed)
    for r in xrange(R):
        pygmmi.fit(gmm_, data, w=w, cutoff=cutoff, rng=rng, tree=tree)
        l[r] = gmm_(data).mean()
        avg.amp[r*K:(r+1)*K] = gmm_.amp
        avg.mean[r*K:(r+1)*K,:] = gmm_.mean
        avg.covar[r*K:(r+1)*K,:,:] = gmm_.covar
    avg.amp /= avg.amp.sum()
    print "execution time %ds" % (datetime.datetime.now() - start).seconds
    plotResults(orig, data, avg, l, patch=ps)

    """
    # 2) run without imputation, incorporating errors
    start = datetime.datetime.now()
    rng = RandomState(seed)
    for r in xrange(R):
        pygmmi.fit(gmm_, data, covar=covar, w=w, cutoff=cutoff, rng=rng)
        l[r] = gmm_(data).mean()
        avg.amp[r*K:(r+1)*K] = gmm_.amp
        avg.mean[r*K:(r+1)*K,:] = gmm_.mean
        avg.covar[r*K:(r+1)*K,:,:] = gmm_.covar
    avg.amp /= avg.amp.sum()
    print "execution time %ds" % (datetime.datetime.now() - start).seconds
    plotResults(orig, data, avg, l, patch=ps)
    """

    # 3) run with imputation, igoring errors
    # We need a better init function to allow the model to
    # start from a good initial location and to explore the
    # volume that is spanned by the missing part of the data
    # NOTE: You want to choose this carefully, depending
    # on the missingness mechanism.
    init_cb = partial(pygmmi.initFromSimpleGMM, w=w, cutoff=cutoff, covar_factor=4., tree=tree)
    start = datetime.datetime.now()
    rng = RandomState(seed)
    for r in xrange(R):
        pygmmi.fit(gmm_, data, init_callback=init_cb, w=w,  cutoff=cutoff, sel_callback=cb, rng=rng, tree=tree)
        l[r] = gmm_(data).mean()
        avg.amp[r*K:(r+1)*K] = gmm_.amp
        avg.mean[r*K:(r+1)*K,:] = gmm_.mean
        avg.covar[r*K:(r+1)*K,:,:] = gmm_.covar
    avg.amp /= avg.amp.sum()
    print "execution time %ds" % (datetime.datetime.now() - start).seconds
    plotResults(orig, data, avg, l, patch=ps)

    # 4) run with imputation, incorporating errors
    start = datetime.datetime.now()
    rng = RandomState(seed)
    for r in xrange(R):
        pygmmi.fit(gmm_, data, covar=covar, init_callback=init_cb, w=w, cutoff=cutoff, sel_callback=cb, rng=rng, tree=tree)
        l[r] = gmm_(data).mean()
        avg.amp[r*K:(r+1)*K] = gmm_.amp
        avg.mean[r*K:(r+1)*K,:] = gmm_.mean
        avg.covar[r*K:(r+1)*K,:,:] = gmm_.covar
    avg.amp /= avg.amp.sum()
    print "execution time %ds" % (datetime.datetime.now() - start).seconds
    plotResults(orig, data, avg, l, patch=ps)
    #plotDifferences(orig, data, avg, R, l, patch=ps)
    #plotCoverage(orig, data, avg, patch=ps, sel_callback=cb)
