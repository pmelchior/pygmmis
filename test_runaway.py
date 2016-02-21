#!/bin/env python

import iemgmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
from functools import partial

def plotResults(orig, data, gmm, patch=None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal')

    # plot inner and outer points
    ax.plot(orig[:,0], orig[:,1], 'bo', mec='None')
    ax.plot(data[:,0], data[:,1], 'o', mfc='None', mec='b')

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
    logL = gmm.logL(orig).mean()
    ax.text(0.05, 0.95, '$\log{\mathcal{L}} = %.3f$' % logL, ha='left', va='top', transform=ax.transAxes)

    ax.set_xlim(-5, 15)
    ax.set_ylim(-5, 15)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def plotTraces(filename='logfile.txt'):
    rw = np.loadtxt(filename)

    grad = np.eye(len(rw))
    grad[0,0] = 0
    for i in xrange(1,len(rw)):
        grad[i,i-1] = -1
    from functools import partial
    gradf = partial(np.dot, grad)

    def gradf_n(x):
        return gradf(x) / gradf(rw[:,5])
    def gradn_f(x):
        return gradf(rw[:,5]) / gradf(x)

    fig = plt.figure(figsize=(6,15))
    ax = fig.add_subplot(511)
    ax.plot(rw[:,0], rw[:,9], label='0')
    ax.plot(rw[:,0], rw[:,12], label='1')
    ax.plot(rw[:,0], rw[:,15], label='2')
    ax.plot(rw[:,0], gmm.amp[0]*np.ones(len(rw)), 'k:')
    ax.plot(rw[:,0], gmm.amp[1]*np.ones(len(rw)), 'k:')
    ax.plot(rw[:,0], gmm.amp[2]*np.ones(len(rw)), 'k:')
    ax.plot(rw[:,0], rw[:,6]/rw[:,5], 'k-', label='N_m/N_o')
    ax.set_ylabel(r'$\alpha_k$')
    ax.legend(frameon=False)

    ax = fig.add_subplot(512)
    ax.plot(rw[:,0], rw[:,2], 'k-', label='lnL')
    ax.plot(rw[:,0], rw[:,3], 'b-', label='ln_o')
    ax.plot(rw[:,0], rw[:,4], 'r-', label='lnL_m')
    ax.legend(frameon=False)

    ax = fig.add_subplot(513)
    ax.plot(rw[:,0], rw[:,5] * gradf(rw[:,7]), label='N*d<lnL_o>/dt')
    ax.plot(rw[:,0], rw[:,6] * gradf(rw[:,8]), label='N_m*d<lnL_m>/dt')
    ax.plot(rw[:,0], gradf(rw[:,6]) * rw[:,8], label='<lnL_m>dN_m/dt')
    ax.plot(rw[:,0], rw[:,5] * gradf(rw[:,7]) + rw[:,6] * gradf(rw[:,8]) + gradf(rw[:,6]) * rw[:,8], 'k--', label='cont')
    ax.plot(rw[:,0], gradf(rw[:,2]), 'k-', label='dlnL/dt')
    ax.legend(frameon=False)

    limit = -1. / rw[:,8] * (rw[:,5] * gradf(rw[:,7]) + rw[:,6] * gradf(rw[:,8]))


    ax = fig.add_subplot(514)
    P_o = np.exp(rw[:,10]) + np.exp(rw[:,13]) + np.exp(rw[:,16])
    P_m = np.exp(rw[:,11]) + np.exp(rw[:,14]) + np.exp(rw[:,17])
    N = len(data)*1.
    ax.plot(rw[:,0], N/P_o * gradf(np.exp(rw[:,11])) - N*P_m / P_o**2 * gradf(np.exp(rw[:,10])), label='0')
    ax.plot(rw[:,0], N/P_o * gradf(np.exp(rw[:,14])) - N*P_m / P_o**2 * gradf(np.exp(rw[:,13])), label='1')
    ax.plot(rw[:,0], N/P_o * gradf(np.exp(rw[:,17])) - N*P_m / P_o**2 * gradf(np.exp(rw[:,16])), label='2')
    ax.plot(rw[:,0], limit*np.ones_like(rw[:,0]), 'k:')
    ax.plot(rw[:,0], gradf(rw[:,6]), 'k-', label='1+2+3')
    ax.set_ylabel(r'$dN_{m,k}/dt$')
    ax.legend(frameon=False)

    ax = fig.add_subplot(515)
    ax.plot(rw[:,0], rw[:,1], 'k-', label='soften')
    ax.set_ylabel('soften')

    #ax.plot(rw[:,0], limit / 3 / (N/P_o * gradf(np.exp(rw[:,11])) - N*P_m / P_o**2 * gradf(np.exp(rw[:,10]))), label='0')
    #ax.plot(rw[:,0], limit / 3 / (N/P_o * gradf(np.exp(rw[:,14])) - N*P_m / P_o**2 * gradf(np.exp(rw[:,13]))), label='1')
    #ax.plot(rw[:,0], limit / 3 / (N/P_o * gradf(np.exp(rw[:,17])) - N*P_m / P_o**2 * gradf(np.exp(rw[:,16]))), label='2')
    #ax.plot(rw[:,0], np.minimum(1, limit / gradf(rw[:,6])), 'k-')
    #ax.set_ylabel('dampening')


    ax.set_xlabel('iteration')
    plt.subplots_adjust(hspace=0.02, bottom=0.06, right=0.95,  top=0.99)
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


if __name__ == '__main__':

    # set up RNG
    seed = None
    from numpy.random import RandomState
    rng = RandomState(seed)
    verbose = True

    # draw N points from 3-component GMM
    N = 400
    D = 2
    gmm = iemgmm.GMM(K=3, D=2)
    gmm.amp[:] = np.array([ 0.42561594,  0.33032903,  0.24405504])
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

    K = 3

    # limit data to within the box
    cb = getHole
    ps = patches.Circle([6.5, 6.], radius=2, fc="none", ec='b', ls='dotted')

    # add isotropic errors on data
    disp = 0.8
    noisy = orig + rng.normal(0, scale=disp, size=(len(orig), D))
    sel = cb(noisy)
    data = noisy[sel]
    covar = np.tile(disp**2 * np.eye(D), (len(data), 1, 1))

    # make sure that the initial placement of the components
    # uses the same RNG for comparison
    init_cb = partial(iemgmm.initializeFromDataMinMax, rng=rng)

    # IEMGMM with imputation, incorporating errors
    logfile = "logfile.txt"
    start = datetime.datetime.now()
    rng = RandomState()
    imp = iemgmm.fit(data, covar=covar, K=K, w=0.1, cutoff=10, sel_callback=cb, init_callback=init_cb, logfile=logfile, verbose=verbose, n_missing=None)
    print "execution time %ds" % (datetime.datetime.now() - start).seconds
    plotResults(orig, data, imp, patch=ps)
    plotTraces(logfile)
