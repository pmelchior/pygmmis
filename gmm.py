#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

D = 2

def draw(amp, mean, covar, size=1, sel_callback=None, invert_callback=False):
    amp /= amp.sum()
    # draw indices for components given amplitudes
    K = amp.size
    ind = np.random.choice(K, size=size, p=amp)
    samples = np.empty((size, mean.shape[1]))
    counter = 0
    if size > K:
        bc = np.bincount(ind)
        components = np.arange(ind.size)[bc > 0]
        for c in components:
            mask = ind == c
            s = mask.sum()
            samples[counter:counter+s] = np.random.multivariate_normal(mean[c], covar[c], size=s)
            counter += s
    else:
        for i in ind:
            samples[counter] = np.random.multivariate_normal(mean[i], covar[i], size=1)
            counter += 1

    # if subsample with selection is required
    if sel_callback is not None:
        sel_ = sel_callback(samples)
        if invert_callback:
            sel_ = np.invert(sel_)
        size_in = sel_.sum()
        if size_in != size:
            ssamples = draw(amp, mean, covar, size=size-size_in, sel_callback=sel_callback, invert_callback=invert_callback)
            samples = np.concatenate((samples[sel_], ssamples))
    return samples

"""
        while True:
            sel_ = sel_callback(samples)
            if invert_callback:
                sel_ = np.invert(sel_)
            size_in = sel_.sum()
            if size_in == size:
                break
            ssamples = draw(amp, mean, covar, size=size-size_in)
            samples = np.concatenate((samples[sel_], ssamples))
        del sel_
    return samples
    """

def logsumLogL(ll):
    """Computes log of sum of likelihoods for GMM components.
    
    This method tries hard to avoid over- or underflow that may arise
    when computing exp(log(p(x | k)).
    
    See appendix A of Bovy, Hogg, Roweis (2009).
    
    Args:
    ll: (K, N) log-likelihoods from K calls to logL_K() with N coordinates
    
    Returns:
    (N, 1) of log of total likelihood
    
    """
    # typo in eq. 58: log(N) -> log(K)
    K = ll.shape[0]
    floatinfo = np.finfo('d')
    underflow = np.log(floatinfo.tiny) - ll.min(axis=0)
    overflow = np.log(floatinfo.max) - ll.max(axis=0) - np.log(K)
    c = np.where(underflow < overflow, underflow, overflow)
    return np.log(np.exp(ll + c).sum(axis=0)) - c

def E(data, amp, mean, covar):
    K = amp.size
    qij = np.empty((data.shape[0], K))
    for j in xrange(K):
        dx = data - mean[j]
        chi2 = np.einsum('...j,j...', dx, np.dot(np.linalg.inv(covar[j]), dx.T))
        qij[:,j] = np.log(amp[j]) - np.log((2*np.pi)**D * np.linalg.det(covar[j]))/2 - chi2/2
    for j in xrange(K):
        qij[:,j] -= logsumLogL(qij.T)
    return qij

def M(data, qij, amp, mean, covar, impute=0):
    K = amp.size
    N = data.shape[0] - impute
    qj = np.exp(logsumLogL(qij))
    if impute:
        qj_in = np.exp(logsumLogL(qij[:-impute]))
        qj_out = np.exp(logsumLogL(qij[-impute:]))
        covar_ = np.empty((D,D))
        
    for j in xrange(K):
        Q_i = np.exp(qij[:,j])
        amp[j] = qj[j]/(N+impute)
        
        # do covar first since we can do this without a copy of mean here
        if impute:
            covar_[:,:] = covar[j]
        covar[j] = 0
        for i in xrange(N):
            covar[j] += Q_i[i] * np.outer(data[i]-mean[j], (data[i]-mean[j]).T)
        if impute == 0:
            covar[j] /= qj[j]
        else:
            covar[j] /= qj_in[j]
            covar[j] += qj_out[j] / qj[j] * covar_
            
        # now update means
        for d in xrange(D):
            mean[j,d] = (data[:,d] * Q_i).sum()/qj[j]

def I(amp, mean, covar, impute=0, sel_callback=None):
    return draw(amp, mean, covar, size=impute, sel_callback=sel_callback, invert_callback=True)
    
def initialize(amp, mean, covar):
    K = amp.size
    # initialize GMM with equal weigths, random positions, fixed covariances
    amp[:] = 1./K
    mean[:,:] = np.random.random(size=(K, D))
    target_size = 0.1
    covar[:,:,:] = np.tile(target_size**2 * np.eye(D), (K,1,1))
            
def run_EM(data, amp, mean, covar, impute=0, sel_callback=None):
    initialize(amp, mean, covar)

    iter = 0
    while iter < 50: 
        try:
            if impute == 0 or iter < 25 or iter % 2 == 0:
                qij = E(data, amp, mean, covar)
                M(data, qij, amp, mean, covar)
            else:
                data_out = I(amp, mean, covar, impute=impute, sel_callback=sel_callback)
                data_ = np.concatenate((data, data_out), axis=0)
                
                qij = E(data_, amp, mean, covar)
                M(data_, qij, amp, mean, covar, impute=impute)
        except np.linalg.linalg.LinAlgError:
            iter = 0
            initialize(amp, mean, covar)
        iter += 1

def plotResults(data, sel, amp, mean, covar, patch=None):
    K = amp.size
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal')

    # plot inner and outer points
    ax.plot(data[:,0][sel], data[:,1][sel], 'bo', mec='None')
    ax.plot(data[:,0][sel==False], data[:,1][sel==False], 'o', mfc='None', mec='b')

    # prediction
    B = 100
    x,y = np.meshgrid(np.linspace(-0.5,1.5,B), np.linspace(-0.5,1.5,B))
    coords = np.dstack((x.flatten(), y.flatten()))[0]

    qij = np.empty((B*B,K))
    for j in xrange(K):
        dx = coords - mean[j]
        chi2 = np.einsum('...j,j...', dx, np.dot(np.linalg.inv(covar[j]), dx.T))
        qij[:,j] = np.log(amp[j]) - np.log((2*np.pi)**D * np.linalg.det(covar[j]))/2 - chi2/2
    p = np.arcsinh(np.exp(logsumLogL(qij.T).reshape((B,B))) / 1e-2)
    cs = ax.contourf(p, 10, extent=(-0.5,1.5,-0.5,1.5), cmap=plt.cm.Greys)
    for c in cs.collections:
        c.set_edgecolor(c.get_facecolor())

    # plot boundary
    if patch is not None:
        import copy
        patch_ = copy.copy(patch)
        if hasattr(patch_, '__iter__'):
            for p in patch_:
                ax.add_artist(p)
        else:
            ax.add_artist(patch_)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def run_test(data, K=3, R=100, sel=None, sel_callback=None):
    # now with imputation
    amp = None
    mean = None
    covar = None
    for r in range(R):
        print r
        amp_ = np.empty(K)
        mean_ = np.empty((K, D))
        covar_ = np.empty((K, D, D))
        if sel is None:
            run_EM(data, amp_, mean_, covar_)
        else:
            run_EM(data, amp_, mean_, covar_, impute=(sel==False).sum(), sel_callback=sel_callback)
        if amp is None:
            amp = amp_
            mean = mean_
            covar = covar_
        else:
            amp = np.concatenate((amp, amp_))
            mean = np.concatenate((mean, mean_), axis=0)
            covar = np.concatenate((covar, covar_), axis=0)
    amp /= amp.sum()
    return amp, mean, covar

def getBox(coords):
    box_limits = np.array([[0,0],[1,1]])
    return (coords[:,0] > box_limits[0,0]) & (coords[:,0] < box_limits[1,0]) & (coords[:,1] > box_limits[0,1]) & (coords[:,1] < box_limits[1,1])

def getHole(coords):
    x,y,r = 0.65, 0.6, 0.2
    return ((coords[:,0] - x)**2 + (coords[:,1] - y)**2 > r**2)

def getBoxWithHole(coords):
    return getBox(coords)*getHole(coords)

def getHalfDensity(coords):
    mask = np.ones(coords.shape[0], dtype='bool')
    mask[np.random.random(coords.shape[0]) < 0.5] = 0
    return mask

def getTaperedDensity(coords):
    mask = np.ones(coords.shape[0], dtype='bool')
    mask[np.random.random(coords.shape[0]) < coords[:,0]] = 0
    return mask

# draw N points from 3-component GMM
N = 500
orig = draw(np.array([ 0.36060026,  0.27986906,  0.206774]),
            np.array([[ 0.08016886,  0.21300697],
                      [ 0.70306351,  0.6709532 ],
                      [ 0.01087670,  0.852077]]),
            np.array([[[ 0.08530014, -0.00314178],
                       [-0.00314178,  0.00541106]],
                      [[ 0.03053402, 0.0125736],
                       [0.0125736,  0.01075791]],
                      [[ 0.00258605,  0.00409287],
                       [ 0.00409287,  0.01065186]]]), size=N)

# limit data to within the box

"""
cb = getHole
ps = patches.Circle([0.65, 0.6], radius=0.2, fc="none", ec='b', ls='dotted', lw=1)
"""
"""
cb = getBox
ps = patches.Rectangle([0,0], 1, 1, fc="none", ec='b', ls='dotted')
"""
"""
cb = getBoxWithHole
ps = [patches.Rectangle([0,0], 1, 1, fc="none", ec='b', ls='dotted'),
      patches.Circle([0.65, 0.6], radius=0.2, fc="none", ec='b', ls='dotted')]

"""
cb = getTaperedDensity
ps = None

sel = cb(orig)
data = orig[sel]

# without imputation
K = 3
R = 10

amp, mean, covar = run_test(data, K=K, R=R)
plotResults(orig, sel, amp, mean, covar, patch=ps)

# with imputation
amp, mean, covar = run_test(data, K=K, R=R, sel=sel, sel_callback=cb)
plotResults(orig, sel, amp, mean, covar, patch=ps)








