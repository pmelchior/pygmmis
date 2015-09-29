#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw(amp, mean, covar, size=1, box_limits=None):
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

    # if subsample within box is needed:
    # make sure that the sampling is from within the box
    if box_limits is not None:
        while True:
            sel_ = (samples[:,0] > box_limits[0,0]) & (samples[:,0] < box_limits[1,0]) & (samples[:,1] > box_limits[0,1]) & (samples[:,1] < box_limits[1,1]) & (samples[:,2] > box_limits[0,2]) & (samples[:,2] < box_limits[1,2])
            size_in = sel_.sum()
            if size_in == size:
                break
            ssamples = draw(amp, mean, covar, size=size-size_in)
            samples = np.concatenate((samples[sel_], ssamples))
        del sel_
    return samples

def getBoxSelection(box_limits, coords, compressed=False):
    sel = (coords[:,0] > box_limits[0,0]) & (coords[:,0] < box_limits[1,0]) & (coords[:,1] > box_limits[0,1]) & (coords[:,1] < box_limits[1,1])
    if compressed:
        return np.nonzero(sel)[0]
    else:
        return sel

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
    floatinfo = np.finfo('d')
    underflow = np.log(floatinfo.tiny) - ll.min(axis=0)
    overflow = np.log(floatinfo.max) - ll.max(axis=0) - np.log(K)
    c = np.where(underflow < overflow, underflow, overflow)
    return np.log(np.exp(ll + c).sum(axis=0)) - c

def E(data, qij, amp, mean, covar):
    for j in xrange(K):
        dx = data - mean[j]
        chi2 = np.einsum('...j,j...', dx, np.dot(np.linalg.inv(covar[j]), dx.T))
        qij[:,j] = np.log(amp[j]) - np.log((2*np.pi)**D * np.linalg.det(covar[j]))/2 - chi2/2
    for j in xrange(K):
        qij[:,j] -= logsumLogL(qij.T)

def M(data, qij, amp, mean, covar):
    qj = np.exp(logsumLogL(qij))
    for j in xrange(K):
        amp[j] = np.exp(qij[:,j]).sum()/N
        covar[j] = 0
        for i in xrange(N):
            covar[j] += np.exp(qij[i,j]) * np.outer(data[i]-mean[j], (data[i]-mean[j]).T)
        covar[j] /= qj[j]
        for d in xrange(D):
            mean[j,d] = (data[:,d] * np.exp(qij[:,j])).sum()/qj[j]

def initialize(amp, mean, covar):
    # initialize GMM with equal weigths, random positions, fixed covariances
    amp[:] = 1./K
    mean[:,:] = np.random.random(size=(K, D))
    target_size = 0.1
    covar[:,:,:] = np.tile(target_size**2 * np.eye(D), (K,1,1))
            
def run_EM(data, amp, mean, covar):
    initialize(amp, mean, covar)
    qij = np.empty((N, K))

    iter = 0
    while iter < 10: 
        try:
            E(data, qij, amp, mean, covar)
            M(data, qij, amp, mean, covar)
        except np.linalg.linalg.LinAlgError:
            iter = 0
            initialize(amp, mean, covar)
        iter += 1

def plotResults(data, sel, amp, mean, covar):
    fig = plt.figure()
    ax = fig.add_subplot(111)

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
    p = np.arcsinh(np.exp(logsumLogL(qij.T).reshape((B,B))) / 1e-4)
    #ax.imshow(p, extent=(-0.5,1.5,-0.5,1.5), cmap=plt.cm.Greys)
    ax.contourf(p, 10, extent=(-0.5,1.5,-0.5,1.5), cmap=plt.cm.Greys)

    # plot boundary
    rect = patches.Rectangle([0,0], 1, 1, fc="none", ec='b', ls='dotted')
    ax.add_artist(rect)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    

    

    

# draw N points from 3-component GMM
D = 2
N = 400
orig = draw(np.array([ 0.36060026,  0.27986906,  0.15506774]),
            np.array([[ 0.24816886,  0.21300697],
                      [ 0.67306351,  0.6109532 ],
                      [ 0.01887670,  0.902077]]),
            np.array([[[ 0.08530014, -0.00314178],
                       [-0.00314178,  0.00541106]],
                      [[ 0.05453402, -0.0195736],
                       [-0.0195736,  0.01475791]],
                      [[ 0.00258605,  0.00409287],
                       [ 0.00409287,  0.01065186]]]), size=N)
# limit data to within the box
box_limits = np.array([[0,0],[1,1]])
sel = getBoxSelection(box_limits, orig)
data = orig[sel]
N = sel.sum()


K = 5
amp = None
mean = None
covar = None
for R in range(100):
    print R
    amp_ = np.empty(K)
    mean_ = np.empty((K, D))
    covar_ = np.empty((K, D, D))
    run_EM(data, amp_, mean_, covar_)
    if amp is None:
        amp = amp_
        mean = mean_
        covar = covar_
    else:
        amp = np.concatenate((amp, amp_))
        mean = np.concatenate((mean, mean_), axis=0)
        covar = np.concatenate((covar, covar_), axis=0)

amp /= amp.sum()
plotResults(orig, sel, amp, mean, covar)
plt.show()






