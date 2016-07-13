import pygmmi
import numpy as np
from functools import partial

L = 1

def binSample(coords, C):
    dl = L*1./C
    N = len(coords)
    from sklearn.neighbors import KDTree
    # chebyshev metric: results in cube selection
    tree = KDTree(coords, leaf_size=N/100, metric="chebyshev")
    # sample position: center of cubes of length K
    skewer = np.arange(C)
    grid = np.meshgrid(skewer, skewer, skewer, indexing="ij")
    grid = np.dstack((grid[0].flatten(), grid[1].flatten(), grid[2].flatten()))[0]
    samples = dl*(grid +0.5)

    # get counts in boxes
    c = tree.query_radius(samples, r=0.5*dl, count_only=True)
    #counts = np.zeros(K**3)
    #counts[mask] = c
    #return counts.reshape(K,K,K)
    return c.reshape(C,C,C)

def initCube(gmm, w=0, rng=np.random):
    #gmm.amp[:] = rng.rand(gmm.K)
    #gmm.amp /= gmm.amp.sum()
    alpha = 100
    gmm.amp[:] = rng.dirichlet(alpha*np.ones(gmm.K)/K, 1)[0]
    gmm.mean[:,:] = rng.rand(gmm.K,gmm.D)
    for k in xrange(gmm.K):
        gmm.covar[k] = np.diag((w + rng.rand(gmm.D) / 30)**2)
    # use random rotations for each component covariance
    # from http://www.mathworks.com/matlabcentral/newsreader/view_thread/298500
    # since we don't care about parity flips we don't have to check
    # the determinant of R (and hence don't need R)
    for k in xrange(gmm.K):
        Q,_ = np.linalg.qr(rng.normal(size=(gmm.D, gmm.D)), mode='complete')
        gmm.covar[k] = np.dot(Q, np.dot(gmm.covar[k], Q.T))

def initToFillCube(gmm, omega=0.5, rng=np.random):
    gmm.amp[k] = 1./gmm.K
    # set model to random positions with equally sized spheres within
    # volumne spanned by data
    min_pos = np.zeros(3)
    max_pos = np.ones(3)
    gmm.mean[k,:] = min_pos + (max_pos-min_pos)*rng.rand(gmm.K, gmm.D)
    # K spheres of radius s [having volume s^D * pi^D/2 / gamma(D/2+1)]
    # should fill fraction omega of cube
    from scipy.special import gamma
    vol_data = np.prod(max_pos-min_pos)
    s = (omega * vol_data / gmm.K * gamma(gmm.D*0.5 + 1))**(1./gmm.D) / np.sqrt(np.pi)
    gmm.covar[k,:,:] = s**2 * np.eye(data.shape[1])

def drawWithNbh(gmm, size=1, rng=np.random):
    # draw indices for components given amplitudes, need to make sure: sum=1
    ind = rng.choice(gmm.K, size=size, p=(gmm.amp/gmm.amp.sum()))
    samples = np.empty((size, gmm.D))
    N_k = np.bincount(ind, minlength=gmm.K)
    nbh = [None for k in xrange(gmm.K)]
    counter = 0
    for k in xrange(gmm.K):
        s = N_k[k]
        samples[counter:counter+s] = rng.multivariate_normal(gmm.mean[k], gmm.covar[k], size=s)
        nbh[k] = np.arange(counter, counter+s)
        counter += s
    return samples, nbh

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def createFigure():
    fig = plt.figure()
    ax = plt.axes([0,0,1,1], projection='3d')#, aspect='equal')
    return fig, ax

def plotPoints(coords, ax=None, depth_shading=True, **kwargs):
    if ax is None:
        fig, ax = createFigure()

    lw = 0
    #if ecolor != 'None':
    #    lw = 0.25
    sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], linewidths=lw, **kwargs)
    # get rid of pesky depth shading in absence of depthshade=False option
    if depth_shading is False:
        sc.set_edgecolors = sc.set_facecolors = lambda *args:None
    plt.show()
    return ax

def slopeSel(coords, gmm=None, rng=np.random):
    return rng.rand(len(coords)) > coords[:,0]

def insideComponent(k, gmm, coords, covar=None, cutoff=5.):
    return gmm.logL_k(k, coords, covar=covar, chi2_only=True) < cutoff

def GMMSel(coords, gmm=None, covar=None, sel_gmm=None, cutoff=3., rng=np.random):
    # compute effective cutoff for chi2 cutoff
    import scipy.stats
    cdf_1d = scipy.stats.norm.cdf(cutoff)
    confidence_1d = 1-(1-cdf_1d)*2
    cutoff_nd = scipy.stats.chi2.ppf(confidence_1d, sel_gmm.D)

    # swiss cheese selection based on a GMM:
    # if within 1 sigma of any component: you're out!
    import multiprocessing
    import parmap
    n_chunks, chunksize = sel_gmm._mp_chunksize()
    inside = np.array(parmap.map(insideComponent, xrange(sel_gmm.K), sel_gmm, coords, covar, cutoff_nd, chunksize=chunksize))
    return np.max(inside, axis=0)

# from http://stackoverflow.com/questions/36740887/how-can-a-python-context-manager-try-to-execute-code
def try_forever(f):
    def decorated(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except:
                pass
    return decorated

def dummy_init(gmm, data, covar=None, rng=np.random):
    pass

if __name__ == "__main__":
    N = 1000
    K = 10
    D = 3
    C = 50
    w = 0.001
    rng = np.random

    # define selection and create Omega in cube:
    # expensive, only do once
    sel_callback = slopeSel
    """
    random = rng.rand(N*100, D)
    sel = sel_callback(random)
    omega_cube = binSample(random[sel], C).astype('float') / binSample(random, C)
    del random
    """
    omega_cube = np.ones((C,C,C))
    for c in xrange(C):
        omega_cube[c,:,:] *= 1 - (c+0.5)/C

    count_cube = np.zeros((C,C,C))
    count__cube = np.zeros((C,C,C))
    count0_cube = np.zeros((C,C,C))

    R = 10
    amp0 = np.empty(R*K)
    frac = np.empty(R*K)
    Omega = np.empty(R*K)
    assoc_frac = np.empty(R*K)

    for r in xrange(R):
        # create original sample from GMM
        gmm0 = pygmmi.GMM(K=K, D=D)
        initCube(gmm0, w=w, rng=rng)
        amp0[r*K:(r+1)*K] = gmm0.amp
        data0, nbh0 = drawWithNbh(gmm0, N, rng=rng)
        count0_cube += binSample(data0, C)

        # fit model after selection
        sel0 = sel_callback(data0)
        data = pygmmi.createShared(data0[sel0])

        # which K: K0 or K/N = const?
        K_ = K#int(K*omega_cube.mean())
        gmm = pygmmi.GMM(K=K_, D=3)
        pygmmi.VERBOSITY = 1
        pygmmi.fit(gmm, data, init_callback=pygmmi.initFromDataAtRandom, w=w, cutoff=5, rng=rng)
        sample = gmm.draw(N, rng=rng)
        count_cube += binSample(sample, C)

        fit_forever = try_forever(pygmmi.fit)
        gmm_ = pygmmi.GMM(K=K_, D=3)
        #fit_forever(gmm_, data, sel_callback=sel_callback, init_callback=pygmmi.initFromDataAtRandom, w=w, cutoff=5, rng=rng)
        gmm_.amp[:] = gmm.amp[:]
        gmm_.mean[:,:] = gmm.mean[:,:]
        gmm_.covar[:,:,:] = 4*gmm.covar[:,:,:]
        fit_forever(gmm_, data, sel_callback=sel_callback, init_callback=dummy_init, w=w, cutoff=5, rng=rng)
        sample = gmm_.draw(N, rng=rng)
        count__cube += binSample(sample, C)

        # find density threshold to be associated with any fit GMM component:
        # below a threshold, the EM algorithm won't bother to put a component.
        # under selection, that threshold applies to the observed sample.
        #
        # 1) compute fraction of observed points for each component of gmm0
        comp0 = np.empty(len(data0), dtype='uint32')
        for k in xrange(gmm0.K):
            comp0[nbh0[k]] = k
        comp = comp0[sel0]
        count0 = np.bincount(comp0, minlength=gmm0.K)
        count = np.bincount(comp, minlength=gmm0.K)
        frac[r*K:(r+1)*K] = count.astype('float') / count.sum()
        Omega[r*K:(r+1)*K] = count.astype('float') / count0

        # 2) test which components have majority of points associated with
        # any fit component
        cutoff = 1
        import scipy.stats
        cdf_1d = scipy.stats.norm.cdf(cutoff)
        confidence_1d = 1-(1-cdf_1d)*2
        for k in xrange(K):
            # select data that is within cutoff of any component of sel_gmm
            sel__ = GMMSel(data0[nbh0[k]], gmm=None, sel_gmm=gmm_, cutoff=cutoff, rng=rng)
            assoc_frac[k + r*K] = sel__.sum() * 1./ nbh0[k].size


    # plot average cell density as function of cell omega:
    # biased estimate will avoid low-omega region and (over)compensate in
    # high-omega regions
    B = 10
    bins = np.linspace(0,1,B+1)

    mean_rho0 = np.empty(B)
    mean_rho = np.empty(B)
    mean_rho_ = np.empty(B)
    mean_omega = np.empty(B)
    std_rho0 = np.empty(B)
    std_rho = np.empty(B)
    std_rho_ = np.empty(B)
    std_omega = np.empty(B)
    for i in range(B):
        mask = (omega_cube > bins[i]) & (omega_cube <= bins[i+1])
        sqrtN = np.sqrt(mask.sum())
        mean_omega[i] = omega_cube[mask].mean()
        std_omega[i] = omega_cube[mask].std()
        mean_rho0[i] = count0_cube[mask].mean()
        std_rho0[i] = count0_cube[mask].std() / sqrtN
        mean_rho[i] = count_cube[mask].mean()
        std_rho[i] = count_cube[mask].std() / sqrtN
        mean_rho_[i] = count__cube[mask].mean()
        std_rho_[i] = count__cube[mask].std() / sqrtN

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bins, np.zeros_like(bins), 'k:')
    ax.plot([0,1], [-1,1], 'k--')
    ax.errorbar(mean_omega, (mean_rho - mean_rho0)/mean_rho0, yerr=np.sqrt(std_rho**2 + std_rho0**2)/mean_rho0, fmt='b-')
    ax.errorbar(mean_omega, (mean_rho_ - mean_rho0)/mean_rho0, yerr=np.sqrt(std_rho_**2 + std_rho0**2)/mean_rho0, fmt='r-')
    ax.set_xlabel('$\Omega$')
    fig.show()


    # plot associated fraction vs observed amplitude
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(frac, assoc_frac, c=Omega, s=100*amp0/amp0.mean(), marker='o')
    xl = ax.get_xlim()
    yl = [0,1.05]
    ax.plot(yl, [confidence_1d, confidence_1d], c='#888888', ls='--')
    ax.plot([1./gmm0.K, 1./gmm0.K], yl, 'k:')
    ax.set_xlim(xmin=-0.005, xmax=xl[1])
    ax.set_ylim(yl)
    ax.set_xlabel('$N^o_k / N^o$')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('$\Omega$')
    fig.show()
