import pygmmis
import numpy as np
import logging
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
    global K
    alpha = K
    gmm.amp[:] = rng.dirichlet(alpha*np.ones(gmm.K)/K, 1)[0]
    gmm.mean[:,:] = rng.rand(gmm.K,gmm.D)
    for k in range(gmm.K):
        gmm.covar[k] = np.diag((w + rng.rand(gmm.D) / 30)**2)
    # use random rotations for each component covariance
    # from http://www.mathworks.com/matlabcentral/newsreader/view_thread/298500
    # since we don't care about parity flips we don't have to check
    # the determinant of R (and hence don't need R)
    for k in range(gmm.K):
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
    nbh = [None for k in range(gmm.K)]
    counter = 0
    for k in range(gmm.K):
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

    #if ecolor != 'None':
    #    lw = 0.25
    sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], **kwargs)
    # get rid of pesky depth shading in absence of depthshade=False option
    if depth_shading is False:
        sc.set_edgecolors = sc.set_facecolors = lambda *args:None
    plt.show()
    return ax

def slopeSel(coords, rng=np.random):
    return rng.rand(len(coords)) > coords[:,0]

def noSel(coords, rng=np.random):
    return np.ones(len(coords), dtype="bool")

def insideComponent(k, gmm, coords, covar=None, cutoff=5.):
    if gmm.amp[k]*K > 0.01:
        return gmm.logL_k(k, coords, covar=covar, chi2_only=True) < cutoff
    else:
        return np.zeros(len(coords), dtype='bool')

def GMMSel(coords, gmm, covar=None, sel_gmm=None, cutoff_nd=3., rng=np.random):
    # swiss cheese selection based on a GMM:
    # if within 1 sigma of any component: you're out!
    import multiprocessing, parmap
    n_chunks, chunksize = sel_gmm._mp_chunksize()
    inside = np.array(parmap.map(insideComponent, range(sel_gmm.K), sel_gmm, coords, covar, cutoff_nd, pm_chunksize=chunksize))
    return np.max(inside, axis=0)

def max_posterior(gmm, U, coords, covar=None):
    import multiprocessing, parmap
    pool = multiprocessing.Pool()
    n_chunks, chunksize = gmm._mp_chunksize()
    log_p = [[] for k in range(gmm.K)]
    log_S = np.zeros(len(coords))
    H = np.zeros(len(coords), dtype="bool")
    k = 0
    for log_p[k], U[k], _ in \
    parmap.starmap(pygmmis._Estep, zip(range(gmm.K), U), gmm, data, covar, None, pm_pool=pool, pm_chunksize=chunksize):
        log_S[U[k]] += np.exp(log_p[k]) # actually S, not logS
        H[U[k]] = 1
        k += 1
    log_S[H] = np.log(log_S[H])

    max_q = np.zeros(len(coords))
    max_k = np.zeros(len(coords), dtype='uint32')
    for k in range(gmm.K):
        q_k = np.exp(log_p[k] - log_S[U[k]])
        max_k[U[k]] = np.where(max_q[U[k]] < q_k, k, max_k[U[k]])
        max_q[U[k]] = np.maximum(max_q[U[k]],q_k)
    return max_k

# from http://stackoverflow.com/questions/36740887/how-can-a-python-context-manager-try-to-execute-code
def try_forever(f):
    def decorated(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except:
                pass
    return decorated

if __name__ == "__main__":
    N = 10000
    K = 50
    D = 3
    C = 50
    w = 0.001
    inner_cutoff = 1

    seed = 42#np.random.randint(1, 10000)
    from numpy.random import RandomState
    rng = RandomState(seed)
    logging.basicConfig(format='%(message)s',level=logging.INFO)

    # define selection and create Omega in cube:
    # expensive, only do once
    sel_callback = partial(slopeSel, rng=rng)
    """
    random = rng.rand(N*100, D)
    sel = sel_callback(random)
    omega_cube = binSample(random[sel], C).astype('float') / binSample(random, C)
    del random
    """
    omega_cube = np.ones((C,C,C))
    for c in range(C):
        omega_cube[c,:,:] *= 1 - (c+0.5)/C

    count_cube = np.zeros((C,C,C))
    count__cube = np.zeros((C,C,C))
    count0_cube = np.zeros((C,C,C))

    R = 10
    amp0 = np.empty(R*K)
    frac = np.empty(R*K)
    Omega = np.empty(R*K)
    assoc_frac = np.empty(R*K)
    posterior = np.empty(R*K)

    cutoff_nd = pygmmis.chi2_cutoff(D, cutoff=inner_cutoff)
    counter = 0
    for r in range(R):
        print ("start")
        # create original sample from GMM
        gmm0 = pygmmis.GMM(K=K, D=D)
        initCube(gmm0, w=w*10, rng=rng) # use larger size floor than in fit
        data0, nbh0 = drawWithNbh(gmm0, N, rng=rng)

        # apply selection
        sel0 = sel_callback(data0)

        # how often is each component used
        comp0 = np.empty(len(data0), dtype='uint32')
        for k in range(gmm0.K):
            comp0[nbh0[k]] = k
        count0 = np.bincount(comp0, minlength=gmm0.K)

        # compute effective Omega
        comp = comp0[sel0]
        count = np.bincount(comp, minlength=gmm0.K)

        frac__ = count.astype('float') / count.sum()
        Omega__ = count.astype('float') / count0

        # restrict to "safe" components
        safe = frac__ >  1./1 * 1./ K
        if safe.sum() < gmm0.K:
            print ("reset to safe components")
            gmm0.amp = gmm0.amp[safe]
            gmm0.amp /= gmm0.amp.sum()
            gmm0.mean = gmm0.mean[safe]
            gmm0.covar = gmm0.covar[safe]

            # redraw data0 and sel0
            data0, nbh0 = drawWithNbh(gmm0, N, rng=rng)
            sel0 = sel_callback(data0)

            # recompute effective Omega and frac
            # how often is each component used
            comp0 = np.empty(len(data0), dtype='uint32')
            for k in range(gmm0.K):
                comp0[nbh0[k]] = k
            count0 = np.bincount(comp0, minlength=gmm0.K)
            comp = comp0[sel0]
            count = np.bincount(comp, minlength=gmm0.K)

            frac__ = count.astype('float') / count.sum()
            Omega__ = count.astype('float') / count0

        frac[counter:counter+gmm0.K] = frac__
        Omega[counter:counter+gmm0.K] = Omega__
        amp0[counter:counter+gmm0.K] = gmm0.amp
        count0_cube += binSample(data0, C)

        # which K: K0 or K/N = const?
        K_ = gmm0.K #int(K*omega_cube.mean())

        # fit model after selection
        data = data0[sel0]

        split_n_merge = K_/3 # 0
        gmm = pygmmis.GMM(K=K_, D=3)
        logL, U = pygmmis.fit(gmm, data, init_method='minmax', w=w, cutoff=5, split_n_merge=split_n_merge, rng=rng)
        sample = gmm.draw(N, rng=rng)
        count_cube += binSample(sample, C)

        fit_forever = try_forever(pygmmis.fit)
        gmm_ = pygmmis.GMM(K=K_, D=3)
        #fit_forever(gmm_, data, sel_callback=sel_callback, init_callback=init_cb, w=w, cutoff=5, split_n_merge=split_n_merge, rng=rng)
        gmm_.amp[:] = gmm.amp[:]
        gmm_.mean[:,:] = gmm.mean[:,:]
        gmm_.covar[:,:,:] = 2*gmm.covar[:,:,:]
        logL_, U_ = fit_forever(gmm_, data, sel_callback=sel_callback, init_method='none', w=w, cutoff=5, split_n_merge=split_n_merge, rng=rng)
        sample_ = gmm_.draw(N, rng=rng)
        """
        gmm_ = gmm
        logL_, U_ = logL, U
        sample_ = sample
        """

        count__cube += binSample(sample_, C)

        # find density threshold to be associated with any fit GMM component:
        # below a threshold, the EM algorithm won't bother to put a component.
        # under selection, that threshold applies to the observed sample.
        #
        # 1) compute fraction of observed points for each component of gmm0
        for k in range(K_):
            # select data that is within cutoff of any component of sel_gmm
            sel__ = GMMSel(data0[nbh0[k]], gmm=None, sel_gmm=gmm_, cutoff_nd=cutoff_nd, rng=rng)
            assoc_frac[k + counter] = sel__.sum() * 1./ nbh0[k].size

        """
        # 2) test which components have majority of points associated with
        # any fit component
        max_k = max_posterior(gmm, U, data0)
        for k in range(K_):
            posterior[k + counter] = np.bincount(max_k[comp0 == k]).max() * 1./ (comp0 == k).sum()
        """

        counter += gmm0.K

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

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(bins, np.zeros_like(bins), ls='--', c='#888888')
    ax.plot([0,1], [-1,1], ls='--', c='#888888')
    angle = 36
    ax.text(0.30, -1+0.47, 'uncorrected $\Omega$', color='#888888', ha='center', va='center', rotation=angle)
    ax.text(0.97, -0.05, 'perfect correction', color='#888888', ha='right', va='top')
    ax.errorbar(mean_omega, (mean_rho - mean_rho0)/mean_rho0, yerr=np.sqrt(std_rho**2 + std_rho0**2)/mean_rho0, fmt='b-', marker='s', label='Standard EM')
    ax.errorbar(mean_omega, (mean_rho_ - mean_rho0)/mean_rho0, yerr=np.sqrt(std_rho_**2 + std_rho0**2)/mean_rho0, fmt='r-', marker='o', label='$\mathtt{GMMis}$')
    ax.set_ylabel(r'$(\tilde{\rho} - \rho)/\rho$')
    ax.set_xlabel('$\Omega$')
    fig.subplots_adjust(bottom=0.12, right=0.97)
    ax.set_xlim(0,1)
    ax.set_ylim(-1,1)
    leg = ax.legend(loc='upper left', frameon=False, numpoints=1)
    fig.show()

    # plot associated fraction vs observed amplitude
    import scipy.stats
    cdf_1d = scipy.stats.norm.cdf(inner_cutoff)
    confidence_1d = 1-(1-cdf_1d)*2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(frac[:counter], assoc_frac[:counter], c=Omega[:counter], s=100*amp0[:counter]/amp0[:counter].mean(), marker='o', rasterized=True, cmap='RdYlBu')
    xl = [-0.005, frac[:counter].max()*1.1]
    yl = [0,1.0]
    ax.plot(xl, [confidence_1d, confidence_1d], c='#888888', ls='--', lw=1)
    ax.text(xl[1]*0.97, 0.68*0.97, '$1\sigma$ region', color='#888888', ha='right', va='top')
    ax.plot([1./gmm0.K, 1./gmm0.K], yl, c='#888888', ls=':', lw=1)
    ax.text(1./gmm0.K + (xl[1]-xl[0])*0.03, yl[0] + 0.03, '$1/K$', color='#888888', ha='left', va='bottom', rotation=90)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_xlabel('$N^o_k / N^o$')
    ax.set_ylabel('$\eta_k$')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.0)
    cb = plt.colorbar(sc, cax=cax)
    ticks = np.linspace(0, 1, 6)
    cb.set_ticks(ticks)
    cb.set_label('$\Omega_k$')
    fig.subplots_adjust(bottom=0.13, right=0.90)
    fig.show()


    cmap = matplotlib.cm.get_cmap('RdYlBu')
    color = np.array([cmap(20),cmap(255)])[sel0.astype('int')]
    #ecolor = np.array(['r','b'])[sel0.astype('int')]
    ax = plotPoints(data0, s=4, c=color,lw=0,rasterized=True, depth_shading=False)
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)

    ax = plotPoints(sample_, s=1, alpha=0.5)
    for k in range(gmm0.K):
        ax.text(gmm_.mean[k,0]+0.03, gmm_.mean[k,1]+0.03, gmm_.mean[k,2]+0.03, "%d" % k, color='r', zorder=1000)
    plotPoints(gmm0.mean, c='g', s=400, ax=ax, alpha=0.5, zorder=100)
    plotPoints(gmm_.mean, c='r', s=400, ax=ax, alpha=0.5, zorder=100)
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)
    """
