import numpy as np
import ctypes

# for multiprocessing: use shared arrays to avoid copies for each thread
# http://stackoverflow.com/questions/5549190/
def createShared(a, dtype=ctypes.c_double):
    import multiprocessing
    shared_array_base = multiprocessing.Array(dtype, a.size)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array[:] = a.flatten()
    shared_array = shared_array.reshape(a.shape)
    return shared_array

# this is to allow multiprocessing pools to operate on class methods:
# https://gist.github.com/bnyeggen/1086393
def _pickle_method(method):
	func_name = method.im_func.__name__
	obj = method.im_self
	cls = method.im_class
	if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
		cls_name = cls.__name__.lstrip('_')
		func_name = '_' + cls_name + func_name
	return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
	for cls in cls.__mro__:
		try:
			func = cls.__dict__[func_name]
		except KeyError:
			pass
		else:
			break
	return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

def logsum(logX, axis=0):
    """Computes log of the sum along give axis from the log of the summands.

    This method tries hard to avoid over- or underflow.
    See appendix A of Bovy, Hogg, Roweis (2009).

    Args:
        logX: array of logarithmic summands
        axis: axis to sum over

    Returns:
        log of the sum, shortened by one axis

    Throws:
        ValueError if logX has length 0 along given axis

    """
    floatinfo = np.finfo(logX.dtype)
    underflow = np.log(floatinfo.tiny) - logX.min(axis=axis)
    overflow = np.log(floatinfo.max) - logX.max(axis=axis) - np.log(logX.shape[axis])
    c = np.where(underflow < overflow, underflow, overflow)
    # adjust the shape of c for addition with logX
    c_shape = [slice(None) for i in xrange(len(logX.shape))]
    c_shape[axis] = None
    return np.log(np.exp(logX + c[c_shape]).sum(axis=axis)) - c

class GMM(object):
    def __init__(self, K=1, D=1):
        self.amp = np.zeros((K))
        self.mean = np.empty((K,D))
        self.covar = np.empty((K,D,D))

    @property
    def K(self):
        return self.amp.size

    @property
    def D(self):
        return self.mean.shape[1]

    def save(self, filename, **kwargs):
        """Save GMM to file.

        Args:
            filename: name for saved file, should end on .npz as the default
                      of numpy.savez(), which is called here
            kwargs:   dictionary of additional information to be stored
                      in the file. Whatever is stored in kwargs, will be loaded
                      into ZDFileInfo.
        Returns:
            None
        """
        np.savez(filename, amp=self.amp, mean=self.mean, covar=self.covar, **kwargs)

    def draw(self, size=1, sel_callback=None, invert_callback=False, rng=np.random):
        # draw indices for components given amplitudes, need to make sure: sum=1
        ind = rng.choice(self.K, size=size, p=(self.amp/self.amp.sum()))
        samples = np.empty((size, self.D))
        counter = 0
        if size > self.K:
            bc = np.bincount(ind, minlength=size)
            components = np.arange(ind.size)[bc > 0]
            for c in components:
                mask = ind == c
                s = mask.sum()
                samples[counter:counter+s] = rng.multivariate_normal(self.mean[c], self.covar[c], size=s)
                counter += s
        else:
            for i in ind:
                samples[counter] = rng.multivariate_normal(self.mean[i], self.covar[i], size=1)
                counter += 1

        # if subsample with selection is required
        if sel_callback is not None:
            sel_ = sel_callback(samples)
            if invert_callback:
                sel_ = np.invert(sel_)
            size_in = sel_.sum()
            if size_in != size:
                ssamples = self.draw(size=size-size_in, sel_callback=sel_callback, invert_callback=invert_callback, rng=rng)
                samples = np.concatenate((samples[sel_], ssamples))
        return samples

    def __call__(self, coords, covar=None, relevant=None, as_log=False):
        if as_log:
            return self.logL(coords, covar=covar, relevant=relevant)
        else:
            return np.exp(self.logL(coords, covar=covar, relevant=relevant))

    def _mp_chunksize(self):
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        chunksize = max(1, self.K/cpu_count)
        n_chunks = min(cpu_count, self.K/chunksize)
        return n_chunks, chunksize

    def _get_chunks(self):
        n_chunks, chunksize = self._mp_chunksize()
        left = self.K - n_chunks*chunksize
        chunks = []
        n = 0
        for i in xrange(n_chunks):
            n_ = n + chunksize
            if left > i:
                n_ += 1
            chunks.append((n, n_))
            n = n_
        return chunks

    def logL(self, coords, covar=None, relevant=None):
        """Log-likelihood of data given all (i.e. the sum of) GMM components

        If covar is None, this method returns
            log(sum_k(p(x | k)))
        of the data values x. If covar is set, the method returns
            log(sum_k(p(y | k))),
        where y = x + noise and noise ~ N(0, covar).

        Args:
            coords: (D,) or (N, D) test coordinates
            covar:  (D, D) or (N, D, D) covariance matrix of data
            relevant: iterable of components relevant for data points
                      see getRelevantComponents()

        Returns:
            (1,) or (N, 1) log(L), depending on shape of data
        """
        # Instead log p (x | k) for each k (which is huge)
        # compute it in stages: first for each chunk, then sum over all chunks
        import multiprocessing
        pool = multiprocessing.Pool()
        chunks = self._get_chunks()
        results = [pool.apply_async(self._logsum_chunk, (chunk, coords, covar)) for chunk in chunks]
        log_p_y_chunk = []
        for r in results:
            log_p_y_chunk.append(r.get())
        pool.close()
        return logsum(np.array(log_p_y_chunk)) # sum over all chunks = all k

    def _logsum_chunk(self, chunk, coords, covar=None):
        # helper function to reduce the memory requirement of logL
        log_p_y_k = np.empty((chunk[1]-chunk[0], len(coords)))
        for i in xrange(chunk[1] - chunk[0]):
            k = chunk[0] + i
            log_p_y_k[i,:] = self.logL_k(k, coords, covar=covar)
        return logsum(log_p_y_k)

    def logL_k(self, k, coords, covar=None, chi2_only=False):
        # compute p(x | k)
        dx = coords - self.mean[k]
        if covar is None:
            T_k = self.covar[k]
        else:
            T_k = self.covar[k] + covar
        chi2 = np.einsum('...i,...ij,...j', dx, np.linalg.inv(T_k), dx)

        if chi2_only:
            return chi2

        # prevent tiny negative determinants to mess up
        (sign, logdet) = np.linalg.slogdet(T_k)
        log2piD2 = np.log(2*np.pi)*(0.5*self.D)
        return np.log(self.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2

    def overlappingWith(self, k, cutoff=5):
        chi2_k = self.logL_k(k, self.mean, covar=self.covar, chi2_only=True)
        return np.flatnonzero(chi2_k < cutoff*cutoff*self.D)


############################
# Begin of fit functions
############################

VERBOSITY = False
VERB_BUFFER = ""

def initFromDataMinMax(gmm, data, covar=None, s=None, k=None, rng=np.random):
    if k is None:
        k = slice(None)
    gmm.amp[k] = 1./gmm.K
    # set model to random positions with equally sized spheres within
    # volumne spanned by data
    min_pos = data.min(axis=0)
    max_pos = data.max(axis=0)
    gmm.mean[k,:] = min_pos + (max_pos-min_pos)*rng.rand(gmm.K, gmm.D)
    # if s is not set: use volume filling argument:
    # K spheres of radius s [having volume s^D * pi^D/2 / gamma(D/2+1)]
    # should completely fill the volume spanned by data.
    if s is None:
        from scipy.special import gamma
        vol_data = np.prod(max_pos-min_pos)
        s = (vol_data / gmm.K * gamma(gmm.D*0.5 + 1))**(1./gmm.D) / np.sqrt(np.pi)
        if VERBOSITY >= 2:
            print "initializing spheres with s=%.2f in data domain" % s
    gmm.covar[k,:,:] = s**2 * np.eye(data.shape[1])

def initFromDataAtRandom(gmm, data, covar=None, s=None, k=None, rng=np.random):
    if k is None:
        k = slice(None)
        k_len = gmm.K
    else:
        try:
            k_len = len(gmm.amp[k])
        except TypeError:
            k_len = 1
    gmm.amp[k] = 1./gmm.K
    # initialize components around data points with uncertainty s
    refs = rng.randint(0, len(data), size=k_len)
    if s is None:
        from scipy.special import gamma
        min_pos = data.min(axis=0)
        max_pos = data.max(axis=0)
        vol_data = np.prod(max_pos-min_pos)
        s = (vol_data / gmm.K * gamma(gmm.D*0.5 + 1))**(1./gmm.D) / np.sqrt(np.pi)
        if VERBOSITY >= 2:
            print "initializing spheres with s=%.2f near data points" % s
    gmm.mean[k,:] = data[refs] + rng.normal(0, s, size=(k_len, data.shape[1]))
    gmm.covar[k,:,:] = s**2 * np.eye(data.shape[1])

# Run a simple GMM to initialize a tricky one:
def initFromSimpleGMM(gmm, data, covar=None, s=None, k=None, rng=np.random, init_callback=initFromDataAtRandom, w=0., cutoff=None, tol=1e-3, covar_factor=1.):
    # 1) run GMM without error and selection (fit is essentially an init fct)
    fit(gmm, data, covar=None, w=w, cutoff=cutoff, sel_callback=None, init_callback=init_callback, tol=tol, rng=rng)
    # 2) adjust the covariance to allow to provide more support
    # in missing volume
    gmm.covar[:,:,:] *= covar_factor

    # if k is set: only use fit init for given k, re-init the others
    if k is not None:
        k_ = set(range(gmm.K))
        try:
            k_len = len(gmm.amp[k])
            k_ -= set(k)
        except TypeError:
            k_ -= set([k])
        init_callback(gmm, k=k_, data=data, covar=covar, rng=rng)


def fit(gmm, data, covar=None, w=0., cutoff=None, sel_callback=None, init_callback=initFromDataAtRandom, tol=1e-3, rng=np.random):

    # init components
    init_callback(gmm, data=data, covar=covar, rng=rng)

    # set up pool
    import multiprocessing
    import parmap
    pool = multiprocessing.Pool()
    n_chunks, chunksize = gmm._mp_chunksize()

    # sum_k p(x|k) -> S
    # extra precautions for cases when some points are treated as outliers
    # and not considered as belonging to any component
    log_S = np.zeros(len(data)) # S = sum_k p(x|k)
    N_ = np.zeros(len(data), dtype='bool') # N == 1 for points in the fit
    neighborhood = [None for k in xrange(gmm.K)]
    log_p = [[] for k in xrange(gmm.K)]
    T_inv = [None for k in xrange(gmm.K)]
    troubled = np.zeros(gmm.K, dtype='bool')

    # compute volumes as proxy for component change
    V = np.linalg.det(gmm.covar)

    # begin EM
    it = 0
    maxiter = max(100, gmm.K)
    if VERBOSITY:
        global VERB_BUFFER
        if sel_callback is None:
            print "ITER\tPOINTS\tLOG_L\tN_STABLE"
        else:
            print "ITER\tPOINTS\tLOG_L\tN_IN\tN_STABLE"

    while it < maxiter: # limit loop in case of slow convergence

        # compute p(i | k) for each k independently in the pool
        # need S = sum_k p(i | k) for further calculation
        # also N = {i | i in neighborhood[k]} for any k
        k = 0
        for log_p[k], neighborhood[k], T_inv[k], troubled[k] in \
        parmap.starmap(_E, zip(xrange(gmm.K), neighborhood), gmm, data, covar, cutoff, pool=pool, chunksize=chunksize):
            log_S[neighborhood[k]] += np.exp(log_p[k]) # actually S, not logS
            N_[neighborhood[k]] = 1
            k += 1

        # need log(S), but since log(0) isn't a good idea, need to restrict to N_
        log_S[N_] = np.log(log_S[N_])
        log_L_ = log_S[N_].mean()
        N = N_.sum()

        if VERBOSITY:
            print ("%d\t%d\t%.3f" % (it, N, log_L_)),

        if sel_callback is not None:
            # with imputation the observed data logL can decrease.
            # revert to previous model if that is the case
            if it > 0 and log_L_ < log_L:
                if VERBOSITY:
                    print "\nmean likelihood decreased: stopping reverting to previous model."
                gmm = gmm_
                break

        # perform M step with M-sums of data and imputations runs
        _M(gmm, neighborhood, log_p, T_inv, log_S, N, data, covar=covar, w=w, sel_callback=sel_callback, cutoff=cutoff, pool=pool, chunksize=chunksize, rng=rng)

        # convergence test:
        if it > 0 and log_L_ - log_L < tol:
            log_L = log_L_
            if VERBOSITY:
                print "\nmean likelihood converged within tolerance %r: stopping here." % tol
            break

        # update all important _ quantities for convergence test(s)
        log_L = log_L_
        if sel_callback is not None:
            gmm_ = gmm # backup if next step gets worse (note: not gmm = gmm_!)

        # check new component volumes and reset neighborhood when it grows by
        # more than 25%
        V_ = np.linalg.det(gmm.covar)
        changed = np.flatnonzero((V_- V)/V > 0.25)
        for c in changed:
            neighborhood[c] = None
            V[c] = V_[c]
        if VERBOSITY:
            print "\t%d" % (gmm.K - changed.size),
        if VERBOSITY >= 2 and changed.size:
            VERB_BUFFER += "\nresetting neighborhoods due to volume change: "
            VERB_BUFFER += ("(" + "%d," * len(changed) + ")") % tuple(changed)

        if troubled.any():
            # can init at random or reset them at center of own neighborhood
            # for another try
            tcs = np.flatnonzero(troubled)
            for c in tcs:
                gmm.mean[c] = np.mean(neighborhood[c], axis=0)
            if VERBOSITY >= 2:
                VERB_BUFFER += "\nresetting instable component center: "
                VERB_BUFFER += ("(" + "%d," * len(tcs) + ")") % tuple(tcs)
            troubled[tcs] = False

        if VERBOSITY:
            print VERB_BUFFER
            if len(VERB_BUFFER):
                VERB_BUFFER = ""

        log_S[:] = 0
        N_[:] = 0
        it += 1

    pool.close()
    return log_L, neighborhood

def _E(k, neighborhood_k, gmm, data, covar=None, cutoff=None):
    # p(x | k) for all x in the vicinity of k
    # determine all points within cutoff sigma from mean[k]
    if neighborhood_k is None:
        dx = data - gmm.mean[k]
    else:
        dx = data[neighborhood_k] - gmm.mean[k]

    if covar is None:
         T_inv_k = None
         chi2 = np.einsum('...i,...ij,...j', dx, np.linalg.inv(gmm.covar[k]), dx)
    else:
        # with data errors: need to create and return T_ik = covar_i + C_k
        # and weight each datum appropriately
        if covar.shape == (gmm.D, gmm.D): # one-for-all
            T_inv_k = np.linalg.inv(gmm.covar[k] + covar)
        else: # each datum has covariance
            T_inv_k = np.linalg.inv(gmm.covar[k] + covar[neighborhood_k].reshape(len(dx), gmm.D, gmm.D))
        chi2 = np.einsum('...i,...ij,...j', dx, T_inv_k, dx)

    # NOTE: close to convergence, we could stop applying the cutoff because
    # changes to neighborhood will be minimal
    troubled = False
    if cutoff is not None:
        indices = chi2 < cutoff*cutoff*gmm.D
        if indices.any():
            chi2 = chi2[indices]
            if covar is not None and covar.shape != (gmm.D, gmm.D):
                T_inv_k = T_inv_k[indices]
            if neighborhood_k is None:
                neighborhood_k = np.flatnonzero(indices)
            else:
                neighborhood_k = neighborhood_k[indices]
        else:
            # return values without cutoff selection
            # but remember to reset component
            troubled = True

    # prevent tiny negative determinants to mess up
    (sign, logdet) = np.linalg.slogdet(gmm.covar[k])

    log2piD2 = np.log(2*np.pi)*(0.5*gmm.D)
    return np.log(gmm.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2, neighborhood_k, T_inv_k, troubled


def _M(gmm, neighborhood, log_p, T_inv, log_S, N, data, covar=None, w=0, cutoff=None, sel_callback=None, pool=None, chunksize=1, rng=np.random):

    # save the M sums from observed data
    A = np.empty(gmm.K)
    M = np.empty((gmm.K, gmm.D))
    C = np.empty((gmm.K, gmm.D, gmm.D))

    # perform sums for M step in the pool
    import parmap
    k = 0
    for A[k], M[k,:], C[k,:,:] in \
    parmap.starmap(_computeMSums, zip(xrange(gmm.K), neighborhood, log_p, T_inv), gmm, data, log_S, pool=pool, chunksize=chunksize):
        k += 1

    if sel_callback is not None:
        over = 1
        tol = 1e-2
        size = N*over
        A2, M2, C2, N2 = _computeIMSums(gmm, size, sel_callback, neighborhood, covar=covar, cutoff=cutoff, pool=pool, chunksize=chunksize, rng=rng)
        A2 /= over
        M2 /= over
        C2 /= over
        N2 /= over

        if VERBOSITY:
            sel_outside = A2 > tol * A
            print "\t%d" % (gmm.K - sel_outside.sum()),
            if VERBOSITY >= 2 and sel_outside.any():
                global VERB_BUFFER
                VERB_BUFFER += "\ncomponent inside fractions: "
                VERB_BUFFER += ("(" + "%.2f," * gmm.K + ")") % tuple(A/(A+A2))
    else:
        A2 = M2 = C2 = N2 = 0

    # M-step for all components using data and data2
    gmm.amp[:] = (A + A2)/ (N + N2)
    # because of finite precision during the imputation: renormalize
    gmm.amp /= gmm.amp.sum()

    gmm.mean[:,:] = (M + M2)/(A + A2)[:,None]
    # minimum covariance term?
    if w > 0:
        # we assume w to be a lower bound of the isotropic dispersion,
        # C_k = w^2 I + ...
        # then eq. 38 in Bovy et al. only ~works for N = 0 because of the
        # prefactor 1 / (q_j + 1) = 1 / (A + 1) in our terminology
        # On average, q_j = N/K, so we'll adopt that to correct.
        w_eff = w**2 * ((N+N2)*1./gmm.K + 1)
        gmm.covar[:,:,:] = (C + C2 + w_eff*np.eye(gmm.D)[None,:,:]) / (A + A2 + 1)[:,None,None]
    else:
        gmm.covar[:,:,:] = (C + C2) / (A + A2)[:,None,None]


def _computeMSums(k, neighborhood_k, log_p_k, T_inv_k, gmm, data, log_S):
    if log_p_k.size:
        # form log_q_ik by dividing with S = sum_k p_ik
        # NOTE:  this modifies log_p_k in place!
        # NOTE2: reshape needed when neighborhood_k is None because of its
        # implicit meaning as np.newaxis (which would create a 2D array)
        log_p_k -= log_S[neighborhood_k].reshape(log_p_k.size)

        # amplitude: A_k = sum_i q_ik
        A_k = np.exp(logsum(log_p_k))

        # in fact: q_ik, but we treat sample index i silently everywhere
        qk = np.exp(log_p_k)

        # data with errors?
        d = data[neighborhood_k].reshape((log_p_k.size, gmm.D))
        if T_inv_k is None:
            # mean: M_k = sum_i x_i q_ik
            M_k = (d * qk[:,None]).sum(axis=0)

            # covariance: C_k = sum_i (x_i - mu_k)^T(x_i - mu_k) q_ik
            d_m = d - gmm.mean[k]
            # funny way of saying: for each point i, do the outer product
            # of d_m with its transpose, multiply with pi[i], and sum over i
            C_k = (qk[:, None, None] * d_m[:, :, None] * d_m[:, None, :]).sum(axis=0)
        else:
            # need temporary variables:
            # b_ik = mu_k + C_k T_ik^-1 (x_i - mu_k)
            # B_ik = C_k - C_k T_ik^-1 C_k
            # to replace pure data-driven means and covariances
            d_m = d - gmm.mean[k]
            b_k = gmm.mean[k] + np.einsum('ij,...jk,...k', gmm.covar[k], T_inv_k, d_m)
            M_k = (b_k * qk[:,None]).sum(axis=0)

            b_k -= gmm.mean[k]
            B_k = gmm.covar[k] - np.einsum('ij,...jk,...kl', gmm.covar[k], T_inv_k, gmm.covar[k])
            C_k = (qk[:, None, None] * (b_k[:, :, None] * b_k[:, None, :] + B_k)).sum(axis=0)
        return A_k, M_k, C_k
    else:
        return 0,0,0

def _computeIMSums(gmm, size, sel_callback, neighborhood, covar=None, cutoff=None, pool=None, chunksize=1, rng=np.random):
    # create fake data with same mechanism as the original data,
    # but invert selection to get the missing part
    data2, covar2, neighborhood2 = _I(gmm, size, sel_callback, cutoff=cutoff, covar=covar, neighborhood=neighborhood, rng=rng)

    A2 = np.zeros(gmm.K)
    M2 = np.zeros((gmm.K, gmm.D))
    C2 = np.zeros((gmm.K, gmm.D, gmm.D))
    N2 = len(data2)

    if N2:
        # similar setup as above, but since imputated points
        # are drawn from the model, we can avoid the caution of
        # dealing with outliers: all points will be considered
        log_S2 = np.zeros(len(data2))
        N2_ = np.zeros(len(data2), dtype='bool')
        log_p2 = [[] for k in xrange(gmm.K)]
        T2_inv = [None for k in xrange(gmm.K)]

        # run E-step: only needed to account for components that
        # overlap outside. Otherwise log_q_ik = 0 for all i,k,
        # i.e. all sums have flat weights
        import parmap
        k = 0
        for log_p2[k], neighborhood2[k], T2_inv[k], _ in \
        parmap.starmap(_E, zip(xrange(gmm.K), neighborhood2), gmm, data2, covar2, None, pool=pool, chunksize=chunksize):
            log_S2[neighborhood2[k]] += np.exp(log_p2[k])
            N2_[neighborhood2[k]] = 1
            k += 1

        log_S2[N2_] = np.log(log_S2[N2_])
        N2 = N2_.sum()

        k = 0
        for A2[k], M2[k,:], C2[k,:,:] in \
        parmap.starmap(_computeMSums, zip(xrange(gmm.K), neighborhood2, log_p2, T2_inv), gmm, data2, log_S2, pool=pool, chunksize=chunksize):
            k += 1

    return A2, M2, C2, N2

def _I(gmm, size, sel_callback, cutoff=3, covar=None, neighborhood=None, covar_reduce_fct=np.mean, rng=np.random):

    data2 = np.empty((size, gmm.D))
    if covar is None:
        covar2 = None
    else:
        if covar.shape == (gmm.D, gmm.D): # one-for-all
            covar2 = covar
        else:
            covar2 = np.empty((size, gmm.D, gmm.D))

    # draw indices for components given amplitudes
    ind = rng.choice(gmm.K, size=size, p=gmm.amp)
    N2 = np.bincount(ind, minlength=gmm.K)

    # keep track which point comes from which component
    component2 = np.empty(size, dtype='uint32')

    # for each component: draw as many points as in ind from a normal
    lower = 0
    for k in np.flatnonzero(N2):
        upper = lower + N2[k]
        if covar is None:
            data2[lower:upper, :] = rng.multivariate_normal(gmm.mean[k], gmm.covar[k], size=N2[k])
        else:
            if covar.shape == (gmm.D, gmm.D): # one-for-all
                covar_k = covar
            else:
                covar_k = covar_reduce_fct(covar[neighborhood[k]], axis=0)
                covar2[lower:upper, :,:] = covar_k
            data2[lower:upper, :] = rng.multivariate_normal(gmm.mean[k], gmm.covar[k] + covar_k, size=N2[k])
        component2[lower:upper] = k
        lower = upper

    # TODO: may want to decide whether to add noise before selector of after
    sel2 = ~sel_callback(data2, gmm=gmm)
    data2 = data2[sel2]
    component2 = component2[sel2]
    if covar is not None and covar.shape != (gmm.D, gmm.D):
        covar2 = covar2[sel2]

    # determine neighborhood between components
    neighborhood2 = [None for k in xrange(gmm.K)]
    for k in xrange(gmm.K):
        overlap_k = gmm.overlappingWith(k, cutoff=cutoff)
        mask2 = np.zeros(len(data2), dtype='bool')
        for j in overlap_k:
            mask2 |= (component2 == j)
        neighborhood2[k] = np.flatnonzero(mask2)

    return data2, covar2, neighborhood2
