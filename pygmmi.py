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


# Blantant copy from Erin Sheldon's esutil
# https://github.com/esheldon/esutil/blob/master/esutil/numpy_util.py
def match1d(arr1input, arr2input, presorted=False):
    """
    NAME:
        match
    CALLING SEQUENCE:
        ind1,ind2 = match(arr1, arr2, presorted=False)
    PURPOSE:
        Match two numpy arrays.  Return the indices of the matches or empty
        arrays if no matches are found.  This means arr1[ind1] == arr2[ind2] is
        true for all corresponding pairs.  arr1 must contain only unique
        inputs, but arr2 may be non-unique.
        If you know arr1 is sorted, set presorted=True and it will run
        even faster
    METHOD:
        uses searchsorted with some sugar.  Much faster than old version
        based on IDL code.
    REVISION HISTORY:
        Created 2015, Eli Rykoff, SLAC.
    """

    # make sure 1D
    arr1 = np.array(arr1input, ndmin=1, copy=False)
    arr2 = np.array(arr2input, ndmin=1, copy=False)

    # check for integer data...
    if (not issubclass(arr1.dtype.type,np.integer) or
        not issubclass(arr2.dtype.type,np.integer)) :
        mess="Error: only works with integer types, got %s %s"
        mess = mess % (arr1.dtype.type,arr2.dtype.type)
        raise ValueError(mess)

    if (arr1.size == 0) or (arr2.size == 0) :
        mess="Error: arr1 and arr2 must each be non-zero length"
        raise ValueError(mess)

    # make sure that arr1 has unique values...
    test=np.unique(arr1)
    if test.size != arr1.size:
        raise ValueError("Error: the arr1input must be unique")

    # sort arr1 if not presorted
    if not presorted:
        st1 = np.argsort(arr1)
    else:
        st1 = None

    # search the sorted array
    sub1=np.searchsorted(arr1,arr2,sorter=st1)

    # check for out-of-bounds at the high end if necessary
    if (arr2.max() > arr1.max()) :
        bad,=np.where(sub1 == arr1.size)
        sub1[bad] = arr1.size-1

    if not presorted:
        sub2,=np.where(arr1[st1[sub1]] == arr2)
        sub1=st1[sub1[sub2]]
    else:
        sub2,=np.where(arr1[sub1] == arr2)
        sub1=sub1[sub2]

    return sub1,sub2


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
            N_k = np.bincount(ind, minlength=self.K)
            for k in xrange(self.K):
                s = N_k[k]
                samples[counter:counter+s] = rng.multivariate_normal(self.mean[k], self.covar[k], size=s)
                counter += s
        else:
            for k in ind:
                samples[counter] = rng.multivariate_normal(self.mean[k], self.covar[k], size=1)
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
    # save backup
    import copy
    gmm_ = copy.deepcopy(gmm)

    # set up pool
    import multiprocessing
    pool = multiprocessing.Pool()
    n_chunks, chunksize = gmm._mp_chunksize()

    # sum_k p(x|k) -> S
    # extra precautions for cases when some points are treated as outliers
    # and not considered as belonging to any component
    log_S = np.zeros(len(data)) # S = sum_k p(x|k)
    H = np.zeros(len(data), dtype='bool') # H == 1 for points in the fit
    log_p = [[] for k in xrange(gmm.K)]
    T_inv = [None for k in xrange(gmm.K)]
    nbh = [None for k in xrange(gmm.K)]

    # compute effective cutoff for chi2 in D dimensions
    if cutoff is not None:
        # note: subsequently the cutoff parameter, e.g. in _E(), refers to this:
        # chi2 < cutoff,
        # while in fit() it means e.g. "cut at 3 sigma".
        # These differing conventions need to be documented well.
        import scipy.stats
        cdf_1d = scipy.stats.norm.cdf(cutoff)
        confidence_1d = 1-(1-cdf_1d)*2
        cutoff_nd = scipy.stats.chi2.ppf(confidence_1d, gmm.D)

    # begin EM
    it = 0
    maxiter = max(100, gmm.K)
    if VERBOSITY:
        global VERB_BUFFER
        print "ITER\tPOINTS\tLOG_L\tN_STABLE"

    while it < maxiter: # limit loop in case of slow convergence

        log_L_, N = _EMstep(gmm, log_p, nbh, T_inv, log_S, H, data, covar=covar, sel_callback=sel_callback, w=w, pool=pool, chunksize=chunksize, cutoff=cutoff_nd, tol=tol, rng=rng)

        if VERBOSITY:
            print ("%d\t%d\t%.3f" % (it, N, log_L_)),

        # convergence tests:
        if it > 0 and log_L_ - log_L < tol:
            log_L = log_L_
            if VERBOSITY:
                print "\nmean likelihood converged within tolerance %r: stopping here." % tol
            break

        if sel_callback is not None:
            # with imputation the observed data logL can decrease.
            # revert to previous model if that is the case
            if it > 0 and log_L_ < log_L:
                if VERBOSITY:
                    print "\nmean likelihood decreased: stopping reverting to previous model."
                gmm = gmm_
                break

        # with a a cutoff: we may have to update the nbhs
        if cutoff is not None:
            # check if component has moved by more than sigma/2
            shift2 = np.einsum('...i,...ij,...j', gmm.mean - gmm_.mean, np.linalg.inv(gmm_.covar), gmm.mean - gmm_.mean)
            moved = shift2 > 0.5**2

            # force update to nbh
            for c in moved:
                nbh[c] = None

            if VERBOSITY:
                print "\t%d" % (gmm.K - moved.size),

            if VERBOSITY >= 2 and moved.any():
                VERB_BUFFER += "\nresetting nbhs of moving components: "
                VERB_BUFFER += ("(" + "%d," * moved.sum() + ")") % tuple(np.flatnonzero(moved))

        if VERBOSITY:
            print VERB_BUFFER
            if len(VERB_BUFFER):
                VERB_BUFFER = ""

        # update all important _ quantities for convergence test(s)
        log_L = log_L_
        # backup to see if components move of if next step gets worse
        # note: gmm = gmm_!
        gmm_.amp[:] = gmm.amp[:]
        gmm_.mean[:,:] = gmm.mean[:,:]
        gmm_.covar[:,:,:] = gmm.covar[:,:,:]

        log_S[:] = 0
        H[:] = 0
        it += 1

    pool.close()
    return log_L, nbh

def _EMstep(gmm, log_p, nbh, T_inv, log_S, H, data, covar=None, sel_callback=None, w=0, pool=None, chunksize=1, cutoff=None, tol=1e-3, rng=np.random):
    import parmap
    # compute p(i | k) for each k independently in the pool
    # need S = sum_k p(i | k) for further calculation
    # also N = {i | i in neighborhood[k]} for any k
    k = 0
    for log_p[k], nbh[k], T_inv[k] in \
    parmap.starmap(_E, zip(xrange(gmm.K), nbh), gmm, data, covar, cutoff, pool=pool, chunksize=chunksize):
        log_S[nbh[k]] += np.exp(log_p[k]) # actually S, not logS
        H[nbh[k]] = 1
        k += 1

    # need log(S), but since log(0) isn't a good idea, need to restrict to N_
    log_S[H] = np.log(log_S[H])
    log_L_ = log_S[H].mean()
    N = H.sum()

    # perform M step with M-sums of data and imputations runs
    _M(gmm, nbh, log_p, T_inv, log_S, H, N, data, covar=covar, w=w, sel_callback=sel_callback, cutoff=cutoff, tol=tol, pool=pool, chunksize=chunksize, rng=rng)
    return log_L_, N


def _E(k, nbh_k, gmm, data, covar=None, cutoff=None):
    # p(x | k) for all x in the vicinity of k
    # determine all points within cutoff sigma from mean[k]
    if nbh_k is None:
        dx = data - gmm.mean[k]
    else:
        dx = data[nbh_k] - gmm.mean[k]

    if covar is None:
         T_inv_k = None
         chi2 = np.einsum('...i,...ij,...j', dx, np.linalg.inv(gmm.covar[k]), dx)
    else:
        # with data errors: need to create and return T_ik = covar_i + C_k
        # and weight each datum appropriately
        if covar.shape == (gmm.D, gmm.D): # one-for-all
            T_inv_k = np.linalg.inv(gmm.covar[k] + covar)
        else: # each datum has covariance
            T_inv_k = np.linalg.inv(gmm.covar[k] + covar[nbh_k].reshape(len(dx), gmm.D, gmm.D))
        chi2 = np.einsum('...i,...ij,...j', dx, T_inv_k, dx)

    # NOTE: close to convergence, we could stop applying the cutoff because
    # changes to nbh will be minimal
    if cutoff is not None:
        indices = chi2 < cutoff
        if indices.any():
            # if all indices are used: probably time to increase nbh
            if indices.all():
                changed = 0
            chi2 = chi2[indices]
            if covar is not None and covar.shape != (gmm.D, gmm.D):
                T_inv_k = T_inv_k[indices]
            if nbh_k is None:
                nbh_k = np.flatnonzero(indices)
            else:
                nbh_k = nbh_k[indices]

    # prevent tiny negative determinants to mess up
    (sign, logdet) = np.linalg.slogdet(gmm.covar[k])

    log2piD2 = np.log(2*np.pi)*(0.5*gmm.D)
    return np.log(gmm.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2, nbh_k, T_inv_k


def _M(gmm, nbh, log_p, T_inv, log_S, H, N, data, covar=None, w=0, cutoff=None, sel_callback=None, tol=1e-3, pool=None, chunksize=1, rng=np.random):

    # save the M sums from observed data
    A = np.empty(gmm.K)                 # sum for amplitudes
    M = np.empty((gmm.K, gmm.D))        # ... means
    C = np.empty((gmm.K, gmm.D, gmm.D)) # ... covariances
    JS = np.empty(gmm.K)                # split criterion, zero if unused

    # perform sums for M step in the pool
    import parmap
    k = 0
    for A[k], M[k,:], C[k,:,:] in \
    parmap.starmap(_computeMSums, zip(xrange(gmm.K), nbh, log_p, T_inv), gmm, data, log_S, pool=pool, chunksize=chunksize):
        k += 1

    A2, M2, C2, N2 = _getIMSums(gmm, nbh, N, covar=covar, cutoff=cutoff, pool=pool, chunksize=chunksize, sel_callback=sel_callback, rng=rng)

    if VERBOSITY:
        sel_outside = A2 > tol * A
        if VERBOSITY >= 2 and sel_outside.any():
            global VERB_BUFFER
            VERB_BUFFER += "\ncomponent inside fractions: "
            VERB_BUFFER += ("(" + "%.2f," * gmm.K + ")") % tuple(A/(A+A2))

    _update(gmm, A, M, C, N, A2, M2, C2, N2, w)


def _update(gmm, A, M, C, N, A2, M2, C2, N2, w, altered=None):
    # M-step for all components using data (and data2, if non-zero sums are set)

    # partial EM: normal update for mean and covar, but constrained for amp
    if altered is None:
        changed = slice(None)
    else:
        changed = altered

    if altered is None:
        gmm.amp[changed] = (A + A2)[changed] / (N + N2)
    else:
        # Bovy eq. 31
        unaltered = np.in1d(xrange(gmm.K), altered, assume_unique=True, invert=True)
        gmm.amp[altered] = (A + A2)[altered] / (A + A2)[altered].sum() * (1 - (gmm.amp[unaltered]).sum())
    # because of finite precision during the imputation: renormalize
    gmm.amp /= gmm.amp.sum()

    gmm.mean[changed,:] = (M + M2)[changed,:]/(A + A2)[changed,None]
    # minimum covariance term?
    if w > 0:
        # we assume w to be a lower bound of the isotropic dispersion,
        # C_k = w^2 I + ...
        # then eq. 38 in Bovy et al. only ~works for N = 0 because of the
        # prefactor 1 / (q_j + 1) = 1 / (A + 1) in our terminology
        # On average, q_j = N/K, so we'll adopt that to correct.
        w_eff = w**2 * ((N+N2)*1./gmm.K + 1)
        gmm.covar[changed,:,:] = (C + C2 + w_eff*np.eye(gmm.D)[None,:,:])[changed,:,:] / (A + A2 + 1)[changed,None,None]
    else:
        gmm.covar[changed,:,:] = (C + C2)[changed,:,:] / (A + A2)[changed,None,None]


def _computeMSums(k, nbh_k, log_p_k, T_inv_k, gmm, data, log_S):
    if log_p_k.size:

        # form log_q_ik by dividing with S = sum_k p_ik
        # NOTE:  this modifies log_p_k in place!
        # NOTE2: reshape needed when nbh_k is None because of its
        # implicit meaning as np.newaxis (which would create a 2D array)
        log_p_k -= log_S[nbh_k].reshape(log_p_k.size)

        # amplitude: A_k = sum_i q_ik
        A_k = np.exp(logsum(log_p_k))

        # in fact: q_ik, but we treat sample index i silently everywhere
        q_k = np.exp(log_p_k)

        # data with errors?
        d = data[nbh_k].reshape((log_p_k.size, gmm.D))
        if T_inv_k is None:
            # mean: M_k = sum_i x_i q_ik
            M_k = (d * q_k[:,None]).sum(axis=0)

            # covariance: C_k = sum_i (x_i - mu_k)^T(x_i - mu_k) q_ik
            d_m = d - gmm.mean[k]
            # funny way of saying: for each point i, do the outer product
            # of d_m with its transpose, multiply with pi[i], and sum over i
            C_k = (q_k[:, None, None] * d_m[:, :, None] * d_m[:, None, :]).sum(axis=0)
        else:
            # need temporary variables:
            # b_ik = mu_k + C_k T_ik^-1 (x_i - mu_k)
            # B_ik = C_k - C_k T_ik^-1 C_k
            # to replace pure data-driven means and covariances
            d_m = d - gmm.mean[k]
            b_k = gmm.mean[k] + np.einsum('ij,...jk,...k', gmm.covar[k], T_inv_k, d_m)
            M_k = (b_k * q_k[:,None]).sum(axis=0)

            b_k -= gmm.mean[k]
            B_k = gmm.covar[k] - np.einsum('ij,...jk,...kl', gmm.covar[k], T_inv_k, gmm.covar[k])
            C_k = (q_k[:, None, None] * (b_k[:, :, None] * b_k[:, None, :] + B_k)).sum(axis=0)
        return A_k, M_k, C_k
    else:
        return 0,0,0


def _getIMSums(gmm, nbh, N, covar=None, cutoff=None, pool=None, chunksize=1, sel_callback=None, over=1, rng=np.random):

    A2 = M2 = C2 = N2 = 0

    if sel_callback is not None:
        tol = 1e-2
        size = N*over

        # create fake data with same mechanism as the original data,
        # but invert selection to get the missing part
        data2, covar2, nbh2 = _I(gmm, size, sel_callback, cutoff=cutoff, covar=covar, nbh=nbh, rng=rng)

        N2 = len(data2)
        if N2 == 0:
            return A2, M2, C2, N2

        A2 = np.zeros(gmm.K)
        M2 = np.zeros((gmm.K, gmm.D))
        C2 = np.zeros((gmm.K, gmm.D, gmm.D))

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
        for log_p2[k], nbh2[k], T2_inv[k] in \
        parmap.starmap(_E, zip(xrange(gmm.K), nbh2), gmm, data2, covar2, None, pool=pool, chunksize=chunksize):
            log_S2[nbh2[k]] += np.exp(log_p2[k])
            N2_[nbh2[k]] = 1
            k += 1

        log_S2[N2_] = np.log(log_S2[N2_])
        N2 = N2_.sum()

        k = 0
        for A2[k], M2[k,:], C2[k,:,:] in \
        parmap.starmap(_computeMSums, zip(xrange(gmm.K), nbh2, log_p2, T2_inv), gmm, data2, log_S2, pool=pool, chunksize=chunksize):
            k += 1

        # normalize over-sampling
        A2 /= over
        M2 /= over
        C2 /= over
        N2 /= over

    return A2, M2, C2, N2


def _overlappingWith(k, gmm, cutoff=5):
    if cutoff is not None:
        chi2_k = gmm.logL_k(k, gmm.mean, covar=gmm.covar, chi2_only=True)
        return np.flatnonzero(chi2_k < cutoff)
    else:
        return np.ones(gmm.K, dtype='bool')


def _I(gmm, size, sel_callback, cutoff=3, covar=None, nbh=None, covar_reduce_fct=np.mean, rng=np.random):

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
            # determine covar2 from given input covariances
            if covar.shape == (gmm.D, gmm.D): # one-for-all
                error_k = covar
            else:
                error_k = covar_reduce_fct(covar[nbh[k]], axis=0)
                covar2[lower:upper, :,:] = error_k
            data2[lower:upper, :] = rng.multivariate_normal(gmm.mean[k], gmm.covar[k] + error_k, size=N2[k])
        component2[lower:upper] = k
        lower = upper

    # TODO: may want to decide whether to add noise before selection or after
    # Here we do noise, then selection, but this is not fundamental
    sel2 = ~sel_callback(data2, gmm=gmm)
    data2 = data2[sel2]
    component2 = component2[sel2]
    if covar is not None and covar.shape != (gmm.D, gmm.D):
        covar2 = covar2[sel2]

    # determine nbh of each component:
    # since components may overlap, simply using component2 is insufficient.
    # for the sake of speed, we simply add all points whose components overlap
    # with the one in question
    nbh2 = [None for k in xrange(gmm.K)]
    for k in xrange(gmm.K):
        overlap_k = _overlappingWith(k, gmm, cutoff=cutoff)
        mask2 = np.zeros(len(data2), dtype='bool')
        for j in overlap_k:
            mask2 |= (component2 == j)
        nbh2[k] = np.flatnonzero(mask2)

    return data2, covar2, nbh2
