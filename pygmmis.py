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


def chi2_cutoff(D, cutoff=3.):
    # compute effective cutoff for chi2 in D dimensions
    import scipy.stats
    cdf_1d = scipy.stats.norm.cdf(cutoff)
    confidence_1d = 1-(1-cdf_1d)*2
    cutoff_nd = scipy.stats.chi2.ppf(confidence_1d, D)
    return cutoff_nd


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

class Background(object):
    def __init__(self, D=1):
        self.amp = 0
        self.D = D
        self.V = None
        self.adjust_amp = True

    @property
    def p(self):
        if self.V is not None:
            return 1./self.V
        else:
            raise RuntimeError("Background.p: volume V not set!")

    def computeVolume(self, data, sel_callback=None, sel_B=10, sel_over=10, rng=np.random):
        # volumne spanned by data
        min_pos = data.min(axis=0)
        max_pos = data.max(axis=0)
        delta_pos = max_pos - min_pos

        # limit to observed subvolume
        if sel_callback is None:
            self.V = np.prod(delta_pos)
        else:
            # divide range into B cells per dimensions
            N_cells = sel_B**3
            N_samples = N_cells * sel_over
            samples = delta_pos * rng.rand(N_samples, self.D) + min_pos
            sel = sel_callback(samples)
            histdd = np.histogramdd(samples[sel], sel_B)[0]
            filled = np.flatnonzero(histdd).size
            # multiply with volume of each cell
            vol_cell = np.prod(delta_pos/sel_B)
            self.V = filled * vol_cell
        return self.V

############################
# Begin of fit functions
############################

VERBOSITY = False

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
def initFromSimpleGMM(gmm, data, covar=None, s=None, k=None, rng=np.random, init_callback=initFromDataAtRandom, w=0., cutoff=None, background=None, tol=1e-3, covar_factor=1.):
    # 1) run GMM without error and selection (fit is essentially an init fct)
    fit(gmm, data, covar=None, w=w, cutoff=cutoff, sel_callback=None, init_callback=init_callback, background=background, tol=tol, rng=rng)
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

# initialize from k-means clusters
# use Algorithm 1 from Bloemer & Bujna (arXiv:1312.5946)
def initFromKMeans(gmm, data, covar=None, rng=np.random):
    from scipy.cluster.vq import kmeans2
    center, label = kmeans2(data, gmm.K)
    for k in xrange(gmm.K):
        mask = (label == k)
        gmm.amp[k] = mask.sum() * 1./len(data)
        gmm.mean[k,:] = data[mask].mean(axis=0)
        d_m = data[mask] - gmm.mean[k]
        # funny way of saying: for each point i, do the outer product
        # of d_m with its transpose and sum over i
        gmm.covar[k,:,:] = (d_m[:, :, None] * d_m[:, None, :]).sum(axis=0) / len(data)


def fit(gmm, data, covar=None, w=0., cutoff=None, sel_callback=None, init_callback=None, background=None, tol=1e-3, split_n_merge=False, rng=np.random):
    # NOTE: If background is set, it implies cutoff=None

    # init components
    if init_callback is not None:
        init_callback(gmm, data=data, covar=covar, rng=rng)
    elif VERBOSITY:
        print "forgoing initialization. hopefully GMM was initialized..."

    # set up pool
    import multiprocessing
    pool = multiprocessing.Pool()
    n_chunks, chunksize = gmm._mp_chunksize()

    # sum_k p(x|k) -> S
    # extra precautions for cases when some points are treated as outliers
    # and not considered as belonging to any component
    log_S = createShared(np.zeros(len(data)))  # S = sum_k p(x|k)
    # FIXME: create sheared boolean array results in
    # AttributeError: 'c_bool' object has no attribute '__array_interface__'
    H = np.zeros(len(data), dtype='bool')      # H == 1 for points in the fit
    log_p = [[] for k in xrange(gmm.K)]   # P = p(x|k) for x in U[k]
    T_inv = [None for k in xrange(gmm.K)] # T = covar(x) + gmm.covar[k]
    U = [None for k in xrange(gmm.K)]     # U = {x close to k}

    if VERBOSITY:
        global VERB_BUFFER

    log_L, N, N2 = _EM(gmm, log_p, U, T_inv, log_S, H, data, covar=covar, sel_callback=sel_callback, w=w, pool=pool, chunksize=chunksize, cutoff=cutoff, background=background, tol=tol, rng=rng)

    # should we try to improve by split'n'merge of components?
    # if so, keep backup copy
    gmm_ = GMM(gmm.K, gmm.D)
    while split_n_merge and gmm.K >= 3:

        gmm_.amp[:] = gmm.amp[:]
        gmm_.mean[:] = gmm.mean[:,:]
        gmm_.covar[:,:,:] = gmm.covar[:,:,:]
        U_ = [U[k].copy() for k in xrange(gmm.K)]

        altered, cleanup = _findSNMComponents(gmm, U, log_p, log_S, N+N2, pool=pool, chunksize=chunksize)

        if VERBOSITY:
            print ("merging %d and %d, splitting %d" % tuple(altered))

        # modify components
        _update_snm(gmm, altered, U, N+N2, cleanup)

        # run partial EM on altered components
        # NOTE: for a partial run, we'd only need the change to Log_S from the
        # altered components. However, the neighborhoods can change from _update_snm
        # or because they move, so that operation is ill-defined.
        # Thus, we'll always run a full E-step, which is pretty cheap for
        # converged neighborhood.
        # The M-step could in principle be run on the altered components only,
        # but there seem to be side effects in what I've tried.
        # Similar to the E-step, the imputation step needs to be run on all
        # components, otherwise the contribution of the altered ones to the mixture
        # would be over-estimated.
        # Effectively, partial runs are as expensive as full runs.
        log_L_, N_, N2_ = _EM(gmm, log_p, U, T_inv, log_S, H, data, covar=covar, sel_callback=sel_callback, w=w, pool=pool, chunksize=chunksize, cutoff=cutoff, background=background, tol=tol, prefix="SNM_P", altered=altered, rng=rng)

        log_L_, N_, N2_ = _EM(gmm, log_p, U, T_inv, log_S, H, data, covar=covar, sel_callback=sel_callback, w=w, pool=pool, chunksize=chunksize, cutoff=cutoff, background=background, tol=tol, prefix="SNM_F", altered=None, rng=rng)

        if log_L >= log_L_:
            # revert to backup
            gmm.amp[:] = gmm_.amp[:]
            gmm.mean[:] = gmm_.mean[:,:]
            gmm.covar[:,:,:] = gmm_.covar[:,:,:]
            U = U_
            if VERBOSITY:
                print ("split'n'merge likelihood decreased: reverting to previous model")
            break

        log_L = log_L_
        split_n_merge -= 1

    pool.close()
    return log_L, U

def _EM(gmm, log_p, U, T_inv, log_S, H, data, covar=None, sel_callback=None, background=None, w=0, pool=None, chunksize=1, cutoff=None, tol=1e-3, prefix="", altered=None, rng=np.random):

    # compute effective cutoff for chi2 in D dimensions
    if cutoff is not None:
        # note: subsequently the cutoff parameter, e.g. in _E(), refers to this:
        # chi2 < cutoff,
        # while in fit() it means e.g. "cut at 3 sigma".
        # These differing conventions need to be documented well.
        cutoff_nd = chi2_cutoff(gmm.D, cutoff=cutoff)

        # store chi2 cutoff for component shifts, use 0.5 sigma
        shift_cutoff = chi2_cutoff(gmm.D, cutoff=min(0.5, cutoff/2))
    else:
        cutoff_nd = None
        shift_cutoff = chi2_cutoff(gmm.D, cutoff=0.5)

    it = 0
    maxiter = max(100, gmm.K)
    if VERBOSITY:
        global VERB_BUFFER
        print("\nITER\tPOINTS\tIMPUTED\tLOG_L\tSTABLE")

    # save backup
    gmm_ = GMM(gmm.K, gmm.D)
    gmm_.amp[:] = gmm.amp[:]
    gmm_.mean[:,:] = gmm.mean[:,:]
    gmm_.covar[:,:,:] = gmm.covar[:,:,:]
    while it < maxiter: # limit loop in case of slow convergence

        log_L_, N, N2 = _EMstep(gmm, log_p, U, T_inv, log_S, H, data, covar=covar, sel_callback=sel_callback, background=background, w=w, pool=pool, chunksize=chunksize, cutoff=cutoff_nd, tol=tol, altered=altered, it=it, rng=rng)

        # check if component has moved by more than sigma/2
        shift2 = np.einsum('...i,...ij,...j', gmm.mean - gmm_.mean, np.linalg.inv(gmm_.covar), gmm.mean - gmm_.mean)
        moved = shift2 > shift_cutoff

        if VERBOSITY:
            print("%s%d\t%d\t%d\t%.3f\t%d" % (prefix, it, N, N2, log_L_, (gmm.K - moved.sum())))

        # convergence tests:
        if it > 0 and log_L_ - log_L < tol:
            # with imputation the observed data logL can decrease.
            # revert to previous model if that is the case
            if sel_callback is not None and log_L_ < log_L:
                gmm.amp[:] = gmm_.amp[:]
                gmm.mean[:,:] = gmm_.mean[:,:]
                gmm.covar[:,:,:] = gmm_.covar[:,:,:]
                if VERBOSITY:
                    print("likelihood decreased: reverting to previous model")
            else:
                log_L = log_L_
                if VERBOSITY:
                    print ("likelihood converged within tolerance %r: stopping here." % tol)
            break

        # force update to U for all moved components
        if cutoff is not None:
            for c in moved:
                U[c] = None

        if VERBOSITY >= 2 and moved.any():
            print ("resetting neighborhoods of moving components: (" + ("%d," * moved.sum() + ")") % tuple(np.flatnonzero(moved)))

        # update all important _ quantities for convergence test(s)
        log_L = log_L_
        # backup to see if components move or if next step gets worse
        # note: not gmm = gmm_ !
        gmm_.amp[:] = gmm.amp[:]
        gmm_.mean[:,:] = gmm.mean[:,:]
        gmm_.covar[:,:,:] = gmm.covar[:,:,:]

        it += 1
    return log_L, N, N2


def _EMstep(gmm, log_p, U, T_inv, log_S, H, data, covar=None, sel_callback=None, background=None, w=0, pool=None, chunksize=1, cutoff=None, tol=1e-3, altered=None, it=0, rng=np.random):
    import parmap

    if background is None:
        # compute p(i | k) for each k independently in the pool
        # need S = sum_k p(i | k) for further calculation
        # also N = {i | i in neighborhood[k]} for any k
        log_S[:] = 0
        H[:] = 0
        k = 0
        for log_p[k], U[k], T_inv[k] in \
        parmap.starmap(_Estep, zip(xrange(gmm.K), U), gmm, data, covar, cutoff, pool=pool, chunksize=chunksize):
            log_S[U[k]] += np.exp(log_p[k]) # actually S, not logS
            H[U[k]] = 1
            k += 1

        # need log(S), but since log(0) isn't a good idea, need to restrict to N_
        log_S[H] = np.log(log_S[H])
        log_L_ = log_S[H].mean()

    # determine which points belong to background:
    # compare uniform background model with GMM,
    # use H to store association to signal vs background
    # that decision conflicts with per-component U's, which also underestimates
    # the probabilities under the signal model.
    # Thus, we ignore any cutoff and compute p(x|k) for all x and k
    else:
        if it == 0:
            log_S[:] = gmm(data, covar=covar)
        else:
            log_S[:] = np.exp(log_S[:])


        p_bg = background.amp * background.p
        q_bg = p_bg / (p_bg + (1-background.amp)*log_S)
        H[:] = q_bg < 0.5

        # recompute background amplitude;
        # for flat log_S, this is identical to summing up samplings with H[i]==0
        if background.adjust_amp:
            background.amp = q_bg.sum() / len(data)
        print "BG:", background.amp, (H==0).sum(), np.median(q_bg), (log_S > 0).all()

        # reset signal U
        for k in xrange(gmm.K):
            U[k] = None

        # don't use cutoff and don't update H:
        # for the signal part: set U[k] = H for the M-step
        k = 0
        log_S[:] = 0
        for log_p[k], U[k], T_inv[k] in \
        parmap.starmap(_Estep, zip(xrange(gmm.K), U), gmm, data, covar, None, pool=pool, chunksize=chunksize):
            log_S += np.exp(log_p[k]) # actually S, not logS
            U[k] = H # shallow copy
            log_p[k] = log_p[k][H]
            k += 1

        log_L_ = np.log((1-background.amp) * log_S + background.amp * background.p).mean()
        log_S[:] = np.log(log_S[:])

    N = H.sum()

    # perform M step with M-sums of data and imputations runs
    N2 = _Mstep(gmm, U, log_p, T_inv, log_S, H, N, data, covar=covar, w=w, sel_callback=sel_callback, cutoff=cutoff, tol=tol, pool=pool, chunksize=chunksize, altered=altered, rng=rng)
    return log_L_, N, N2


def _Estep(k, U_k, gmm, data, covar=None, cutoff=None):
    # p(x | k) for all x in the vicinity of k
    # determine all points within cutoff sigma from mean[k]
    if U_k is None:
        dx = data - gmm.mean[k]
    else:
        dx = data[U_k] - gmm.mean[k]

    if covar is None:
         T_inv_k = None
         chi2 = np.einsum('...i,...ij,...j', dx, np.linalg.inv(gmm.covar[k]), dx)
    else:
        # with data errors: need to create and return T_ik = covar_i + C_k
        # and weight each datum appropriately
        if covar.shape == (gmm.D, gmm.D): # one-for-all
            T_inv_k = np.linalg.inv(gmm.covar[k] + covar)
        else: # each datum has covariance
            T_inv_k = np.linalg.inv(gmm.covar[k] + covar[U_k].reshape(len(dx), gmm.D, gmm.D))
        chi2 = np.einsum('...i,...ij,...j', dx, T_inv_k, dx)

    # NOTE: close to convergence, we could stop applying the cutoff because
    # changes to U will be minimal
    if cutoff is not None:
        indices = chi2 < cutoff
        if indices.any():
            chi2 = chi2[indices]
            if covar is not None and covar.shape != (gmm.D, gmm.D):
                T_inv_k = T_inv_k[indices]
            if U_k is None:
                U_k = np.flatnonzero(indices)
            else:
                U_k = U_k[indices]

    # prevent tiny negative determinants to mess up
    (sign, logdet) = np.linalg.slogdet(gmm.covar[k])

    log2piD2 = np.log(2*np.pi)*(0.5*gmm.D)
    return np.log(gmm.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2, U_k, T_inv_k


def _Mstep(gmm, U, log_p, T_inv, log_S, H, N, data, covar=None, w=0, cutoff=None, sel_callback=None, tol=1e-3, pool=None, chunksize=1, altered=None, rng=np.random):

    # save the M sums from observed data
    A = np.empty(gmm.K)                 # sum for amplitudes
    M = np.empty((gmm.K, gmm.D))        # ... means
    C = np.empty((gmm.K, gmm.D, gmm.D)) # ... covariances

    # perform sums for M step in the pool
    # FIXME: in a partial run, could work on altered components only;
    # however, there seem to be side effects or race conditions
    import parmap
    k = 0
    for A[k], M[k,:], C[k,:,:] in \
    parmap.starmap(_computeMSums, zip(xrange(gmm.K), U, log_p, T_inv), gmm, data, log_S, pool=pool, chunksize=chunksize):
        k += 1

    # imputation step needs to be done with all components, even if only
    # altered ones are relevant for a partial run, otherwise their contribution
    # gets over-estimated
    A2, M2, C2, N2 = _getIMSums(gmm, U, N, covar=covar, cutoff=cutoff, pool=pool, chunksize=chunksize, sel_callback=sel_callback, rng=rng)

    if VERBOSITY:
        sel_outside = A2 > tol * A
        if VERBOSITY >= 2 and sel_outside.any():
            print ("component inside fractions: " + ("(" + "%.2f," * gmm.K + ")") % tuple(A/(A+A2)))

    _update(gmm, A, M, C, N, A2, M2, C2, N2, w, altered=altered)
    return N2


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


def _computeMSums(k, U_k, log_p_k, T_inv_k, gmm, data, log_S):
    if log_p_k.size:
        # get log_q_ik by dividing with S = sum_k p_ik
        # NOTE:  this modifies log_p_k in place, but is only relevant
        # within this method since the call is parallel and its arguments
        # therefore don't get updated.
        # NOTE: reshape needed when U_k is None because of its
        # implicit meaning as np.newaxis (which would create a 2D array)
        log_p_k -= log_S[U_k].reshape(log_p_k.size)

        # amplitude: A_k = sum_i q_ik
        A_k = np.exp(logsum(log_p_k))

        # in fact: q_ik, but we treat sample index i silently everywhere
        q_k = np.exp(log_p_k)

        # data with errors?
        d = data[U_k].reshape((log_p_k.size, gmm.D))
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


def _getIMSums(gmm, U, N, covar=None, cutoff=None, pool=None, chunksize=1, sel_callback=None, over=1, rng=np.random):

    A2 = M2 = C2 = N2 = 0

    if sel_callback is not None:
        tol = 1e-2
        size = N*over

        # create fake data with same mechanism as the original data,
        # but invert selection to get the missing part
        data2, covar2, U2 = _I(gmm, size, sel_callback, cutoff=cutoff, covar=covar, U=U, rng=rng)

        N2 = len(data2)
        if N2 == 0:
            return A2, M2, C2, N2

        A2 = np.zeros(gmm.K)
        M2 = np.zeros((gmm.K, gmm.D))
        C2 = np.zeros((gmm.K, gmm.D, gmm.D))

        # similar setup as above: E-step plus the sums from the Mstep.
        # the latter will be added to the original data sums
        log_S2 = np.zeros(len(data2))
        H2 = np.zeros(len(data2), dtype='bool')
        log_p2 = [[] for k in xrange(gmm.K)]
        T2_inv = [None for k in xrange(gmm.K)]

        # run E-step on data2
        import parmap
        k = 0
        for log_p2[k], U2[k], T2_inv[k] in \
        parmap.starmap(_Estep, zip(xrange(gmm.K), U2), gmm, data2, covar2, cutoff, pool=pool, chunksize=chunksize):
            log_S2[U2[k]] += np.exp(log_p2[k])
            H2[U2[k]] = 1
            k += 1

        log_S2[H2] = np.log(log_S2[H2])
        N2 = H2.sum()

        k = 0
        for A2[k], M2[k,:], C2[k,:,:] in \
        parmap.starmap(_computeMSums, zip(xrange(gmm.K), U2, log_p2, T2_inv), gmm, data2, log_S2, pool=pool, chunksize=chunksize):
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


def _I(gmm, size, sel_callback, cutoff=3, covar=None, U=None, covar_reduce_fct=np.mean, rng=np.random):

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
                error_k = covar_reduce_fct(covar[U[k]], axis=0)
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

    # determine U of each component:
    # since components may overlap, simply using component2 is insufficient.
    # for the sake of speed, we simply add all points whose components overlap
    # with the one in question
    # FIXME: This method does not account for added noise that could "connect"
    # two otherwise disjoint components.
    U2 = [None for k in xrange(gmm.K)]
    for k in xrange(gmm.K):
        overlap_k = _overlappingWith(k, gmm, cutoff=cutoff)
        mask2 = np.zeros(len(data2), dtype='bool')
        for j in overlap_k:
            mask2 |= (component2 == j)
        U2[k] = np.flatnonzero(mask2)

    return data2, covar2, U2


def _JS(k, gmm, log_p, log_S, U, A):
    # compute Kullback-Leiber divergence
    log_q_k = log_p[k] - log_S[U[k]]
    return np.dot(np.exp(log_q_k), log_q_k - np.log(A[k]) - log_p[k] + np.log(gmm.amp[k])) / A[k]

def _findSNMComponents(gmm, U, log_p, log_S, N, pool=None, chunksize=1):
    # find those components that are most similar
    JM = np.zeros((gmm.K, gmm.K))
    # compute log_q (posterior for k given i), but use normalized probabilities
    # to allow for merging of empty components
    log_q = [log_p[k] - log_S[U[k]] - np.log(gmm.amp[k]) for k in xrange(gmm.K)]
    for k in xrange(gmm.K):
        # don't need diagonal (can merge), and JM is symmetric
        for j in xrange(k+1, gmm.K):
            # get index list for intersection of U of k and l
            # FIXME: match1d fails if either U is empty
            # SOLUTION: merge empty U, split another
            i_k, i_j = match1d(U[k], U[j], presorted=True)
            JM[k,j] = np.dot(np.exp(log_q[k][i_k]), np.exp(log_q[j][i_j]))
    merge_jk = np.unravel_index(JM.argmax(), JM.shape)
    # if all Us are disjunct, JM is blank and merge_jk = [0,0]
    # merge two smallest components and clean up from the bottom
    cleanup = False
    if merge_jk[0] == 0 and merge_jk[1] == 0:
        global VERBOSITY
        if VERBOSITY >= 2:
            print "neighborhoods disjunct. merging components %d and %d" % tuple(merge_jk)
        merge_jk = np.argsort(gmm.amp)[:2]
        cleanup = True


    # split the one whose p(x|k) deviate most from current Gaussian
    # ask for the three worst components to avoid split being in merge_jk
    """
    JS = np.empty(gmm.K)
    import parmap
    k = 0
    A = gmm.amp * N
    for JS[k] in \
    parmap.map(_JS, xrange(gmm.K), gmm, log_p, log_S, U, A, pool=pool, chunksize=chunksize):
        k += 1
    """
    # get largest Eigenvalue, weighed by amplitude
    # Large EV implies extended object, which often is caused by coverving
    # multiple clusters. This happes also for almost empty components, which
    # should rather be merged than split, hence amplitude weights.
    EV = np.linalg.svd(gmm.covar, compute_uv=False)
    JS = EV[:,0] * gmm.amp
    split_l3 = np.argsort(JS)[-3:][::-1]

    # check that the three indices are unique
    altered = np.array([merge_jk[0], merge_jk[1], split_l3[0]])
    if split_l3[0] in merge_jk:
        if split_l3[1] not in merge_jk:
            altered[2] = split_l3[1]
        else:
            altered[2] = split_l3[2]
    return altered, cleanup

def _update_snm(gmm, altered, U, N, cleanup):
    # reconstruct A from gmm.amp
    A = gmm.amp * N

    # update parameters and U
    # merge 0 and 1, store in 0, Bovy eq. 39
    gmm.amp[altered[0]] = gmm.amp[altered[0:2]].sum()
    if not cleanup:
        gmm.mean[altered[0]] = np.sum(gmm.mean[altered[0:2]] * A[altered[0:2]][:,None], axis=0) / A[altered[0:2]].sum()
        gmm.covar[altered[0]] = np.sum(gmm.covar[altered[0:2]] * A[altered[0:2]][:,None,None], axis=0) / A[altered[0:2]].sum()
        U[altered[0]] = np.union1d(U[altered[0]], U[altered[1]])
    else:
        # if we're cleaning up the weakest components:
        # merging does not lead to valid component parameters as the original
        # ones can be anywhere. Simply adopt second one.
        gmm.mean[altered[0],:] = gmm.mean[altered[1],:]
        gmm.covar[altered[0],:,:] = gmm.covar[altered[1],:,:]
        U[altered[0]] = U[altered[1]]

    # split 2, store in 1 and 2
    # following SVD method in Zhang 2003, with alpha=1/2, u = 1/4
    gmm.amp[altered[1]] = gmm.amp[altered[2]] = gmm.amp[altered[2]] / 2
    _, radius2, rotation = np.linalg.svd(gmm.covar[altered[2]])
    dl = np.sqrt(radius2[0]) *  rotation[0] / 4
    gmm.mean[altered[1]] = gmm.mean[altered[2]] - dl
    gmm.mean[altered[2]] = gmm.mean[altered[2]] + dl
    gmm.covar[altered[1:]] = np.linalg.det(gmm.covar[altered[2]])**(1./gmm.D) * np.eye(gmm.D)
    U[altered[1]] = U[altered[2]].copy() # now 1 and 2 have same U
