from __future__ import division
import numpy as np
import scipy.special, scipy.stats
import ctypes

import logging
logger = logging.getLogger("pygmmis")

# set up multiprocessing
import multiprocessing
import parmap

def createShared(a, dtype=ctypes.c_double):
    """Create a shared array to be used for multiprocessing's processes.

    Taken from http://stackoverflow.com/questions/5549190/

    Works only for float, double, int, long types (e.g. no bool).

    Args:
        numpy array, arbitrary shape

    Returns:
        numpy array whose container is a multiprocessing.Array
    """
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

import types
# python 2 -> 3 adjustments
try:
    import copy_reg
except ImportError:
    import copyreg as copy_reg
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

try:
    xrange
except NameError:
    xrange = range

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
        logX: numpy array of logarithmic summands
        axis (int): axis to sum over

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
    return np.log(np.exp(logX + c[tuple(c_shape)]).sum(axis=axis)) - c


def chi2_cutoff(D, cutoff=3.):
    """D-dimensional eqiuvalent of "n sigma" cut.

    Evaluates the quantile function of the chi-squared distribution to determine
    the limit for the chi^2 of samples wrt to GMM so that they satisfy the
    68-95-99.7 percent rule of the 1D Normal distribution.

    Args:
        D (int): dimensions of the feature space
        cutoff (float): 1D equivalent cut [in units of sigma]

    Returns:
        float: upper limit for chi-squared in D dimensions
    """
    cdf_1d = scipy.stats.norm.cdf(cutoff)
    confidence_1d = 1-(1-cdf_1d)*2
    cutoff_nd = scipy.stats.chi2.ppf(confidence_1d, D)
    return cutoff_nd

def covar_callback_default(coords, default=None):
    N,D = coords.shape
    if default.shape != (D,D):
        raise RuntimeError("covar_callback received improper default covariance %r" % default)
    # no need to copy since a single covariance matrix is sufficient
    # return np.tile(default, (N,1,1))
    return default


class GMM(object):
    """Gaussian mixture model with K components in D dimensions.

    Attributes:
        amp: numpy array (K,), component amplitudes
        mean: numpy array (K,D), component means
        covar: numpy array (K,D,D), component covariances
    """
    def __init__(self, K=0, D=0):
        """Create the arrays for amp, mean, covar."""
        self.amp = np.zeros((K))
        self.mean = np.empty((K,D))
        self.covar = np.empty((K,D,D))

    @property
    def K(self):
        """int: number of components, depends on size of amp."""
        return self.amp.size

    @property
    def D(self):
        """int: dimensions of the feature space."""
        return self.mean.shape[1]

    def save(self, filename, **kwargs):
        """Save GMM to file.

        Args:
            filename (str): name for saved file, should end on .npz as the default
                of numpy.savez(), which is called here
            kwargs:  dictionary of additional information to be stored in file.

        Returns:
            None
        """
        np.savez(filename, amp=self.amp, mean=self.mean, covar=self.covar, **kwargs)

    def load(self, filename):
        """Load GMM from file.

        Additional arguments stored by save() will be ignored.

        Args:
            filename (str): name for file create with save().

        Returns:
            None
        """
        F = np.load(filename)
        self.amp = F["amp"]
        self.mean = F["mean"]
        self.covar = F["covar"]
        F.close()

    @staticmethod
    def from_file(filename):
        """Load GMM from file.

        Additional arguments stored by save() will be ignored.

        Args:
            filename (str): name for file create with save().

        Returns:
            GMM
        """
        gmm = GMM()
        gmm.load(filename)
        return gmm

    def draw(self, size=1, rng=np.random):
        """Draw samples from the GMM.

        Args:
            size (int): number of samples to draw
            rng: numpy.random.RandomState for deterministic draw

        Returns:
            numpy array (size,D)
        """
        # draw indices for components given amplitudes, need to make sure: sum=1
        ind = rng.choice(self.K, size=size, p=self.amp/self.amp.sum())
        N = np.bincount(ind, minlength=self.K)

        # for each component: draw as many points as in ind from a normal
        samples = np.empty((size, self.D))
        lower = 0
        for k in np.flatnonzero(N):
            upper = lower + N[k]
            samples[lower:upper, :] = rng.multivariate_normal(self.mean[k], self.covar[k], size=N[k])
            lower = upper
        return samples

    def __call__(self, coords, covar=None, as_log=False):
        """Evaluate model PDF at given coordinates.

        see logL() for details.

        Args:
            coords: numpy array (D,) or (N, D) of test coordinates
            covar:  numpy array (D, D) or (N, D, D) covariance matrix of coords
            as_log (bool): return log(p) instead p

        Returns:
            numpy array (1,) or (N, 1) of PDF (or its log)
        """
        if as_log:
            return self.logL(coords, covar=covar)
        else:
            return np.exp(self.logL(coords, covar=covar))

    def _mp_chunksize(self):
        # find how many components to distribute over available threads
        cpu_count = multiprocessing.cpu_count()
        chunksize = max(1, self.K//cpu_count)
        n_chunks = min(cpu_count, self.K//chunksize)
        return n_chunks, chunksize

    def _get_chunks(self):
        # split all component in ideal-sized chunks
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

    def logL(self, coords, covar=None):
        """Log-likelihood of coords given all (i.e. the sum of) GMM components

        Distributes computation over all threads on the machine.

        If covar is None, this method returns
            log(sum_k(p(x | k)))
        of the data values x. If covar is set, the method returns
            log(sum_k(p(y | k))),
        where y = x + noise and noise ~ N(0, covar).

        Args:
            coords: numpy array (D,) or (N, D) of test coordinates
            covar:  numpy array (D, D) or (N, D, D) covariance matrix of coords

        Returns:
            numpy array (1,) or (N, 1) log(L), depending on shape of data
        """
        # Instead log p (x | k) for each k (which is huge)
        # compute it in stages: first for each chunk, then sum over all chunks
        pool = multiprocessing.Pool()
        chunks = self._get_chunks()
        results = [pool.apply_async(self._logsum_chunk, (chunk, coords, covar)) for chunk in chunks]
        log_p_y_chunk = []
        for r in results:
            log_p_y_chunk.append(r.get())
        pool.close()
        pool.join()
        return logsum(np.array(log_p_y_chunk)) # sum over all chunks = all k

    def _logsum_chunk(self, chunk, coords, covar=None):
        # helper function to reduce the memory requirement of logL
        log_p_y_k = np.empty((chunk[1]-chunk[0], len(coords)))
        for i in xrange(chunk[1] - chunk[0]):
            k = chunk[0] + i
            log_p_y_k[i,:] = self.logL_k(k, coords, covar=covar)
        return logsum(log_p_y_k)

    def logL_k(self, k, coords, covar=None, chi2_only=False):
        """Log-likelihood of coords given only component k.

        Args:
            k (int): component index
            coords: numpy array (D,) or (N, D) of test coordinates
            covar:  numpy array (D, D) or (N, D, D) covariance matrix of coords
            chi2_only (bool): only compute deltaX^T Sigma_k^-1 deltaX

        Returns:
            numpy array (1,) or (N, 1) log(L), depending on shape of data
        """
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
    """Background object to be used in conjuction with GMM.

    For a normalizable uniform distribution, a support footprint must be set.
    It should be sufficiently large to explain all non-clusters samples.

    Attributes:
        amp (float): mixing amplitude
        footprint: numpy array, (D,2) of rectangular volume
        adjust_amp (bool): whether amp will be adjusted as part of the fit
        amp_max (float): maximum value of amp allowed if adjust_amp=True
    """
    def __init__(self, footprint, amp=0):
        """Initialize Background with a footprint.

        Args:
            footprint: numpy array, (D,2) of rectangular volume

        Returns:
            None
        """
        self.amp = amp
        self.footprint = footprint
        self.adjust_amp = True
        self.amp_max = 1
        self.amp_min = 0

    @property
    def p(self):
        """Probability of the background model.

        Returns:
            float, equal to 1/volume, where volume is given by footprint.
        """
        volume = np.prod(self.footprint[1] - self.footprint[0])
        return 1/volume

    def draw(self, size=1, rng=np.random):
        """Draw samples from uniform background.

        Args:
            size (int): number of samples to draw
            rng: numpy.random.RandomState for deterministic draw

        Returns:
            numpy array (size, D)
        """
        dx = self.footprint[1] - self.footprint[0]
        return self.footprint[0] + dx*rng.rand(size,len(self.footprint[0]))


############################
# Begin of fit functions
############################

def initFromDataMinMax(gmm, data, covar=None, s=None, k=None, rng=np.random):
    """Initialization callback for uniform random component means.

    Component amplitudes are set at 1/gmm.K, covariances are set to
    s**2*np.eye(D), and means are distributed randomly over the range that is
    covered by data.

    If s is not given, it will be set such that the volume of all components
    completely fills the space covered by data.

    Args:
        gmm: A GMM to be initialized
        data: numpy array (N,D) to define the range of the component means
        covar: ignored in this callback
        s (float): if set, sets component variances
        k (iterable): list of components to set, is None sets all components
        rng: numpy.random.RandomState for deterministic behavior

    Returns:
        None
    """
    if k is None:
        k = slice(None)
    gmm.amp[k] = 1/gmm.K
    # set model to random positions with equally sized spheres within
    # volumne spanned by data
    min_pos = data.min(axis=0)
    max_pos = data.max(axis=0)
    gmm.mean[k,:] = min_pos + (max_pos-min_pos)*rng.rand(gmm.K, gmm.D)
    # if s is not set: use volume filling argument:
    # K spheres of radius s [having volume s^D * pi^D/2 / gamma(D/2+1)]
    # should completely fill the volume spanned by data.
    if s is None:
        vol_data = np.prod(max_pos-min_pos)
        s = (vol_data / gmm.K * scipy.special.gamma(gmm.D*0.5 + 1))**(1/gmm.D) / np.sqrt(np.pi)
        logger.info("initializing spheres with s=%.2f in data domain" % s)

    gmm.covar[k,:,:] = s**2 * np.eye(data.shape[1])

def initFromDataAtRandom(gmm, data, covar=None, s=None, k=None, rng=np.random):
    """Initialization callback for component means to follow data on scales > s.

    Component amplitudes are set to 1/gmm.K, covariances are set to
    s**2*np.eye(D). For each mean, a data sample is selected at random, and a
    multivariant Gaussian offset is added, whose variance is given by s**2.

    If s is not given, it will be set such that the volume of all components
    completely fills the space covered by data.

    Args:
        gmm: A GMM to be initialized
        data: numpy array (N,D) to define the range of the component means
        covar: ignored in this callback
        s (float): if set, sets component variances
        k (iterable): list of components to set, is None sets all components
        rng: numpy.random.RandomState for deterministic behavior

    Returns:
        None
    """
    if k is None:
        k = slice(None)
        k_len = gmm.K
    else:
        try:
            k_len = len(gmm.amp[k])
        except TypeError:
            k_len = 1
    gmm.amp[k] = 1/gmm.K
    # initialize components around data points with uncertainty s
    refs = rng.randint(0, len(data), size=k_len)
    D = data.shape[1]
    if s is None:
        min_pos = data.min(axis=0)
        max_pos = data.max(axis=0)
        vol_data = np.prod(max_pos-min_pos)
        s = (vol_data / gmm.K * scipy.special.gamma(gmm.D*0.5 + 1))**(1/gmm.D) / np.sqrt(np.pi)
        logger.info("initializing spheres with s=%.2f near data points" % s)

    gmm.mean[k,:] = data[refs] + rng.multivariate_normal(np.zeros(D), s**2 * np.eye(D), size=k_len)
    gmm.covar[k,:,:] = s**2 * np.eye(data.shape[1])

def initFromKMeans(gmm, data, covar=None, rng=np.random):
    """Initialization callback from a k-means clustering run.

    See Algorithm 1 from Bloemer & Bujna (arXiv:1312.5946)
    NOTE: The result of this call are not deterministic even if rng is set
    because scipy.cluster.vq.kmeans2 uses its own initialization.

    Args:
        gmm: A GMM to be initialized
        data: numpy array (N,D) to define the range of the component means
        covar: ignored in this callback
        rng: numpy.random.RandomState for deterministic behavior

    Returns:
        None
    """
    from scipy.cluster.vq import kmeans2
    center, label = kmeans2(data, gmm.K)
    for k in xrange(gmm.K):
        mask = (label == k)
        gmm.amp[k] = mask.sum() / len(data)
        gmm.mean[k,:] = data[mask].mean(axis=0)
        d_m = data[mask] - gmm.mean[k]
        # funny way of saying: for each point i, do the outer product
        # of d_m with its transpose and sum over i
        gmm.covar[k,:,:] = (d_m[:, :, None] * d_m[:, None, :]).sum(axis=0) / len(data)


def fit(gmm, data, covar=None, R=None, init_method='random', w=0., cutoff=None, sel_callback=None, oversampling=10, covar_callback=None, background=None, tol=1e-3, miniter=1, maxiter=1000, frozen=None, split_n_merge=False, rng=np.random):
    """Fit GMM to data.

    If given, init_callback is called to set up the GMM components. Then, the
    EM sequence is repeated until the mean log-likelihood converges within tol.

    Args:
        gmm: an instance if GMM
        data: numpy array (N,D)
        covar: sample noise covariance; numpy array (N,D,D) or (D,D) if i.i.d.
        R: sample projection matrix; numpy array (N,D,D)
        init_method (string): one of ['random', 'minmax', 'kmeans', 'none']
            defines the method to initialize the GMM components
        w (float): minimum covariance regularization
        cutoff (float): size of component neighborhood [in 1D equivalent sigmas]
        sel_callback: completeness callback to generate imputation samples.
        oversampling (int): number of imputation samples per data sample.
            only used if sel_callback is set.
            value of 1 is fine but results are noisy. Set as high as feasible.
        covar_callback: covariance callback for imputation samples.
            needs to be present if sel_callback and covar are set.
        background: an instance of Background if simultaneous fitting is desired
        tol (float): tolerance for covergence of mean log-likelihood
        maxiter (int): maximum number of iterations of EM
        frozen (iterable or dict): index list of components that are not updated
        split_n_merge (int): number of split & merge attempts
        rng: numpy.random.RandomState for deterministic behavior

    Notes:
        If frozen is a simple list, it will be assumed that is applies to mean
        and covariance of the specified components. It can also be a dictionary
        with the keys "mean" and "covar" to specify them separately.
        In either case, amplitudes will be updated to reflect any changes made.
        If frozen["amp"] is set, it will use this list instead.

    Returns:
        mean log-likelihood (float), component neighborhoods (list of ints)

    Throws:
        RuntimeError for inconsistent argument combinations
    """

    N = len(data)
    # if there are data (features) missing, i.e. masked as np.nan, set them to zeros
    # and create/set covariance elements to very large value to reduce its weight
    # to effectively zero
    missing = np.isnan(data)
    if missing.any():
        data_ = createShared(data.copy())
        data_[missing] = 0 # value does not matter as long as it's not nan
        if covar is None:
            covar = np.zeros((gmm.D, gmm.D))
            # need to create covar_callback if imputation is requested
            if sel_callback is not None:
                from functools import partial
                covar_callback = partial(covar_callback_default, default=np.zeros((gmm.D, gmm.D)))
        if covar.shape == (gmm.D, gmm.D):
            covar_ = createShared(np.tile(covar, (N,1,1)))
        else:
            covar_ = createShared(covar.copy())

        large = 1e10
        for d in range(gmm.D):
            covar_[missing[:,d],d,d] += large
            covar_[missing[:,d],d,d] += large
    else:
        data_ = createShared(data.copy())
        if covar is None or covar.shape == (gmm.D, gmm.D):
            covar_ = covar
        else:
            covar_ = createShared(covar.copy())

    # init components
    if init_method.lower() not in ['random', 'minmax', 'kmeans', 'none']:
        raise NotImplementedError("init_mehod %s not in ['random', 'minmax', 'kmeans', 'none']" % init_method)
    if init_method.lower() == 'random':
        initFromDataAtRandom(gmm, data_, covar=covar_, rng=rng)
    if init_method.lower() == 'minmax':
        initFromDataMinMax(gmm, data_, covar=covar_, rng=rng)
    if init_method.lower() == 'kmeans':
        initFromKMeans(gmm, data_, covar=covar_, rng=rng)

    # test if callbacks are consistent
    if sel_callback is not None and covar is not None and covar_callback is None:
        raise NotImplementedError("covar is set, but covar_callback is None: imputation samples inconsistent")

    # set up pool
    pool = multiprocessing.Pool()
    n_chunks, chunksize = gmm._mp_chunksize()

    # containers
    # precautions for cases when some points are treated as outliers
    # and not considered as belonging to any component
    log_S = createShared(np.zeros(N))          # S = sum_k p(x|k)
    log_p = [[] for k in xrange(gmm.K)]        # P = p(x|k) for x in U[k]
    T_inv = [None for k in xrange(gmm.K)]      # T = covar(x) + gmm.covar[k]
    U = [None for k in xrange(gmm.K)]          # U = {x close to k}
    p_bg = None
    if background is not None:
        gmm.amp *= 1 - background.amp          # GMM amp + BG amp = 1
        p_bg = [None]                          # p_bg = p(x|BG), no log because values are larger
        if covar is not None:
            # check if covar is diagonal and issue warning if not
            mess = "background model will only consider diagonal elements of covar"
            nondiag = ~np.eye(gmm.D, dtype='bool')
            if covar.shape == (gmm.D, gmm.D):
                if (covar[nondiag] != 0).any():
                    logger.warning(mess)
            else:
                if (covar[np.tile(nondiag,(N,1,1))] != 0).any():
                    logger.warning(mess)

    # check if all component parameters can be changed
    changeable = {"amp": slice(None), "mean": slice(None), "covar": slice(None)}
    if frozen is not None:
        if all(isinstance(item, int) for item in frozen):
            changeable['amp'] = changeable['mean'] = changeable['covar'] = np.in1d(xrange(gmm.K), frozen, assume_unique=True, invert=True)
        elif hasattr(frozen, 'keys') and np.in1d(["amp","mean","covar"], tuple(frozen.keys()), assume_unique=True).any():
            if "amp" in frozen.keys():
                changeable['amp'] = np.in1d(xrange(gmm.K), frozen['amp'], assume_unique=True, invert=True)
            if "mean" in frozen.keys():
                changeable['mean'] = np.in1d(xrange(gmm.K), frozen['mean'], assume_unique=True, invert=True)
            if "covar" in frozen.keys():
                changeable['covar'] = np.in1d(xrange(gmm.K), frozen['covar'], assume_unique=True, invert=True)
        else:
            raise NotImplementedError("frozen should be list of indices or dictionary with keys in ['amp','mean','covar']")

    try:
        log_L, N, N2 = _EM(gmm, log_p, U, T_inv, log_S, data_, covar=covar_, R=R, sel_callback=sel_callback, oversampling=oversampling, covar_callback=covar_callback, w=w, pool=pool, chunksize=chunksize, cutoff=cutoff, background=background, p_bg=p_bg, changeable=changeable, miniter=miniter, maxiter=maxiter, tol=tol, rng=rng)
    except Exception:
        # cleanup
        pool.close()
        pool.join()
        del data_, covar_, log_S
        raise

    # should we try to improve by split'n'merge of components?
    # if so, keep backup copy
    gmm_ = None
    if frozen is not None and split_n_merge:
        logger.warning("forgoing split'n'merge because some components are frozen")
    else:
        while split_n_merge and gmm.K >= 3:

            if gmm_ is None:
                gmm_ = GMM(gmm.K, gmm.D)

            gmm_.amp[:] = gmm.amp[:]
            gmm_.mean[:] = gmm.mean[:,:]
            gmm_.covar[:,:,:] = gmm.covar[:,:,:]
            U_ = [U[k].copy() for k in xrange(gmm.K)]

            changing, cleanup = _findSNMComponents(gmm, U, log_p, log_S, N+N2, pool=pool, chunksize=chunksize)
            logger.info("merging %d and %d, splitting %d" % tuple(changing))

            # modify components
            _update_snm(gmm, changing, U, N+N2, cleanup)

            # run partial EM on changeable components
            # NOTE: for a partial run, we'd only need the change to Log_S from the
            # changeable components. However, the neighborhoods can change from _update_snm
            # or because they move, so that operation is ill-defined.
            # Thus, we'll always run a full E-step, which is pretty cheap for
            # converged neighborhood.
            # The M-step could in principle be run on the changeable components only,
            # but there seem to be side effects in what I've tried.
            # Similar to the E-step, the imputation step needs to be run on all
            # components, otherwise the contribution of the changeable ones to the mixture
            # would be over-estimated.
            # Effectively, partial runs are as expensive as full runs.

            changeable['amp'] = changeable['mean'] = changeable['covar'] = np.in1d(xrange(gmm.K), changing, assume_unique=True)
            log_L_, N_, N2_ = _EM(gmm, log_p, U, T_inv, log_S, data_, covar=covar_, R=R,  sel_callback=sel_callback, oversampling=oversampling, covar_callback=covar_callback, w=w, pool=pool, chunksize=chunksize, cutoff=cutoff, background=background, p_bg=p_bg, maxiter=maxiter, tol=tol, prefix="SNM_P", changeable=changeable, rng=rng)

            changeable['amp'] = changeable['mean'] = changeable['covar'] = slice(None)
            log_L_, N_, N2_ = _EM(gmm, log_p, U, T_inv, log_S, data_, covar=covar_, R=R,  sel_callback=sel_callback, oversampling=oversampling, covar_callback=covar_callback, w=w, pool=pool, chunksize=chunksize, cutoff=cutoff, background=background, p_bg=p_bg, maxiter=maxiter, tol=tol, prefix="SNM_F", changeable=changeable, rng=rng)

            if log_L >= log_L_:
                # revert to backup
                gmm.amp[:] = gmm_.amp[:]
                gmm.mean[:] = gmm_.mean[:,:]
                gmm.covar[:,:,:] = gmm_.covar[:,:,:]
                U = U_
                logger.info ("split'n'merge likelihood decreased: reverting to previous model")
                break

            log_L = log_L_
            split_n_merge -= 1

    pool.close()
    pool.join()
    del data_, covar_, log_S
    return log_L, U

# run EM sequence
def _EM(gmm, log_p, U, T_inv, log_S, data, covar=None, R=None, sel_callback=None, oversampling=10, covar_callback=None, background=None, p_bg=None, w=0, pool=None, chunksize=1, cutoff=None, miniter=1, maxiter=1000, tol=1e-3, prefix="", changeable=None, rng=np.random):

    # compute effective cutoff for chi2 in D dimensions
    if cutoff is not None:
        # note: subsequently the cutoff parameter, e.g. in _E(), refers to this:
        # chi2 < cutoff,
        # while in fit() it means e.g. "cut at 3 sigma".
        # These differing conventions need to be documented well.
        cutoff_nd = chi2_cutoff(gmm.D, cutoff=cutoff)

        # store chi2 cutoff for component shifts, use 0.5 sigma
        shift_cutoff = chi2_cutoff(gmm.D, cutoff=min(0.1, cutoff/2))
    else:
        cutoff_nd = None
        shift_cutoff = chi2_cutoff(gmm.D, cutoff=0.1)

    if sel_callback is not None:
        omega = createShared(sel_callback(data).astype("float"))
        if np.any(omega == 0):
            logger.warning("Selection probability Omega = 0 for an observed sample.")
            logger.warning("Selection callback likely incorrect! Bad things will happen!")
    else:
        omega = None

    it = 0
    header = "ITER\tSAMPLES"
    if sel_callback is not None:
        header += "\tIMPUTED\tORIG"
    if background is not None:
        header += "\tBG_AMP"
    header += "\tLOG_L\tSTABLE"
    logger.info(header)

    # save backup
    gmm_ = GMM(gmm.K, gmm.D)
    gmm_.amp[:] = gmm.amp[:]
    gmm_.mean[:,:] = gmm.mean[:,:]
    gmm_.covar[:,:,:] = gmm.covar[:,:,:]
    N0 = len(data) # size of original (unobscured) data set (signal and background)
    N2 = 0         # size of imputed signal sample
    if background is not None:
        bg_amp_ = background.amp

    while it < maxiter: # limit loop in case of slow convergence
        log_L_, N, N2_, N0_ = _EMstep(gmm, log_p, U, T_inv, log_S, N0, data, covar=covar, R=R, sel_callback=sel_callback, omega=omega, oversampling=oversampling, covar_callback=covar_callback, background=background, p_bg=p_bg , w=w, pool=pool, chunksize=chunksize, cutoff=cutoff_nd, tol=tol, changeable=changeable, it=it, rng=rng)

        # check if component has moved by more than sigma/2
        shift2 = np.einsum('...i,...ij,...j', gmm.mean - gmm_.mean, np.linalg.inv(gmm_.covar), gmm.mean - gmm_.mean)
        moved = np.flatnonzero(shift2 > shift_cutoff)
        status_mess = "%s%d\t%d" % (prefix, it, N)
        if sel_callback is not None:
            status_mess += "\t%.2f\t%.2f" % (N2_, N0_)
        if background is not None:
            status_mess += "\t%.3f" % bg_amp_
        status_mess += "\t%.3f\t%d" % (log_L_, gmm.K - moved.size)
        logger.info(status_mess)

        # convergence tests
        if it > miniter:
            if sel_callback is None:
                if np.abs(log_L_ - log_L) < tol * np.abs(log_L) and moved.size == 0:
                    log_L = log_L_
                    logger.info("likelihood converged within relative tolerance %r: stopping here." % tol)
                    break
            else:
                if np.abs(N0_ - N0) < tol * N0 and np.abs(N2_ - N2) < tol * N2 and moved.size == 0:
                    log_L = log_L_
                    logger.info("imputation sample size converged within relative tolerance %r: stopping here." % tol)
                    break

        # force update to U for all moved components
        if cutoff is not None:
            for k in moved:
                U[k] = None

        if moved.size:
            logger.debug("resetting neighborhoods of moving components: (" + ("%d," * moved.size + ")") % tuple(moved))

        # update all important _ quantities for convergence test(s)
        log_L = log_L_
        N0 = N0_
        N2 = N2_

        # backup to see if components move or if next step gets worse
        # note: not gmm = gmm_ !
        gmm_.amp[:] = gmm.amp[:]
        gmm_.mean[:,:] = gmm.mean[:,:]
        gmm_.covar[:,:,:] = gmm.covar[:,:,:]
        if background is not None:
            bg_amp_ = background.amp

        it += 1

    return log_L, N, N2

# run one EM step
def _EMstep(gmm, log_p, U, T_inv, log_S, N0, data, covar=None, R=None, sel_callback=None, omega=None, oversampling=10, covar_callback=None, background=None, p_bg=None, w=0, pool=None, chunksize=1, cutoff=None, tol=1e-3, changeable=None, it=0, rng=np.random):

    # NOTE: T_inv (in fact (T_ik)^-1 for all samples i and components k)
    # is very large and is unfortunately duplicated in the parallelized _Mstep.
    # If memory is too limited, one can recompute T_inv in _Msums() instead.
    log_L = _Estep(gmm, log_p, U, T_inv, log_S, data, covar=covar, R=R, omega=omega, background=background, p_bg=p_bg, pool=pool, chunksize=chunksize, cutoff=cutoff, it=it)
    A,M,C,N,B = _Mstep(gmm, U, log_p, T_inv, log_S, data, covar=covar, R=R, p_bg=p_bg, pool=pool, chunksize=chunksize)

    A2 = M2 = C2 = B2 = N2 = 0

    # here the magic happens: imputation from the current model
    if sel_callback is not None:

        # if there are projections / missing data, we don't know how to
        # generate those for the imputation samples
        # NOTE: in principle, if there are only missing data, i.e. R is 1_D,
        # we could ignore missingness for data2 because we'll do an analytic
        # marginalization. This doesn't work if R is a non-trivial matrix.
        if R is not None:
            raise NotImplementedError("R is not None: imputation samples likely inconsistent")

        # create fake data with same mechanism as the original data,
        # but invert selection to get the missing part
        data2, covar2, N0, omega2 = draw(gmm, len(data)*oversampling, sel_callback=sel_callback, orig_size=N0*oversampling, invert_sel=True, covar_callback=covar_callback, background=background, rng=rng)
        data2 = createShared(data2)
        if not(covar2 is None or covar2.shape == (gmm.D, gmm.D)):
            covar2 = createShared(covar2)

        N0 = N0/oversampling
        U2 = [None for k in xrange(gmm.K)]

        if len(data2) > 0:
            log_S2 = np.zeros(len(data2))
            log_p2 = [[] for k in xrange(gmm.K)]
            T2_inv = [None for k in xrange(gmm.K)]
            R2 = None
            if background is not None:
                p_bg2 = [None]
            else:
                p_bg2 = None

            log_L2 = _Estep(gmm, log_p2, U2, T2_inv, log_S2, data2, covar=covar2, R=R2, omega=None, background=background, p_bg=p_bg2, pool=pool, chunksize=chunksize, cutoff=cutoff, it=it)
            A2,M2,C2,N2,B2 = _Mstep(gmm, U2, log_p2, T2_inv, log_S2, data2, covar=covar2, R=R2, p_bg=p_bg2, pool=pool, chunksize=chunksize)

            # normalize for oversampling
            A2 /= oversampling
            M2 /= oversampling
            C2 /= oversampling
            B2 /= oversampling
            N2 = N2/oversampling # need floating point precision in update

            # check if components have outside selection
            sel_outside = A2 > tol * A
            if sel_outside.any():
                logger.debug("component inside fractions: " + ("(" + "%.2f," * gmm.K + ")") % tuple(A/(A+A2)))

        # correct the observed likelihood for the overall normalization constant of
        # of the data process with selection:
        # logL(x | gmm) = sum_k p_k(x) / Z(gmm), with Z(gmm) = int dx sum_k p_k(x) = 1
        # becomes
        # logL(x | gmm) = sum_k Omega(x) p_k(x) / Z'(gmm),
        # with Z'(gmm) = int dx Omega(x) sum_k p_k(x), which we can gt by MC integration
        log_L -= N * np.log((omega.sum() + omega2.sum() / oversampling) / (N + N2))

    _update(gmm, A, M, C, N, B, A2, M2, C2, N2, B2, w, changeable=changeable, background=background)

    return log_L, N, N2, N0

# perform E step calculations.
# If cutoff is set, this will also set the neighborhoods U
def _Estep(gmm, log_p, U, T_inv, log_S, data, covar=None, R=None, omega=None, background=None, p_bg=None, pool=None, chunksize=1, cutoff=None, it=0, rng=np.random):
    # compute p(i | k) for each k independently in the pool
    # need S = sum_k p(i | k) for further calculation
    log_S[:] = 0

    # H = {i | i in neighborhood[k]} for any k, needed for outliers below
    # TODO: Use only when cutoff is set
    H = np.zeros(len(data), dtype="bool")

    k = 0
    for log_p[k], U[k], T_inv[k] in \
    parmap.starmap(_Esum, zip(xrange(gmm.K), U), gmm, data, covar, R, cutoff, pm_pool=pool, pm_chunksize=chunksize):
        log_S[U[k]] += np.exp(log_p[k]) # actually S, not logS
        H[U[k]] = 1
        k += 1

    if background is not None:
        p_bg[0] = background.amp * background.p
        if covar is not None:
            # This is the zeroth moment of a truncated Normal error distribution
            # Its calculation is simple only of the covariance is diagonal!
            # See e.g. Manjunath & Wilhem (2012) if not
            error = np.ones(len(data))
            x0,x1 = background.footprint
            for d in range(gmm.D):
                if covar.shape == (gmm.D, gmm.D): # one-for-all
                    denom = np.sqrt(2 * covar[d,d])
                else:
                    denom = np.sqrt(2 * covar[:,d,d])
                # CAUTION: The erf is approximate and returns 0
                # Thus, we don't add the logs but multiple the value itself
                # underrun is not a big problem here
                error *= np.real(scipy.special.erf((data[:,d] - x0[d])/denom)  - scipy.special.erf((data[:,d] - x1[d])/denom)) / 2
            p_bg[0] *= error
        log_S[:] = np.log(log_S + p_bg[0])
        if omega is not None:
            log_S += np.log(omega)
        log_L = log_S.sum()
    else:
        # need log(S), but since log(0) isn't a good idea, need to restrict to H
        log_S[H] = np.log(log_S[H])
        if omega is not None:
            log_S += np.log(omega)
        log_L = log_S[H].sum()

    return log_L

# compute chi^2, and apply selections on component neighborhood based in chi^2
def _Esum(k, U_k, gmm, data, covar=None, R=None, cutoff=None):
    # since U_k could be None, need explicit reshape
    d_ = data[U_k].reshape(-1, gmm.D)
    if covar is not None:
        if covar.shape == (gmm.D, gmm.D): # one-for-all
            covar_ = covar
        else: # each datum has covariance
            covar_ = covar[U_k].reshape(-1, gmm.D, gmm.D)
    else:
        covar_ = 0
    if R is not None:
        R_ = R[U_k].reshape(-1, gmm.D, gmm.D)

    # p(x | k) for all x in the vicinity of k
    # determine all points within cutoff sigma from mean[k]
    if R is None:
        dx = d_ - gmm.mean[k]
    else:
        dx = d_ - np.dot(R_, gmm.mean[k])

    if covar is None and R is None:
         T_inv_k = None
         chi2 = np.einsum('...i,...ij,...j', dx, np.linalg.inv(gmm.covar[k]), dx)
    else:
        # with data errors: need to create and return T_ik = covar_i + C_k
        # and weight each datum appropriately
        if R is None:
            T_inv_k = np.linalg.inv(gmm.covar[k] + covar_)
        else: # need to project out missing elements: T_ik = R_i C_k R_i^R + covar_i
            T_inv_k = np.linalg.inv(np.einsum('...ij,jk,...lk', R_, gmm.covar[k], R_) + covar_)
        chi2 = np.einsum('...i,...ij,...j', dx, T_inv_k, dx)

    # NOTE: close to convergence, we could stop applying the cutoff because
    # changes to U will be minimal
    if cutoff is not None:
        indices = chi2 < cutoff
        chi2 = chi2[indices]
        if (covar is not None and covar.shape != (gmm.D, gmm.D)) or R is not None:
            T_inv_k = T_inv_k[indices]
        if U_k is None:
            U_k = np.flatnonzero(indices)
        else:
            U_k = U_k[indices]

    # prevent tiny negative determinants to mess up
    if covar is None:
        (sign, logdet) = np.linalg.slogdet(gmm.covar[k])
    else:
        (sign, logdet) = np.linalg.slogdet(T_inv_k)
        sign *= -1 # since det(T^-1) = 1/det(T)

    log2piD2 = np.log(2*np.pi)*(0.5*gmm.D)
    return np.log(gmm.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2, U_k, T_inv_k

# get zeroth, first, second moments of the data weighted with p_k(x) avgd over x
def _Mstep(gmm, U, log_p, T_inv, log_S, data, covar=None, R=None, p_bg=None, pool=None, chunksize=1):

    # save the M sums from observed data
    A = np.empty(gmm.K)                 # sum for amplitudes
    M = np.empty((gmm.K, gmm.D))        # ... means
    C = np.empty((gmm.K, gmm.D, gmm.D)) # ... covariances
    N = len(data)

    # perform sums for M step in the pool
    # NOTE: in a partial run, could work on changeable components only;
    # however, there seem to be side effects or race conditions
    k = 0
    for A[k], M[k,:], C[k,:,:] in \
    parmap.starmap(_Msums, zip(xrange(gmm.K), U, log_p, T_inv), gmm, data, R, log_S, pm_pool=pool, pm_chunksize=chunksize):
        k += 1

    if p_bg is not None:
        q_bg = p_bg[0] / np.exp(log_S)
        B = q_bg.sum() # equivalent to A_k in _Msums, but done without logs
    else:
        B = 0

    return A,M,C,N,B

# compute moments for the Mstep
def _Msums(k, U_k, log_p_k, T_inv_k, gmm, data, R, log_S):
    if log_p_k.size == 0:
        return 0,0,0

    # get log_q_ik by dividing with S = sum_k p_ik
    # NOTE:  this modifies log_p_k in place, but is only relevant
    # within this method since the call is parallel and its arguments
    # therefore don't get updated across components.

    # NOTE: reshape needed when U_k is None because of its
    # implicit meaning as np.newaxis
    log_p_k -= log_S[U_k].reshape(log_p_k.size)
    d = data[U_k].reshape((log_p_k.size, gmm.D))
    if R is not None:
        R_ = R[U_k].reshape((log_p_k.size, gmm.D, gmm.D))

    # amplitude: A_k = sum_i q_ik
    A_k = np.exp(logsum(log_p_k))

    # in fact: q_ik, but we treat sample index i silently everywhere
    q_k = np.exp(log_p_k)

    if R is None:
        d_m = d - gmm.mean[k]
    else:
        d_m = d - np.dot(R_, gmm.mean[k])

    # data with errors?
    if T_inv_k is None and R is None:
        # mean: M_k = sum_i x_i q_ik
        M_k = (d * q_k[:,None]).sum(axis=0)

        # covariance: C_k = sum_i (x_i - mu_k)^T(x_i - mu_k) q_ik
        # funny way of saying: for each point i, do the outer product
        # of d_m with its transpose, multiply with pi[i], and sum over i
        C_k = (q_k[:, None, None] * d_m[:, :, None] * d_m[:, None, :]).sum(axis=0)
    else:
        if R is None: # that means T_ik is not None
            # b_ik = mu_k + C_k T_ik^-1 (x_i - mu_k)
            # B_ik = C_k - C_k T_ik^-1 C_k
            b_k = gmm.mean[k] + np.einsum('ij,...jk,...k', gmm.covar[k], T_inv_k, d_m)
            B_k = gmm.covar[k] - np.einsum('ij,...jk,...kl', gmm.covar[k], T_inv_k, gmm.covar[k])
        else:
            # F_ik = C_k R_i^T T_ik^-1
            F_k = np.einsum('ij,...kj,...kl', gmm.covar[k], R_, T_inv_k)
            b_k = gmm.mean[k] + np.einsum('...ij,...j', F_k, d_m)
            B_k = gmm.covar[k] - np.einsum('...ij,...jk,kl', F_k, R_, gmm.covar[k])

            #b_k = gmm.mean[k] + np.einsum('ij,...jk,...k', gmm.covar[k], T_inv_k, d_m)
            #B_k = gmm.covar[k] - np.einsum('ij,...jk,...kl', gmm.covar[k], T_inv_k, gmm.covar[k])
        M_k = (b_k * q_k[:,None]).sum(axis=0)
        b_k -= gmm.mean[k]
        C_k = (q_k[:, None, None] * (b_k[:, :, None] * b_k[:, None, :] + B_k)).sum(axis=0)
    return A_k, M_k, C_k


# update component with the moment matrices.
# If changeable is set, update only those components and renormalize the amplitudes
def _update(gmm, A, M, C, N, B, A2, M2, C2, N2, B2, w, changeable=None, background=None):

    # recompute background amplitude
    if background is not None and background.adjust_amp:
        background.amp = max(min((B + B2) / (N + N2), background.amp_max), background.amp_min)

    # amp update:
    # for partial update: need to update amp for any component that is changeable
    if not hasattr(changeable['amp'], '__iter__'): # it's a slice(None), not a bool array
        gmm.amp[changeable['amp']] = (A + A2)[changeable['amp']] / (N + N2)
    else:
        # Bovy eq. 31, with correction for bg.amp if needed
        if background is None:
            total = 1
        else:
            total = 1 - background.amp
        gmm.amp[changeable['amp']] = (A + A2)[changeable['amp']] / (A + A2)[changeable['amp']].sum() * (total - (gmm.amp[~changeable['amp']]).sum())

    # mean updateL
    gmm.mean[changeable['mean'],:] = (M + M2)[changeable['mean'],:]/(A + A2)[changeable['mean'],None]

    # covar updateL
    # minimum covariance term?
    if w > 0:
        # we assume w to be a lower bound of the isotropic dispersion,
        # C_k = w^2 I + ...
        # then eq. 38 in Bovy et al. only ~works for N = 0 because of the
        # prefactor 1 / (q_j + 1) = 1 / (A + 1) in our terminology
        # On average, q_j = N/K, so we'll adopt that to correct.
        w_eff = w**2 * ((N+N2)/gmm.K + 1)
        gmm.covar[changeable['covar'],:,:] = (C + C2 + w_eff*np.eye(gmm.D)[None,:,:])[changeable['covar'],:,:] / (A + A2 + 1)[changeable['covar'],None,None]
    else:
        gmm.covar[changeable['covar'],:,:] = (C + C2)[changeable['covar'],:,:] / (A + A2)[changeable['covar'],None,None]

# draw from the model (+ background) and apply appropriate covariances
def _drawGMM_BG(gmm, size, covar_callback=None, background=None, rng=np.random):
    # draw sample from model, or from background+model
    if background is None:
        data2 = gmm.draw(int(np.round(size)), rng=rng)
    else:
        # model is GMM + Background
        bg_size = int(background.amp * size)
        data2 = np.concatenate((gmm.draw(int(np.round(size-bg_size)), rng=rng), background.draw(int(np.round(bg_size)), rng=rng)))

    # add noise
    # NOTE: When background is set, adding noise is problematic if
    # scattering them out is more likely than in.
    # This can be avoided when the background footprint is large compared to
    # selection region
    if covar_callback is not None:
        covar2 = covar_callback(data2)
        if covar2.shape == (gmm.D, gmm.D): # one-for-all
            noise = rng.multivariate_normal(np.zeros(gmm.D), covar2, size=len(data2))
        else:
            # create noise from unit covariance and then dot with eigenvalue
            # decomposition of covar2 to get a the right noise distribution:
            # n' = R V^1/2 n, where covar = R V R^-1
            # faster than drawing one sample per each covariance
            noise = rng.multivariate_normal(np.zeros(gmm.D), np.eye(gmm.D), size=len(data2))
            val, rot = np.linalg.eigh(covar2)
            val = np.maximum(val,0) # to prevent univariate errors to underflow
            noise = np.einsum('...ij,...j', rot, np.sqrt(val)*noise)
        data2 += noise
    else:
        covar2 = None
    return data2, covar2


def draw(gmm, obs_size, sel_callback=None, invert_sel=False, orig_size=None, covar_callback=None, background=None, rng=np.random):
    """Draw from the GMM (and the Background) with noise and selection.

    Draws orig_size samples from the GMM and the Background, if set; calls
    covar_callback if set and applies resulting covariances; the calls
    sel_callback on the (noisy) samples and returns those matching ones.

    If the number is resulting samples is inconsistent with obs_size, i.e.
    outside of the 68 percent confidence limit of a Poisson draw, it will
    update its estimate for the original sample size orig_size.
    An estimate can be provided with orig_size, otherwise it will use obs_size.

    Note:
        If sel_callback is set, the number of returned samples is not
        necessarily given by obs_size.

    Args:
        gmm: an instance if GMM
        obs_size (int): number of observed samples
        sel_callback: completeness callback to generate imputation samples.
        invert_sel (bool): whether to invert the result of sel_callback
        orig_size (int): an estimate of the original size of the sample.
        background: an instance of Background
        covar_callback: covariance callback for imputation samples.
        rng: numpy.random.RandomState for deterministic behavior

    Returns:
        sample: nunmpy array (N_orig, D)
        covar_sample: numpy array (N_orig, D, D) or None of covar_callback=None
        N_orig (int): updated estimate of orig_size if sel_callback is set

    Throws:
        RuntimeError for inconsistent argument combinations
    """

    if orig_size is None:
        orig_size = int(obs_size)

    # draw from model (with background) and add noise.
    # TODO: may want to decide whether to add noise before selection or after
    # Here we do noise, then selection, but this is not fundamental
    data, covar = _drawGMM_BG(gmm, orig_size, covar_callback=covar_callback, background=background, rng=rng)

    # apply selection
    if sel_callback is not None:
        omega = sel_callback(data)
        sel = rng.rand(len(data)) < omega

        # check if predicted observed size is consistent with observed data
        # 68% confidence interval for Poisson variate: observed size
        alpha = 0.32
        lower = 0.5*scipy.stats.chi2.ppf(alpha/2, 2*obs_size)
        upper = 0.5*scipy.stats.chi2.ppf(1 - alpha/2, 2*obs_size + 2)
        obs_size_ = sel.sum()
        while obs_size_ > upper or obs_size_ < lower:
            orig_size = int(orig_size / obs_size_ * obs_size)
            data, covar = _drawGMM_BG(gmm, orig_size, covar_callback=covar_callback, background=background, rng=rng)
            omega = sel_callback(data)
            sel = rng.rand(len(data)) < omega
            obs_size_ = sel.sum()

        if invert_sel:
            sel = ~sel
        data = data[sel]
        omega = omega[sel]
        if covar_callback is not None and covar.shape != (gmm.D, gmm.D):
            covar = covar[sel]

    return data, covar, orig_size, omega


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
        logger.debug("neighborhoods disjunct. merging components %d and %d" % tuple(merge_jk))
        merge_jk = np.argsort(gmm.amp)[:2]
        cleanup = True


    # split the one whose p(x|k) deviate most from current Gaussian
    # ask for the three worst components to avoid split being in merge_jk
    """
    JS = np.empty(gmm.K)
    k = 0
    A = gmm.amp * N
    for JS[k] in \
    parmap.map(_JS, xrange(gmm.K), gmm, log_p, log_S, U, A, pm_pool=pool, pm_chunksize=chunksize):
        k += 1
    """
    # get largest Eigenvalue, weighed by amplitude
    # Large EV implies extended object, which often is caused by coverving
    # multiple clusters. This happes also for almost empty components, which
    # should rather be merged than split, hence amplitude weights.
    # TODO: replace with linalg.eigvalsh, but eigenvalues are not always ordered
    EV = np.linalg.svd(gmm.covar, compute_uv=False)
    JS = EV[:,0] * gmm.amp
    split_l3 = np.argsort(JS)[-3:][::-1]

    # check that the three indices are unique
    changing = np.array([merge_jk[0], merge_jk[1], split_l3[0]])
    if split_l3[0] in merge_jk:
        if split_l3[1] not in merge_jk:
            changing[2] = split_l3[1]
        else:
            changing[2] = split_l3[2]
    return changing, cleanup


def _update_snm(gmm, changeable, U, N, cleanup):
    # reconstruct A from gmm.amp
    A = gmm.amp * N

    # update parameters and U
    # merge 0 and 1, store in 0, Bovy eq. 39
    gmm.amp[changeable[0]] = gmm.amp[changeable[0:2]].sum()
    if not cleanup:
        gmm.mean[changeable[0]] = np.sum(gmm.mean[changeable[0:2]] * A[changeable[0:2]][:,None], axis=0) / A[changeable[0:2]].sum()
        gmm.covar[changeable[0]] = np.sum(gmm.covar[changeable[0:2]] * A[changeable[0:2]][:,None,None], axis=0) / A[changeable[0:2]].sum()
        U[changeable[0]] = np.union1d(U[changeable[0]], U[changeable[1]])
    else:
        # if we're cleaning up the weakest components:
        # merging does not lead to valid component parameters as the original
        # ones can be anywhere. Simply adopt second one.
        gmm.mean[changeable[0],:] = gmm.mean[changeable[1],:]
        gmm.covar[changeable[0],:,:] = gmm.covar[changeable[1],:,:]
        U[changeable[0]] = U[changeable[1]]

    # split 2, store in 1 and 2
    # following SVD method in Zhang 2003, with alpha=1/2, u = 1/4
    gmm.amp[changeable[1]] = gmm.amp[changeable[2]] = gmm.amp[changeable[2]] / 2
    # TODO: replace with linalg.eigvalsh, but eigenvalues are not always ordered
    _, radius2, rotation = np.linalg.svd(gmm.covar[changeable[2]])
    dl = np.sqrt(radius2[0]) *  rotation[0] / 4
    gmm.mean[changeable[1]] = gmm.mean[changeable[2]] - dl
    gmm.mean[changeable[2]] = gmm.mean[changeable[2]] + dl
    gmm.covar[changeable[1:]] = np.linalg.det(gmm.covar[changeable[2]])**(1/gmm.D) * np.eye(gmm.D)
    U[changeable[1]] = U[changeable[2]].copy() # now 1 and 2 have same U


# L-fold cross-validation of the fit function.
# all parameters for fit must be supplied with kwargs.
# the rng seed will be fixed for the CV runs so that all random effects are the
# same for each run.
def cv_fit(gmm, data, L=10, **kwargs):
    N = len(data)
    lcv = np.empty(N)
    logger.info("running %d-fold cross-validation ..." % L)

    # CV and stacking can't have probabilistic inits that depends on
    # data or subsets thereof
    init_callback = kwargs.get("init_callback", None)
    if init_callback is not None:
        raise RuntimeError("Cross-validation can only be used consistently with init_callback=None")

    # make sure we know what the RNG is,
    # fix state of RNG to make behavior of fit reproducable
    rng = kwargs.get("rng", np.random)
    rng_state = rng.get_state()

    # need to copy the gmm when init_cb is None
    # otherwise runs start from different init positions
    gmm0 = GMM(K=gmm.K, D=gmm.D)
    gmm0.amp[:,] = gmm.amp[:]
    gmm0.mean[:,:] = gmm.mean[:,:]
    gmm0.covar[:,:,:] = gmm.covar[:,:,:]

    # same for bg if present
    bg = kwargs.get("background", None)
    if bg is not None:
        bg_amp0 = bg.amp

    # to L-fold CV here, need to split covar too if set
    covar = kwargs.pop("covar", None)
    for i in xrange(L):
        rng.set_state(rng_state)
        mask = np.arange(N) % L == i
        if covar is None or covar.shape == (gmm.D, gmm.D):
            fit(gmm, data[~mask], covar=covar, **kwargs)
            lcv[mask] = gmm.logL(data[mask], covar=covar)
        else:
            fit(gmm, data[~mask], covar=covar[~mask], **kwargs)
            lcv[mask] = gmm.logL(data[mask], covar=covar[mask])

        # undo for consistency
        gmm.amp[:,] = gmm0.amp[:]
        gmm.mean[:,:] = gmm0.mean[:,:]
        gmm.covar[:,:,:] = gmm0.covar[:,:,:]
        if bg is not None:
            bg.amp = bg_amp0

    return lcv


def stack(gmms, weights):
    # build stacked model by combining all gmms and applying weights to amps
    stacked = GMM(K=0, D=gmms[0].D)
    for m in xrange(len(gmms)):
        stacked.amp = np.concatenate((stacked.amp[:], weights[m]*gmms[m].amp[:]))
        stacked.mean = np.concatenate((stacked.mean[:,:], gmms[m].mean[:,:]))
        stacked.covar = np.concatenate((stacked.covar[:,:,:], gmms[m].covar[:,:,:]))
    stacked.amp /= stacked.amp.sum()
    return stacked


def stack_fit(gmms, data, kwargs, L=10, tol=1e-5, rng=np.random):
    M = len(gmms)
    N = len(data)
    lcvs = np.empty((M,N))

    for m in xrange(M):
        # run CV to get cross-validation likelihood
        rng_state = rng.get_state()
        lcvs[m,:] = cv_fit(gmms[m], data, L=L, **(kwargs[m]))
        rng.set_state(rng_state)
        # run normal fit on all data
        fit(gmms[m], data, **(kwargs[m]))

    # determine the weights that maximize the stacked estimator likelihood
    # run a tiny EM on lcvs to get them
    beta = np.ones(M)/M
    log_p_k = np.empty_like(lcvs)
    log_S = np.empty(N)
    it = 0
    logger.info("optimizing stacking weights\n")
    logger.info("ITER\tLOG_L")

    while True and it < 20:
        log_p_k[:,:] = lcvs + np.log(beta)[:,None]
        log_S[:] = logsum(log_p_k)
        log_p_k[:,:] -= log_S
        beta[:] = np.exp(logsum(log_p_k, axis=1)) / N
        logL_ = log_S.mean()
        logger.info("STACK%d\t%.4f" % (it, logL_))

        if it > 0 and logL_ - logL < tol:
            break
        logL = logL_
        it += 1
    return stack(gmms, beta)
