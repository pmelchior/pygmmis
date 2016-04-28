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
    def __init__(self, K=1, D=1, verbose=False):
        self.verbose = verbose
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

    def setRelevantComponents(self, relevant=None):
        # restore from backup copy if it exits
        try:
            self.amp = self._amp_cp
            self.mean = self._mean_cp
            self.covar = self._covar_cp
        except AttributeError:
            pass
        if relevant is not None:
            # copy all coeffs to backup and only show relevant ones to the outside
            self._amp_cp = self.amp.copy()
            self._mean_cp = self.mean.copy()
            self._covar_cp = self.covar.copy()

            self.amp = self.amp[relevant]
            self.amp /= self.amp.sum()
            self.mean = self.mean[relevant]
            self.covar = self.covar[relevant]

    def findRelevantComponents(self, coords, covar=None, method="chi2", cutoff=3):
        if method.upper() == "CHI2":
            # uses all components that have at least one point in data within
            # chi2 cutoff.
            import multiprocessing
            import parmap
            chunksize = int(np.ceil(self.K*1./multiprocessing.cpu_count()))
            k = 0
            relevant = set()
            for has_relevant_points in parmap.map(self._pointsAboveChi2Cutoff, xrange(self.K), coords, covar, cutoff, chunksize=chunksize):
                if has_relevant_points:
                    relevant.add(k)
                k += 1
            return list(relevant)

        if method.upper() == "RADIUS":
            # search coords for neighbors around each compoenent within
            # cutoff radius
            from sklearn.neighbors import KDTree
            tree = KDTree(coords)
            relevant_points = tree.query_radius(self.mean, r=cutoff, count_only=True)
            return np.nonzero(relevant_points > 0)[0]

        raise NotImplementedError("GMM.findRelevantComponents: method '%s' not implemented!" % method)


    def _pointsAboveChi2Cutoff(self, k, coords, covar=None, cutoff=3):
        # helper function to reduce memory requirement of findRelevantComponents():
        # avoids return of entire chi2 vector per component
        return (self.logL_k(k, coords, covar=covar, chi2_only=True) < cutoff*cutoff*self.D).sum()

    def draw(self, size=1, sel_callback=None, invert_callback=False, rng=np.random):
        # draw indices for components given amplitudes
        ind = rng.choice(self.K, size=size, p=self.amp)
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
        cpu_count = multiprocessing.cpu_count()
        chunksize = int(np.ceil(self.K*1./cpu_count))
        chunks = [(i*chunksize, min(self.K, (i+1)*chunksize)) for i in xrange(min(self.K, cpu_count))]
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

def initializeFromDataMinMax(gmm, K, data=None, covar=None, s=None, rng=np.random):
    gmm.amp[:] = np.ones(K)/K # now gmm.K works
    # set model to random positions with equally sized spheres within
    # volumne spanned by data
    min_pos = data.min(axis=0)
    max_pos = data.max(axis=0)
    gmm.mean[:,:] = min_pos + (max_pos-min_pos)*rng.rand(gmm.K, gmm.D)
    # if s is not set: use volume filling argument:
    # K spheres of radius s [having volume s^D * pi^D/2 / gamma(D/2+1)]
    # should completely fill the volume spanned by data.
    if s is None:
        from scipy.special import gamma
        vol_data = np.prod(max_pos-min_pos)
        s = (vol_data / gmm.K * gamma(gmm.D*0.5 + 1))**(1./gmm.D) / np.sqrt(np.pi)
        if gmm.verbose >= 2:
            print "initializing spheres with s=%.2f in data domain" % s
    gmm.covar[:,:,:] = np.tile(s**2 * np.eye(data.shape[1]), (gmm.K,1,1))

def initializeFromDataAtRandom(gmm, K, data=None, covar=None, s=None, rng=np.random):
    gmm.amp[:] = np.ones(K)/K
    # initialize components around data points with uncertainty s
    refs = rng.randint(0, len(data), size=K)
    if s is None:
        from scipy.special import gamma
        min_pos = data.min(axis=0)
        max_pos = data.max(axis=0)
        vol_data = np.prod(max_pos-min_pos)
        s = (vol_data / gmm.K * gamma(gmm.D*0.5 + 1))**(1./gmm.D) / np.sqrt(np.pi)
        if gmm.verbose >= 2:
            print "initializing spheres with s=%.2f near data points" % s
    gmm.mean[:,:] = data[refs] + rng.normal(0, s, size=(gmm.K,data.shape[1]))
    gmm.covar[:,:,:] = np.tile(s**2 * np.eye(data.shape[1]), (gmm.K,1,1))


def fit(data, covar=None, K=1, w=0., cutoff=None, sel_callback=None, N_missing=None, init_callback=initializeFromDataMinMax, tol=1e-3, verbose=False):
    gmm = GMM(K=K, D=data.shape[1], verbose=verbose)

    if sel_callback is None:
        # init components
        init_callback(gmm, K, data, covar)
    else:
        if covar is None:
            # run default EM first
            gmm = fit(data, covar=None, K=K, w=w, cutoff=cutoff, sel_callback=None, init_callback=init_callback, tol=tol, verbose=verbose)
            # inflate covar to accommodate changes from selection
            gmm.covar *= 4
        else:
            gmm = fit(data, covar=None, K=K, w=w, cutoff=cutoff, sel_callback=sel_callback, init_callback=init_callback, tol=tol, verbose=verbose)

    # set up pool
    import multiprocessing
    import parmap
    pool = multiprocessing.Pool()
    chunksize = int(np.ceil(gmm.K*1./multiprocessing.cpu_count()))

    # sum_k p(x|k) -> S
    # extra precautions for cases when some points are treated as outliers
    # and not considered as belonging to any component
    S = np.zeros(len(data)) # S = sum_k p(x|k)
    log_S = np.empty(len(data))
    N = np.zeros(len(data), dtype='bool') # N == 1 for points in the fit
    neighborhood = [None for k in xrange(gmm.K)]
    log_p = [[] for k in xrange(gmm.K)]
    T_inv = [None for k in xrange(gmm.K)]

    # save volumes to see which components change
    V = np.linalg.det(gmm.covar)

    # save the M sums from observed data
    A = np.empty(gmm.K)
    M = np.empty((gmm.K, gmm.D))
    C = np.empty((gmm.K, gmm.D, gmm.D))

    # moments of components under selection; need to be global for final update
    M0 = M1 = M2 = None
    if sel_callback is not None:
        M0 = np.empty(gmm.K)
        M1 = np.empty((gmm.K, gmm.D))
        M2 = np.empty((gmm.K, gmm.D, gmm.D))

    # begin EM
    it = 0
    maxiter = max(100, gmm.K)
    while it < maxiter: # limit loop in case of no convergence

        # compute p(i | k) for each k independently in the pool
        # need S = sum_k p(i | k) for further calculation
        # also N = {i | i in neighborhood[k]} for any k
        k = 0
        for log_p[k], neighborhood[k], T_inv[k] in \
        parmap.starmap(_E, zip(xrange(gmm.K), neighborhood), gmm, data, covar, cutoff, pool=pool, chunksize=chunksize):
            S[neighborhood[k]] += np.exp(log_p[k])
            N[neighborhood[k]] = 1
            k += 1

        # since log(0) isn't a good idea, need to restrict to N
        log_S[N] = np.log(S[N])
        log_L_ = log_S[N].mean()
        N_ = N.sum()

        if gmm.verbose:
            print ("%d\t%d\t%.4f" % (it, N_, log_L_)),
            if sel_callback is None:
                print ""

        # perform sums for M step in the pool
        k = 0
        for A[k], M[k], C[k] in \
        parmap.starmap(_computeMSums, zip(xrange(gmm.K), neighborhood, log_p, T_inv), gmm, data, log_S, pool=pool, chunksize=chunksize):
            k += 1

        # need to do MC integral of p(missing | k):
        # get missing data by imputation from the current model
        if sel_callback is not None:
            # with imputation the observed data logL can decrease.
            # revert to previous model if that is the case
            if it > 0 and log_L_ < log_L:
                if gmm.verbose:
                    print "\nmean likelihood decreased: stopping reverting to previous model."
                gmm = gmm_
                break

            k = 0
            for M0[k], M1[k], M2[k] in \
            parmap.map(_computeMoments, xrange(gmm.K), gmm, sel_callback, pool=pool, chunksize=chunksize):
                k += 1

            if gmm.verbose:
                if verbose >= 2:
                    print ("\t%.2f" * gmm.K) % tuple(M0)
                else:
                    print (M0 < 1).sum()

        # perform M step with M-sums of data and imputations runs
        _M(gmm, A, M, C, N_, w, M0, M1, M2)

        # convergence test:
        if it > 0 and log_L_ - log_L < tol:
            if gmm.verbose >= 2:
                print "mean likelihood converged within tolerance %r: stopping here." % tol
            break

        # update all important _ quantities for convergence test(s)
        log_L = log_L_
        if sel_callback is not None:
            gmm_ = gmm # backup if next step gets worse (note: not gmm = gmm_!)

        # check new component volumes and reset sel when it grows by
        # more then 25%
        V_ = np.linalg.det(gmm.covar)
        changed = np.flatnonzero((V_- V)/V > 0.25)
        for c in changed:
            neighborhood[c] = None
            V[c] = V_[c]
        if gmm.verbose >= 2 and changed.any():
            print "resetting neighborhoods due to volume change: ",
            print ("(" + "%d," * len(changed) + ")") % tuple(changed)
        S[:] = 0
        N[:] = 0
        it += 1

    if sel_callback is not None:
        # update component amplitudes for the missing fraction
        gmm.amp[:] /= M0[:] / M0.sum()
        gmm.amp /= gmm.amp.sum()

    pool.close()
    return gmm

def _E(k, neighborhood_k, gmm, data, covar=None, cutoff=None):
    # p(x | k) for all x in the vicinity of k
    # determine all points within cutoff sigma from mean[k]
    if cutoff is None or neighborhood_k is None:
        dx = data - gmm.mean[k]
    else:
        dx = data[neighborhood_k] - gmm.mean[k]

    if covar is None:
         T_inv_k = None
         chi2 = np.einsum('...i,...ij,...j', dx, np.linalg.inv(gmm.covar[k]), dx)
    else:
        # with data errors: need to create and return T_ik = covar_i + C_k
        # and weight each datum appropriately
        T_inv_k = np.linalg.inv(gmm.covar[k] + covar[neighborhood_k].reshape(len(dx), gmm.D, gmm.D))
        chi2 = np.einsum('...i,...ij,...j', dx, T_inv_k, dx)

    # NOTE: close to convergence, we could stop applying the cutoff because
    # changes to neighborhood will be minimal
    if cutoff is not None:
        indices = np.flatnonzero(chi2 < cutoff*cutoff*gmm.D)
        chi2 = chi2[indices]
        if covar is not None:
            T_inv_k = T_inv_k[indices]
        if neighborhood_k is None:
            neighborhood_k = indices
        else:
            neighborhood_k = neighborhood_k[indices]

    # prevent tiny negative determinants to mess up
    (sign, logdet) = np.linalg.slogdet(gmm.covar[k])

    log2piD2 = np.log(2*np.pi)*(0.5*gmm.D)
    return np.log(gmm.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2, neighborhood_k, T_inv_k

def _M(gmm, A, M, C, n_points, w=0., M0=None, M1=None, M2=None):
    # compute the new amplitude from the observed points only!
    # adjustment for missing points at the very end
    gmm.amp[:] = A / n_points

    # minimum covariance term?
    if w > 0:
        # we assume w to be a lower bound of the isotropic dispersion,
        # C_k = w^2 I + ...
        # then eq. 38 in Bovy et al. only ~works for N = 0 because of the
        # prefactor 1 / (q_j + 1) = 1 / (A + 1) in our terminology
        # On average, q_j = N/K, so we'll adopt that to correct.
        w_eff = w**2 * (n_points*1./gmm.K + 1)
        C_ = (C + w_eff*np.eye(gmm.D)[None,:,:]) / (A + 1)[:,None,None]
    else:
        C_ = C / A[:,None,None]

    if M0 is None:
        gmm.mean[:,:] = M / A[:,None]
        gmm.covar[:,:,:] = C_
    else:
        # only update components for which the selection corrections are
        # reasonably well determined.
        good = M0 > 0.1
        # also check of covariance remains positive definite:
        # instead of eigenvalues, use cholesky:
        # http://stackoverflow.com/questions/16266720/
        try:
            np.linalg.cholesky(C_ + gmm.covar - M2)
        except np.linalg.LinAlgError:
            for k in xrange(gmm.K):
                try:
                    np.linalg.cholesky(C_[k] + gmm.covar[k] - M2[k])
                except np.linalg.LinAlgError:
                    good[k] = 0

        if good.all():
            gmm.mean[:,:] = M / A[:,None] + gmm.mean - M1
            gmm.covar[:,:,:] = C_ + gmm.covar - M2
        else:
            gmm.mean[good,:] = M[good,:] / A[good,None] + gmm.mean[good,:] - M1[good,:]
            gmm.covar[good,:,:] = C_[good,:,:] + gmm.covar[good,:,:] - M2[good,:,:]
            # freeze components when the move off too much
            gmm.mean[~good,:] = gmm.mean[~good,:]
            gmm.covar[~good,:,:] = gmm.covar[~good,:,:]

def _computeMSums(k, neighborhood_k, log_p_k, T_inv_k, gmm, data, log_S):
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

def _draw_select(mean, covar, N, sel_callback=None, invert_callback=False, return_sel=False):
    # simple helper function to draw samples from a single component
    # potentially with a selection function
    data = np.random.multivariate_normal(mean, covar, size=N)
    if sel_callback is not None:
        sel = sel_callback(data)
        if invert_callback:
            sel = np.invert(sel)
        if return_sel:
            return data, sel
        return data[sel]
    return data

def _sampleMoments(k, gmm, d, tol, N, Nmax, sel_callback=None, invert_callback=False):
    while True:
        M0 = len(d)
        M1 = d.sum(axis=0) / M0
        varM1 = ((d - M1)**2).sum(axis=0) / (M0-1)
        if (varM1 / M0 < (M1 * tol)**2).all() or N >= Nmax:
            break
        # sample size to achieve tol
        M0_ = int((varM1 / (M1*tol)**2).max())
        # extra points (beyond M0) to achive, correct for selection loss
        # make sure to at least double N (for the effort), but to stay within Nmax
        extra = min(Nmax, max(N, int((M0_ - M0) / (M0 * 1./ N))))
        d = np.concatenate((d, _draw_select(gmm.mean[k], gmm.covar[k], extra, sel_callback, invert_callback)))
        N += extra

    # covariance: sum_i (x_i - mu_k)^T(x_i - mu_k)
    # Note: error on covariance larger than tol, but correction of means
    # more important, so we'll ignore that
    d_m = d - gmm.mean[k]
    M2 = (d_m[:, :, None] * d_m[:, None, :]).sum(axis=0) / M0
    M0 = M0 * 1./N
    return N, M0, M1, M2

def _computeMoments(k, gmm, sel_callback=None, tol=1e-2):
    N = int(10/tol)
    Nmax = 100000
    d, sel = _draw_select(gmm.mean[k], gmm.covar[k], N, sel_callback=sel_callback, return_sel=True)
    N_ = sel.sum()
    # selection affects component
    if N_ < N * (1 - tol):

        # component most inside: use inside draws
        if N_ > N / 2:
            N, M0, M1, M2 = _sampleMoments(k, gmm, d[sel], tol, N, Nmax, sel_callback=sel_callback)

        # predict inside moments from outside draws
        else:
            N, M0_, M1_, M2_ = _sampleMoments(k, gmm, d[~sel], tol, N, Nmax, sel_callback=sel_callback, invert_callback=True)
            M0 = max(tol, 1 - M0_) # prevent division by zero
            M1 = (N*gmm.mean[k] - M1_* M0_ * N) / M0 / N
            M2 = (N*gmm.covar[k] - M2_* M0_ * N) / M0 / N
    else:
        M0 = 1
        M1 = gmm.mean[k]
        M2 = gmm.covar[k]
    return M0, M1, M2
