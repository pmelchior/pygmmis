import numpy as np
import multiprocessing
import ctypes

# for multiprocessing: use shared arrays to avoid copies for each thread
# http://stackoverflow.com/questions/5549190/
def createShared(a, dtype=ctypes.c_double):
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

class GMM(object):
    def __init__(self, K=1, D=1, verbose=False):
        # if rng is None:
        #     self.rng = np.random
        # else:
        #     self.rng = rng
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
        if kwargs is None:
            kwargs = {"amp": self.amp, "mean": self.mean, "covar": self.covar}
        else:
            kwargs['amp'] = self.amp
            kwargs['mean'] = self.mean
            kwargs['covar'] = self.covar
        np.savez(filename, **kwargs)

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

    def logL(self, data, covar=None):
        """Log-likelihood of data given all (i.e. the sum of) GMM components

        See appendix A of Bovy, Hogg, Roweis (2009).

        If covar is set, the data error will be incorporated, eq. 9, resulting
        in log(sum_k(p(y | k))). Otherwise this method return log(sum_k(p(x | k)))
        of the truth values.

        Args:
            data:   (D,) or (N, D) test coordinates
            covar:  (D, D) or (N, D, D) covariance matrix of data

        Returns:
            (1,) or (N, 1) log(L), depending on shape of data
        """

        log_p = np.empty((self.K, len(data)))
        for k in xrange(self.K):
            log_p[k,:] = self.logL_k(k, data, covar=covar)
        return self.logsumLogX(log_p) # sum over all k

    def logL_k(self, k, data, covar=None):
        # compute p(x | k)
        log2piD2 = np.log(2*np.pi)*(0.5*self.D)
        dx = data - self.mean[k]
        if covar is None:
            chi2 = np.einsum('...j,j...', dx, np.dot(np.linalg.inv(self.covar[k]), dx.T))
            # prevent tiny negative determinants to mess up
            (sign, logdet) = np.linalg.slogdet(self.covar[k])
        else:
            T_k = self.covar[k] + covar
            chi2 = np.einsum('...i,...ij,...j', dx, np.linalg.inv(T_k), dx)
            (sign, logdet) = np.linalg.slogdet(T_k)
        return np.log(self.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2

    @staticmethod
    def logsumLogX(logX):
        """Computes log of the sum given the log of the summands.

        This method tries hard to avoid over- or underflow.
        See appendix A of Bovy, Hogg, Roweis (2009).

        Args:
        logX: (K, N) log-likelihoods from K calls to logL_K() with N coordinates

        Returns:
        (N, 1) of log of total likelihood

        """
        floatinfo = np.finfo(logX.dtype)
        underflow = np.log(floatinfo.tiny) - logX.min(axis=0)
        overflow = np.log(floatinfo.max) - logX.max(axis=0) - np.log(logX.shape[0])
        c = np.where(underflow < overflow, underflow, overflow)
        return np.log(np.exp(logX + c).sum(axis=0)) - c


############################
# Begin of fit functions
############################

def initializeFromDataMinMax(gmm, K, data=None, covar=None, s=None, rng=np.random):
    gmm.amp[:] = np.ones(K)/K # now gmm.K works
    # set model to random positions with equally sized spheres within
    # volumne spanned by data
    min_pos = data.min(axis=0)
    max_pos = data.max(axis=0)
    gmm.mean[:,:] = min_pos + (max_pos-min_pos)*np.random.rand(gmm.K, gmm.D)
    # if s is not set: use volume filling argument:
    # K spheres of radius s [having volume s^D * pi^D/2 / gamma(D/2+1)]
    # should completely fill the volume spanned by data.
    if s is None:
        from scipy.special import gamma
        vol_data = np.prod(max_pos-min_pos)
        s = (vol_data / gmm.K * gamma(gmm.D*0.5 + 1))**(1./gmm.D) / np.sqrt(np.pi)
        if gmm.verbose:
            print "initializing spheres with s=%.2f" % s
    gmm.covar[:,:,:] = np.tile(s**2 * np.eye(data.shape[1]), (gmm.K,1,1))

def initializeFromDataAtRandom(gmm, K, data=None, covar=None, s=None, rng=np.random):
    gmm.amp[:] = np.ones(K)/K
    # initialize components around data points with uncertainty s
    refs = np.random.randint(0, len(coords), size=K)
    gmm.mean[:,:] = coords[refs] + np.random.normal(0, s, size=(K,3))
    gmm.covar[:,:,:] = np.tile(s**2 * np.eye(data.shape[1]), (K,1,1))


def fit(data, covar=None, K=1, w=0., cutoff=None, sel_callback=None, n_missing=None, init_callback=initializeFromDataMinMax, tol=1e-3, verbose=False):
    gmm = GMM(K=K, D=data.shape[1], verbose=verbose)

    # init function as generic call
    init_callback(gmm, K, data, covar)

    return _run_EM(gmm, data, covar=covar, w=w, cutoff=cutoff, sel_callback=sel_callback, n_missing=n_missing, tol=tol)

def _run_EM(gmm, data, covar=None, w=0., cutoff=None, sel_callback=None, n_missing=None, tol=1e-3):
    maxiter = max(100, gmm.K)

    pool = multiprocessing.Pool()

    # sum_k p(x|k) -> S
    # extra precautions for cases when some points are treated as outliers
    # and not considered as belonging to any component
    S = np.zeros(len(data)) # S = sum_k p(x|k)
    log_S = createShared(np.empty(len(data)))
    N = np.zeros(len(data), dtype='bool') # N == 1 for points in the fit
    neighborhood = [None for k in xrange(gmm.K)]
    log_p = [[] for k in xrange(gmm.K)]
    T_inv = [None for k in xrange(gmm.K)]

    # save volumes to see which components change
    V = np.linalg.det(gmm.covar)

    # save the M sums from the non-imputed data
    A = np.empty(gmm.K)
    M = np.empty((gmm.K, gmm.D))
    C = np.empty((gmm.K, gmm.D, gmm.D))
    P = np.empty(gmm.K)

    # same for imputed data: set below if needed
    A2 = M2 = C2 = P2 = None

    # begin EM
    it = 0
    logL = None
    logL_obs = None
    n_impute = None
    n_guess = None
    while it < maxiter: # limit loop in case of no convergence

        # compute p(i | k) for each k independently in the pool
        # need S = sum_k p(i | k) for further calculation
        # also N = {i | i in neighborhood[k]} for any k
        results = [pool.apply_async(_E, (gmm, k, data, covar, neighborhood[k], cutoff)) for k in xrange(gmm.K)]
        for r in results:
            k, log_p[k], neighborhood[k], T_inv[k] = r.get()
            S[neighborhood[k]] += np.exp(log_p[k])
            N[neighborhood[k]] = 1
            if gmm.verbose >= 2:
                print "  k=%d: amp=%.3f pos=(%.1f, %.1f) s=%.2f |I| = %d <S> = %.3f" % (k, gmm.amp[k], gmm.mean[k][0], gmm.mean[k][1], np.linalg.det(gmm.covar[k])**(0.5/gmm.D), log_p[k].size, np.log(S[neighborhood[k]]).mean())

        # since log(0) isn't a good idea, need to restrict to N
        log_S[N] = np.log(S[N])
        logL_ = logL_obs_ = _logsum(log_S[N])

        if gmm.verbose:
            print ("%d\t%d\t%.4f" % (it, N.sum(), logL_)),

        # perform sums for M step in the pool
        results = [pool.apply_async(_computeMSums, (gmm, k, data, log_p[k], log_S, neighborhood[k], T_inv[k])) for k in xrange(gmm.K)]
        for r in results:
            k, A[k], M[k], C[k], P[k] = r.get()

        # need to do MC integral of p(missing | k):
        # get missing data by imputation from the current model
        if sel_callback is not None:
            RD = 200
            soften = 1 - np.exp(-it*0.1)
            RDs = int(RD*soften)
            logL2 = 0
            A2 = np.zeros(gmm.K)
            M2 = np.zeros((gmm.K, gmm.D))
            C2 = np.zeros((gmm.K, gmm.D, gmm.D))
            P2 = np.zeros(gmm.K)
            n_impute = 0

            results = [pool.apply_async(_computeIMSums, (gmm, sel_callback, len(data), n_missing, n_guess, cutoff, it*rd)) for rd in xrange(RDs)]
            for r in results:
                A2_, M2_, C2_, P2_, logL2_, n_impute_ = r.get()
                A2 += A2_
                M2 += M2_
                C2 += C2_
                P2 += P2_
                logL2 += logL2_
                n_impute += n_impute_

            if soften > 0 and RDs > 0:
                A2 *= soften / RDs
                M2 *= soften / RDs
                C2 *= soften / RDs
                P2 *= soften / RDs
                logL2 /= RDs
                logL_ += logL2 + np.log(soften)
                n_impute = n_impute * soften / RDs

                # update n_guess with <n_impute>
                n_guess = n_impute / soften

                if gmm.verbose:
                    print "\t%d\t%d\t%.2f\t%.4f\t%.4f" % (RDs, n_impute, soften, logL2, logL_)

        if gmm.verbose:
            print  ""

        # convergence test:
        if it > 5 and logL_ - logL < tol and logL_obs_ <= logL_obs:
            break
        else:
            logL = logL_
            logL_obs = logL_obs_

        # perform M step with M-sums of data and imputations runs
        _M(gmm, A, M, C, P, N.sum(), w, A2, M2, C2, P2, n_impute)

        # check new component volumes and reset sel when it grows by
        # more then 25%
        V_ = np.linalg.det(gmm.covar)
        changed = np.flatnonzero((V_- V)/V > 0.25)
        for c in changed:
            neighborhood[c] = None
            V[c] = V_[c]
            if gmm.verbose:
                print " resetting neighborhood[%d] due to volume change" % c

        S[:] = 0
        N[:] = 0
        it += 1
    return gmm

def _E(gmm, k, data, covar=None, neighborhood_k=None, cutoff=None):
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

    # NOTE: close to convergence, we can stop applying the cutoff because
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
    return k, np.log(gmm.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2, neighborhood_k, T_inv_k

def logL(gmm, data, covar=None, cutoff=None):
    S = np.zeros(len(data))
    for k in xrange(gmm.K):
        # to minimize the components with tiny p(x|k) in sum:
        # only use those within cutoff
        k, log_p_k, neighborhood_k, T_inv_k = _E(gmm, k, data, covar=covar, cutoff=cutoff)
        S[neighborhood_k] += np.exp(log_p_k)
    return np.log(S)

def _logsum(l):
    """Computes log of a sum, given the log of the elements.

    This method tries hard to avoid over- or underflow that may arise
    when computing exp(ll).

    See appendix A of Bovy, Hogg, Roweis (2009).

    Args:
        l:  (N,1) log of whatever

    """
    floatinfo = np.finfo(l.dtype)
    underflow = np.log(floatinfo.tiny) - l.min()
    overflow = np.log(floatinfo.max) - l.max() - np.log(l.size)
    if underflow < overflow:
        c = underflow
    else:
        c = overflow
    return np.log(np.exp(l + c).sum()) - c

def _M(gmm, A, M, C, P, n_points, w=0., A2=None, M2=None, C2=None, P2=None, n_points2=None):

    # if imputation was run
    if A2 is not None:
        A += A2
        M += M2
        C += C2
        n_points = n_points2 + n_points

        # imputation correction
        frac_p_out = P2 / (P + P2)
        C += gmm.covar * frac_p_out[:,None,None]

    # the actual update
    gmm.amp[:] = A / n_points
    gmm.mean[:,:] = M / A[:,None]
    if w > 0:
        # we assume w to be a lower bound of the isotropic dispersion,
        # C_k = w^2 I + ...
        # then eq. 38 in Bovy et al. only ~works for N = 0 because of the
        # prefactor 1 / (q_j + 1) = 1 / (A + 1) in our terminology
        # On average, q_j = N/K, so we'll adopt that to correct.
        w_eff = w**2 * (n_points*1./gmm.K + 1)
        gmm.covar[:,:,:] = (C + w_eff*np.eye(gmm.D)[None,:,:]) / (A + 1)[:,None,None]
    else:
        gmm.covar[:,:,:] = C / A[:,None,None]

def _computeMSums(gmm, k, data, log_p_k, log_S, neighborhood_k, T_inv_k=None):
    # needed for imputation correction: P_k = sum_i p_ik
    P_k = np.exp(_logsum(log_p_k))

    # form log_q_ik by dividing with S = sum_k p_ik
    # NOTE:  this modifies log_p_k in place!
    # NOTE2: reshape needed when neighborhood_k is None because of its
    # mpliciti meaning as np.newaxis (which would create a 2D array)
    log_p_k -= log_S[neighborhood_k].reshape(log_p_k.size)

    # amplitude: A_k = sum_i q_ik
    A_k = np.exp(_logsum(log_p_k))

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
    return k, A_k, M_k, C_k, P_k

def _computeIMSums(gmm, sel_callback, len_data, n_missing, n_guess, cutoff, seed=None):
    # create imputated data
    data2 = _I(gmm, sel_callback, len_data, n_missing=n_missing, n_guess=n_guess)
    A2 = np.zeros(gmm.K)
    M2 = np.zeros((gmm.K, gmm.D))
    C2 = np.zeros((gmm.K, gmm.D, gmm.D))
    P2 = np.zeros(gmm.K)
    logL2 = 0
    n_impute = len(data2)

    if len(data2):
        # similar setup as above, but since imputated points
        # are drawn from the model, we can avoid the caution of
        # dealing with outliers: all points will be considered
        S2 = np.zeros(len(data2))
        neighborhood2 = [None for k in xrange(gmm.K)]
        log_p2 = [[] for k in xrange(gmm.K)]

        # run E now on data2
        # then combine respective sums in M step
        for k in xrange(gmm.K):
            k, log_p2[k], neighborhood2[k], _ = _E(gmm, k, data2, cutoff=cutoff)
            S2[neighborhood2[k]] += np.exp(log_p2[k])

        log_S2 = np.log(S2)
        logL2 = _logsum(log_S2)

        for k in xrange(gmm.K):
            # with small imputation sets: neighborhood2[k] might be empty
            if neighborhood2[k] is None or neighborhood2[k].size:
                k, A2[k], M2[k], C2[k], P2[k] = _computeMSums(gmm, k, data2, log_p2[k], log_S2, neighborhood2[k])
    return A2, M2, C2, P2, logL2, n_impute

def _I(gmm, sel_callback, len_data=None, n_missing=None, n_guess=None, alpha=0.05):
    # create imputation sample from the current model
    if n_missing is not None:
        return gmm.draw(size=n_missing, sel_callback=sel_callback, invert_callback=True)
    else:
        if len_data is None:
            raise RuntimeError("I: need len_data when n_missing is None!")

        # confidence interval for Poisson variate len_data
        from scipy.stats import chi2
        lower = 0.5*chi2.ppf(alpha/2, 2*len_data)
        upper = 0.5*chi2.ppf(1 - alpha/2, 2*len_data + 2)

        # N: current guess of the whole sample, of which we can only
        # see the observable fraction len_data
        N = len_data
        if n_guess is not None:
            N += n_guess
        while True:
            # draw N without any selection
            sample = gmm.draw(N)
            # check if observed fraction is compatible with Poisson(len_data)
            sel = sel_callback(sample)
            N_o = sel.sum() # predicted observed
            if lower <= N_o and N_o <= upper:
                return sample[sel==False]
            else:
                # update N assuming N_o/N is ~correct and independent of N
                N = max(int(N*1.0/N_o * len_data), len_data)
