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
        import multiprocessing
        import parmap
        chunksize = int(np.ceil(self.K*1./multiprocessing.cpu_count()))
        # log p (x | k) for each k
        log_p = parmap.map(self.logL_k, xrange(self.K), coords, covar, False, chunksize=chunksize)
        return self.logsumLogX(np.array(log_p)) # sum over all k

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


def fit(data, covar=None, K=1, w=0., cutoff=None, sel_callback=None, N_missing=None, init_callback=initializeFromDataMinMax, tol=1e-3, verbose=False, logfile=None):
    gmm = GMM(K=K, D=data.shape[1], verbose=verbose)

    # init function as generic call
    init_callback(gmm, K, data, covar)

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

    # save the M sums from the non-imputed data
    A_ = np.empty(gmm.K)
    M_ = np.empty((gmm.K, gmm.D))
    C_ = np.empty((gmm.K, gmm.D, gmm.D))
    P_ = np.empty(gmm.K)
    A = np.empty(gmm.K)

    # "global" imputation variables
    log_S2_mean = N_imp = N_guess = troubled = None
    A2 = np.empty(gmm.K)
    A2_ = None
    M2_ = None
    C2_ = None
    P2_ = None
    N_imp_ = None

    if logfile is not None:
        logfile = open(logfile, 'w')

    # begin EM
    it = 0
    log_L = None
    log_L_obs = None
    log_S_mean = None
    maxiter = max(100, gmm.K)
    conv_iter = 5
    while it < maxiter: # limit loop in case of no convergence

        # compute p(i | k) for each k independently in the pool
        # need S = sum_k p(i | k) for further calculation
        # also N = {i | i in neighborhood[k]} for any k
        k = 0
        for log_p[k], neighborhood[k], T_inv[k] in \
        parmap.starmap(_E, zip(xrange(gmm.K), neighborhood), gmm, data, covar, cutoff, pool=pool, chunksize=chunksize):
            S[neighborhood[k]] += np.exp(log_p[k])
            N[neighborhood[k]] = 1
            if gmm.verbose >= 2:
                print "  k=%d: amp=%.3f pos=(%.1f, %.1f) s=%.2f |I| = %d <S> = %.3f" % (k, gmm.amp[k], gmm.mean[k][0], gmm.mean[k][1], np.linalg.det(gmm.covar[k])**(0.5/gmm.D), log_p[k].size, np.log(S[neighborhood[k]]).mean())
            k += 1

        # since log(0) isn't a good idea, need to restrict to N
        log_S[N] = np.log(S[N])
        log_S_mean_ = log_S[N].mean()
        log_L_ = log_L_obs_ = log_S[N].sum()
        N_ = N.sum()

        if gmm.verbose:
            print ("%d\t%d\t%.4f\t%.4f" % (it, N_, log_L_, log_S_mean_)),
            if sel_callback is None:
                print ""

        # perform sums for M step in the pool
        k = 0
        for A_[k], M_[k], C_[k], P_[k] in \
        parmap.starmap(_computeMSums, zip(xrange(gmm.K), neighborhood, log_p, T_inv), gmm, data, log_S, pool=pool, chunksize=chunksize):
            k += 1

        # need to do MC integral of p(missing | k):
        # get missing data by imputation from the current model
        if sel_callback is not None:

            # with imputation the observed data logL can decrease:
            # revert to previous model is that is that case
            if it > conv_iter and log_S_mean_ < log_S_mean:
                if gmm.verbose:
                    print "\nmean likelihood decreased: stopping reverting to previous model."
                gmm = gmm_
                break

            RD = 200
            soften =  1./(1+np.exp(-(it-4.)/2))
            RDs = int(RD*soften)
            log_L2_ = 0
            log_S2_mean_ = 0
            A2_ = np.zeros(gmm.K)
            M2_ = np.zeros((gmm.K, gmm.D))
            C2_ = np.zeros((gmm.K, gmm.D, gmm.D))
            P2_ = np.zeros(gmm.K)
            N_imp_ = 0

            chunksize_I = int(np.ceil(RDs)*1./multiprocessing.cpu_count())
            for A2__, M2__, C2__, P2__, log_L2__, log_S2_mean__, N_imp__ in \
            parmap.map(_computeIMSums, it*np.arange(RDs), gmm, sel_callback, N_, N_missing, N_guess, cutoff, pool=pool, chunksize=chunksize_I):
                A2_ += A2__
                M2_ += M2__
                C2_ += C2__
                P2_ += P2__
                log_L2_ += log_L2__
                log_S2_mean_ += log_S2_mean__
                N_imp_ += N_imp__

            if soften > 0 and RDs > 0:
                A2_ *= soften / RDs
                M2_ *= soften / RDs
                C2_ *= soften / RDs
                P2_ *= soften / RDs
                log_S2_mean_ /= RDs
                log_L2_ *= soften / RDs
                log_L_ += log_L2_
                N_imp_ = N_imp_ * soften / RDs

                # update N_guess with <N_imp>
                N_guess = N_imp_ / soften

                if gmm.verbose:
                    print ("\t%d\t%d\t%.2f\t%.4f\t%.4f" % (RDs, N_imp_, soften, log_L2_, log_L_))

            # compute changes of log_S, log_S2, N_imp
            # to check if imputation gets instable
            if it > 0: # need to wait out the first iteration
                d_log_S_mean = log_S_mean_ - log_S_mean
                d_log_S2_mean = log_S2_mean_ - log_S2_mean
                d_N_imp = N_imp_ - N_imp
                # limit is upper bound for d/dt N_imp such that d/dt L_tot = 0
                limit = -1./(log_S_mean_) * (N_ * d_log_S_mean + N_imp_ * d_log_S2_mean)
                # determine which component(s) drive imputation
                # if they exceed limit, they will be frozed in the M step
                A_o_ = A_.sum()
                A_m_ = A2_.sum()
                d_A_o = A_ - A
                d_A_m = A2_ - A2
                d_N_m = N_ / A_o_ * d_A_m - N_* A_m_ / A_o_**2 * d_A_o
                troubled_ = d_N_m  / limit >= 1
                # see if we can softly unthaw components that are not
                # troubled anymore
                if troubled is not None:
                    improved = troubled & (troubled_ == False)
                    # allow increase of from A2 to A2_ in proportion to
                    # room under the limit (i.e. d_N_m / limit)
                    A2_[improved] = np.minimum(A2_[improved], A2[improved] + A2_[improved]*(1 - np.maximum(0, d_N_m[improved]/limit)))
                    # need to correct N_imp_, otherwise normalization wrong
                    N_imp_ *= A2_.sum() / A_m_
                troubled = troubled_

        if logfile is not None:
            # create dummies when there was no imputation
            if sel_callback is None:
                logL2 = soften = N_imp = log_S2_mean = -1
                P2 = np.ones_like(P)
            logfile.write("%d\t%.3f\t%.4f\t%.4f\t%.4f\t%.1f\t%.6f\t%.4f\t%.4f" % (it, soften, log_L_, log_L_obs_, log_L2_, N_, N_imp,  log_S_mean, log_S2_mean))
            for k in xrange(gmm.K):
                logfile.write("\t%.3f\t%.4f\t%.4f" % (gmm.amp[k], np.log(A_[k]), np.log(A2_[k])))
            logfile.write("\n")

        # perform M step with M-sums of data and imputations runs
        _M(gmm, A_, M_, C_, P_, N_, w, A2_, M2_, C2_, P2_, N_imp_, troubled)

        # convergence test:
        if it > conv_iter and log_S_mean_ - log_S_mean < tol:
            if gmm.verbose:
                print "mean likelihood converged within tolerance %r: stopping here." % tol
            break

        # update all important _ quantities for convergence test(s)
        log_L = log_L_
        log_L_obs = log_L_obs_
        A[:] = A_[:]
        log_S_mean = log_S_mean_
        if sel_callback is not None:
            A2[:] = A2_[:]
            log_S2_mean = log_S2_mean_
            N_imp = N_imp_
            gmm_ = gmm # backup if next step gets worse (note: not gmm = gmm_!)

        # check new component volumes and reset sel when it grows by
        # more then 25%
        V_ = np.linalg.det(gmm.covar)
        changed = np.flatnonzero((V_- V)/V > 0.25)
        for c in changed:
            neighborhood[c] = None
            V[c] = V_[c]
            if gmm.verbose >= 2:
                print " resetting neighborhood[%d] due to volume change" % c
        S[:] = 0
        N[:] = 0
        it += 1

    if logfile is not None:
        logfile.close()
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
    return np.log(gmm.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2, neighborhood_k, T_inv_k

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

def _M(gmm, A, M, C, P, n_points, w=0., A2=None, M2=None, C2=None, P2=None, n_points2=None, troubled=None):

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
    amp_ = A / n_points

    # if there are troubled components: freeze their amplitude
    # and reweight the others according to their free amplitudes
    if troubled is not None and troubled.any():
        if not troubled.all():
            gmm.amp[troubled==False] = amp_[troubled==False]
            # allow troubled components to decrease (unlinkely, but why not)
            gmm.amp[troubled] = np.minimum(gmm.amp[troubled], amp_[troubled])
            # sum over troubled and good components
            sat = gmm.amp[troubled].sum()
            sag = gmm.amp[troubled==False].sum()
            # ensure sum_km amp_k =1, while fixing the troubled components
            # Note: a simple amp /= amp.sum() does not work well because
            # the troubled components rise, so that the others typically
            # have less weight in the sum. This way the total sum of the
            # troubled components is constant
            gmm.amp[troubled==False] *= (1-sat) / sag
    else:
        gmm.amp[:] = amp_[:]

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

def _computeMSums(k, neighborhood_k, log_p_k, T_inv_k, gmm, data, log_S):
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
    return A_k, M_k, C_k, P_k

def _computeIMSums(seed, gmm, sel_callback, len_data, n_missing, N_guess, cutoff):
    # create imputated data
    data2 = _I(gmm, sel_callback, len_data, n_missing=n_missing, N_guess=N_guess)
    covar2 = T2_inv = None
    A2 = np.zeros(gmm.K)
    M2 = np.zeros((gmm.K, gmm.D))
    C2 = np.zeros((gmm.K, gmm.D, gmm.D))
    P2 = np.zeros(gmm.K)
    log_L2 = 0
    log_S2_mean = 0
    N_imp = len(data2)

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
            log_p2[k], neighborhood2[k], _ = _E(k, neighborhood2[k], gmm, data2, covar2, cutoff=cutoff)
            S2[neighborhood2[k]] += np.exp(log_p2[k])

        log_S2 = np.log(S2)
        log_S2_mean = log_S2.mean()
        log_L2 = log_S2.sum()

        for k in xrange(gmm.K):
            # with small imputation sets: neighborhood2[k] might be empty
            if neighborhood2[k] is None or neighborhood2[k].size:
                A2[k], M2[k], C2[k], P2[k] = _computeMSums(k, neighborhood2[k], log_p2[k], T2_inv, gmm, data2, log_S2)
    return A2, M2, C2, P2, log_L2, log_S2_mean, N_imp

def _I(gmm, sel_callback, len_data=None, n_missing=None, N_guess=None, alpha=0.05):
    # create imputation sample from the current model
    if n_missing is not None:
        sample =  gmm.draw(size=n_missing, sel_callback=sel_callback, invert_callback=True)
        return sample
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
        if N_guess is not None:
            N += N_guess
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
