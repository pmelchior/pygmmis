#!/bin/env python

from iemgmm import GMM
import numpy as np

class ICGMM(GMM):

    def __init__(self, K=1, D=1, data=None, s=None, w=0., cutoff=None, sel_callback=None, n_missing=None, init_callback=None, rng=None, verbose=False):
        GMM.__init__(self, K=K, D=D, rng=rng, verbose=verbose)
        
        if data is not None:
            self.D = data.shape[1]
            self.w = w
            if init_callback is not None:
                self.amp, self.mean, self.covar = init_callback(K)
            else:
                self._initializeModel(K, s, data)
            self._run_EM(data, cutoff=cutoff, sel_callback=sel_callback, n_missing=n_missing)
            
    # no imputation for now
    def _run_EM(self, data, cutoff=None, sel_callback=None, n_missing=None, tol=1e-3):
        maxiter = 100

        # sum_k p(x|k) -> S
        # extra precautions for cases when some points are treated as outliers
        # and not considered as belonging to any component
        S = np.zeros(len(data)) # S = sum_k p(x|k)
        log_S = np.empty(len(data))
        N = np.zeros(len(data), dtype='bool') # N == 1 for points in the fit
        sel = [None for k in xrange(self.K)]
        log_p = [[] for k in xrange(self.K)]

        # save volumes to see which components change
        V = np.linalg.det(self.covar)

        # save the M sums from the non-imputed data
        A = np.empty(self.K)
        M = np.empty((self.K, self.D))
        C = np.empty((self.K, self.D, self.D))
        P = np.empty(self.K)
        
        # same for imputed data: set below if needed
        A2 = M2 = C2 = P2 = None

        # begin EM
        it = 0
        logL = None
        logL_obs = None
        n_impute = None
        n_guess = None
        while it < maxiter: # limit loop in case of no convergence

            # compute p(i | k) for each k independently
            # need S = sum_k p(i | k) for further calculation
            for k in xrange(self.K):
                log_p[k] = self._E(k, data, sel, cutoff=cutoff)
                S[sel[k]] += np.exp(log_p[k])
                N[sel[k]] = 1
                if self.verbose:
                    print "  k = %d: |I| = %d <S> = %.3f" % (k, log_p[k].size, np.log(S[sel[k]]).mean())

            # since log(0) isn't a good idea, need to restrict to N
            log_S[N] = np.log(S[N])
            logL_ = logL_obs_ = self._logsum(log_S[N])

            for k in xrange(self.K):
                A[k], M[k], C[k], P[k] = self._computeMSums(k, data, log_p[k], log_S, sel[k])

            # need to do MC integral of p(missing | k):
            # get missing data by imputation from the current model
            if it > 0 and sel_callback is not None:
                rd = 0
                RD = 200
                logL2_ = np.empty(RD)
                A2 = np.zeros(self.K)
                M2 = np.zeros((self.K, self.D))
                C2 = np.zeros((self.K, self.D, self.D))
                P2 = np.zeros(self.K)
                n_impute = 0
                while rd < RD:
                    
                    # create imputated data
                    data2 = self._I(sel_callback, len(data), n_missing=n_missing, n_guess=n_guess)

                    # similar setup as above, but since imputated points
                    # are drawn from the model, we can avoid the caution of
                    # dealing with outliers: all points will be considered
                    S2 = np.zeros(len(data2))
                    sel2 = [None for k in xrange(self.K)]
                    log_p2 = [[] for k in xrange(self.K)]
            
                    # run E now on data2
                    # then combine respective sums in M step
                    for k in xrange(self.K):
                        log_p2[k] = self._E(k, data2, sel2, cutoff=cutoff)
                        S2[sel2[k]] += np.exp(log_p2[k])
                        if self.verbose:
                            print "    k = %d: |I2| = %d <S> = %.3f" % (k, log_p2[k].size, np.log(S2[sel2[k]]).mean())

                    log_S2 = np.log(S2)
                    logL2_[rd] = self._logsum(log_S2)
                    n_impute += len(data2)
                    
                    for k in xrange(self.K):
                        # with small imputation sets: sel2[k] might be empty
                        if sel2[k].size:
                            a2, m2, c2, p2 = self._computeMSums(k, data2, log_p2[k], log_S2, sel2[k])
                            A2[k] += a2
                            M2[k] += m2
                            C2[k] += c2
                            P2[k] += p2

                    rd += 1

                    # some convergence criterion
                    if False:
                        break
                    
                A2 /= rd
                M2 /= rd
                C2 /= rd
                P2 /= rd
                n_impute = 1./rd * n_impute
                logL_ += logL2_[:rd].mean()

                # update n_guess with <n_impute>
                n_guess = n_impute

                print "%d\t%d\t%.4f\t%d\t%d\t%.4f\t%.4f\t%.4f" % (it, N.sum(), self._logsum(log_S[N]),  rd, n_impute, logL2_[:rd].mean(), logL2_[:rd].std()/rd**0.5, logL_)
            else:
                print "%d\t%d\t%.4f" % (it, N.sum(), logL_)
                
            # convergence test:
            if it > 5 and logL_ - logL < tol and logL_obs_ < logL_obs:
                break
            else:
                logL = logL_
                logL_obs = logL_obs_

            # perform M step with M-sums of data and imputations runs
            self._M(A, M, C, P, N.sum(), A2, M2, C2, P2, n_impute)

            # check new component volumes and reset sel when it grows by
            # more then 25%
            V_ = np.linalg.det(self.covar)
            changed = np.flatnonzero((V_- V)/V > 0.25)
            for c in changed:
                sel[c] = None
                V[c] = V_[c]
                if self.verbose:
                    print " resetting sel[%d] due to volume change" % c

            S[:] = 0
            N[:] = 0
            it += 1

            
    def logL(self, data, cutoff=None):
        S = np.zeros(len(data))
        sel = [None for k in xrange(self.K)]
        for k in xrange(self.K):
            # to minimize the components with tiny p(x|k) in sum:
            # only use those within cutoff
            log_p_k = self._E(k, data, sel, cutoff=cutoff)
            S[sel[k]] += np.exp(log_p_k)
        return np.log(S)

    def _E(self, k, data, sel, cutoff=None):
        # p(x | k) for all x in the vicinity of k
        # determine all points within cutoff sigma from mean[k]
        if cutoff is None or sel[k] is None:
            dx = data - self.mean[k]
        else:
            dx = data[sel[k]] - self.mean[k]
        chi2 = np.einsum('...j,j...', dx, np.dot(np.linalg.inv(self.covar[k]), dx.T))
        # close to convergence, we can stop applying the cutoff because
        # changes to sel will be minimal
        if cutoff is not None:
            indices = np.flatnonzero(chi2 < cutoff*cutoff*self.D)
            chi2 = chi2[indices]
            if sel[k] is None:
                sel[k] = indices
            else:
                sel[k] = sel[k][indices]
        
        # prevent tiny negative determinants to mess up
        (sign, logdet) = np.linalg.slogdet(self.covar[k])

        log2piD2 = np.log(2*np.pi)*(0.5*self.D)
        return np.log(self.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2


    def _M(self, A, M, C, P, n_points, A2=None, M2=None, C2=None, P2=None, n_points2=None):

        # if imputation was run
        if A2 is not None:
            # make sure imputated points don't dominate
            problems = (A2 > A)
            if problems.any():
                rescale = np.ones(self.K)
                rescale[problems] = A[problems] / A2[problems]
                A2 *= rescale
                P2 *= rescale
                M2 *= rescale[:,None]
                C2 *= rescale[:,None,None]

            A += A2
            M += M2
            C += C2
            n_points = n_points2 + n_points

            # imputation correction
            frac_p_out = P2 / (P + P2)
            C += self.covar * frac_p_out[:,None,None]

        # the actual update
        self.amp[:] = A / n_points
        # because of possible rescaling above: need to renormalize
        # Note: the reduces the weight of a component (wrt others) if
        # it becomes dominated by imputation points.
        self.amp /= self.amp.sum()
        
        self.mean[:,:] = M / A[:,None]
        if self.w > 0:
            self.covar[:,:,:] = (C + (self.w*np.eye(self.D))[None,:,:]) / (A + 1)[:,None,None]
        else:
            self.covar[:,:,:] = C / A[:,None,None]
            

    def _computeMSums(self, k, data, log_p_k, log_S, sel_k):
        # needed for imputation correction: P_k = sum_i p_ik
        P_k = np.exp(self._logsum(log_p_k))
            
        # reshape needed when sel_k is None because of its implicit meaning
        # as np.newaxis (which would create a 2D array)
        # NOTE: this modifies log_q_k in place!
        log_p_k -= log_S[sel_k].reshape(log_p_k.size)

        # amplitude: A_k = sum_i q_ik
        A_k = np.exp(self._logsum(log_p_k))

        # mean: M_k = sum_i x_i q_ik
        qk = np.exp(log_p_k)
        d = data[sel_k].reshape((log_p_k.size, self.D))
        M_k = (d * qk[:,None]).sum(axis=0)

        # covariance: C_k = sum_i (x_i - mu_k)^T(x_i - mu_k) q_ik
        d_m = d - self.mean[k]
        # funny way of saying: for each point i, do the outer product
        # of d_m with its transpose, multiply with pi[i], and sum over i
        C_k = (qk[:, None, None] * d_m[:, :, None] * d_m[:, None, :]).sum(axis=0)
        return A_k, M_k, C_k, P_k
        

    def _logsum(self, l):
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


