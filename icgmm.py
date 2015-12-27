#!/bin/env python

from iemgmm import GMM
import numpy as np

class ICGMM(GMM):

    def __init__(self, K=1, D=1, data=None, s=None, w=0., cutoff=None, n_impute=0, sel_callback=None, verbose=False):
        self.verbose = verbose
        if data is not None:
            self.D = data.shape[1]
            self.w = w
            self.initializeModel(K, s, data)     # will need randoms if non-uniform
            self.sel = [None for k in xrange(K)] # selection of points per component
            self._run_EM(data, cutoff=cutoff, n_impute=n_impute, sel_callback=sel_callback)
        else:
            self.D = D
            self.amp = np.zeros((K))
            self.mean = np.empty((K,D))
            self.covar = np.empty((K,D,D))
            
    # no imputation for now
    def _run_EM(self, data, cutoff=None, n_impute=0, sel_callback=None, tol=1e-3):
        maxiter = 100

        # sum_k p(x|k) -> S
        # extra precautions for cases when some points are treated as outliers
        # and not considered as belonging to any component
        S = np.zeros(len(data)) # S = sum_k p(x|k)
        log_S = np.empty(len(data))
        A = np.zeros(len(data), dtype='bool') # A == 1 for points in the fit

        # standard EM
        if n_impute == 0:
            it = 0
            logL0 = None
            while it < maxiter: # limit loop in case of no convergence
                amp_ = self.amp.copy()
                mean_ = self.mean.copy()
                covar_ = self.covar.copy()
                try:
                    # compute p(i | k) for each k independently
                    # only need S = sum_k p(i | k) for further calculation
                    log_p = []
                    for k in xrange(self.K):
                        log_p.append(self._E(data, k, cutoff=cutoff))
                        S[self.sel[k]] += np.exp(log_p[k])
                        A[self.sel[k]] = 1
                        if self.verbose:
                            print "  k = %d: |I| = %d <S> = %.3f" % (k, self.sel[k].size, np.log(S[self.sel[k]]).mean())

                    # FIXME: this will throw an error of points are outside
                    # of any selection, but it causes no other problems
                    log_S[A] = np.log(S[A])
                    logL_ = log_S[A].mean()
                    for k in xrange(self.K):
                        self._M(data, k, log_p[k], log_S, A.sum())
                        
                    if self.verbose:
                        print " iter %d: <S> = %.3f\t|A| = %d" % (it, logL_, A.sum())
                    # convergence test
                    if it > 0 and logL_ - logL0 < tol:
                        break
                    else:
                        logL0 = logL_

                    it += 1
                    S[:] = 0
                    A[:] = 0
                
                except np.linalg.linalg.LinAlgError:
                    if self.verbose:
                        print "warning: ran into trouble, stopping fit at previous position"
                    self.amp = amp_
                    self.mean = mean_
                    self.covar = covar_
                    break
                
    def logL(self, data):
        S = np.zeros(len(data))
        log_p = []
        for k in xrange(self.K):
            log_p.append(self._E(data, k))
            S += np.exp(log_p[k])
        return np.log(S)

    def _E(self, data, k, cutoff=None, return_index=False):
        # p(x | k) for all x in the vicinity of k
        # determine all points within cutoff sigma from mean[k]
        if cutoff is None or self.sel[k] is None:
            dx = data - self.mean[k]
        else:
            dx = data[self.sel[k]] - self.mean[k]
        chi2 = np.einsum('...j,j...', dx, np.dot(np.linalg.inv(self.covar[k]), dx.T))
        # close to convergence, we can stop applying the cutoff because
        # changes to sel will be minimal
        if cutoff is not None:
            indices = np.flatnonzero(chi2 < cutoff**2*self.D)
            chi2 = chi2[indices]
            if self.sel[k] is None:
                self.sel[k] = indices
            else:
                self.sel[k] = self.sel[k][indices]
        
        # prevent tiny negative determinants to mess up
        (sign, logdet) = np.linalg.slogdet(self.covar[k])

        log2piD2 = np.log(2*np.pi)*(0.5*self.D)
        log_p = np.log(self.amp[k]) - log2piD2 - sign*logdet/2 - chi2/2

        if return_index:
            return log_p, self.sel[k]
        else:
            return log_p

    def _M(self, data, k, log_q_k, log_S, all_points):
        log_q_k -= log_S[self.sel[k]]
        sum_i_q_k = np.exp(self._logsum(log_q_k))
        self.amp[k] = sum_i_q_k/all_points

        qk = np.exp(log_q_k)
        # mean
        self.mean[k] = (data[self.sel[k]] * qk[:,None]).sum(axis=0)/sum_i_q_k

        # funny way of saying: for each point i, do the outer product
        # of d_m with its transpose, multiply with pi[i], and sum over i
        d_m = data[self.sel[k]] - self.mean[k]
        self.covar[k] = (qk[:, None, None] * d_m[:, :, None] * d_m[:, None, :]).sum(axis=0)
            
        # Bayesian regularization term
        if self.w > 0:
            self.covar[k] += self.w*np.eye(self.D)
            self.covar[k] /= sum_i_q_k + 1
        else:
            self.covar[k] /= sum_i_q_k

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


