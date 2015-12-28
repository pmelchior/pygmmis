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
        sel = [None for k in xrange(self.K)]
        log_p = [[] for k in xrange(self.K)]
        
        # standard EM
        it = 0
        logL0 = None
        while it < maxiter: # limit loop in case of no convergence
            amp_ = self.amp.copy()
            mean_ = self.mean.copy()
            covar_ = self.covar.copy()
            # compute p(i | k) for each k independently
            # only need S = sum_k p(i | k) for further calculation
            for k in xrange(self.K):
                log_p[k] = self._E(k, data, sel, cutoff=cutoff)
                S[sel[k]] += np.exp(log_p[k])
                A[sel[k]] = 1
                if self.verbose:
                    print "  k = %d: |I| = %d <S> = %.3f" % (k, log_p[k].size, np.log(S[sel[k]]).mean())

            # since log(0) isn't a good idea, need to restrict to A
            log_S[A] = np.log(S[A])
            logL_ = log_S[A].mean()
            for k in xrange(self.K):
                self._M(k, data, log_p[k], log_S, sel[k], A.sum())

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
            
        # do we need imputation?
        if n_impute > 0:

            if self.verbose:
                print " starting imputation with ~%d extra points" % n_impute 
            
            # for each iteration, draw several fake data sets
            # estimate mean and std of their logL, test for convergence,
            # and adopt their _mean_ model for the next iteration
            it = 0
            logL0 = None
            RD = 5 # repeated draws for imputation
            logL__ = np.empty(RD)
            amp__ = np.empty((RD, self.K))
            mean__ = np.empty((RD, self.K, self.D))
            covar__ = np.empty((RD, self.K, self.D, self.D))
            
            while it < maxiter:
                amp_ = self.amp.copy()
                mean_ = self.mean.copy()
                covar_ = self.covar.copy()

                # run E step on data first as above
                if it == 0:
                    # need to undo what previous run of M did
                    for k in xrange(self.K):
                        log_p[k] += log_S[sel[k]].reshape(log_p[k].size)
                else:
                    for k in xrange(self.K):
                        log_p[k] = self._E(k, data, sel, cutoff=cutoff)
                        S[sel[k]] += np.exp(log_p[k])
                        A[sel[k]] = 1
                        if self.verbose:
                            print "  k = %d: |I| = %d <S> = %.3f" % (k, log_p[k].size, np.log(S[sel[k]]).mean())
                    log_S[A] = np.log(S[A])
                
                rd = 0
                while rd < RD:
                    
                    # reset model to current
                    self.amp[:] = amp_[:]
                    self.mean[:,:] = mean_[:,:]
                    self.covar[:,:,:] = covar_[:,:,:]
                    
                    # create imputated data
                    data2 = self._I(n_impute, sel_callback=sel_callback)

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
                            print "  k = %d: |I2| = %d <S> = %.3f" % (k, log_p2[k].size, np.log(S2[sel2[k]]).mean())

                    log_S2 = np.log(S2)
                    logL__[rd] = np.concatenate((log_S, log_S2)).mean()
                    for k in xrange(self.K):
                        self._M2(k, data, data2, log_p[k], log_p2[k], log_S, log_S2, sel[k], sel2[k], A.sum() + len(data2))

                    # save model
                    amp__[rd,:] = self.amp[:]
                    mean__[rd,:,:] = self.mean[:,:]
                    covar__[rd,:,:,:] = self.covar[:,:,:]
                    
                    if self.verbose:
                        print "   iter %d/%d: %.3f\t|A| = %d" % (it, rd, logL__[rd], A.sum() + n_impute)

                    rd += 1
                    
                # convergence test:
                # in principle one can do Welch's t-test wrt iteration before
                # but the actual risk here is a run-away, which
                # drastically _reduces_ the likelihood, at which point we abort
                if self.verbose:
                    print " iter %d: %.3f" % (it, np.array(logL__).mean())

                if it > 0 and logL__.mean() - logL0 < tol:
                    break
                else:
                    logL0 = logL__.mean()

                # because the components remain ordered, we can
                # adopt the mean of the repeated draws as new model
                self.amp = amp__.mean(axis=0) 
                self.mean = mean__.mean(axis=0) 
                self.covar = covar__.mean(axis=0)
                
                it += 1
                S[:] = 0
                A[:] = 0


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

    def _M(self, k, data, log_p_k, log_S, sel_k, n_points, n_impute=0):

        # maybe needed for later _M2 calls: P_k = sum_i p_ik
        if n_impute > 0:
            P_k = np.exp(self._logsum(log_q_k))
        else:
            P_k = None
            
        # reshape needed when sel_k is None because of its implicit meaning
        # as np.newaxis (which would create a 2D array)
        # NOTE: this modifies log_q_k in place!
        log_p_k -= log_S[sel_k].reshape(log_p_k.size)

        # amplitude: A_k = sum_i q_ik
        A_k = np.exp(self._logsum(log_p_k))
        self.amp[k] = A_k/n_points

        # mean: M_k = sum_i x_i q_ik
        qk = np.exp(log_p_k)
        d = data[sel_k].reshape((log_p_k.size, self.D))
        M_k = (d * qk[:,None]).sum(axis=0)
        self.mean[k,:] = M_k/A_k

        # covariance: C_k = sum_i (x_i - mu_k)^T(x_i - mu_k) q_ik
        d_m = d - self.mean[k]
        # funny way of saying: for each point i, do the outer product
        # of d_m with its transpose, multiply with pi[i], and sum over i
        C_k = (qk[:, None, None] * d_m[:, :, None] * d_m[:, None, :]).sum(axis=0)
        # Bayesian regularization term
        if self.w > 0:
            self.covar[k,:,:] = (C_k + self.w*np.eye(self.D)) / (A_k + 1)
        else:
            self.covar[k,:,:] /= C_k/A_k

        return A_k, M_k, C_k, P_k
        

    def _M2(self, k, data, data2, log_q_k, log_q2_k, log_S, log_S2, sel_k, sel2_k, n_points):
        # before we modify log_p, we need to store the fractional probability
        # of imputed points (compared to all) for each component
        frac_p_k_out = np.exp(self._logsum(log_q2_k))
        frac_p_k_out /= np.exp(self._logsum(log_q_k)) + frac_p_k_out
        
        # reshape needed when sel_k is None because of its implicit meaning
        # as np.newaxis (which would create a 2D array)
        log_q_k -= log_S[sel_k].reshape(log_q_k.size)
        log_q2_k -= log_S2[sel2_k].reshape(log_q2_k.size)
        sum_i_q_k = np.exp(self._logsum(log_q_k)) + np.exp(self._logsum(log_q2_k))
        # amplitude
        self.amp[k] = sum_i_q_k/n_points

        # mean
        qk = np.exp(log_q_k)
        q2k = np.exp(log_q2_k)
        d = data[sel_k].reshape((log_q_k.size, self.D))
        d2 = data2[sel2_k].reshape((log_q2_k.size, self.D))
        self.mean[k,:] = ((d * qk[:,None]).sum(axis=0) +
                          (d2 * q2k[:,None]).sum(axis=0))/sum_i_q_k
                               
        # covariance
        d_m = d - self.mean[k]
        d2_m = d2 - self.mean[k]

        # imputation correction
        self.covar[k] *= frac_p_k_out
                               
        # funny way of saying: for each point i, do the outer product
        # of d_m with its transpose, multiply with pi[i], and sum over i
        self.covar[k,:,:] += (qk[:, None, None] * d_m[:, :, None] * d_m[:, None, :]).sum(axis=0) + (q2k[:, None, None] * d2_m[:, :, None] * d2_m[:, None, :]).sum(axis=0)

        # Bayesian regularization term
        if self.w > 0:
            self.covar[k,:,:] += self.w*np.eye(self.D)
            self.covar[k,:,:] /= sum_i_q_k + 1
        else:
            self.covar[k,:,:] /= sum_i_q_k

        # need to undo change to log_q_k because we need to reuse
        # FIXME: stupid!!!
        log_q_k += log_S[sel_k].reshape(log_q_k.size)

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


