#!/bin/env python

from iemgmm import GMM
import numpy as np

class ICGMM(GMM):

    def __init__(self, K=1, D=1, data=None, s=None, w=0., cutoff=None, sel_callback=None, n_missing=None, init_callback=None, rng=None, verbose=False):
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        self.verbose = verbose
        
        if data is not None:
            self.D = data.shape[1]
            self.w = w
            if init_callback is not None:
                self.amp, self.mean, self.covar = init_callback(K)
            else:
                self._initializeModel(K, s, data)
            self._run_EM(data, cutoff=cutoff, sel_callback=sel_callback, n_missing=n_missing)
        else:
            self.D = D
            self.amp = np.zeros((K))
            self.mean = np.empty((K,D))
            self.covar = np.empty((K,D,D))
            
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

        # standard EM
        it = 0
        logL0 = None
        while it < maxiter: # limit loop in case of no convergence
            amp_ = self.amp.copy()
            mean_ = self.mean.copy()
            covar_ = self.covar.copy()
            # compute p(i | k) for each k independently
            # only need S = sum_k p(i | k) for further calculation
            if it > 0:
                S[:] = 0
                N[:] = 0
            for k in xrange(self.K):
                log_p[k] = self._E(k, data, sel, cutoff=cutoff)
                S[sel[k]] += np.exp(log_p[k])
                N[sel[k]] = 1
                if self.verbose:
                    print "  k = %d: |I| = %d <S> = %.3f" % (k, log_p[k].size, np.log(S[sel[k]]).mean())

            # since log(0) isn't a good idea, need to restrict to N
            log_S[N] = np.log(S[N])
            logL_ = log_S[N].mean()
            for k in xrange(self.K):
                self._M(k, data, log_p[k], log_S, sel[k], N.sum())

            if self.verbose:
                print " iter %d: <S> = %.3f\t|N| = %d" % (it, logL_, N.sum())
                
            # convergence test:
            if it > 0 and logL_ - logL0 < tol:
                break
            else:
                logL0 = logL_

            it += 1

        # do we need imputation?
        if sel_callback is not None:

            if self.verbose:
                print "starting imputation:"
            
            # for each iteration, draw several fake data sets
            # estimate mean and std of their logL, test for convergence,
            # and adopt their _mean_ model for the next iteration
            it = 0
            logL0 = None
            n_guess = None
            RD = 5 # repeated draws for imputation
            logL__ = np.empty(RD)
            amp__ = np.empty((RD, self.K))
            mean__ = np.empty((RD, self.K, self.D))
            covar__ = np.empty((RD, self.K, self.D, self.D))
            n_impute__ = np.empty(RD)

            # save the M sums from the non-imputed data
            A = np.empty(self.K)
            M = np.empty((self.K, self.D))
            C = np.empty((self.K, self.D, self.D))
            P = np.empty(self.K)

            # save volumes to see which components change
            V = np.linalg.det(self.covar)
            
            while it < maxiter:

                # save existing params to be used for each imputation run
                amp_ = self.amp.copy()
                mean_ = self.mean.copy()
                covar_ = self.covar.copy()

                # run E on data first as above, and store intermediate sums
                # of M step for re-use during each imputation run
                S[:] = 0
                N[:] = 0
                for k in xrange(self.K):
                    log_p[k] = self._E(k, data, sel, cutoff=cutoff)
                    S[sel[k]] += np.exp(log_p[k])
                    N[sel[k]] = 1
                    if self.verbose:
                        print "    k = %d: |I| = %d <S> = %.3f" % (k, log_p[k].size, np.log(S[sel[k]]).mean())
                log_S[N] = np.log(S[N])
                for k in xrange(self.K):
                    A[k], M[k], C[k], P[k] = self._computeMSums(k, data, log_p[k], log_S, sel[k], impute=True)
                
                rd = 0

                mean_log_S2 = 0
                tot_S2 = 0
                mean_N2 = 0
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
                    logL__[rd] = np.concatenate((log_S[N], log_S2)).mean()
                    n_impute__[rd] = len(data2)
                    for k in xrange(self.K):
                        self._M2(k, data2, log_p2[k], log_S2, sel2[k], A[k], M[k], C[k], P[k], N.sum())
                    self.amp /= self.amp.sum()
                    
                    # save model
                    amp__[rd,:] = self.amp[:]
                    mean__[rd,:,:] = self.mean[:,:]
                    covar__[rd,:,:,:] = self.covar[:,:,:]
                    
                    if self.verbose:
                        print "   iter %d/%d: %.3f\t|N| = %d" % (it, rd, logL__[rd], N.sum() + len(data2))

                    mean_log_S2 += log_S2.mean()
                    tot_S2 += np.exp(self._logsum(log_S2))
                    mean_N2 += len(data2)

                    # reset model to current and repeat
                    rd += 1
                    if rd < RD:
                        self.amp[:] = amp_[:]
                        self.mean[:,:] = mean_[:,:]
                        self.covar[:,:,:] = covar_[:,:,:]
                    
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

                #print "%d\t%d\t%.4f\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%.4f" % (it, N.sum(), log_S[N].mean(), np.exp(self._logsum(log_S[N])), mean_N2 / RD, mean_log_S2 / RD, tot_S2 / RD, logL__.mean(), np.exp(self._logsum(log_S[N])) + tot_S2 / RD) 

                # because the components remain ordered, we can
                # adopt the mean of the repeated draws as new model
                self.amp = amp__.mean(axis=0) 
                self.mean = mean__.mean(axis=0) 
                self.covar = covar__.mean(axis=0)

                # update n_guess with <n_impute>
                n_guess = n_impute__.mean()

                # check new component volumes and reset sel when it grows by
                # more then 25%
                V_ = np.linalg.det(self.covar)
                changed = np.flatnonzero((V_- V)/V > 0.25)
                for c in changed:
                    sel[c] = None
                    if self.verbose:
                        print " resetting sel[%d] due to volume change" % c
                V[:] = V_[:]
                
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

    def _M(self, k, data, log_p_k, log_S, sel_k, n_points, impute=False):

        # get the relevant sums
        A_k, M_k, C_k, P_k = self._computeMSums(k, data, log_p_k, log_S, sel_k, impute=impute)
        
        # amplitude: A_k = sum_i q_ik
        self.amp[k] = A_k/n_points

        # mean: M_k = sum_i x_i q_ik
        self.mean[k,:] = M_k/A_k

        # covariance: C_k = sum_i (x_i - mu_k)^T(x_i - mu_k) q_ik
        # w/o Bayesian regularization term
        if self.w > 0:
            self.covar[k,:,:] = (C_k + self.w*np.eye(self.D)) / (A_k + 1)
        else:
            self.covar[k,:,:] /= C_k/A_k

    def _M2(self, k, data2, log_p2_k, log_S2, sel2_k, A_k, M_k, C_k, P_k, n_points):
        # get the relevant sums from data2
        n_points2 = len(data2)
        A2_k, M2_k, C2_k, P2_k = self._computeMSums(k, data2, log_p2_k, log_S2, sel2_k, impute=True)

        
        #print "\t%d\t%.4f\t%.4f\t%.4f\t%.4f" % (k, P_k, P2_k, A_k, A2_k)
                                    
        # this is now the sum_i q_ik for i in [data, data2]
        sum_i_q_k = A_k + A2_k
        
        """
        if A2_k > A_k:
            M2_k *= A_k / A2_k
            C2_k *= A_k / A2_k
            A2_k = A_k
            P2_k = P_k
        """
        
        # amplitude
        self.amp[k] = sum_i_q_k/(n_points + n_points2)
        
        # mean
        self.mean[k,:] = (M_k + M2_k)/sum_i_q_k
        
        # covariance
        # imputation correction
        frac_p_k_out = P2_k / (P_k + P2_k)
        
        # Bayesian regularization term
        if self.w > 0:
            self.covar[k,:,:] = (C_k + C2_k + self.covar[k] * frac_p_k_out + self.w*np.eye(self.D)) / (sum_i_q_k + 1)
        else:
            self.covar[k,:,:] = (C_k + C2_k + self.covar[k] * frac_p_k_out) / sum_i_q_k

    def _computeMSums(self, k, data, log_p_k, log_S, sel_k, impute=False):
        # maybe needed for later _M2 calls: P_k = sum_i p_ik
        if impute:
            P_k = np.exp(self._logsum(log_p_k))
        else:
            P_k = None
            
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


