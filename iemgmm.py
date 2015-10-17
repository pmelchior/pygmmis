#!/bin/env python

import numpy as np

class GMM:
    def __init__(self, K=1, D=1):
        self.K = K
        self.D = D
        self.amp = None
        self.mean = None
        self.covar = None
        
    def draw(self, size=1, sel_callback=None, invert_callback=False):
        # draw indices for components given amplitudes
        try:
            ind = np.random.choice(self.K, size=size, p=self.amp)
        except (FloatingPointError, ValueError):
            print self.K, size, self.amp
            raise FloatingPointError
        samples = np.empty((size, self.D))
        counter = 0
        if size > self.K:
            bc = np.bincount(ind)
            components = np.arange(ind.size)[bc > 0]
            for c in components:
                mask = ind == c
                s = mask.sum()
                samples[counter:counter+s] = np.random.multivariate_normal(self.mean[c], self.covar[c], size=s)
                counter += s
        else:
            for i in ind:
                samples[counter] = np.random.multivariate_normal(self.mean[i], self.covar[i], size=1)
                counter += 1

        # if subsample with selection is required
        if sel_callback is not None:
            sel_ = sel_callback(samples)
            if invert_callback:
                sel_ = np.invert(sel_)
            size_in = sel_.sum()
            if size_in != size:
                ssamples = self.draw(size=size-size_in, sel_callback=sel_callback, invert_callback=invert_callback)
                samples = np.concatenate((samples[sel_], ssamples))
        return samples

    def logsumLogL(self, ll):
        """Computes log of sum of likelihoods for GMM components.

        This method tries hard to avoid over- or underflow that may arise
        when computing exp(log(p(x | k)).

        See appendix A of Bovy, Hogg, Roweis (2009).

        Args:
        ll: (K, N) log-likelihoods from K calls to logL_K() with N coordinates

        Returns:
        (N, 1) of log of total likelihood

        """
        floatinfo = np.finfo('float64')
        underflow = np.log(floatinfo.tiny) - ll.min(axis=0)
        overflow = np.log(floatinfo.max) - ll.max(axis=0) - np.log(ll.shape[0])
        c = np.where(underflow < overflow, underflow, overflow)
        return np.log(np.exp(ll + c).sum(axis=0)) - c



class IEMGMM(GMM):
    
    def __init__(self, data, K=1, R=1, s=1., w=0., verbose=False, sel=None, sel_callback=None):
        self.K = K
        self.D = data.shape[1]
        self.R = R
        self.verbose = verbose
        self._fit(data, s=s, w=w, sel=sel, sel_callback=sel_callback)
                 
    def _fit(self, data, s=1., w=0, sel=None, sel_callback=None):
        amp_r = np.empty((self.R * self.K))
        mean_r = np.empty((self.R * self.K, self.D))
        covar_r = np.empty((self.R * self.K, self.D, self.D))
        self.ll = np.empty(self.R)
        for r in range(self.R):
            if self.verbose:
                print "fitting model %d..." % r
            if sel is None:
                self._run_EM(data, s=s, w=w)
            else:
                self._run_EM(data, s=s, w=w, impute=(sel==False).sum(), sel_callback=sel_callback)

            # save the current model likelihood
            self.ll[r] = self.logL(data).mean()

            # store for mixture later
            amp_r[r*self.K:(r+1)*self.K] = self.amp
            mean_r[r*self.K:(r+1)*self.K] = self.mean
            covar_r[r*self.K:(r+1)*self.K] = self.covar

        self.amp = amp_r / amp_r.sum()
        self.mean = mean_r
        self.covar = covar_r
        
    def _run_EM(self, data, s=1., w=0, impute=0, sel_callback=None, tol=1e-3):
        self.initializeModel(data, s)
        maxiter = 100
        
        # standard EM
        if impute == 0:
            it = 0
            logL0 = None
            while it < maxiter: # limit loop in case of no convergence
                amp_ = self.amp.copy()
                mean_ = self.mean.copy()
                covar_ = self.covar.copy()
                try:
                    qij = self.E(data)
                    # compute logL from E before M modifies qij
                    logL_ = self.logsumLogL(qij.T).mean()
                    self.M(data, qij, w=w)
                    if self.verbose:
                        print " iter %d: %.3f" % (it, logL_)

                    # convergence test
                    if it > 0 and logL_ - logL0 < tol:
                        break
                    else:
                        logL0 = logL_
                    it += 1
                    amp_ = self.amp.copy()
                    mean_ = self.mean.copy()
                    covar_ = self.covar.copy()
                except np.linalg.linalg.LinAlgError:
                    if self.verbose:
                        print "warning: ran into trouble, stopping fit at previous position"
                    self.amp = amp_
                    self.mean = mean_
                    self.covar = covar_
                    break
        # with imputation
        else:
            # run standard EM first, with larger tolerance
            self._run_EM(data, s=s, w=w, tol=tol*10)

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
                
                rd = 0
                while rd < RD:
                    
                    # reset model to current
                    self.amp[:] = amp_[:]
                    self.mean[:,:] = mean_[:,:]
                    self.covar[:,:,:] = covar_[:,:,:]
                    
                    try:
                        data_out = self.I(impute=impute, sel_callback=sel_callback)
                        data_ = np.concatenate((data, data_out), axis=0)

                        # perform EM on extended data
                        qij = self.E(data_)
                        logL__[rd] = self.logsumLogL(qij.T).mean()
                        self.M(data_, qij, w=w, impute=impute)

                        # save model
                        amp__[rd,:] = self.amp[:]
                        mean__[rd,:,:] = self.mean[:,:]
                        covar__[rd,:,:,:] = self.covar[:,:,:]
                        if self.verbose:
                            print "   iter %d/%d: %.3f" % (it, rd, logL__[rd])
                    except np.linalg.linalg.LinAlgError:
                        rd -= 1
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

    def initializeModel(self, data, s):
        # set model to random positions with equally sized spheres
        self.amp = np.ones(self.K)/self.K
        min_pos = data.min(axis=0)
        max_pos = data.max(axis=0)
        self.mean = min_pos + (max_pos-min_pos)*np.random.random(size=(self.K, self.D))
        self.covar = np.tile(s**2 * np.eye(self.D), (self.K,1,1))

    def logL(self, data):
        qij = self.E(data)
        return self.logsumLogL(qij.T)

    def E(self, data):
        # use amp.size instead of self.K here because repeated fits will obtain
        # probablities and logL from E, and they would otherwise only
        # experience the first model
        qij = np.empty((data.shape[0], self.amp.size))
        log2piD2 = np.log(2*np.pi)*(0.5*self.D)
        for j in xrange(self.amp.size):
            dx = data - self.mean[j]
            chi2 = np.einsum('...j,j...', dx, np.dot(np.linalg.inv(self.covar[j]), dx.T))
            # prevent tiny negative determinants to mess up
            (sign, logdet) = np.linalg.slogdet(self.covar[j])
            qij[:,j] = np.log(self.amp[j]) - log2piD2 - sign*logdet/2 - chi2/2
        return qij

    def M(self, data, qij, w=0, impute=0):
        N = data.shape[0]
        
        # log of fractional probability
        qi = self.logsumLogL(qij.T)
        for j in xrange(self.K):
            qij[:,j] -= qi
        qj = self.logsumLogL(qij)
        pj = np.exp(qj)

        # amplitude update
        self.amp = pj/N
        
        # covariance: with imputation we need add penalty term from
        # the conditional probability of drawing points from the model:
        # p_out / p * Sigma_j (i.e. fractional prob of imputed points)
        if impute == 0:
            self.covar[:,:,:] = 0
        else:
            pj_out = np.exp(self.logsumLogL(qij[-impute:]))
            self.covar *= (pj_out / pj)[:, None, None]
            
        for j in xrange(self.K):
            pi = np.exp(qij[:,j])

            # mean
            self.mean[j] = (data * pi[:,None]).sum(axis=0)/pj[j]

            # funny way of saying: for each point i, do the outer product
            # of d_m with its transpose, multiply with pi[i], and sum over i
            d_m = data - self.mean[j]
            self.covar[j] += (pi[:, None, None] * d_m[:, :, None] * d_m[:, None, :]).sum(axis=0)
            
            # Bayesian regularization term
            if w > 0:
                self.covar[j]+= w*np.eye(self.D)
                self.covar[j] /= pj[j] + 1
            else:
                self.covar[j] /= pj[j]


    def I(self, impute=0, sel_callback=None):
        # create imputation sample from the current model
        # we don know the number if missing values exactly, so
        # draw from a Poisson distribution
        n_samples = np.random.poisson(impute)
        return self.draw(size=n_samples, sel_callback=sel_callback, invert_callback=True)



            











