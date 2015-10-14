#!/bin/env python

import numpy as np
#np.seterr(all='raise')
class IEMGMM:
    def __init__(self, K=1, D=1, R=1):
        self.K = K
        self.D = D
        self.R = R
        self.amp = None
        self.mean = None
        self.covar = None
        
    def fit(self, data, s=1., sel=None, sel_callback=None):
        amp_ = None
        mean_ = None
        covar_ = None
        for r in range(self.R):
            print r
            if sel is None:
                self.run_EM(data, s=s)
            else:
                self.run_EM(data, s=s, impute=(sel==False).sum(), sel_callback=sel_callback)
            if amp_ is None:
                amp_ = self.amp.copy()
                mean_ = self.mean.copy()
                covar_ = self.covar.copy()
            else:
                amp_ = np.concatenate((amp_, self.amp))
                mean_ = np.concatenate((mean_, self.mean), axis=0)
                covar_ = np.concatenate((covar_, self.covar), axis=0)
                
        self.amp = amp_ / amp_.sum()
        self.mean = mean_
        self.covar = covar_
        self.K *= self.R # need to tell model that it has repeated runs

    def run_EM(self, data, s=1., impute=0, sel_callback=None, tol=1e-3):
        self.initializeModel(s)
        maxiter = 100

        # standard EM
        if impute == 0:
            it = 0
            logL0 = None
            while it < maxiter: # limit loop in case of no convergence
                try:
                    qij = self.E(data)
                    # compute logL from E before M modifies qij
                    logL_ = self.logsumLogL(qij.T).mean()
                    self.M(data, qij)
                    print " iter %d: %.3f" % (it, logL_)

                    # convergence test
                    if it > 0 and logL_ - logL0 < tol:
                        break
                    else:
                        logL0 = logL_
                except np.linalg.linalg.LinAlgError:
                    it = 0
                    self.initializeModel(s)
                it += 1
        # with imputation
        else:
            # run standard EM first
            self.run_EM(data, s=s)

            # for each iteration, draw several fake data sets
            # estimate mean and std of their logL
            # test for convergence with Welch's t-test
            # and adopt their _mean_ model for the next iteration
            it = 0
            logL0 = None
            RD = 10 # repeated draws for imputation
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
                    self.amp = amp_.copy()
                    self.mean = mean_.copy()
                    self.covar = covar_.copy()
                    
                    try:
                        data_out = self.I(impute=impute, sel_callback=sel_callback)
                        data_ = np.concatenate((data, data_out), axis=0)

                        # perform EM on extended data
                        qij = self.E(data_)
                        logL__[rd] = self.logsumLogL(qij.T).mean()
                        self.M(data_, qij, impute=impute)

                        # save model
                        amp__[rd,:] = self.amp
                        mean__[rd,:,:] = self.mean
                        covar__[rd,:,:,:] = self.covar
                        print "   iter %d/%d: %.3f" % (it, rd, logL__[rd])
                    except np.linalg.linalg.LinAlgError:
                        rd -= 1
                    rd += 1
                    
                # convergence test:
                # in principle one can do Welch's t-test wrt iteration before
                # but the actual risk here is a run-away, which
                # drastically _reduces_ the likelihood, at which point we abort
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

    def initializeModel(self, s):
        # set model to random positions with equally sized spheres
        self.amp = np.ones(self.K)/self.K
        self.mean = np.random.random(size=(self.K, self.D))
        self.covar = np.tile(s**2 * np.eye(self.D), (self.K,1,1))

    def logL(self, data):
        qij = self.E(data)
        return self.logsumLogL(qij.T)

    def E(self, data):
        qij = np.empty((data.shape[0], self.K))
        for j in xrange(self.K):
            dx = data - self.mean[j]
            chi2 = np.einsum('...j,j...', dx, np.dot(np.linalg.inv(self.covar[j]), dx.T))
            qij[:,j] = np.log(self.amp[j]) - np.log((2*np.pi)**self.D * np.linalg.det(self.covar[j]))/2 - chi2/2
        return qij

    def M(self, data, qij, impute=0):
        N = data.shape[0] - impute
        qi = self.logsumLogL(qij.T)
        for j in xrange(self.K):
            qij[:,j] -= qi
        pj = np.exp(self.logsumLogL(qij))
        if impute:
            pj_in = np.exp(self.logsumLogL(qij[:-impute]))
            pj_out = np.exp(self.logsumLogL(qij[-impute:]))
            covar_ = np.empty((self.D,self.D))

        for j in xrange(self.K):
            P_i = np.exp(qij[:,j])
            self.amp[j] = pj[j]/(N+impute)

            # do covar first since we can do this without a copy of mean here
            if impute:
                covar_[:,:] = self.covar[j]
            self.covar[j] = 0
            for i in xrange(N):
                self.covar[j] += P_i[i] * np.outer(data[i]-self.mean[j], (data[i]-self.mean[j]).T)
            if impute == 0:
                self.covar[j] /= pj[j]
            else:
                self.covar[j] /= pj_in[j]
                self.covar[j] += pj_out[j] / pj[j] * covar_

            # now update means
            for d in xrange(self.D):
                self.mean[j,d] = (data[:,d] * P_i).sum()/pj[j]

    def I(self, impute=0, sel_callback=None):
        return self.draw(size=impute, sel_callback=sel_callback, invert_callback=True)

    def draw(self, size=1, sel_callback=None, invert_callback=False):
        # draw indices for components given amplitudes
        ind = np.random.choice(self.K, size=size, p=self.amp)
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
        # typo in eq. 58: log(N) -> log(K)
        floatinfo = np.finfo('d')
        underflow = np.log(floatinfo.tiny) - ll.min(axis=0)
        overflow = np.log(floatinfo.max) - ll.max(axis=0) - np.log(self.K)
        c = np.where(underflow < overflow, underflow, overflow)
        return np.log(np.exp(ll + c).sum(axis=0)) - c


            











