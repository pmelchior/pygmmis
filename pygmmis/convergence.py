import logging

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import t as studentT


def line(x, m, c):
    return (m*x) + c

def fit_line(x, y, yerr=None):
    """returns [m, c], dist"""
    p, pcov = curve_fit(line, x, y, None, yerr, absolute_sigma=True)
    return p, pcov, np.sqrt(np.diag(pcov))


def pvalue2sigma(pvalue):
    return norm(0, 1).ppf(1 - (pvalue/2))

def sigma2pvalue(sigma):
    return norm(0, 1).cdf(-abs(sigma)) * 2


class ConvergenceDetector(object):
    def __init__(self, tolerance, significance, burnin):
        self.tolerance = tolerance
        self.significance = significance
        self.pvalue = sigma2pvalue(significance)
        self.burnin = burnin
        self.last_check = False


    def test_convergence(self, backend):
        n = len(backend)
        if n <= self.burnin:
            return False, (np.mean(backend), np.std(backend)), (None, None)
        (significant_gradient, big_gradient), mu_std, info = self._convergence_test(backend)
        gradient, pvalue = info
        mu, std = mu_std
        # better_than_initial = backend[-1] > backend[0] - self.tolerance
        # if not better_than_initial:
        #     logging.debug("{}-{}: Still decreasing from starting point")
        #     return False, info
        if (not big_gradient) and significant_gradient:
            if not self.last_check:
                logging.info("{}-{}: Gradient is significant but flat within {}, double checking".format(self.burnin, n, self.tolerance))
                self.last_check = True
                return False, mu_std, info
            else:
                logging.info("{}-{}: Converged within {}".format(self.burnin, n, self.tolerance))
                return True, mu_std, info
        if (not significant_gradient):
            if not self.last_check:
                logging.debug("{}-{}: Gradient {} is not significant (p={} >= {}), probably converged, double checking".format(self.burnin, n, gradient, pvalue, self.pvalue))
                self.last_check = True
                return False, mu_std, info
            else:
                logging.debug( "{}-{}: Double checked, gradient {} is not significant (p={} >= {}), converged".format(self.burnin, n, gradient, pvalue, self.pvalue))
                return True, mu_std, info

        if significant_gradient and big_gradient:
            logging.debug("{}-{}: Gradient {} is significant (p={}) and not flat, keep going".format(self.burnin, n, gradient, pvalue))
            self.burnin = len(backend)
            self.last_check = False
            return False, mu_std, info


    def _convergence_test(self, backend):
        """
        :return: tuple: (converged, result_is_significant), (gradient, pvalue)
        """
        array = backend[self.burnin:]
        n = len(array)
        x = np.arange(len(array))
        p, pcov, perr = fit_line(x, array)
        gradient = p[0]
        model = line(x, *p)
        gradient_err = np.sqrt(np.sum((model - array)**2) / (n-2) / np.sum((x - x.mean())**2))

        tstat = gradient / gradient_err
        pvalue = (1 - studentT.cdf(abs(tstat), df=n-2)) * 2.  # two-sided t-test
        significant_gradient = pvalue < self.pvalue
        big_gradient = abs(gradient) > self.tolerance
        return (significant_gradient, big_gradient), (np.mean(array), np.std(array)), (gradient, pvalue)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(11)

    arr = np.load('tests/loglike.npy')
    arr = np.concatenate([arr, np.random.normal(arr[-10:].mean(), arr[-10:].std(), size=200)])
    # arr = np.random.normal(-13.4, 2, size=200)
    x = np.arange(len(arr))
    plt.plot(x, arr)

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    detector = ConvergenceDetector(tolerance=0, significance=1, burnin=0)

    # will detect "convergence" in a normal dist when significance~0.29
    # noise of std=1, requires same significance to detect


    step = 10
    for i in range(step, len(arr)+step, step):
        converged, info = detector.test_convergence(arr[:i])
        if converged:
            break
    plt.axvline(i)

