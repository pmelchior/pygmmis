from warnings import warn
import logging
import numpy as np
from scipy.optimize import curve_fit, bisect
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
    return norm(0, 1).cdf(sigma) * 2


class ConvergenceDetector(object):
    def __init__(self, tolerance, significance, burnin):
        self.tolerance = tolerance
        self.significance = significance
        self.pvalue = 1 - norm(0, 1).cdf(significance)
        self.burnin = burnin


    def test_converged(self, backend):
        n = len(backend)
        if n <= self.burnin:
            return False

        # increasing = (self.backend[-1] - self.tolerance) > self.backend[-2]
        # decreasing = (self.backend[-1] + self.tolerance) < self.backend[-2]
        # higher_than_first = self.backend[-1] > (self.backend[self.burnin] + self.tolerance)

        # if increasing:
        #     self.burnin = len(self.backend)
        #     logging.info("Gradient does not show convergence, keep going")
        #     return False
        # elif decreasing:
        #     logging.info("LogL not increasing as it should. Bad start point? Waiting until it increases beyond the initial guess")
        # else:
        (significant_gradient, big_gradient), (gradient, pvalue) = self._convergence_test(backend)
        info = (gradient, pvalue)
        if (not big_gradient) and significant_gradient:
            logging.info("{}-{}: Converged within {}".format(self.burnin, n, self.tolerance))
            return True, info
        if (not significant_gradient) and (not big_gradient):
            logging.debug("{}-{}: Gradient {} is not significant (p={} >= {}), converged".format(self.burnin, n, gradient, pvalue, self.pvalue))
            return True, info
        if significant_gradient and big_gradient:
            logging.debug("{}-{}: Gradient {} is significant (p={}) and not flat, keep going".format(self.burnin, n, gradient, pvalue))
            self.burnin = len(backend)
            # if gradient > 0:
            #     self.burnin = len(backend)
            # else:
            #     logging.info("{}-{}: LogL not increasing as it should. Bad start point? Waiting until it increases beyond the initial guess".format(self.burnin, n))
        return False, info


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
        return (significant_gradient, big_gradient), (gradient, pvalue)


    # def estimate_critical_pvalue(self, nsteps, gradient, ):

    #
    #
    # def find_burnin_point(self):
    #     """
    #     Returns the index at which the likelihood starts to monotonically increase (within a tolerance)
    #     :return: int
    #     """
    #     try:
    #         return np.where([self.backend[:] < self.backend[0]])[0][-1]  # find the last point which is smaller than index 0
    #     except IndexError:
    #         return 0






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(11)

    arr = np.load('tests/loglike.npy')
    arr = np.concatenate([arr, np.random.normal(arr[-10:].mean(), arr[-10:].std(), size=200)])
    # arr =
    x = np.arange(len(arr))
    # arr += (x*1e-8)
    plt.plot(x, arr)


    logging.basicConfig(format='%(message)s', level=logging.INFO)
    detector = ConvergenceDetector(tolerance=1e-6, significance=3, burnin=0)

    step = 20
    for i in range(step, len(arr)+step, step):
        converged, info = detector.test_converged(arr[:i])
        if converged:
            break
    plt.axvline(i)



    #
    # n = len(x)
    # tscore = p[0] / perr[0] * np.sqrt(n)
    # pvalue = 1 - studentT.cdf(tscore, df=n-2)
    # sigma = pvalue2sigma(pvalue)
    #
