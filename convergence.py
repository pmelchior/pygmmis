from warnings import warn

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

# def _gradient_significance(dist, tol=0):
#     """prob gradient within tolerance, prob of gradient > tol, prob gradient < tol"""
#     return dist.cdf(tol) - dist.cdf(-tol), 1 - dist.cdf(tol), dist.cdf(-tol)

def studentt_test(p, perr, tolerance, n):
    """
    Returns p value for a measured p +- perr with n measurements
    """
    T = studentT(n - 2)
    upper_tstat = (p - tolerance) / perr #/ np.sqrt(n)
    lower_tstat = (p + tolerance) / perr #/ np.sqrt(n)
    upper_p = (1 - T.cdf(abs(upper_tstat)))
    lower_p = (1 - T.cdf(abs(lower_tstat)))
    return lower_p, upper_p

def pvalue2sigma(pvalue):
    return norm(0, 1).ppf(1 - (pvalue/2))

def sigma2pvalue(sigma):
    return norm(0, 1).cdf(sigma) * 2

def gradient_significance(x, y, yerr):
    p, pcov, perr = fit_line(x, y, yerr)
    # pwithin, pabove, pbelow =_gradient_significance(dist, tolerance)
    # return pwithin, pabove, pbelow, p[0]
    return studentt_test(p[0], perr[0], len(x))


class ConvergenceDetector(object):
    def __init__(self, sigma=3, tolerance=0):
        self.tolerance = tolerance
        self.sigma = sigma
        self.p = norm(0, 1).cdf(sigma)

    def gradient_significance(self, y, yerr):
        x = np.arange(len(y))
        return gradient_significance(x, y, yerr, self.tolerance)

    def test_not_increasing(self, values, errors=0):
        values, errors = np.atleast_1d(values, errors)
        err = np.sqrt(errors[-1]**2 + errors[0]**2)
        if ((values[-1] - values[0]) / err) > self.sigma:
            return False, (None, None, None), 0, (values[-1] - values[0]) / len(values)

        test_condition = (values[-1]) < (values[:-1])
        try:
            where = np.where(test_condition)[0][0]
        except IndexError:
            where = 0

        within, above, below, m = self.gradient_significance(values[where:], errors[where:])
        if (below > self.p):
            warn("Significant downward trend detected")
        converged = above < self.p
        return converged, (within, above, below), where, m


    # def convergence_probability(self, values, errors=0, fractional_step=0.1):
    #     vs = np.atleast_1d(values)
    #     es = np.zeros_like(values)
    #     es[:] = errors
    #     step = int(len(vs) * fractional_step)
    #     steps = range(step, len(vs), step)
    #     return np.asarray([self.test_not_increasing(vs[s-step:s], es[s-step:s])[1][0] for s in steps])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # arr = np.load('tests/loglike.npy')[80:]
    arr = np.random.normal(0, 1, size=1000)
    x = np.arange(len(arr))
    arr += (x*0.0001)
    p, pcov, perr = fit_line(x, arr)

    plt.plot(x, arr)
    plt.plot(x, line(x, *p))

    n = len(x)
    tscore = p[0] / perr[0] #* np.sqrt(n)
    pvalue = 1 - studentT.cdf(tscore, df=n-2)
    sigma = pvalue2sigma(pvalue)

