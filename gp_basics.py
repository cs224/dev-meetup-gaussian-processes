
import numpy as np, pandas as pd
import functools

class GPCovarianceFunction:
    def covariance(self, x1, x2):
        return np.subtract.outer(x1, x2)**2

# http://www.cs.toronto.edu/~duvenaud/cookbook/index.html

class GPCovarianceFunctionWhiteNoise:
    def __init__(self, sigma):
        self.sigma = float(sigma)

    def covariance(self, x1, x2):
        lr = np.zeros(shape=(len(x1), len(x2)))
        np.fill_diagonal(lr, self.sigma ** 2)
        return lr

class GPCovarianceFunctionSquaredExponential:
    def __init__(self, l, sigma):
        self.l = float(l)
        self.sigma = float(sigma)

    def covariance(self, x1, x2):
        return (self.sigma**2) * np.exp( -0.5 * (np.subtract.outer(x1, x2)**2) / (self.l ** 2))

class GPCovarianceFunctionExpSine2Kernel:
    def __init__(self, l, period, sigma):
        self.l = float(l)
        self.period = float(period)
        self.sigma = float(sigma)

    def covariance(self, x1, x2):
        return (self.sigma**2) * np.exp( -2.0 * np.power(np.sin((np.pi / self.period) * np.abs(np.subtract.outer(x1, x2))),
                                                         2) / (self.l ** 2))

class GPCovarianceFunctionSum:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def covariance(self, x1, x2):
        s1 = self.k1.covariance(x1, x2)
        s2 = self.k2.covariance(x1, x2)
        s = s1 + s2
        return  s

class GPCovarianceFunctionProduct:
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def covariance(self, x1, x2):
        p1 = self.k1.covariance(x1, x2)
        p2 = self.k2.covariance(x1, x2)
        p = p1 * p2
        return  p

def conditional(x_new, x, y, cov, sigma_n=0):
    if not isinstance(cov, list):
        cov = [cov]

    if len(cov) < 2:
        total_covariance_function = cov[0]
    else:
        total_covariance_function = functools.reduce(lambda a, x: GPCovarianceFunctionSum(a, x), cov)

    A = total_covariance_function.covariance(x_new, x_new)
    C = total_covariance_function.covariance(x_new, x)
    B = total_covariance_function.covariance(x, x) + np.power(sigma_n,2)*np.diag(np.ones(len(x)))

    mu = [np.linalg.inv(B).dot(C.T).T.dot(y).squeeze()]
    sigma = [(A - C.dot(np.linalg.inv(B).dot(C.T))).squeeze()]

    for i in range(0, len(cov)):
        partial_covariance_function = cov[i]
        C_ = partial_covariance_function.covariance(x_new, x)
        mu_ = np.linalg.inv(B).dot(C_.T).T.dot(y).squeeze()
        mu.append(mu_)

    return (mu, sigma)

def predict(x_new, x, y, cov, sigma_n=0):
    l_y_pred, l_sigmas = conditional(x_new, x, y, cov=cov, sigma_n=sigma_n)
    if len(l_sigmas[0].shape) > 1:
        return l_y_pred, [np.diagonal(ls) for ls in l_sigmas]
    else:
        return l_y_pred, l_sigmas
