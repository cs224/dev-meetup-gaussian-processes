
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

class GP(object):
    @classmethod
    def kernel_bell_shape(cls, x, y, delta=1.0):
        return np.exp(-1 / 2.0 * np.power(x - y, 2) / delta)

    @classmethod
    def kernel_laplacian(cls, x, y, delta=1):
        return np.exp(-1 / 2.0 * np.abs(x - y) / delta)

    @classmethod
    def generate_kernel(cls, kernel, delta=1):
        def wrapper(*args, **kwargs):
            kwargs.update({"delta": delta})
            return kernel(*args, **kwargs)

        return wrapper

    def __init__(self, x, y, cov_f=None, R=0):
        super().__init__()
        self.x = x
        self.y = y
        self.N = len(self.x)
        self.R = R

        self.sigma = []
        self.mean = []
        self.cov_f = cov_f if cov_f else self.kernel_bell_shape
        self.setup_sigma()

    @classmethod
    def calculate_sigma(cls, x, cov_f, R=0):
        N = len(x)
        sigma = np.ones((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                cov = cov_f(x[i], x[j])
                sigma[i][j] = cov
                sigma[j][i] = cov

        sigma = sigma + R * np.eye(N)
        return sigma

    def setup_sigma(self):
        self.sigma = self.calculate_sigma(self.x, self.cov_f, self.R)

    def predict(self, x):
        cov = 1 + self.R * self.cov_f(x, x)
        sigma_1_2 = np.zeros((self.N, 1))
        for i in range(self.N):
            sigma_1_2[i] = self.cov_f(self.x[i], x)

        # SIGMA_1_2 * SIGMA_1_1.I * (Y.T -M)
        # M IS ZERO
        m_expt = (sigma_1_2.T * np.mat(self.sigma).I) * np.mat(self.y).T
        # sigma_expt = cov - (sigma_1_2.T * np.mat(self.sigma).I) * sigma_1_2
        sigma_expt = cov + self.R - (sigma_1_2.T * np.mat(self.sigma).I) * sigma_1_2
        return m_expt, sigma_expt

    @staticmethod
    def get_probability(sigma, y, R):
        multiplier = np.power(np.linalg.det(2 * np.pi * sigma), -0.5)
        return multiplier * np.exp(
            (-0.5) * (np.mat(y) * np.dot(np.mat(sigma).I, y).T))

    def optimize(self, R_list, B_list):

        def cov_f_proxy(delta, f):
            def wrapper(*args, **kwargs):
                kwargs.update({"delta": delta})
                return f(*args, **kwargs)

            return wrapper

        best = (0, 0, 0)
        history = []
        for r in R_list:
            best_beta = (0, 0)
            for b in B_list:
                sigma = gaus.calculate_sigma(self.x, cov_f_proxy(b, self.cov_f), r)
                marginal = b * float(self.get_probability(sigma, self.y, r))
                if marginal > best_beta[0]:
                    best_beta = (marginal, b)
            history.append((best_beta[0], r, best_beta[1]))
        return sorted(history)[-1], np.mat(history)



