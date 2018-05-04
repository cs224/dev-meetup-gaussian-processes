
# https://youtrack.jetbrains.com/issue/PY-26546 import pandas causes console crash with signal 11: SIGSEGV
# https://youtrack.jetbrains.com/issue/PY-29882 import pandas causes console crash with signal 11: SIGSEGV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns, gpflow
import statsmodels.api as sm

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# pd.set_option('display.float_format', lambda x: '%.2f' % x)
np.set_printoptions(edgeitems=10)
np.set_printoptions(suppress=True)
np.core.arrayprint._line_width = 180


mauna_loa_atmospheric_CO2_concentration_data = sm.datasets.get_rdataset("CO2")

data1 = mauna_loa_atmospheric_CO2_concentration_data.data
data1.head()

x = np.array(data1.time)
y = np.array(data1.value)
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

import bokeh.plotting, bokeh.models, bokeh.io, bokeh.palettes
p = bokeh.plotting.figure(title="Fit to the Mauna Loa Data", #x_axis_type='datetime',
                          plot_width=900, plot_height=600)
p.yaxis.axis_label = 'CO2 [ppm]'
p.xaxis.axis_label = 'Date'

# true value
p.circle(x, y, color="black", legend="Observed data")
p.legend.location = "top_left"
bokeh.plotting.show(p)

plt.plot(x,y)
# plt.show()

k1 = gpflow.kernels.RBF(1, variance=(66.0 ** 2), lengthscales=67.0)
k2_exp_sine_squred_gamma= 2.0 / 1.3 ** 2.0
k2_exp_sine_squred_period = 1.0
k2 = gpflow.kernels.RBF(1, variance=(2.4 ** 2.0), lengthscales=90.0) * gpflow.kernels.Periodic(1, period=k2_exp_sine_squred_period, variance=1.0,  lengthscales=1.0/k2_exp_sine_squred_gamma)
# k3 = how to do a rational quadratic term in GPflow?
k4 = gpflow.kernels.RBF(1, variance=(0.18 ** 2), lengthscales=1.6) + gpflow.kernels.White(1, variance=0.19)
kernel = k1 + k2 + k4

m = gpflow.models.GPR(X, Y, kern=kernel)
m.likelihood.variance = 0.01
m.as_pandas_table()

opt = gpflow.train.ScipyOptimizer()
opt.minimize(m)

m.as_pandas_table()

k1.as_pandas_table()

# k1.compute_K()
# k1.compute_K_symm

import functools

def conditional2(x_new, x, y, cov, sigma_n=0):
    if not isinstance(cov, list):
        cov = [cov]

    if len(cov) < 2:
        total_covariance_function = cov[0]
    else:
        # total_covariance_function = None
        total_covariance_function = functools.reduce(lambda a, x: a + x, cov)

    A = total_covariance_function.compute_K_symm(x_new)
    C = total_covariance_function.compute_K(x_new, x)
    B = total_covariance_function.compute_K_symm(x) + np.power(sigma_n,2)*np.diag(np.ones(len(x)))

    mu = [np.linalg.inv(B).dot(C.T).T.dot(y).squeeze()]
    sigma = [(A - C.dot(np.linalg.inv(B).dot(C.T))).squeeze()]

    for i in range(0, len(cov)):
        partial_covariance_function = cov[i]
        C_ = partial_covariance_function.compute_K(x_new, x)
        mu_ = np.linalg.inv(B).dot(C_.T).T.dot(y).squeeze()
        mu.append(mu_)

    return (mu, sigma)

def predict2(x_new, x, y, cov, sigma_n=0):
    l_y_pred, l_sigmas = conditional2(x_new, x, y, cov=cov, sigma_n=sigma_n)
    if len(l_sigmas[0].shape) > 1:
        return l_y_pred, [np.diagonal(ls) for ls in l_sigmas]
    else:
        return l_y_pred, l_sigmas

xx = np.linspace(1990, 2000, 1000)
XX = xx.reshape(-1, 1)
# y_pred, sigmas = predict2(xx, x, y, cov=[k1, k2, k4], sigma_n=np.sqrt(m.likelihood.variance.value))
y_pred, sigmas = predict2(XX, X, Y, cov=[k1, k2, k4], sigma_n=np.sqrt(m.likelihood.variance.value))
# y_pred[0]
# len(y_pred)
# np.allclose()