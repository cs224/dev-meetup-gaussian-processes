
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
plt.show()