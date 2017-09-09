# dev-meetup-gaussian-processes

This dev-meetup is split in several smaller notebooks to keep each one of the notebooks focused.

* [From Linear Regression to Gaussian Processes](https://nbviewer.jupyter.org/github/cs224/dev-meetup-gaussian-processes/blob/master/from_linear_regression_to_gaussian_processes.ipynb)
explains the weight-space view of Gaussian Processes. It also explains what a kernel or 
covariance function is and how you arrive at this concept relatively naturally once you introduce
projections of inputs into feature space.
* [Sampling from Gaussian Process by Hand](https://nbviewer.jupyter.org/github/cs224/dev-meetup-gaussian-processes/blob/master/sampling_from_gaussian_by_hand.ipynb) shows how
to sample functions from a gaussian process in a similar way like you would sample numbers from
a normal probability distribution.
* [Gaussian Process Parameter Effects](https://nbviewer.jupyter.org/github/cs224/dev-meetup-gaussian-processes/blob/master/gaussian_process_parameter_effects.ipynb) shows what effects
the modification of the hyper parameters of the convariance funcion (a.k.a. kernel) has on the recovered
regression line.
* [CO2 Mauna Loa Gaussian Process Regression](https://nbviewer.jupyter.org/github/cs224/dev-meetup-gaussian-processes/blob/master/co2-mauna-loa-gaussian-process-regression.ipynb)
shows how to perform the decomposition of additive effects to explain parts of the overall model.
