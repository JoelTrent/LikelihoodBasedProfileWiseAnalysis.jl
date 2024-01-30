# Two-Species Logistic Model

The two-species logistic model with a Gaussian data distribution [simpsonprofile2023](@cite) has the following differential equations for the population densities of the two species ``C_1(t)\geq0`` and ``C_2(t)\geq0`` :
```math
    \frac{\mathrm{d}C(t)}{\mathrm{d}t} = \lambda_1 C_1(t) \Bigg[1-\frac{S(t)}{K}\Bigg],
```
```math
    \frac{\mathrm{d}C(t)}{\mathrm{d}t} = \lambda_2 C_2(t) \Bigg[1-\frac{S(t)}{K}\Bigg] - \delta C_2(t) \Bigg[\frac{C_1(t)}{K}\Bigg],
```
where ``S(t) = C_1(t)+C_2(t)``, and the full parameter vector is given by ``\theta = (\lambda_1, \lambda_2, K, \delta, C_1(0), C_2(0), \sigma)``. The corresponding additive Gaussian data distribution, with an estimated standard deviation, has a density function for the observed data given by:
```math
    y_i \sim p(y_i ; \theta) \sim \mathcal{N}(z_i(\theta^M), \theta^\textrm{o} \mathbb{I}) \sim \mathcal{N}(z_i(\theta^M), \sigma^2_N \mathbb{I}) ,
```
where ``\theta^M = (\lambda_1, \lambda_2, K, \delta, C_1(0), C_2(0))``, ``\theta^\textrm{o} = \sigma``, ``z_i(\theta^M)=z(t_i; \theta^M) = (C_1(t_i; \theta^M), C_2(t_i; \theta^M))`` from the previous equations, meaning at each ``t_i`` we have an observation of both ``C_1(t)`` and ``C_2(t)``, ``y_i^\textrm{o}=(C_{1,\,i}^\textrm{o}, C_{2,\,i}^\textrm{o})``, and ``\mathbb{I}`` is a ``2\times2`` identity matrix.

This model uses real data, so no `true' parameter values exist. Instead, the MLE values of parameters are used for coverage simulations ``\hat{\theta} =(0.00293, 0.00315, 0.00164, 78.8, 0.289, 0.0293, 1.83)``. The corresponding lower and upper parameter bounds are ``a = (0.0001, 0.0001, 0, 60, 0.01, 0.001, 0.1)`` and ``b = (0.01, 0.01, 0.01, 90, 1, 1, 3)``; the lower bounds for all the parameters apart from ``\delta`` were zero [simpsonprofile2023](@cite) but were increased slightly to increase stability. Observation times are ``t_{1:I} = (0, 769, 1140, 1488, 1876, 2233, 2602, 2889, 3213, 3621, 4028)``. Smaller nuisance parameter bounds are used for univariate profiles, although they are wider than those used in [simpsonprofile2023](@cite): ``a_{\text{nuisance},j} =\max(a_j,\, \hat{\theta}_j\div2.5 ), \, j \in 1,2,...,7`` and ``b_{\text{nuisance},j} =\min(b_j,\, \hat{\theta}_j\times2.5 ), \, j \in 1,2,...,7``. The original implementation can be found at [https://github.com/ProfMJSimpson/profile_predictions](https://github.com/ProfMJSimpson/profile_predictions).

Real observations, the MLE model trajectory and the MLE 95% population reference set under this parameterisation can be seen in the figure below:

## With Logit-Normal Data Distribution

If we instead use a more statistically realistic logit-normal distribution, defined on (0,1), instead of an additive Gaussian data distribution, the density function for the observed data becomes:
```math
    y_i \sim \text{LogitNormal}(\text{logit}(z_i(\theta^M)), \sigma^2\mathbb{I}).
```
where ``\theta^M = (\lambda_1, \lambda_2, K, \delta, C_1(0), C_2(0))``, ``\theta^\textrm{o} = \sigma`` and ``\text{logit}(p)=\log(p\div (1-p))``. The model trajectory, ``z_i(\theta^M)``, is assumed to be a proportion ``\in (0,1)``.

The 'true' parameter values used for coverage simulations are similar to the MLE values of the parameters when using this data distribution, albeit with a lower value of ``\sigma``: ``\theta = (0.003, 0.0004, 0.0004, 80.0, 0.4, 1.2, 0.1)``. Parameter bounds have been adjusted slightly to ``a = (0.0005, 0.00001, 0.00001, 60, 0.01, 0.1, 0.01)`` and `b = (0.01, 0.005, 0.005, 98, 2, 3, 1)`` and no 'special' nuisance parameter bounds are specified.

For coverage testing of predictive quantities using the sampled full parameter confidence set, we use much more well-informed parameter bounds that may be overly constrained: ``a_\text{sampling} = (0.0022, 0.00001, 0.0001, 73, 0.25, 0.7, 0.03)`` and ``b_\text{sampling} = (0.0036, 0.001, 0.0009, 85, 0.65, 2, 0.2)``.

Example observations, the true model trajectory and the 95% population reference set under this parameterisation can be seen in the figure below:

##