# Background

Here we cover the background / general idea of the PWA workflow as seen in [simpsonprofilewise2023](@cite). We also introduce the idea of reference tolerance sets for predictions of realisations/observations within the workflow as formally introduced in my [Masters thesis](https://github.com/JoelTrent/UoA_MastersWorking).

This is a discussion of the workflow formulation from my [Masters thesis](https://github.com/JoelTrent/UoA_MastersWorking). Discussions which formed the basis of this section, minus reference tolerance intervals are in [simpsonprofilewise2023](@cite) and [murphyimplementing2023](@cite). It is a little notation heavy, but hopefully it helps explain the sets and intervals available in the PWA workflow.

This summary uses models of the form 'deterministic mathematical model + error model' as in [simpsonprofilewise2023](@cite) and [murphyimplementing2023](@cite), but it can also be used with stochastic models [simpsonreliable2022](@cite).

## Observed Data

Within the PWA workflow, observed data ``y_i^\textrm{o}`` is measured at discrete time points ``t_i``. The 'o' superscript distinguishes the observed data from the random variable ``y`` that generated the data. Given ``I`` observations, where ``i=1,2,3,..., I``, observed data is collected into the vector, ``y_{1:I}^\textrm{o}``, which corresponds to the time points ``t_{1:I}``. For multiple observations from the same time point and model component distinct indices in ``1:I`` are used. For an observation of multiple model components at the same time point, we will use the same index in ``1:I``, such that, e.g. ``y_{i}^\textrm{o} = (x_i^\textrm{o}, y_i^\textrm{o})``.

## Mechanistic Mathematical Model

Deterministic mechanistic models take the form:
```math
    z = f(\theta^M, t).
```

where ``\theta^M`` is a vector of mechanistic model parameters, and ``z`` is a scalar or vector of model solutions containing no error. Model solutions evaluated at discrete time points ``t_i`` are defined as ``z_i(\theta^M) = z(t_i; \theta^M)``. This is also referred to as the model trajectory. In the same way as observations, ``z_{1:I}(\theta^M)`` is a vector of the model solution evaluated at ``t_{1:I}``, while ``z(\theta^M)`` represents the continuous model solution. 

## Data Distribution Parameters

The PWA workflow assumes that the observable data distribution can be characterised by a data/statistical model with parameters ``\phi(\theta)``. The data distribution can be considered to come from an underlying mechanistic model with measurement error [simpsonprofilewise2023](@cite), [murphyimplementing2023](@cite).

If we assume that the observed data come from an underlying mechanistic model with measurement error, then we let: 
```math
    \phi(\theta) = (z(\theta^M), \theta^\textrm{o}),
```
where typically (and here) the model solution is used as the mean or median parameter of the data distribution (error model) and ``\theta^\textrm{o}`` is any additional observation parameters like the observation error standard deviation. This gives us the full parameter vector, ``\theta = (\theta^M, \theta^\textrm{o})``, for the mechanistic model and error model. For simplicity, the additional observation parameters, ``\theta^\textrm{o}``, may be specified and regarded as known [simpsonprofilewise2023](@cite). However, unless they are truly known (as in, e.g. a synthetic coverage experiment), the likelihood function is known as an estimated likelihood function because it does not account for the uncertainty in ``\theta^\textrm{o}`` [pawitanall2001](@cite).

At time points ``t_i`` we define:
```math
    \phi_i(\theta) = (z_i(\theta^M), \theta^\textrm{o}),
```
where ``z_i(\theta^M)`` is the mechanistic model solution at observation ``i``.

In general, given data distribution parameters, we obtain a density function for the observed data ``y`` dependent on parameters, ``\theta``, of the form:
```math
    y \sim p(y;\theta) = p(y;\phi(\theta)).
```

When measured at time point ``t_i``, this becomes:
```math
    y_i \sim p(y_i;\theta) = p(y_i;\phi_i(\theta)).
```

When the data distribution parameters are for an error model such as the additive Gaussian, log-normal and logit-normal models, the density function for ``y`` can be represented in the following ways.

For the additive Gaussian model, this is represented as [murphyimplementing2023](@cite):
```math
    y_i \sim p(y_i ; \theta) \sim \mathcal{N}(\phi_i(\theta)) \sim \mathcal{N}(z_i(\theta^M), \theta^\textrm{o}) \sim \mathcal{N}(z_i(\theta^M), \sigma^2_N),
```
where ``\theta^\textrm{o} = \sigma_N``. This is equivalent to representing the observed data as the model trajectory plus error from a normal distribution with zero mean and variance, ``\sigma^2_N``.

For the log-normal model, this is represented as [murphyimplementing2023](@cite):
```math
    y_i \sim \text{LogNormal}(\log(z_i(\theta^M)), \sigma^2_L),
```
where ``\theta^\textrm{o} = \sigma_L``.

The logit-normal model is represented similarly to the log-normal model, noting that the mechanistic model solution is a proportion defined ``\in (0,1)``:
```math
    y_i \sim \text{LogitNormal}(\text{logit}(z_i(\theta^M)), \sigma^2_{L}),
```
where ``\theta^\textrm{o} = \sigma_L`` and ``\text{logit}(p)=\log(p\div (1-p))``.

## Likelihood Function

If we have a vector of independent observations ``y_{1:I}^\textrm{o}`` from our density function, ``y``, which is a function of parameter ``\theta``, we can define the normalised likelihood function, ``\hat{\mathcal{L}}(\theta ; y_{1:I}^{\textrm{o}})``, and in particular the normalised log-likelihood function, ``\hat{\ell} \left(\theta \, ; \, y_{1:I}^{\textrm{o}}\right)``. The 'hat', ``\hat{}``, on ``\mathcal{L}`` and ``\ell`` is used to represent that the functions are normalised.

The normalised likelihood function is [sprottstatistical2008](@cite):
```math
    \hat{\mathcal{L}}(\theta ; y_{1:I}^{\textrm{o}}) =
    \frac{p(y_{1:I}^{\textrm{o}} ; \phi(\theta) )}{\sup\limits_{\theta} \hspace{0.1cm} p(y_{1:I}^{\textrm{o}};\phi(\theta))}.
```
where the numerator is the likelihood function evaluated for data distribution parameters ``\phi(\theta)`` at observations ``y_{1:I}^{\textrm{o}}`` and the denominator is the maximum likelihood estimate (MLE) of the likelihood function as ``\theta`` varies. The MLE will be estimated using numerical optimisation, with bounds on parameters, which have value ``\hat{\theta}`` at the MLE.

Here, we will work with the normalised log-likelihood function of the form:
```math
\begin{equation}
\begin{split}
    \hat{\ell} \left(\theta \, ; \, y_{1:I}^{\textrm{o}}\right) &= \log \hat{\mathcal{L}}(\theta ; y_{1:I}^{\textrm{o}})\\
    &= \log p(y_{1:I}^{\textrm{o}} ; \phi(\theta) ) - \sup_{\theta} \log p(y_{1:I}^{\textrm{o}} ; \phi(\theta) )\\
    &= \sum_{i=1}^{I} \log p(y_i^{\textrm{o}} ; \phi(\theta) )- \sup_{\theta}\sum_{i=1}^{I} \log p(y_i^{\textrm{o}} ; \phi(\theta)).
\end{split}
\end{equation}
```

Normalisation of the log-likelihood function means that ``\hat{\ell} \left(\theta \, ; \, y_{1:I}^{\textrm{o}}\right) \leq 0`` and ``\hat{\ell} (\hat{\theta} \, ; \, y_{1:I}^{\textrm{o}}) = 0``.

We generally recommend using the `loglikelihood` function implemented in [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) to determine the value of the log-likelihood function, as in our examples. This is straightforward to compute for the error models described in [Data Distribution Parameters](@ref) given observed data ``y_{1:I}^\textrm{o}``.

## Profile Likelihood Function

Given a likelihood function, we can define a profile log-likelihood function as a function of interest parameters, ``\psi``, and nuisance parameters, ``\omega``. These parameters represent a partitioning of the parameter vector ``\theta = (\psi, \omega)``. The normalised profile log-likelihood function is then:

```math
    \hat{\ell}_p \left(\psi \, ; \, y_{1:I}^{\textrm{o}}\right) = \sup_{\omega \, | \, \psi} \hat{\ell}_p \left(\psi, \, \omega \, ; \, y_{1:I}^{\textrm{o}}\right),
```
where for given values of the interest parameter ``\psi``, the values of ``\omega`` are optimised out, meaning they are set to the values that maximise the function. We expect that this function will be continuous for the models considered.

## Confidence Sets For Parameters

Using the log-likelihood function, we define approximate likelihood-based confidence sets for the full parameter vector ``\theta`` [pawitanall2001](@cite), [sprottstatistical2008](@cite):
```math
    \mathcal{C}_{\theta, 1-\alpha}(y_{1:I}^{\textrm{o}}) = \{ \theta \, | \, \hat{\ell} \left(\theta \, ; \, y_{1:I}^{\textrm{o}}\right) \geq \ell_c \},
```
where ``\ell_c`` is a threshold chosen so that the approximate coverage of the confidence interval is ``1-\alpha``. For sufficiently regular problems (see Section \ref{sssec:loglikelihood_approx}) the threshold is calibrated using the chi-square distribution. Regular means that the likelihood function is well approximated by a quadratic function around the MLE [pawitanall2001](@cite):
```math
\ell_c = - \frac{\Delta_{\nu, 1-\alpha}}{2},
```
where ``\Delta_{\nu, 1-\alpha}`` is the ``1-\alpha`` quantile of the ``\chi^2`` distribution with ``\nu`` degrees of freedom. We will refer to these as full parameter confidence sets. For full parameter confidence sets, we set ``\nu`` equal to the number of parameters, ``|\theta|``. 

Profile likelihood-based confidence sets for the interest parameter(s) ``\psi`` take the form:
```math
    \mathcal{C}^\psi_{\theta, 1-\alpha}(y_{1:I}^{\textrm{o}}) = \{ \theta=(\psi, \omega) \,| \, \hat{\ell}_p \left(\psi \, ; \, y_{1:I}^{\textrm{o}}\right) \geq \ell_c \},
```
where we instead set ``\nu`` equal to the dimensionality of the interest parameters (e.g. for confidence sets with a single interest parameter ``\nu=1`` and with two interest parameters ``\nu=2``). We also record the optimised out values of nuisance parameters, ``\omega``, in the confidence set for the interest parameter. Recording these nuisance parameters allows this set to be propagated forward into predictive quantities; they do not have to be recorded otherwise. Simultaneous profile likelihood-based confidence sets for interest parameters can be obtained by setting ``\nu=|\theta|`` [rauestructural2009](@cite). 

We refer to profile likelihood-based confidence sets for one and two interest parameters as univariate profiles and bivariate profiles, respectively. We let the labels 'univariate' or 'bivariate' relate to profiles formed using ``\nu`` equal to the dimensionality of the interest parameter, ``|\psi|``. If the profiles are instead created using the simultaneous asymptotic threshold with ``\nu=|\theta|``, they will be referred to as 'simultaneous univariate' or 'simultaneous bivariate' profiles. Setting ``\nu`` to any other value may not have a clear statistical interpretation.

## Confidence Sets For Data Distribution Parameters

By propagating forward full parameter confidence sets and profile likelihood-based confidence sets using the mapping ``\phi(\theta)``, we define approximate likelihood-based confidence sets for the data distribution parameters ``\phi``. We refer to the data distribution parameters, ``\phi``, as 'predictive' quantities [simpsonprofilewise2023](@cite). We use square brackets in the following equations to show that the confidence set for data distribution parameters is the set image of the parameter confidence set under the mapping ``\phi(\theta)``. 

For the likelihood-based confidence set for data distribution parameters from full parameter confidence sets, this is defined as:
```math
    \mathcal{C}_{\phi,1-\alpha}(y_{1:I}^\textrm{o}) = \{\phi[\mathcal{C}_{\theta,1-\alpha}(y_{1:I}^\textrm{o})]\} = \{\phi(\theta) \ \vert\ \ \theta \in \mathcal{C}_{\theta,1-\alpha}(y_{1:I}^\textrm{o})\}.
```

This set definition implies that the confidence set ``\mathcal{C}_{\phi,1-\alpha}(y_{1:I}^\textrm{o})`` has at least a coverage of ``1-\alpha`` (is conservative) given the following relationship [simpsonprofilewise2023](@cite):
```math
    \theta \in \mathcal{C}_{\theta,1-\alpha} \implies \phi(\theta) \in \mathcal{C}_{\phi,1-\alpha}.
```

For a profile likelihood-based confidence set, a profile-wise confidence set, this is defined as:
```math
    \mathcal{C}_{\phi,1-\alpha}^\psi(y_{1:I}^\textrm{o}) = \{\phi[\mathcal{C}^\psi_{\theta,1-\alpha}(y_{1:I}^\textrm{o})]\} = \{\phi(\theta) \ \vert\ \theta \in \mathcal{C}^\psi_{\theta,1-\alpha}(y_{1:I}^\textrm{o})\}.
```

More conservative profile-wise confidence sets for data distribution parameters can be formed by taking the union of individual profile confidence sets:
```math
    \mathcal{C}_{\phi,1-\alpha} \approx \bigcup_\psi \mathcal{C}^\psi_{\phi,1-\alpha}.
```

Additionally, we can obtain more conservative profile-wise confidence sets for data distribution parameters by forming simultaneous profile confidence sets with ``\nu=|\theta|`` rather than ``\nu=|\psi|``.

When considered in a mechanistic model context, this allows us to form a trajectory confidence set for the mechanistic model solution, which is typically treated as the mean or median data distribution parameter [simpsonprofilewise2023](@cite), [murphyimplementing2023](@cite). If we wish to form a trajectory confidence set for the mechanistic model solution, we consider just that component of the data distribution confidence set. In this case, if the parameter confidence set has the correct coverage properties over all parameters simultaneously, we expect the trajectory confidence set, ``\mathcal{C}_{\phi,1-\alpha}``, to display _curvewise_ (simultaneous) coverage properties, where the true model solution is fully contained within the confidence set [murphyimplementing2023](@cite).  

We refer to likelihood-based confidence sets for the model trajectory from full parameter confidence sets as full trajectory confidence sets. Similarly, we refer to the sets for the data distribution parameters as full data distribution confidence sets. Additionally, we refer to profile-wise confidence sets for the model trajectory from univariate and bivariate profiles as profile-wise trajectory confidence sets. 

## Reference Tolerance Sets For Observed Data

W define ``(1-\delta, 1-\alpha)`` reference tolerance sets for observed data given ``1-\delta`` population reference sets [wrightcalculating1999](@cite), [katkiassessing2005](@cite). For more background, please see my [Masters thesis](https://github.com/JoelTrent/UoA_MastersWorking). This is in contrast to more traditional prediction sets for observed data.

The ``1-\delta`` population reference interval refers to an interval that contains ``1-\delta`` of the population (i.e. of observations) [wrightcalculating1999](@cite), [katkiassessing2005](@cite). These population reference intervals are also 'predictive' quantities. Here, a ``1-\delta`` population reference set refers to the set of ``1-\delta`` population reference intervals across time points, ``t_j`` (and similarly for reference tolerance sets and reference tolerance intervals). In general, we will take a ``1-\delta`` population reference interval to be given by the ``1-\delta`` highest density region [hyndmancomputing1996](@cite) of population observations at ``t_j``, as this allows the interval to represent a 'typical' observation best. For the error models discussed in [Data Distribution Parameters](@ref) we use [UnivariateUnimodalHighestDensityRegion.jl](https://joeltrent.github.io/UnivariateUnimodalHighestDensityRegion.jl/stable/) as seen in [Predefined Error models](@ref).

Similarly, a ``(1-\delta, 1-\alpha)`` reference tolerance interval is a tolerance interval that contains the ``1-\delta`` reference interval with probability ``1-\alpha``. We refer to these as reference tolerance intervals unless the ``(1-\delta, 1-\alpha)`` designation is important for clarity. These intervals can be used as approximate prediction intervals; they appear to be appropriate for trapping at least ``1-\delta`` of observations with confidence ``1-\alpha``, which is a weaker condition than trapping the ``1-\delta`` population reference interval. As our notation suggests, a reference tolerance interval can be formed that has a coverage property at a different confidence level to the size of the reference interval (i.e. ``\delta \neq \alpha``).

Therefore, given a desired confidence level ``1-\alpha``, we form a ``1-\alpha`` likelihood-based confidence set for data distribution parameters, ``\mathcal{C}_{\phi,1-\alpha}``, from the full parameter confidence set. Then, for each ``\phi`` in the data distribution parameter set we construct a ``1-\delta`` reference set, ``\mathcal{A}_{y,1-\delta}(y^\textrm{o}_{1:I})``, where ``\phi`` is related to ``y`` as in the density function equation in [Data Distribution Parameters](@ref). This reference set is constructed by taking a ``1-\delta`` region of the data distribution. For symmetric data distributions, we take the ``\delta/2`` and ``1-\delta/2`` quantiles of the probability distribution. For asymmetric distributions, we take the ``1-\delta`` highest density region [hyndmancomputing1996](@cite). If the asymmetric distribution is unimodal we can use [UnivariateUnimodalHighestDensityRegion.jl](https://joeltrent.github.io/UnivariateUnimodalHighestDensityRegion.jl/stable/) to evaluate the highest density region. We then take the union across the reference sets formed from each ``\phi`` to obtain ``(1-\delta, 1-\alpha)`` reference tolerance sets for observed data, ``y^\textrm{o}``, from full parameter confidence sets:
```math
    \mathcal{C}_{y, (1-\delta, 1-\alpha)}(y^\textrm{o}_{1:I}) \approx \bigcup_{\phi \ \in \ \mathcal{C}_{\phi,1-\alpha}(y^\textrm{o}_{1:I})} \mathcal{A}_{y,1-\delta}(y^\textrm{o}_{1:I}).
```
Because each ``\mathcal{A}_{y,1-\delta}(y^\textrm{o}_{1:I})`` can only be guaranteed to contain the population reference set if it was derived using the true parameter values, we refer to these only as _reference sets_. Similarly, because ``\mathcal{C}_{y,(1-\delta, 1-\alpha)}(y^\textrm{o}_{1:I})`` is obtained by taking the union over each reference set, it is much more likely that one of these was obtained from the true parameter values. Hence, we refer to these as _reference tolerance sets_. Usefully, if the data distribution parameter confidence set, ``\mathcal{C}_{\phi,1-\alpha}``, has curvewise coverage properties then we also expect the reference tolerance set, ``\mathcal{C}_{y,(1-\delta, 1-\alpha)}(y^\textrm{o}_{1:I})``, to have curvewise coverage properties. We refer to these as full reference tolerance sets.

We do the same thing for profile-wise reference tolerance sets, beginning from profile-wise data distribution confidence sets:
```math
    \mathcal{C}^\psi_{y,(1-\delta, 1-\alpha)}(y^\textrm{o}_{1:I}) \approx \bigcup_{\phi \ \in \ \mathcal{C}^\psi_{\phi,1-\alpha}(y^\textrm{o}_{1:I})} \mathcal{A}^\psi_{1-\delta}(y^\textrm{o}_{1:I}).
```

More conservative profile-wise reference tolerance sets can again be formed by taking the union of reference tolerance sets from individual profiles:
```math
    \mathcal{C}_{y,(1-\delta, 1-\alpha)} \approx \bigcup_\psi \mathcal{C}^\psi_{y,(1-\delta, 1-\alpha)}.
```
We refer to these as profile-wise reference tolerance sets.

### Full Reference Tolerance Set Coverage

The coverage of the full reference tolerance set in the PWA workflow is straightforward to prove. The ``(1-\delta, 1-\alpha)`` reference tolerance set is constructed from the ``1-\alpha`` confidence set for data distribution parameters, which is constructed from the ``1-\alpha`` confidence set for parameters. Here let ``1-\delta = 1-\alpha = 0.95 = 95\%``. If a full parameter confidence set has 95% coverage and the model is well-specified, then 95% of the time it will trap the true parameters. Therefore, the full data distribution confidence set will also have 95% coverage. Additionally, the true data distribution parameters give the true ``1-\delta`` reference interval for the population at each time point. Hence, if the full data distribution confidence set has 95% coverage, then our full reference tolerance set will contain the 95% reference set _at least_ 95% of the time. 

In the mechanistic model case, if the full data distribution confidence set has curvewise coverage of the model solution, then our full reference tolerance set will also have curvewise coverage of the population reference set. That is, 95% of the full reference tolerance sets constructed in this fashion under repeated sampling of new data will contain the 95% population reference set (all 95% population reference intervals). A similar idea to what we demonstrate here, but in a different context, is found in [sattenupper1995](@cite). 

For example, consider a single parameter scalar model, ``z(\theta^M) = \theta^M``, from which we have obtained ``I=100`` observations corrupted by i.i.d. Gaussian noise ``\sim \mathcal{N}(0,\theta^\textrm{o})``. Let ``\theta^M=2`` and ``\theta^\textrm{o}=1``. The density function for observations is represented as:
```math
    y \sim p(y ; \theta)  \sim \mathcal{N}(\phi(\theta)) \sim \mathcal{N}(z(\theta^M), \theta^\textrm{o}) \sim \mathcal{N}(z(\theta^M), \sigma),
```
where ``\theta^\textrm{o}=\sigma`` and ``\theta = (\theta^M, \theta^\textrm{o})``. Resultantly, our parameter vector, ``\theta``, has two parameters. 

The 95% population reference interval can be formed by considering the 2.5% and 97.5% quantiles of this density function under the true parameterisation, ``\theta =[0.040, 3.960]``. Then, given the ``I`` observations we form a 95% confidence set for ``\theta``, ``\mathcal{C}_{\theta, 0.95}``. The mapping of ``\theta`` onto the data distribution parameters ``\phi`` is 1:1, hence we also have a 95% confidence set for ``\phi``, ``\mathcal{C}_{\phi, 0.95}``. We then use each ``\phi \in \mathcal{C}_{\phi, 0.95}`` to form 95\% reference intervals, ``\mathcal{A}_{y, 0.95}``. We then take the union across each of these reference intervals to form (95%, 95%) reference tolerance intervals, ``\mathcal{C}_{y, (0.95,0.95)}``. If the true parameterisation is in ``\mathcal{C}_{\theta, 0.95}`` with 95% coverage, then it is also in ``\mathcal{C}_{\phi, 0.95}`` with 95% coverage and our full reference tolerance interval, ``\mathcal{C}_{y, (0.95, 0.95)}``, will also contain the _at least_ 95% reference interval with _at least_ 95% coverage. 

The 'at least' statement for both the reference interval and the coverage of this interval occurs for the following reasons. For the coverage statement, if ``\theta \notin \mathcal{C}_{\theta, 0.95}``, then it is possible that we predict the population reference interval using the incorrect parameter or similarly from the union of reference intervals from many incorrect parameters. Similarly, for the reference interval statement, if ``\theta \in \mathcal{C}_{\theta, 0.95}``, then our 95% reference interval from that ``\theta`` is the 95% population reference interval. Other parameter values in the parameter confidence set may predict reference intervals outside of this range. Hence, the union of these intervals will contain at least the 95% population reference interval.

Code to visualise and test this example is seen below:

#### Setup

```julia
using Random, Distributions
using LaTeXStrings
using LikelihoodBasedProfileWiseAnalysis

θ_true = [2,1]
true_dist = Normal(θ_true[1], θ_true[2])
n = 100
Random.seed!(3)
y_obs = rand(true_dist, n) 

ref_interval = quantile(true_dist, [0.025, 0.975])

data = (y_obs=y_obs, dist=Normal(0, θ_true[2]), t=["z"])

function lnlike(θ, data)
    return sum(loglikelihood(Normal(0, θ[2]), data.y_obs .- θ[1]))
end

function predictfunction(θ, data, t=["z"])
    return [θ[1]*1.0]
end

errorfunction(a,b,c) = normal_error_σ_estimated(a,b,c, 2)
```

#### Parameter Confidence Set Evaluation

```julia
model = initialise_LikelihoodModel(lnlike, predictfunction, errorfunction, data, [:μ, :σ], [2.,1.], [-1., 0.01], [5., 5.], [1.,1.]);

univariate_confidenceintervals!(model, num_points_in_interval=300)
dimensional_likelihood_samples!(model, 2, 1000000)
```

#### Confidence Set Plots

```julia
using Plots; gr()
format=(size=(400,400), dpi=300, title="", legend_position=:topright)

plt = plot_univariate_profiles(model; format...)
vline!(plt[1], [θ_true[1]], label=L"\theta^M", xlabel=L"\theta^M", lw=2, linestyle=:dash)
vline!(plt[2], [θ_true[2]], label=L"\theta^\textrm{o}", xlabel=L"\theta^M", lw=2, linestyle=:dash)
display(plt[1])
display(plt[2])

plt = plot_bivariate_profiles(model; for_dim_samples=true, markeralpha=0.4, max_internal_points=10000, ylabel=latexstring("\\theta^\\textrm{o}"), xlabel=latexstring("\\theta^M"), format...)
scatter!(plt[1], [θ_true[1]], [θ_true[2]], label="θtrue", color="black", ms=5, msw=0)
display(plt[1])
```

#### Profile-Wise Reference Intervals and Reference Tolerance Intervals From Univariate Profiles

```julia
generate_predictions_univariate!(model, ["z"], 1.0)
lq = model.uni_predictions_dict[1].realisations.lq
uq = model.uni_predictions_dict[1].realisations.uq

plt = plot(1:length(lq), transpose(uq); xlabel="Confidence Set Sample", ylabel="Interval", label="Upper", palette=:Paired_6, format...)
plot!(1:length(lq), transpose(lq), label="Lower")

lq = model.uni_predictions_dict[2].realisations.lq
uq = model.uni_predictions_dict[2].realisations.uq

plt = plot(1:length(lq), transpose(uq); xlabel="Confidence Set Sample", ylabel="Interval", label="Upper", palette=:Paired_6, format...)
plot!(1:length(lq), transpose(lq), label="Lower")

extrema1 = model.uni_predictions_dict[1].realisations.extrema
extrema2 = model.uni_predictions_dict[2].realisations.extrema
extrema = [min(extrema1[1],extrema2[1]) max(extrema1[2], extrema2[2])]

using StatsPlots
plt = plot(true_dist; xlabel=latexstring("y"), label="Density", fill=(0, 0.3), palette=:Paired_6, format...)
vline!(ref_interval, label="Reference", lw=2)
vline!(transpose(extrema1), label="Tolerance, "*L"\psi=\theta^M", linestyle=:dash, lw=2)
vline!(transpose(extrema2), label="Tolerance, "*L"\psi=\theta^\textrm{o}", linestyle=:dash, lw=3)
vline!(transpose(extrema), label="Tolerance, union", linestyle=:dashdot, lw=2, alpha=0.7)
```

#### Full Reference Tolerance Interval

```julia
generate_predictions_dim_samples!(model, ["z"], 1.0)
lq = model.dim_predictions_dict[1].realisations.lq
uq = model.dim_predictions_dict[1].realisations.uq
extrema=model.dim_predictions_dict[1].realisations.extrema

plt = plot(1:length(lq), transpose(uq); xlabel="Confidence Set Sample", ylabel="Interval", label="Upper", palette=:Paired_6, format...)
plot!(1:length(lq), transpose(lq), label="Lower")

plt = plot(true_dist; xlabel=latexstring("y"),label="Density", fill=(0, 0.3), palette=:Paired_6, format...)
vline!(ref_interval, label="Reference", lw=2)
vline!(transpose(extrema), label="Tolerance", linestyle=:dash, lw=2)
```