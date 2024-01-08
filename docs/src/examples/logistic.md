# Logistic Model

The logistic model with a normal data distribution [simpsonprofilewise2023](@cite) has the following differential equation for the population density ``C(t)\geq0``:
```math
\frac{\mathrm{d}C(t)}{\mathrm{d}t} = \lambda C(t) \Bigg[1-\frac{C(t)}{K}\Bigg],
```
where the model parameter vector is given by ``\theta^M = (\lambda, K, C(0))``. The corresponding additive Gaussian data distribution, with a fixed standard deviation, has a density function for the observed data given by:
```math
y_i \sim p(y_i ; \theta) \sim \mathcal{N}(z_i(\theta^M), \sigma^2_N),
```
where ``z_i(\theta^M)=z(t_i; \theta^M)`` is the model solution of the first Equation at ``t_i`` and ``\sigma=10``.

The true parameter values are ``\theta^M =(0.01, 100, 10)``. The corresponding lower and upper parameter bounds are ``a = (0, 50, 0)`` and ``b = (0.05,150,50)``. Observation times are ``t_{1:I} = 0,100,200,...,1000``. The original implementation can be found at [https://github.com/ProfMJSimpson/Workflow](https://github.com/ProfMJSimpson/Workflow). Example realisations, the true model trajectory and 95% population reference set under this parameterisation can be seen in the Figure below:

<!-- ![](assets/figures/.png) -->

## Initial Setup

Here we add three worker processes, which matches the number of univariate and bivariate profiles. For coverage testing we recommend setting this number as discussed in [Import Package and Set Up Distributed Environment](@ref). 

```julia
using Distributed
if nprocs()==1; addprocs(3, env=["JULIA_NUM_THREADS"=>"1"]) end
@everywhere using Random, Distributions
@everywhere using PlaceholderLikelihood
```

## Model and Likelihood Function Definition

`data` is a named tuple...

```julia
@everywhere function solvedmodel(t, θ)
    return (θ[2]*θ[3]) ./ ((θ[2]-θ[3]) .* (exp.(-θ[1] .* t)) .+ θ[3])
end

@everywhere function loglhood(θ, data)
    y=solvedmodel(data.t, θ)
    e=sum(loglikelihood(data.dist, data.y_obs .- y))
    return e
end
```

## Initial Data and Parameter Definition

```julia
# true parameters
λ_true=0.01; K_true=100.0; C0_true=10.0; t=0:100:1000; 
@everywhere global σ=10.0;
θ_true=[λ_true, K_true, C0_true]
y_true = solvedmodel(t, θ_true)
y_obs = [19.27, 20.14, 37.23, 74.87, 88.51, 82.91, 123.88, 103.25, 78.89, 87.87, 113.0]

# Named tuple of all data required within the log-likelihood function
data = (y_obs=y_obs, t=t, dist=Normal(0, σ))

# Bounds on model parameters 
λ_min, λ_max = (0.00, 0.05)
K_min, K_max = (50., 150.)
C0_min, C0_max = (0.0, 50.)
lb = [λ_min, K_min, C0_min]
ub = [λ_max, K_max, C0_max]

θnames = [:λ, :K, :C0]
θG = θ_true
par_magnitudes = [0.005, 10, 10]
```

## LikelihoodModel Initialisation

Here we choose to set some optimization settings, `opt_settings`, which are used when determining the maximum likelihood estimate ``\hat{\theta}``. If different settings are not provided to functions for profiling, then these settings (which are now contained in the [`LikelihoodModel`](@ref)), will be used.

```julia
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5))
model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings);
```

## Full Parameter Vector Confidence Set Evaluation

To evaluate the full parameter vector confidence set at a 95% confidence level we use:

```julia
full_likelihood_sample!!(model, 30000)
```

## Profiling

### Univariate Profiles

To find the confidence intervals for all three parameters at a 95% confidence level (the default), we use:

```julia
univariate_confidenceintervals!(model)
```

If we instead wish to find these intervals at a 99% confidence interval we use:

```julia
univariate_confidenceintervals!(model, confidence_level=0.99)
```

Similarly, if we wish to find simultaneous 95% confidence intervals for the parameters we set the degrees of freedom to the number of model parameters (instead of `1`).

```julia
univariate_confidenceintervals!(model, dof=model.core.num_pars) # model.core.num_pars=3
```

To find asymptotic confidence intervals using the ellipse approximation, we change the specified profile type to [`EllipseApproxAnalytical`](@ref) or [`EllipseApprox`](@ref). When parameter constraints are not in the way these will produce the same result for well-identified models:

```julia
univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
univariate_confidenceintervals!(model, profile_type=EllipseApprox())
```

If we want to visualise the univariate profiles across the range defined by each confidence interval then we need to evaluate points inside each interval. We can also evaluate some points to the left and right of each interval to observe the behaviour of the profile log-likelihood function outside of this range:

```julia
get_points_in_intervals!(model, 20, additional_width=0.2)
```

This can also be done within [`univariate_confidenceintervals`](@ref) using the `num_points_in_interval` and `additional_width` keyword arguments.

### Bivariate Profiles

To evaluate the bivariate boundaries for all three bivariate parameter combinations, here we use the [`IterativeBoundaryMethod`](@ref), which uses a 20 point ellipse approximation of the boundary as a starting guess using [`RadialMLEMethod`](@ref). The boundaries in this example are reasonably convex, which makes this starting guess appropriate. To speed up computation we provide stronger optimization settings.

```julia
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
bivariate_confidenceprofiles!(model, 50, method=IterativeBoundaryMethod(20, 5, 5, 0.15, 1.0, use_ellipse=true), optimizationsettings=opt_settings)
```

Similarly, to evaluate the analytical ellipse boundaries using [EllipseSampling](https://joeltrent.github.io/EllipseSampling.jl/stable) we use:

```julia
bivariate_confidenceprofiles!(model, 50, profile_type=EllipseApproxAnalytical(), method=AnalyticalEllipseMethod(0.15, 1.0))
```

To efficiently sample 100 points within the bivariate boundaries using a rejection sampling approach we use:

```julia
sample_bivariate_internal_points!(model, 100)
```

### Plots of Profiles

To visualise plots of these profiles we load [Plots](https://docs.juliaplots.org/stable/) alongside a plotting backend. Here we use [GR](https://github.com/jheinen/GR.jl).

```julia
using Plots; gr()
```

Univariate and bivariate profiles can either be visualised individually or in comparison to profiles at the same confidence level and degrees of freedom. 

Here we compare the univariate profiles formed at a 95% confidence level.
```julia
plts = plot_univariate_profiles_comparison(model, confidence_level=0.95)

plt = plot(plts..., layout=(1,3))
display(plt)
```

Similarly, here we compare the bivariate profiles formed at a 95% confidence level.
```julia
plts = plot_bivariate_profiles_comparison(model, confidence_level=0.95)

plt = plot(plts..., layout=(1,3))
display(plt)
```

## Predictions

To make predictions for the model trajectory and ``1-\delta`` population reference set we define the following functions, which then need to be added to our [`LikelihoodModel`](@ref). These could also be added in [`initialise_LikelihoodModel`](@ref).

```julia
@everywhere function predictFunc(θ, data, t=data.t); solvedmodel(t, θ) end
@everywhere function errorFunc(predictions, θ, region); normal_error_σ_known(predictions, θ, region, σ) end

add_prediction_function!(model, predictFunc)
add_error_function!(model, errorFunc)
```

To generate profile-wise predictions for each of the evaluated profiles we first define the desired time points for prediction and then evaluate the approximate model trajectory confidence sets and ``(1-\delta, 1-\alpha)`` population reference tolerance sets. By default the population reference tolerance set evaluates reference interval regions at the same level as the confidence level (``1-\delta = 1-\alpha``); however, this is not required.

```julia
t_pred=0:5:1000

generate_predictions_univariate!(model, t_pred)
generate_predictions_bivariate!(model, t_pred)
generate_predictions_dim_samples!(model, t_pred) # for the full likelihood sample
```

### Plotting Predictions

We can plot the predictions of individual profiles or the union of all profiles at a given number of interest parameters, confidence level, degrees of freedom and reference interval region (if relevant). When plotting the union of these predictions we can compare it to the result of the full likelihood sample, which here used [`LatinHypercubeSamples`](@ref), the default.

```julia
using Plots; gr()
plot_predictions_union(model, t_pred, 1, compare_to_full_sample_type=LatinHypercubeSamples()) # univariate profiles
```

```julia
plot_predictions_union(model, t_pred, 2, compare_to_full_sample_type=LatinHypercubeSamples()) # bivariate profiles
```

## Coverage Testing

### Data Generation

```julia
# DATA GENERATION FUNCTION AND ARGUMENTS
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    y_obs = generator_args.y_true .+ rand(generator_args.dist, length(generator_args.t))
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

@everywhere function reference_set_generator(θtrue, generator_args::NamedTuple, confidence_level::Float64)
    lq, uq = errorFunc(generator_args.y_true, θtrue, confidence_level)
    return (lq, uq)
end

training_gen_args = (y_true=y_true, t=t, dist=Normal(0, σ), is_test_set=false)
testing_gen_args = (y_true=solvedmodel(t_pred, θ_true), t=t_pred, dist=Normal(0, σ), is_test_set=true)
```