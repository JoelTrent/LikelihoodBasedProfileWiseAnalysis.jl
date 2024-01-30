# Gaussian Approximation of a Binomial Distribution

This model is taken from an unreleased paper by the authors of the workflow [simpsonprofilewise2023](@cite). The Binomial distribution is defined as:

```math
    X \sim \text{B}(n,\,p),
```
where ``n`` is the number of trials and ``p`` is the probability of success. For sufficiently large ``n``, this distribution has the following Gaussian approximation:
```math
    y_i \sim p(y_i ; \theta) X \sim \mathcal{N}(np,\, \sqrt{np(1-p)}),
```
where ``np`` is the mean number of successes and ``\sqrt{np(1-p)}`` is the standard deviation of this mean. We take the model parameter vector as ``\theta = (n,p)``. 

The true parameter values are ``\theta =(100, 0.2)``. The corresponding lower and upper parameter bounds are ``a = (0.0001, 0.0001)`` and ``b = (500,1.0)``. There is no observation time as such, but we take ten samples of the Gaussian approximation under the true parameterisation, ``y^\textrm{o}_{1:I}=[21.9, 22.3, 12.8, 16.4, 16.4, 20.3, 16.2, 20.0, 19.7, 24.4]``. 

## Initial Setup 

```julia
using Random, Distributions
using PlaceholderLikelihood
```

## Model and Likelihood Function Definition

`data` is a named tuple...

```julia
distrib_θ(θ) = Normal(θ[1] * θ[2], sqrt(θ[1] * θ[2] * (1 - θ[2])))

function loglhood_θ(θ, data)
    return sum(logpdf.(distrib_θ(θ), data.samples))
end

function predictfunction_θ(θ, data, t=["n*p"]); [prod(θ)] end
```

## Initial Data and Parameter Definition

```julia
# true parameters
θ_true = [200.0, 0.2]

# Named tuple of all data required within the log-likelihood function
data = (samples=[21.9, 22.3, 12.8, 16.4, 16.4, 20.3, 16.2, 20.0, 19.7, 24.4],)

# Bounds on model parameters
lb = [0.0001, 0.0001]
ub = [500.0, 1.0]

θnames = [:n, :p]
θG = [50, 0.3]
par_magnitudes = [100, 1]
```

## LikelihoodModel Initialisation

```julia
model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes)
```

## Evaluating a Concave Boundary

This example is particularly interesting because it contains a very concave bivariate boundary - the [`IterativeBoundaryMethod`](@ref) thus becomes very appropriate to use. However, evaluting this many points may be prohibitive on higher dimensional models.

```julia
bivariate_confidenceprofiles!(model, 200, method=IterativeBoundaryMethod(3,1,1, 0.15, 1.0, use_ellipse=true))
```

## Visualising the Progress of the IterativeBoundaryMethod

We can visualise the progress of the [`IterativeBoundaryMethod`](@ref) using [`plot_bivariate_profiles_iterativeboundary_gif`](@ref).

```julia
using Plots; gr()

format = (size=(600, 400), dpi=300, title="",
    legend_position=:topright, palette=:Paired)
plot_bivariate_profiles_iterativeboundary_gif(model, 0.2, 0.2; markeralpha=0.5, color=2, save_as_separate_plots=false, format...)
```

## Coordinate Transformation

This is an example that particularly benefits from a coordinate transformation to improve the regularity of the log-likelihood. A natural log transformation is particularly appropriate. 

### Redefining Functions

Here we define the new log-likelihood and prediction functions which define the backwards mapping from our logged parameterisation to the original parameterisation. We also define a function which specifies the forward parameter transformation, from the original to the logged parameterisation.

```julia
function loglhood_Θ(Θ, data)
    return loglhood_Θ(exp.(Θ), data)
end

function predictfunctions_Θ(Θ, data, t=["n*p"]); [prod(exp.(Θ))] end

function forward_parameter_transformLog(θ)
    return log.(θ)
end
```

### Transforming Parameter Definitions

To update the parameter bounds we can use [`transformbounds_NLopt`](@ref), which solves an integer program to determine how the old bounds map to the new bounds given the specified transformation.

```julia
lb_Θ, ub_Θ = transformbounds_NLopt(forward_parameter_transformLog, lb, ub)

Θnames = [:ln_n, :ln_p]
ΘG = forward_parameter_transformLog(θG)
par_magnitudes = [2, 1]
```

### LikelihoodModel Initialisation

```julia
model = initialise_LikelihoodModel(loglhood_Θ, data, Θnames, ΘG, lb_Θ, ub_Θ, par_magnitudes)
```

### Re-evaluating the Bivariate Boundary

Re-evaluating the bivariate boundary of the log-likelihood function after the transformation reveals a much more convex shape.

```julia
bivariate_confidenceprofiles!(model, 40, method=RadialMLEMethod())

using Plots; gr()

plots = plot_bivariate_profiles(model, 0.2, 0.2, include_internal_points=true, markeralpha=0.9)
display(plots[1])
```