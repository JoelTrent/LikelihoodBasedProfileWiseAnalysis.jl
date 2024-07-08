```@meta
CollapsedDocStrings = true
```
# Initialisation

## Model Initialisation

To initialise a model for profile likelihood evaluation we use [`initialise_LikelihoodModel`](@ref) which returns a struct of type [`LikelihoodModel`](@ref). This struct contains all the information we require for the PWA workflow and will also contain computed profiles and profile-wise predictions.

```@docs
initialise_LikelihoodModel
```

## Model Representation

```@docs
LikelihoodModel
CoreLikelihoodModel
BaseLikelihoodModel
```

## Ellipse Approximation

In order to evaluate the ellipse approximation of the normalised log-likelihood function we need to evaluate the observed Fisher information matrix (FIM) [pawitanall2001](@cite). The observed FIM is a quadratic approximation of the curvature of the log-likelihood function at the maximum likelihood estimate (MLE) for parameters. This then allows us to define the ellipse approximation of the log-likelihood function, represented as the [`EllipseApproxAnalytical`](@ref) and [`EllipseApprox`](@ref) profile types. These have corresponding equations to evaluate their approximation to the log-likelihood function, as given by [`LikelihoodBasedProfileWiseAnalysis.analytic_ellipse_loglike`](@ref) and [`LikelihoodBasedProfileWiseAnalysis.ellipse_loglike`](@ref), respectively.

```@docs
getMLE_ellipse_approximation!
check_ellipse_approx_exists!
EllipseMLEApprox
```

## Modifying Parameter Magnitudes and Bounds

If desired, the defined parameter magnitudes and bounds contained with a [`LikelihoodModel`](@ref) can be updated using [`setmagnitudes!`](@ref) and [`setbounds!`](@ref).

```@docs
setmagnitudes!
setbounds!
```

## Parameter Transformations

To assist with parameter transformations we provide functions for transforming parameter bounds given a monotonic forward mapping from the original parameter space to the transformed parameter space.

```@docs
transformbounds
transformbounds_NLopt
```

### Example Usage

For example, if we want to log transform the parameterisation given a defined log-likelihood function and lower and upper bounds we:

Calculate the new, log-transformed bounds using the forward transformation `log.(θ)`. This is easy enough to be done without [`transformbounds_NLopt`](@ref), but other more complex transformations may not be.

```julia
lb, ub = ..., ...

f(x) = log.(x)
lb_log, ub_log = transformbounds_NLopt(f, lb, ub)
```

Define a new log-likelihood function which uses the backward transformation `exp.(θ)` from the transformed parameter space to the original space.

```julia
function loglike(θ, data)
    ...
end

function loglike_log(Θ, data)
    return loglike(exp.(Θ), data)
end
```

## Optimization Settings

We can set our default optimisation settings using a [`OptimizationSettings`](@ref) struct. This will be contained within the [`CoreLikelihoodModel`](@ref) field of a [`LikelihoodModel`](@ref) and can be passed as an option to [`initialise_LikelihoodModel`](@ref). Unless different ones are passed to functions for computing profiles, they will also be used for that purpose. It may be useful to compute the MLE using conservative settings for accuracy, and then use less conservative settings for the optimisation of nuisance parameters along parameter profiles.

```@docs
default_OptimizationSettings
create_OptimizationSettings
set_OptimizationSettings!
OptimizationSettings
optimise
optimise_unbounded
```

## Index

```@index
Pages = ["initialisation.md"]
```