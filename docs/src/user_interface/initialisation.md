# Initialisation

```@index
Pages = ["initialisation.md"]
```

## Model Initialisation

To initialise a model for profile likelihood evaluation we use [`initialise_LikelihoodModel`](@ref) which returns a struct of type [`LikelihoodModel`](@ref).

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

```@docs
getMLE_ellipse_approximation!
check_ellipse_approx_exists!
EllipseMLEApprox
```

## Modifying Parameter Magnitudes and Bounds

```@docs
setmagnitudes!
setbounds!
```

## Parameter Transformations

```@docs
transformbounds
transformbounds_NLopt
```

## Optimization Settings

We can set our default optimisation settings using a [`OptimizationSettings`](@ref) struct. This will be contained within the [`CoreLikelihoodModel`](@ref) field of a [`LikelihoodModel`](@ref) and can be passed as an option to [`initialise_LikelihoodModel`](@ref).

```@docs
default_OptimizationSettings
create_OptimizationSettings
set_OptimizationSettings!
OptimizationSettings
optimise
optimise_unbounded
```