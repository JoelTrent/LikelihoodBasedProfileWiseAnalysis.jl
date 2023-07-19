# Initialisation

```@index
Pages = ["initialisation.md"]
```

## Model Initialisation

To initialise a model for profile likelihood evaluation we use [`initialiseLikelihoodModel`](@ref) which returns a struct of type [`LikelihoodModel`](@ref).

```@docs
initialiseLikelihoodModel
```

## Optimization Settings

We can set our default optimisation settings using a [`OptimizationSettings`](@ref) struct. This will be contained within [`CoreLikelihoodModel`](@ref) and can be passed as an option to [`initialiseLikelihoodModel`](@ref).

```@docs
defaultOptimizationSettings
OptimizationSettings
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

## Optimisation of Nuisance Parameters

This is presently done using a predefined NLopt optimiser. 
```@docs
optimise
optimise_unbounded
```