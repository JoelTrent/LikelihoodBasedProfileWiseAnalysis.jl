# Initialisation

```@index
Pages = ["initialisation.md"]
```

## Model Initialisation

```@docs
initialiseLikelihoodModel
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