```@index
Pages = ["initialisation.md"]
```

# Initialisation

```@docs
initialiseLikelihoodModel
LikelihoodModel
CoreLikelihoodModel
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