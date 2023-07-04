# Bivariate Profiles

```@index
Pages = ["bivariate.md"]
```

```@docs
bivariate_confidenceprofiles!
```

## Methods For Finding Boundaries

```@docs
AbstractBivariateMethod
AbstractBivariateVectorMethod
bivariate_methods
IterativeBoundaryMethod
RadialMLEMethod
RadialRandomMethod
SimultaneousMethod
Fix1AxisMethod
AnalyticalEllipseMethod
ContinuationMethod
```

## Sampling Internal Points From Boundaries

```@docs
sample_bivariate_internal_points!
bivariate_hull_methods
AbstractBivariateHullMethod
ConvexHullMethod
ConcaveHullMethod
MPPHullMethod
```

## Merging Boundaries From Multiple Methods

To improve sampling performance, it may be worth finding bivariate boundaries using a combination of methods, where one method has more guaranteed boundary coverage and the other gives a more random search of interest parameter space, such as combining [`IterativeBoundaryMethod`](@ref) with [`SimultaneousMethod`](@ref). 

```@docs
CombinedBivariateMethod
combine_bivariate_boundaries!
```