# Bivariate Profiles

```@index
Pages = ["bivariate.md"]
```

```@docs
bivariate_confidenceprofiles!
```

## Methods For Finding Boundaries

We provide several heuristics for evaluating the boundaries of bivariate profiles; they're listed in order of computational efficiency. Available methods can be checked using [`bivariate_methods`](@ref). We recommend evaluating up to 50 points; 10-30 should be appropriate for getting a reasonable approximation of a bivariate boundary, particularly if it is relatively elliptical. Note: [`AnalyticalEllipseMethod`] exclusively works with the [`EllipseApproxAnalytical`] profile type; it is also highly recommended if that profile type is of interest.

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

In order to cheaply sample interval points within the found boundary of a bivariate profile we use [`sample_bivariate_internal_points!`](@ref).

```@docs
sample_bivariate_internal_points!
bivariate_hull_methods
AbstractBivariateHullMethod
ConvexHullMethod
ConcaveHullMethod
MPPHullMethod
```

## Merging Boundaries From Multiple Methods

To improve the performance of internal point sampling, it may be worth finding bivariate boundaries using a combination of methods, where one method has more guaranteed boundary coverage and the other gives a more random search of interest parameter space, such as combining [`IterativeBoundaryMethod`](@ref) with [`SimultaneousMethod`](@ref) into a [`CombinedBivariateMethod`](@ref). 

```@docs
CombinedBivariateMethod
combine_bivariate_boundaries!
```