```@meta
CollapsedDocStrings = true
```
# Univariate Profiles

The key function for evaluating univariate profiles (the parameter confidence interval and corresponding points along the profile as desired) is [`univariate_confidenceintervals!`](@ref).  The evaluated univariate profile(s) will be contained within a [`UnivariateConfidenceStruct`](@ref) that is stored in the [`LikelihoodModel`](@ref).

```@docs
univariate_confidenceintervals!
get_points_in_intervals!
get_uni_confidence_intervals
get_uni_confidence_interval
get_uni_confidence_interval_points
```

## Index

```@index
Pages = ["univariate.md"]
```