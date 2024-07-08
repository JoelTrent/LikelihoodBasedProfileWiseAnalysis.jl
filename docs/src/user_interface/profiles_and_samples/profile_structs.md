```@meta
CollapsedDocStrings = true
```
# Structs and Profile Types

## Structs

The following structs are used to contain information on computed univariate, bivariate and sampled dimensional profiles. After evaluation they are stored in the [`LikelihoodModel`](@ref).

```@docs
PointsAndLogLikelihood
AbstractConfidenceStruct
UnivariateConfidenceStruct
BivariateConfidenceStruct
SampledConfidenceStruct
```

## Profile Types

Profile type is a Struct that specifies whether the profile to be taken uses the true loglikelihood function or an ellipse approximation of the loglikelihood function centred at the MLE (with optional use of parameter bounds).

```@docs
AbstractProfileType
AbstractEllipseProfileType
LogLikelihood
EllipseApprox
EllipseApproxAnalytical
```

## Index

```@index
Pages = ["profile_structs.md"]
```