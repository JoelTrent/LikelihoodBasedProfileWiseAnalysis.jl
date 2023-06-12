```@index
Pages = ["profile_structs.md"]
```

# Structs and Profile Types

## Structs

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