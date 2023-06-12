```@index
Pages = ["user_interface.md"]
```

# User Interface


## Main Model Structs and Initialisation

```@docs
LikelihoodModel
CoreLikelihoodModel
EllipseMLEApprox
initialiseLikelihoodModel
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

## Parameter Profiles and Samples

### Structs

```@docs
PointsAndLogLikelihood
AbstractConfidenceStruct
UnivariateConfidenceStruct
BivariateConfidenceStruct
SampledConfidenceStruct
```

### Profile Types

Profile type is a Struct that specifies whether the profile to be taken uses the true loglikelihood function or an ellipse approximation of the loglikelihood function centred at the MLE (with optional use of parameter bounds).

```@docs
AbstractProfileType
AbstractEllipseProfileType
LogLikelihood
EllipseApprox
EllipseApproxAnalytical
```

### Univariate Profiles

```@docs
univariate_confidenceintervals!
get_points_in_interval!
```

### Bivariate Profiles

```
bivariate_confidenceprofiles!
```

#### Methods

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
ContinuationMethod, 
```

### Dimensional Samples

We can generate samples within the loglikelihood boundary at any dimension of model interest parameters.

#### Sample Types

```@docs
AbstractSampleType
UniformGridSamples
UniformRandomSamples
LatinHypercubeSamples
```

#### Full Likelihood Sampling


```@docs
full_likelihood_sample!
```

#### Dimensional Likelihood Sampling

Note: dimensional likelihood samples can be 'full' likelihood samples as well.

```@docs
dimensional_likelihood_sample!
bivariate_concave_hull
```

## Predictions From Profiles and Samples

In the event that a prediction function has not been added to the [`LikelihoodModel`](@ref) struct yet, we can add one using [`add_prediction_function!`](@ref)
```@docs
add_prediction_function!
check_prediction_function_exists
```

Then to generate predictions we can use one of three functions, depending on whether we want to generate predictions from univariate or bivariate profiles, or dimensional samples.

```@docs
generate_predictions_univariate!
generate_predictions_bivariate! 
generate_predictions_dim_samples!
```

## Plots

### Univariate Profiles

```@docs
plot_univariate_profiles
plot_univariate_profiles_comparison
```

### Bivariate Profiles and Samples

```@docs
plot_bivariate_profiles
plot_bivariate_profiles_comparison
plot_bivariate_profiles_iterativeboundary_gif
```

### Predictions

```@docs
plot_predictions_individual
plot_predictions_union
```