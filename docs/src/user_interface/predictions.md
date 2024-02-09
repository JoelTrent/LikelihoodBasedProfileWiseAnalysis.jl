# Predictions

## Adding a Prediction Function

In the event that a prediction function has not been added to the [`LikelihoodModel`](@ref) struct yet, we can add one using [`add_prediction_function!`](@ref). This prediction function is for model solutions/trajectory and not the additional error we account for when predicting realisations.

```@docs
add_prediction_function!
check_prediction_function_exists
```

## Adding an Error Function

Similarly, if a error model function (the data distribution) has not been added to the [`LikelihoodModel`](@ref) struct yet, we can add one using [`add_error_function!`](@ref). This function is used to evaluate `region` reference intervals around the model trajectory given the data distribution defined as the error model.

```@docs
add_error_function!
```

## Predefined Error models

For convenience, we define example error model functions for four data distributions: Gaussian, log-normal, logit-normal and poisson. We provide versions where the standard deviation parameter σ is known (i.e. fixed at a given value) and where it's estimated (it should be a parameter within the parameter vector).

```@docs
normal_error_σ_known
normal_error_σ_estimated
lognormal_error_σ_known
lognormal_error_σ_estimated
logitnormal_error_σ_known
logitnormal_error_σ_estimated
poisson_error
```

## Prediction Generation

Then to generate predictions we can use one of three functions, depending on whether we want to generate predictions from univariate or bivariate profiles, or dimensional samples.

```@docs
generate_predictions_univariate!
generate_predictions_bivariate! 
generate_predictions_dim_samples!
```

## Structs

Predictions are stored in a [`PredictionStruct`](@ref) which will also contain a [`PredictionRealisationsStruct`](@ref) if the corresponding reference tolerance intervals have been evaluated.

```@docs
AbstractPredictionStruct
PredictionStruct
PredictionRealisationsStruct
```

## Index

```@index
Pages = ["predictions.md"]
```