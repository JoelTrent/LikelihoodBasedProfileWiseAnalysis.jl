# Predictions

```@index
Pages = ["predictions.md"]
```

## Adding a Prediction Function

In the event that a prediction function has not been added to the [`LikelihoodModel`](@ref) struct yet, we can add one using [`add_prediction_function!`](@ref). This prediction function is for model solutions and not the additional error we account for when predicting realisations.

```@docs
add_prediction_function!
check_prediction_function_exists
```

## Predictions for Realisations

```@docs
add_error_function!
normal_error_σ_known
normal_error_σ_estimated
lognormal_error_σ_known
lognormal_error_σ_estimated
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

```@docs
AbstractPredictionStruct
PredictionStruct
PredictionRealisationsStruct
```