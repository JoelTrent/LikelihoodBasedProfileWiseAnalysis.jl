# Predictions

```@index
Pages = ["predictions.md"]
```

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

## Structs

```@docs
AbstractPredictionStruct
PredictionStruct
```