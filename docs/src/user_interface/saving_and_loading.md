# Saving and Loading LikelihoodModels

Models can be saved easily using [BSON.jl](https://github.com/JuliaIO/BSON.jl/tree/master) as they are defined using a Julia struct. 

The only thing to be careful of is ensuring that empty dataframe rows within the model are removed prior to saving using [`trim_model_dfs!`](@ref) (as they include undefined references). 

```julia
using LikelihoodBasedProfileWiseAnalysis

# function definitions and initialisation...
model = 
trim_model_dfs!(model)
using BSON: @save

@save "mymodel.bson" model
```

To load `model` in a new session/file:

```julia
using LikelihoodBasedProfileWiseAnalysis
using BSON: @load

# function definitions....

@load "mymodel.bson" model
```

## Potential Issues When Loading

There are a couple of things to watch out for if the model saved had functions defined in `model.core`:
- The log-likelihood function (and if defined, the prediction and error functions) must be defined with the same name and be available in the scope we are loading `model` in. Functions used within `model.core.optimizationsettings` must also be defined in the scope we are loading `model` in (i.e. by loading `LikelihoodBasedProfileWiseAnalysis`).
- The variable name `model` has when saved is the same name it needs to be loaded with.

## Fixing Issues

The first of these issues we can get around by converting our [`CoreLikelihoodModel`](@ref) to a [`BaseLikelihoodModel`](@ref) before saving. The only difference between these two structs is that [`BaseLikelihoodModel`](@ref) doesn't contain fields for the log-likelihood, prediction and error functions. This means we can load a saved `model` without needing those functions defined in the local scope, which may be useful for workflows where the computation is performed in one file and plotting of outputs is performed in another file.

We can use [`remove_functions_from_core!`](@ref) to perform this task, pulling out the original `model.core` so we can put it back into `model` later if desired: 

```julia
core_original = remove_functions_from_core!(model)
@save "mymodel.bson" model

# restore the core that has the functions if desired
model.core = core_original 
```

If we want to add the log-likelihood function to this loaded version of the `model` we can use [`add_loglikelihood_function!`](@ref) after loading `LikelihoodBasedProfileWiseAnalysis`.

```julia
using LikelihoodBasedProfileWiseAnalysis
# log-likelihood function function definition
function loglikefunction(θ, data); return ... end
add_loglikelihood_function!(model, loglikefunction)
```

!!! danger "Missing log-likelihood function"
    Trying to use a profile-related function will result in an error if the log-likelihood function is not defined in `model.core`.

The prediction function can be added in the same fashion using [`add_prediction_function!`](@ref). However, the log-likelihood function must have been added first.

The error function can also be added in the same fashion using [`add_error_function!`](@ref). However, the log-likelihood must have been added first.

## Saving and Loading Functions

```@docs
trim_model_dfs!
remove_functions_from_core!
add_loglikelihood_function!
```