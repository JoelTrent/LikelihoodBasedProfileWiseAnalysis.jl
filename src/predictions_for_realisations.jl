"""
    add_error_function!(model::LikelihoodModel, errorfunction::Function)

Adds an error function, `errorfunction`, to `model`. Modifies `model` in place. Model must contain a prediction function `model.core.predictfunction` to add the error function.
    
# Requirements for `errorfunction`
- A function to generate lower and upper confidence quartiles (reference intervals) of each prediction realisation from `predictfunction`.
- Takes three arguments, `predictions`, `θ`, and `region` in that order, where `predictions` is the array of predictions generated by `predictfunction`, `θ` is the same as for `loglikefunction` and `region` is the highest density region to evaluate quartiles of the error at each prediction point. 
- The output of the function should be a lower quartile and a upper quartile array, each with the same dimensions as `predictions`.

# Predefined error functions available to use
- [`normal_error_σ_known`](@ref) 
- [`normal_error_σ_estimated`](@ref)
- [`lognormal_error_σ_known`](@ref)
- [`lognormal_error_σ_estimated`](@ref)
- [`logitnormal_error_σ_known`](@ref)
- [`logitnormal_error_σ_estimated`](@ref)
- [`poisson_error`](@ref)
"""
function add_error_function!(model::LikelihoodModel,
    errorfunction::Function)

    check_prediction_function_exists(model) || return nothing
    errorfunction(model.core.ymle, model.core.θmle, 0.95)

    corelikelihoodmodel = model.core
    model.core = @set corelikelihoodmodel.errorfunction = errorfunction

    return nothing
end

"""
    normal_error_σ_known(predictions::AbstractArray, 
        θ::AbstractVector, 
        region::Float64, 
        σ::Real)

Use a normal error model to quantify the uncertainty in the predictions of realisations, with known value of `σ`.

To use this function as the error model you must create a new function with only the first three arguments, which calls this function with a set value of `σ`.

Two equivalent examples of this specification with `σ` set to 1.3:
```julia
errorfunction(predictions, θ, region) = normal_error_σ_known(predictions, θ, region, 1.3)

function errorfunction(predictions, θ, region) 
    normal_error_σ_known(predictions, θ, region, 1.3)
end
```
"""
function normal_error_σ_known(predictions::AbstractArray, θ::AbstractVector, region::Float64, σ::Real)
    lq = predictions .+ quantile(Normal(0, σ), (1.0-region)/2.0)
    uq = predictions .+ quantile(Normal(0, σ), 1.0 - ((1.0-region)/2.0))
    return lq, uq
end

"""
    normal_error_σ_estimated(predictions::AbstractArray, 
        θ::AbstractVector, 
        region::Float64, 
        σ_θindex::Int)

Use a normal error model to quantify the uncertainty in the predictions of realisations, where `σ` is an estimated model parameter in `θ`.

To use this function as the error model you must create a new function with only the first three arguments, which calls this function with the index, `σ_θindex`, of `σ` in `θ`.

Two equivalent examples of this specification with `σ` stored at the end of the `θ` parameter vector, which has 4 elements (length 4):
```julia
errorfunction(predictions, θ, region) = normal_error_σ_estimated(predictions, θ, region, 4)

function errorfunction(predictions, θ, region) 
    normal_error_σ_estimated(predictions, θ, region, 4)
end
```
"""
function normal_error_σ_estimated(predictions::AbstractArray, θ::AbstractVector, region::Float64, σ_θindex::Int)
    lq = predictions .+ quantile(Normal(0, θ[σ_θindex]), (1.0-region)/2.0)
    uq = predictions .+ quantile(Normal(0, θ[σ_θindex]), 1.0 - ((1.0-region)/2.0))
    return lq, uq
end

"""
    lognormal_error_σ_known(predictions::AbstractArray, 
        θ::AbstractVector, 
        region::Float64, 
        σ::Real)

Use a log normal error model to quantify the uncertainty in the predictions of realisations, with known value of `σ`.

To use this function as the error model you must create a new function with only the first three arguments, which calls this function with a set value of `σ`.

Two equivalent examples of this specification with `σ` set to 1.3:
```julia
errorfunction(predictions, θ, region) = lognormal_error_σ_known(predictions, θ, region, 1.3)

function errorfunction(predictions, θ, region) 
    lognormal_error_σ_known(predictions, θ, region, 1.3)
end
```
"""
function lognormal_error_σ_known(predictions::AbstractArray, θ::AbstractVector, region::Float64, σ::Real)
    lq, uq = zeros(size(predictions)), zeros(size(predictions))

    for i in eachindex(predictions)
        dist = LogNormal(log(predictions[i]), σ)
        lq[i], uq[i] = univariate_unimodal_HDR(dist, region, NLopt.LN_BOBYQA())
    end
    return lq, uq
end

"""
    lognormal_error_σ_estimated(predictions::AbstractArray, 
        θ::AbstractVector, 
        region::Float64, 
        σ_θindex::Int)

Use a log normal error model to quantify the uncertainty in the predictions of realisations, where `σ` is an estimated model parameter in `θ`.

To use this function as the error model you must create a new function with only the first three arguments, which calls this function with the index, `σ_θindex`, of `σ` in `θ`.

Two equivalent examples of this specification with `σ` stored at the end of the `θ` parameter vector, which has 4 elements (length 4):
```julia
errorfunction(predictions, θ, region) = lognormal_error_σ_estimated(predictions, θ, region, 4)

function errorfunction(predictions, θ, region) 
    lognormal_error_σ_estimated(predictions, θ, region, 4)
end
```
"""
function lognormal_error_σ_estimated(predictions::AbstractArray, θ::AbstractVector, region::Float64, σ_θindex::Int)
    lq, uq = zeros(size(predictions)), zeros(size(predictions))

    for i in eachindex(predictions)
        dist = LogNormal(log(predictions[i]), θ[σ_θindex])
        lq[i], uq[i] = univariate_unimodal_HDR(dist, region, NLopt.LN_BOBYQA())
    end
    return lq, uq
end

"""
    logitnormal_error_σ_known(predictions::AbstractArray, 
        θ::AbstractVector, 
        region::Float64, 
        σ::Real)

Use a logit-normal error model to quantify the uncertainty in the predictions of realisations, with known value of `σ`. Predictions are required to be defined ∈ (0,1).

Output is the correct highest density region for around `σ < 1.43` (at all values of `predictions`). If `predictions` are closer to `0` or `1.0` than `0.5`, then higher values of `σ` (closer to `2`) will be acceptable as the distribution will still be unimodal. Otherwise, the distribution will not be unimodal and won't identify the correct high density regions.

To use this function as the error model you must create a new function with only the first three arguments, which calls this function with a set value of `σ`.

Two equivalent examples of this specification with `σ` set to 0.9:
```julia
errorfunction(predictions, θ, region) = logitnormal_error_σ_known(predictions, θ, region, 0.9)

function errorfunction(predictions, θ, region) 
    logitnormal_error_σ_known(predictions, θ, region, 0.9)
end
```
"""
function logitnormal_error_σ_known(predictions::AbstractArray, θ::AbstractVector, region::Float64, σ::Real)
    lq, uq = zeros(size(predictions)), zeros(size(predictions))

    for i in eachindex(predictions)
        dist = LogitNormal(logit(predictions[i]), σ)
        lq[i], uq[i] = univariate_unimodal_HDR(dist, region, NLopt.LN_BOBYQA())
    end
    return lq, uq
end

"""
    logitnormal_error_σ_estimated(predictions::AbstractArray, 
        θ::AbstractVector, 
        region::Float64, 
        σ_θindex::Int)

Use a logit-normal error model to quantify the uncertainty in the predictions of realisations, where `σ` is an estimated model parameter in `θ`. Predictions are required to be defined ∈ (0,1).

Output is the correct highest density region for around `σ < 1.43` (at all values of `predictions`). If `predictions` are closer to `0` or `1.0` than `0.5`, then higher values of `σ` (closer to `2`) will be acceptable as the distribution will still be unimodal. Otherwise, the distribution will not be unimodal and won't identify the correct high density regions.

To use this function as the error model you must create a new function with only the first three arguments, which calls this function with the index, `σ_θindex`, of `σ` in `θ`.

Two equivalent examples of this specification with `σ` stored at the end of the `θ` parameter vector, which has 4 elements (length 4):
```julia
errorfunction(predictions, θ, region) = logitnormal_error_σ_estimated(predictions, θ, region, 4)

function errorfunction(predictions, θ, region) 
    logitnormal_error_σ_estimated(predictions, θ, region, 4)
end
```
"""
function logitnormal_error_σ_estimated(predictions::AbstractArray, θ::AbstractVector, region::Float64, σ_θindex::Int)
    lq, uq = zeros(size(predictions)), zeros(size(predictions))

    for i in eachindex(predictions)
        dist = LogitNormal(logit(predictions[i]), θ[σ_θindex])
        lq[i], uq[i] = univariate_unimodal_HDR(dist, region, NLopt.LN_BOBYQA())
    end
    return lq, uq
end

"""
    poisson_error(predictions::AbstractArray, 
        θ::AbstractVector, 
        region::Float64)

Use a poisson error model to quantify the uncertainty in the predictions of realisations, where each prediction is the mean of the error model.

To use this function as the error model you don't need to create a new function, just specify this function directly.
"""
function poisson_error(predictions::AbstractArray, θ::AbstractVector, region::Float64)
    lq, uq = zeros(size(predictions)), zeros(size(predictions))

    for i in eachindex(predictions)
        dist = Poisson(predictions[i])
        lq[i], uq[i] = univariate_unimodal_HDR(dist, region, NLopt.LN_BOBYQA())
    end
    return lq, uq
end

"""
    predict_realisations(errorfunction::Function, 
        predictions::AbstractArray, 
        θ::AbstractVector,
        region::Float64)

Uses `errorfunction` to make prediction for realisations; it forms `region` reference intervals for `region` population reference intervals. Returns two arrays of the same dimension as `predictions`, one containing the lower quantile and one containing the upper quantile (the extrema) of the reference interval.
"""
function predict_realisations(errorfunction::Function, predictions::AbstractArray, θ::AbstractVector, region::Float64)
    lq, uq = errorfunction(predictions, θ, region)
    return lq, uq
end