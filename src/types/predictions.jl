"""
    AbstractPredictionStruct

Supertype for the predictions storage struct.

# Subtypes

[`PredictionStruct`](@ref)
"""
abstract type AbstractPredictionStruct end

"""
    PredictionStruct(predictions::Array{Real}, extrema::Array{Real})

Struct for containing evaluated predictions corresponding to confidence profiles.

# Fields
- `predictions`: array of model predictions evaluated at the parameters given by a particular confidence profile parameter set. If a model has multiple predictive variables, it assumes that `model.predictfunction` stores the prediction for each variable in its columns. We are going to store values for each variable in the 3rd dimension (row=dim1, col=dim2, page/sheet=dim3). Each column corresponds to a column of the confidence profile parameter set. 
- `extrema`: extrema of the predictions array.

# Supertype Hiearachy

`PredictionStruct <: AbstractPredictionStruct <: Any`
"""
struct PredictionStruct <: AbstractPredictionStruct
    predictions::Array{<:Real}
    extrema::Array{<:Real}
end

"""
    Base.merge(a::PredictionStruct, b::PredictionStruct)

Specifies how to merge two variables with type [`PredictionStruct`](@ref).
"""
function Base.merge(a::PredictionStruct, b::PredictionStruct)
    return PredictionStruct(hcat(a.predictions, b.predictions), 
        hcat(min.(a.extrema[:,1], b.extrema[:,1]), max.(a.extrema[:,2], b.extrema[:,2])))
end