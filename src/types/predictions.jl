"""
    AbstractPredictionStruct

Supertype for the predictions storage struct.

# Subtypes

[`PredictionRealisationsStruct`](@ref)

[`PredictionStruct`](@ref)
"""
abstract type AbstractPredictionStruct end

"""
    PredictionRealisationsStruct(lq::Array{<:Real}, uq::Array{<:Real}, extrema::Array{<:Real})

Struct for containing evaluated lower and upper confidence quartiles of prediction realisations corresponding to confidence profiles.

# Fields
- `lq`: array of the lower confidence quartile of the error model evaluated at each prediction point in a corresponding [`PredictionStruct`](@ref).
- `uq`: array of the upper confidence quartile of the error model evaluated at each prediction point in a corresponding [`PredictionStruct`](@ref).
- `extrema`: extrema of the `lq` and `uq` arrays.

# Supertype Hiearachy

`PredictionRealisationsStruct <: AbstractPredictionStruct <: Any`
"""
struct PredictionRealisationsStruct <: AbstractPredictionStruct
    lq::Array{<:Real}
    uq::Array{<:Real}
    extrema::Array{<:Real}

    function PredictionRealisationsStruct(lq::Array{<:Real}=zeros(0, 0),
        uq::Array{<:Real}=zeros(0, 0),
        extrema::Array{<:Real}=zeros(0, 0))
        return new(lq, uq, extrema)
    end
end

"""
    PredictionStruct(predictions::Array{Real}, extrema::Array{Real}, realisations::PredictionRealisationsStruct)

Struct for containing evaluated predictions corresponding to confidence profiles.

# Fields
- `predictions`: array of model predictions evaluated at the parameters given by a particular confidence profile parameter set. If a model has multiple response variables, it assumes that `model.core.predictfunction` stores the prediction for each variable in its columns. Values for each response variable are stored in the 3rd dimension (row=dim1, col=dim2, page/sheet=dim3). Each column corresponds to a column of the confidence profile parameter set. 
- `extrema`: extrema of the predictions array.
- `realisations`: a [`PredictionRealisationsStruct`](@ref) struct 

# Supertype Hiearachy

`PredictionStruct <: AbstractPredictionStruct <: Any`
"""
struct PredictionStruct <: AbstractPredictionStruct
    predictions::Array{<:Real}
    extrema::Array{<:Real}
    realisations::PredictionRealisationsStruct
    
    function PredictionStruct(predictions::Array{<:Real}, extrema::Array{<:Real},
            realisations::PredictionRealisationsStruct=PredictionRealisationsStruct())
        return new(predictions, extrema, realisations)
    end
end

"""
    Base.merge(a::PredictionStruct, b::PredictionStruct)

Specifies how to merge two variables with type [`PredictionStruct`](@ref).
"""
function Base.merge(a::PredictionStruct, b::PredictionStruct)
    return PredictionStruct(hcat(a.predictions, b.predictions), 
        hcat(min.(a.extrema[:,1], b.extrema[:,1]), max.(a.extrema[:,2], b.extrema[:,2])),
        merge(a.realisations, b.realisations))
end

"""
    Base.merge(a::PredictionRealisationsStruct, b::PredictionRealisationsStruct)

Specifies how to merge two variables with type [`PredictionRealisationsStruct`](@ref).
"""
function Base.merge(a::PredictionRealisationsStruct, b::PredictionRealisationsStruct)
    if isempty(a.lq) && isempty(b.lq)
        return PredictionRealisationsStruct()
    end
    if isempty(a.lq)
        return b
    end
    return a
end