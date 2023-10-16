"""
    variablemappingranges(num_pars::T, index::T) where T <: Int

Returns two tuples of ranges that map variables in nuisance parameter space (ω) to their corresponding indexes in parameter space (θ) given an interest parameter ψ at `index`.
"""
function variablemappingranges(num_pars::T, index::T) where {T<:Int}
    θranges = (1:(index-1), (index+1):num_pars)
    ωranges = (1:(index-1), index:(num_pars-1))
    return θranges, ωranges
end


"""
    variablemappingranges(num_pars::T, index1::T, index2::T) where T <: Int

Returns two tuples of ranges that map variables in nuisance parameter space (ω) to their corresponding indexes in parameter space (θ), given interest parameters ψ at `index1` and `index2` and `index1 < index2`.
"""
function variablemappingranges(num_pars::T, index1::T, index2::T) where {T<:Int}
    θranges = (1:(index1-1), (index1+1):(index2-1), (index2+1):num_pars)
    ωranges = (1:(index1-1), index1:(index2-2), (index2-1):(num_pars-2))
    return θranges, ωranges
end

"""
    variablemapping!(θ::Union{Vector, SubArray}, ω::Union{Vector, SubArray}, θranges::Tuple{T, T}, ωranges::Tuple{T, T}) where T <: UnitRange

Modifies the array `θ` in place, mapping the variable values in the nuisance parameter array `ω` to their corresponding indexes in the parameter array `θ`, where the ranges are determined by [`PlaceholderLikelihood.variablemappingranges`](@ref). For one interest parameter.
"""
function variablemapping!(θ::Union{Vector, SubArray},
                            ω::Union{Vector, SubArray},
                            θranges::Tuple{T, T}, 
                            ωranges::Tuple{T, T}) where T <: UnitRange
    θ[θranges[1]] .= @view(ω[ωranges[1]])
    θ[θranges[2]] .= @view(ω[ωranges[2]])
    return θ
end

"""
    variablemapping!(θ::Union{Vector, SubArray}, ω::Union{Vector, SubArray}, θranges::Tuple{T,T,T}, ωranges::Tuple{T,T,T})) where T <: UnitRange

Modifies the array `θ` in place, mapping the variable values in the nuisance parameter array `ω` to their corresponding indexes in the parameter array `θ`, where the ranges are determined by [`PlaceholderLikelihood.variablemappingranges`](@ref). For two interest parameters.
"""
function variablemapping!(θ::Union{Vector, SubArray},
                            ω::Union{Vector, SubArray},
                            θranges::Tuple{T,T,T},
                            ωranges::Tuple{T,T,T}) where {T<:UnitRange}
    θ[θranges[1]] .= @view(ω[ωranges[1]])
    θ[θranges[2]] .= @view(ω[ωranges[2]])
    θ[θranges[3]] .= @view(ω[ωranges[3]])
    return θ
end

"""
    boundsmapping!(newbounds::Vector, bounds::AbstractVector, index::Int)

Modifies `newbounds` in place, mapping all the values in `bounds` in order into `newbounds` with the exception of the value at `index`, which is the interest parameter.
"""
function boundsmapping!(newbounds::Vector, bounds::AbstractVector, index::Int)
    newbounds[1:(index-1)] .= @view(bounds[1:(index-1)])
    newbounds[index:end]   .= @view(bounds[(index+1):end])
    return nothing
end

"""
    boundsmapping!(newbounds::Vector, bounds::AbstractVector, index1::Int, index2::Int)

Modifies `newbounds` in place, mapping all the values in `bounds` in order into `newbounds` with the exception of the values at `index1` and `index2`, which are the interest parameters and `index1 < index2`.
"""
function boundsmapping!(newbounds::Vector, bounds::Union{AbstractVector, SubArray}, index1::Int, index2::Int)
    newbounds[1:(index1-1)]      .= @view(bounds[1:(index1-1)])
    newbounds[index1:(index2-2)] .= @view(bounds[(index1+1):(index2-1)])
    newbounds[(index2-1):end]    .= @view(bounds[(index2+1):end])
    return nothing
end

"""
    init_nuisance_parameters(model::LikelihoodModel, index::Int)

Initialises the lower and upper bounds, and initial guess for nuisance parameters using [`PlaceholderLikelihood.boundsmapping!`](@ref) and ranges that map variables between nuisance parameter and parameter space using [`PlaceholderLikelihood.variablemappingranges`](@ref), given an interest parameter at `index`. The initial guess for nuisance parameters is their corresponding value at the maximum likelihood estimate (`model.core.θmle`).
"""
function init_nuisance_parameters(model::LikelihoodModel, index::Int, θlb_nuisance::AbstractVector{<:Float64}, θub_nuisance::AbstractVector{<:Float64})
    newLb     = zeros(model.core.num_pars-1) 
    newUb     = zeros(model.core.num_pars-1)
    initGuess = zeros(model.core.num_pars-1)

    boundsmapping!(newLb, θlb_nuisance, index)
    boundsmapping!(newUb, θub_nuisance, index)
    boundsmapping!(initGuess, model.core.θmle, index)

    θranges, ωranges = variablemappingranges(model.core.num_pars, index)

    return newLb, newUb, initGuess, θranges, ωranges
end

"""
    init_nuisance_parameters(model::LikelihoodModel, index::Int)

Initialises the lower and upper bounds, and initial guess for nuisance parameters using [`PlaceholderLikelihood.boundsmapping!`](@ref) and ranges that map variables between nuisance parameter and parameter space using [`PlaceholderLikelihood.variablemappingranges`](@ref), given interest parameters at `index1` and `index2` where `index1 < index2`. The initial guess for nuisance parameters is their corresponding value at the maximum likelihood estimate (`model.core.θmle`).
"""
function init_nuisance_parameters(model::LikelihoodModel, index1::Int, index2::Int, θlb_nuisance::AbstractVector{<:Float64}, θub_nuisance::AbstractVector{<:Float64})
    newLb     = zeros(model.core.num_pars - 2)
    newUb     = zeros(model.core.num_pars - 2)
    initGuess = zeros(model.core.num_pars - 2)

    boundsmapping!(newLb, θlb_nuisance, index1, index2)
    boundsmapping!(newUb, θub_nuisance, index1, index2)
    boundsmapping!(initGuess, model.core.θmle, index1, index2)

    θranges, ωranges = variablemappingranges(model.core.num_pars, index1, index2)

    return newLb, newUb, initGuess, θranges, ωranges
end

"""
    init_nuisance_parameters(model::LikelihoodModel, index::Int)

Initialises the lower and upper bounds, and initial guess for nuisance parameters and indices that map variables between parameter and nuisance parameter space, given interest parameters in `θindices`. The initial guess for nuisance parameters is their corresponding value at the maximum likelihood estimate (`model.core.θmle`).
"""
function init_nuisance_parameters(model::LikelihoodModel, θindices::Vector{Int}, num_dims::Int, θlb_nuisance::AbstractVector{<:Float64}, θub_nuisance::AbstractVector{<:Float64})

    ωindices  = setdiff(1:model.core.num_pars, θindices)
    newLb     = zeros(model.core.num_pars-num_dims) 
    newUb     = zeros(model.core.num_pars-num_dims)
    initGuess = zeros(model.core.num_pars-num_dims)

    newLb     .= θlb_nuisance[ωindices]
    newUb     .= θub_nuisance[ωindices]
    initGuess .= model.core.θmle[ωindices]

    return newLb, newUb, initGuess, ωindices
end