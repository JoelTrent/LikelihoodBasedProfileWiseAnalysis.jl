function variablemapping1dranges(num_pars::T, index::T) where T <: Int
    θranges = (1:(index-1), (index+1):num_pars)
    ωranges = (1:(index-1), index:(num_pars-1))
    return θranges, ωranges
end

function variablemapping1d!(θ::Union{Vector, SubArray},
                            ω::Union{Vector, SubArray},
                            θranges::Tuple{T, T}, 
                            ωranges::Tuple{T, T}) where T <: UnitRange
    θ[θranges[1]] .= @view(ω[ωranges[1]])
    θ[θranges[2]] .= @view(ω[ωranges[2]])
    return θ
end

function boundsmapping1d!(newbounds::Vector{<:Float64}, bounds::AbstractVector{<:Real}, index::Int)
    newbounds[1:(index-1)] .= @view(bounds[1:(index-1)])
    newbounds[index:end]   .= @view(bounds[(index+1):end])
    return nothing
end

function init_univariate_parameters(model::LikelihoodModel, 
                                    index)
    newLb     = zeros(model.core.num_pars-1) 
    newUb     = zeros(model.core.num_pars-1)
    initGuess = zeros(model.core.num_pars-1)

    boundsmapping1d!(newLb, model.core.θlb, index)
    boundsmapping1d!(newUb, model.core.θub, index)
    boundsmapping1d!(initGuess, model.core.θmle, index)

    θranges, ωranges = variablemapping1dranges(model.core.num_pars, index)

    return newLb, newUb, initGuess, θranges, ωranges
end