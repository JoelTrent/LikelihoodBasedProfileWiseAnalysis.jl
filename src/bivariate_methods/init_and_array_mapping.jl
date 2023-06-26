function variablemapping2dranges(num_pars::T, index1::T, index2::T) where T <: Int
    θranges = (1:(index1-1), (index1+1):(index2-1), (index2+1):num_pars)
    ωranges = (1:(index1-1), index1:(index2-2), (index2-1):(num_pars-2) )
    return θranges, ωranges
end

function variablemapping2d!(θ::Union{Vector, SubArray}, 
                            ω::Union{Vector, SubArray}, 
                            θranges::Tuple{T, T, T}, 
                            ωranges::Tuple{T, T, T}) where T <: UnitRange
    θ[θranges[1]] .= @view(ω[ωranges[1]])
    θ[θranges[2]] .= @view(ω[ωranges[2]])
    θ[θranges[3]] .= @view(ω[ωranges[3]])
    return θ
end

# we know index1 < index2 by construction. If index1 and index2 are user provided, enforce this relationship 
function boundsmapping2d!(newbounds::Vector{<:Float64}, 
                            bounds::Union{AbstractVector{<:Real}, SubArray{Real}}, 
                            ind1::Int,
                            ind2::Int)
    newbounds[1:(ind1-1)]    .= @view(bounds[1:(ind1-1)])
    newbounds[ind1:(ind2-2)] .= @view(bounds[(ind1+1):(ind2-1)])
    newbounds[(ind2-1):end]  .= @view(bounds[(ind2+1):end])
    return nothing
end

function init_bivariate_parameters(model::LikelihoodModel, 
                                    ind1::Int, 
                                    ind2::Int)
    newLb     = zeros(model.core.num_pars-2) 
    newUb     = zeros(model.core.num_pars-2)
    initGuess = zeros(model.core.num_pars-2)

    boundsmapping2d!(newLb, model.core.θlb, ind1, ind2)
    boundsmapping2d!(newUb, model.core.θub, ind1, ind2)
    boundsmapping2d!(initGuess, model.core.θmle, ind1, ind2)

    θranges, ωranges = variablemapping2dranges(model.core.num_pars, ind1, ind2)

    return newLb, newUb, initGuess, θranges, ωranges
end