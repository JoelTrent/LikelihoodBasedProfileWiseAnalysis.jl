"""
    checkforInf(x::AbstractVector{<:Real})

Warns via a message if any of the bounds returned given the provided forward transformation are +/-Inf.

# Arguments
- `x`: vector of transformed bounds. 
"""
function checkforInf(x::AbstractVector{<:Real})
    if any(isinf.(x))
        @warn "the specified transformation causes some of the returned bounds to be +/-Inf. Altering the initial bounds is recommended to prevent this from occurring."
    end
    return nothing
end

"""
    transformbounds(transformfun::Function, 
        lb::AbstractVector{<:Real}, 
        ub::AbstractVector{<:Real}, 
        independentParameterIndexes::Vector{<:Int}=Int[], 
        dependentParameterIndexes::Vector{<:Int}=Int[])

Given a monotonic (increasing or decreasing) function, `transformfun`, that describes a parameter transformation, return the lower and upper bounds in the transformed space that correspond to the lower and upper bounds in the original space. Uses a heuristic to evaluate the bound transformation. We assume that the ordering of parameters stay the same for the purposes of `independentParameterIndexes` and `dependentParameterIndexes`.

# Arguments
- `transformfun`: a function describing the forward transformation between parameter space `θ` and `Θ`. Should take in a single argument, `θ`, a vector of parameter values in the original space and return `Θ`, a vector parameter values in the transformed space. These vectors need to be the same length as `lb` and `ub`.
- `lb`: a vector of lower bounds on parameters. 
- `ub`: a vector of upper bounds on parameters. 
- `independentParameterIndexes`: a vector of parameter indexes where the new parameter `Θ[i]` depends only on `transformfun(θ[i])`.
- `dependentParameterIndexes`: a vector of parameter indexes where the new parameter `Θ[i]` depends on `transformfun(θ[i], θ[j], j!=i)`.

The heuristic used for dependent parameters may fail if there are multiple local minima for the appropriate bounds to use. In this case [`transformbounds_NLopt`](@ref) should be used.

Warns if any of the returned bounds are `Inf` using [`LikelihoodBasedProfileWiseAnalysis.checkforInf`](@ref).
"""
function transformbounds(transformfun::Function, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real},
    independentParameterIndexes::Vector{<:Int}=Int[], dependentParameterIndexes::Vector{<:Int}=Int[])

    (length(transformfun(lb)) == length(lb) && length(lb) == length(ub)) || throw(ArgumentError("The length of lb must be the same as the length of ub and transformfun(lb)"))

    newlb, newub = zeros(length(lb)), zeros(length(lb))

    if isempty(dependentParameterIndexes)
        potentialbounds = zeros(2, length(lb))
        potentialbounds[1,:] .= transformfun(lb)
        potentialbounds[2,:] .= transformfun(ub)

        newlb[:] .= minimum(potentialbounds, dims=1)[:]
        newub[:] .= maximum(potentialbounds, dims=1)[:]

        checkforInf.((newlb, newub))
        return newlb, newub
    end

    if !isempty(independentParameterIndexes)
        potentialbounds = zeros(2, length(lb))
        potentialbounds[1,:] .= transformfun(lb)
        potentialbounds[2,:] .= transformfun(ub)
        
        newlb[independentParameterIndexes] .= minimum(@view(potentialbounds[:, independentParameterIndexes]), dims=1)[:]
        newub[independentParameterIndexes] .= maximum(@view(potentialbounds[:, independentParameterIndexes]), dims=1)[:]
    end

    for i in dependentParameterIndexes # MAKE THIS PART VECTORISED? E.g by making it recursive?
        maximisingbounds = copy(ub)
        currentMax = transformfun(maximisingbounds)[i]
        for j in eachindex(ub)
            maximisingbounds[j] = lb[j]
            candidate = transformfun(maximisingbounds)[i]

            if candidate > currentMax
                currentMax = candidate
            else
                maximisingbounds[j] = ub[j]
            end
        end

        newub[i] = currentMax * 1.0
    end

    for i in dependentParameterIndexes
        minimisingbounds = copy(lb)
        currentMin = transformfun(minimisingbounds)[i]
        for j in eachindex(ub)
            minimisingbounds[j] = ub[j]
            candidate = transformfun(minimisingbounds)[i]

            if candidate < currentMin
                currentMin = candidate
            else
                minimisingbounds[j] = lb[j]
            end
        end

        newlb[i] = currentMin * 1.0
    end

    checkforInf.((newlb, newub))
    return newlb, newub
end

"""
    transformbounds_NLopt(transformfun::Function, 
        lb::AbstractVector{<:Real}, 
        ub::AbstractVector{<:Real})

Given a monotonic (increasing or decreasing) function, `transformfun`, that describes a parameter transformation, return the lower and upper bounds in the transformed space that correspond to the lower and upper bounds in the original space. Uses a naturally binary integer programme if `transformfun` is monotonic on θ between `lb` and `ub`.

# Arguments
- `transformfun`: a function describing the forward transformation between parameter space `θ` and `Θ`. Should take in a single argument, `θ`, a vector of parameter values in the original space and return `Θ`, a vector parameter values in the transformed space. These vectors need to be the same length as `lb` and `ub`.
- `lb`: a vector of lower bounds on parameters. 
- `ub`: a vector of upper bounds on parameters. 

Warns if any of the returned bounds are `Inf` using [`LikelihoodBasedProfileWiseAnalysis.checkforInf`](@ref).
"""
function transformbounds_NLopt(transformfun::Function, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real})

    function bounds_transform(x,_)
        return minOrMax * transformfun( (((1.0 .- x) .* lb) .+ (x .* ub)) )[NLP_transformIndex]
    end
    
    # variables will be binary integer automatically due to how the obj function is setup
    # IF the transformation function applied to each θ[i] is monotonic between lb[i] and ub[i]
    num_vars = length(ub)
    
    newlb = zeros(num_vars)
    newub = zeros(num_vars)
    initialGuess = fill(0.5, num_vars)
    NLPlb = zeros(num_vars)
    NLPub = ones(num_vars)
    optimizationsettings = default_OptimizationSettings()

    minOrMax = -1.0
    NLP_transformIndex=0
    for i in 1:num_vars
        NLP_transformIndex += 1
        
        minOrMax = 1.0
        newlb[i] = optimise(bounds_transform, optimizationsettings, initialGuess, NLPlb, NLPub)[2] * -1.0
        
        minOrMax = -1.0
        newub[i] = optimise(bounds_transform, optimizationsettings, initialGuess, NLPlb, NLPub)[2]
    end

    checkforInf.((newlb, newub))
    return newlb, newub
end