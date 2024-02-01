"""
    dimensional_optimiser!(θs_opt::Union{Vector, SubArray}, 
        q::NamedTuple, 
        options::OptimizationSettings, 
        targetll::Real)

Given a log-likelihood function (`q.consistent.loglikefunction`) which is bounded in parameter space, this function finds the values of the nuisance parameters ω that optimise the function for fixed values of the interest parameters ψ (which are already in `θs_opt`) and returns the log-likelihood value minus the confidence boundary target threshold. The returned function value will be zero at the locations of the approximate confidence boundary for ψ. Nuisance parameter values are stored in the corresponding indices of θs_opt, modifying the array in place.
"""
function dimensional_optimiser!(θs_opt::Union{Vector,SubArray},
    q::NamedTuple, 
    options::OptimizationSettings, 
    targetll::Real)
    
    function fun(ω, q)
        θs = zeros(eltype(ω), q.consistent.num_pars)
        θs[q.θindices] .= θs_opt[q.θindices]
        θs[q.ωindices] .= ω
        @timeit_debug timer "Likelihood evaluation" begin
            return -q.consistent.loglikefunction(θs, q.consistent.data)
        end
    end

    @timeit_debug timer "Likelihood nuisance parameter optimisation" begin
        (xopt,fopt)=optimise(fun, q, options, q.initGuess, q.newLb, q.newUb)
    end
    llb=fopt-targetll
    θs_opt[q.ωindices] .= xopt
    return llb
end

"""
    valid_points(model::LikelihoodModel, 
        p::NamedTuple, 
        grid::Matrix{Float64},
        grid_size::Int,
        confidence_level::Float64, 
        dof::Int,
        num_dims::Int,
        use_threads::Bool,
        channel::RemoteChannel)

Given a `grid` of `grid_size` points in interest parameter space with `num_dims` dimensions, this function finds the values of nuisance parameters that maximise the log-likelihood function at each point and returns the points that are within the `confidence_level`, `dof`, log-likelihood threshold as a `num_dims * n` array, alongside a vector of their log-likelihood values. Log-likelihood values are standardised to 0.0 at the MLE point.
"""
function valid_points(model::LikelihoodModel, 
                        q::NamedTuple, 
                        grid::Matrix{Float64},
                        grid_size::Int,
                        confidence_level::Float64,  
                        dof::Int,
                        num_dims::Int,
                        optimizationsettings::OptimizationSettings,
                        use_threads::Bool,
                        channel::RemoteChannel)

    valid_point = falses(grid_size)
    ll_values = zeros(grid_size)
    targetll = get_target_loglikelihood(model, confidence_level, LogLikelihood(), dof)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=grid_size) 
    @floop ex for i in axes(grid,2)
        ll_values[i] = dimensional_optimiser!(@view(grid[:,i]), q, optimizationsettings, targetll)
        put!(channel, true)
    end
    valid_point .= ll_values .≥ 0.0

    valid_ll_values = ll_values[valid_point]
    valid_ll_values .= valid_ll_values .+ get_target_loglikelihood(model, confidence_level,
                                                        EllipseApproxAnalytical(), dof)

    return grid[:,valid_point], valid_ll_values
end

"""
    check_if_bounds_supplied(model::LikelihoodModel,
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real})

Returns the model bounds on interest parameter space if lb and ub are empty, and lb and ub for interest parameter indices otherwise.  
"""
function check_if_bounds_supplied(model::LikelihoodModel,
                                    θindices::Vector{Int},
                                    lb::AbstractVector{<:Real},
                                    ub::AbstractVector{<:Real})
    if isempty(lb)
        lb = model.core.θlb[θindices]
    else
        length(lb) == length(θindices) || throw(ArgumentError(string("lb must be of length ", length(θindices))))
    end
    if isempty(ub)
        ub = model.core.θub[θindices]
    else
        length(ub) == length(θindices)  || throw(ArgumentError(string("ub must be of length ", length(θindices))))
    end
    return lb, ub
end

"""
    uniform_grid(model::LikelihoodModel,
        θindices::Vector{Int},
        points_per_dimension::Union{Int, Vector{Int}},
        confidence_level::Float64,
        dof::Int,
        lb::AbstractVector{<:Real}=Float64[],
        ub::AbstractVector{<:Real}=Float64[],
        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
        optimizationsettings::OptimizationSettings=default_OptimizationSettings();
        use_threads=true,
        arguments_checked::Bool=false,
        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(Inf)))

Creates a uniform grid on interest parameter space `θindices` with `points_per_dimension` in each interest dimension, uniformly spaced between `lb[θindices]` and `ub[θindices]` if supplied or between the bounds contained in `model.core`. The grid is then passed to [`LikelihoodBasedProfileWiseAnalysis.valid_points`](@ref) to determine the values of nuisance parameters that maximise log-likelihood function at each grid point. All grid points within the `confidence_level` log-likelihood threshold are then saved as a [`SampledConfidenceStruct`](@ref). Points are saved alongside a vector of their log-likelihood values. Log-likelihood values are standardised to 0.0 at the MLE point.

For the [`UniformGridSamples`](@ref) sample type.
"""
function uniform_grid(model::LikelihoodModel,
                        θindices::Vector{Int},
                        points_per_dimension::Union{Int, Vector{Int}},
                        confidence_level::Float64,
                        dof::Int,
                        lb::AbstractVector{<:Real}=Float64[],
                        ub::AbstractVector{<:Real}=Float64[],
                        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                        optimizationsettings::OptimizationSettings=default_OptimizationSettings();
                        use_threads=true,
                        arguments_checked::Bool=false,
                        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(Inf)))

    num_dims = length(θindices); num_dims > 0 || throw(ArgumentError("θindices must not be empty"))
    if points_per_dimension isa Vector{Int}
        num_dims == length(points_per_dimension) || throw(ArgumentError(string("points_per_dimension must be of length ", num_dims)))
        all(points_per_dimension .> 0) || throw(DomainError("points_per_dimension must be a vector of strictly positive integers"))
    else
        points_per_dimension > 0 || throw(DomainError("points_per_dimension must be a strictly positive integer"))
        points_per_dimension = fill(points_per_dimension, num_dims)
    end

    newLb, newUb, initGuess, ωindices = init_nuisance_parameters(model, θindices, num_dims, θlb_nuisance, θub_nuisance)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), dof)
    q=(θindices=θindices, newLb=newLb, newUb=newUb, initGuess=initGuess,
        ωindices=ωindices, consistent=consistent)

    lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, θindices, lb, ub)

    ranges = LinRange.(lb, ub, points_per_dimension)
    grid_iterator = Iterators.product(ranges...)
    grid_size = prod(points_per_dimension)

    grid = zeros(model.core.num_pars, grid_size)
    for (i, point) in enumerate(grid_iterator)
        grid[θindices, i] .= point
    end

    pnts, lls = valid_points(model, q, grid, grid_size, confidence_level, dof, num_dims, 
                                optimizationsettings, use_threads, channel)
    return SampledConfidenceStruct(pnts, lls)
end

# function uniform_random_blocks(model::LikelihoodModel,
#                         θindices::Vector{Int},
#                         num_points::Int,
#                         confidence_level::Float64,
#                         lb::AbstractVector{<:Real}=Float64[],
#                         ub::AbstractVector{<:Real}=Float64[];
#                         block_size=20000,
#                         use_threads::Bool=true,
#                         arguments_checked::Bool=false,
#                         channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(num_points+1)))

#     num_dims = length(θindices); num_dims > 0 || throw(ArgumentError("θindices must not be empty"))
#     if !arguments_checked
#         num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
#     end
#     newLb, newUb, initGuess, ωindices = init_nuisance_parameters(model, θindices, num_dims)
#     consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), num_dims)
#     p=(θindices=θindices, newLb=newLb, newUb=newUb, initGuess=initGuess,
#         ωindices=ωindices, consistent=consistent)
    
#     lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, θindices, lb, ub)

#     num_blocks = div(num_points, block_size)
#     remainder = mod(num_points, block_size)

#     valid_grid = zeros(model.core.num_pars, 0)
#     valid_ll_values = zeros(0)

#     block_sizes = [block_size for _ in 1:num_blocks]
#     if remainder > 0; append!(block_sizes, remainder) end
#     cumulative_sizes = cumsum(block_sizes)
#     preallocate_j = findfirst((cumulative_sizes ./ num_points) .≥ 0.05)

#     current_len = 0
#     estimated_size = num_points
#     grid = zeros(model.core.num_pars, block_size)
#     for (i, block) in enumerate(block_sizes)

#         if block < block_size; grid = zeros(model.core.num_pars, block) end

#         for dim in 1:num_dims
#             grid[θindices[dim], :] .= rand(Uniform(lb[dim], ub[dim]), block)
#         end

#         pnts, lls = valid_points(model, p, grid, block, confidence_level, num_dims, use_threads, channel)

#         if i ≤ preallocate_j
#             valid_grid = hcat(valid_grid, @view(pnts[:,:]))
#             valid_ll_values = vcat(valid_ll_values, @view(lls[:]))
#             continue
#         end
#         if (preallocate_j + 1) == i
#             # estimate preallocation size
#             current_len = length(valid_ll_values)
#             accept_rate = current_len/cumulative_sizes[preallocate_j]
#             safety_factor = 1.25
#             estimated_size = convert(Int, round(accept_rate * num_points * safety_factor, RoundDown))

#             valid_grid = hcat(valid_grid, zeros(model.core.num_pars, estimated_size-current_len))
#             resize!(valid_ll_values, estimated_size)
#         end

#         num_to_add = length(lls)
#         if (current_len + num_to_add) ≤ estimated_size
#             rnge = current_len+1:current_len+num_to_add
#             valid_grid[:, rnge] .= pnts
#             valid_ll_values[rnge] .= lls
#             current_len += num_to_add
#         else
#             # do something here
#             continue
#         end
#     end

#     return SampledConfidenceStruct(valid_grid[:,1:current_len], valid_ll_values[1:current_len])
# end

"""
    uniform_random(model::LikelihoodModel,
        num_points::Int,
        confidence_level::Float64,
        dof::Int,
        lb::AbstractVector{<:Real}=Float64[],
        ub::AbstractVector{<:Real}=Float64[],
        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
        optimizationsettings::OptimizationSettings=default_OptimizationSettings();
        use_threads::Bool=true,
        use_distributed::Bool=false,
        arguments_checked::Bool=false,
        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(num_points+1)))

Creates a grid of `num_points` uniform random points on interest parameter space `θindices` sampled between `lb[θindices]` and `ub[θindices]` if supplied or between the bounds contained in `model.core`. The grid is then passed to [`LikelihoodBasedProfileWiseAnalysis.valid_points`](@ref) to determine the values of nuisance parameters that maximise log-likelihood function at each grid point. All grid points within the `confidence_level` log-likelihood threshold are then saved as a [`SampledConfidenceStruct`](@ref). Points are saved alongside a vector of their log-likelihood values. Log-likelihood values are standardised to 0.0 at the MLE point.
    
For the [`UniformRandomSamples`](@ref) sample type.
"""
function uniform_random(model::LikelihoodModel,
                        θindices::Vector{Int},
                        num_points::Int,
                        confidence_level::Float64,
                        dof::Int,
                        lb::AbstractVector{<:Real}=Float64[],
                        ub::AbstractVector{<:Real}=Float64[],
                        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                        optimizationsettings::OptimizationSettings=default_OptimizationSettings();
                        use_threads::Bool=true,
                        arguments_checked::Bool=false,
                        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(num_points+1)))

    num_dims = length(θindices); num_dims > 0 || throw(ArgumentError("θindices must not be empty"))
    if !arguments_checked
        num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
    end
    
    newLb, newUb, initGuess, ωindices = init_nuisance_parameters(model, θindices, num_dims, θlb_nuisance, θub_nuisance)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), dof)
    q=(θindices=θindices, newLb=newLb, newUb=newUb, initGuess=initGuess,
        ωindices=ωindices, consistent=consistent)
    
    lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, θindices, lb, ub)

    grid = zeros(model.core.num_pars, num_points)

    for dim in 1:num_dims
        grid[θindices[dim], :] .= rand(Uniform(lb[dim], ub[dim]), num_points)
    end

    pnts, lls = valid_points(model, q, grid, num_points, confidence_level, dof, num_dims, 
                                optimizationsettings, use_threads, channel)
    return SampledConfidenceStruct(pnts, lls)
end

"""
    LHS(model::LikelihoodModel,
        θindices::Vector{Int},
        num_points::Int,
        confidence_level::Float64,
        dof::Int,
        lb::AbstractVector{<:Real}=Float64[],
        ub::AbstractVector{<:Real}=Float64[],
        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
        optimizationsettings::OptimizationSettings=default_OptimizationSettings();
        use_threads::Bool=true,
        arguments_checked::Bool=false,
        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(num_points+1)))

Creates a grid of `num_points` points on interest parameter space `θindices` using a Latin Hypercube sampling plan between `lb[θindices]` and `ub[θindices]` if supplied or between the bounds contained in `model.core`. The grid is then passed to [`LikelihoodBasedProfileWiseAnalysis.valid_points`](@ref) to determine the values of nuisance parameters that maximise log-likelihood function at each grid point. All grid points within the `confidence_level` log-likelihood threshold are then saved as a [`SampledConfidenceStruct`](@ref). Points are saved alongside a vector of their log-likelihood values. Log-likelihood values are standardised to 0.0 at the MLE point.

For the [`LatinHypercubeSamples`](@ref) sample type.
"""
function LHS(model::LikelihoodModel,
            θindices::Vector{Int},
            num_points::Int,
            confidence_level::Float64,
            dof::Int,
            lb::AbstractVector{<:Real}=Float64[],
            ub::AbstractVector{<:Real}=Float64[],
            θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
            θub_nuisance::AbstractVector{<:Real}=model.core.θub,
            optimizationsettings::OptimizationSettings=default_OptimizationSettings();
            use_threads::Bool=true,
            arguments_checked::Bool=false,
            channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(num_points+1)))
    
    num_dims = length(θindices); num_dims > 0 || throw(ArgumentError("θindices must not be empty"))
    if !arguments_checked
        num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
        lb, ub = check_if_bounds_supplied(model, θindices, lb, ub)
    end

    newLb, newUb, initGuess, ωindices = init_nuisance_parameters(model, θindices, num_dims, θlb_nuisance, θub_nuisance)
    consistent = get_consistent_tuple(model, confidence_level, LogLikelihood(), dof)
    q=(θindices=θindices, newLb=newLb, newUb=newUb, initGuess=initGuess,
        ωindices=ωindices, consistent=consistent)

    grid = zeros(model.core.num_pars, num_points)
    
    scale_range = [(lb[i], ub[i]) for i in 1:num_dims]
    grid[θindices, :] = permutedims(scaleLHC(randomLHC(num_points, num_dims), scale_range))

    # grid = permutedims(scaleLHC(LHCoptim(num_points, num_dims, num_gens; kwargs...)[1], scale_range))
    
    pnts, lls = valid_points(model, q, grid, num_points, confidence_level, dof, num_dims, 
                                optimizationsettings, use_threads, channel)
    return SampledConfidenceStruct(pnts, lls)
end

"""
    dimensional_likelihood_sample(model::LikelihoodModel,
        θindices::Vector{Int},
        num_points::Union{Int, Vector{Int}},
        confidence_level::Float64,
        dof::Int,
        sample_type::AbstractSampleType,
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real},
        θlb_nuisance::AbstractVector{<:Real},
        θub_nuisance::AbstractVector{<:Real},
        optimizationsettings::OptimizationSettings,
        use_threads::Bool,
        channel::RemoteChannel)

Calls the desired method for sampling interest parameter space, `sample_type`, and returns a [`SampledConfidenceStruct`](@ref) containing any points that were found within the `confidence_level`, `dof, log-likelihood threshold.
"""
function dimensional_likelihood_sample(model::LikelihoodModel,
                                    θindices::Vector{Int},
                                    num_points::Union{Int, Vector{Int}},
                                    confidence_level::Float64,
                                    dof::Int,
                                    sample_type::AbstractSampleType,
                                    lb::AbstractVector{<:Real},
                                    ub::AbstractVector{<:Real}, 
                                    θlb_nuisance::AbstractVector{<:Real},
                                    θub_nuisance::AbstractVector{<:Real},
                                    optimizationsettings::OptimizationSettings,
                                    use_threads::Bool,
                                    channel::RemoteChannel)

    try         
        @timeit_debug timer "Dimensional likelihood sample" begin
            if sample_type isa UniformGridSamples
                sample_struct = uniform_grid(model, θindices, num_points, confidence_level, dof, lb, ub,
                                                θlb_nuisance, θub_nuisance,
                                                optimizationsettings;
                                                use_threads=use_threads, arguments_checked=true,
                                                channel=channel)
            elseif sample_type isa UniformRandomSamples
                sample_struct = uniform_random(model, θindices, num_points, confidence_level, dof, lb, ub,
                                                θlb_nuisance, θub_nuisance,
                                                optimizationsettings;             
                                                use_threads=use_threads, arguments_checked=true,
                                                channel=channel)
            elseif sample_type isa LatinHypercubeSamples
                sample_struct = LHS(model, θindices, num_points, confidence_level, dof, lb, ub,
                                    θlb_nuisance, θub_nuisance,
                                    optimizationsettings;
                                    use_threads=use_threads, arguments_checked=true, channel=channel)
            end 
            return sample_struct
        end
    catch
        @error string("an error occurred when computing a dimensional sample with settings: ",
            (sample_type=sample_type, confidence_level=confidence_level,
                θindices=θindices))
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
            println(stdout)
        end
    end
    return nothing
end

"""
    dimensional_likelihood_samples!(model::LikelihoodModel,
        θindices::Vector{Vector{Int}},
        num_points_to_sample::Union{Int, Vector{Int}};
        <keyword arguments>)

Samples `num_points_to_sample` points from interest parameter space, for each interest parameter combination in `θindices`, determining the values of nuisance parameters that maximise log-likelihood function at each, saving all points that are inside the `confidence_level` log-likelihood threshold. Saves these samples by modifying `model` in place.

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `θindices`: a vector of vectors of parameter indexes for the combinations of interest parameters to samples points from.
- `num_points_to_sample`: integer number of points to sample (for [`UniformRandomSamples`](@ref) and [`LatinHypercubeSamples`](@ref) sample types). For the [`UniformGridSamples`](@ref) sample type, if integer it is the number of points to grid over in each parameter dimension. If it is a vector of integers each index of the vector is the number of points to grid over in the corresponding parameter dimension. For example, [1,2] would mean a single point in dimension 1 and two points in dimension 2. 

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to find samples within. Default is `0.95` (95%).
- `sample_type`: the sampling method used to sample parameter space. Available sample types are [`UniformGridSamples`](@ref), [`UniformRandomSamples`](@ref) and [`LatinHypercubeSamples`](@ref). Default is `LatinHypercubeSamples()` ([`LatinHypercubeSamples`](@ref)).
- `lb`: optional vector of lower bounds on interest parameters. Use to specify interest parameter lower bounds to sample over that are different than those contained in `model.core` (must be the same length as original bounds). Default is `Float64[]` (use lower bounds from `model.core`).
- `ub`: optional vector of upper bounds on interest parameters. Use to specify interest parameter upper bounds to sample over that are different than those contained in `model.core` (must be the same length as original bounds). Default is `Float64[]` (use upper bounds from `model.core`).
- `θlb_nuisance`: a vector of lower bounds on nuisance parameters, require `θlb_nuisance .≤ model.core.θmle`. Default is `model.core.θlb`. 
- `θub_nuisance`: a vector of upper bounds on nuisance parameters, require `θub_nuisance .≥ model.core.θmle`. Default is `model.core.θub`.
- `existing_profiles`: `Symbol ∈ [:ignore, :overwrite]` specifying what to do if samples already exist for a given `confidence_level` and `sample_type`. Default is `:overwrite`.
- `optimizationsettings`: a [`OptimizationSettings`](@ref) containing the optimisation settings used to find optimal values of nuisance parameters for a given interest parameter values. Default is `missing` (will use `model.core.optimizationsettings`).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of `θcombinations` completed and estimated time of completion. Default is `model.show_progress`.
- `use_distributed`: boolean variable specifying whether to use a normal for loop or a `@distributed` for loop across combinations of interest parameters. Set this variable to `false` if [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is not being used. Default is `true`.
- `use_threads`: boolean variable specifying, if `use_distributed` is false, to use a parallelised for loop across `Threads.nthreads()` threads to evaluate the log-likelihood at each sampled point. Default is `true`.

# Details

Using [`dimensional_likelihood_sample`](@ref) this function calls the sample method specified by `sample_type` for each set of interest parameters in `[θindices]` (depending on the setting for `existing_profiles` and `confidence_level` if these samples already exist). Updates `model.dim_samples_df` for each successful sample and saves their results as a [`SampledConfidenceStruct`](@ref) in `model.dim_samples_dict`, where the keys for the dictionary is the row number in `model.dim_samples_df` of the corresponding sample. `model.dim_samples_df.num_points` is the number of points within the confidence boundary from those sampled.

!!! note "Support for `dof`"
    Setting the degrees of freedom of a sampled parameter confidence set to a value other than the interest parameter dimensionality is not currently supported (e.g. as supported for univariate and bivariate profiles). Support may be added in the future, with a slight change in the API of this function.

# Extended help

## Parallel Computing Implementation

If [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is being used `use_distributed` is `true`, then the dimensional samples of distinct interest parameter combinations will be computed in parallel across `Distributed.nworkers()` workers. If `use_distributed` is `false` and `use_threads` is `true` then the dimensional samples of each distinct interest parameter combination will be computed in parallel across `Threads.nthreads()` threads. It is highly recommended to set `use_threads` to `true` in that situation.

## Iteration Speed Of the Progress Meter

The time/it value is the time it takes for each point chosen under the specified sampling scheme to be evaluated as valid or not, for each interest parameter combination. A point is valid if the log-likelihood function value at that point is greater than the confidence log-likelihood threshold.
"""
function dimensional_likelihood_samples!(model::LikelihoodModel,
                                        θindices::Vector{Vector{Int}},
                                        num_points_to_sample::Union{Int, Vector{Int}};
                                        confidence_level::Float64=0.95,
                                        sample_type::AbstractSampleType=LatinHypercubeSamples(),
                                        lb::AbstractVector{<:Real}=Float64[],
                                        ub::AbstractVector{<:Real}=Float64[],
                                        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                                        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                                        existing_profiles::Symbol=:overwrite,
                                        optimizationsettings::Union{OptimizationSettings,Missing}=missing,
                                        show_progress::Bool=model.show_progress,
                                        use_distributed::Bool=true,
                                        use_threads::Bool=true)

    function argument_handling()
        model.core isa CoreLikelihoodModel || throw(ArgumentError("model does not contain a log-likelihood function. Add it using add_loglikelihood_function!"))

        if num_points_to_sample isa Int
            num_points_to_sample > 0 || throw(DomainError("num_points_to_sample must be a strictly positive integer"))
        else
            minimum(num_points_to_sample) > 0 || throw(DomainError("num_points_to_sample must contain strictly positive integers"))

            sample_type isa UniformGridSamples || throw(ArgumentError(string("num_points_to_sample must be an integer for ", sample_type, " sample_type")))

            (length(num_points_to_sample) == length(θindices[1]) &&
                diff([extrema(length.(θindices))...])[1] == 0) || 
                throw(ArgumentError("num_points_to_sample must have the same length as each vector of interest parameters in num_points_to_sample"))
        end
        existing_profiles ∈ [:ignore, :overwrite] || throw(ArgumentError("existing_profiles can only take value :ignore or :overwrite"))

        (!use_distributed && use_threads && timeit_debug_enabled()) &&
            throw(ArgumentError("use_threads cannot be true when debug timings from TimerOutputs are enabled and use_distributed is false. Either set use_threads to false or disable debug timings using `LikelihoodBasedProfileWiseAnalysis.TimerOutputs.disable_debug_timings(LikelihoodBasedProfileWiseAnalysis)`"))

        # error handle confidence_level
        get_target_loglikelihood(model, confidence_level, LogLikelihood(), 1)

        length(θlb_nuisance) == model.core.num_pars || throw(ArgumentError("θlb_nuisance must have the same length as the number of model parameters"))
        length(θub_nuisance) == model.core.num_pars || throw(ArgumentError("θub_nuisance must have the same length as the number of model parameters"))
        all(θlb_nuisance .≤ model.core.θmle) || throw(DomainError("θlb_nuisance must be less than or equal to model.core.θmle"))
        all(θub_nuisance .≥ model.core.θmle) || throw(DomainError("θub_nuisance must be greater than or equal to model.core.θmle"))
        
        θindices = θindices[.!isempty.(θindices)]
        sort!.(θindices); unique!.(θindices)
        sort!(θindices); unique!(θindices)
        1 ≤ first.(θindices)[1] && maximum(last.(θindices)) ≤ model.core.num_pars || throw(DomainError("θindices can only contain parameter indexes between 1 and the number of model parameters"))

        return nothing
    end
    
    argument_handling()
    optimizationsettings = ismissing(optimizationsettings) ? model.core.optimizationsettings : optimizationsettings
    lb, ub = check_if_bounds_supplied(model, lb, ub)

    # check if any of θindices is for the full likelihood - do this outside main for loop
    for (i, θs) in enumerate(θindices)
        if length(θs) == model.core.num_pars
            full_likelihood_sample!(model::LikelihoodModel, num_points_to_sample;
                                    confidence_level=confidence_level,
                                    sample_type=sample_type,
                                    lb=lb, ub=ub, use_threads=use_threads,
                                    use_distributed=use_distributed,
                                    existing_profiles=existing_profiles)
            θindices = θindices[setdiff(1:length(θindices), i)]
            break
        end
    end

    init_dim_samples_row_exists!(model, θindices, sample_type)

    θs_to_keep = trues(length(θindices))
    θs_to_overwrite = falses(length(θindices))
    num_to_overwrite = 0
    # check if profile has already been evaluated
    # in this case we only have :ignore and :overwrite
    for (i, θs) in enumerate(θindices)
        if model.dim_samples_row_exists[(θs, length(θs), sample_type)][confidence_level] != 0
            θs_to_keep[i] = false
            θs_to_overwrite[i] = true
            num_to_overwrite += 1
        end
    end
    if existing_profiles == :ignore
        θindices = θindices[θs_to_keep]
        θs_to_overwrite = θs_to_overwrite[θs_to_keep]
        num_to_overwrite = 0
    end
    
    len_θindices = length(θindices)
    len_θindices > 0 || return nothing

    num_rows_required = ((len_θindices-num_to_overwrite) + model.num_dim_samples) - nrow(model.dim_samples_df)

    if num_rows_required > 0
        add_dim_samples_rows!(model, num_rows_required)
    end

    if sample_type isa UniformGridSamples
        if num_points_to_sample isa Int
            totaltasks = sum(num_points_to_sample .^ length.(θindices))
            tasks_per_profile = totaltasks / length(θindices)
        else
            tasks_per_profile = prod(num_points_to_sample)
            totaltasks = length(θindices) * tasks_per_profile
        end
    else
        tasks_per_profile = num_points_to_sample
        totaltasks = length(θindices) * tasks_per_profile
    end
    channel_buffer_size = min(ceil(Int, tasks_per_profile*0.2), 400)
    channel = RemoteChannel(() -> Channel{Bool}(channel_buffer_size))
    
    p = Progress(totaltasks; desc="Computing dimensional profile samples: ",
                dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    @sync begin
        @async while take!(channel)
            next!(p)
        end

        @async begin
            if use_distributed
                profiles_to_add = @distributed (vcat) for θs in θindices
                    [(θs, dimensional_likelihood_sample(model, θs, num_points_to_sample,
                                                        confidence_level, length(θs), sample_type,
                                                        lb[θs], ub[θs], 
                                                        θlb_nuisance, θub_nuisance,
                                                        optimizationsettings, 
                                                        false, channel))]
                end
                
                for (i, (θs, sample_struct)) in enumerate(profiles_to_add)
                    if isnothing(sample_struct); continue end
                    
                    num_points_kept = length(sample_struct.ll)
                    if num_points_kept == 0
                        @warn string("no sampled points for θindices, ", θs, ", were in the confidence region of the profile likelihood within the supplied bounds: try increasing num_points_to_sample or changing the bounds")
                        continue
                    end
                    
                    if θs_to_overwrite[i]
                        row_ind = model.dim_samples_row_exists[(θs, length(θs), sample_type)][confidence_level]
                    else
                        model.num_dim_samples += 1
                        row_ind = model.num_dim_samples * 1
                        model.dim_samples_row_exists[(θs, length(θs), sample_type)][confidence_level] = row_ind
                    end
                    
                    model.dim_samples_dict[row_ind] = sample_struct
                    
                    set_dim_samples_row!(model, row_ind, θs, true, confidence_level, length(θs), sample_type,
                    num_points_kept)
                end
            else
                for (i, θs) in enumerate(θindices)
                    sample_struct = dimensional_likelihood_sample(model, θs, num_points_to_sample,
                                                        confidence_level, length(θs), sample_type,
                                                        lb[θs], ub[θs],
                                                        θlb_nuisance, θub_nuisance, 
                                                        optimizationsettings, 
                                                        use_threads, channel)

                    if isnothing(sample_struct); continue end
                    
                    num_points_kept = length(sample_struct.ll)
                    if num_points_kept == 0
                        @warn string("no sampled points for θindices, ", θs, ", were in the confidence region of the profile likelihood within the supplied bounds: try increasing num_points_to_sample or changing the bounds")
                        continue
                    end
                    
                    if θs_to_overwrite[i]
                        row_ind = model.dim_samples_row_exists[(θs, length(θs), sample_type)][confidence_level]
                    else
                        model.num_dim_samples += 1
                        row_ind = model.num_dim_samples * 1
                        model.dim_samples_row_exists[(θs, length(θs), sample_type)][confidence_level] = row_ind
                    end
                    
                    model.dim_samples_dict[row_ind] = sample_struct
                    
                    set_dim_samples_row!(model, row_ind, θs, true, confidence_level, length(θs), sample_type,
                        num_points_kept)
                end
            end
            put!(channel, false)
        end
    end

    return nothing
end

"""
    dimensional_likelihood_samples!(model::LikelihoodModel,
        θnames::Vector{Vector{Symbol}},
        num_points_to_sample::Union{Int, Vector{Int}};
        <keyword arguments>)

Samples just the provided `θnames` interest parameter sets, provided as a vector of vectors.
"""
function dimensional_likelihood_samples!(model::LikelihoodModel,
                                            θnames::Vector{Vector{Symbol}},
                                            num_points_to_sample::Union{Int, Vector{Int}};
                                            confidence_level::Float64=0.95,
                                            sample_type::AbstractSampleType=LatinHypercubeSamples(),
                                            lb::AbstractVector{<:Real}=Float64[],
                                            ub::AbstractVector{<:Real}=Float64[],
                                            θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                                            θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                                            existing_profiles::Symbol=:overwrite,
                                            optimizationsettings::Union{OptimizationSettings,Missing}=missing,
                                            show_progress::Bool=model.show_progress,
                                            use_distributed::Bool=true,
                                            use_threads::Bool=true)

    θindices = convertθnames_toindices(model, θnames)

    dimensional_likelihood_samples!(model, θindices, num_points_to_sample,
                                    confidence_level=confidence_level, sample_type=sample_type,
                                    lb=lb, ub=ub,
                                    θlb_nuisance=θlb_nuisance, θub_nuisance=θub_nuisance,
                                    existing_profiles=existing_profiles,
                                    optimizationsettings=optimizationsettings,
                                    use_distributed=use_distributed,
                                    show_progress=show_progress,
                                    use_threads=use_threads)
    return nothing
end

"""
    dimensional_likelihood_samples!(model::LikelihoodModel,
        sample_dimension::Int,
        sample_m_random_combinations::Int,
        num_points_to_sample::Union{Int, Vector{Int}}
        <keyword arguments>)

Samples m random combinations of `sample_dimension` model parameters (sampling without replacement), where `0 < m ≤ binomial(model.core.num_pars, sample_dimension)`.
"""
function dimensional_likelihood_samples!(model::LikelihoodModel,
                                            sample_dimension::Int,
                                            sample_m_random_combinations::Int,
                                            num_points_to_sample::Union{Int, Vector{Int}};
                                            confidence_level::Float64=0.95,
                                            sample_type::AbstractSampleType=LatinHypercubeSamples(),
                                            lb::AbstractVector{<:Real}=Float64[],
                                            ub::AbstractVector{<:Real}=Float64[],
                                            θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                                            θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                                            existing_profiles::Symbol=:overwrite,
                                            optimizationsettings::Union{OptimizationSettings,Missing}=missing,
                                            show_progress::Bool=model.show_progress,
                                            use_distributed::Bool=true,
                                            use_threads::Bool=true)

    sample_m_random_combinations = max(0, min(sample_m_random_combinations, binomial(model.core.num_pars, sample_dimension)))
    sample_m_random_combinations > 0 || throw(DomainError("sample_m_random_combinations must be a strictly positive integer"))

    θcombinations = sample(collect(combinations(1:model.core.num_pars, sample_dimension)),
                            sample_m_random_combinations, replace=false, ordered=true)

    dimensional_likelihood_samples!(model, θcombinations, num_points_to_sample,
                                    confidence_level=confidence_level, sample_type=sample_type,
                                    lb=lb, ub=ub,
                                    θlb_nuisance=θlb_nuisance, θub_nuisance=θub_nuisance,
                                    existing_profiles=existing_profiles,
                                    optimizationsettings=optimizationsettings,
                                    show_progress=show_progress,
                                    use_distributed=use_distributed,
                                    use_threads=use_threads)
    return nothing
end

"""
    dimensional_likelihood_samples!(model::LikelihoodModel,
        sample_dimension::Int,
        num_points_to_sample::Union{Int, Vector{Int}};
        <keyword arguments>)

Samples all combinations of `sample_dimension` model parameters.
"""
function dimensional_likelihood_samples!(model::LikelihoodModel,
                                            sample_dimension::Int,
                                            num_points_to_sample::Union{Int, Vector{Int}};
                                            confidence_level::Float64=0.95,
                                            sample_type::AbstractSampleType=LatinHypercubeSamples(),
                                            lb::AbstractVector{<:Real}=Float64[],
                                            ub::AbstractVector{<:Real}=Float64[],
                                            θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                                            θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                                            existing_profiles::Symbol=:overwrite,
                                            optimizationsettings::Union{OptimizationSettings,Missing}=missing,
                                            show_progress::Bool=model.show_progress,
                                            use_distributed::Bool=true,
                                            use_threads::Bool=true)

    θcombinations = collect(combinations(1:model.core.num_pars, sample_dimension))

    dimensional_likelihood_samples!(model, θcombinations, num_points_to_sample,
                                    confidence_level=confidence_level, sample_type=sample_type,
                                    lb=lb, ub=ub,
                                    θlb_nuisance=θlb_nuisance, θub_nuisance=θub_nuisance,
                                    existing_profiles=existing_profiles,
                                    optimizationsettings=optimizationsettings,
                                    show_progress=show_progress,
                                    use_distributed=use_distributed,
                                    use_threads=use_threads)
    return nothing
end