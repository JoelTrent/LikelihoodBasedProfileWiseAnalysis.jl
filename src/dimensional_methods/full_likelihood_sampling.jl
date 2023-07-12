"""
    add_dim_samples_rows!(model::LikelihoodModel, num_rows_to_add::Int)

Adds `num_rows_to_add` free rows to `model.dim_samples_df` by vertically concatenating the existing DataFrame and free rows using [`PlaceholderLikelihood.init_dim_samples_df`](@ref).
"""
function add_dim_samples_rows!(model::LikelihoodModel, 
                                num_rows_to_add::Int)
    new_rows = init_dim_samples_df(num_rows_to_add, 
                                    existing_largest_row=nrow(model.dim_samples_df))

    model.dim_samples_df = vcat(model.dim_samples_df, new_rows)
    return nothing
end

"""
    set_dim_samples_row!(model::LikelihoodModel, 
        row_ind::Int,
        θindices::Vector{Int},
        not_evaluated_predictions::Bool,
        confidence_level::Float64,
        sample_type::AbstractSampleType,
        num_points::Int)

Sets the relevant fields of row `row_ind` in `model.dim_samples_df` after a profile has been evaluated.
"""
function set_dim_samples_row!(model::LikelihoodModel, 
                                row_ind::Int,
                                θindices::Vector{Int},
                                not_evaluated_predictions::Bool,
                                confidence_level::Float64,
                                sample_type::AbstractSampleType,
                                num_points::Int)
    model.dim_samples_df[row_ind, 2:end] .= θindices,
                                            length(θindices),
                                            not_evaluated_predictions,
                                            confidence_level,
                                            sample_type,
                                            num_points
    return nothing
end

"""
    valid_points(model::LikelihoodModel, 
        grid::Matrix{Float64},
        grid_size::Int,
        confidence_level::Float64, 
        num_dims::Int,
        use_threads::Bool,
        use_distributed::Bool,
        channel::RemoteChannel)

Given a `grid` of `grid_size` points in full parameter space with `num_dims` dimensions, this function evaluates the log-likelihood function at each point and returns the points that are within the `confidence_level` log-likelihood threshold as a `num_dims * n` array, alongside a vector of their log-likelihood values. Log-likelihood values are standardised to 0.0 at the MLE point.
"""
function valid_points(model::LikelihoodModel, 
                        grid::Matrix{Float64},
                        grid_size::Int,
                        confidence_level::Float64, 
                        num_dims::Int,
                        use_threads::Bool,
                        use_distributed::Bool,
                        channel::RemoteChannel)
    
    valid_point = falses(grid_size)
    ll_values = zeros(grid_size)
    targetll = get_target_loglikelihood(model, confidence_level, LogLikelihood(), num_dims)

    if use_distributed
        ll_values_shared = SharedArray{Float64}(grid_size)

        @distributed (+) for i in axes(grid, 2)
            ll_values_shared[i] = model.core.loglikefunction(grid[:, i], model.core.data) - targetll
            put!(channel, true)
            i
        end

        ll_values[:]   .= ll_values_shared
        valid_point[:] .= ll_values .>= 0.0

    else
        ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=grid_size) 
        @floop ex for i in axes(grid,2)
            ll_values[i] = model.core.loglikefunction(grid[:,i], model.core.data)-targetll
            if ll_values[i] >= 0.0
                valid_point[i] = true
            end
            put!(channel, true)
        end
    end

    valid_ll_values = ll_values[valid_point]
    valid_ll_values .= valid_ll_values .+ get_target_loglikelihood(model, confidence_level,
                                                        EllipseApproxAnalytical(), num_dims)

    put!(channel, false)

    return grid[:,valid_point], valid_ll_values
end

"""
    check_if_bounds_supplied(model::LikelihoodModel,
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real})

Returns the model bounds on full parameter space if lb and ub are empty, and lb and ub otherwise.  
"""
function check_if_bounds_supplied(model::LikelihoodModel,
                                    lb::AbstractVector{<:Real},
                                    ub::AbstractVector{<:Real})
    if isempty(lb)
        lb = model.core.θlb
    else
        length(lb) == model.core.num_pars || throw(ArgumentError(string("lb must be of length ", model.core.num_pars)))
    end
    if isempty(ub)
        ub = model.core.θub
    else
        length(ub) == model.core.num_pars  || throw(ArgumentError(string("ub must be of length ", model.core.num_pars)))
    end
    return lb, ub
end

"""
    uniform_grid(model::LikelihoodModel,
        points_per_dimension::Union{Int, Vector{Int}},
        confidence_level::Float64,
        lb::AbstractVector{<:Real}=Float64[],
        ub::AbstractVector{<:Real}=Float64[];
        use_threads::Bool=true,
        use_distributed::Bool=false,
        arguments_checked::Bool=false,
        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(Inf)))

Creates a uniform grid with `points_per_dimension` in each dimension, uniformly spaced between `lb` and `ub` if supplied or between the bounds contained in `model.core`.The log-likelihood function is evaluated at each grid point and all grid points within the `confidence_level` log-likelihood threshold are saved as a [`SampledConfidenceStruct`](@ref). Points are saved alongside a vector of their log-likelihood values. Log-likelihood values are standardised to 0.0 at the MLE point.

For the [`UniformGridSamples`](@ref) sample type.
"""
function uniform_grid(model::LikelihoodModel,
                        points_per_dimension::Union{Int, Vector{Int}},
                        confidence_level::Float64,
                        lb::AbstractVector{<:Real}=Float64[],
                        ub::AbstractVector{<:Real}=Float64[];
                        use_threads::Bool=true,
                        use_distributed::Bool=false,
                        arguments_checked::Bool=false,
                        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(Inf)))

    num_dims = model.core.num_pars

    if points_per_dimension isa Vector{Int}
        num_dims == length(points_per_dimension) || throw(ArgumentError(string("points_per_dimension must be of length ", num_dims)))
        all(points_per_dimension .> 0) || throw(DomainError("points_per_dimension must be a vector of strictly positive integers"))
    else
        points_per_dimension > 0 || throw(DomainError("points_per_dimension must be a strictly positive integer"))
        points_per_dimension = fill(points_per_dimension, num_dims)
    end
    lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, lb, ub)

    ranges = LinRange.(lb, ub, points_per_dimension)
    grid_iter = Iterators.product(ranges...)
    grid_size = prod(points_per_dimension)
    grid = zeros(num_dims, grid_size)

    for (i, point) in enumerate(grid_iter)
        grid[:, i] .= point
    end

    pnts, lls = valid_points(model, grid, grid_size, confidence_level, num_dims, use_threads, use_distributed, channel)
    return SampledConfidenceStruct(pnts, lls)
end

"""
    uniform_random(model::LikelihoodModel,
        num_points::Int,
        confidence_level::Float64,
        lb::AbstractVector{<:Real}=Float64[],
        ub::AbstractVector{<:Real}=Float64[];
        use_threads::Bool=true,
        use_distributed::Bool=false,
        arguments_checked::Bool=false,
        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(num_points+1)))

Creates a grid of `num_points` uniform random points sampled between `lb` and `ub` if supplied or between the bounds contained in `model.core`. The log-likelihood function is evaluated at each grid point and all grid points within the `confidence_level` log-likelihood threshold are saved as a [`SampledConfidenceStruct`](@ref). Points are saved alongside a vector of their log-likelihood values. Log-likelihood values are standardised to 0.0 at the MLE point.

For the [`UniformRandomSamples`](@ref) sample type.
"""
function uniform_random(model::LikelihoodModel,
                        num_points::Int,
                        confidence_level::Float64,
                        lb::AbstractVector{<:Real}=Float64[],
                        ub::AbstractVector{<:Real}=Float64[];
                        use_threads::Bool=true,
                        use_distributed::Bool=false,
                        arguments_checked::Bool=false,
                        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(num_points+1)))

    num_dims = model.core.num_pars
    if !arguments_checked
        num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
    end
    
    lb, ub = arguments_checked ? (lb, ub) : check_if_bounds_supplied(model, lb, ub)

    grid = zeros(num_dims, num_points)

    for dim in 1:num_dims
        grid[dim, :] .= rand(Uniform(lb[dim], ub[dim]), num_points)
    end

    pnts, lls = valid_points(model, grid, num_points, confidence_level, num_dims, use_threads, use_distributed, channel)
    return SampledConfidenceStruct(pnts, lls)
end

"""
    LHS(model::LikelihoodModel,
        num_points::Int,
        confidence_level::Float64,
        lb::AbstractVector{<:Real}=Float64[],
        ub::AbstractVector{<:Real}=Float64[];
        use_threads::Bool=true,
        use_distributed::Bool=false,
        arguments_checked::Bool=false,
        channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(num_points+1)))

Creates a grid of `num_points` points sampled using a Latin Hypercube sampling plan between `lb` and `ub` if supplied or between the bounds contained in `model.core`. The log-likelihood function is evaluated at each grid point and all grid points within the `confidence_level` log-likelihood threshold are saved as a [`SampledConfidenceStruct`](@ref). Points are saved alongside a vector of their log-likelihood values. Log-likelihood values are standardised to 0.0 at the MLE point.

For the [`LatinHypercubeSamples`](@ref) sample type.
"""
function LHS(model::LikelihoodModel,
            num_points::Int,
            confidence_level::Float64,
            lb::AbstractVector{<:Real}=Float64[],
            ub::AbstractVector{<:Real}=Float64[];
            use_threads::Bool=true,
            use_distributed::Bool=false,
            arguments_checked::Bool=false,
            channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(num_points+1)))
    
    num_dims = model.core.num_pars
    if !arguments_checked
        num_points > 0 || throw(DomainError("num_points must be a strictly positive integer"))
        lb, ub = check_if_bounds_supplied(model, lb, ub)
    end
    
    scale_range = [(lb[i], ub[i]) for i in 1:num_dims]
    grid = permutedims(scaleLHC(randomLHC(num_points, num_dims), scale_range))

    # grid = permutedims(scaleLHC(LHCoptim(num_points, num_dims, num_gens; kwargs...)[1], scale_range))
    
    pnts, lls = valid_points(model, grid, num_points, confidence_level, num_dims, use_threads, use_distributed, channel)
    return SampledConfidenceStruct(pnts, lls)
end

"""
    full_likelihood_sample(model::LikelihoodModel,
        num_points::Union{Int, Vector{Int}},
        confidence_level::Float64,
        sample_type::AbstractSampleType,
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real},
        use_threads::Bool,
        use_distributed::Bool,
        channel::RemoteChannel)

Calls the desired method for sampling parameter space, `sample_type`, and returns a [`SampledConfidenceStruct`](@ref) containing any points that were found within the `confidence_level` log-likelihood threshold.
"""
function full_likelihood_sample(model::LikelihoodModel,
                                    num_points::Union{Int, Vector{Int}},
                                    confidence_level::Float64,
                                    sample_type::AbstractSampleType,
                                    lb::AbstractVector{<:Real},
                                    ub::AbstractVector{<:Real},
                                    use_threads::Bool,
                                    use_distributed::Bool,
                                    channel::RemoteChannel)

    if sample_type isa UniformGridSamples
        sample_struct = uniform_grid(model, num_points, confidence_level, lb, ub;
                                        use_threads=use_threads, use_distributed=use_distributed,
                                        arguments_checked=true, channel=channel)
    elseif sample_type isa UniformRandomSamples
        sample_struct = uniform_random(model, num_points, confidence_level, lb, ub;             
                                        use_threads=use_threads, use_distributed=use_distributed,
                                        arguments_checked=true, channel=channel)
    elseif sample_type isa LatinHypercubeSamples
        sample_struct = LHS(model, num_points, confidence_level, lb, ub;
                            use_threads=use_threads, use_distributed=use_distributed,
                            arguments_checked=true, channel=channel)
    end
    return sample_struct
end

"""
    full_likelihood_sample!(model::LikelihoodModel,
        num_points_to_sample::Union{Int, Vector{Int}};
        <keyword arguments>)

Samples `num_points_to_sample` points from full parameter space, evaluating the log-likelihood function at each, saving all points that are inside the `confidence_level` log-likelihood threshold. Saves this sample by modifying `model` in place.

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `num_points_to_sample`: integer number of points to sample (for [`UniformRandomSamples`](@ref) and [`LatinHypercubeSamples`](@ref) sample types). For the [`UniformGridSamples`](@ref) sample type, if integer it is the number of points to grid over in each parameter dimension. If it is a vector of integers each index of the vector is the number of points to grid over in the corresponding parameter dimension. For example, [1,2] would mean a single point in dimension 1 and two points in dimension 2. 

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level which . Default is 0.95 (95%).
- `sample_type`: the sampling method used to sample parameter space. Available sample types are [`UniformGridSamples`](@ref), [`UniformRandomSamples`](@ref) and [`LatinHypercubeSamples`](@ref). Default is `LatinHypercubeSamples()` ([`LatinHypercubeSamples`](@ref)).
- `lb`: optional vector of lower bounds on parameters. Use to specify parameter lower bounds to sample over that are different than those contained in `model.core`. Default is `Float64[]` (use lower bounds from `model.core`).
- `ub`: optional vector of upper bounds on parameters. Use to specify parameter upper bounds to sample over that are different than those contained in `model.core`. Default is `Float64[]` (use upper bounds from `model.core`).
- `use_distributed`: boolean variable specifying whether to use a threaded for loop or distributed for loop to evaluate the log-likelihood at each sampled point. This should be set to true if Julia instances have been started with low numbers of threads or distributed computing is being used. Default is `false`.
- `use_threads`: boolean variable specifying, if `use_distributed` is false, whether to use a parallelised for loop across `Threads.nthreads()` threads or a non-parallel for loop to evaluate the log-likelihood at each sampled point. Default is true.
- `existing_profiles`: `Symbol ∈ [:ignore, :overwrite]` specifying what to do if samples already exist for a given `confidence_level` and `sample_type`.  Default is `:overwrite`.
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of `θcombinations` completed and estimated time of completion. Default is `model.show_progress`.

# Details

Using [`full_likelihood_sample`](@ref) this function calls the sample method specified by `sample_type` (depending on the setting for `existing_profiles` and `confidence_level` if a full likelihood sample already exists). Updates `model.dim_samples_df` if the sample is successful and saves the results as a [`SampledConfidenceStruct`](@ref) in `model.dim_samples_dict`, where the keys for the dictionary is the row number in `model.dim_samples_df` of the corresponding sample.

## Parallel Computing Implementation

If [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is being used and `use_distributed` is `true`,the log-likelihood value of sampled points will be computed in parallel across `Distributed.nworkers()` workers. If `use_distributed` is `false` and `use_threads` is `true` then the log-likelihood value of sampled points will be computed in parallel across `Threads.nthreads()` threads.

## Iteration Speed Of the Progress Meter

The time/it value is the time it takes for each point chosen under the specified sampling scheme to be evaluated as valid or not. A point is valid if the log-likelihood function value at that point is greater than the confidence log-likelihood threshold.
"""
function full_likelihood_sample!(model::LikelihoodModel,
                                    num_points_to_sample::Union{Int, Vector{Int}};
                                    confidence_level::Float64=0.95,
                                    sample_type::AbstractSampleType=LatinHypercubeSamples(),
                                    lb::AbstractVector{<:Real}=Float64[],
                                    ub::AbstractVector{<:Real}=Float64[],
                                    use_distributed::Bool=false,
                                    use_threads::Bool=true,
                                    existing_profiles::Symbol=:overwrite,
                                    show_progress::Bool=model.show_progress)

    if num_points_to_sample isa Int
        num_points_to_sample > 0 || throw(DomainError("num_points_to_sample must be a strictly positive integer"))
    else
        minimum(num_points_to_sample) > 0 || throw(DomainError("num_points_to_sample must contain strictly positive integers"))

        sample_type isa UniformGridSamples || throw(ArgumentError(string("num_points_to_sample must be an integer for ", sample_type, " sample_type")))
    end
    existing_profiles ∈ [:ignore, :overwrite] || throw(ArgumentError("existing_profiles can only take value :ignore or :overwrite"))
    lb, ub = check_if_bounds_supplied(model, lb, ub)

    # error handle confidence_level
    get_target_loglikelihood(model, confidence_level, LogLikelihood(), 1)

    init_dim_samples_row_exists!(model, sample_type)

    requires_overwrite = model.dim_samples_row_exists[sample_type][confidence_level] != 0
    if existing_profiles == :ignore && requires_overwrite; return nothing end

    if sample_type isa UniformGridSamples
        totaltasks = num_points_to_sample isa Int ? num_points_to_sample^model.core.num_pars : prod(num_points_to_sample)
    else
        totaltasks = num_points_to_sample
    end

    channel_buffer_size = min(ceil(Int, totaltasks*0.2), 400)
    channel = RemoteChannel(() -> Channel{Bool}(channel_buffer_size))
    p = Progress(totaltasks; desc="Computing full likelihood samples: ",
                dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    local sample_struct::SampledConfidenceStruct
    @sync begin
        @async while take!(channel)
            next!(p)
        end

        @async begin
            try
                sample_struct = full_likelihood_sample(model, num_points_to_sample, confidence_level, sample_type, lb, ub, use_threads, use_distributed, channel)
            catch
                @error string("an error occurred when computing a full likelihood sample with settings: ",
                    (sample_type=sample_type, confidence_level=confidence_level))
                for (exc, bt) in current_exceptions()
                    showerror(stdout, exc, bt)
                    println(stdout)
                    println(stdout)
                end
                return nothing
            end
            put!(channel, false)
        end
    end

    num_points_kept = length(sample_struct.ll)
    
    if num_points_kept == 0
        @warn "no sampled points were in the confidence region of the full likelihood within the supplied bounds: try increasing num_points_to_sample or changing the bounds"
        return nothing
    end

    if requires_overwrite
        row_ind = model.dim_samples_row_exists[sample_type][confidence_level]
    else
        model.num_dim_samples += 1
        row_ind = model.num_dim_samples * 1
        if (model.num_dim_samples - nrow(model.dim_samples_df)) > 0
            add_dim_samples_rows!(model, 1)
        end
        model.dim_samples_row_exists[sample_type][confidence_level] = row_ind
    end

    model.dim_samples_dict[row_ind] = sample_struct
    set_dim_samples_row!(model, row_ind, collect(1:model.core.num_pars), true, confidence_level, sample_type, num_points_kept)

    return nothing
end