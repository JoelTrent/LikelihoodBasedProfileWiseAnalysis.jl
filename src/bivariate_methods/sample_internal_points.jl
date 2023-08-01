"""
    update_biv_dict_internal!(model::LikelihoodModel,
        biv_row_number::Int,
        points::PointsAndLogLikelihood)

Updates the `internal_points` field of a [`BivariateConfidenceStruct`](@ref), for the profile related to `biv_row_number` stored at `model.biv_profiles_dict[biv_row_number]`, with the internal points stored in `points`.
"""
function update_biv_dict_internal!(model::LikelihoodModel,
    biv_row_number::Int,
    points::PointsAndLogLikelihood)

    interval_struct = model.biv_profiles_dict[biv_row_number]
    model.biv_profiles_dict[biv_row_number] = @set interval_struct.internal_points = points

    return nothing
end

"""
    sample_internal_points_LHC(model::LikelihoodModel, 
        target_num_points::Int, 
        θindices::Tuple{Int,Int}, 
        profile_type::AbstractProfileType,
        conf_struct::BivariateConfidenceStruct, 
        confidence_level::Float64,
        boundary_not_ordered::Bool,
        hullmethod::AbstractBivariateHullMethod,
        use_threads::Bool)

Given a `hullmethod` of type [`AbstractBivariateHullMethod`](@ref) which creates a 2D polygon hull from a set of boundary and internal points (method dependent) in `conf_struct` as a representation of the true confidence boundary, sample points from the bounding box, with edges parallel to x and y axes, of a 2D polygon hull using a heuristically optimised Latin Hypercube sampling plan to find approximately `target_num_points` within the polygon, rejecting any that are not inside the log-likelihood threshold at that `confidence_level` and `profile_type`.   
"""
function sample_internal_points_LHC(model::LikelihoodModel,
                                    target_num_points::Int,
                                    θindices::Tuple{Int,Int},
                                    profile_type::AbstractProfileType,
                                    conf_struct::BivariateConfidenceStruct,
                                    confidence_level::Float64,
                                    boundary_not_ordered::Bool,
                                    hullmethod::AbstractBivariateHullMethod,
                                    optimizationsettings::OptimizationSettings,
                                    use_threads::Bool)

    bivariate_optimiser = get_bivariate_opt_func(profile_type, RadialMLEMethod())
    biv_opt_is_ellipse_analytical = bivariate_optimiser == bivariateψ_ellipse_analytical_vectorsearch
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)
    
    ind1, ind2 = θindices
    newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(model, ind1, ind2)

    q=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, ωranges=ωranges, consistent=consistent)
    
    # Use shoelace algorithm to determine polygon area (similar to LazySets.jl `_area_polygon` implementation)
    function polygon_area(polygon::AbstractMatrix)
        n = size(polygonhull, 2)
        @inbounds A = polygon[1,n]*polygon[2,1] - polygon[1,1]*polygon[2,n]
        for i in 1:(n-1)
            @inbounds A += polygon[1,i]*polygon[2,i+1] - polygon[1,i+1]*polygon[2,i]
        end
        return abs(A/2)
    end

    polygonhull = construct_polygon_hull(model, [ind1, ind2], conf_struct, confidence_level,
        boundary_not_ordered, hullmethod, true)

    poly_area = polygon_area(polygonhull)
    lb, ub = vec(minimum(polygonhull, dims=2)), vec(maximum(polygonhull, dims=2))
    bbox_area = (ub[1]-lb[1]) * (ub[2]-lb[2])
    
    if isapprox(0.0, poly_area, atol=1e-14) || isapprox(0.0, bbox_area, atol=1e-14)
        @warn string("Polygon hull or bounding box of hull have an area of approximately zero for parameters ", model.core.θnames[ind1]," and ", model.core.θnames[ind2], ". No internal points were found.")
        return conf_struct.internal_points, 1.0
    end
    
    polygonhull_num_points = size(polygonhull, 2)
    nodes = permutedims(polygonhull)
    edges = zeros(Int, polygonhull_num_points, 2)
    for i in 1:polygonhull_num_points-1
        edges[i,:] .= i, i+1
    end
    edges[end,:] .= polygonhull_num_points, 1
    
    num_points = ceil(Int, target_num_points * (bbox_area / poly_area))
    scale_range = [(lb[i], ub[i]) for i in 1:2]
    sample_points = permutedims(scaleLHC(LHCoptim(num_points, 2, 100)[1], scale_range))
    is_internal = falses(num_points)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_points)
    let is_internal = falses(num_points)
        let sample_points=sample_points
            @floop ex for i in axes(sample_points, 2)
                is_inside, is_boundary = inpoly2(sample_points[:, i], nodes, edges)

                if is_inside || is_boundary
                    is_internal[i] = true
                end
            end
        end
        sample_points = sample_points[:, is_internal]
    end

    num_sample_points = size(sample_points,2)
    _internal_points = zeros(model.core.num_pars, num_sample_points)
    _ll = zeros(num_sample_points)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_sample_points)
    let sample_points=sample_points
        @floop ex for i in axes(sample_points, 2)
            FLoops.@init pointa = zeros(2)
            FLoops.@init uhat = zeros(2)
            FLoops.@init ω_opt = zeros(model.core.num_pars - 2)
            p = (ω_opt=ω_opt, pointa=pointa, uhat=uhat, q=q, options=optimizationsettings)

            p.pointa .= sample_points[:, i]
            _ll[i] = bivariate_optimiser(0.0, p)

            if _ll[i] ≥ 0.0
                _internal_points[[ind1, ind2], i] .= p.pointa
                if !biv_opt_is_ellipse_analytical
                    variablemapping!(@view(_internal_points[:, i]), p.ω_opt, θranges, ωranges)
                end
            end
        end
    end
    is_internal = _ll .≥ 0.0
    internal_points = _internal_points[:, is_internal]
    ll = _ll[is_internal]
    rejection_rate = (num_sample_points - sum(is_internal)) / num_sample_points

    ll .= ll .+ get_target_loglikelihood(model, confidence_level, EllipseApproxAnalytical(), 2)

    if biv_opt_is_ellipse_analytical
        get_ωs_bivariate_ellipse_analytical!(@view(internal_points[[ind1, ind2], :]), size(internal_points, 2),
            consistent, ind1, ind2,
            model.core.num_pars, initGuess,
            θranges, ωranges, 
            optimizationsettings, use_threads, internal_points)
    end

    return merge(conf_struct.internal_points, PointsAndLogLikelihood(internal_points, ll)), rejection_rate
end

"""
    sample_internal_points_uniform_random(model::LikelihoodModel, 
        num_points::Int, 
        θindices::Tuple{Int,Int}, 
        profile_type::AbstractProfileType,
        conf_struct::BivariateConfidenceStruct, 
        confidence_level::Float64,
        boundary_not_ordered::Bool,
        hullmethod::AbstractBivariateHullMethod,
        optimizationsettings::OptimizationSettings,
        use_threads::Bool)

Given a `hullmethod` of type [`AbstractBivariateHullMethod`](@ref) which creates a 2D polygon hull from a set of boundary and internal points (method dependent) in `conf_struct` as a representation of the true confidence boundary, sample points from the polygon hull homogeneously until `num_points` are found, rejecting any that are not inside the log-likelihood threshold at that `confidence_level` and `profile_type`.   
"""
function sample_internal_points_uniform_random(model::LikelihoodModel,
                                                num_points::Int, 
                                                θindices::Tuple{Int,Int}, 
                                                profile_type::AbstractProfileType,
                                                conf_struct::BivariateConfidenceStruct, 
                                                confidence_level::Float64,
                                                boundary_not_ordered::Bool,
                                                hullmethod::AbstractBivariateHullMethod,
                                                optimizationsettings::OptimizationSettings,
                                                use_threads::Bool)

    bivariate_optimiser = get_bivariate_opt_func(profile_type, RadialMLEMethod())
    biv_opt_is_ellipse_analytical = bivariate_optimiser == bivariateψ_ellipse_analytical_vectorsearch
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)

    ind1, ind2 = θindices
    newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(model, ind1, ind2)

    q=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, ωranges=ωranges, consistent=consistent)

    mesh = construct_polygon_hull(model, [ind1, ind2], conf_struct, confidence_level,
                                    boundary_not_ordered, hullmethod, false)
    
    internal_points = zeros(model.core.num_pars, num_points)
    ll = zeros(num_points)
    i=1
    num_rejected=0
    while i ≤ num_points
        num_sample_points = num_points-(i-1)
        sample_points = reduce(hcat, [point.coords for point in collect(sample(mesh, HomogeneousSampling(num_sample_points)))])
        
        let is_rejected = falses(num_sample_points), _ll = zeros(num_sample_points), _internal_points = zeros(model.core.num_pars, num_sample_points)
            ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_sample_points)
            @floop ex for j in axes(sample_points, 2)
                FLoops.@init pointa = zeros(2)
                FLoops.@init uhat = zeros(2)
                FLoops.@init ω_opt = zeros(model.core.num_pars - 2)
                p = (ω_opt=ω_opt, pointa=pointa, uhat=uhat, q=q, options=optimizationsettings)

                p.pointa .= sample_points[:,j]
                _ll[j] = bivariate_optimiser(0.0, p)

                if _ll[j] ≥ 0.0
                    _internal_points[[ind1, ind2], j] .= p.pointa
                    if !biv_opt_is_ellipse_analytical
                        variablemapping!(@view(_internal_points[:, j]), p.ω_opt, θranges, ωranges)
                    end
                else
                    is_rejected[j] = true
                end
            end
            new_num_rejected = sum(is_rejected)
            is_accepted = .!is_rejected
            num_rejected += new_num_rejected
            num_accepted = num_sample_points-new_num_rejected

            ll[i:(i+num_accepted-1)] .= _ll[is_accepted]
            internal_points[:, i:(i+num_accepted-1)] .= _internal_points[:, is_accepted]
            i += num_accepted
        end
    end

    rejection_rate = num_rejected / (num_rejected+i-1)
    ll .= ll .+ get_target_loglikelihood(model, confidence_level, EllipseApproxAnalytical(), 2)

    if biv_opt_is_ellipse_analytical
        get_ωs_bivariate_ellipse_analytical!(@view(internal_points[[ind1, ind2], :]), num_points,
            consistent, ind1, ind2,
            model.core.num_pars, initGuess,
            θranges, ωranges,
            optimizationsettings, use_threads, internal_points)
    end

    return merge(conf_struct.internal_points, PointsAndLogLikelihood(internal_points, ll)), rejection_rate
end

"""
    sample_internal_points_single_row(model::LikelihoodModel,
        sub_df::Union{DataFrame, SubDataFrame},
        i::Int,
        biv_row_number::Int, 
        num_points::Int, 
        sample_type::AbstractSampleType,
        hullmethod::AbstractBivariateHullMethod, 
        t::Union{AbstractVector,Missing},
        evaluate_predictions_for_samples::Bool,
        proportion_of_predictions_to_keep::Real,
        optimizationsettings::OptimizationSettings,
        use_threads::Bool,
        channel::RemoteChannel)

Sample internal points from the bivariate profile given by a valid row number in `model.biv_profiles_df` using either homogeneous sampling (`UniformRandomSamples()`) to find exactly `num_points` or a Latin Hypercube sampling plan (`LatinHypercubeSamples()`) to find approximately `num_points`.
"""
function sample_internal_points_single_row(model::LikelihoodModel,
    sub_df::Union{DataFrame, SubDataFrame},
    i::Int,
    biv_row_number::Int, 
    num_points::Int, 
    sample_type::AbstractSampleType,
    hullmethod::AbstractBivariateHullMethod, 
    t::Union{AbstractVector,Missing},
    evaluate_predictions_for_samples::Bool,
    proportion_of_predictions_to_keep::Real,
    optimizationsettings::OptimizationSettings,
    use_threads::Bool,
    channel::RemoteChannel)

    try
        θindices = model.biv_profiles_df.θindices[biv_row_number]
        profile_type = model.biv_profiles_df.profile_type[biv_row_number]
        conf_struct = model.biv_profiles_dict[biv_row_number]
        confidence_level = model.biv_profiles_df[biv_row_number, :conf_level]
        boundary_not_ordered = model.biv_profiles_df.boundary_not_ordered[biv_row_number]

        @timeit_debug timer "Sample bivariate internal points" begin
            if sample_type isa LatinHypercubeSamples
                internal_points, rejection_rate = sample_internal_points_LHC(model, num_points, θindices,
                    profile_type, conf_struct, confidence_level, boundary_not_ordered, hullmethod,
                    optimizationsettings, use_threads)
            end
            internal_points, rejection_rate = sample_internal_points_uniform_random(model, num_points, θindices,
                profile_type, conf_struct, confidence_level, boundary_not_ordered, hullmethod,
                optimizationsettings, use_threads)
        end

        if evaluate_predictions_for_samples && !sub_df[i, :not_evaluated_predictions]
            predict_struct = model.biv_predictions_dict[biv_row_number]

            new_predict_struct = generate_prediction(model.core.predictfunction,
                model.core.data, t, model.core.ymle,
                internal_points.points[:, (end-num_points+1):end],
                proportion_of_predictions_to_keep)

            merged_predict_struct = merge(predict_struct, new_predict_struct)
        else
            merged_predict_struct = missing
        end
        put!(channel, true)

        return internal_points, rejection_rate, merged_predict_struct
    catch
        @error string("an error occurred when finding the points inside a bivariate confidence boundary with settings: ",
            (biv_row_number=biv_row_number, num_points=num_points,
                sample_type=sample_type, hullmethod=hullmethod))
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
            println(stdout)
        end
    end
    return nothing
end

"""
    sample_bivariate_internal_points!(model::LikelihoodModel,
        num_points::Int;
        <keyword arguments>)

Samples `num_points` internal points in interest parameter space of existing bivariate profiles that meet the requirements of the bivariate method of [`PlaceholderLikelihood.desired_df_subset`](@ref) (see Keyword Arguments). Modifies `model` in place, with sampled internal points appended to the internal points field of each [`BivariateConfidenceStruct`](@ref).

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `num_points`: number of internal points to sample from within a polygon hull approximation of a bivariate boundary and append to it's array of internal points.

# Keyword Arguments
- `confidence_levels`: a vector of confidence levels. If empty, all confidence levels of bivariate profiles will be considered for finding interval points. Otherwise, only confidence levels in `confidence_levels` will be considered. Default is `Float64[]` (any confidence level).
- `profile_types`: a vector of [`AbstractProfileType`](@ref) structs. If empty, all profile types of bivariate profiles are considered. Otherwise, only profiles with matching profile types will be considered. Default is `AbstractProfileType[]` (any profile type).
- `methods`: a vector of [`AbstractBivariateMethod`](@ref) structs. If empty all methods used to find bivariate profiles are considered. Otherwise, only profiles with matching method types will be considered (struct arguments do not need to be the same). Default is `AbstractBivariateMethod[]` (any bivariate method).
- `hullmethod`: method of type [`AbstractBivariateHullMethod`](@ref) used to create a 2D polygon hull that approximates the bivariate boundary from a set of boundary points and internal points (method dependent). For available methods see [`bivariate_hull_methods()`](@ref). Default is `MPPHullMethod()` ([`MPPHullMethod`](@ref)).
- `sample_type`: either a [`UniformRandomSamples`](@ref) or [`LatinHypercubeSamples`](@ref) struct for how to sample internal points from the polygon hull. [`UniformRandomSamples`](@ref) are homogeneously sampled from the polygon and [`LatinHypercubeSamples`](@ref) use the intersection of a heuristically optimised Latin Hypercube sampling plan with the polygon. Default is `LatinHypercubeSamples()` ([`LatinHypercubeSamples`](@ref)).
- `t`: vector of timepoints to evaluate predictions at for each new sampled internal point from a bivariate boundary that has already had predictions evaluated. The vector must be the same vector used to produce these previous predictions, otherwise points will not be sampled from this boundary. Default is `missing`.
- `evaluate_predictions_for_samples`: boolean variable specifying whether to evaluate predictions for sampled points given predictions have been evaluated for the boundary they were sampled from. If `false`, then existing predictions will be forgotten by the `model` and overwritten the next time predictions are evaluated for each profile internal points were sampled from. Default is `true`.
- `proportion_of_predictions_to_keep`: The proportion of predictions from `num_points` internal points to save. Does not impact the extrema calculated from predictions. Default is `1.0`.
- `optimizationsettings`: a [`OptimizationSettings`](@ref) struct containing the optimisation settings used to find optimal values of nuisance parameters for a given pair of interest parameter values. Default is `missing` (will use `model.core.optimizationsettings`).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of `θs_to_profile` completed and estimated time of completion. Default is `model.show_progress`.
- `use_distributed`: boolean variable specifying whether to use a normal for loop or a `@distributed` for loop across combinations of interest parameters. Set this variable to `false` if [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is not being used. Default is `true`.
- `use_threads`: boolean variable specifying, if the number of workers for distributed computing is not greater than 1 (`!Distributed.nworkers()>1`), to use a parallelised for loop across `Threads.nthreads()` threads to evaluate the log-likelihood at each sampled point. Default is `false`.

# Details

For each bivariate profile that meets the requirements of [`PlaceholderLikelihood.desired_df_subset`](@ref) it creates a 2D polygon hull from it's set of boundary and internal points (method dependent) using `hullmethod` and samples points from the hull using `sample_type` until `num_points` are found, rejecting any that are not inside the log-likelihood threshold at that `confidence_level` and `profile_type`. For [`LatinHypercubeSamples`](@ref) this will be approximately `num_points`, whereas for [`UniformRandomSamples`](@ref) this will be exactly `num_points`. Nuisance parameters of each point in bivariate interest parameter space are found by maximising the log-likelihood function given by the `profile_type` of the profile.

It is highly recommended to view the docstrings of each `hullmethod` as the rejection rate of sampled points and the representation accuracy / coverage of the true confidence boundary varies between them, which can impact both computational performance and sampling coverage. For example, given the same set of boundary and internal points, [`ConvexHullMethod`](@ref) will produce a polygon hull that contains at least as much of the true confidence boundary as the other methods, but may have a higher rejection rate than other methods leading to higher computational cost.

## Parallel Computing Implementation

If [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is being used and `use_distributed` is `true` then the internal samples of distinct interest parameter combinations will be computed in parallel across `Distributed.nworkers()` workers. If `use_distributed` is `false` and `use_threads` is `true` then the internal samples of each distinct interest parameter combination will be computed in parallel across `Threads.nthreads()` threads. It is highly recommended to set `use_threads` to `true` in that situation.

## Iteration Speed Of the Progress Meter

An iteration within the progress meter is specified as the time it takes for all internal points within a bivariate boundary to be found.
"""
function sample_bivariate_internal_points!(model::LikelihoodModel,
                                    num_points::Int;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                    sample_type::AbstractSampleType=LatinHypercubeSamples(),
                                    hullmethod::AbstractBivariateHullMethod=MPPHullMethod(),
                                    t::Union{AbstractVector, Missing}=missing,
                                    evaluate_predictions_for_samples::Bool=true,
                                    proportion_of_predictions_to_keep::Real=1.0,
                                    optimizationsettings::Union{OptimizationSettings,Missing}=missing,
                                    show_progress::Bool=model.show_progress,
                                    use_distributed::Bool=true,
                                    use_threads::Bool=false)

    function argument_handling()
        0 < num_points || throw(DomainError("num_points must be a strictly positive integer"))
        model.core isa CoreLikelihoodModel || throw(ArgumentError("model does not contain a log-likelihood function. Add it using add_loglikelihood_function!"))
        sample_type isa UniformGridSamples && throw(ArgumentError("sample_bivariate_internal_points! is not defined for sample_type=UniformGridSamples()"))

        (!use_distributed && use_threads && timeit_debug_enabled()) &&
            throw(ArgumentError("use_threads cannot be true when debug timings from TimerOutputs are enabled and use_distributed is false. Either set use_threads to false or disable debug timings using `PlaceholderLikelihood.TimerOutputs.disable_debug_timings(PlaceholderLikelihood)`"))
        return nothing
    end

    argument_handling()
    optimizationsettings = ismissing(optimizationsettings) ? model.core.optimizationsettings : optimizationsettings
    
    sub_df = desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, Tuple{Int,Int}[], 
                confidence_levels, profile_types, methods)

    if nrow(sub_df) < 1
        return nothing
    end

    predictions_evaluated = .!sub_df.not_evaluated_predictions
    if evaluate_predictions_for_samples && any(predictions_evaluated)
        row_subset = falses(sum(predictions_evaluated))
        for (i, row_ind) in enumerate(sub_df[predictions_evaluated, :row_ind])
            if check_prediction_function_exists(model::LikelihoodModel) === false
                continue
            end

            if ismissing(t)
                @warn string("t must be specified to evaluate predictions for internal points (and be the same as the t vector used to generate previous predictions with generate_prediction_bivariate!). Bivariate profile row ", row_ind, " not evaluated")
                continue
            end

            predict_struct = model.biv_predictions_dict[row_ind]
            if size(predict_struct.predictions, 1) != length(t)
                @warn string("additional samples not evaluated for the bivariate profile at row ", row_ind, " because the length of t is not the same as the length of existing predictions")
                continue
            end
            row_subset[i] = true
        end

        sub_df = @view(sub_df[row_subset, :])
        if nrow(sub_df) < 1
            return nothing
        end
    end

    rejection_df = DataFrame(rejection_rate=zeros(nrow(sub_df)), biv_df_row_ind=zeros(Int, nrow(sub_df)))

    p = Progress(nrow(sub_df); desc="Computing points inside bivariate confidence boundaries: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)
    channel = RemoteChannel(() -> Channel{Bool}(1))

    @sync begin
        @async while take!(channel)
            next!(p)
        end
        @async begin
            if use_distributed
                internal_samples = @distributed (vcat) for i in 1:nrow(sub_df)
                    [(i, sub_df[i, :row_ind], 
                        sample_internal_points_single_row(model, sub_df, i, sub_df[i, :row_ind], num_points, sample_type, 
                            hullmethod, t, evaluate_predictions_for_samples, proportion_of_predictions_to_keep, 
                            optimizationsettings, use_threads, channel))]
                end
                
                for (i, row_ind, samples) in internal_samples
                    if isnothing(samples); continue end
                    
                    internal_points, rejection_rate, merged_predict_struct = samples
                    if ismissing(internal_points); continue end
                    
                    update_biv_dict_internal!(model, row_ind, internal_points)
                    rejection_df[i, :] .= rejection_rate, row_ind
                    sub_df[i, :not_evaluated_internal_points] = false
                    
                    if ismissing(merged_predict_struct)
                        sub_df[i, :not_evaluated_predictions] = true
                        continue
                    end
                    model.biv_predictions_dict[row_ind] = merged_predict_struct
                end
            else
                for i in 1:nrow(sub_df)
                    row_ind = sub_df[i, :row_ind]
                    samples = sample_internal_points_single_row(model, sub_df, i, sub_df[i, :row_ind], num_points, sample_type, 
                            hullmethod, t, evaluate_predictions_for_samples, proportion_of_predictions_to_keep, 
                            optimizationsettings, use_threads, channel)
                
                    if isnothing(samples); continue end
                    
                    internal_points, rejection_rate, merged_predict_struct = samples
                    if ismissing(internal_points); continue end
                    
                    update_biv_dict_internal!(model, row_ind, internal_points)
                    rejection_df[i, :] .= rejection_rate, row_ind
                    sub_df[i, :not_evaluated_internal_points] = false
                    
                    if ismissing(merged_predict_struct)
                        sub_df[i, :not_evaluated_predictions] = true
                        continue
                    end
                    model.biv_predictions_dict[row_ind] = merged_predict_struct
                end
            end
            put!(channel, false)
        end
    end

    rejection_df = rejection_df[rejection_df.biv_df_row_ind .!= 0, :]

    return rejection_df
end