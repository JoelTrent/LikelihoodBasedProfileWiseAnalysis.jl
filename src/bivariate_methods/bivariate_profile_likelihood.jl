"""
    get_bivariate_confidence_set(model::LikelihoodModel, biv_row_number::Int)

Returns the [`BivariateConfidenceStruct`](@ref) corresponding to the profile in row `biv_row_number` of `model.biv_profiles_df`
"""
function get_bivariate_confidence_set(model::LikelihoodModel, biv_row_number::Int)
    return model.biv_profiles_dict[biv_row_number]
end

"""
    add_biv_profiles_rows!(model::LikelihoodModel, num_rows_to_add::Int)

Adds `num_rows_to_add` rows to `model.biv_profiles_df`. 
"""
function add_biv_profiles_rows!(model::LikelihoodModel, num_rows_to_add::Int)
    new_rows = init_biv_profiles_df(num_rows_to_add, 
                                    existing_largest_row=nrow(model.biv_profiles_df))

    model.biv_profiles_df = vcat(model.biv_profiles_df, new_rows)
    return nothing
end

"""
    set_biv_profiles_row!(model::LikelihoodModel, 
        row_ind::Int, 
        θcombination::Tuple{Int, Int},
        not_evaluated_internal_points::Bool, 
        not_evaluated_predictions::Bool,
        boundary_not_ordered::Bool,
        confidence_level::Float64, 
        dof::Int,
        profile_type::AbstractProfileType,
        method::AbstractBivariateMethod, 
        num_points::Int)

Sets the columns of row `row_ind` of `model.biv_profiles_df` to contain the relevant info about a just conducted profile. `model.biv_profiles_dict` contains the profile for row `row_ind` at key `row_ind`.  
"""
function set_biv_profiles_row!(model::LikelihoodModel,
                                    row_ind::Int,
                                    θcombination::Tuple{Int, Int},
                                    not_evaluated_internal_points::Bool,
                                    not_evaluated_predictions::Bool,
                                    boundary_not_ordered::Bool,
                                    confidence_level::Float64,
                                    dof::Int,
                                    profile_type::AbstractProfileType,
                                    method::AbstractBivariateMethod,
                                    num_points::Int)
    model.biv_profiles_df[row_ind, 2:end-1] .= θcombination, 
                                                not_evaluated_internal_points,
                                                not_evaluated_predictions,
                                                boundary_not_ordered,
                                                confidence_level,
                                                dof,
                                                profile_type,
                                                method,
                                                num_points
    return nothing
end

"""
    get_bivariate_opt_func(profile_type::AbstractProfileType, method::AbstractBivariateMethod)

Returns the correct bivariate optimisation function used to find the optimal values of nuisance parameters at a set of interest parameters for the `profile_type` log-likelihood function. The optimisation function returns the value of the `profile_type` log-likelihood function as well as finding the optimal nuisance parameters and saving these in one of it's inputs.
"""
function get_bivariate_opt_func(profile_type::AbstractProfileType, method::AbstractBivariateMethod)
    if method isa AnalyticalEllipseMethod
        return bivariateψ_ellipse_analytical
    elseif method isa Fix1AxisMethod
        if profile_type isa EllipseApproxAnalytical
            return bivariateψ_ellipse_analytical
        elseif profile_type isa LogLikelihood || profile_type isa EllipseApprox
            return bivariateψ!
        end

    elseif method isa AbstractBivariateVectorMethod
        if profile_type isa EllipseApproxAnalytical
            return bivariateψ_ellipse_analytical_vectorsearch
        elseif profile_type isa LogLikelihood || profile_type isa EllipseApprox
            return bivariateψ_vectorsearch!
        end
    end

    return missing
end

"""
    get_ωs_bivariate_ellipse_analytical!(boundary,
        num_points::Int,
        consistent::NamedTuple, 
        ind1::Int, 
        ind2::Int, 
        num_pars::Int,
        initGuess::Vector{<:Float64}, 
        θranges::Tuple{T, T, T}, 
        ωranges::Tuple{T, T, T},
        optimizationsettings::OptimizationSettings,
        samples_all_pars::Union{Missing, Matrix{Float64}}=missing) where T<:UnitRange

Determines the nuisance parameters for a [`EllipseApproxAnalytical`](@ref) boundary profile by optimising over the unbounded ellipse approximation of the log-likelihood centred at the MLE. At higher confidence levels, where the ellipse approximation is less accurate, it is likely that predictions produced by running the model with these optimised nuisance parameters will be unrealistic and/or the parameters themselves may be infeasible for the model. 
"""
function get_ωs_bivariate_ellipse_analytical!(boundary,
                                                num_points::Int,
                                                consistent::NamedTuple, 
                                                ind1::Int, 
                                                ind2::Int, 
                                                num_pars::Int,
                                                initGuess::Vector{<:Float64}, 
                                                θranges::Tuple{T, T, T}, 
                                                ωranges::Tuple{T, T, T},
                                                optimizationsettings::OptimizationSettings,
                                                use_threads::Bool,
                                                samples_all_pars::Union{Missing, Matrix{Float64}}=missing) where T<:UnitRange

    q=(ind1=ind1, ind2=ind2, initGuess=initGuess,
        θranges=θranges, ωranges=ωranges, consistent=consistent)
    p=(q=q, options=optimizationsettings)
    
    if ismissing(samples_all_pars)
        samples_all_pars = zeros(num_pars, num_points)
        samples_all_pars[[ind1, ind2], :] .= boundary
    end
    
    _samples_all_pars=samples_all_pars
    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_points)
    @floop ex for i in 1:num_points
        variablemapping!(@view(_samples_all_pars[:, i]), bivariateψ_ellipse_unbounded(boundary[:,i], p), θranges, ωranges)
    end

    return _samples_all_pars
end

"""
    bivariate_confidenceprofile(bivariate_optimiser::Function,
        model::LikelihoodModel, 
        num_points::Int,
        confidence_level::Float64,
        consistent::NamedTuple,
        ind1::Int,
        ind2::Int,
        dof::Int,
        profile_type::AbstractProfileType,
        method::AbstractBivariateMethod,
        θlb_nuisance::AbstractVector{<:Real},
        θub_nuisance::AbstractVector{<:Real},
        mle_targetll::Float64,
        save_internal_points::Bool,
        find_zero_atol::Real,
        optimizationsettings::OptimizationSettings,
        channel::RemoteChannel)

Returns a [`BivariateConfidenceStruct`](@ref) containing the `num_points` boundary points and internal points (if `save_internal_points=true`) for the specified combination of parameters `ind1` and `ind2`, and `profile_type` at `confidence_level` using `method`. Calls the desired `method`. Called by [`bivariate_confidenceprofiles!`](@ref).
"""
function bivariate_confidenceprofile(bivariate_optimiser::Function,
                                        model::LikelihoodModel, 
                                        num_points::Int,
                                        confidence_level::Float64,
                                        consistent::NamedTuple,
                                        ind1::Int,
                                        ind2::Int,
                                        dof::Int,
                                        profile_type::AbstractProfileType,
                                        method::AbstractBivariateMethod,
                                        θlb_nuisance::AbstractVector{<:Real},
                                        θub_nuisance::AbstractVector{<:Real},
                                        mle_targetll::Float64,
                                        save_internal_points::Bool,
                                        find_zero_atol::Real,
                                        optimizationsettings::OptimizationSettings,
                                        use_threads::Bool,
                                        channel::RemoteChannel)

    try 
        @timeit_debug timer "Bivariate confidence boundary" begin
            internal=PointsAndLogLikelihood(zeros(model.core.num_pars,0), zeros(0))
            if method isa AnalyticalEllipseMethod
                boundary_ellipse = generate_N_clustered_points(
                                            num_points, consistent.data_analytic.Γmle, 
                                            consistent.data_analytic.θmle, ind1, ind2,
                                            confidence_level=confidence_level,
                                            dof=dof,
                                            start_point_shift=method.ellipse_start_point_shift,
                                            sqrt_distortion=method.ellipse_sqrt_distortion)

                _, _, initGuess, θranges, ωranges = init_nuisance_parameters(model, ind1, ind2, θlb_nuisance, θub_nuisance)

                boundary = get_ωs_bivariate_ellipse_analytical!(
                                    boundary_ellipse, 
                                    num_points,
                                    consistent, ind1, ind2, 
                                    model.core.num_pars, initGuess,
                                    θranges, ωranges,
                                    optimizationsettings, use_threads)

                put!(channel, true)
                
            elseif method isa Fix1AxisMethod
                boundary, internal = bivariate_confidenceprofile_fix1axis(
                                        bivariate_optimiser, model, 
                                        num_points, consistent, ind1, ind2,
                                        θlb_nuisance, θub_nuisance,
                                        mle_targetll, save_internal_points,
                                        find_zero_atol, optimizationsettings, 
                                        use_threads, channel)
                
            elseif method isa SimultaneousMethod
                boundary, internal = bivariate_confidenceprofile_vectorsearch(
                                        bivariate_optimiser, model, 
                                        num_points, consistent, ind1, ind2, dof,
                                        θlb_nuisance, θub_nuisance,
                                        mle_targetll, save_internal_points, 
                                        find_zero_atol, optimizationsettings,
                                        use_threads, channel,
                                        min_proportion_unique=method.min_proportion_unique,
                                        use_MLE_point=method.use_MLE_point)

            elseif method isa RadialRandomMethod
                boundary, internal = bivariate_confidenceprofile_vectorsearch(
                                        bivariate_optimiser, model, 
                                        num_points, consistent, ind1, ind2, dof,
                                        θlb_nuisance, θub_nuisance,
                                        mle_targetll, save_internal_points, 
                                        find_zero_atol, optimizationsettings,
                                        use_threads, channel,
                                        num_radial_directions=method.num_radial_directions,
                                        use_MLE_point=method.use_MLE_point)

            elseif method isa RadialMLEMethod
                boundary, internal = bivariate_confidenceprofile_vectorsearch(
                                        bivariate_optimiser, model, 
                                        num_points, consistent, ind1, ind2, dof,
                                        θlb_nuisance, θub_nuisance,
                                        mle_targetll, save_internal_points, 
                                        find_zero_atol, optimizationsettings,
                                        use_threads, channel,
                                        ellipse_confidence_level=confidence_level,
                                        ellipse_start_point_shift=method.ellipse_start_point_shift,
                                        ellipse_sqrt_distortion=method.ellipse_sqrt_distortion)
            
            elseif method isa IterativeBoundaryMethod
                boundary, internal = bivariate_confidenceprofile_iterativeboundary(
                                        bivariate_optimiser, model,
                                        num_points, consistent, ind1, ind2, dof,
                                        θlb_nuisance, θub_nuisance,
                                        method.initial_num_points, method.angle_points_per_iter,
                                        method.edge_points_per_iter, method.radial_start_point_shift,
                                        method.ellipse_sqrt_distortion, confidence_level, 
                                        method.use_ellipse,
                                        mle_targetll, save_internal_points,
                                        find_zero_atol, optimizationsettings, 
                                        use_threads, channel)
            end

            return BivariateConfidenceStruct(boundary, internal)
        end
    catch
        @error string("an error occurred when finding the bivariate boundary with settings: ",
            (profile_type=profile_type, method=method, confidence_level=confidence_level, 
            θcombination=[ind1, ind2]))
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
            println(stdout)
        end
    end

    return nothing
end

function get_bivariate_method_tasknumbers(method::AbstractBivariateMethod, num_points::Int)
    if method isa AnalyticalEllipseMethod
        return 1
    else
        return num_points
    end
end

"""
    bivariate_confidenceprofiles!(model::LikelihoodModel, 
        θcombinations::Vector{Vector{Int}}, 
        num_points::Int; 
        <keyword arguments>)

Finds `num_points` `profile_type` boundary points at a specified `confidence_level` for each combination of two interest parameters using a specified `method`, optionally saving any found internal points. Saves these profiles by modifying `model` in place.
    
# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `θcombinations`: vector of pairs of parameters to profile, as a vector of vectors of model parameter indexes.
- `num_points`: positive number of points to find on the boundary at the specified confidence level. Depending on the method, if a region of the user-provided bounds is inside the boundary some of these points will be on the bounds and inside the boundary. Set to at least 3 within the function as some methods need at least three points to work.

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level on which to find the `profile_type` boundary. Default is `0.95` (95%).
- `dof`: an integer ∈ [2, `model.core.num_pars`] for the degrees of freedom used to define the asymptotic threshold ([`LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood`](@ref)) which defines the boundary of the bivariate profile. For bivariate profiles that are considered individually, it should be set to `2`. For profiles that are considered simultaneously, it should be set to `model.core.num_pars`. Default is `2`. Setting it to `model.core.num_pars` should be reasonable when making predictions for well-identified models with `<10` parameters. Note: values other than `2` and `model.core.num_pars` may not have a clear statistical interpretation.
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `method`: a method of type [`AbstractBivariateMethod`](@ref). For a list of available methods use `bivariate_methods()` ([`bivariate_methods`](@ref)). Default is `RadialRandomMethod(5)` ([`RadialRandomMethod`](@ref)).
- `θlb_nuisance`: a vector of lower bounds on nuisance parameters, require `θlb_nuisance .≤ model.core.θmle`. Default is `model.core.θlb`. 
- `θub_nuisance`: a vector of upper bounds on nuisance parameters, require `θub_nuisance .≥ model.core.θmle`. Default is `model.core.θub`.
- `save_internal_points`: boolean variable specifying whether to save points found inside the boundary during boundary computation. Internal points can be plotted in bivariate profile plots and will be used to generate predictions from a given bivariate profile. Default is `true`.
- `existing_profiles`: `Symbol ∈ [:ignore, :merge, :overwrite]` specifying what to do if profiles already exist for a given `θcombination`, `confidence_level`, `profile_type` and `method`. See below for each symbol's meanings. Default is `:merge`.
- `find_zero_atol`: a `Real` number greater than zero for the absolute tolerance of the log-likelihood function value from the target value to be used when searching for confidence intervals. Default is `model.find_zero_atol`.
- `optimizationsettings`: a [`OptimizationSettings`](@ref) struct containing the optimisation settings used to find optimal values of nuisance parameters for a given pair of interest parameter values. Default is `missing` (will use `model.core.optimizationsettings`).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of `θcombinations` completed and estimated time of completion. Default is `model.show_progress`.
- `use_distributed`: boolean variable specifying whether to use a normal for loop or a `@distributed` for loop across combinations of interest parameters. Set this variable to `false` if [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is not being used. Default is `true`.
- `use_threads`: boolean variable specifying, if `use_distributed` is false, whether to use parallelised for loops across `Threads.nthreads()` threads or a non-parallel for loops to find boundary points from `methods` where boundary points are found independently. Default is `true`.
    - [`Fix1AxisMethod`](@ref) and [`RadialMLEMethod`](@ref) parallelise the finding point pair step and the finding the boundary from point pairs step.
    - [`SimultaneousMethod`](@ref) and [`RadialRandomMethod`](@ref) do not parallelise the finding point pair step but parallelise finding the boundary from point pairs.
    - [`IterativeBoundaryMethod`](@ref) parallelises finding the initial boundary but not the following boundary improvement steps.
    - [`AnalyticalEllipseMethod`](@ref) does not require parallelisation.

!!! note "existing_profiles meanings"
    - :ignore means profiles that already exist will not be recomputed even if they contain fewer `num_points` boundary points. 
    - :merge means profiles that already exist will be merged with profiles from the current algorithm run to reach `num_points`. If the existing profile already has at least `num_points` boundary points then that profile will not be recomputed. Otherwise, the specified method will be run starting from the difference between `num_points` and the number of points in the existing profile. The result of that method run will be merged with the existing profile. Predictions evaluated from the existing profile will be forgotten. To keep these predictions see extended help below.
    - :overwrite means profiles that already exist will be overwritten, regardless of how many points they contain. Predictions evaluated from the existing profile will be forgotten. To keep these predictions see extended help below.

# Details

Using [`LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile`](@ref) this function calls the algorithm/method specified by `method` for each interest parameter combination in `θcombinations` (depending on the setting for `existing_profiles` and `num_points` if these profiles already exist). Nuisance parameters of each point in bivariate interest parameter space are found by maximising the log-likelihood function given by `profile_type`. Updates `model.biv_profiles_df` for each successful profile and saves their results as a [`BivariateConfidenceStruct`](@ref) in `model.biv_profiles_dict`, where the keys for the dictionary is the row number in `model.biv_profiles_df` of the corresponding profile. `model.biv_profiles_df.num_points` is the number of points found on the bivariate boundary (it does not include the number of saved internal points).

# Extended help

## Valid bounds

For methods that use points placed on parameter bounds to bracket for the confidence boundary, the bracketing method utilised via Roots.jl's [`find_zero`](https://juliamath.github.io/Roots.jl/stable/reference/#Roots.find_zero) will be unlikely to converge to the true confidence boundary for a given pair of interest parameters if the bounds on either parameter are +/- Inf or the log-likelihood function evaluates to +/- Inf. Bounds should be set to prevent this from occurring.

## Preventing predictions from being forgotten when merging or overwriting profiles

To prevent predictions from being lost from existing profiles that would be overwritten when calling [`bivariate_confidenceprofiles!`](@ref), existing profiles should be converted into a [`CombinedBivariateMethod`], prior to running new bivariate profiles. To do this use [`combine_bivariate_boundaries!`](@ref) on `model` with keyword argument `not_evaluated_predictions` set to `false`.

## Distributed Computing Implementation

If [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is being used and `use_distributed` is `true`, then the bivariate profiles of distinct interest parameter combinations will be computed in parallel across `Distributed.nworkers()` workers. If `use_distributed` is `false` and `use_threads` is `true` then for methods where finding boundary points is independent they will be computed in parallel across `Threads.nthreads()` threads for each pair of interest parameters.

## Iteration Speed Of the Progress Meter

The time/it value is the time it takes for each new boundary point to be found (for all methods except for [`AnalyticalEllipseMethod`](@ref)). For [`AnalyticalEllipseMethod`](@ref) this is the time it takes to find all points on the boundary of the ellipse of two interest parameters.
"""
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        θcombinations::Vector{Vector{Int}}, 
                                        num_points::Int; 
                                        confidence_level::Float64=0.95, 
                                        dof::Int=2,
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=RadialRandomMethod(5),
                                        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                                        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                                        save_internal_points::Bool=true,
                                        existing_profiles::Symbol=:merge,
                                        find_zero_atol::Real=model.find_zero_atol,
                                        optimizationsettings::Union{OptimizationSettings,Missing}=missing,
                                        show_progress::Bool=model.show_progress,
                                        use_distributed::Bool=true,
                                        use_threads::Bool=true)
                                    
    function argument_handling!()
        existing_profiles ∈ [:ignore, :merge, :overwrite] || throw(ArgumentError("existing_profiles can only take value :ignore, :merge or :overwrite"))

        method isa CombinedBivariateMethod && throw(ArgumentError("CombinedBivariateMethod is not a valid method"))
        model.core isa CoreLikelihoodModel || throw(ArgumentError("model does not contain a log-likelihood function. Add it using add_loglikelihood_function!"))
        find_zero_atol ≥ 0.0 || throw(DomainError("find_zero_atol must be greater than or equal to zero"))
        
        # for each combination, enforce ind1 < ind2 and make sure only unique combinations are run
        sort!.(θcombinations); unique!.(θcombinations)
        sort!(θcombinations); unique!(θcombinations)
        1 ≤ first.(θcombinations)[1] && maximum(last.(θcombinations)) ≤ model.core.num_pars || throw(DomainError("θcombinations can only contain parameter indexes between 1 and the number of model parameters"))
        
        extrema(length.(θcombinations)) == (2,2) || throw(ArgumentError("θcombinations must only contain vectors of length 2"))

        length(θlb_nuisance) == model.core.num_pars || throw(ArgumentError("θlb_nuisance must have the same length as the number of model parameters"))
        length(θub_nuisance) == model.core.num_pars || throw(ArgumentError("θub_nuisance must have the same length as the number of model parameters"))
        all(θlb_nuisance .≤ model.core.θmle) || throw(DomainError("θlb_nuisance must be less than or equal to model.core.θmle"))
        all(θub_nuisance .≥ model.core.θmle) || throw(DomainError("θub_nuisance must be greater than or equal to model.core.θmle"))

        (dof ≥ 2) || throw(DomainError("dof must be greater than or equal to 2. Setting to 2 is recommended"))

        (!use_distributed && use_threads && timeit_debug_enabled()) &&
            throw(ArgumentError("use_threads cannot be true when debug timings from TimerOutputs are enabled and use_distributed is false. Either set use_threads to false or disable debug timings using `LikelihoodBasedProfileWiseAnalysis.TimerOutputs.disable_debug_timings(LikelihoodBasedProfileWiseAnalysis)`"))
        return nothing
    end
    
    argument_handling!()
    optimizationsettings = ismissing(optimizationsettings) ? model.core.optimizationsettings : optimizationsettings
    
    # need at least 3 boundary points for some algorithms to work
    num_points = max(3, num_points)
    
    if profile_type isa AbstractEllipseProfileType
        check_ellipse_approx_exists!(model)
    end

    if method isa AnalyticalEllipseMethod && !(profile_type isa EllipseApproxAnalytical)
        check_ellipse_approx_exists!(model)
        profile_type = EllipseApproxAnalytical()
    end

    boundary_not_ordered = !(method isa AnalyticalEllipseMethod || method isa RadialMLEMethod)

    bivariate_optimiser = get_bivariate_opt_func(profile_type, method)
    consistent = get_consistent_tuple(model, confidence_level, profile_type, dof)
    mle_targetll = get_target_loglikelihood(model, confidence_level, EllipseApproxAnalytical(), dof)

    init_biv_profile_row_exists!(model, θcombinations, dof, profile_type, method)

    θcombinations_to_keep = trues(length(θcombinations))
    θcombinations_to_reuse = falses(length(θcombinations))
    num_to_reuse = 0

    for (i, (ind1, ind2)) in enumerate(θcombinations)
        if model.biv_profile_row_exists[((ind1, ind2), dof, profile_type, method)][confidence_level] != 0
            θcombinations_to_keep[i] = false
            θcombinations_to_reuse[i] = true
            num_to_reuse += 1
        end
    end
    if existing_profiles == :ignore
        θcombinations = θcombinations[θcombinations_to_keep]
        θcombinations_to_merge = θcombinations_to_reuse[θcombinations_to_keep]
        num_to_reuse = 0
    elseif existing_profiles == :merge
        θcombinations_to_merge = θcombinations_to_reuse
    elseif existing_profiles == :overwrite
        θcombinations_to_merge = falses(length(θcombinations))
    end

    len_θcombinations = length(θcombinations)
    len_θcombinations > 0 || return nothing

    num_rows_required = ((len_θcombinations-num_to_reuse) + model.num_biv_profiles) - nrow(model.biv_profiles_df)

    if num_rows_required > 0
        add_biv_profiles_rows!(model, num_rows_required)
    end

    num_new_points = zeros(Int, len_θcombinations) .+ num_points
    if existing_profiles == :merge
        pos_new_points = trues(len_θcombinations)
        for (i, (ind1, ind2)) in enumerate(θcombinations)
            if θcombinations_to_merge[i]
                local row_ind = model.biv_profile_row_exists[((ind1, ind2), dof, profile_type, method)][confidence_level]
                num_new_points[i] = num_new_points[i] - model.biv_profiles_df[row_ind, :num_points]
                pos_new_points[i] = num_new_points[i] > 0
            end
        end
        θcombinations = θcombinations[pos_new_points]
        num_new_points = num_new_points[pos_new_points]
        θcombinations_to_reuse = θcombinations_to_reuse[pos_new_points]
        θcombinations_to_merge = θcombinations_to_merge[pos_new_points]
    end

    len_θcombinations = length(θcombinations)
    len_θcombinations > 0 || return nothing

    tasks_per_profile = get_bivariate_method_tasknumbers.(Ref(method), num_new_points)
    totaltasks = sum(tasks_per_profile)
    # channel_buffer_size = min(ceil(Int, tasks_per_profile * 0.05), 30)
    channel_buffer_size = 2
    channel = RemoteChannel(() -> Channel{Bool}(channel_buffer_size))
    p = Progress(totaltasks; desc="Computing bivariate profiles: ",
                dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    @sync begin
        @async while take!(channel)
            next!(p)
        end

        if use_distributed
            @async begin
                profiles_to_add = @distributed (vcat) for i in 1:len_θcombinations
                    [((θcombinations[i][1], θcombinations[i][2]), 
                        bivariate_confidenceprofile(bivariate_optimiser, model, num_new_points[i],
                                                    confidence_level, consistent, 
                                                    θcombinations[i][1], θcombinations[i][2], dof, profile_type,
                                                    method, θlb_nuisance, θub_nuisance, mle_targetll,
                                                    save_internal_points,
                                                    find_zero_atol,
                                                    optimizationsettings, false,
                                                    channel))]
                end
                put!(channel, false)

                for (i, (inds, boundary_struct)) in enumerate(profiles_to_add)
                    if isnothing(boundary_struct); continue end
                    
                    if θcombinations_to_reuse[i]
                        row_ind = model.biv_profile_row_exists[(inds, dof, profile_type, method)][confidence_level]
                    else
                        model.num_biv_profiles += 1
                        row_ind = model.num_biv_profiles * 1
                        model.biv_profile_row_exists[(inds, dof, profile_type, method)][confidence_level] = row_ind
                    end

                    if θcombinations_to_merge[i]
                        model.biv_profiles_dict[row_ind] = merge(model.biv_profiles_dict[row_ind], boundary_struct)
                    else
                        model.biv_profiles_dict[row_ind] = boundary_struct
                    end

                    set_biv_profiles_row!(model, row_ind, inds, !save_internal_points, true, boundary_not_ordered,
                        confidence_level, dof, profile_type, method, num_points)        
                end
            end
        else
            @async begin
                for i in 1:len_θcombinations
                    inds = (θcombinations[i][1], θcombinations[i][2])
                    boundary_struct = bivariate_confidenceprofile(bivariate_optimiser, model,
                                        num_new_points[i],
                                        confidence_level, consistent,
                                        θcombinations[i][1], θcombinations[i][2], dof, profile_type,
                                        method, θlb_nuisance, θub_nuisance, mle_targetll,
                                        save_internal_points, 
                                        find_zero_atol, 
                                        optimizationsettings, use_threads,
                                        channel)
    
                    if isnothing(boundary_struct); continue end

                    if θcombinations_to_reuse[i]
                        row_ind = model.biv_profile_row_exists[(inds, dof, profile_type, method)][confidence_level]
                    else
                        model.num_biv_profiles += 1
                        row_ind = model.num_biv_profiles * 1
                        model.biv_profile_row_exists[(inds, dof, profile_type, method)][confidence_level] = row_ind
                    end

                    if θcombinations_to_merge[i]
                        model.biv_profiles_dict[row_ind] = merge(model.biv_profiles_dict[row_ind], boundary_struct)
                    else
                        model.biv_profiles_dict[row_ind] = boundary_struct
                    end

                    set_biv_profiles_row!(model, row_ind, inds, !save_internal_points, true, boundary_not_ordered,
                        confidence_level, dof, profile_type, method, num_points)
                    end
                put!(channel, false)
            end
        end
    end

    return nothing
end

"""
    bivariate_confidenceprofiles!(model::LikelihoodModel, 
        θcombinations_symbols::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}}, 
        num_points::Int; 
        <keyword arguments>)

Profiles just the provided `θcombinations_symbols` parameter pairs, provided as either a vector of vectors or a vector of tuples.
"""
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        θcombinations_symbols::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}}, 
                                        num_points::Int;
                                        confidence_level::Float64=0.95, 
                                        dof::Int=2,
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=RadialRandomMethod(5),
                                        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                                        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                                        save_internal_points::Bool=true,
                                        existing_profiles::Symbol=:merge,
                                        find_zero_atol::Real=model.find_zero_atol,
                                        optimizationsettings::Union{OptimizationSettings,Missing}=missing,
                                        show_progress::Bool=model.show_progress,
                                        use_distributed::Bool=true,
                                        use_threads::Bool=true)

    θcombinations = convertθnames_toindices(model, θcombinations_symbols)

    bivariate_confidenceprofiles!(model, θcombinations, num_points, 
            confidence_level=confidence_level, dof=dof, 
            profile_type=profile_type, 
            θlb_nuisance=θlb_nuisance, θub_nuisance=θub_nuisance,
            method=method,
            save_internal_points=save_internal_points,
            existing_profiles=existing_profiles,
            find_zero_atol=find_zero_atol,
            optimizationsettings=optimizationsettings,
            show_progress=show_progress,
            use_distributed=use_distributed,
            use_threads=use_threads)
    return nothing
end

"""
    bivariate_confidenceprofiles!(model::LikelihoodModel, 
        profile_m_random_combinations::Int, 
        num_points::Int; 
        <keyword arguments>)

Profiles m random two-way combinations of model parameters (sampling without replacement), where `0 < m ≤ binomial(model.core.num_pars,2)`.
"""
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        profile_m_random_combinations::Int, 
                                        num_points::Int;
                                        confidence_level::Float64=0.95, 
                                        dof::Int=2, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=RadialRandomMethod(5),
                                        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                                        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                                        save_internal_points::Bool=true,
                                        existing_profiles::Symbol=:merge,
                                        find_zero_atol::Real=model.find_zero_atol,
                                        optimizationsettings::Union{OptimizationSettings,Missing}=missing,
                                        show_progress::Bool=model.show_progress,
                                        use_distributed::Bool=true,
                                        use_threads::Bool=true)

    profile_m_random_combinations = max(0, min(profile_m_random_combinations, binomial(model.core.num_pars, 2)))
    profile_m_random_combinations > 0 || throw(DomainError("profile_m_random_combinations must be a strictly positive integer"))

    θcombinations = sample(collect(combinations(1:model.core.num_pars, 2)),
                            profile_m_random_combinations, replace=false, ordered=true)

    bivariate_confidenceprofiles!(model, θcombinations, num_points, 
            confidence_level=confidence_level, dof=dof,
            profile_type=profile_type, 
            θlb_nuisance=θlb_nuisance, θub_nuisance=θub_nuisance,
            method=method,
            save_internal_points=save_internal_points,
            existing_profiles=existing_profiles,
            find_zero_atol=find_zero_atol,
            optimizationsettings=optimizationsettings,
            show_progress=show_progress,
            use_distributed=use_distributed,
            use_threads=use_threads)
    return nothing
end

"""
    bivariate_confidenceprofiles!(model::LikelihoodModel, 
        num_points::Int; 
        <keyword arguments>)

Profiles all two-way combinations of model parameters.
"""
function bivariate_confidenceprofiles!(model::LikelihoodModel, 
                                        num_points::Int; 
                                        confidence_level::Float64=0.95, 
                                        dof::Int=2, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        method::AbstractBivariateMethod=RadialRandomMethod(5),
                                        θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
                                        θub_nuisance::AbstractVector{<:Real}=model.core.θub,
                                        save_internal_points::Bool=true,
                                        existing_profiles::Symbol=:merge,
                                        find_zero_atol::Real=model.find_zero_atol,
                                        optimizationsettings::Union{OptimizationSettings,Missing}=missing,
                                        show_progress::Bool=model.show_progress,
                                        use_distributed::Bool=true,
                                        use_threads::Bool=true)

    θcombinations = collect(combinations(1:model.core.num_pars, 2))

    bivariate_confidenceprofiles!(model, θcombinations, num_points, 
            confidence_level=confidence_level, dof=dof, 
            profile_type=profile_type, 
            θlb_nuisance=θlb_nuisance, θub_nuisance=θub_nuisance,
            method=method,
            save_internal_points=save_internal_points,
            existing_profiles=existing_profiles,
            find_zero_atol=find_zero_atol,
            optimizationsettings=optimizationsettings,
            show_progress=show_progress,
            use_distributed=use_distributed,
            use_threads=use_threads)
    return nothing
end

