"""
    check_bivariate_boundary_coverage(data_generator::Function,
        generator_args::Union{Tuple,NamedTuple},
        model::LikelihoodModel,
        N::Int,
        num_points::Union{Int, Vector{<:Int}},
        num_points_to_sample::Union{Int, Vector{<:Int}},
        θtrue::AbstractVector{<:Real},
        θcombinations::Union{Vector{Vector{Int}}, Vector{Tuple{Int,Int}}},
        θinitialguess::AbstractVector{<:Real}=θtrue; 
        <keyword arguments>)

Performs a simulation to estimate the coverage of approximate bivariate confidence boundaries with `num_points` constructed using `method` and `hullmethod` for two-way sets of interest parameters in `θcombinations` given a model of the true bivariate confidence boundary by: 

1. Repeatedly drawing new observed data using `data_generator` for fixed true parameter values, θtrue and fitting the model. 
2. `num_points_to_sample` points are then sampled in interest parameter space using `sample_type` and those that are inside the true bivariate confidence boundary are extracted. 
3. Then bivariate confidence boundaries of `num_points` are found using `method` and `hullmethod` is used to construct 2D polygon hulls of the boundary points. 
4. Finally, the percentage of extracted samples that are contained within the 2D polygon hull is extracted. The median and mean percentage (coverage) across all `N` simulations of the true boundary is recorded and returned with a default 95% simulation quantile interval within a DataFrame. The median may be more reliable for use than the mean due to expected coverage approaching 1.0 when the polygon is a very good representation of the boundary. The 95% simulation quantile interval is the 2.5% and 97.5% quantiles of the coverage across the `N simulations`. 

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. Outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`.
- `generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `generator_args`. 
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `N`: a positive number of coverage simulations.
- `num_points`: positive number of points to find on the boundary at the specified confidence level using a single `method`. Or a vector of positive numbers of boundary points to find for each method in `method` (if `method` is a vector of [`AbstractBivariateMethod`](@ref)). Set to at least 3 within the function as some methods need at least three points to work. 
- `num_points_to_sample`: integer number of points to sample (for [`UniformRandomSamples`](@ref) and [`LatinHypercubeSamples`](@ref) sample types) from interest parameter space. For the [`UniformGridSamples`](@ref) sample type, if integer it is the number of points to grid over in each parameter dimension. If it is a vector of integers each index of the vector is the number of points to grid over in the corresponding parameter dimension. For example, [1,2] would mean a single point in dimension 1 and two points in dimension 2. 
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θcombinations`: a vector of pairs of parameters to profile, as a vector of vectors of model parameter indexes.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to evaluate the confidence interval coverage at. Default is `0.95` (95%).
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `method`: a method of type [`AbstractBivariateMethod`](@ref) or a vector of methods of type [`AbstractBivariateMethod`](@ref) (if so `num_points` needs to be a vector of the same length). For a list of available methods use `bivariate_methods()` ([`bivariate_methods`](@ref)). Default is `RadialRandomMethod(3)` ([`RadialRandomMethod`](@ref)).
- `sample_type`: the sampling method used to sample parameter space of type [`AbstractSampleType`]. Default is `LatinHypercubeSamples()` ([`LatinHypercubeSamples`](@ref)).
- `hullmethod`: method of type [`AbstractBivariateHullMethod`](@ref) used to create a 2D polygon hull that approximates the bivariate boundary from a set of boundary points and internal points (method dependent) (or vector of type [`AbstractBivariateHullMethod`](@ref) if comparison between hull methods is1 desired). For available methods see [`bivariate_hull_methods()`](@ref). Default is `MPPHullMethod()` ([`MPPHullMethod`](@ref)).
- `θlb_nuisance`: a vector of lower bounds on nuisance parameters, require `θlb_nuisance .≤ model.core.θmle`. Default is `model.core.θlb`. 
- `θub_nuisance`: a vector of upper bounds on nuisance parameters, require `θub_nuisance .≥ model.core.θmle`. Default is `model.core.θub`.
- `optimizationsettings`: a [`OptimizationSettings`](@ref) containing the optimisation settings used to find optimal values of nuisance parameters for a given interest parameter value. Default is `missing` (will use `default_OptimizationSettings()` (see [`default_OptimizationSettings`](@ref)).
- `coverage_estimate_quantile_level`: a number ∈ (0.0, 1.0) for the level of the quantile interval of the estimated coverage (intervals are formed from simulation quantiles). Default is `0.95` (95%).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations (false) or across the individual bivariate boundary calculations within each iteration (true). Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of the approximations of the true bivariate parameter confidence boundaries. Namely, how well the approximation contains the true boundary. The simulation uses [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) to parallelise the workload.

Tests how well the boundary polygon created by a `method` with a given number of points and turned into a polygon hull using `hullmethod` contains the theoretical boundary by testing how many samples from a [`AbstractSampleType`](@ref) within the true boundary are within the boundary polygon.

If [`MPPHullMethod`](@ref) is the `hullmethod` used, it is expected that the approximation of the true bivariate parameter confidence boundary created by [`bivariate_confidenceprofiles!`](@ref) will be an exact representation, as the number of boundary points approaches infinity. For [`ConcaveHullMethod`](@ref) this is also likely to be the case, but it may fail due to being a heuristic. For [`ConvexHullMethod`](@ref) this will be true if the true boundary is convex. If the true boundary is concave then the approximation that uses [`ConvexHullMethod`](@ref) will fully contain the true boundary, but will also contain parameter space that is not part of the true boundary. 

This check is useful for determining how to most efficiently sample internal points from bivariate confidence boundaries with [`sample_bivariate_internal_points`] as it shows how the interaction between the `method`, `hullmethod` and the number of boundary points impact the coverage of the true boundary. For example, using [`ConvexHullMethod`](@ref) will generally give the highest coverage of the true boundary, but may cause the rejection rate to be higher because it contains a greater area that is not part of the true boundary.

The uncertainty in estimates of the coverage under the simulated model will become more accurate as the number of simulations, `N`, is increased. Simulation quantile intervals for the coverage estimate are provided to quantify this uncertainty. 

!!! note "Simultaneous bivariate profiles"
    Calculating the coverage for approximations of simultaneous bivariate profiles is not currently supported (i.e. for `dof ≠ 2`)

!!! note "Recommended setting for distributed_over_parameters"
    - If the number of processes available to use is significantly greater than the number of model parameters or only a few pairs of model parameters are being checked for coverage, `false` is recommended.   
    - If system memory or model size in system memory is a concern, or the number of processes available is similar or less than the number of pairs of model parameters being checked, `true` will likely be more appropriate. 
    - When set to `false`, a separate [`LikelihoodModel`](@ref) struct will be used by each process, as opposed to only one when set to `true`, which could cause a memory issue for larger models. 
"""
function check_bivariate_boundary_coverage(data_generator::Function,
    generator_args::Union{Tuple,NamedTuple},
    model::LikelihoodModel,
    N::Int,
    num_points::Union{Int, Vector{<:Int}},
    num_points_to_sample::Union{Int, Vector{<:Int}},
    θtrue::AbstractVector{<:Real},
    θcombinations::Union{Vector{Vector{Int}}, Vector{Tuple{Int,Int}}},
    θinitialguess::AbstractVector{<:Real}=θtrue;
    confidence_level::Float64=0.95,
    profile_type::AbstractProfileType=LogLikelihood(),
    method::Union{AbstractBivariateMethod,Vector{<:AbstractBivariateMethod}}=RadialRandomMethod(3),
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    hullmethod::Union{AbstractBivariateHullMethod, Vector{<:AbstractBivariateHullMethod}}=MPPHullMethod(),
    θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
    θub_nuisance::AbstractVector{<:Real}=model.core.θub,
    coverage_estimate_quantile_level::Float64=0.95,
    optimizationsettings::Union{OptimizationSettings,Missing}=missing,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false)

    function argument_handling!()
        length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))
        length(θinitialguess) == model.core.num_pars || throw(ArgumentError("θinitialguess must have the same length as the number of model parameters"))

        (0.0 < coverage_estimate_quantile_level && coverage_estimate_quantile_level < 1.0) || throw(DomainError("coverage_estimate_quantile_level must be in the open interval (0,1)"))
        get_target_loglikelihood(model, confidence_level, profile_type, 2)

        if num_points_to_sample isa Int
            num_points_to_sample > 0 || throw(DomainError("num_points_to_sample must be a strictly positive integer"))
        else
            minimum(num_points_to_sample) > 0 || throw(DomainError("num_points_to_sample must contain strictly positive integers"))

            sample_type isa UniformGridSamples || throw(ArgumentError(string("num_points_to_sample must be an integer for ", sample_type, " sample_type")))

            (length(num_points_to_sample) == length(θcombinations[1]) &&
            diff([extrema(length.(θcombinations))...])[1] == 0) ||
                throw(ArgumentError("num_points_to_sample must have the same length as each vector of interest parameters in num_points_to_sample"))
        end

        !xor(num_points isa Vector, method isa Vector) || throw(ArgumentError("num_points and method must both be a Vector, or both be a Int and AbstractBivariateMethod, respectively, at the same time (xnor gate)"))
        combine_methods = num_points isa Vector
        if combine_methods
            (length(num_points) == length(method)) || throw(ArgumentError("num_points must have the same length as method, each index in num_points corresponds to the number of boundary points for the corresponding index in method"))
        end

        N > 0 || throw(DomainError("N must be greater than 0"))

        length(θlb_nuisance) == model.core.num_pars || throw(ArgumentError("θlb_nuisance must have the same length as the number of model parameters"))
        length(θub_nuisance) == model.core.num_pars || throw(ArgumentError("θub_nuisance must have the same length as the number of model parameters"))
        all(θlb_nuisance .≤ model.core.θmle) || throw(DomainError("θlb_nuisance must be less than or equal to model.core.θmle"))
        all(θub_nuisance .≥ model.core.θmle) || throw(DomainError("θub_nuisance must be greater than or equal to model.core.θmle"))

        if hullmethod isa Vector
            unique!(hullmethod)
        else
            hullmethod = [hullmethod]
        end

        if θcombinations isa Vector{Tuple{Int, Int}}
            θcombinations = [[combo...] for combo in θcombinations]
        end

        # for each combination, enforce ind1 < ind2 and make sure only unique combinations are run
        sort!.(θcombinations)
        unique!.(θcombinations)
        sort!(θcombinations)
        unique!(θcombinations)

        1 ≤ first.(θcombinations)[1] && maximum(last.(θcombinations)) ≤ model.core.num_pars || throw(DomainError("θcombinations can only contain parameter indexes between 1 and the number of model parameters"))
        extrema(length.(θcombinations)) == (2, 2) || throw(ArgumentError("θcombinations must only contain vectors of length 2"))
        return nothing
    end

    local combine_methods::Bool
    argument_handling!()

    len_θs = length(θcombinations)
    combo_to_index = Dict{Tuple{Int,Int},Int}(Tuple(combo) => index for (index, combo) in enumerate(θcombinations))

    full_len = len_θs * length(hullmethod)
    coverage = zeros(full_len, N)

    bivariate_optimiser = get_bivariate_opt_func(profile_type, RadialMLEMethod())
    biv_opt_is_ellipse_analytical = bivariate_optimiser == bivariateψ_ellipse_analytical_vectorsearch
    min_num_dim_points = 20

    data = [data_generator(θtrue, generator_args) for _ in 1:N]

    channel = RemoteChannel(() -> Channel{Bool}(3))
    progress = Progress(N; desc="Computing bivariate boundary coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for i in 1:N
            new_data = data[i]

            m_new = initialise_LikelihoodModel(model.core.loglikefunction, new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, model.core.θmagnitudes; biv_row_preallocation_size=len_θs, show_progress=false, optimizationsettings=model.core.optimizationsettings)

            lb, ub = correct_θbounds_nuisance(m_new, θlb_nuisance, θub_nuisance)
            dimensional_likelihood_samples!(m_new, θcombinations, num_points_to_sample;
                confidence_level=confidence_level, sample_type=sample_type, 
                θlb_nuisance=lb, θub_nuisance=ub,
                show_progress=false, optimizationsettings=optimizationsettings)

            if combine_methods
                for (j, methodj) in enumerate(method)
                    bivariate_confidenceprofiles!(m_new, θcombinations, num_points[j];
                        confidence_level=confidence_level, profile_type=profile_type, method=methodj,
                        θlb_nuisance=lb, θub_nuisance=ub,
                        show_progress=false, optimizationsettings=optimizationsettings, use_threads=false)
                end
                combine_bivariate_boundaries!(m_new, confidence_level=confidence_level,
                    not_evaluated_predictions=true)
            else
                bivariate_confidenceprofiles!(m_new, θcombinations, num_points;
                    confidence_level=confidence_level, profile_type=profile_type, method=method,
                    θlb_nuisance=lb, θub_nuisance=ub,
                    show_progress=false, optimizationsettings=optimizationsettings, use_threads=false)
            end

            for (h, hmethod) in enumerate(hullmethod)
                for row_ind in 1:m_new.num_biv_profiles
                    θindices_tuple = m_new.biv_profiles_df[row_ind, :θindices]
                    θindices = [θindices_tuple...]

                    conf_struct = m_new.biv_profiles_dict[row_ind]
                    polygonhull = construct_polygon_hull(m_new, θindices, conf_struct, confidence_level, 2,
                        m_new.biv_profiles_df[row_ind, :boundary_not_ordered], hmethod, true)

                    polygonhull_num_points = size(polygonhull, 2)
                    nodes = permutedims(polygonhull)
                    edges = zeros(Int, polygonhull_num_points, 2)
                    for j in 1:polygonhull_num_points-1; edges[j,:] .= j, j+1 end
                    edges[end,:] .= polygonhull_num_points, 1

                    dim_samples_row = m_new.dim_samples_row_exists[(θindices, 2, sample_type)][confidence_level]
                    if dim_samples_row == 0; continue end

                    num_dim_points = m_new.dim_samples_df[dim_samples_row, :num_points]
                    if num_dim_points < min_num_dim_points; continue end

                    dimensional_samples = m_new.dim_samples_dict[dim_samples_row].points[θindices, :]

                    count=0
                    for j in axes(dimensional_samples, 2)
                        is_inside, is_boundary = inpoly2(dimensional_samples[:,j], nodes, edges)

                        if is_inside || is_boundary
                            count += 1
                        end
                    end
                    coverage[combo_to_index[θindices_tuple] + ((h-1)*len_θs), i] = count / num_dim_points
                end
            end
            next!(progress)
        end

    else
        coverage_shared = SharedArray(zeros(full_len, N))
        @sync begin
            @async while take!(channel)
                next!(progress)
            end

            @async begin
                @distributed (+) for i in 1:N
                    new_data = data[i]

                    m_new = initialise_LikelihoodModel(model.core.loglikefunction, new_data,
                        model.core.θnames, θinitialguess, model.core.θlb, model.core.θub,
                        model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false, 
                        optimizationsettings=model.core.optimizationsettings)

                    lb, ub = correct_θbounds_nuisance(m_new, θlb_nuisance, θub_nuisance)                    

                    dimensional_likelihood_samples!(m_new, θcombinations, num_points_to_sample;
                        confidence_level=confidence_level, sample_type=sample_type,
                        θlb_nuisance=lb, θub_nuisance=ub,
                        show_progress=false, optimizationsettings=optimizationsettings)
                    
                    if combine_methods
                        for (j, methodj) in enumerate(method)
                            bivariate_confidenceprofiles!(m_new, θcombinations, num_points[j];
                                confidence_level=confidence_level, profile_type=profile_type, method=methodj,
                                θlb_nuisance=lb, θub_nuisance=ub,
                                show_progress=false, optimizationsettings=optimizationsettings, 
                                use_distributed=false, use_threads=false)
                        end
                        combine_bivariate_boundaries!(m_new, confidence_level=confidence_level,
                            not_evaluated_predictions=true)
                    else
                        bivariate_confidenceprofiles!(m_new, θcombinations, num_points;
                            confidence_level=confidence_level, profile_type=profile_type, method=method,
                            θlb_nuisance=lb, θub_nuisance=ub,
                            show_progress=false, optimizationsettings=optimizationsettings, 
                            use_distributed=false, use_threads=false)
                    end

                    for (h, hmethod) in enumerate(hullmethod)
                        for row_ind in 1:m_new.num_biv_profiles
                            θindices_tuple = m_new.biv_profiles_df[row_ind, :θindices]
                            θindices = [θindices_tuple...]

                            conf_struct = m_new.biv_profiles_dict[row_ind]
                            polygonhull = construct_polygon_hull(m_new, θindices, conf_struct, confidence_level, 2,
                                m_new.biv_profiles_df[row_ind, :boundary_not_ordered], hmethod, true)

                            polygonhull_num_points = size(polygonhull, 2)
                            nodes = permutedims(polygonhull)
                            edges = zeros(Int, polygonhull_num_points, 2)
                            for j in 1:polygonhull_num_points-1; edges[j,:] .= j, j+1 end
                            edges[end,:] .= polygonhull_num_points, 1

                            dim_samples_row = m_new.dim_samples_row_exists[(θindices, 2, sample_type)][confidence_level]
                            if dim_samples_row == 0; continue end

                            num_dim_points = m_new.dim_samples_df[dim_samples_row, :num_points]
                            if num_dim_points < min_num_dim_points; continue end
                            
                            dimensional_samples = m_new.dim_samples_dict[dim_samples_row].points[θindices, :]

                            count = 0
                            for j in axes(dimensional_samples, 2)
                                is_inside, is_boundary = inpoly2(dimensional_samples[:,j], nodes, edges)

                                if is_inside || is_boundary
                                    count += 1
                                end
                            end
                            coverage_shared[combo_to_index[θindices_tuple]+((h-1)*len_θs), i] = count / size(dimensional_samples, 2)
                        end
                    end
                    put!(channel, true); i^2
                end
                put!(channel, false)
            end
        end
        coverage .= coverage_shared
    end

    quantile_intervals = zeros(full_len, 2)
    lower_quantile = (1.0-coverage_estimate_quantile_level) / 2.0 
    for i in 1:full_len
        quantile_intervals[i, :] .= quantile(coverage[i, coverage[i,:] .!= 0.0], [lower_quantile, 1 - lower_quantile])
    end

    coverage_mean = [mean(coverage[i, coverage[i, :].!=0.0]) for i in 1:full_len]
    coverage_median = [median(coverage[i, coverage[i, :].!=0.0]) for i in 1:full_len]

    return DataFrame(θnames=[model.core.θnames[[combo...]] for _ in hullmethod for combo in θcombinations],
                        θindices=[combo for _ in hullmethod for combo in θcombinations], 
                        hullmethod=[string(hmethod) for hmethod in hullmethod for _ in θcombinations], 
                        coverage_median=coverage_median, 
                        coverage_mean=coverage_mean, 
                        coverage_lb=quantile_intervals[:, 1], coverage_ub=quantile_intervals[:, 2],
                        num_boundary_points=fill(sum(num_points), full_len))
end