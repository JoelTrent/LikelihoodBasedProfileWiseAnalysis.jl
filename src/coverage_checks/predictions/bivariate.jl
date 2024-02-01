"""
    check_bivariate_prediction_coverage(data_generator::Function, 
        generator_args::Union{Tuple, NamedTuple},
        t::AbstractVector,
        model::LikelihoodModel, 
        N::Int, 
        num_points::Union{Int, Vector{<:Int}},
        θtrue::AbstractVector{<:Real}, 
        θcombinations::Union{Vector{Vector{Int}}, Vector{Tuple{Int,Int}}},
        θinitialguess::AbstractVector{<:Real}=θtrue; 
        <keyword arguments>)

Performs a simulation to estimate the prediction coverage of bivariate confidence profiles for parameters in `θcombinations` given a model by: 
    
1. Repeatedly drawing new observed data using `data_generator` for fixed true parameter values, `θtrue`, and fixed true prediction value. 
2. Fitting the model and bivariate confidence boundaries. 
3. Sampling points within the polygon hull of the confidence boundaries.
4. Evaluating predictions from the points in the profile and finding the prediction extrema.
5. Checking whether the prediction extrema contain the true prediction value(s), in a pointwise and simultaneous fashion. The estimated simultaneous coverage is returned with a default 95% confidence interval within a DataFrame. 

The prediction coverage from combining the prediction sets of multiple confidence profiles, choosing 1 to `length(θcombinations)` random combinations of `θcombinations`, is also evaluated (i.e. the final result is the union over all profiles in `θcombinations`). 

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. Outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`.
- `generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `generator_args`. 
- `t`: a vector of time points to compute predictions and evaluate coverage at.
- `model`: a [`LikelihoodModel`](@ref) containing model information.
- `N`: a positive number of coverage simulations.
- `num_points`: positive number of points to find on the boundary at the specified confidence level using a single `method`. Or a vector of positive numbers of boundary points to find for each method in `method` (if `method` is a vector of [`AbstractBivariateMethod`](@ref)). Set to at least 3 within the function as some methods need at least three points to work. 
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θcombinations`: a vector of pairs of parameters to profile, as a vector of vectors of model parameter indexes.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `num_internal_points`: an integer number of points to optionally evaluate within the a polygon hull approximation of a bivariate boundary for each interest parameter pair using [`sample_bivariate_internal_points`](@ref). Default is `0`. 
- `hullmethod`: method of type [`AbstractBivariateHullMethod`](@ref) used to create a 2D polygon hull that approximates the bivariate boundary from a set of boundary points and internal points (method dependent). For available methods see [`bivariate_hull_methods()`](@ref). Default is `MPPHullMethod()` ([`MPPHullMethod`](@ref)).
- `sample_type`: either a [`UniformRandomSamples`](@ref) or [`LatinHypercubeSamples`](@ref) struct for how to sample internal points from the polygon hull. [`UniformRandomSamples`](@ref) are homogeneously sampled from the polygon and [`LatinHypercubeSamples`](@ref) use the intersection of a heuristically optimised Latin Hypercube sampling plan with the polygon. Default is `LatinHypercubeSamples()` ([`LatinHypercubeSamples`](@ref)).
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level on which to find the `profile_type` boundary. Default is `0.95` (95%).
- `dof`: an integer ∈ [2, `model.core.num_pars`] for the degrees of freedom used to define the asymptotic threshold ([`LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood`](@ref)) which defines the boundary of the bivariate profile. For bivariate profiles that are considered individually, it should be set to `2`. For profiles that are considered simultaneously, it should be set to `model.core.num_pars`. Default is `2`. Setting it to `model.core.num_pars` should be reasonable when making predictions for well-identified models with `<10` parameters. Note: values other than `2` and `model.core.num_pars` may not have a clear statistical interpretation.
- `θlb_nuisance`: a vector of lower bounds on nuisance parameters, require `θlb_nuisance .≤ model.core.θmle`. Default is `model.core.θlb`. 
- `θub_nuisance`: a vector of upper bounds on nuisance parameters, require `θub_nuisance .≥ model.core.θmle`. Default is `model.core.θub`.
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is `0.95` (95%).
- `optimizationsettings`: a [`OptimizationSettings`](@ref) containing the optimisation settings used to find optimal values of nuisance parameters for a given interest parameter value. Default is `missing` (will use `default_OptimizationSettings()` (see [`default_OptimizationSettings`](@ref)).
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is `0.95` (95%).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations (false) or across the individual confidence interval calculations within each iteration (true). Default is `false`.
- `manual_GC_calls`: boolean variable specifying whether to manually call garbage collection, `GC.gc()`, after every 10 iterations (`distributed_over_parameters=true`) or after every iteration on that worker (`distributed_over_parameters=false`). May be important to correctly free up memory for coverage simulations that use distributed or threaded workloads for Julia versions prior to v1.10.0.  Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of propagating bivariate parameter confidence intervals into prediction space. The simulation uses [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) to parallelise the workload.

The uncertainty in estimates of the prediction coverage under the simulated model will decrease as the number of simulations, `N`, is increased. Confidence intervals for the coverage estimate are provided to quantify this uncertainty. The confidence interval for the estimated coverage is a Clopper-Pearson interval on a binomial test generated using [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/).

!!! note "Recommended setting for distributed_over_parameters"
    - If the number of processes available to use is significantly greater than the number of model parameters or only a few pairs of model parameters are being checked for coverage, `false` is recommended.   
    - If system memory or model size in system memory is a concern, or the number of processes available is similar or less than the number of pairs of model parameters being checked, `true` will likely be more appropriate. 
    - When set to `false`, a separate [`LikelihoodModel`](@ref) struct will be used by each process, as opposed to only one when set to `true`, which could cause a memory issue for larger models. 

!!! danger "May not work correctly on bimodal confidence boundaries"
    The current implementation constructs a single polygon with minimum polygon perimeter from the set of boundary points as the confidence boundary. If there are multiple distinct boundaries represented, then there will be edges connecting the distinct boundaries which the true parameter might be inside (but not inside either of the distinct boundaries). 
"""
function check_bivariate_prediction_coverage(data_generator::Function, 
    generator_args::Union{Tuple, NamedTuple},
    t::AbstractVector,
    model::LikelihoodModel, 
    N::Int, 
    num_points::Union{Int, Vector{<:Int}},
    θtrue::AbstractVector{<:Real}, 
    θcombinations::Union{Vector{Vector{Int}}, Vector{Tuple{Int,Int}}},
    θinitialguess::AbstractVector{<:Real}=θtrue;
    num_internal_points::Int=0,
    hullmethod::AbstractBivariateHullMethod=MPPHullMethod(),
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    confidence_level::Float64=0.95,
    dof::Int=2,
    profile_type::AbstractProfileType=LogLikelihood(), 
    method::Union{AbstractBivariateMethod,Vector{<:AbstractBivariateMethod}}=RadialRandomMethod(3),
    θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
    θub_nuisance::AbstractVector{<:Real}=model.core.θub,
    coverage_estimate_confidence_level::Float64=0.95,
    optimizationsettings::Union{OptimizationSettings,Missing}=missing,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false,
    manual_GC_calls::Bool=false)

    function argument_handling!()
        length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))
        length(θinitialguess) == model.core.num_pars || throw(ArgumentError("θinitialguess must have the same length as the number of model parameters"))
        check_prediction_function_exists(model::LikelihoodModel) || throw(ArgumentError("see warning message"))
        
        num_internal_points >= 0 || throw(DomainError("num_internal_points must be a strictly positive integer"))
        
        (0.0 < coverage_estimate_confidence_level && coverage_estimate_confidence_level < 1.0) || throw(DomainError("coverage_estimate_confidence_level must be in the open interval (0,1)"))
        get_target_loglikelihood(model, confidence_level, profile_type, 1)
        
        !xor(num_points isa Vector, method isa Vector) || throw(ArgumentError("num_points and method must both be a Vector, or both be a Int and AbstractBivariateMethod, respectively, at the same time (xnor gate)"))
        combine_methods = num_points isa Vector 
        if combine_methods
            (length(num_points) == length(method)) || throw(ArgumentError("num_points must have the same length as method, each index in num_points corresponds to the number of boundary points for the corresponding index in method"))
        end

        N > 0 || throw(DomainError("N must be greater than 0"))
        (dof ≥ 2) || throw(DomainError("dof must be greater than or equal to 2. Setting to 2 is generally recommended"))

        length(θlb_nuisance) == model.core.num_pars || throw(ArgumentError("θlb_nuisance must have the same length as the number of model parameters"))
        length(θub_nuisance) == model.core.num_pars || throw(ArgumentError("θub_nuisance must have the same length as the number of model parameters"))
        all(θlb_nuisance .≤ model.core.θmle) || throw(DomainError("θlb_nuisance must be less than or equal to model.core.θmle"))
        all(θub_nuisance .≥ model.core.θmle) || throw(DomainError("θub_nuisance must be greater than or equal to model.core.θmle"))

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

    y_true = model.core.predictfunction(θtrue, model.core.data, t)
    multiple_outputs = ndims(y_true) == 2

    len_θs = length(θcombinations)

    successes = zeros(Int, len_θs*2)
    successes_pointwise = [zeros(size(y_true)) for _ in 1:(len_θs*2)]
    iteration_is_included = falses(N)

    data = [data_generator(θtrue, generator_args) for _ in 1:N]

    channel = RemoteChannel(() -> Channel{Bool}(1))
    p = Progress(N; desc="Computing bivariate prediction coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for i in 1:N
            new_data = data[i]

            m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction,
                new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, 
                model.core.θmagnitudes; biv_row_preallocation_size=len_θs, show_progress=false,
                optimizationsettings=model.core.optimizationsettings)

            lb, ub = correct_θbounds_nuisance(m_new, θlb_nuisance, θub_nuisance)

            if combine_methods
                for (j, methodj) in enumerate(method) 
                    bivariate_confidenceprofiles!(m_new, deepcopy(θcombinations), num_points[j];
                        confidence_level=confidence_level, dof=dof, 
                        profile_type=profile_type, method=methodj, θlb_nuisance=lb, θub_nuisance=ub,
                        use_threads=false,
                        optimizationsettings=optimizationsettings)
                end
                combine_bivariate_boundaries!(m_new, confidence_level=confidence_level, dof=dof,
                    not_evaluated_predictions=true)
            else
                bivariate_confidenceprofiles!(m_new, deepcopy(θcombinations), num_points;
                    confidence_level=confidence_level, dof=dof,
                    profile_type=profile_type, method=method, θlb_nuisance=lb, θub_nuisance=ub,
                    use_threads=false,
                    optimizationsettings=optimizationsettings)
            end
            
            if num_internal_points > 0 
                sample_bivariate_internal_points!(m_new, num_internal_points,
                    sample_type=sample_type, hullmethod=hullmethod, 
                    θlb_nuisance=lb, θub_nuisance=ub, use_threads=false,
                    optimizationsettings=optimizationsettings)
            end

            generate_predictions_bivariate!(m_new, t, 0.0)

            indiv_cov, union_cov, iteration_is_included[i] = evaluate_coverage(m_new, y_true, :bivariate, multiple_outputs, len_θs)            
            successes[1:len_θs] .+= first.(indiv_cov)
            successes[len_θs+1:end] .+= first.(union_cov)

            successes_pointwise[1:len_θs] .+= last.(indiv_cov)
            successes_pointwise[len_θs+1:end] .+= last.(union_cov)

            next!(p)
            if manual_GC_calls && rem(i, 5) == 0
                @everywhere GC.gc()
            end
        end
    else
        successes_bool = SharedArray{Bool}(len_θs*2, N)
        successes_bool .= false
        iteration_is_included_shared = SharedArray{Bool}(N)
        iteration_is_included_shared .= false
        @sync begin
            @async while take!(channel)
                next!(p)
            end

            @async begin
                successes_pointwise_bool = @distributed (vcat) for i in 1:N
                    new_data = data[i]

                    m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction, 
                        new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, 
                        model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false,
                        optimizationsettings=model.core.optimizationsettings)

                    lb, ub = correct_θbounds_nuisance(m_new, θlb_nuisance, θub_nuisance)

                    if combine_methods
                        for (j, methodj) in enumerate(method) 
                            bivariate_confidenceprofiles!(m_new, deepcopy(θcombinations), num_points[j];
                                confidence_level=confidence_level, dof=dof, 
                                profile_type=profile_type, method=methodj,
                                θlb_nuisance=lb, θub_nuisance=ub,
                                use_distributed=false, use_threads=false,
                                optimizationsettings=optimizationsettings)
                        end
                        combine_bivariate_boundaries!(m_new, confidence_level=confidence_level, dof=dof,
                            not_evaluated_predictions=true)
                    else
                        bivariate_confidenceprofiles!(m_new, deepcopy(θcombinations), num_points;
                            confidence_level=confidence_level, dof=dof,
                            profile_type=profile_type, method=method, θlb_nuisance=lb, θub_nuisance=ub,
                            use_distributed=false, use_threads=false,
                            optimizationsettings=optimizationsettings)
                    end
                    
                    if num_internal_points > 0 
                        sample_bivariate_internal_points!(m_new, num_internal_points,
                            sample_type=sample_type, hullmethod=hullmethod,
                            θlb_nuisance=lb, θub_nuisance=ub, 
                            use_distributed=false, use_threads=false,
                            optimizationsettings=optimizationsettings)
                    end
        
                    generate_predictions_bivariate!(m_new, t, 0.0, use_distributed=false)

                    indiv_cov, union_cov, iteration_is_included_shared[i] = evaluate_coverage(m_new, y_true, :bivariate, multiple_outputs, len_θs)
                    successes_bool[1:len_θs, i] .= first.(indiv_cov)
                    successes_bool[len_θs+1:end, i] .= first.(union_cov)

                    if manual_GC_calls
                        GC.gc()
                    end
                    put!(channel, true)
                    (vcat(last.(indiv_cov), last.(union_cov)),)
                end
                put!(channel, false)
                iteration_is_included .= iteration_is_included_shared
                for pointwise_bool in successes_pointwise_bool
                    successes_pointwise .+= first(pointwise_bool)
                end
            end
        end
        successes .= sum(successes_bool, dims=2)
    end

    N_counted = sum(iteration_is_included)
    coverage = successes ./ N_counted
    conf_ints = zeros(len_θs*2, 2)
    for i in 1:(len_θs*2)
        conf_ints[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N_counted), 
            level=coverage_estimate_confidence_level) 
        successes_pointwise[i] = successes_pointwise[i] ./ N_counted
    end

    num_boundary_points = zeros(Int, len_θs*2)
    num_boundary_points[1:len_θs] .= sum(num_points)
    num_boundary_points[len_θs+1:end] .= sum(num_points) .* collect(1:len_θs)
    num_internal_points_all = zeros(Int, len_θs*2)
    num_internal_points_all[1:len_θs] .= num_internal_points
    num_internal_points_all[len_θs+1:end] .= num_internal_points .* collect(1:len_θs)

    if manual_GC_calls
        GC.gc()
    end

    return DataFrame(θname=[[model.core.θnames[[combo...]] for combo in θcombinations]..., fill("", len_θs)...], θindices=[θcombinations..., fill([0], len_θs)...],
        n_random_combinations=[fill(0, len_θs)..., collect(1:len_θs)...],
        simultaneous_coverage=coverage, coverage_lb=conf_ints[:,1], coverage_ub=conf_ints[:,2],
        pointwise_coverage=successes_pointwise,
        num_boundary_points=num_boundary_points,
        num_internal_points=num_internal_points_all)
end

"""
    check_bivariate_prediction_realisations_coverage(data_generator::Function, 
        reference_set_generator::Function,
        training_generator_args::Union{Tuple, NamedTuple},
        testing_generator_args::Union{Tuple, NamedTuple},
        t::AbstractVector,
        model::LikelihoodModel, 
        N::Int, 
        num_points::Union{Int, Vector{<:Int}},
        θtrue::AbstractVector{<:Real}, 
        θcombinations::Union{Vector{Vector{Int}}, Vector{Tuple{Int,Int}}},
        θinitialguess::AbstractVector{<:Real}=θtrue; 
        <keyword arguments>)

Performs a simulation to estimate the prediction reference set and realisation coverage of bivariate confidence profiles for parameters in `θcombinations` given a model by:

1. Constructing the `confidence_level` reference set for predictions from the fixed true parameter values, `θ_true`.
2. Repeatedly drawing new observed training data using `data_generator` and `training_generator_args` for fixed true parameter values, `θtrue`, and fixed true prediction value. 
3. Fitting the model using training data.
4. Fitting the model and bivariate confidence boundaries using training data. 
5. Sampling points within the polygon hull of the confidence boundaries.
6. Evaluating predictions from the points in the profile and finding the prediction extrema (reference tolerance sets).
7. Drawing new observed testing data using `data_generator` and `training_generator_args` for fixed true parameter values, `θtrue`, and fixed true prediction value. 
8. Checking whether the prediction extrema (reference tolerance set) contains the prediction reference set from Step 1, in a pointwise and simultaneous fashion. 
9. Checking whether the prediction extrema contain the observed testing data, in a pointwise and simultaneous fashion. 
10. The estimated simultaneous coverage of the reference set and the prediction realisations (observed testing data) is returned with a default 95% confidence interval, alongside pointwise coverage, within a DataFrame. We also provided an alternate 'simultaneous' statistic for prediction realisation coverage; rather than testing whether 100% of prediction realisations are covered we test whether `simultaneous_alternate_proportion` proportion of prediction realisations are covered. 

The prediction coverage from combining the prediction reference sets of multiple confidence profiles, choosing 1 to `length(θcombinations)` random combinations of `θcombinations`, is also evaluated (i.e. the final result is the union over all profiles in `θcombinations`). 

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. When used with `training_generator_args`, it outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`. When used with `testing_generator_args`, it outputs an array containing the observed data to use as the test data set.
- `reference_set_generator`: a function with three arguments which generates the `confidence_level` data reference set for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The three arguments must be the vector of true model parameters, `θtrue`, a Tuple or NamedTuple, `generator_args`, and a number (0.0, 1.0) for the confidence level at which to evaluate the reference set. When used with `testing_generator_args` it outputs a tuple of two arrays, `(lq, uq)`, which contain the lower and upper quantiles of the reference set. 
- `training_generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at, used to create the training set of data. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `training_generator_args`. 
- `testing_generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at, used to create the test data set. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `testing_generator_args`. 
- `t`: a vector of time points to compute predictions and evaluate coverage at, which are the same as the time points used to create the test data set.
- `model`: a [`LikelihoodModel`](@ref) containing model information.
- `N`: a positive number of coverage simulations.
- `num_points`: positive number of points to find on the boundary at the specified confidence level using a single `method`. Or a vector of positive numbers of boundary points to find for each method in `method` (if `method` is a vector of [`AbstractBivariateMethod`](@ref)). Set to at least 3 within the function as some methods need at least three points to work. 
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θcombinations`: a vector of pairs of parameters to profile, as a vector of vectors of model parameter indexes.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `num_internal_points`: an integer number of points to optionally evaluate within the a polygon hull approximation of a bivariate boundary for each interest parameter pair using [`sample_bivariate_internal_points`](@ref). Default is `0`. 
- `hullmethod`: method of type [`AbstractBivariateHullMethod`](@ref) used to create a 2D polygon hull that approximates the bivariate boundary from a set of boundary points and internal points (method dependent). For available methods see [`bivariate_hull_methods()`](@ref). Default is `MPPHullMethod()` ([`MPPHullMethod`](@ref)).
- `sample_type`: either a [`UniformRandomSamples`](@ref) or [`LatinHypercubeSamples`](@ref) struct for how to sample internal points from the polygon hull. [`UniformRandomSamples`](@ref) are homogeneously sampled from the polygon and [`LatinHypercubeSamples`](@ref) use the intersection of a heuristically optimised Latin Hypercube sampling plan with the polygon. Default is `LatinHypercubeSamples()` ([`LatinHypercubeSamples`](@ref)).
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level on which to find the `profile_type` boundary. Default is `0.95` (95%).
- `dof`: an integer ∈ [2, `model.core.num_pars`] for the degrees of freedom used to define the asymptotic threshold ([`LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood`](@ref)) which defines the boundary of the bivariate profile. For bivariate profiles that are considered individually, it should be set to `2`. For profiles that are considered simultaneously, it should be set to `model.core.num_pars`. Default is `2`. Setting it to `model.core.num_pars` should be reasonable when making predictions for well-identified models with `<10` parameters. Note: values other than `2` and `model.core.num_pars` may not have a clear statistical interpretation.
- `region`: a `Real` number ∈ [0, 1] specifying the proportion of the density of the error model from which to evaluate the highest density region. Default is `0.95`.
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `θlb_nuisance`: a vector of lower bounds on nuisance parameters, require `θlb_nuisance .≤ model.core.θmle`. Default is `model.core.θlb`. 
- `θub_nuisance`: a vector of upper bounds on nuisance parameters, require `θub_nuisance .≥ model.core.θmle`. Default is `model.core.θub`.
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is `0.95` (95%).
- `simultaneous_alternate_proportion`: a number ∈ (0.0, 1.0) for the alternate 'simultaneous' coverage statistic, testing whether at least this proportion of prediction realisations are covered. Recommended to be equal to `region`. Default is `0.95` (95%).
- `optimizationsettings`: a [`OptimizationSettings`](@ref) containing the optimisation settings used to find optimal values of nuisance parameters for a given interest parameter value. Default is `missing` (will use `default_OptimizationSettings()` (see [`default_OptimizationSettings`](@ref)).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations (false) or across the individual confidence interval calculations within each iteration (true). Default is `false`.
- `manual_GC_calls`: boolean variable specifying whether to manually call garbage collection, `GC.gc()`, after every 10 iterations (`distributed_over_parameters=true`) or after every iteration on that worker (`distributed_over_parameters=false`). May be important to correctly free up memory for coverage simulations that use distributed or threaded workloads for Julia versions prior to v1.10.0.  Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of propagating bivariate parameter confidence intervals into prediction realisation space. The simulation uses [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) to parallelise the workload.

The uncertainty in estimates of the prediction realisation coverage under the simulated model will decrease as the number of simulations, `N`, is increased. Confidence intervals for the coverage estimate are provided to quantify this uncertainty. The confidence interval for the estimated coverage is a Clopper-Pearson interval on a binomial test generated using [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/).

!!! note "Recommended setting for distributed_over_parameters"
    - If the number of processes available to use is significantly greater than the number of model parameters or only a few pairs of model parameters are being checked for coverage, `false` is recommended.   
    - If system memory or model size in system memory is a concern, or the number of processes available is similar or less than the number of pairs of model parameters being checked, `true` will likely be more appropriate. 
    - When set to `false`, a separate [`LikelihoodModel`](@ref) struct will be used by each process, as opposed to only one when set to `true`, which could cause a memory issue for larger models. 

!!! danger "May not work correctly on bimodal confidence boundaries"
    The current implementation constructs a single polygon with minimum polygon perimeter from the set of boundary points as the confidence boundary. If there are multiple distinct boundaries represented, then there will be edges connecting the distinct boundaries which the true parameter might be inside (but not inside either of the distinct boundaries). 
"""
function check_bivariate_prediction_realisations_coverage(data_generator::Function,
    reference_set_generator::Function,
    training_generator_args::Union{Tuple,NamedTuple},
    testing_generator_args::Union{Tuple,NamedTuple},
    t::AbstractVector,
    model::LikelihoodModel,
    N::Int,
    num_points::Union{Int,Vector{<:Int}},
    θtrue::AbstractVector{<:Real},
    θcombinations::Union{Vector{Vector{Int}},Vector{Tuple{Int,Int}}},
    θinitialguess::AbstractVector{<:Real}=θtrue;
    num_internal_points::Int=0,
    hullmethod::AbstractBivariateHullMethod=MPPHullMethod(),
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    confidence_level::Float64=0.95,
    dof::Int=2,
    region::Float64=0.95,
    profile_type::AbstractProfileType=LogLikelihood(),
    method::Union{AbstractBivariateMethod,Vector{<:AbstractBivariateMethod}}=RadialRandomMethod(3),
    θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
    θub_nuisance::AbstractVector{<:Real}=model.core.θub,
    coverage_estimate_confidence_level::Float64=0.95,
    simultaneous_alternate_proportion::Float64=0.95,
    optimizationsettings::Union{OptimizationSettings,Missing}=missing,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false,
    manual_GC_calls::Bool=false)

    function argument_handling!()
        length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))
        length(θinitialguess) == model.core.num_pars || throw(ArgumentError("θinitialguess must have the same length as the number of model parameters"))
        check_prediction_function_exists(model::LikelihoodModel) || throw(ArgumentError("see warning message"))

        num_internal_points >= 0 || throw(DomainError("num_internal_points must be a strictly positive integer"))

        (0.0 < coverage_estimate_confidence_level && coverage_estimate_confidence_level < 1.0) || throw(DomainError("coverage_estimate_confidence_level must be in the open interval (0,1)"))
        get_target_loglikelihood(model, confidence_level, profile_type, 1)

        (0.0 <= region <= 1.0) || throw(DomainError("region must be in the closed interval [0.0, 1.0]"))

        !xor(num_points isa Vector, method isa Vector) || throw(ArgumentError("num_points and method must both be a Vector, or both be a Int and AbstractBivariateMethod, respectively, at the same time (xnor gate)"))
        combine_methods = num_points isa Vector
        if combine_methods
            (length(num_points) == length(method)) || throw(ArgumentError("num_points must have the same length as method, each index in num_points corresponds to the number of boundary points for the corresponding index in method"))
        end

        N > 0 || throw(DomainError("N must be greater than 0"))
        (dof ≥ 2) || throw(DomainError("dof must be greater than or equal to 2. Setting to 2 is generally recommended"))

        length(θlb_nuisance) == model.core.num_pars || throw(ArgumentError("θlb_nuisance must have the same length as the number of model parameters"))
        length(θub_nuisance) == model.core.num_pars || throw(ArgumentError("θub_nuisance must have the same length as the number of model parameters"))
        all(θlb_nuisance .≤ model.core.θmle) || throw(DomainError("θlb_nuisance must be less than or equal to model.core.θmle"))
        all(θub_nuisance .≥ model.core.θmle) || throw(DomainError("θub_nuisance must be greater than or equal to model.core.θmle"))

        !ismissing(model.core.errorfunction) || throw(ArgumentError("model must contain an error function for creating prediction realisation confidence intervals. Add one when creating model with initialise_LikelihoodModel or using add_error_function!"))

        if θcombinations isa Vector{Tuple{Int,Int}}
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

    y_true = model.core.predictfunction(θtrue, model.core.data, t)
    multiple_outputs = ndims(y_true) == 2

    len_θs = length(θcombinations)

    successes_reference = zeros(Int, len_θs*2)
    successes_reference_pointwise = [zeros(size(y_true)) for _ in 1:(len_θs*2)]
    successes = zeros(Int, len_θs * 2)
    successes_alternate = zeros(Int, len_θs * 2)
    successes_pointwise = [zeros(size(y_true)) for _ in 1:(len_θs*2)]
    iteration_is_included = falses(N)

    data_training = [data_generator(θtrue, training_generator_args) for _ in 1:N]
    data_testing = [data_generator(θtrue, testing_generator_args) for _ in 1:N]
    reference_set_testing = reference_set_generator(θtrue, testing_generator_args, region)

    channel = RemoteChannel(() -> Channel{Bool}(1))
    p = Progress(N; desc="Computing bivariate prediction realisation coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for i in 1:N
            new_data = data_training[i]

            m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction,
                model.core.errorfunction,
                new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub,
                model.core.θmagnitudes; biv_row_preallocation_size=len_θs, show_progress=false,
                optimizationsettings=model.core.optimizationsettings)

            lb, ub = correct_θbounds_nuisance(m_new, θlb_nuisance, θub_nuisance)

            if combine_methods
                for (j, methodj) in enumerate(method)
                    bivariate_confidenceprofiles!(m_new, deepcopy(θcombinations), num_points[j];
                        confidence_level=confidence_level, dof=dof, 
                        profile_type=profile_type, method=methodj,
                        θlb_nuisance=lb, θub_nuisance=ub,
                        use_threads=false,
                        optimizationsettings=optimizationsettings)
                end
                combine_bivariate_boundaries!(m_new, confidence_level=confidence_level, dof=dof,
                    not_evaluated_predictions=true)
            else
                bivariate_confidenceprofiles!(m_new, deepcopy(θcombinations), num_points;
                    confidence_level=confidence_level, dof=dof, 
                    profile_type=profile_type, method=method,
                    θlb_nuisance=lb, θub_nuisance=ub,
                    use_threads=false,
                    optimizationsettings=optimizationsettings)
            end

            if num_internal_points > 0
                sample_bivariate_internal_points!(m_new, num_internal_points,
                    sample_type=sample_type, hullmethod=hullmethod, 
                    θlb_nuisance=lb, θub_nuisance=ub, use_threads=false,
                    optimizationsettings=optimizationsettings)
            end

            generate_predictions_bivariate!(m_new, t, 0.0, region=region)

            indiv_cov, union_cov, iteration_is_included[i] = evaluate_coverage_realisations(m_new, data_testing[i], :bivariate, multiple_outputs, len_θs)
            successes[1:len_θs] .+= first.(indiv_cov)
            successes[len_θs+1:end] .+= first.(union_cov)

            successes_pointwise[1:len_θs] .+= last.(indiv_cov)
            successes_pointwise[len_θs+1:end] .+= last.(union_cov)

            successes_alternate[1:len_θs] .+= evaluate_conf_simultaneous_coverage.(last.(indiv_cov), Ref(simultaneous_alternate_proportion))
            successes_alternate[len_θs+1:end] .+= evaluate_conf_simultaneous_coverage.(last.(union_cov), Ref(simultaneous_alternate_proportion))

            indiv_cov, union_cov = evaluate_coverage_reference_sets(m_new, reference_set_testing, :bivariate, multiple_outputs, len_θs, iteration_is_included[i])
            successes_reference[1:len_θs] .+= first.(indiv_cov)
            successes_reference[len_θs+1:end] .+= first.(union_cov)

            successes_reference_pointwise[1:len_θs] .+= last.(indiv_cov)
            successes_reference_pointwise[len_θs+1:end] .+= last.(union_cov)
            next!(p)
            if manual_GC_calls && rem(i, 5) == 0
                @everywhere GC.gc()
            end
        end
    else
        successes_reference_bool = SharedArray{Bool}(len_θs*2, N)
        successes_reference_bool .= false
        successes_alternate_bool = SharedArray{Bool}(len_θs*2, N)
        successes_alternate_bool .= false
        successes_bool = SharedArray{Bool}(len_θs*2, N)
        successes_bool .= false
        iteration_is_included_shared = SharedArray{Bool}(N)
        iteration_is_included_shared .= false
        @sync begin
            @async while take!(channel)
                next!(p)
            end

            @async begin
                successes_pointwise_bool_ = @distributed (vcat) for i in 1:N
                    new_data = data_training[i]

                    m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction,
                        model.core.errorfunction,
                        new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub,
                        model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false,
                        optimizationsettings=model.core.optimizationsettings)

                    lb, ub = correct_θbounds_nuisance(m_new, θlb_nuisance, θub_nuisance)

                    if combine_methods
                        for (j, methodj) in enumerate(method)
                            bivariate_confidenceprofiles!(m_new, deepcopy(θcombinations), num_points[j];
                                confidence_level=confidence_level, dof=dof, 
                                profile_type=profile_type, method=methodj,
                                θlb_nuisance=lb, θub_nuisance=ub,
                                use_distributed=false, use_threads=false,
                                optimizationsettings=optimizationsettings)
                        end
                        combine_bivariate_boundaries!(m_new, confidence_level=confidence_level, dof=dof,
                            not_evaluated_predictions=true)
                    else
                        bivariate_confidenceprofiles!(m_new, deepcopy(θcombinations), num_points;
                            confidence_level=confidence_level, dof=dof,
                            profile_type=profile_type, method=method,
                            θlb_nuisance=lb, θub_nuisance=ub,
                            use_distributed=false, use_threads=false,
                            optimizationsettings=optimizationsettings)
                    end

                    if num_internal_points > 0
                        sample_bivariate_internal_points!(m_new, num_internal_points,
                            sample_type=sample_type, hullmethod=hullmethod,
                            θlb_nuisance=lb, θub_nuisance=ub, 
                            use_distributed=false, use_threads=false,
                            optimizationsettings=optimizationsettings)
                    end

                    generate_predictions_bivariate!(m_new, t, 0.0, region=region, use_distributed=false)

                    indiv_cov, union_cov, iteration_is_included_shared[i] = evaluate_coverage_realisations(m_new, data_testing[i], :bivariate, multiple_outputs, len_θs)
                    successes_bool[1:len_θs, i] .= first.(indiv_cov)
                    successes_bool[len_θs+1:end, i] .= first.(union_cov)

                    successes_alternate_bool[1:len_θs, i] .= evaluate_conf_simultaneous_coverage.(last.(indiv_cov), Ref(simultaneous_alternate_proportion))
                    successes_alternate_bool[len_θs+1:end, i] .= evaluate_conf_simultaneous_coverage.(last.(union_cov), Ref(simultaneous_alternate_proportion))

                    indiv_cov_ref, union_cov_ref = evaluate_coverage_reference_sets(m_new, reference_set_testing, :bivariate, multiple_outputs, len_θs, iteration_is_included_shared[i])
                    successes_reference_bool[1:len_θs, i] .= first.(indiv_cov_ref)
                    successes_reference_bool[len_θs+1:end, i] .= first.(union_cov_ref)

                    if manual_GC_calls
                        GC.gc()
                    end
                    put!(channel, true)
                    (vcat(last.(indiv_cov), last.(union_cov)), vcat(last.(indiv_cov_ref), last.(union_cov_ref)))
                end
                put!(channel, false)
                iteration_is_included .= iteration_is_included_shared
                for (pointwise_bool, pointwise_reference_bool) in successes_pointwise_bool_
                    successes_pointwise .+= pointwise_bool
                    successes_reference_pointwise .+= pointwise_reference_bool
                end
            end
        end
        successes .= sum(successes_bool, dims=2)
        successes_alternate .= sum(successes_alternate_bool, dims=2)
        successes_reference .= sum(successes_reference_bool, dims=2)
    end

    N_counted = sum(iteration_is_included)
    coverage_realisations = successes ./ N_counted
    coverage_reference_sets = successes_reference ./ N_counted
    coverage_realisations_alternate = successes_alternate ./ N_counted
    conf_ints_realisations = zeros(len_θs*2, 2)
    conf_ints_realisations_alternate = zeros(len_θs*2, 2)
    conf_ints_reference_sets = zeros(len_θs*2, 2)
    for i in 1:(len_θs*2)
        conf_ints_realisations[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N_counted),
            level=coverage_estimate_confidence_level)
        conf_ints_realisations_alternate[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes_alternate[i], N_counted),
            level=coverage_estimate_confidence_level)
        conf_ints_reference_sets[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes_reference[i], N_counted),
            level=coverage_estimate_confidence_level)
        successes_pointwise[i] = successes_pointwise[i] ./ N_counted
        successes_reference_pointwise[i] = successes_reference_pointwise[i] ./ N_counted
    end

    num_boundary_points = zeros(Int, len_θs*2)
    num_boundary_points[1:len_θs] .= sum(num_points)
    num_boundary_points[len_θs+1:end] .= sum(num_points) .* collect(1:len_θs)
    num_internal_points_all = zeros(Int, len_θs*2)
    num_internal_points_all[1:len_θs] .= num_internal_points
    num_internal_points_all[len_θs+1:end] .= num_internal_points .* collect(1:len_θs)

    if manual_GC_calls
        GC.gc()
    end

    return DataFrame(θname=[[model.core.θnames[[combo...]] for combo in θcombinations]..., fill("", len_θs)...], θindices=[θcombinations..., fill([0], len_θs)...],
        n_random_combinations=[fill(0, len_θs)..., collect(1:len_θs)...],
        simultaneous_coverage_reference_sets=coverage_reference_sets, coverage_reference_sets_lb=conf_ints_reference_sets[:, 1], coverage_reference_sets_ub=conf_ints_reference_sets[:, 2],
        pointwise_coverage_reference_sets=successes_reference_pointwise,
        simultaneous_coverage_realisations=coverage_realisations, coverage_realisations_lb=conf_ints_realisations[:, 1], coverage_realisations_ub=conf_ints_realisations[:, 2],
        simultaneous_coverage_realisations_alternate=coverage_realisations_alternate, coverage_realisations_alternate_lb=conf_ints_realisations_alternate[:, 1],
        coverage_realisations_alternate_ub=conf_ints_realisations_alternate[:, 2],
        pointwise_coverage_realisations=successes_pointwise,
        num_boundary_points=num_boundary_points,
        num_internal_points=num_internal_points_all)
end