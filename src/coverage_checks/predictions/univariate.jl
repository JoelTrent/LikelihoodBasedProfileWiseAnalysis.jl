"""
    check_univariate_prediction_coverage(data_generator::Function, 
        generator_args::Union{Tuple, NamedTuple},
        model::LikelihoodModel, 
        N::Int, 
        θtrue::AbstractVector{<:Real}, 
        θs::AbstractVector{<:Int64},
        θinitialguess::AbstractVector{<:Real}=θtrue; 
        <keyword arguments>)

Performs a simulation to estimate the prediction coverage of univariate confidence profiles for parameters in `θs` given a model by: 
    
1. Repeatedly drawing new observed data using `data_generator` for fixed true parameter values, `θtrue`, and fixed true prediction value. 
2. Fitting the model and univariate confidence intervals. 
3. Sampling points along the profile within the confidence intervals.
4. Evaluating predictions from the points in the profile and finding the prediction extrema.
5. Checking whether the prediction extrema contain the true prediction value(s), in a pointwise and simultaneous fashion. The estimated simultaneous coverage is returned with a default 95% confidence interval within a DataFrame. 

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. Outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`.
- `generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `generator_args`. 
- `t`: a vector of time points to compute predictions and evaluate coverage at.
- `model`: a [`LikelihoodModel`](@ref) containing model information.
- `N`: a positive number of coverage simulations.
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θs`: a vector of parameters to profile, as a vector of model parameter indexes.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `num_points_in_interval`: an integer number of points to optionally evaluate within the confidence interval for each interest parameter using [`get_points_in_intervals!`](@ref). Points are linearly spaced in the interval. Useful for predictions from univariate profiles. Default is `0`. 
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to evaluate the confidence interval coverage at. Default is `0.95` (95%).
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `θlb_nuisance`: a vector of lower bounds on nuisance parameters, require `θlb_nuisance .≤ model.core.θmle`. Default is `model.core.θlb`. 
- `θub_nuisance`: a vector of upper bounds on nuisance parameters, require `θub_nuisance .≥ model.core.θmle`. Default is `model.core.θub`.
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is `0.95` (95%).
- `optimizationsettings`: a [`OptimizationSettings`](@ref) containing the optimisation settings used to find optimal values of nuisance parameters for a given interest parameter value. Default is `missing` (will use `default_OptimizationSettings()` (see [`default_OptimizationSettings`](@ref)).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations (false) or across the individual confidence interval calculations within each iteration (true). Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of propagating univariate parameter confidence intervals into prediction space. The simulation uses [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) to parallelise the workload.

The uncertainty in estimates of the prediction coverage under the simulated model will decrease as the number of simulations, `N`, is increased. Confidence intervals for the coverage estimate are provided to quantify this uncertainty. The confidence interval for the estimated coverage is a Clopper-Pearson interval on a binomial test generated using [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/).

!!! note "Recommended setting for distributed_over_parameters"
    - If the number of processes available to use is significantly greater than the number of model parameters or only a few model parameters are being checked for coverage, `false` is recommended.   
    - If system memory or model size in system memory is a concern, or the number of processes available is similar or less than the number of model parameters being checked, `true` will likely be more appropriate. 
    - When set to `false`, a separate [`LikelihoodModel`](@ref) struct will be used by each process, as opposed to only one when set to `true`, which could cause a memory issue for larger models. 

!!! danger "Not intended for use on bimodal univariate profile likelihoods"
    The current implementation only considers two extremes of the log-likelihood and whether the truth is between these two points. If the profile likelihood function is bimodal, it's possible the method has only found one set of correct confidence intervals (estimated coverage will be correct, but less than expected) or found one extrema on distinct sets (estimated coverage may be incorrect and will either be larger than expected or much lower than expected). 
"""
function check_univariate_prediction_coverage(data_generator::Function, 
    generator_args::Union{Tuple, NamedTuple},
    t::AbstractVector,
    model::LikelihoodModel, 
    N::Int, 
    θtrue::AbstractVector{<:Real}, 
    θs::AbstractVector{<:Int64},
    θinitialguess::AbstractVector{<:Real}=θtrue;
    num_points_in_interval::Int=0,
    confidence_level::Float64=0.95, 
    profile_type::AbstractProfileType=LogLikelihood(),
    θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
    θub_nuisance::AbstractVector{<:Real}=model.core.θub,
    coverage_estimate_confidence_level::Float64=0.95,
    optimizationsettings::Union{OptimizationSettings,Missing}=missing,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false)

    function argument_handling!()
        length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))
        length(θinitialguess) == model.core.num_pars || throw(ArgumentError("θinitialguess must have the same length as the number of model parameters"))
        check_prediction_function_exists(model::LikelihoodModel) || throw(ArgumentError("see warning message"))
        
        num_points_in_interval >= 0 || throw(DomainError("num_points_in_interval must be a strictly positive integer"))
        
        (0.0 < coverage_estimate_confidence_level && coverage_estimate_confidence_level < 1.0) || throw(DomainError("coverage_estimate_confidence_level must be in the open interval (0,1)"))
        get_target_loglikelihood(model, confidence_level, profile_type, 1)
        
        N > 0 || throw(DomainError("N must be greater than 0"))
        
        length(θlb_nuisance) == model.core.num_pars || throw(ArgumentError("θlb_nuisance must have the same length as the number of model parameters"))
        length(θub_nuisance) == model.core.num_pars || throw(ArgumentError("θub_nuisance must have the same length as the number of model parameters"))
        all(θlb_nuisance .≤ model.core.θmle) || throw(DomainError("θlb_nuisance must be less than or equal to model.core.θmle"))
        all(θub_nuisance .≥ model.core.θmle) || throw(DomainError("θub_nuisance must be greater than or equal to model.core.θmle"))

        (sort(θs); unique!(θs))
        1 ≤ θs[1] && θs[end] ≤ model.core.num_pars || throw(DomainError("θs can only contain parameter indexes between 1 and the number of model parameters"))
        return nothing
    end
    
    argument_handling!()
    y_true = model.core.predictfunction(θtrue, model.core.data, t)
    multiple_outputs = ndims(y_true) == 2

    len_θs = length(θs)
    # θs_to_θi = Dict{Int,Int}(θindex => θi for (θi, θindex) in enumerate(θs))

    successes = zeros(Int, len_θs+1)
    successes_pointwise = [zeros(size(y_true)) for _ in 1:(len_θs+1)]
    iteration_is_included = falses(N) 

    data = [data_generator(θtrue, generator_args) for _ in 1:N]

    channel = RemoteChannel(() -> Channel{Bool}(1))
    p = Progress(N; desc="Computing univariate prediction coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for i in 1:N
            new_data = data[i]

            m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction,
                new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, 
                model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false,
                optimizationsettings=model.core.optimizationsettings)

            lb, ub = correct_θbounds_nuisance(m_new, θlb_nuisance, θub_nuisance)

            univariate_confidenceintervals!(m_new, deepcopy(θs);
                num_points_in_interval=num_points_in_interval, confidence_level=confidence_level, 
                profile_type=profile_type, use_threads=false,
                θlb_nuisance=lb, θub_nuisance=ub,
                optimizationsettings=optimizationsettings)

            generate_predictions_univariate!(m_new, t, 0.0)

            indiv_cov, union_cov, iteration_is_included[i] = evaluate_coverage(m_new, y_true, :univariate, multiple_outputs, len_θs)
            successes[1:len_θs] .+= first.(indiv_cov)
            successes[end] += first(union_cov)

            successes_pointwise[1:len_θs] .+= last.(indiv_cov)
            successes_pointwise[end] += last(union_cov)

            # for row_ind in 1:m_new.num_uni_profiles
            #     θindex = m_new.uni_profiles_df[row_ind, :θindex]

            #     successes[θs_to_θi[θindex]] .+= first(indiv_cov[row_ind])
            #     successes[end] += first(union_cov)

            #     successes_pointwise[θs_to_θi[θindex]] .+= last(indiv_cov[row_ind])
            #     successes_pointwise[end] += last(union_cov)
            # end

            next!(p)
        end
    else
        successes_bool = SharedArray{Bool}(len_θs+1, N)
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

                    univariate_confidenceintervals!(m_new, deepcopy(θs); 
                        num_points_in_interval=num_points_in_interval, confidence_level=confidence_level, 
                        profile_type=profile_type, use_distributed=false, use_threads=false,
                        θlb_nuisance=lb, θub_nuisance=ub,
                        optimizationsettings=optimizationsettings)

                    generate_predictions_univariate!(m_new, t, 0.0, use_distributed=false)

                    indiv_cov, union_cov, iteration_is_included_shared[i] = evaluate_coverage(m_new, y_true, :univariate, multiple_outputs, len_θs)
                    successes_bool[1:len_θs, i] .= first.(indiv_cov)
                    successes_bool[end, i] = first(union_cov)

                    put!(channel, true)
                    (vcat(last.(indiv_cov), [last(union_cov)]),)
                end
                put!(channel, false)
                iteration_is_included .= iteration_is_included_shared
                for (i, pointwise_bool) in enumerate(successes_pointwise_bool)
                    successes_pointwise .+= first(pointwise_bool)
                end
            end
        end
        successes .= sum(successes_bool, dims=2)
    end

    N_counted = sum(iteration_is_included)
    coverage = successes ./ N_counted
    conf_ints = zeros(len_θs+1, 2)
    for i in 1:(len_θs+1)
        conf_ints[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N_counted),
            level=coverage_estimate_confidence_level) 
        successes_pointwise[i] = successes_pointwise[i] ./ N_counted
    end

    points_in_interval = zeros(Int, len_θs+1)
    points_in_interval[1:len_θs] .= num_points_in_interval
    points_in_interval[end] = num_points_in_interval*len_θs

    return DataFrame(θname=[model.core.θnames[θs]..., :union], θindex=[θs..., θs], 
        simultaneous_coverage=coverage, coverage_lb=conf_ints[:,1], coverage_ub=conf_ints[:,2],
        pointwise_coverage=successes_pointwise,
        num_points_in_interval=points_in_interval)
end

"""
    check_univariate_prediction_realisations_coverage(data_generator::Function, 
        training_generator_args::Union{Tuple, NamedTuple},
        testing_generator_args::Union{Tuple, NamedTuple},
        model::LikelihoodModel, 
        N::Int, 
        θtrue::AbstractVector{<:Real}, 
        θs::AbstractVector{<:Int64},
        θinitialguess::AbstractVector{<:Real}=θtrue; 
        <keyword arguments>)

Performs a simulation to estimate the prediction reference set and realisation coverage of univariate confidence profiles for parameters in `θs` given a model by: 

1. Constructing the `confidence_level` reference set for predictions from the fixed true parameter values, `θ_true`.
2. Repeatedly drawing new observed training data using `data_generator` and `training_generator_args` for fixed true parameter values, `θtrue`, and fixed true prediction value. 
3. Fitting the model using training data.
4. Fitting the model and univariate confidence intervals using training data. 
5. Sampling points along the profile within the confidence intervals.
6. Evaluating predictions from the points in the profile and finding the prediction extrema (reference tolerance sets).
7. Drawing new observed testing data using `data_generator` and `training_generator_args` for fixed true parameter values, `θtrue`, and fixed true prediction value. 
8. Checking whether the prediction extrema (reference tolerance set) contains the prediction reference set from Step 1, in a pointwise and simultaneous fashion. 
9. Checking whether the prediction extrema contain the observed testing data, in a pointwise and simultaneous fashion. 
10. The estimated simultaneous coverage of the reference set and the prediction realisations (observed testing data) is returned with a default 95% confidence interval, alongside pointwise coverage, within a DataFrame.  

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. When used with `training_generator_args`, it outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`. When used with `testing_generator_args`, it outputs an array containing the observed data to use as the test data set.
- `reference_set_generator`: a function with three arguments which generates the `confidence_level` data reference set for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The three arguments must be the vector of true model parameters, `θtrue`, a Tuple or NamedTuple, `generator_args`, and a number (0.0, 1.0) for the confidence level at which to evaluate the reference set. When used with `testing_generator_args` it outputs a tuple of two arrays, `(lq, uq)`, which contain the lower and upper quantiles of the reference set. 
- `training_generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at, used to create the training set of data. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `training_generator_args`. 
- `testing_generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at, used to create the test data set. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `testing_generator_args`. 
- `t`: a vector of time points to compute predictions and evaluate coverage at, which are the same as the time points used to create the test data set.
- `model`: a [`LikelihoodModel`](@ref) containing model information.
- `N`: a positive number of coverage simulations.
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θs`: a vector of parameters to profile, as a vector of model parameter indexes.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `num_points_in_interval`: an integer number of points to optionally evaluate within the confidence interval for each interest parameter using [`get_points_in_intervals!`](@ref). Points are linearly spaced in the interval. Useful for predictions from univariate profiles. Default is `0`. 
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to evaluate the confidence interval coverage at. Default is `0.95` (95%).
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `θlb_nuisance`: a vector of lower bounds on nuisance parameters, require `θlb_nuisance .≤ model.core.θmle`. Default is `model.core.θlb`. 
- `θub_nuisance`: a vector of upper bounds on nuisance parameters, require `θub_nuisance .≥ model.core.θmle`. Default is `model.core.θub`.
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is `0.95` (95%).
- `optimizationsettings`: a [`OptimizationSettings`](@ref) containing the optimisation settings used to find optimal values of nuisance parameters for a given interest parameter value. Default is `missing` (will use `default_OptimizationSettings()` (see [`default_OptimizationSettings`](@ref)).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations (false) or across the individual confidence interval calculations within each iteration (true). Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of propagating univariate parameter confidence intervals into prediction realisation space. The simulation uses [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) to parallelise the workload.

The uncertainty in estimates of the prediction realisation coverage under the simulated model will decrease as the number of simulations, `N`, is increased. Confidence intervals for the coverage estimate are provided to quantify this uncertainty. The confidence interval for the estimated coverage is a Clopper-Pearson interval on a binomial test generated using [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/).

!!! note "Recommended setting for distributed_over_parameters"
    - If the number of processes available to use is significantly greater than the number of model parameters or only a few model parameters are being checked for coverage, `false` is recommended.   
    - If system memory or model size in system memory is a concern, or the number of processes available is similar or less than the number of model parameters being checked, `true` will likely be more appropriate. 
    - When set to `false`, a separate [`LikelihoodModel`](@ref) struct will be used by each process, as opposed to only one when set to `true`, which could cause a memory issue for larger models. 

!!! danger "Not intended for use on bimodal univariate profile likelihoods"
    The current implementation only considers two extremes of the log-likelihood and whether the truth is between these two points. If the profile likelihood function is bimodal, it's possible the method has only found one set of correct confidence intervals (estimated coverage will be correct, but less than expected) or found one extrema on distinct sets (estimated coverage may be incorrect and will either be larger than expected or much lower than expected). 
"""
function check_univariate_prediction_realisations_coverage(data_generator::Function,
    reference_set_generator::Function,
    training_generator_args::Union{Tuple,NamedTuple},
    testing_generator_args::Union{Tuple,NamedTuple},
    t::AbstractVector,
    model::LikelihoodModel,
    N::Int,
    θtrue::AbstractVector{<:Real},
    θs::AbstractVector{<:Int64},
    θinitialguess::AbstractVector{<:Real}=θtrue;
    num_points_in_interval::Int=0,
    confidence_level::Float64=0.95,
    profile_type::AbstractProfileType=LogLikelihood(),
    θlb_nuisance::AbstractVector{<:Real}=model.core.θlb,
    θub_nuisance::AbstractVector{<:Real}=model.core.θub,
    coverage_estimate_confidence_level::Float64=0.95,
    optimizationsettings::Union{OptimizationSettings,Missing}=missing,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false)

    function argument_handling!()
        length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))
        length(θinitialguess) == model.core.num_pars || throw(ArgumentError("θinitialguess must have the same length as the number of model parameters"))
        check_prediction_function_exists(model::LikelihoodModel) || throw(ArgumentError("see warning message"))

        num_points_in_interval >= 0 || throw(DomainError("num_points_in_interval must be a strictly positive integer"))

        (0.0 < coverage_estimate_confidence_level && coverage_estimate_confidence_level < 1.0) || throw(DomainError("coverage_estimate_confidence_level must be in the open interval (0,1)"))
        get_target_loglikelihood(model, confidence_level, profile_type, 1)

        N > 0 || throw(DomainError("N must be greater than 0"))

        length(θlb_nuisance) == model.core.num_pars || throw(ArgumentError("θlb_nuisance must have the same length as the number of model parameters"))
        length(θub_nuisance) == model.core.num_pars || throw(ArgumentError("θub_nuisance must have the same length as the number of model parameters"))
        all(θlb_nuisance .≤ model.core.θmle) || throw(DomainError("θlb_nuisance must be less than or equal to model.core.θmle"))
        all(θub_nuisance .≥ model.core.θmle) || throw(DomainError("θub_nuisance must be greater than or equal to model.core.θmle"))

        !ismissing(model.core.errorfunction) || throw(ArgumentError("model must contain an error function for creating prediction realisation confidence intervals. Add one when creating model with initialise_LikelihoodModel or using add_error_function!"))

        (sort(θs); unique!(θs))
        1 ≤ θs[1] && θs[end] ≤ model.core.num_pars || throw(DomainError("θs can only contain parameter indexes between 1 and the number of model parameters"))
        return nothing
    end

    argument_handling!()

    bonferroni_confidence_level = 1.0 - ((1.0-confidence_level)/2.0)

    y_true = model.core.predictfunction(θtrue, model.core.data, t)
    multiple_outputs = ndims(y_true) == 2

    len_θs = length(θs)
    # θs_to_θi = Dict{Int,Int}(θindex => θi for (θi, θindex) in enumerate(θs))

    successes_reference = zeros(Int, len_θs + 1)
    successes_reference_pointwise = [zeros(size(y_true)) for _ in 1:(len_θs+1)]
    successes = zeros(Int, len_θs + 1)
    successes_pointwise = [zeros(size(y_true)) for _ in 1:(len_θs+1)]
    iteration_is_included = falses(N)

    data_training = [data_generator(θtrue, training_generator_args) for _ in 1:N]
    data_testing = [data_generator(θtrue, testing_generator_args) for _ in 1:N]
    reference_set_testing = reference_set_generator(θtrue, testing_generator_args, confidence_level)

    channel = RemoteChannel(() -> Channel{Bool}(1))
    p = Progress(N; desc="Computing univariate prediction realisation coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for i in 1:N
            new_data = data_training[i]

            m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction,
                model.core.errorfunction,
                new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub,
                model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false,
                optimizationsettings=model.core.optimizationsettings)

            lb, ub = correct_θbounds_nuisance(m_new, θlb_nuisance, θub_nuisance)

            univariate_confidenceintervals!(m_new, deepcopy(θs);
                num_points_in_interval=num_points_in_interval, confidence_level=bonferroni_confidence_level,
                θlb_nuisance=lb, θub_nuisance=ub,
                profile_type=profile_type, use_threads=false,
                optimizationsettings=optimizationsettings)

            generate_predictions_univariate!(m_new, t, 0.0)

            indiv_cov, union_cov, iteration_is_included[i] = evaluate_coverage_realisations(m_new, data_testing[i], :univariate, multiple_outputs, len_θs)
            successes[1:len_θs] .+= first.(indiv_cov)
            successes[end] += first(union_cov)

            successes_pointwise[1:len_θs] .+= last.(indiv_cov)
            successes_pointwise[end] += last(union_cov)

            indiv_cov, union_cov = evaluate_coverage_reference_sets(m_new, reference_set_testing, :univariate, multiple_outputs, len_θs, iteration_is_included[i])
            successes_reference[1:len_θs] .+= first.(indiv_cov)
            successes_reference[end] += first(union_cov)

            successes_reference_pointwise[1:len_θs] .+= last.(indiv_cov)
            successes_reference_pointwise[end] += last(union_cov)

            next!(p)
        end
    else
        successes_reference_bool = SharedArray{Bool}(len_θs + 1, N)
        successes_reference_bool .= false
        successes_bool = SharedArray{Bool}(len_θs + 1, N)
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

                    univariate_confidenceintervals!(m_new, deepcopy(θs);
                        num_points_in_interval=num_points_in_interval, confidence_level=bonferroni_confidence_level,
                        profile_type=profile_type, use_distributed=false,
                        θlb_nuisance=lb, θub_nuisance=ub, use_threads=false,
                        optimizationsettings=optimizationsettings)

                    generate_predictions_univariate!(m_new, t, 0.0, use_distributed=false)

                    indiv_cov, union_cov, iteration_is_included_shared[i] = evaluate_coverage_realisations(m_new, data_testing[i], :univariate, multiple_outputs, len_θs)
                    successes_bool[1:len_θs, i] .= first.(indiv_cov)
                    successes_bool[end, i] = first(union_cov)

                    indiv_cov_ref, union_cov_ref = evaluate_coverage_reference_sets(m_new, reference_set_testing, :univariate, multiple_outputs, len_θs, iteration_is_included_shared[i])
                    successes_reference_bool[1:len_θs, i] .= first.(indiv_cov_ref)
                    successes_reference_bool[end, i] = first(union_cov_ref)

                    put!(channel, true)
                    (vcat(last.(indiv_cov), [last(union_cov)]), vcat(last.(indiv_cov_ref), [last(union_cov_ref)]))
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
        successes_reference .= sum(successes_reference_bool, dims=2)
    end

    N_counted = sum(iteration_is_included)
    coverage_realisations = successes ./ N_counted
    coverage_reference_sets = successes_reference ./ N_counted
    conf_ints_realisations = zeros(len_θs + 1, 2)
    conf_ints_reference_sets = zeros(len_θs + 1, 2)
    for i in 1:(len_θs+1)
        conf_ints_realisations[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N_counted),
            level=coverage_estimate_confidence_level)
        conf_ints_reference_sets[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes_reference[i], N_counted),
            level=coverage_estimate_confidence_level)
        successes_pointwise[i] = successes_pointwise[i] ./ N_counted
        successes_reference_pointwise[i] = successes_reference_pointwise[i] ./ N_counted
    end

    points_in_interval = zeros(Int, len_θs + 1)
    points_in_interval[1:len_θs] .= num_points_in_interval
    points_in_interval[end] = num_points_in_interval * len_θs


    return DataFrame(θname=[model.core.θnames[θs]..., :union], θindex=[θs..., θs],
        simultaneous_coverage_reference_sets=coverage_reference_sets, coverage_reference_sets_lb=conf_ints_reference_sets[:, 1], coverage_reference_sets_ub=conf_ints_reference_sets[:, 2],
        pointwise_coverage_reference_sets=successes_reference_pointwise,
        simultaneous_coverage_realisations=coverage_realisations, coverage_realisations_lb=conf_ints_realisations[:, 1], coverage_realisations_ub=conf_ints_realisations[:, 2],
        pointwise_coverage_realisations=successes_pointwise,
        num_points_in_interval=points_in_interval)
end