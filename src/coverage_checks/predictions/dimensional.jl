"""
    check_dimensional_prediction_coverage(data_generator::Function, 
        generator_args::Union{Tuple, NamedTuple},
        t::AbstractVector,
        model::LikelihoodModel, 
        N::Int, 
        num_points_to_sample::Union{Int, Vector{<:Int}},
        θtrue::AbstractVector{<:Real}, 
        θindices::Union{Vector{Vector{Int}}, Vector{Vector{Symbol}}},
        θinitialguess::AbstractVector{<:Real}=θtrue;
        <keyword arguments>)

Performs a simulation to estimate the prediction coverage of dimensional confidence samples (including full likelihood samples) for parameters in `θindices` given a model by: 
    
1. Repeatedly drawing new observed data using `data_generator` for fixed true parameter values, θtrue, and fixed true prediction value. 
2. Fitting the model. 
3. Sampling points using `sample_type`.
4. Evaluating predictions from the points in the samples and finding the prediction extrema.
5. Checking whether the prediction extrema contain the true prediction value(s), in a pointwise and simultaneous fashion. The estimated simultaneous coverage is returned with a default 95% confidence interval within a DataFrame. 

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. Outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`.
- `generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `generator_args`. 
- `t`: a vector of time points to compute predictions and evaluate coverage at.
- `model`: a [`LikelihoodModel`](@ref) containing model information.
- `N`: a positive number of coverage simulations.
- `num_points_to_sample`: integer number of points to sample (for [`UniformRandomSamples`](@ref) and [`LatinHypercubeSamples`](@ref) sample types). For the [`UniformGridSamples`](@ref) sample type, if integer it is the number of points to grid over in each parameter dimension. If it is a vector of integers each index of the vector is the number of points to grid over in the corresponding parameter dimension. For example, [1,2] would mean a single point in dimension 1 and two points in dimension 2. 
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θindices`: a vector of vectors of parameter indexes for the combinations of interest parameters to samples points from.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to find samples within and evaluate coverage at. Default is `0.95` (95%).
- `sample_type`: the sampling method used to sample parameter space. Available sample types are [`UniformGridSamples`](@ref), [`UniformRandomSamples`](@ref) and [`LatinHypercubeSamples`](@ref). Default is `LatinHypercubeSamples()` ([`LatinHypercubeSamples`](@ref)).
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is `0.95` (95%).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations (false) or across the individual confidence interval calculations within each iteration (true). Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of propagating dimensional samples into prediction space. The simulation uses [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) to parallelise the workload.

The uncertainty in estimates of the prediction coverage under the simulated model will decrease as the number of simulations, `N`, is increased. Confidence intervals for the coverage estimate are provided to quantify this uncertainty. The confidence interval for the estimated coverage is a Clopper-Pearson interval on a binomial test generated using [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/).

!!! note "Recommended setting for distributed_over_parameters"
    - If the number of processes available to use is significantly greater than the number of model parameters or only a few pairs of model parameters are being checked for coverage, `false` is recommended.   
    - If system memory or model size in system memory is a concern, or the number of processes available is similar or less than the number of pairs of model parameters being checked, `true` will likely be more appropriate. 
    - When set to `false`, a separate [`LikelihoodModel`](@ref) struct will be used by each process, as opposed to only one when set to `true`, which could cause a memory issue for larger models. 
"""
function check_dimensional_prediction_coverage(data_generator::Function, 
    generator_args::Union{Tuple, NamedTuple},
    t::AbstractVector,
    model::LikelihoodModel, 
    N::Int, 
    num_points_to_sample::Union{Int, Vector{<:Int}},
    θtrue::AbstractVector{<:Real}, 
    θindices::Union{Vector{Vector{Int}}, Vector{Vector{Symbol}}},
    θinitialguess::AbstractVector{<:Real}=θtrue;
    confidence_level::Float64=0.95, 
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    coverage_estimate_confidence_level::Float64=0.95,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false)

    function argument_handling!()
        length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))
        length(θinitialguess) == model.core.num_pars || throw(ArgumentError("θinitialguess must have the same length as the number of model parameters"))
        check_prediction_function_exists(model::LikelihoodModel) || throw(ArgumentError("see warning message"))
        
        if num_points_to_sample isa Int
            num_points_to_sample > 0 || throw(DomainError("num_points_to_sample must be a strictly positive integer"))
        else
            minimum(num_points_to_sample) > 0 || throw(DomainError("num_points_to_sample must contain strictly positive integers"))

            sample_type isa UniformGridSamples || throw(ArgumentError(string("num_points_to_sample must be an integer for ", sample_type, " sample_type")))

            (length(num_points_to_sample) == length(θindices[1]) &&
                diff([extrema(length.(θindices))...])[1] == 0) || 
                throw(ArgumentError("num_points_to_sample must have the same length as each vector of interest parameters in num_points_to_sample"))
        end
        
        (0.0 < coverage_estimate_confidence_level && coverage_estimate_confidence_level < 1.0) || throw(DomainError("coverage_estimate_confidence_level must be in the open interval (0,1)"))
        get_target_loglikelihood(model, confidence_level, LogLikelihood(), 1)

        N > 0 || throw(DomainError("N must be greater than 0"))

        if θindices isa Vector{Vector{Symbol}}
            θindices = convertθnames_toindices(model, θnames)
        end

        θindices = θindices[.!isempty.(θindices)]
        sort!.(θindices); unique!.(θindices)
        sort!(θindices); unique!(θindices)
        1 ≤ first.(θindices)[1] && maximum(last.(θindices)) ≤ model.core.num_pars || throw(DomainError("θindices can only contain parameter indexes between 1 and the number of model parameters"))

        if any(length.(θindices) .== model.core.num_pars)
            i = findfirst(length.(θindices) .== model.core.num_pars)
            new_inds = vcat(i, setdiff(1:length(θindices), i))
            θindices = θindices[new_inds]
        end
        return nothing
    end
    argument_handling!()

    y_true = model.core.predictfunction(θtrue, model.core.data, t)
    multiple_outputs = ndims(y_true) == 2

    len_θs = length(θindices)

    successes = zeros(Int, len_θs+1)
    successes_pointwise = [zeros(size(y_true)) for _ in 1:(len_θs+1)]
    iteration_is_included = falses(N)

    data = [data_generator(θtrue, generator_args) for _ in 1:N]

    channel = RemoteChannel(() -> Channel{Bool}(1))
    p = Progress(N; desc="Computing dimensional samples prediction coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for i in 1:N
            new_data = data[i]

            m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction,
                new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, 
                model.core.θmagnitudes; biv_row_preallocation_size=len_θs, show_progress=false)

            dimensional_likelihood_samples!(m_new, deepcopy(θindices), num_points_to_sample,
                confidence_level=confidence_level, sample_type=sample_type,
                use_threads=false)

            generate_predictions_dim_samples!(m_new, t, 0.0)

            indiv_cov, union_cov, iteration_is_included[i] = evaluate_coverage(m_new, y_true, :dimensional, multiple_outputs, len_θs)            
            successes[1:len_θs] .+= first.(indiv_cov)
            successes[end] += first(union_cov)

            successes_pointwise[1:len_θs] .+= last.(indiv_cov)
            successes_pointwise[end] += last(union_cov)

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
                        model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false)

                    dimensional_likelihood_samples!(m_new, deepcopy(θindices), num_points_to_sample,
                        confidence_level=confidence_level, sample_type=sample_type,
                        use_distributed=false, use_threads=false)
        
                    generate_predictions_dim_samples!(m_new, t, 0.0, use_distributed=false)

                    indiv_cov, union_cov, iteration_is_included_shared[i] = evaluate_coverage(m_new, y_true, :dimensional, multiple_outputs, len_θs)
                    successes_bool[1:len_θs, i] .= first.(indiv_cov)
                    successes_bool[end, i] = first(union_cov)

                    put!(channel, true)
                    (vcat(last.(indiv_cov), [last(union_cov)]),)
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
    conf_ints = zeros(len_θs+1, 2)
    for i in 1:(len_θs+1)
        conf_ints[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N_counted), 
            level=coverage_estimate_confidence_level) 
        successes_pointwise[i] = successes_pointwise[i] ./ N_counted
    end

    return DataFrame(θname=[[model.core.θnames[[combo...]] for combo in θindices]..., :union], θindices=[θindices..., θindices], 
        simultaneous_coverage=coverage, coverage_lb=conf_ints[:,1], coverage_ub=conf_ints[:,2],
        pointwise_coverage=successes_pointwise)
end

"""
    check_dimensional_prediction_realisations_coverage(data_generator::Function, 
        training_generator_args::Union{Tuple,NamedTuple},
        testing_generator_args::Union{Tuple,NamedTuple},
        t::AbstractVector,
        model::LikelihoodModel, 
        N::Int, 
        num_points_to_sample::Union{Int, Vector{<:Int}},
        θtrue::AbstractVector{<:Real}, 
        θindices::Union{Vector{Vector{Int}}, Vector{Vector{Symbol}}},
        θinitialguess::AbstractVector{<:Real}=θtrue;
        <keyword arguments>)

Performs a simulation to estimate the prediction realisation coverage of dimensional confidence samples (including full likelihood samples) for parameters in `θindices` given a model by: 
    
1. Repeatedly drawing new observed training data using `data_generator` and `training_generator_args` for fixed true parameter values, θtrue, and fixed true prediction value. 
2. Fitting the model using training data.
3. Sampling points using `sample_type`.
4. Evaluating predictions from the points in the samples and finding the prediction extrema.
5. Drawing new observed testing data using `data_generator` and `training_generator_args` for fixed true paramter values, θtrue, and fixed true prediction value. 
6. Checking whether the prediction extrema contain the observed testing data, in a pointwise and simultaneous fashion. The estimated simultaneous coverage is returned with a default 95% confidence interval within a DataFrame. 

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. When used with `training_generator_args`, it outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`. When used with `testing_generator_args`, it outputs an array containing the observed data to use as the test data set.
- `training_generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at, used to create the training set of data. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `training_generator_args`. 
- `testing_generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at, used to create the test data set. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `testing_generator_args`. 
- `t`: a vector of time points to compute predictions and evaluate coverage at, which are the same as the time points used to create the test data set.
- `model`: a [`LikelihoodModel`](@ref) containing model information.
- `N`: a positive number of coverage simulations.
- `num_points_to_sample`: integer number of points to sample (for [`UniformRandomSamples`](@ref) and [`LatinHypercubeSamples`](@ref) sample types). For the [`UniformGridSamples`](@ref) sample type, if integer it is the number of points to grid over in each parameter dimension. If it is a vector of integers each index of the vector is the number of points to grid over in the corresponding parameter dimension. For example, [1,2] would mean a single point in dimension 1 and two points in dimension 2. 
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θindices`: a vector of vectors of parameter indexes for the combinations of interest parameters to samples points from.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to find samples within and evaluate coverage at. Default is `0.95` (95%).
- `sample_type`: the sampling method used to sample parameter space. Available sample types are [`UniformGridSamples`](@ref), [`UniformRandomSamples`](@ref) and [`LatinHypercubeSamples`](@ref). Default is `LatinHypercubeSamples()` ([`LatinHypercubeSamples`](@ref)).
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is `0.95` (95%).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations (false) or across the individual confidence interval calculations within each iteration (true). Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of propagating dimensional samples into prediction realisation space. The simulation uses [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) to parallelise the workload.

The uncertainty in estimates of the prediction realisation coverage under the simulated model will decrease as the number of simulations, `N`, is increased. Confidence intervals for the coverage estimate are provided to quantify this uncertainty. The confidence interval for the estimated coverage is a Clopper-Pearson interval on a binomial test generated using [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/).

!!! note "Recommended setting for distributed_over_parameters"
    - If the number of processes available to use is significantly greater than the number of model parameters or only a few pairs of model parameters are being checked for coverage, `false` is recommended.   
    - If system memory or model size in system memory is a concern, or the number of processes available is similar or less than the number of pairs of model parameters being checked, `true` will likely be more appropriate. 
    - When set to `false`, a separate [`LikelihoodModel`](@ref) struct will be used by each process, as opposed to only one when set to `true`, which could cause a memory issue for larger models. 
"""
function check_dimensional_prediction_realisations_coverage(data_generator::Function,
    training_generator_args::Union{Tuple,NamedTuple},
    testing_generator_args::Union{Tuple,NamedTuple},
    t::AbstractVector,
    model::LikelihoodModel,
    N::Int,
    num_points_to_sample::Union{Int,Vector{<:Int}},
    θtrue::AbstractVector{<:Real},
    θindices::Union{Vector{Vector{Int}},Vector{Vector{Symbol}}},
    θinitialguess::AbstractVector{<:Real}=θtrue;
    confidence_level::Float64=0.95,
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    coverage_estimate_confidence_level::Float64=0.95,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false)

    function argument_handling!()
        length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))
        length(θinitialguess) == model.core.num_pars || throw(ArgumentError("θinitialguess must have the same length as the number of model parameters"))
        check_prediction_function_exists(model::LikelihoodModel) || throw(ArgumentError("see warning message"))

        if num_points_to_sample isa Int
            num_points_to_sample > 0 || throw(DomainError("num_points_to_sample must be a strictly positive integer"))
        else
            minimum(num_points_to_sample) > 0 || throw(DomainError("num_points_to_sample must contain strictly positive integers"))

            sample_type isa UniformGridSamples || throw(ArgumentError(string("num_points_to_sample must be an integer for ", sample_type, " sample_type")))

            (length(num_points_to_sample) == length(θindices[1]) &&
             diff([extrema(length.(θindices))...])[1] == 0) ||
                throw(ArgumentError("num_points_to_sample must have the same length as each vector of interest parameters in num_points_to_sample"))
        end

        (0.0 < coverage_estimate_confidence_level && coverage_estimate_confidence_level < 1.0) || throw(DomainError("coverage_estimate_confidence_level must be in the open interval (0,1)"))
        get_target_loglikelihood(model, confidence_level, LogLikelihood(), 1)

        N > 0 || throw(DomainError("N must be greater than 0"))

        !ismissing(model.core.errorfunction) || throw(ArgumentError("model must contain an error function for creating prediction realisation confidence intervals. Add one when creating model with initialise_LikelihoodModel or using add_error_function!"))

        if θindices isa Vector{Vector{Symbol}}
            θindices = convertθnames_toindices(model, θnames)
        end

        θindices = θindices[.!isempty.(θindices)]
        sort!.(θindices)
        unique!.(θindices)
        sort!(θindices)
        unique!(θindices)
        1 ≤ first.(θindices)[1] && maximum(last.(θindices)) ≤ model.core.num_pars || throw(DomainError("θindices can only contain parameter indexes between 1 and the number of model parameters"))

        if any(length.(θindices) .== model.core.num_pars)
            i = findfirst(length.(θindices) .== model.core.num_pars)
            new_inds = vcat(i, setdiff(1:length(θindices), i))
            θindices = θindices[new_inds]
        end
        return nothing
    end
    argument_handling!()

    bonferroni_confidence_level = 1.0 - ((1.0-confidence_level)/2.0)

    y_true = model.core.predictfunction(θtrue, model.core.data, t)
    multiple_outputs = ndims(y_true) == 2

    len_θs = length(θindices)

    successes = zeros(Int, len_θs + 1)
    successes_pointwise = [zeros(size(y_true)) for _ in 1:(len_θs+1)]
    iteration_is_included = falses(N)

    data_training = [data_generator(θtrue, training_generator_args) for _ in 1:N]
    data_testing = [data_generator(θtrue, testing_generator_args) for _ in 1:N]

    channel = RemoteChannel(() -> Channel{Bool}(1))
    p = Progress(N; desc="Computing dimensional samples prediction realisation coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for i in 1:N
            new_data = data_training[i]

            m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction,
                model.core.errorfunction,
                new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub,
                model.core.θmagnitudes; biv_row_preallocation_size=len_θs, show_progress=false)

            dimensional_likelihood_samples!(m_new, deepcopy(θindices), num_points_to_sample,
                confidence_level=bonferroni_confidence_level, sample_type=sample_type,
                use_threads=false)

            generate_predictions_dim_samples!(m_new, t, 0.0)

            indiv_cov, union_cov, iteration_is_included[i] = evaluate_coverage_realisations(m_new, data_testing[i], :dimensional, multiple_outputs, len_θs)
            successes[1:len_θs] .+= first.(indiv_cov)
            successes[end] += first(union_cov)

            successes_pointwise[1:len_θs] .+= last.(indiv_cov)
            successes_pointwise[end] += last(union_cov)

            next!(p)
        end
    else
        successes_bool = SharedArray{Bool}(len_θs + 1, N)
        successes_bool .= false
        iteration_is_included_shared = SharedArray{Bool}(N)
        iteration_is_included_shared .= false
        @sync begin
            @async while take!(channel)
                next!(p)
            end

            @async begin
                successes_pointwise_bool = @distributed (vcat) for i in 1:N
                    new_data = data_training[i]

                    m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction,
                        model.core.errorfunction,
                        new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub,
                        model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false)

                    dimensional_likelihood_samples!(m_new, deepcopy(θindices), num_points_to_sample,
                        confidence_level=bonferroni_confidence_level, sample_type=sample_type,
                        use_distributed=false, use_threads=false)

                    generate_predictions_dim_samples!(m_new, t, 0.0, use_distributed=false)

                    indiv_cov, union_cov, iteration_is_included_shared[i] = evaluate_coverage_realisations(m_new, data_testing[i], :dimensional, multiple_outputs, len_θs)
                    successes_bool[1:len_θs, i] .= first.(indiv_cov)
                    successes_bool[end, i] = first(union_cov)

                    put!(channel, true)
                    (vcat(last.(indiv_cov), [last(union_cov)]),)
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
    conf_ints = zeros(len_θs + 1, 2)
    for i in 1:(len_θs+1)
        conf_ints[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N_counted),
            level=coverage_estimate_confidence_level)
        successes_pointwise[i] = successes_pointwise[i] ./ N_counted
    end

    return DataFrame(θname=[[model.core.θnames[[combo...]] for combo in θindices]..., :union], θindices=[θindices..., θindices],
        simultaneous_coverage=coverage, coverage_lb=conf_ints[:, 1], coverage_ub=conf_ints[:, 2],
        pointwise_coverage=successes_pointwise)
end