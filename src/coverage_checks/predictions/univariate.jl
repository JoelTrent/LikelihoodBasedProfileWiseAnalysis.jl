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
    
1. Repeatedly drawing new observed data using `data_generator` for fixed true parameter values, θtrue, and fixed true prediction value. 
2. Fitting the model and univariate confidence intervals. 
3. Sampling points along the profile within the confidence intervals.
4. Evaluating predictions from the points in the profile and finding the prediction extrema.
5. Checking whether the prediction extrema contain the true prediction value(s), in a pointwise and simultaneous fashion. The estimated simultaneous coverage is returned with a default 95% confidence interval within a DataFrame. 

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. Outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`.
- `generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `generator_args`. 
- `t`: a vector of time points to compute predictions and evaluate coverage at.
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `N`: a positive number of coverage simulations.
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θs`: a vector of parameters to profile, as a vector of model parameter indexes.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `num_points_in_interval`: an integer number of points to optionally evaluate within the confidence interval for each interest parameter using [`get_points_in_intervals!`](@ref). Points are linearly spaced in the interval. Useful for plots that visualise the confidence interval or for predictions from univariate profiles. Default is `0`. 
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to evaluate the confidence interval coverage at. Default is `0.95` (95%).
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is `0.95` (95%).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations (false) or across the individual confidence interval calculations within each iteration (true). Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of propagating parameter confidence intervals into prediction space. The simulation uses [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) to parallelise the workload.

The uncertainty in estimates of the coverage under the simulated model will decrease as the number of simulations, `N`, is increased. Confidence intervals for the coverage estimate are provided to quantify this uncertainty. The confidence interval for the estimated coverage is a Clopper-Pearson interval on a binomial test generated using [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/).

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
    coverage_estimate_confidence_level::Float64=0.95,
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
        
        (sort(θs); unique!(θs))
        1 ≤ θs[1] && θs[end] ≤ model.core.num_pars || throw(DomainError("θs can only contain parameter indexes between 1 and the number of model parameters"))
        return nothing
    end
    
    argument_handling!()
    y_true = model.core.predictfunction(θtrue, model.core.data, t)
    multiple_outputs = ndims(y_true) == 2

    len_θs = length(θs)
    θi_to_θs = Dict{Int,Int}(θi => θs for (θs, θi) in enumerate(θs))

    successes = zeros(Int, len_θs+1)
    successes_pointwise = [zeros(size(y_true)) for _ in 1:(len_θs+1)]

    data = [data_generator(θtrue, generator_args) for _ in 1:N]

    channel = RemoteChannel(() -> Channel{Bool}(1))
    p = Progress(N; desc="Computing univariate prediction coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for i in 1:N
            new_data = data[i]

            m_new = initialise_LikelihoodModel(model.core.loglikefunction, model.core.predictfunction,
                new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, 
                model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false)

            univariate_confidenceintervals!(m_new, θs; 
                num_points_in_interval=num_points_in_interval, confidence_level=confidence_level, 
                profile_type=profile_type, show_progress=false, use_threads=false)

            generate_predictions_univariate!(m_new, t, 0.0, show_progress=false)

            indiv_cov, union_cov = evaluate_coverage(m_new, y_true, :univariate, multiple_outputs)            
            successes[1:len_θs] .+= first.(indiv_cov)
            successes[end] += first(union_cov)

            successes_pointwise[1:len_θs] .+= last.(indiv_cov)
            successes_pointwise[end] += last(union_cov)

            next!(p)
        end
    else
        successes_bool = SharedArray{Bool}(len_θs+1, N)
        successes_bool .= false
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

                    univariate_confidenceintervals!(m_new, θs; 
                        num_points_in_interval=num_points_in_interval, confidence_level=confidence_level, 
                        profile_type=profile_type, show_progress=false, use_distributed=false, use_threads=false)

                    generate_predictions_univariate!(m_new, t, 0.0, show_progress=false, use_distributed=false)

                    indiv_cov, union_cov = evaluate_coverage(m_new, y_true, :univariate, multiple_outputs)
                    successes_bool[1:len_θs, i] .= first.(indiv_cov)
                    successes_bool[end, i] = first(union_cov)

                    put!(channel, true)
                    (vcat(last.(indiv_cov), [last(union_cov)]),)
                end
                put!(channel, false)
                for pointwise_bool in successes_pointwise_bool
                    successes_pointwise .+= first(pointwise_bool)
                end
            end
        end
        successes .= sum(successes_bool, dims=2)
    end

    coverage = successes ./ N
    conf_ints = zeros(len_θs+1, 2)
    for i in 1:(len_θs+1)
        conf_ints[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N), 
            level=coverage_estimate_confidence_level) 
        successes_pointwise[i] = successes_pointwise[i] ./ N
    end

    return DataFrame(θname=[model.core.θnames[θs]..., :union], θindex=[θs..., θs], 
        simultaneous_coverage=coverage, coverage_lb=conf_ints[:,1], coverage_ub=conf_ints[:,2],
        pointwise_coverage=successes_pointwise)
end