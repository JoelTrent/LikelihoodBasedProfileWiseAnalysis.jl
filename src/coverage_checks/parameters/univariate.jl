"""
    check_univariate_parameter_coverage(data_generator::Function, 
        generator_args::Union{Tuple, NamedTuple},
        model::LikelihoodModel, 
        N::Int, 
        θtrue::AbstractVector{<:Real}, 
        θs::AbstractVector{<:Int64},
        θinitialguess::AbstractVector{<:Real}=θtrue; 
        confidence_level::Float64=0.95, 
        profile_type::AbstractProfileType=LogLikelihood(), 
        θs_is_unique::Bool=false,
        coverage_estimate_confidence_level::Float64=0.95,
        show_progress::Bool=model.show_progress,
        distributed_over_parameters::Bool=false)

Performs a simulation to estimate the coverage of univariate confidence intervals for parameters in `θs` given a model by: repeatedly drawing new observed data using `data_generator` for fixed true parameter values, θtrue, fitting the model and univariate confidence intervals, and checking whether the confidence interval for the parameters of interest contain the true parameter value in `θtrue`. The estimated coverage is returned with a default 95% confidence interval within a DataFrame. 

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. Outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`.
- `generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `generator_args`. 
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `N`: a positive number of coverage simulations.
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θs`: a vector of parameters to profile, as a vector of model parameter indexes.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to evaluate the confidence interval coverage at. Default is 0.95 (95%).
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `θs_is_unique`: boolean variable specifying whether all parameter indexes in `θs` are ordered by parameter index (ascending) and unique. Default is `false`.
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is 0.95 (95%).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations or across the individual confidence interval calculations within each iteration. Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of parameter confidence intervals. For a 95% confidence interval of a interest parameter `θi` it is expected that under repeated experiments from an underlying true model (data generation) which are used to construct a confidence interval for `θi` using the method used in [`univariate_confidenceintervals!`](@ref), 95% of the intervals constructed would contain the true value for `θi`. In our simulation where the values of our true parameters, `θtrue`, are known this is equivalent to whether the confidence interval for `θi` contains the value `θtrue[θi]`. 

The uncertainty in estimates of the coverage under the simulated model will decrease as the number of simulations, `N`, is increased. Confidence intervals for the coverage estimate are provided to quantify this uncertainty. The confidence interval for the estimated coverage is a Clopper-Pearson interval on a binomial test generated using [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/).

!!! note "Recommended setting for distributed_over_parameters"
    - If the number of processes available to use is significantly greater than the number of model parameters or only a few model parameters are being checked for coverage, `false` is recommended.   
    - If system memory or model size in system memory is a concern, or the number of processes available is similar or less than the number of model parameters being checked, `true` will likely be more appropriate. 
    - When set to `false`, a separate [`LikelihoodModel`](@ref) struct will be used by each process, as opposed to only one when set to `true`, which could be an issue for larger models. 

!!! danger "Not intended for use on bimodal univariate profile likelihoods"
    The current implementation only considers two extremes of the log-likelihood and whether the truth is between these two points. If the profile likelihood function is bimodal, it's possible the method has only found one set of correct confidence intervals (estimated coverage will be correct, but less than expected) or found one extrema on distinct sets (estimated coverage may be incorrect and will either be larger than expected or much lower than expected). 
"""
function check_univariate_parameter_coverage(data_generator::Function, 
    generator_args::Union{Tuple, NamedTuple},
    model::LikelihoodModel, 
    N::Int, 
    θtrue::AbstractVector{<:Real}, 
    θs::AbstractVector{<:Int64},
    θinitialguess::AbstractVector{<:Real}=θtrue; 
    confidence_level::Float64=0.95, 
    profile_type::AbstractProfileType=LogLikelihood(), 
    θs_is_unique::Bool=false,
    coverage_estimate_confidence_level::Float64=0.95,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false)

    length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))     

    N > 0 || throw(DomainError("N must be greater than 0"))

    θs_is_unique || (sort(θs); unique!(θs))
    1 ≤ θs[1] && θs[end] ≤ model.core.num_pars || throw(DomainError("θs can only contain parameter indexes between 1 and the number of model parameters"))

    len_θs = length(θs)
    θi_to_θs = Dict{Int,Int}(θi => θs for (θs, θi) in enumerate(θs))

    successes = zeros(Int, len_θs)

    channel = RemoteChannel(() -> Channel{Bool}(1))
    p = Progress(N; desc="Computing univariate parameter coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for _ in 1:N
            new_data = data_generator(θtrue, generator_args)

            m_new = initialiseLikelihoodModel(model.core.loglikefunction, new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false)

            univariate_confidenceintervals!(m_new, θs; 
                confidence_level=confidence_level, profile_type=profile_type, θs_is_unique=true, show_progress=false)

            for row_ind in 1:m_new.num_uni_profiles
                θi = m_new.uni_profiles_df[row_ind, :θindex]

                # check if interval contains θtrue[θi]
                interval_struct = get_uni_confidence_interval_points(m_new, row_ind)
                interval = @inbounds interval_struct.points[θi, 1:2]

                θtrue_i = θtrue[θi]
                if interval[1] ≤ θtrue_i && θtrue_i ≤ interval[2]
                    successes[θi_to_θs[θi]] += 1
                end
            end
            next!(p)
        end

    else
        successes_bool = SharedArray{Bool}(len_θs, N)
        successes_bool .= false
        @sync begin
            # this task prints the progress bar
            @async while take!(channel)
                next!(p)
            end

            # this task does the computation
            @async begin
                @distributed (+) for i in 1:N
                    new_data = data_generator(θtrue, generator_args)

                    m_new = initialiseLikelihoodModel(model.core.loglikefunction, new_data, 
                        model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, 
                        model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false)

                    univariate_confidenceintervals!(m_new, θs; 
                        confidence_level=confidence_level, profile_type=profile_type, θs_is_unique=true, show_progress=false, use_distributed=false)

                    for row_ind in 1:m_new.num_uni_profiles
                        θi = m_new.uni_profiles_df[row_ind, :θindex]
                        
                        # check if interval contains θtrue[θi]
                        interval_struct = get_uni_confidence_interval_points(m_new, row_ind)
                        interval = @inbounds interval_struct.points[θi, 1:2]
                        
                        θtrue_i = θtrue[θi]
                        if interval[1] ≤ θtrue_i && θtrue_i ≤ interval[2]
                            successes_bool[θi_to_θs[θi], i] = true
                        end
                    end
                    put!(channel, true); i^2
                end
                put!(channel, false)
            end
        end
        successes .= sum(successes_bool, dims=2)
    end

    coverage = successes ./ N
    conf_ints = zeros(len_θs, 2)
    for i in 1:len_θs; conf_ints[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N), level=coverage_estimate_confidence_level) end
    
    return DataFrame(θname=model.core.θnames[θs], θindex=θs, coverage=coverage, coverage_lb=conf_ints[:,1], coverage_ub=conf_ints[:,2])
end