# some function to simulate data from at fixed time points

# Parameter truth as input

# Number of iterations to run OR method for coverage convergence  (e.g. value hasn't changed by more than x% in last y iterations.)

# Probably easiest to have user first initialise a model using initialiseLikelihoodModel - and then we can use the fields of this model to re-init all the next ones ?
# OR have all the arguments for initialiseLikelihoodModel as arguments for this function.

"""
    check_univariate_parameter_coverage(data_generator::Function, 
        model::LikelihoodModel, 
        θtrue::AbstractVector{<:Real}, 
        N::Int, 
        θs::AbstractVector{<:Int64},
        θinitialguess::AbstractVector{<:Real}=θtrue; 
        confidence_level::Float64=0.95, 
        profile_type::AbstractProfileType=LogLikelihood(), 
        θs_is_unique::Bool=false)

"""
function check_univariate_parameter_coverage(data_generator::Function, 
    generator_args::Union{Tuple, NamedTuple},
    model::LikelihoodModel, 
    θtrue::AbstractVector{<:Real}, 
    N::Int, 
    θs::AbstractVector{<:Int64},
    θinitialguess::AbstractVector{<:Real}=θtrue; 
    confidence_level::Float64=0.95, 
    profile_type::AbstractProfileType=LogLikelihood(), 
    θs_is_unique::Bool=false,
    coverage_estimate_confidence_level::Float64=0.95)

    length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))     

    N > 0 || throw(DomainError("N must be greater than 0"))

    θs_is_unique || (sort(θs); unique!(θs))
    1 ≤ θs[1] && θs[end] ≤ model.core.num_pars || throw(DomainError("θs can only contain parameter indexes between 1 and the number of model parameters"))

    len_θs = length(θs)
    θi_to_θs = Dict{Int,Int}(θi => θs for (θs, θi) in enumerate(θs))

    successes, total = zeros(Int, len_θs), 0
    not_converged=true

    while not_converged
        new_data = data_generator(θtrue, generator_args)

        m_new = initialiseLikelihoodModel(model.core.loglikefunction, new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false)

        univariate_confidenceintervals!(m_new, θs; 
            confidence_level=confidence_level, profile_type=profile_type, θs_is_unique=true)

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
        total += 1
        if rem(total, 10) == 0; println(total) end
        if total == N; not_converged=false end
    end

    coverage = successes ./ N
    conf_ints = zeros(len_θs, 2)
    for i in 1:len_θs; conf_ints[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N), level=coverage_estimate_confidence_level) end
    
    return DataFrame(θname=model.core.θnames[θs], θindex=θs, coverage=coverage, coverage_lb=conf_ints[:,1], coverage_ub=conf_ints[:,2])
end