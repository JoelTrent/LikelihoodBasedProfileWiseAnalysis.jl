"""
    correct_θbounds_nuisance(m::LikelihoodModel,
        θlb_nuisance::AbstractVector{<:Float64},
        θub_nuisance::AbstractVector{<:Float64})

Makes sure that nuisance parameter bounds contain the MLE parameter values - if not, set that part of the nuisance parameter bound to the bounds in `model.core`.
"""
function correct_θbounds_nuisance(m::LikelihoodModel,
    θlb_nuisance::AbstractVector{<:Float64},
    θub_nuisance::AbstractVector{<:Float64})
    if all(θlb_nuisance .< m.core.θmle) && all(θub_nuisance .> m.core.θmle)
        return θlb_nuisance, θub_nuisance
    end

    lb_new = θlb_nuisance .* 1.0
    ub_new = θub_nuisance .* 1.0
    if any(θlb_nuisance .≥ m.core.θmle)
        inds = θlb_nuisance .≥ m.core.θmle
        lb_new[inds] .= m.core.θlb[inds]
    end
    if any(θub_nuisance .≤ m.core.θmle)
        inds = θub_nuisance .≤ m.core.θmle
        ub_new[inds] .= m.core.θub[inds]
    end
    return lb_new, ub_new
end

"""
    setmagnitudes!(model::LikelihoodModel, θmagnitudes::AbstractVector{<:Real})

Updates the magnitudes of each parameter in `model` from `model.core.θmagnitudes` to `θmagnitudes`.
"""
function setmagnitudes!(model::LikelihoodModel, θmagnitudes::AbstractVector{<:Real})

    length(θmagnitudes) == model.core.num_pars || throw(ArgumentError(string("θmagnitudes must have the same length as the number of model parameters (", model.core.num_pars, ")")))

    model.core.θmagnitudes .= θmagnitudes .* 1.0
    return nothing
end

"""
    setbounds!(model::LikelihoodModel; 
        lb::AbstractVector{<:Real}=Float64[], 
        ub::AbstractVector{<:Real}=Float64[])

Updates the parameter bounds in `model` from `model.core.θlb` to `lb` if specified and from `model.core.θub` to `ub` if specified. `lb` and `ub` are keyword arguments.
"""
function setbounds!(model::LikelihoodModel;
                    lb::AbstractVector{<:Real}=Float64[],
                    ub::AbstractVector{<:Real}=Float64[])

    if !isempty(lb)
        length(lb) == model.core.num_pars || throw(ArgumentError(string("lb must have the same length as the number of model parameters (", model.core.num_pars, ")")))
        model.core.θlb .= lb .* 1.0
    end
    if !isempty(ub)
        length(ub) == model.core.num_pars || throw(ArgumentError(string("ub must have the same length as the number of model parameters (", model.core.num_pars, ")")))
        model.core.θub .= ub .* 1.0
    end
    return nothing
end

"""
    convertθnames_toindices(model::LikelihoodModel, 
        θnames_to_convert::Vector{<:Symbol})

Converts a vector of symbols representing parameters in `model` to a vector of each symbol's corresponding index in `model.core.θnames`.
"""
function convertθnames_toindices(model::LikelihoodModel, θnames_to_convert::Vector{<:Symbol})

    indices = zeros(Int, length(θnames_to_convert))

    for (i, name) in enumerate(θnames_to_convert)
        indices[i] = model.core.θname_to_index[name]
    end

    return indices
end

"""
    convertθnames_toindices(model::LikelihoodModel, 
        θnames_to_convert::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}})

Converts a vector of vectors or tuples containing symbols representing parameters in `model` to a vector of vectors containing each symbol's corresponding index in `model.core.θnames`.
"""
function convertθnames_toindices(model::LikelihoodModel, 
                                    θnames_to_convert::Union{Vector{Vector{Symbol}}, Vector{Tuple{Symbol, Symbol}}})

    indices = [zeros(Int, dim) for dim in length.(θnames_to_convert)]

    for (i, names) in enumerate(θnames_to_convert)
        indices[i] .= getindex.(Ref(model.core.θname_to_index), names)
    end

    return indices
end

"""
    ll_correction(model::LikelihoodModel, 
        profile_type::AbstractProfileType, 
        ll::Float64)

If a `profile_type` is `LogLikelihood()`, it corrects `ll` such that an input log-likelihood value (which has value of zero at the MLE) will now have a value of `model.core.maximisedmle` at the MLE. Otherwise, a copy of `ll` is returned, as both ellipse approximation profile types have a log-likelihood value of 0.0 at the MLE.
"""
function ll_correction(model::LikelihoodModel, profile_type::AbstractProfileType, ll::Float64)
    if profile_type isa LogLikelihood
        return ll + model.core.maximisedmle
    end
    return ll * 1.0
end

"""
    get_target_loglikelihood(model::LikelihoodModel, 
        confidence_level::Float64, 
        profile_type::AbstractProfileType, 
        dof::Int)

Returns the target log-likelihood / threshold at a confidence level and degrees of freedom, `dof` (typically the number of interest parameters OR the number of model parameters), required for a particular `profile_type` to be in the confidence set. Uses [`LikelihoodBasedProfileWiseAnalysis.ll_correction`](@ref).
"""
function get_target_loglikelihood(model::LikelihoodModel, 
                                    confidence_level::Float64,
                                    profile_type::AbstractProfileType,
                                    dof::Int)

    (0.0 ≤ confidence_level && confidence_level < 1.0) || throw(DomainError("confidence_level must be in the interval [0,1)"))
    (dof ≤ model.core.num_pars) || throw(DomainError("dof must be less than or equal to the number of model parameters"))

    llstar = -quantile(Chisq(dof), confidence_level) / 2.0

    return ll_correction(model, profile_type, llstar)
end

"""
    get_consistent_tuple(model::LikelihoodModel, 
        confidence_level::Float64, 
        profile_type::AbstractProfileType, 
        dof::Int)

Returns a tuple containing the values needed for log-likelihood evaluation and finding function zeros, including the target log-likelihood, number of model parameters, log-likelihood function to use and `data` tuple for evaluating the log-likelihood function.
"""
function get_consistent_tuple(model::LikelihoodModel, 
                                confidence_level::Float64, 
                                profile_type::AbstractProfileType, 
                                dof::Int)

    targetll = get_target_loglikelihood(model, confidence_level, profile_type, dof)

    if profile_type isa LogLikelihood 
        return (targetll=targetll, num_pars=model.core.num_pars,
                loglikefunction=model.core.loglikefunction, data=model.core.data)
    elseif profile_type isa AbstractEllipseProfileType
        return (targetll=targetll, num_pars=model.core.num_pars, 
                loglikefunction=ellipse_loglike, 
                data=(θmle=model.core.θmle, Hmle=model.ellipse_MLE_approx.Hmle),
                data_analytic=(θmle=model.core.θmle, Γmle=model.ellipse_MLE_approx.Γmle))
    end

    return (missing)
end

"""
    desired_df_subset(df::DataFrame, 
        num_used_rows::Int, 
        confidence_levels::Union{Float64, Vector{<:Float64}},
        dofs::Union{Int, Vector{<:Int}}, 
        sample_types::Vector{<:AbstractSampleType}; 
        sample_dimension::Int=0, 
        regions::Union{Real, Vector{<:Real}}=Float64[],
        for_prediction_generation::Bool=false, 
        for_prediction_plots::Bool=false, 
        include_higher_confidence_levels::Bool=false)

Returns a view of `df` that includes only valid rows ∈ `1:num_used_rows`, and rows that contain all of the values specified within function arguments. For dimensional samples.

# Arguments
- `df`: a DataFrame - `model.dim_samples_df`.
- `num_used_rows`: the number of valid rows in `df` - `model.num_dim_samples`.
- `confidence_levels`: a vector of confidence levels or a `Float64` of a single confidence level. If empty, all confidence levels in `df` are allowed. Otherwise, if `include_higher_confidence_levels == true` and `confidence_levels` is a `Float64`, all confidence levels greater than or equal to `confidence_levels` are allowed. Else, only matching confidence levels in `df` are allowed.
- `dofs`: a vector of integer degrees of freedom or a `Int` of a single degree of freedom. If empty, all degrees of freedom for dimensional profiles are allowed. Otherwise, only matching degrees of freedom in `df` are allowed.
- `sample_types`: a vector of `AbstractSampleType` structs. If empty, all sample types in `df` are allowed. Otherwise, only matching sample types in `df` are allowed.

# Keyword Arguments
- `sample_dimension`: an integer greater than or equal to 0; if non-zero only matching dimensions of interest parameters in `df` are allowed, otherwise all are allowed. Default is `0`.
- `regions`: a vector of `Real` numbers ∈ [0, 1] or a single `Real` number specifying the regions in `df` that are allowed. If empty, all regions are allowed. Otherwise, only matching regions in `df` are allowed. Default is `Float64[]`.
- `for_prediction_generation`: a boolean specifying whether only rows which have not had predictions evaluated are allowed. As predictions do not need to be generated for rows which already have them evaluated. 
- `for_prediction_plots`: a boolean specifying whether only rows which have had predictions evaluated are allowed. As prediction plots can only include rows which have evaluated predictions. 
- `include_higher_confidence_levels`: a boolean specifying whether all confidence levels greater than or equal to `confidence_levels` are allowed. Useful for prediction plots as a dimensional sample can be evaluated at a high confidence level (e.g. 0.95) and then used at a lower confidence level (e.g. 0.9), extracting only the sample points that are in the 0.9 confidence set.
"""
function desired_df_subset(df::DataFrame, 
                            num_used_rows::Int,
                            confidence_levels::Union{Float64, Vector{<:Float64}},
                            dofs::Union{Int, Vector{<:Int}}, 
                            sample_types::Vector{<:AbstractSampleType};
                            sample_dimension::Int=0,
                            regions::Union{Real, Vector{<:Real}}=Float64[],
                            for_prediction_generation::Bool=false,
                            for_prediction_plots::Bool=false,
                            include_higher_confidence_levels::Bool=false)
    
    df_sub = @view(df[1:num_used_rows, :])    
    row_subset = df_sub.num_points .> 0
    if for_prediction_generation
        row_subset .= row_subset .& df_sub.not_evaluated_predictions
    end
    if for_prediction_plots
        row_subset .= row_subset .& .!(df_sub.not_evaluated_predictions)
    end
    if sample_dimension > 0
        row_subset .= row_subset .& (df_sub.dimension .== sample_dimension)
    end

    if !isempty(confidence_levels)
        if include_higher_confidence_levels
            row_subset .= row_subset .& (df_sub.conf_level .>= confidence_levels::Float64)
        else
            row_subset .= row_subset .& (df_sub.conf_level .∈ Ref(confidence_levels))
        end
    end
    if !isempty(dofs)
        row_subset .= row_subset .& (df_sub.dof .∈ Ref(dofs))
    end
    if !isempty(regions)
        row_subset .= row_subset .& (df_sub.region .∈ Ref(regions))
    end
    if !isempty(sample_types)
        row_subset .= row_subset .& (df_sub.sample_type .∈ Ref(sample_types))
    end

    return @view(df_sub[row_subset, :])
end

"""
    desired_df_subset(df::DataFrame, 
        num_used_rows::Int, 
        θs_of_interest::Vector{<:Int}, 
        confidence_levels::Union{Float64, Vector{<:Float64}}, 
        dofs::Union{Int, Vector{<:Int}}, 
        profile_types::Vector{<:AbstractProfileType}; 
        regions::Union{Real, Vector{<:Real}}=Float64[],
        for_points_in_interval::Tuple{Bool,Int,Real}=(false,0,0), 
        for_prediction_generation::Bool=false, 
        for_prediction_plots::Bool=false)

Returns a view of `df` that includes only valid rows ∈ `1:num_used_rows`, and rows that contain all of the values specified within function arguments. For univariate profiles.

# Arguments
- `df`: a DataFrame - `model.uni_profiles_df`.
- `num_used_rows`: the number of valid rows in `df` - `model.num_uni_profiles`.
- `confidence_levels`: a vector of confidence levels or a `Float64` of a single confidence level. If empty, all confidence levels in `df` are allowed. Otherwise, only matching confidence levels in `df` are allowed.
- `dofs`: a vector of integer degrees of freedom or a `Int` of a single degree of freedom. If empty, all degrees of freedom for univariate profiles are allowed. Otherwise, only matching degrees of freedom in `df` are allowed.
- `profile_types`: a vector of `AbstractProfileType` structs. If empty, all profile types in `df` are allowed. Otherwise, only matching profile types in `df` are allowed.

# Keyword Arguments
- `regions`: a vector of `Real` numbers ∈ [0, 1] or a single `Real` number specifying the regions in `df` that are allowed. If empty, all regions are allowed. Otherwise, only matching regions in `df` are allowed. Default is `Float64[]`.
- `for_points_in_interval`: a tuple used for only extracting the rows that need to have points in the confidence interval evaluated by [`get_points_in_intervals!`](@ref). Default is `(false, 0, 0)`.
- `for_prediction_generation`: a boolean specifying whether only rows which have not had predictions evaluated are allowed. As predictions do not need to be generated for rows which already have them evaluated. 
- `for_prediction_plots`: a boolean specifying whether only rows which have had predictions evaluated are allowed. As prediction plots can only include rows which have evaluated predictions. 
"""
function desired_df_subset(df::DataFrame, 
                            num_used_rows::Int,
                            θs_of_interest::Vector{<:Int},
                            confidence_levels::Union{Float64, Vector{<:Float64}},
                            dofs::Union{Int, Vector{<:Int}},
                            profile_types::Vector{<:AbstractProfileType};
                            regions::Union{Float64, Vector{<:Float64}}=Float64[],
                            for_points_in_interval::Tuple{Bool,Int,Real}=(false,0,0),
                            for_prediction_generation::Bool=false,
                            for_prediction_plots::Bool=false)

    df_sub = @view(df[1:num_used_rows, :])    
    row_subset = df_sub.num_points .> 0

    if for_points_in_interval[1]
        num_points_in_interval, additional_width = for_points_in_interval[2:3]
        row_subset .= row_subset .& ((df_sub.num_points .!= (num_points_in_interval+2)) .| 
                        (df_sub.additional_width .!= additional_width))
    end

    if for_prediction_generation
        row_subset .= row_subset .& df_sub.not_evaluated_predictions
    end
    if for_prediction_plots
        row_subset .= row_subset .& .!(df_sub.not_evaluated_predictions)
    end

    if !isempty(θs_of_interest) 
        row_subset .= row_subset .& (df_sub.θindex .∈ Ref(θs_of_interest))
    end
    if !isempty(confidence_levels)
        row_subset .= row_subset .& (df_sub.conf_level .∈ Ref(confidence_levels))
    end
    if !isempty(dofs)
        row_subset .= row_subset .& (df_sub.dof .∈ Ref(dofs))
    end
    if !isempty(regions)
        row_subset .= row_subset .& (df_sub.region .∈ Ref(regions))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .& (df_sub.profile_type .∈ Ref(profile_types))
    end

    return @view(df_sub[row_subset, :])
end


"""
    desired_df_subset(df::DataFrame, 
        num_used_rows::Int, 
        θs_of_interest::Vector{Tuple{Int,Int}}, 
        confidence_levels::Union{Float64, Vector{<:Float64}}, 
        dofs::Union{Int, Vector{<:Int}}, 
        profile_types::Vector{<:AbstractProfileType}, 
        methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[];
        regions::Union{Real, Vector{<:Real}}=Float64[], 
        for_prediction_generation::Bool=false, 
        for_prediction_plots::Bool=false, 
        include_lower_confidence_levels::Bool=false)

Returns a view of `df` that includes only valid rows ∈ `1:num_used_rows`, and rows that contain all of the values specified within function arguments. For bivariate profiles.

# Arguments
- `df`: a DataFrame - `model.biv_profiles_df`.
- `num_used_rows`: the number of valid rows in `df` - `model.num_biv_profiles`.
- `confidence_levels`: a vector of confidence levels or a `Float64` of a single confidence level. If empty, all confidence levels in `df` are allowed. Otherwise, if `include_lower_confidence_levels == true` and `confidence_levels` is a `Float64`, all confidence levels less than or equal to `confidence_levels` are allowed. Else, only matching confidence levels in `df` are allowed.
- `dofs`: a vector of integer degrees of freedom or a `Int` of a integer degree of freedom. If empty, all degrees of freedom for bivariate profiles are allowed. Otherwise, only matching degrees of freedom in `df` are allowed.
- `profile_types`: a vector of `AbstractProfileType` structs. If empty, all profile types in `df` are allowed. Otherwise, only matching profile types in `df` are allowed.
- `methods`: a vector of `AbstractBivariateMethod` structs. If empty, all methods in `df` are allowed. Otherwise, only methods of the same type in `df` are allowed.

# Keyword Arguments
- `regions`: a vector of `Real` numbers ∈ [0, 1] or a single `Real` number specifying the regions in `df` that are allowed. If empty, all regions are allowed. Otherwise, only matching regions in `df` are allowed. Default is `Float64[]`.
- `for_prediction_generation`: a boolean specifying whether only rows which have not had predictions evaluated are allowed. As predictions do not need to be generated for rows which already have them evaluated. 
- `for_prediction_plots`: a boolean specifying whether only rows which have had predictions evaluated are allowed. As prediction plots can only include rows which have evaluated predictions. 
- `remove_combined_method`: a boolean specifiying whether rows with `method` of type [`CombinedBivariateMethod`](@ref) should be removed. Needed by [`combine_bivariate_boundaries!`](@ref) to ensure that combined boundaries are not removed from `df`.
- `include_lower_confidence_levels`: a boolean specifying whether all confidence levels less than or equal to `confidence_levels` are allowed. Useful for prediction plots if a given set of bivariate profiles has few internal points evaluated, meaning some information about predictions may be missing. Bivariate profiles at lower confidence levels are by definition inside the desired confidence profile and may provide additional information on predictions.
"""
function desired_df_subset(df::DataFrame, 
                            num_used_rows::Int,
                            θs_of_interest::Vector{Tuple{Int,Int}},
                            confidence_levels::Union{Float64, Vector{<:Float64}},
                            dofs::Union{Int, Vector{<:Int}},
                            profile_types::Vector{<:AbstractProfileType},
                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[];
                            regions::Union{Real, Vector{<:Real}}=Float64[],
                            for_prediction_generation::Bool=false,
                            for_prediction_plots::Bool=false,
                            remove_combined_method::Bool=false,
                            include_lower_confidence_levels::Bool=false)

    df_sub = @view(df[1:num_used_rows, :])    
    row_subset = df_sub.num_points .> 0
    if for_prediction_generation
        row_subset .= row_subset .& df_sub.not_evaluated_predictions
    end
    if for_prediction_plots
        row_subset .= row_subset .& .!(df_sub.not_evaluated_predictions)
    end

    if !isempty(θs_of_interest) 
        row_subset .= row_subset .& (df_sub.θindices .∈ Ref(θs_of_interest))
    end
    if !isempty(confidence_levels)
        if include_lower_confidence_levels
            row_subset .= row_subset .& (df_sub.conf_level .<= confidence_levels::Float64)
        else
            row_subset .= row_subset .& (df_sub.conf_level .∈ Ref(confidence_levels))
        end
    end
    if !isempty(dofs)
        row_subset .= row_subset .& (df_sub.dof .∈ Ref(dofs))
    end
    if !isempty(regions)
        row_subset .= row_subset .& (df_sub.region .∈ Ref(regions))
    end
    if !isempty(profile_types)
        row_subset .= row_subset .& (df_sub.profile_type .∈ Ref(profile_types))
    end
    if !isempty(methods)
        row_subset .= row_subset .& (typeof.(df_sub.method) .∈ Ref(typeof.(methods)))
    end
    if remove_combined_method
        row_subset .= row_subset .& (typeof.(df_sub.method) .∉ Ref([CombinedBivariateMethod]))
    end

    return @view(df_sub[row_subset, :])
end
