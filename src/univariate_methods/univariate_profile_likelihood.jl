"""
    get_uni_confidence_interval_points(model::LikelihoodModel, uni_row_number::Int)

Returns the interval points [`PointsAndLogLikelihood`](@ref) struct corresponding to the profile in row `uni_row_number` of `model.uni_profiles_df`.
"""
function get_uni_confidence_interval_points(model::LikelihoodModel, uni_row_number::Int)
    return model.uni_profiles_dict[uni_row_number].interval_points
end

"""
    get_uni_confidence_interval(model::LikelihoodModel, uni_row_number::Int)

Returns the confidence interval corresponding to the profile in row `uni_row_number` of `model.uni_profiles_df` as a vector of length two. If an entry has value `NaN`, that side of the confidence interval is outside the corresponding bound on the interest parameter.
"""
function get_uni_confidence_interval(model::LikelihoodModel, uni_row_number::Int)
    return model.uni_profiles_dict[uni_row_number].confidence_interval
end

"""
    get_interval_brackets(model::LikelihoodModel, 
        θi::Int, 
        confidence_level::Float64, 
        profile_type::AbstractProfileType)

Returns updated interval brackets (`Float64` vectors of length two) if smaller or larger confidence level profiles exist for θi, such that the region to bracket over for the left and right sides of the confidence internval is smallest. Otherwise, returns empty brackets (`Float64[]`).
"""
function get_interval_brackets(model::LikelihoodModel, 
                                θi::Int,
                                confidence_level::Float64,
                                profile_type::AbstractProfileType)

    prof_keys = collect(keys(model.uni_profile_row_exists[(θi, profile_type)]))
    len_keys = length(prof_keys)
    if len_keys > 1
        sort!(prof_keys)
        conf_ind = findfirst(isequal(confidence_level), prof_keys)

        bracket_l, bracket_r = [0.0, 0.0], [0.0, 0.0]

        if conf_ind>1
            bracket_l[2], bracket_r[1] = get_uni_confidence_interval(model, model.uni_profile_row_exists[(θi, profile_type)][prof_keys[conf_ind-1]])
        else
            bracket_l[2], bracket_r[1] = model.core.θmle[θi], model.core.θmle[θi]
        end
        
        if conf_ind<len_keys
            bracket_l[1], bracket_r[2] = get_uni_confidence_interval(model, model.uni_profile_row_exists[(θi, profile_type)][prof_keys[conf_ind+1]])
        else
            bracket_l[1], bracket_r[2] = model.core.θlb[θi], model.core.θub[θi]
        end
    else
        bracket_l, bracket_r = Float64[], Float64[]
    end
    return bracket_l, bracket_r
end

"""
    add_uni_profiles_rows!(model::LikelihoodModel, num_rows_to_add::Int)

Adds `num_rows_to_add` free rows to `model.uni_profiles_df` by vertically concatenating the existing DataFrame and free rows using [`PlaceholderLikelihood.init_uni_profiles_df`](@ref).
"""
function add_uni_profiles_rows!(model::LikelihoodModel, 
                                num_rows_to_add::Int)
    new_rows = init_uni_profiles_df(num_rows_to_add, 
                                    existing_largest_row=nrow(model.uni_profiles_df))

    model.uni_profiles_df = vcat(model.uni_profiles_df, new_rows)
    return nothing
end

"""
    set_uni_profiles_row!(model::LikelihoodModel, 
        row_ind::Int, 
        θi::Int,
        not_evaluated_internal_points::Bool, 
        not_evaluated_predictions::Bool,
        confidence_level::Float64, 
        profile_type::AbstractProfileType, 
        num_points::Int, 
        additional_width::Real)

Sets the relevant fields of row `row_ind` in `model.uni_profiles_df` after a profile has been evaluated.
"""
function set_uni_profiles_row!(model::LikelihoodModel,
                                    row_ind::Int,
                                    θi::Int,
                                    not_evaluated_internal_points::Bool,
                                    not_evaluated_predictions::Bool,
                                    confidence_level::Float64,
                                    profile_type::AbstractProfileType,
                                    num_points::Int,
                                    additional_width::Real)
    model.uni_profiles_df[row_ind, 2:end] .= θi*1, 
                                            not_evaluated_internal_points,
                                            not_evaluated_predictions,
                                            confidence_level,
                                            profile_type,
                                            num_points,
                                            additional_width
    return nothing
end

"""
    get_univariate_opt_func(profile_type::AbstractProfileType=LogLikelihood())

Returns the correct univariate optimisation function used to for find the optimal values of nuisance parameters for a given interest parameter value for the `profile_type` log-likelihood function. The optimisation function returns the value of the `profile_type` log-likelihood function as well as finding the optimal nuisance parameters and saving these in one of it's inputs.
    
Will be [`PlaceholderLikelihood.univariateΨ`](@ref) for the [`LogLikelihood()`](@ref) and [`EllipseApprox()`](@ref) profiles types and [`PlaceholderLikelihood.univariateΨ_ellipse_unbounded`](@ref) for the [`EllipseApproxAnalytical`](@ref) profile type.
"""
function get_univariate_opt_func(profile_type::AbstractProfileType=LogLikelihood())

    if profile_type isa LogLikelihood || profile_type isa EllipseApprox
        return univariateΨ
    elseif profile_type isa EllipseApproxAnalytical
        return univariateΨ_ellipse_unbounded #univariateΨ_ellipse_analytical
    end

    return (missing)
end

"""
    univariate_confidenceinterval!(p::Progress,
        univariate_optimiser::Function, 
        model::LikelihoodModel, 
        consistent::NamedTuple, 
        θi::Int, 
        confidence_level::Float64,
        profile_type::AbstractProfileType,
        mle_targetll::Float64,
        use_existing_profiles::Bool,
        num_points_in_interval::Int,
        additional_width::Real)

Returns a [`UnivariateConfidenceStruct`](@ref) containing the likelihood-based confidence interval for interest parameter `θi` at `confidence_level`, and any additional points within the interval if `num_points_in_interval > 0` as well as outside the interval if `num_points_in_interval > 0` and `additional_width > 0`. Log-likelihood values, standardised to 0.0 at the MLE point, for all points found in the interval are also stored in the [`UnivariateConfidenceStruct`](@ref).

If `use_existing_profiles=true` then the brackets used to find each side of the confidence interval (between each side of the bounds for `θi` and the MLE point), will be updated and made smaller if confidence profiles for `θi` already exist at lower and higher confidence levels. 

Called by [`univariate_confidenceintervals!`](@ref).
"""
function univariate_confidenceinterval(univariate_optimiser::Function, 
                                        model::LikelihoodModel, 
                                        consistent::NamedTuple, 
                                        θi::Int, 
                                        confidence_level::Float64,
                                        profile_type::AbstractProfileType,
                                        mle_targetll::Float64,
                                        use_existing_profiles::Bool,
                                        num_points_in_interval::Int,
                                        additional_width::Real,
                                        channel::RemoteChannel)

    interval = zeros(2)
    ll = zeros(2)
    interval_points = zeros(model.core.num_pars, 2)
    newLb, newUb, initGuess, θranges, λranges = init_univariate_parameters(model, θi)

    p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, λranges=λranges, consistent=consistent, 
        λ_opt=zeros(model.core.num_pars-1))

    if use_existing_profiles
        bracket_l, bracket_r = get_interval_brackets(model, θi, confidence_level,
                                                        profile_type)
    else
        bracket_l, bracket_r = Float64[], Float64[]
    end

    if isempty(bracket_l)
        bracket_l = [model.core.θlb[θi], model.core.θmle[θi]]
    end
    if isempty(bracket_r)
        bracket_r = [model.core.θmle[θi], model.core.θub[θi]]
    end

    if univariate_optimiser == univariateΨ_ellipse_unbounded

        interval .= analytic_ellipse_loglike_1D_soln(θi, consistent.data_analytic, mle_targetll)
        
        if interval[1] >= bracket_l[1]
            interval_points[θi,1] = interval[1]
            univariate_optimiser(interval[1], p)
            variablemapping1d!(@view(interval_points[:, 1]), p.λ_opt, θranges, λranges)
            ll[1] = mle_targetll
        else
            interval[1]=NaN 
        end

        if interval[2] <= bracket_r[2]
            interval_points[θi,2] = interval[2]
            univariate_optimiser(interval[2], p)
            variablemapping1d!(@view(interval_points[:, 2]), p.λ_opt, θranges, λranges)
            ll[2] = mle_targetll
        else 
            interval[2]=NaN
        end

    else
        # by definition, g(θmle[i],p) == abs(llstar) > 0, so only have to check one side of interval to make sure it brackets a zero
        g = univariate_optimiser(bracket_l[1], p)
        if g < 0.0
            # make bracket a tiny bit smaller
            if isinf(g); bracket_l[1] = bracket_l[1] + 1e-8 * diff(bracket_l)[1] end

            interval[1] = find_zero(univariate_optimiser, bracket_l, Roots.Brent(), p=p) 
            interval_points[θi,1] = interval[1]
            univariate_optimiser(interval[1], p)
            variablemapping1d!(@view(interval_points[:,1]), p.λ_opt, θranges, λranges)
            ll[1] = mle_targetll
        else
            interval[1] = NaN
        end

        g = univariate_optimiser(bracket_r[2], p)
        if g < 0.0
            # make bracket a tiny bit smaller
            if isinf(g); bracket_r[2] = bracket_r[2] - 1e-8 * diff(bracket_r)[1] end

            interval[2] = find_zero(univariate_optimiser, bracket_r, Roots.Brent(), p=p)
            interval_points[θi,2] = interval[2]
            univariate_optimiser(interval[2], p)
            variablemapping1d!(@view(interval_points[:,2]), p.λ_opt, θranges, λranges)
            ll[2] = mle_targetll
        else
            interval[2] = NaN
        end
    end

    if isnan(interval[1])
        interval_points[θi,1] = bracket_l[1] * 1.0
        ll[1] = univariate_optimiser(bracket_l[1], p) + mle_targetll
        variablemapping1d!(@view(interval_points[:, 1]), p.λ_opt, θranges, λranges)
    end
    if isnan(interval[2])         
        interval_points[θi,2] = bracket_r[2] * 1.0
        ll[2] = univariate_optimiser(bracket_r[2], p) + mle_targetll
        variablemapping1d!(@view(interval_points[:, 2]), p.λ_opt, θranges, λranges)
    end
  
    points = PointsAndLogLikelihood(interval_points, ll, [1,2])

    if num_points_in_interval > 0
        points = get_points_in_interval_single_row(univariate_optimiser, model,
                                                    num_points_in_interval, θi,
                                                    profile_type, points, additional_width)
    end

    put!(channel, true)
    return UnivariateConfidenceStruct(interval, points)
end

"""
    univariate_confidenceintervals!(model::LikelihoodModel, 
        θs_to_profile::Vector{<:Int64}=collect(1:model.core.num_pars); 
        <keyword arguments>)

Computes likelihood-based confidence interval profiles for the provided `θs_to_profile` interest parameters, where `θs_to_profile` is a vector of `Int` corresponding to the parameter indexes in `model.core.θnames`. Saves these profiles by modifying `model` in place.

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `θs_to_profile`: vector of parameters to profile, as a vector of model parameter indexes. Default is `collect(1:model.core.num_pars)`, or all parameters.

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to evaluate the confidence interval. Default is 0.95 (95%).
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `use_existing_profiles`: boolean variable specifying whether to use existing profiles of a parameter `θi` to decrease the width of the bracket used to search for the desired confidence interval using [`PlaceholderLikelihood.get_interval_brackets`](@ref). Default is `false`.
- `θs_is_unique`: boolean variable specifying whether all parameter indexes in `θs_to_profile` are ordered by parameter index (ascending) and unique. Default is `false`.
- `num_points_in_interval`: an integer number of points to optionally evaluate within the confidence interval for each interest parameter using [`get_points_in_interval!`](@ref). Points are linearly spaced in the interval and have their optimised log-likelihood value recorded. Useful for plots that visualise the confidence interval or for predictions from univariate profiles. Default is 0. 
- `additional_width`: a `Real` number greater than or equal to zero. Specifies the additional width to optionally evaluate outside the confidence interval's width if `num_points_in_interval` is greater than 0 using [`get_points_in_interval!`](@ref). Half of this additional width will be placed on either side of the confidence interval. If the additional width goes outside a bound on the parameter, only up to the bound will be considered. The spacing of points in the additional width will try to match the spacing of points evaluated inside the interval. Useful for plots that visualise the confidence interval as it shows the trend of the log-likelihood profile outside the interval range. Default is 0.0.
- `existing_profiles`: `Symbol ∈ [:ignore, :overwrite]` specifying what to do if profiles already exist for a given interest parameter, `confidence_level` and `profile_type`. See below for each symbol's meanings. Default is `:merge`.
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of `θs_to_profile` completed and estimated time of completion. Default is `model.show_progress`.

!!! note "existing_profiles meanings"

    - :ignore means profiles that already exist will not be recomputed. 
    - :overwrite means profiles that already exist will be overwritten.

# Details

Using [`univariate_confidenceinterval`](@ref) this function uses a bracketing method for each interest parameter in `θs_to_profile` (depending on the setting for `existing_profiles` if these profiles already exist). Updates `model.uni_profiles_df` for each successful profile and saves their results as a [`UnivariateConfidenceStruct`](@ref) in `model.uni_profiles_dict`, where the keys for the dictionary is the row number in `model.uni_profiles_df` of the corresponding profile.
"""
function univariate_confidenceintervals!(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Int64}=collect(1:model.core.num_pars); 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        use_existing_profiles::Bool=false,
                                        θs_is_unique::Bool=false,
                                        num_points_in_interval::Int=0,
                                        additional_width::Real=0.0,
                                        existing_profiles::Symbol=:ignore,
                                        show_progress::Bool=model.show_progress,
                                        use_distributed::Bool=true)

    num_points_in_interval >= 0 || throw(DomainError("num_points_in_interval must be a strictly positive integer"))
    additional_width >= 0 || throw(DomainError("additional_width must be greater than or equal to zero"))
    existing_profiles ∈ [:ignore, :overwrite] || throw(ArgumentError("existing_profiles can only take value :ignore or :overwrite"))

    additional_width = num_points_in_interval > 0 ? additional_width : 0.0

    if profile_type isa AbstractEllipseProfileType
        check_ellipse_approx_exists!(model)
    end

    univariate_optimiser = get_univariate_opt_func(profile_type)
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 1)
    mle_targetll = get_target_loglikelihood(model, confidence_level, EllipseApproxAnalytical(), 1)

    θs_is_unique || (sort(θs_to_profile); unique!(θs_to_profile))

    1 ≤ θs_to_profile[1] && θs_to_profile[end] ≤ model.core.num_pars || throw(DomainError("θs_to_profile can only contain parameter indexes between 1 and the number of model parameters"))

    init_uni_profile_row_exists!(model, θs_to_profile, profile_type)

    θs_to_keep = trues(length(θs_to_profile))
    θs_to_overwrite = falses(length(θs_to_profile))
    num_to_overwrite = 0
    # check if profile has already been evaluated
    # in this case we only have :ignore and :overwrite
    for (i, θi) in enumerate(θs_to_profile)
        if model.uni_profile_row_exists[(θi, profile_type)][confidence_level] != 0
            θs_to_keep[i] = false
            θs_to_overwrite[i] = true
            num_to_overwrite += 1
        end
    end
    if existing_profiles == :ignore
        θs_to_profile = θs_to_profile[θs_to_keep]
        θs_to_overwrite = θs_to_overwrite[θs_to_keep]
        num_to_overwrite = 0
    end

    len_θs_to_profile = length(θs_to_profile)
    len_θs_to_profile > 0 || return nothing

    num_rows_required = ((len_θs_to_profile-num_to_overwrite) + model.num_uni_profiles) - nrow(model.uni_profiles_df)

    if num_rows_required > 0
        add_uni_profiles_rows!(model, num_rows_required)
    end

    not_evaluated_internal_points = num_points_in_interval > 0 ? false : true

    channel = RemoteChannel(() -> Channel{Bool}(1))
    p = Progress(length(θs_to_profile); desc="Computing univariate profiles: ",
                dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    @sync begin
        # this task prints the progress bar
        @async while take!(channel)
            next!(p)
        end
        if use_distributed

            # this task does the computation
            @async begin
                profiles_to_add = @distributed (vcat) for θi in θs_to_profile
                    [(θi, univariate_confidenceinterval(univariate_optimiser, model,
                                                                consistent, θi, 
                                                                confidence_level, profile_type,
                                                                mle_targetll,
                                                                use_existing_profiles,
                                                                num_points_in_interval,
                                                                additional_width, channel))]
                end
                put!(channel, false)

                for (i, (θi, interval_struct)) in enumerate(profiles_to_add)
                    if θs_to_overwrite[i]
                        row_ind = model.uni_profile_row_exists[(θi, profile_type)][confidence_level]
                    else
                        model.num_uni_profiles += 1
                        row_ind = model.num_uni_profiles * 1
                        model.uni_profile_row_exists[(θi, profile_type)][confidence_level] = row_ind
                    end

                    model.uni_profiles_dict[row_ind] = interval_struct

                    set_uni_profiles_row!(model, row_ind, θi, not_evaluated_internal_points, true, confidence_level, 
                                            profile_type, num_points_in_interval+2, additional_width)
                end        
            end

        else

            # this task does the computation
            @async begin
                for (i, θi) in enumerate(θs_to_profile)
                    interval_struct = univariate_confidenceinterval(univariate_optimiser, model,
                        consistent, θi,
                        confidence_level, profile_type,
                        mle_targetll,
                        use_existing_profiles,
                        num_points_in_interval,
                        additional_width, channel)

                    if θs_to_overwrite[i]
                        row_ind = model.uni_profile_row_exists[(θi, profile_type)][confidence_level]
                    else
                        model.num_uni_profiles += 1
                        row_ind = model.num_uni_profiles * 1
                        model.uni_profile_row_exists[(θi, profile_type)][confidence_level] = row_ind
                    end

                    model.uni_profiles_dict[row_ind] = interval_struct

                    set_uni_profiles_row!(model, row_ind, θi, not_evaluated_internal_points, true, confidence_level,
                        profile_type, num_points_in_interval + 2, additional_width)
                end
                put!(channel, false)
            end
        end
    end
    
    return nothing
end

"""
    univariate_confidenceintervals!(model::LikelihoodModel, 
        θs_to_profile::Vector{<:Symbol}; 
        <keyword arguments>)

Profiles only the provided `θs_to_profile` interest parameters, where `θs_to_profile` is a vector of `Symbol` corresponding to the parameter symbols in `model.core.θnames`.
"""
function univariate_confidenceintervals!(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Symbol}; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        use_existing_profiles::Bool=false,
                                        θs_is_unique::Bool=false,
                                        num_points_in_interval::Int=0,
                                        additional_width::Real=0.0,
                                        existing_profiles::Symbol=:ignore,
                                        show_progress::Bool=model.show_progress,
                                        use_distributed::Bool=true)

    indices_to_profile = convertθnames_toindices(model, θs_to_profile)
    univariate_confidenceintervals!(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type,
                                use_existing_profiles=use_existing_profiles,
                                θs_is_unique=θs_is_unique,
                                num_points_in_interval=num_points_in_interval,
                                additional_width=additional_width,
                                existing_profiles=existing_profiles,
                                show_progress=show_progress,
                                use_distributed=use_distributed)
    return nothing
end

"""
    univariate_confidenceintervals!(model::LikelihoodModel, 
        profile_m_random_combinations::Int; 
        <keyword arguments>)

Profiles m random interest parameters (sampling without replacement), where `0 < m ≤ model.core.num_pars`.

!!! warning "θs_is_unique keyword argument"

    `θs_is_unique` is not a valid keyword argument for this function method as it internally produces the parameter combinations which are guaranteed to be unique.
"""
function univariate_confidenceintervals!(model::LikelihoodModel, 
                                        profile_m_random_parameters::Int; 
                                        confidence_level::Float64=0.95, 
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        use_existing_profiles::Bool=false,
                                        num_points_in_interval::Int=0,
                                        additional_width::Real=0.0,
                                        existing_profiles::Symbol=:ignore,
                                        show_progress::Bool=model.show_progress,
                                        use_distributed::Bool=true)

    profile_m_random_parameters = max(0, min(profile_m_random_parameters, model.core.num_pars))
    profile_m_random_parameters > 0 || throw(DomainError("profile_m_random_parameters must be a strictly positive integer"))

    indices_to_profile = sample(1:model.core.num_pars, profile_m_random_parameters, replace=false)

    univariate_confidenceintervals!(model, indices_to_profile, confidence_level=confidence_level,
                                profile_type=profile_type,
                                use_existing_profiles=use_existing_profiles,
                                θs_is_unique=true,
                                num_points_in_interval=num_points_in_interval,
                                additional_width=additional_width,
                                existing_profiles=existing_profiles,
                                show_progress=show_progress,
                                use_distributed=use_distributed)
    return nothing
end