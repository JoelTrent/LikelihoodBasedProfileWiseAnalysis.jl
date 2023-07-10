"""
    update_uni_dict_internal!(model::LikelihoodModel,
        uni_row_number::Int,
        points::PointsAndLogLikelihood)

Updates the `interval_points` field of a [`UnivariateConfidenceStruct`](@ref), for the profile related to `uni_row_number` stored at `model.uni_profiles_dict[uni_row_number]`, with the interval points stored in `points`.
"""
function update_uni_dict_internal!(model::LikelihoodModel,
                                    uni_row_number::Int,
                                    points::PointsAndLogLikelihood)

    interval_struct = model.uni_profiles_dict[uni_row_number]
    model.uni_profiles_dict[uni_row_number] = @set interval_struct.interval_points = points

    return nothing
end

# function update_uni_df_internal_points!(model::LikelihoodModel,
#                                         uni_row_number::Int,
#                                         num_points_in_interval::Int)
# 
#     model.uni_profiles_df[uni_row_number, [:not_evaluated_internal_points, :num_points]] .= false, num_points_in_interval+2
# 
#     return nothing
# end

"""
    get_points_in_interval_single_row(univariate_optimiser::Function, 
        model::LikelihoodModel,
        num_points_in_interval::Int,
        θi::Int,
        profile_type::AbstractProfileType,
        current_interval_points::PointsAndLogLikelihood,
        additional_width::Real=0.0)

Method for getting `num_points_in_interval` points inside a confidence interval for parameter `θi`, directly called by [`PlaceholderLikelihood.univariate_confidenceinterval`](@ref) and called via it's other method for [`get_points_in_interval!`](@ref). Optionally adds `additional_width` outside of the confidence interval, so long as a parameter bound is not reached. If a bound is reached, up until the bound will be considered instead.
"""
function get_points_in_interval_single_row(univariate_optimiser::Function, 
                                model::LikelihoodModel,
                                num_points_in_interval::Int,
                                θi::Int,
                                profile_type::AbstractProfileType,
                                current_interval_points::PointsAndLogLikelihood,
                                additional_width::Real=0.0)

    num_points_in_interval > 0 || throw(DomainError("num_points_in_interval must be a strictly positive integer"))
    additional_width >= 0 || throw(DomainError("additional_width must be greater than or equal to zero"))
    
    boundary_indices = current_interval_points.boundary_col_indices
    
    newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(model, θi)
    
    consistent = get_consistent_tuple(model, 0.0, profile_type, 1)
    p=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, ωranges=ωranges, consistent=consistent, 
        ω_opt=zeros(model.core.num_pars-1))
    
    if additional_width > 0.0
        boundary = current_interval_points.points[θi, boundary_indices]
        boundary_width = diff(boundary)[1]
        half_add_width = boundary_width * (additional_width / 2.0)
        interval_to_eval = [max(boundary[1]-half_add_width, model.core.θlb[θi]), 
                                min(boundary[2]+half_add_width, model.core.θub[θi])]
        
        interval_width = diff(interval_to_eval)[1]

        additional_widths = [boundary[1]-interval_to_eval[1], interval_to_eval[2]-boundary[2]]
        points_in_each_interval = [0, num_points_in_interval+2, 0]

        point_locations = Float64[]

        
        if additional_widths[1] > 0.0
            num_points = convert(Int, max(1.0, round((additional_widths[1]/interval_width)*num_points_in_interval, RoundDown)))
            points_in_each_interval[1] = num_points

            append!(point_locations, LinRange(interval_to_eval[1], 
                                                boundary[1], 
                                                points_in_each_interval[1]+1)[1:(end-1)]
                    )
        end

        append!(point_locations, LinRange(boundary[1], boundary[2], num_points_in_interval+2)
                    )

        if additional_widths[2] > 0.0
            num_points = convert(Int, max(1.0, round((additional_widths[2]/interval_width)*num_points_in_interval, RoundDown)))
            points_in_each_interval[3] = num_points

            append!(point_locations, LinRange(boundary[2],
                                                interval_to_eval[2],
                                                points_in_each_interval[3]+1)[2:end]
                    )
        end
        new_boundary_indices = [points_in_each_interval[1]+1, points_in_each_interval[1]+points_in_each_interval[2]]

    else
        new_boundary_indices = [1, num_points_in_interval+2]
        point_locations = LinRange(current_interval_points.points[θi, boundary_indices[1]], 
                                    current_interval_points.points[θi, boundary_indices[2]], 
                                    num_points_in_interval+2)
    end

    total_points = length(point_locations)

    ll = zeros(total_points)
    interval_points = zeros(model.core.num_pars, total_points)

    ll[new_boundary_indices] .= current_interval_points.ll[boundary_indices]

    interval_points[:,new_boundary_indices[1]] .= current_interval_points.points[:,boundary_indices[1]]
    interval_points[:,new_boundary_indices[2]] .= current_interval_points.points[:,boundary_indices[2]]

    iter_inds = setdiff(1:total_points, new_boundary_indices)

    for i in iter_inds
        ll[i] = univariate_optimiser(point_locations[i], p)
        variablemapping!(@view(interval_points[:,i]), p.ω_opt, θranges, ωranges)
        # p.initGuess .= p.ω_opt .* 1.0
    end
    interval_points[θi,iter_inds] .= point_locations[iter_inds]

    return PointsAndLogLikelihood(interval_points, ll, new_boundary_indices)
end

"""
    get_points_in_interval_single_row(model::LikelihoodModel,
        uni_row_number::Int,
        num_points_in_interval::Int,
        additional_width::Real)

Alternate method called by [`get_points_in_interval!`](@ref).
"""
function get_points_in_interval_single_row(model::LikelihoodModel,
                                uni_row_number::Int,
                                num_points_in_interval::Int,
                                additional_width::Real)

    θi = model.uni_profiles_df.θindex[uni_row_number]
    profile_type = model.uni_profiles_df.profile_type[uni_row_number]
    univariate_optimiser = get_univariate_opt_func(profile_type)
    current_interval_points = get_uni_confidence_interval_points(model, uni_row_number)
    
    return get_points_in_interval_single_row(univariate_optimiser, model, num_points_in_interval, 
                                                θi, profile_type, current_interval_points, additional_width)
end

"""
    get_points_in_interval!(model::LikelihoodModel, 
        num_points_in_interval::Int; 
        confidence_levels::Vector{<:Float64}=Float64[], 
        profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[], 
        additional_width::Real=0.0)

Evaluate and save `num_points_in_interval` linearly spaced points between the confidence intervals of existing univariate profiles that meet the requirements of [`PlaceholderLikelihood.desired_df_subset`](@ref) (see Keyword Arguments), as well as any additional width on the sides of the interval. Modifies `model` in place.

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `num_points_in_interval`: an integer number of points to evaluate within the confidence interval. Points are linearly spaced in the interval and have their optimised log-likelihood value recorded. Useful for plots that visualise the confidence interval or for predictions from univariate profiles. 

# Keyword Arguments
- `confidence_levels`: a vector of confidence levels. If empty, all confidence levels of univariate profiles will be considered for finding interval points. Otherwise, only confidence levels of univariate profiles in `confidence_levels` will be considered. Default is `Float64[]` (any confidence level).
- `profile_types`: a vector of `AbstractProfileType` structs. If empty, all profile types of univariate profiles are considered. Otherwise, only univariate profiles with matching profile types will be considered. Default is `AbstractProfileType[]` (any profile type).
- `additional_width`: a `Real` number greater than or equal to zero. Specifies the additional width to optionally evaluate outside the confidence interval's width. Half of this additional width will be placed on either side of the confidence interval. If the additional width goes outside a bound on the parameter, only up to the bound will be considered. The spacing of points in the additional width will try to match the spacing of points evaluated inside the interval. Useful for plots that visualise the confidence interval as it shows the trend of the log-likelihood profile outside the interval range. Default is 0.0.

# Details

Interval points and their corresponding log-likelihood values are stored in the `interval_points` field of a [`UnivariateConfidenceStruct`](@ref). These are updated using [`PlaceholderLikelihood.update_uni_dict_internal!`](@ref).
"""
function get_points_in_interval!(model::LikelihoodModel,
                                    num_points_in_interval::Int;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    additional_width::Real=0.0
                                    )

    0 < num_points_in_interval || throw(DomainError("num_points_in_interval must be a strictly positive integer"))
    additional_width >= 0 || throw(DomainError("additional_width must be greater than or equal to zero"))
    
    sub_df = desired_df_subset(model.uni_profiles_df, model.num_uni_profiles, Int[], 
                confidence_levels, profile_types, 
                for_points_in_interval=(true, num_points_in_interval, additional_width))

    if nrow(sub_df) < 1
        return nothing
    end

    for i in 1:nrow(sub_df)
        points = get_points_in_interval_single_row(model, sub_df[i, :row_ind], 
                                            num_points_in_interval, additional_width)

        update_uni_dict_internal!(model, sub_df[i, :row_ind], points)
    end

    sub_df[:, :not_evaluated_internal_points] .= false
    sub_df[:, :not_evaluated_predictions] .= true
    sub_df[:, :num_points] .= num_points_in_interval+2
    sub_df[:, :additional_width] .= additional_width

    return nothing
end