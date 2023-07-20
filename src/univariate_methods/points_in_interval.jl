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
        additional_width::Real,
        use_threads::Bool)

Method for getting `num_points_in_interval` points inside a confidence interval for parameter `θi`, directly called by [`PlaceholderLikelihood.univariate_confidenceinterval`](@ref) and called via it's other method for [`get_points_in_intervals!`](@ref). Adds `additional_width` outside of the confidence interval, so long as a parameter bound is not reached. If a bound is reached, up until the bound will be considered instead.
"""
function get_points_in_interval_single_row(univariate_optimiser::Function, 
                                model::LikelihoodModel,
                                num_points_in_interval::Int,
                                θi::Int,
                                profile_type::AbstractProfileType,
                                current_interval_points::PointsAndLogLikelihood,
                                additional_width::Real,
                                use_threads::Bool)

    num_points_in_interval > 0 || throw(DomainError("num_points_in_interval must be a strictly positive integer"))
    additional_width >= 0 || throw(DomainError("additional_width must be greater than or equal to zero"))
    
    boundary_indices = current_interval_points.boundary_col_indices
    
    newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(model, θi)
    
    consistent = get_consistent_tuple(model, 0.0, profile_type, 1)
    q=(ind=θi, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, ωranges=ωranges, consistent=consistent, 
        options=model.core.optimizationsettings)
    
    point_locations = zeros(0)

    if additional_width > 0.0
        boundary = current_interval_points.points[θi, boundary_indices]
        boundary_width = diff(boundary)[1]
        half_add_width = boundary_width * (additional_width / 2.0)
        interval_to_eval = [max(boundary[1]-half_add_width, model.core.θlb[θi]), 
                                min(boundary[2]+half_add_width, model.core.θub[θi])]
        
        interval_width = diff(interval_to_eval)[1]

        additional_widths = [boundary[1]-interval_to_eval[1], interval_to_eval[2]-boundary[2]]
        points_in_each_interval = [0, num_points_in_interval+2, 0]

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

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=length(iter_inds)) 
    let point_locations = point_locations
        @floop ex for i in iter_inds
            FLoops.@init ω_opt = zeros(model.core.num_pars-1)
            p = (ω_opt=ω_opt, q=q, options=model.core.optimizationsettings)
            ll[i] = univariate_optimiser(point_locations[i], p)
            variablemapping!(@view(interval_points[:,i]), p.ω_opt, θranges, ωranges)
            # p.initGuess .= p.ω_opt .* 1.0
        end
    end
    interval_points[θi,iter_inds] .= point_locations[iter_inds]

    return PointsAndLogLikelihood(interval_points, ll, new_boundary_indices)
end

"""
    get_points_in_interval_single_row(model::LikelihoodModel,
        uni_row_number::Int,
        num_points_in_interval::Int,
        additional_width::Real
        use_threads::Bool,
        channel::RemoteChannel)

Alternate method called by [`get_points_in_intervals!`](@ref).
"""
function get_points_in_interval_single_row(model::LikelihoodModel,
                                uni_row_number::Int,
                                num_points_in_interval::Int,
                                additional_width::Real,
                                use_threads::Bool,
                                channel::RemoteChannel)

    try
        @timeit_debug timer "Univariate points in interval" begin
            θi = model.uni_profiles_df.θindex[uni_row_number]
            profile_type = model.uni_profiles_df.profile_type[uni_row_number]
            univariate_optimiser = get_univariate_opt_func(profile_type)
            current_interval_points = get_uni_confidence_interval_points(model, uni_row_number)
            
            output = get_points_in_interval_single_row(univariate_optimiser, model, num_points_in_interval,
                θi, profile_type, current_interval_points, additional_width, use_threads)
            put!(channel, true)
            return output 
        end
    catch
        @error string("an error occurred when finding the points inside a univariate confidence interval with settings: ",
            (uni_row_number=uni_row_number, num_points_in_interval=num_points_in_interval,
                additional_width=additional_width))
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
            println(stdout)
        end
    end
    return nothing
end

"""
    get_points_in_intervals!(model::LikelihoodModel, 
        num_points_in_interval::Int; 
        <keyword arguments>)

Evaluate and save `num_points_in_interval` linearly spaced points between the confidence intervals of existing univariate profiles that meet the requirements of the univariate method of [`PlaceholderLikelihood.desired_df_subset`](@ref) (see Keyword Arguments), as well as any additional width on the sides of the interval. Modifies `model` in place.

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `num_points_in_interval`: an integer number of points to evaluate within the confidence interval. Points are linearly spaced in the interval and have their optimised log-likelihood value recorded (standardised to 0.0 at the MLE point). Useful for plots that visualise the confidence interval or for predictions from univariate profiles. 

# Keyword Arguments
- `additional_width`: a `Real` number greater than or equal to zero. Specifies the additional width to optionally evaluate outside the confidence interval's width. Half of this additional width will be placed on either side of the confidence interval. If the additional width goes outside a bound on the parameter, only up to the bound will be considered. The spacing of points in the additional width will try to match the spacing of points evaluated inside the interval. Useful for plots that visualise the confidence interval as it shows the trend of the log-likelihood profile outside the interval range. Default is `0.0`.
- `confidence_levels`: a vector of confidence levels. If empty, all confidence levels of univariate profiles will be considered for finding interval points. Otherwise, only confidence levels in `confidence_levels` will be considered. Default is `Float64[]` (any confidence level).
- `profile_types`: a vector of `AbstractProfileType` structs. If empty, all profile types of univariate profiles are considered. Otherwise, only profiles with matching profile types will be considered. Default is `AbstractProfileType[]` (any profile type).
- `not_evaluated_predictions`: a boolean specifying whether to only get points in intervals of profiles that have not had predictions evaluated (true) or for all profiles (false). If `false`, then any existing predictions will be forgotten by the `model` and overwritten the next time predictions are evaluated for each profile. Default is `true`.
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of `θs_to_profile` completed and estimated time of completion. Default is `model.show_progress`.
- `use_threads`: boolean variable specifying, if the number of workers for distributed computing is not greater than 1 (`!Distributed.nworkers()>1`), to use a parallelised for loop across `Threads.nthreads()` threads to evaluate the interval points. Default is `false`.

# Details

Interval points and their corresponding log-likelihood values are stored in the `interval_points` field of a [`UnivariateConfidenceStruct`](@ref). These are updated using [`PlaceholderLikelihood.update_uni_dict_internal!`](@ref). Nuisance parameters of each point in univariate interest parameter space are found by maximising the log-likelihood function given by the `profile_type` of the profile. 

If [`get_points_in_intervals!`](@ref) has already been used on a univariate profile, with the same values of `num_points_in_interval` and `additional_width`, it will not be recomputed for that profile.

## Parallel Computing Implementation

If [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is being used then each set of interval points for distinct interest parameters will be computed in parallel across `Distributed.nworkers()` workers. If it is not being used (`Distributed.nworkers()` is equal to `1`) and `use_threads` is `true` then the interval points of each interest parameter will be computed in parallel across `Threads.nthreads()` threads . It is highly recommended to set `use_threads` to `true` in that situation.

## Iteration Speed Of the Progress Meter

An iteration within the progress meter is specified as the time it takes for all internal points within a univariate confidence interval to be found (as well as any outside, if `additional_width` is greater than zero).
"""
function get_points_in_intervals!(model::LikelihoodModel,
                                    num_points_in_interval::Int;
                                    additional_width::Real=0.0,
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    not_evaluated_predictions::Bool=true,
                                    show_progress::Bool=model.show_progress,
                                    use_threads::Bool=false)

    function argument_handling()
        num_points_in_interval > 0 || throw(DomainError("num_points_in_interval must be a strictly positive integer"))
        additional_width >= 0 || throw(DomainError("additional_width must be greater than or equal to zero"))
    
        (use_threads && timeit_debug_enabled()) &&
            throw(ArgumentError("use_threads cannot be true when debug timings from TimerOutputs are enabled. Either set use_threads to false or disable debug timings using `PlaceholderLikelihood.TimerOutputs.disable_debug_timings(PlaceholderLikelihood)`"))

        (use_threads && nworkers() > 1) &&
            throw(ArgumentError("use_threads cannot be true when the number of workers for distributed computing is greater than 1 (`Distributed.nworkers()>1`). Either set use_threads to false or remove these workers using `Distributed.rmprocs(workers())`"))
    end

    argument_handling()

    sub_df = desired_df_subset(model.uni_profiles_df, model.num_uni_profiles, Int[], 
                confidence_levels, profile_types, 
                for_points_in_interval=(true, num_points_in_interval, additional_width),
                for_prediction_generation=not_evaluated_predictions)

    if nrow(sub_df) < 1
        return nothing
    end

    p = Progress(nrow(sub_df); desc="Computing points in univariate confidence intervals: ",
                dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)
    channel = RemoteChannel(() -> Channel{Bool}(1))

    @sync begin
        @async while take!(channel)
            next!(p)
        end
        @async begin
            points_to_add = @distributed (vcat) for i in 1:nrow(sub_df)
                [(sub_df[i, :row_ind], get_points_in_interval_single_row(model, sub_df[i, :row_ind], 
                                            num_points_in_interval, additional_width, use_threads, channel))]
            end
            put!(channel, false)
            
            for (i, (row_ind, points)) in enumerate(points_to_add)
                if isnothing(points); continue end
                update_uni_dict_internal!(model, row_ind, points)
            end
        end
    end

    sub_df[:, :not_evaluated_internal_points] .= false
    sub_df[:, :not_evaluated_predictions] .= true
    sub_df[:, :num_points] .= num_points_in_interval+2
    sub_df[:, :additional_width] .= additional_width

    return nothing
end