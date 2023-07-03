"""
    rebuild_bivariate_datastructures!(model::LikelihoodModel)

Rebuilds all the bivariate datastructures so that the :row_ind of rows in `model.biv_profiles_df` begins at 1 and increases in increments of 1, updating dictionary keys with the new row indexes. 
"""
function rebuild_bivariate_datastructures!(model::LikelihoodModel)
    # rebalance dictionaries according to new row indexes
    for i in 1:model.num_biv_profiles
        row_ind = model.biv_profiles_df[i, :row_ind]

        if i != row_ind
            conf_struct = pop!(model.biv_profiles_dict, row_ind)
            model.biv_profiles_dict[i] = conf_struct

            if !model.biv_profiles_df[i, :not_evaluated_predictions]
                predict_struct = pop!(model.biv_predictions_dict, row_ind)
                model.biv_predictions_dict[i] = predict_struct
            end

            model.biv_profiles_df[i, :row_ind] = i
            model.biv_profile_row_exists[(model.biv_profiles_df[i, :θindices],
                                            model.biv_profiles_df[i, :profile_type],
                                            model.biv_profiles_df[i, :method])
                                            ][model.biv_profiles_df[i, :conf_level]] = i
        end
    end
    return nothing
end

"""
    combine_bivariate_boundaries!(model::LikelihoodModel;
        confidence_level::Float64=0.95,
        profile_type::AbstractProfileType=LogLikelihood(),
        methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
        not_evaluated_predictions::Bool=true)

Combines the `confidence_level` bivariate boundaries of `profile_type` found using `methods` into a single [`CombinedBivariateMethod`] boundary for each interest parameter, modifying `model` destructively in place. Rows of `model.biv_profiles_df` to combine are found using [`PlaceholderLikelihood.desired_df_subset`s](@ref). Dictionary entries and dataframe rows of boundaries that have beeen combined will be deleted and the datastructures will be rebuilt according to the new row indices of `model.biv_profiles_df`. 

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level of `profile_type` boundaries to combine. Default is 0.95 (95%).
- `profile_type`: the profile type of boundaries to combine. Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `methods`: a vector of methods of type [`AbstractBivariateMethod`](@ref) for combining boundaries found using those method types. `methods` should not contain [`CombinedBivariateMethod`](@ref), but the case where it is included in `methods` is handled (it will be removed from the vector). Default is AbstractBivariateMethod[] (boundaries found using all methods are combined).
- `not_evaluated_predictions`: a boolean specifiying whether to combine only boundaries that have either had or not had predictions evaluated. If predictions are evaluated for the combined struct (if it exists) but not for the rows to combine with it, they will not be combined, and vice versa. Default is true.
"""
function combine_bivariate_boundaries!(model::LikelihoodModel;
                                        confidence_level::Float64=0.95,
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                        not_evaluated_predictions::Bool=true)

    
    if !isempty(methods) && CombinedBivariateMethod() ∈ [methods]
        setdiff!(methods, [CombinedBivariateMethod()])
    end

    if not_evaluated_predictions
        sub_df = desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, Tuple{Int,Int}[],
            confidence_level, [profile_type], methods, for_prediction_generation=true, remove_combined_method=true)
    else
        sub_df = desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, Tuple{Int,Int}[],
            confidence_level, [profile_type], methods, for_prediction_plots=true, remove_combined_method=true)
    end

    if nrow(sub_df) < 1
        return nothing
    end

    rows_to_combine = sub_df.row_ind .* 1
    θcombinations = unique(sub_df.θindices)
    len_θcombinations = length(θcombinations)
    init_biv_profile_row_exists!(model, θcombinations, profile_type, CombinedBivariateMethod())

    num_to_reuse = 0
    for i in eachindex(θcombinations)
        if model.biv_profile_row_exists[(θcombinations[i], profile_type, CombinedBivariateMethod())][confidence_level] != 0
            num_to_reuse += 1
        end
    end
    num_rows_required = ((len_θcombinations - num_to_reuse) + model.num_biv_profiles) - nrow(model.biv_profiles_df)

    if num_rows_required > 0
        add_biv_profiles_rows!(model, num_rows_required)
    end

    row_was_combined = falses(nrow(model.biv_profiles_df))

    # for each θcombination, create a combined struct (if needed) and add to datastructures
    # destructively remove combined rows from datastructures that aren't `model.biv_profiles_df` as we go
    for i in eachindex(θcombinations)

        combo_row = model.biv_profile_row_exists[(θcombinations[i], profile_type, CombinedBivariateMethod())][confidence_level]
        if combo_row != 0 && (model.biv_profiles_df[combo_row, :not_evaluated_predictions] != not_evaluated_predictions)
            continue
        end

        desired_rows = sub_df.θindices .== Ref(θcombinations[i])
        rows = rows_to_combine[desired_rows]
        row_was_combined[rows] .= true
        for row in rows
            model.biv_profile_row_exists[(θcombinations[i], profile_type, model.biv_profiles_df[row, :method])][confidence_level] = 0
        end

        local predict_struct::AbstractPredictionStruct
        if !not_evaluated_predictions
            if combo_row != 0
                predict_struct = model.biv_predictions_dict[combo_row]
            end

            for row in rows
                if @isdefined predict_struct
                    predict_struct = merge(predict_struct, pop!(model.biv_predictions_dict, row))
                else
                    predict_struct = pop!(model.biv_predictions_dict, row)
                end
            end
        end

        local conf_struct::BivariateConfidenceStruct
        if combo_row != 0
            conf_struct = model.biv_profiles_dict[combo_row]
        end
        for row in rows
            if @isdefined conf_struct
                conf_struct = merge(conf_struct, pop!(model.biv_profiles_dict, row))
            else
                conf_struct = pop!(model.biv_profiles_dict, row)
            end
        end
        if combo_row == 0
            model.num_biv_profiles +=1
            combo_row = model.num_biv_profiles*1
        end

        not_evaluated_internal_points = size(conf_struct.internal_points.points, 2) == 0
        set_biv_profiles_row!(model, combo_row, θcombinations[i], not_evaluated_internal_points, not_evaluated_predictions,
                                true, confidence_level, profile_type, CombinedBivariateMethod(), 
                                size(conf_struct.confidence_boundary, 2))
        
        model.biv_profiles_dict[combo_row] = conf_struct
        if !not_evaluated_predictions
           model.biv_predictions_dict[combo_row] = predict_struct 
        end
    end

    rows_to_keep = collect(1:model.num_biv_profiles)[.!row_was_combined]

    model.biv_profiles_df = model.biv_profiles_df[rows_to_keep, :]
    model.num_biv_profiles = nrow(model.biv_profiles_df)

    rebuild_bivariate_datastructures!(model::LikelihoodModel)

    return nothing
end