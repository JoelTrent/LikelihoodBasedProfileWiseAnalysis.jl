"""
    predictions_can_be_merged(model::LikelihoodModel, rows::AbstractVector{<:Int})

Returns true if all the prediction struct arrays have the same size in the 1st (and 3rd if relevant) dimensions. Return false otherwise.
"""
function predictions_can_be_merged(model::LikelihoodModel, combo_row::Int, rows::AbstractVector{<:Int})

    all_rows = combo_row != 0 ? vcat([combo_row], rows) : rows .* 1
    expected_size = size(model.biv_predictions_dict[all_rows[1]].extrema)
    
    for row in all_rows[2:end]
        if expected_size != size(model.biv_predictions_dict[row].extrema)
            return false, (all_rows[1], row)
        end
    end

    return true, (0, 0)
end


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
                                            model.biv_profiles_df[i, :dof],
                                            model.biv_profiles_df[i, :profile_type],
                                            model.biv_profiles_df[i, :method])
                                            ][model.biv_profiles_df[i, :conf_level]] = i
        end
    end
    return nothing
end

"""
    combine_bivariate_boundaries!(model::LikelihoodModel;
        <keyword arguments>)

Combines the `confidence_level` bivariate boundaries at `dof` of `profile_type` found using `methods` into a single [`CombinedBivariateMethod`](@ref) boundary for each set of interest parameters, modifying `model` destructively in place. Rows of `model.biv_profiles_df` to combine are found using the bivariate method of [`PlaceholderLikelihood.desired_df_subset`](@ref). Dictionary entries and dataframe rows of boundaries that have beeen combined will be deleted and the datastructures will be rebuilt according to the new row indices of `model.biv_profiles_df`. 

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level of `profile_type` boundaries to combine. Default is 0.95 (95%).
- `dof`: a integer ∈ [2, model.core.num_pars] for the degrees of freedom of `profile_type` boundaries to combine. Default is 2.
- `profile_type`: the profile type of boundaries to combine. Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `methods`: a vector of methods of type [`AbstractBivariateMethod`](@ref) for combining boundaries found using those method types. `methods` should not contain [`CombinedBivariateMethod`](@ref), but the case where it is included in `methods` is handled: it will be removed from the vector. Default is `AbstractBivariateMethod[]` (boundaries found using all methods are combined).
- `not_evaluated_predictions`: a boolean specifiying whether to combine only boundaries that have not had or have had predictions evaluated. If predictions are evaluated for the combined struct (if it exists) but not for the rows to combine with it, they will not be combined, and vice versa. Default is `true` (combine boundaries that have not had predictions evaluated).

!!! info "Combining predictions"
    If predictions have been evaluated: the time points at which predictions have been evaluated at must be the same for all of the boundaires that are being combined.
"""
function combine_bivariate_boundaries!(model::LikelihoodModel;
                                        confidence_level::Float64=0.95,
                                        dof::Int=2,
                                        profile_type::AbstractProfileType=LogLikelihood(),
                                        methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                        not_evaluated_predictions::Bool=true)

    if !isempty(methods) && CombinedBivariateMethod() ∈ [methods]
        setdiff!(methods, [CombinedBivariateMethod()])
    end

    if not_evaluated_predictions
        sub_df = desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, Tuple{Int,Int}[],
            confidence_level, dof, [profile_type], methods, for_prediction_generation=true, remove_combined_method=true)
    else
        sub_df = desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, Tuple{Int,Int}[],
            confidence_level, dof, [profile_type], methods, for_prediction_plots=true, remove_combined_method=true)
    end

    if nrow(sub_df) < 1
        return nothing
    end

    rows_to_combine = sub_df.row_ind .* 1
    θcombinations = unique(sub_df.θindices)
    len_θcombinations = length(θcombinations)
    init_biv_profile_row_exists!(model, θcombinations, dof, profile_type, CombinedBivariateMethod())

    num_to_reuse = 0
    for i in eachindex(θcombinations)
        if model.biv_profile_row_exists[(θcombinations[i], dof, profile_type, CombinedBivariateMethod())][confidence_level] != 0
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

        combo_row = model.biv_profile_row_exists[(θcombinations[i], dof, profile_type, CombinedBivariateMethod())][confidence_level]
        if combo_row != 0 && (model.biv_profiles_df[combo_row, :not_evaluated_predictions] != not_evaluated_predictions)
            continue
        end

        desired_rows = sub_df.θindices .== Ref(θcombinations[i])
        rows = rows_to_combine[desired_rows]

        if !not_evaluated_predictions
            merge_allowed, error_rows = predictions_can_be_merged(model, combo_row, rows)
            if !merge_allowed
                @warn string("the boundaries for parameter combination ", θcombinations[i], 
                                " could not be combined because the evaluated predictions of bivariate profile row ", 
                                error_rows[1], " and ", error_rows[2], " have different dimensions and cannot be safely merged")
                continue
            end
        end
        
        row_was_combined[rows] .= true
        for row in rows
            model.biv_profile_row_exists[(θcombinations[i], dof, profile_type, model.biv_profiles_df[row, :method])][confidence_level] = 0
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
                                true, confidence_level, dof, profile_type, CombinedBivariateMethod(), 
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