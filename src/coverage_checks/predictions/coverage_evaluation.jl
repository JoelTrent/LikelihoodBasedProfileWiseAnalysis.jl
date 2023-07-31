"""
    union_of_prediction_extrema(df::DataFrame, dict::Dict)

Compute the extrema union of the extrema of the predictions of each profile in the rows of `df`, with extrema stored in `dict`.
"""
function union_of_prediction_extrema(df::DataFrame, dict::Dict, multiple_outputs::Bool)
    extrema_union = zeros(size(dict[df.row_ind[1]].extrema))

    for row_ind in df.row_ind
        if multiple_outputs
            extrema_union[:,1,:] .= min.(extrema_union[:,1,:], dict[row_ind].extrema[:,1,:])
            extrema_union[:,2,:] .= max.(extrema_union[:,2,:], dict[row_ind].extrema[:,2,:])
        else
            extrema_union[:,1] .= min.(extrema_union[:,1], dict[row_ind].extrema[:,1])
            extrema_union[:,2] .= max.(extrema_union[:,2], dict[row_ind].extrema[:,2])
        end
    end
    return extrema_union
end

"""
simultaneous coverage requires y_true to be inside y_pred for every column in y_true (every observed variable)

pointwise coverage is 
"""
function evaluate_coverage(y_true::Array, y_pred_extrema::Array, multiple_outputs::Bool)
    pointwise = falses(size(y_true))
    if multiple_outputs
        for col in axes(y_true, 2)
            pointwise[:, col] .= Base.isbetween.(y_pred_extrema[:,1,col], y_true[:,col], y_pred_extrema[:,2,col])
        end
        return all(pointwise), pointwise
    end

    pointwise .= Base.isbetween.(y_pred_extrema[:,1], y_true, y_pred_extrema[:,2])
    return all(pointwise), pointwise
end

"""

"""
function evaluate_coverage(model::LikelihoodModel, y_true::Array, profile_kind::Symbol, multiple_outputs::Bool)
    if profile_kind == :univariate
        df, dict = model.uni_profiles_df, model.uni_predictions_dict
    elseif profile_kind == :bivariate
        df, dict = model.biv_profiles_df, model.biv_predictions_dict
    elseif profile_kind == :dimensional
        df, dict = model.dim_samples_df, model.dim_predictions_dict
    end

    individual_coverage = [evaluate_coverage(y_true, dict[row_ind].extrema, multiple_outputs) for row_ind in df.row_ind]

    extrema_union = union_of_prediction_extrema(df, dict, multiple_outputs)
    union_coverage = evaluate_coverage(y_true, extrema_union, multiple_outputs)

    return individual_coverage, union_coverage
end