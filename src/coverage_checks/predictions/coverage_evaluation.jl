"""
    union_of_prediction_extrema(df::DataFrame, dict::Dict)

Compute the extrema union of the extrema of the predictions of each profile in the rows of `df`, with extrema stored in `dict`.
"""
function union_of_prediction_extrema(df::Union{DataFrame, SubDataFrame}, dict::Dict, multiple_outputs::Bool)
    extrema_union = zeros(size(dict[df.row_ind[1]].extrema))

    for (i, row_ind) in enumerate(df.row_ind)
        if i == 1
            extrema_union .= dict[row_ind].extrema
            continue
        end

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
    union_of_prediction_realisations_extrema(df::DataFrame, dict::Dict)

Compute the extrema union of the extrema of the prediction realisations of each profile in the rows of `df`, with extrema stored in `dict`.
"""
function union_of_prediction_realisations_extrema(df::Union{DataFrame, SubDataFrame}, dict::Dict, multiple_outputs::Bool)
    extrema_union = zeros(size(dict[df.row_ind[1]].extrema))

    for (i, row_ind) in enumerate(df.row_ind)
        if i == 1
            extrema_union .= dict[row_ind].realisations.extrema
            continue
        end

        if multiple_outputs
            extrema_union[:,1,:] .= min.(extrema_union[:,1,:], dict[row_ind].realisations.extrema[:,1,:])
            extrema_union[:,2,:] .= max.(extrema_union[:,2,:], dict[row_ind].realisations.extrema[:,2,:])
        else
            extrema_union[:,1] .= min.(extrema_union[:,1], dict[row_ind].realisations.extrema[:,1])
            extrema_union[:,2] .= max.(extrema_union[:,2], dict[row_ind].realisations.extrema[:,2])
        end
    end
    return extrema_union
end

"""
    evaluate_conf_simultaneous_coverage(pointwise::BitArray, 
        region::Float64)

Evaluate whether at least `region` proportion of entries in `pointwise` are `true`. This is simultaneous coverage with a different goal; i.e. `region` proportion of points are contained simultaneously.
"""
function evaluate_conf_simultaneous_coverage(pointwise::BitArray, region::Float64)
    return (sum(pointwise)/length(pointwise)) ≥ region
end

"""
    evaluate_coverage(y_true::Array, 
        y_pred_extrema::Union{Array,Missing}, 
        multiple_outputs::Bool)

simultaneous coverage requires y_true to be inside y_pred for every row and column in y_true (every observed variable)

pointwise coverage is on a row and column basis; whether y_true is inside y_pred at each individual row and column 
"""
function evaluate_coverage(y_true::Array, y_pred_extrema::Union{Array,Missing}, multiple_outputs::Bool)
    pointwise = falses(size(y_true))
    if ismissing(y_pred_extrema)
        return false, pointwise
    end

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
    evaluate_coverage(model::LikelihoodModel, 
        y_true::Array, 
        profile_kind::Symbol, 
        multiple_outputs::Bool, 
        len_θs::Int)
"""
function evaluate_coverage(model::LikelihoodModel, y_true::Array, profile_kind::Symbol, multiple_outputs::Bool, len_θs::Int)
    missing_profiles=false
    num_missing=0
    if profile_kind == :univariate
        df, dict = model.uni_profiles_df, model.uni_predictions_dict
        missing_profiles = len_θs != model.num_uni_profiles
        num_missing = len_θs - model.num_uni_profiles
    elseif profile_kind == :bivariate
        df, dict = model.biv_profiles_df, model.biv_predictions_dict
        missing_profiles = len_θs != model.num_biv_profiles
        num_missing = len_θs - model.num_biv_profiles
    elseif profile_kind == :dimensional
        df, dict = model.dim_samples_df, model.dim_predictions_dict
        missing_profiles = len_θs != model.num_dim_samples
        num_missing = len_θs - model.num_dim_samples
    end

    if missing_profiles
        @error string(num_missing, " out of ", len_θs, " profiles failed to run. This iteration will not count towards the coverage statistic")
        individual_coverage = [evaluate_coverage(y_true, missing, multiple_outputs) for _ in 1:len_θs]
        union_coverage = [evaluate_coverage(y_true, missing, multiple_outputs) for _ in 1:len_θs]

        return individual_coverage, union_coverage, false
    end
    
    individual_coverage = [evaluate_coverage(y_true, dict[row_ind].extrema, multiple_outputs) for row_ind in df.row_ind]
    
    union_coverage = []
    for i in 1:len_θs
        sub_df = @view df[sample(1:len_θs, i, replace=false, ordered=true), :]
        extrema_union = union_of_prediction_extrema(sub_df, dict, multiple_outputs)
        push!(union_coverage, evaluate_coverage(y_true, extrema_union, multiple_outputs))
    end

    return individual_coverage, union_coverage, true
end

"""
    evaluate_coverage_realisations(model::LikelihoodModel, 
        testing_data::Array, 
        profile_kind::Symbol, 
        multiple_outputs::Bool, 
        len_θs::Int)
"""
function evaluate_coverage_realisations(model::LikelihoodModel, testing_data::Array, profile_kind::Symbol, multiple_outputs::Bool, len_θs::Int)
    missing_profiles = false
    num_missing = 0
    if profile_kind == :univariate
        df, dict = model.uni_profiles_df, model.uni_predictions_dict
        missing_profiles = len_θs != model.num_uni_profiles
        num_missing = len_θs - model.num_uni_profiles
    elseif profile_kind == :bivariate
        df, dict = model.biv_profiles_df, model.biv_predictions_dict
        missing_profiles = len_θs != model.num_biv_profiles
        num_missing = len_θs - model.num_biv_profiles
    elseif profile_kind == :dimensional
        df, dict = model.dim_samples_df, model.dim_predictions_dict
        missing_profiles = len_θs != model.num_dim_samples
        num_missing = len_θs - model.num_dim_samples
    end

    if missing_profiles
        @error string(num_missing, " out of ", len_θs, " profiles failed to run. This iteration will not count towards the coverage statistic")
        individual_coverage = [evaluate_coverage(testing_data, missing, multiple_outputs) for _ in 1:len_θs]
        union_coverage = [evaluate_coverage(testing_data, missing, multiple_outputs) for _ in 1:len_θs]

        return individual_coverage, union_coverage, false
    end

    individual_coverage = [evaluate_coverage(testing_data, dict[row_ind].realisations.extrema, multiple_outputs) for row_ind in df.row_ind]

    union_coverage = []
    for i in 1:len_θs
        sub_df = @view df[sample(1:len_θs, i, replace=false, ordered=true), :]
        extrema_union = union_of_prediction_realisations_extrema(sub_df, dict, multiple_outputs)
        push!(union_coverage, evaluate_coverage(testing_data, extrema_union, multiple_outputs))
    end

    return individual_coverage, union_coverage, true
end



"""
    evaluate_coverage_reference_sets(y_true::Array, 
        y_pred_extrema::Union{Array,Missing}, 
        multiple_outputs::Bool)
simultaneous coverage requires y_true to be inside y_pred for every row and column in y_true (every observed variable)

pointwise coverage is on a row and column basis; whether y_true is inside y_pred at each individual row and column 
"""
function evaluate_coverage_reference_sets(reference_set_data::Tuple{Array,Array}, y_pred_extrema::Union{Array,Missing}, multiple_outputs::Bool)
    ref_set_lq, ref_set_uq = reference_set_data
    pointwise = falses(size(ref_set_lq))
    if ismissing(y_pred_extrema)
        return false, pointwise
    end

    if multiple_outputs
        for col in axes(ref_set_lq, 2)
            pointwise[:, col] .= Base.isbetween.(y_pred_extrema[:, 1, col], ref_set_lq[:, col], y_pred_extrema[:, 2, col]) .&& 
                Base.isbetween.(y_pred_extrema[:, 1, col], ref_set_uq[:, col], y_pred_extrema[:, 2, col])
        end
        return all(pointwise), pointwise
    end

    pointwise .= Base.isbetween.(y_pred_extrema[:,1], ref_set_lq, y_pred_extrema[:,2]) .&&
        Base.isbetween.(y_pred_extrema[:,1], ref_set_uq, y_pred_extrema[:,2])
    return all(pointwise), pointwise
end

"""
    evaluate_coverage_reference_sets(model::LikelihoodModel, 
        reference_set_data::Tuple{Array,Array}, 
        profile_kind::Symbol, 
        multiple_outputs::Bool, 
        len_θs::Int, 
        not_missing_profiles::Bool)
"""
function evaluate_coverage_reference_sets(model::LikelihoodModel, 
        reference_set_data::Tuple{Array,Array}, 
        profile_kind::Symbol, 
        multiple_outputs::Bool, 
        len_θs::Int, 
        not_missing_profiles::Bool)

    if profile_kind == :univariate
        df, dict = model.uni_profiles_df, model.uni_predictions_dict
    elseif profile_kind == :bivariate
        df, dict = model.biv_profiles_df, model.biv_predictions_dict
    elseif profile_kind == :dimensional
        df, dict = model.dim_samples_df, model.dim_predictions_dict
    end

    if !not_missing_profiles
        individual_coverage = [evaluate_coverage_reference_sets(reference_set_data, missing, multiple_outputs) for _ in 1:len_θs]
        union_coverage = [evaluate_coverage_reference_sets(reference_set_data, missing, multiple_outputs) for _ in 1:len_θs]
        return individual_coverage, union_coverage
    end

    individual_coverage = [evaluate_coverage_reference_sets(reference_set_data, dict[row_ind].realisations.extrema, multiple_outputs) for row_ind in df.row_ind]

    union_coverage = []
    for i in 1:len_θs
        sub_df = @view df[sample(1:len_θs, i, replace=false, ordered=true), :]
        extrema_union = union_of_prediction_realisations_extrema(sub_df, dict, multiple_outputs)
        push!(union_coverage, evaluate_coverage_reference_sets(reference_set_data, extrema_union, multiple_outputs))
    end
    
    return individual_coverage, union_coverage
end