"""
    add_prediction_function!(model::LikelihoodModel, predictfunction::Function)

Adds a prediction function, `predictfunction`, to `model` and evaluates the predicted response variable(s) at the data points using the maximum likelihood estimate for model parameters. Modifies `model` in place.
    
# Requirements for `predictfunction`
- A function to generate model predictions from that is paired with the `loglikefunction`. 
- Takes three arguments, `θ`, `data` and `t`, in that order, where `θ` and `data` are the same as for `loglikefunction` and `t` needs to be an optional third argument. 
- When `t` is not specified, the prediction function should be evaluated for the same time points/independent variable as the data. When `t` is specified, the prediction function should be evaluated for those specified time points/independent variable. A good practice is to include the time points of the data in the argument, `data`, so that the function can be specified as `predictfunction(θ, data, t=data.t)` (here `data` is a `NamedTuple`).
- The output of the function should be a 1D vector when there is a single predicted response variable or a 2D array when there are multiple predicted response variables. 
- The prediction(s) for each response variable should be stored in the columns of the array. In Julia, vectors are stored column-wise, so in the case where there is only one response variable, this will already be the case.
- The number of rows of each predicted response variable should be the same length as the vector `t` used to evaluate the response.
"""
function add_prediction_function!(model::LikelihoodModel,
                                    predictfunction::Function)

    ymle = predictfunction(model.core.θmle, model.core.data)
    ndims(ymle) ∈ SA[1, 2] || throw(DimensionMismatch(("predictfunction must return an array with 1D or 2D outputs")))

    corelikelihoodmodel = model.core
    model.core = @set corelikelihoodmodel.predictfunction = predictfunction
    model.core = @set corelikelihoodmodel.ymle = ymle

    return nothing
end

"""
    check_prediction_function_exists(model::LikelihoodModel)

Checks if a prediction function is stored in `model`, returning true if there is. Otherwise false is returned and a warning is logged. Requirements for a prediction function can be seen in [`add_prediction_function!`](@ref).
"""
function check_prediction_function_exists(model::LikelihoodModel)
    if model.core isa BaseLikelihoodModel
        @warn "the LikelihoodModel does not contain a log-likelihood function or a function for evaluating predictions. Please add a log-likelihood function using add_loglikelihood_function! then add a prediction function using add_prediction_function!"
        return false
    end
    if ismissing(model.core.predictfunction) 
        @warn "the LikelihoodModel does not contain a function for evaluating predictions. Please add a prediction function using add_prediction_function!"
        return false
    end

    return true
end

"""
    generate_prediction(predictfunction::Function,
        errorfunction::Function,
        data,
        t::AbstractVector,
        data_ymle::AbstractArray{<:Real},
        parameter_points::Matrix{Float64},
        proportion_to_keep::Real,
        region::Real,
        channel::Union{RemoteChannel,Missing}=missing)

Generates the predictions for response variables from a `predictfunction` which meets the requirements specified in [`add_prediction_function!`](@ref), given `data`, at time points `t` for each parameter combination in the columns of `parameter_points`. The extrema of all predictions is computed and `proportion_to_keep` of the individual predictions are kept. `errorfunction` is used to predict the lower and upper quartiles of realisations at each prediction point; the highest density `region` is returned.
    
Returns a [`PredictionStruct`] containing the kept predictions, prediction extrema, lower and upper quartiles of realisations from the error model at `confidence_level` at each predicted point and the realisation extrema. 

The prediction at each timepoint is stored in the corresponding row (1st dimension). The prediction for each parameter combination is stored in the corresponding column (2nd dimension). The prediction for multiple response variables is stored in the 3rd dimension.
"""
function generate_prediction(predictfunction::Function,
                                errorfunction::Function,
                                data,
                                t::AbstractVector,
                                data_ymle::AbstractArray{<:Real},
                                parameter_points::Matrix{Float64},
                                proportion_to_keep::Real,
                                region::Real,
                                channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(Inf)))
    try
        num_points = size(parameter_points, 2)
        
        if ndims(data_ymle) > 2
            error("this function has not been written to handle predictions that are stored in higher than 2 dimensions")
        end

        if ndims(data_ymle) == 2
            predictions = zeros(length(t), num_points, size(data_ymle, 2))

            for i in 1:num_points
                predictions[:,i,:] .= predictfunction(parameter_points[:,i], data, t)
                put!(channel, true)
            end

            lq, uq = zeros(length(t), num_points, size(data_ymle, 2)), zeros(length(t), num_points, size(data_ymle, 2))
            for i in 1:num_points
                lq[:,i,:], uq[:,i,:] = predict_realisations(errorfunction, predictions[:,i,:], parameter_points[:,i], region)
            end

        else
            predictions = zeros(length(t), num_points)

            for i in 1:num_points
                predictions[:,i] .= predictfunction(parameter_points[:,i], data, t)
                put!(channel, true)
            end

            lq, uq = zeros(length(t), num_points), zeros(length(t), num_points)
            for i in 1:num_points
                lq[:,i], uq[:,i] = predict_realisations(errorfunction, predictions[:,i], parameter_points[:,i], region)
            end
        end
        
        extrema = hcat(minimum(predictions, dims=2), maximum(predictions, dims=2))
        extrema_realisations = hcat(min.(minimum(lq, dims=2), minimum(uq, dims=2)), 
                                    max.(maximum(lq, dims=2), maximum(uq, dims=2)))
    

        num_to_keep = convert(Int, round(num_points*proportion_to_keep, RoundUp))
        if num_points < 2
            predict_struct = PredictionStruct(predictions, extrema, 
                PredictionRealisationsStruct(lq, uq, extrema_realisations))
            return predict_struct
        elseif num_to_keep < 2
            num_to_keep = 2
        end

        keep_i = sample(1:num_points, num_to_keep, replace=false, ordered=true)
        if ndims(data_ymle) == 2
            predict_struct = PredictionStruct(predictions[:,keep_i,:], extrema, 
                PredictionRealisationsStruct(lq[:,keep_i,:], uq[:,keep_i,:], extrema_realisations))
        else
            predict_struct = PredictionStruct(predictions[:,keep_i], extrema, 
                PredictionRealisationsStruct(lq[:,keep_i], uq[:,keep_i], extrema_realisations))
        end

        return predict_struct

    catch
        @error "an error occurred when generating a prediction"
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
            println(stdout)
        end
    end
    return nothing
end

"""
    generate_prediction(predictfunction::Function,
        errorfunction::Missing,
        data,
        t::AbstractVector,
        data_ymle::AbstractArray{<:Real},
        parameter_points::Matrix{Float64},
        proportion_to_keep::Real,
        region::Real,
        channel::Union{RemoteChannel,Missing}=missing)

Generates the predictions for response variables from a `predictfunction` which meets the requirements specified in [`add_prediction_function!`](@ref), given `data`, at time points `t` for each parameter combination in the columns of `parameter_points`. The extrema of all predictions is computed and `proportion_to_keep` of the individual predictions are kept.
    
Returns a [`PredictionStruct`] containing the kept predictions and prediction extrema. 

The prediction at each timepoint is stored in the corresponding row (1st dimension). The prediction for each parameter combination is stored in the corresponding column (2nd dimension). The prediction for multiple response variables is stored in the 3rd dimension.
"""
function generate_prediction(predictfunction::Function,
                                errorfunction::Missing,
                                data,
                                t::AbstractVector,
                                data_ymle::AbstractArray{<:Real},
                                parameter_points::Matrix{Float64},
                                proportion_to_keep::Real,
                                region::Real,
                                channel::RemoteChannel=RemoteChannel(() -> Channel{Bool}(Inf)))
    try
        num_points = size(parameter_points, 2)
        
        if ndims(data_ymle) > 2
            error("this function has not been written to handle predictions that are stored in higher than 2 dimensions")
        end

        if ndims(data_ymle) == 2
            predictions = zeros(length(t), num_points, size(data_ymle, 2))

            for i in 1:num_points
                predictions[:,i,:] .= predictfunction(parameter_points[:,i], data, t)
                put!(channel, true)
            end

        else
            predictions = zeros(length(t), num_points)

            for i in 1:num_points
                predictions[:,i] .= predictfunction(parameter_points[:,i], data, t)
                put!(channel, true)
            end
        end
        
        extrema = hcat(minimum(predictions, dims=2), maximum(predictions, dims=2))

        num_to_keep = convert(Int, round(num_points*proportion_to_keep, RoundUp))
        if num_points < 2
            predict_struct = PredictionStruct(predictions, extrema)
            return predict_struct
        elseif num_to_keep < 2
            num_to_keep = 2
        end

        keep_i = sample(1:num_points, num_to_keep, replace=false, ordered=true)
        if ndims(data_ymle) == 2
            predict_struct = PredictionStruct(predictions[:, keep_i, :], extrema)
        else
            predict_struct = PredictionStruct(predictions[:, keep_i], extrema)
        end

        return predict_struct

    catch
        @error "an error occurred when generating a prediction"
        for (exc, bt) in current_exceptions()
            showerror(stdout, exc, bt)
            println(stdout)
            println(stdout)
        end
    end
    return nothing
end

"""
    generate_prediction_univariate(model::LikelihoodModel,
        errorfunction::Union{Function, Missing},
        sub_df,
        row_i::Int,
        t::AbstractVector,
        proportion_to_keep::Real,  
        channel::RemoteChannel)

Generates predictions for the univariate profile in `sub_df` that corresponds to `row_i` at timepoints `t`.
"""
function generate_prediction_univariate(model::LikelihoodModel,
                                        sub_df,
                                        row_i::Int,
                                        t::AbstractVector,
                                        proportion_to_keep::Real, 
                                        channel::RemoteChannel)

    interval_points = get_uni_confidence_interval_points(model, sub_df[row_i, :row_ind])
    boundary_col_indices = interval_points.boundary_col_indices
    actual_internal = interval_points.ll .≥ get_target_loglikelihood(model, sub_df[row_i, :conf_level], EllipseApproxAnalytical(), sub_df[row_i, :dof])
    internal_indices = collect(1:length(interval_points.ll))[actual_internal]
    boundary_and_internal = union(boundary_col_indices, internal_indices)
    
    return generate_prediction(model.core.predictfunction, 
                model.core.errorfunction,
                model.core.data, t, model.core.ymle,
                interval_points.points[:, boundary_and_internal], proportion_to_keep, 
                sub_df[row_i, :region], channel)
end

"""
    generate_prediction_bivariate(model::LikelihoodModel,
        sub_df,
        row_i::Int,
        t::AbstractVector,
        proportion_to_keep::Real, 
        channel::RemoteChannel)

Generates predictions for the bivariate profile in `sub_df` that corresponds to `row_i` at timepoints `t`.
"""
function generate_prediction_bivariate(model::LikelihoodModel,
                                        sub_df,
                                        row_i::Int,
                                        t::AbstractVector,
                                        proportion_to_keep::Real, 
                                        channel::RemoteChannel)

    conf_struct = model.biv_profiles_dict[sub_df[row_i, :row_ind]]

    if !isempty(conf_struct.internal_points.points)
        return generate_prediction(model.core.predictfunction,
                                    model.core.errorfunction,
                                    model.core.data, t, model.core.ymle,
                                    hcat(conf_struct.confidence_boundary, conf_struct.internal_points.points), 
                                    proportion_to_keep, 
                                    sub_df[row_i, :region], channel)
    end
    return generate_prediction(model.core.predictfunction,
                                model.core.errorfunction,
                                model.core.data, t, model.core.ymle,
                                conf_struct.confidence_boundary, 
                                proportion_to_keep, 
                                sub_df[row_i, :region], channel)
end

"""
    generate_predictions_univariate!(model::LikelihoodModel,
        t::AbstractVector,
        proportion_to_keep::Real;
        <keyword arguments>)

Evalute and save `proportion_to_keep` individual predictions and their extrema from existing univariate profiles that meet the requirements of the univariate method of [`PlaceholderLikelihood.desired_df_subset`](@ref) (see Keyword Arguments) at time points `t`. Modifies `model` in place.

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `t`: a vector of time points to compute predictions at.
- `proportion_to_keep`: a `Real` number ∈ [0.0,1.0] of the proportion of individual predictions to save. Default is 1.0.

# Keyword Arguments
- `region`: a `Real` number ∈ [0, 1] specifying the proportion of the density of the error model from which to evaluate the highest density region. Default is `0.95`.
- `confidence_levels`: a vector of confidence levels. If empty, all confidence levels of univariate profiles will be considered for evaluating predictions from. Otherwise, only confidence levels in `confidence_levels` will be considered. Default is `Float64[]` (any confidence level).
- `dofs`: a vector of integer degrees of freedom used to define the asymptotic threshold for the extremities of a univariate profile. If empty, all degrees of freedom for univariate profiles will be considered for evaluating predictions from. Otherwise, only degrees of freedom in `dofs` will be considered. Default is `Int[]` (any degree of freedom).
- `profile_types`: a vector of `AbstractProfileType` structs. If empty, all profile types of univariate profiles are considered. Otherwise, only profiles with matching profile types will be considered. Default is `AbstractProfileType[]` (any profile type).
- `overwrite_predictions`: boolean variable specifying whether to re-evaluate and overwrite predictions for univariate profiles that have already had predictions evaluated. Set to `true` if predictions need to be evaluated for a new vector of time points. Default is `false`.
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of predictions evaluated and estimated time of completion. Default is `model.show_progress`.
- `use_distributed`: boolean variable specifying whether to use a normal for loop or a `@distributed` for loop across univariate profiles. Default is `true`.

# Details

For each univariate profile that meets the requirements of [`PlaceholderLikelihood.desired_df_subset`](@ref), it uses [`PlaceholderLikelihood.generate_prediction`](@ref) to generates the predictions for every parameter point in the profiles. The extrema of these predictions are computed (these are approximate simultaneous confidence bands for the prediction mean). The extrema and `proportion_to_keep` of the individual predictions are saved as a [`PredictionStruct`](@ref) in `model.uni_predictions_dict`, where the keys for the dictionary is the row number in `model.uni_profiles_df` of the corresponding profile.

## Distributed Computing Implementation

If [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is being used and `use_distributed` is `true` then the predictions from each univariate profile will be computed in parallel across `Distributed.nworkers()` workers.

## Iteration Speed Of the Progress Meter

The time/it value is the time it takes for a prediction to be evaluated from a single point in parameter space.
"""
function generate_predictions_univariate!(model::LikelihoodModel,
                                            t::AbstractVector,
                                            proportion_to_keep::Real=1.0;
                                            region::Real=0.95,
                                            confidence_levels::Vector{<:Float64}=Float64[],
                                            dofs::Vector{<:Int}=Int[],
                                            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                            overwrite_predictions::Bool=false,
                                            show_progress::Bool=model.show_progress,
                                            use_distributed::Bool=true)

    check_prediction_function_exists(model) || return nothing

    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the closed interval [0.0, 1.0]"))
    (0.0 <= region <= 1.0) || throw(DomainError("region must be in the closed interval [0.0, 1.0]"))
    sub_df = desired_df_subset(model.uni_profiles_df, model.num_uni_profiles, Int[], confidence_levels, dofs, profile_types, 
                                for_prediction_generation=!overwrite_predictions)

    if nrow(sub_df) < 1
        return nothing
    end

    sub_df[:, :region] .= region*1.

    totaltasks = sum(sub_df.num_points)
    channel_buffer_size = min(ceil(Int, totaltasks * 0.05), 50)
    channel = RemoteChannel(() -> Channel{Bool}(channel_buffer_size))
    p = Progress(totaltasks; desc="Generating univariate profile predictions: ",
                dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    @sync begin
        @async while take!(channel)
            next!(p)
        end

        @async begin
            if use_distributed
                predictions = @distributed (vcat) for i in 1:nrow(sub_df)
                    [generate_prediction_univariate(model, sub_df, i, t, proportion_to_keep, channel)]
                end

                for (i, predict_struct) in enumerate(predictions)
                    if isnothing(predict_struct); continue end

                    model.uni_predictions_dict[sub_df[i, :row_ind]] = predict_struct
                    sub_df[i, :not_evaluated_predictions] = false
                end
            else
                for i in 1:nrow(sub_df)
                    predict_struct = generate_prediction_univariate(model, sub_df, i, t, proportion_to_keep, channel)
                    if isnothing(predict_struct); continue end

                    model.uni_predictions_dict[sub_df[i, :row_ind]] = predict_struct
                    sub_df[i, :not_evaluated_predictions] = false
                end
            end
            put!(channel, false)
        end
    end

    return nothing
end

"""
    generate_predictions_bivariate!(model::LikelihoodModel,
        t::AbstractVector,
        proportion_to_keep::Real;
        <keyword arguments>)

Evalute and save `proportion_to_keep` individual predictions and their extrema from existing bivariate profiles that meet the requirements of the bivariate method of [`PlaceholderLikelihood.desired_df_subset`](@ref) (see Keyword Arguments) at time points `t`. Modifies `model` in place.

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `t`: a vector of time points to compute predictions at.
- `proportion_to_keep`: a `Real` number ∈ [0.0,1.0] of the proportion of individual predictions to save. Default is 1.0.

# Keyword Arguments
- `region`: a `Real` number ∈ [0, 1] specifying the proportion of the density of the error model from which to evaluate the highest density region. Default is `0.95`.
- `confidence_levels`: a vector of confidence levels. If empty, all confidence levels of bivariate profiles will be considered for evaluating predictions from. Otherwise, only confidence levels in `confidence_levels` will be considered. Default is `Float64[]` (any confidence level).
- `profile_types`: a vector of `AbstractProfileType` structs. If empty, all profile types of bivariate profiles are considered. Otherwise, only profiles with matching profile types will be considered. Default is `AbstractProfileType[]` (any profile type).
- `methods`: a vector of `AbstractBivariateMethod` structs. If empty all methods used to find bivariate profiles are considered. Otherwise, only profiles with matching method types will be considered (struct arguments do not need to be the same). Default is `AbstractBivariateMethod[]` (any bivariate method).
- `overwrite_predictions`: boolean variable specifying whether to re-evaluate and overwrite predictions for bivariate profiles that have already had predictions evaluated. Set to `true` if predictions need to be evaluated for a new vector of time points. Default is `false`.
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of predictions evaluated and estimated time of completion. Default is `model.show_progress`.
- `use_distributed`: boolean variable specifying whether to use a normal for loop or a `@distributed` for loop across bivariate profiles. Default is `true`.

# Details

For each bivariate profile that meets the requirements of [`PlaceholderLikelihood.desired_df_subset`](@ref), it uses [`PlaceholderLikelihood.generate_prediction`](@ref) to generates the predictions for every parameter point in the profiles. The extrema of these predictions are computed (these are approximate simultaneous confidence bands for the prediction mean). The extrema and `proportion_to_keep` of the individual predictions are saved as a [`PredictionStruct`](@ref) in `model.biv_predictions_dict`, where the keys for the dictionary is the row number in `model.biv_profiles_df` of the corresponding profile.

## Distributed Computing Implementation

If [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is being used and `use_distributed` is `true` then the predictions from each bivariate profile will be computed in parallel across `Distributed.nworkers()` workers.

## Iteration Speed Of the Progress Meter

The time/it value is the time it takes for a prediction to be evaluated from a single point in parameter space.
"""
function generate_predictions_bivariate!(model::LikelihoodModel,
                                            t::AbstractVector,
                                            proportion_to_keep::Real=1.0;
                                            region::Real=0.95,
                                            confidence_levels::Vector{<:Float64}=Float64[],
                                            profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                            methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                            overwrite_predictions::Bool=false,
                                            show_progress::Bool=model.show_progress, 
                                            use_distributed::Bool=true)

    check_prediction_function_exists(model) || return nothing
    
    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the closed interval [0.0, 1.0]"))
    (0.0 <= region <= 1.0) || throw(DomainError("region must be in the closed interval [0.0, 1.0]"))
    sub_df = desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, Tuple{Int, Int}[], 
                                confidence_levels, profile_types, methods, for_prediction_generation=!overwrite_predictions)

    if nrow(sub_df) < 1
        return nothing
    end

    sub_df[:, :region] .= region * 1.0

    totaltasks = sum(sub_df.num_points) + 
        sum([length(model.biv_profiles_dict[row_ind].internal_points.ll) for row_ind in sub_df.row_ind])
    channel_buffer_size = min(ceil(Int, totaltasks * 0.05), 100)
    channel = RemoteChannel(() -> Channel{Bool}(channel_buffer_size))
    p = Progress(totaltasks; desc="Generating bivariate profile predictions: ",
                dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    @sync begin
        @async while take!(channel)
            next!(p)
        end

        @async begin
            if use_distributed
                predictions = @distributed (vcat) for i in 1:nrow(sub_df)
                    [generate_prediction_bivariate(model, sub_df, i,
                                                    t, proportion_to_keep, channel)]
                end

                for (i, predict_struct) in enumerate(predictions)
                    if isnothing(predict_struct); continue end
                    
                    model.biv_predictions_dict[sub_df[i, :row_ind]] = predict_struct
                    sub_df[i, :not_evaluated_predictions] = false
                end
            else
                for i in 1:nrow(sub_df)
                    predict_struct = generate_prediction_bivariate(model, sub_df, i, t, proportion_to_keep, channel)
                    if isnothing(predict_struct); continue end
                    
                    model.biv_predictions_dict[sub_df[i, :row_ind]] = predict_struct
                    sub_df[i, :not_evaluated_predictions] = false
                end
            end
            put!(channel, false)
        end
    end

    return nothing
end

"""
    generate_predictions_dim_samples!(model::LikelihoodModel,
        t::AbstractVector,
        proportion_to_keep::Real;
        <keyword arguments>)

Evalute and save `proportion_to_keep` individual predictions and their extrema from existing dimensional samples that meet the requirements of the dimensional method of [`PlaceholderLikelihood.desired_df_subset`](@ref) (see Keyword Arguments) at time points `t`. Modifies `model` in place.

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `t`: a vector of time points to compute predictions at.
- `proportion_to_keep`: a `Real` number ∈ [0.0,1.0] of the proportion of individual predictions to save. Default is 1.0.

# Keyword Arguments
- `region`: a `Real` number ∈ [0, 1] specifying the proportion of the density of the error model from which to evaluate the highest density region. Default is `0.95`.
- `confidence_levels`: a vector of confidence levels. If empty, all confidence levels of dimensional samples will be considered for evaluating predictions from. Otherwise, only confidence levels in `confidence_levels` will be considered. Default is `Float64[]` (any confidence level).
- `sample_types`: a vector of [`AbstractSampleType`](@ref) structs. If empty, all sample types used to find dimensional samples are considered. Otherwise, only samples with matching sample types will be considered. Default is `AbstractSampleType[]` (any sample type).
- `overwrite_predictions`: boolean variable specifying whether to re-evaluate and overwrite predictions for dimensional samples that have already had predictions evaluated. Set to `true` if predictions need to be evaluated for a new vector of time points. Default is `false`.
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of predictions evaluated and estimated time of completion. Default is `model.show_progress`.
- `use_distributed`: boolean variable specifying whether to use a normal for loop or a `@distributed` for loop across dimensional samples. Default is `true`.

# Details

For each dimensional sample that meets the requirements of [`PlaceholderLikelihood.desired_df_subset`](@ref), it uses [`PlaceholderLikelihood.generate_prediction`](@ref) to generates the predictions for every parameter point in the samples. The extrema of these predictions are computed (these are approximate simultaneous confidence bands for the prediction mean). The extrema and `proportion_to_keep` of the individual predictions are saved as a [`PredictionStruct`](@ref) in `model.dim_predictions_dict`, where the keys for the dictionary is the row number in `model.dim_samples_df` of the corresponding sample.

## Distributed Computing Implementation

If [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) is being used and `use_distributed` is `true` then the predictions from each dimensional sample will be computed in parallel across `Distributed.nworkers()` workers.

## Iteration Speed Of the Progress Meter

The time/it value is the time it takes for a prediction to be evaluated from a single point in parameter space.
"""
function generate_predictions_dim_samples!(model::LikelihoodModel,
                                            t::AbstractVector,
                                            proportion_to_keep::Real=1.0;
                                            region::Real=0.95,
                                            confidence_levels::Vector{<:Float64}=Float64[],
                                            sample_types::Vector{<:AbstractSampleType}=AbstractSampleType[],
                                            overwrite_predictions::Bool=false,
                                            show_progress::Bool=model.show_progress, 
                                            use_distributed::Bool=true)

    check_prediction_function_exists(model) || return nothing
    
    (0.0 <= proportion_to_keep <= 1.0) || throw(DomainError("proportion_to_keep must be in the closed interval [0.0, 1.0]"))
    (0.0 <= region <= 1.0) || throw(DomainError("region must be in the closed interval [0.0, 1.0]"))
    sub_df = desired_df_subset(model.dim_samples_df, model.num_dim_samples, confidence_levels, sample_types, 
                                for_prediction_generation=!overwrite_predictions)

    if nrow(sub_df) < 1
        return nothing
    end

    sub_df[:, :region] .= region * 1.0

    totaltasks = sum(sub_df.num_points)
    channel_buffer_size = min(ceil(Int, totaltasks * 0.05), 400)
    channel = RemoteChannel(() -> Channel{Bool}(channel_buffer_size))
    p = Progress(totaltasks; desc="Generating dimensional profile sample predictions: ",
                dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    @sync begin
        @async while take!(channel)
            next!(p)
        end

        @async begin
            if use_distributed
                predictions = @distributed (vcat) for i in 1:nrow(sub_df)
                    parameter_points = model.dim_samples_dict[sub_df[i, :row_ind]].points
                    [generate_prediction(model.core.predictfunction, model.core.errorfunction, model.core.data, t,
                                                        model.core.ymle, parameter_points, proportion_to_keep, sub_df[i,:conf_level], channel)]
                end

                for (i, predict_struct) in enumerate(predictions)
                    if isnothing(predict_struct); continue end
                    
                    model.dim_predictions_dict[sub_df[i, :row_ind]] = predict_struct
                    sub_df[i, :not_evaluated_predictions] = false
                end
            else
                for i in 1:nrow(sub_df)
                    parameter_points = model.dim_samples_dict[sub_df[i, :row_ind]].points
                    predict_struct = generate_prediction(model.core.predictfunction, model.core.errorfunction, model.core.data, t,
                                                        model.core.ymle, parameter_points, proportion_to_keep, sub_df[i,:conf_level], channel)
                    if isnothing(predict_struct); continue end

                    model.dim_predictions_dict[sub_df[i, :row_ind]] = predict_struct
                    sub_df[i, :not_evaluated_predictions] = false
                end
            end
            put!(channel, false)
        end
    end

    return nothing
end