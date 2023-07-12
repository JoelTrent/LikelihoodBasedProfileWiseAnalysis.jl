"""
    update_biv_dict_internal!(model::LikelihoodModel,
        biv_row_number::Int,
        points::PointsAndLogLikelihood)

Updates the `internal_points` field of a [`BivariateConfidenceStruct`](@ref), for the profile related to `biv_row_number` stored at `model.biv_profiles_dict[biv_row_number]`, with the internal points stored in `points`.
"""
function update_biv_dict_internal!(model::LikelihoodModel,
    biv_row_number::Int,
    points::PointsAndLogLikelihood)

    interval_struct = model.biv_profiles_dict[biv_row_number]
    model.biv_profiles_dict[biv_row_number] = @set interval_struct.internal_points = points

    return nothing
end

"""
    sample_internal_points_single_row(model::LikelihoodModel, 
        num_points::Int, 
        θindices::Tuple{Int,Int}, 
        profile_type::AbstractProfileType,
        conf_struct::BivariateConfidenceStruct, 
        confidence_level::Float64,
        boundary_not_ordered::Bool,
        hullmethod::AbstractBivariateHullMethod)

Given a `hullmethod` of type [`AbstractBivariateHullMethod`](@ref) which creates a 2D polygon hull from a set of boundary and internal points (method dependent) in `conf_struct` as a representation of the theoretical boundary, sample points from the polygon hull homogeneously until `num_points` are found, rejecting any that are not inside the log-likelihood threshold at that `confidence_level` and `profile_type`.   
"""
function sample_internal_points_single_row(model::LikelihoodModel, 
                                            num_points::Int, 
                                            θindices::Tuple{Int,Int}, 
                                            profile_type::AbstractProfileType,
                                            conf_struct::BivariateConfidenceStruct, 
                                            confidence_level::Float64,
                                            boundary_not_ordered::Bool,
                                            hullmethod::AbstractBivariateHullMethod)

    bivariate_optimiser = get_bivariate_opt_func(profile_type, RadialMLEMethod())
    biv_opt_is_ellipse_analytical = bivariate_optimiser == bivariateψ_ellipse_analytical_vectorsearch
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)
    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]

    ind1, ind2 = θindices
    newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(model, ind1, ind2)

    if biv_opt_is_ellipse_analytical
        p = (ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
            θranges=θranges, ωranges=ωranges, consistent=consistent)
    else
        p = (ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
            θranges=θranges, ωranges=ωranges, consistent=consistent, ω_opt=zeros(model.core.num_pars - 2))
    end

    mesh = construct_polygon_hull(model, [ind1, ind2], conf_struct, confidence_level,
                                    boundary_not_ordered, hullmethod, false)
    
    internal_points = zeros(model.core.num_pars, num_points)
    ll = zeros(num_points)
    i=1
    num_rejected=0
    while i ≤ num_points

        sample_points = reduce(hcat, [point.coords for point in collect(sample(mesh, HomogeneousSampling(num_points-(i-1))))])

        for j in axes(sample_points, 2)
            p.pointa .= sample_points[:,j]
            ll[i] = bivariate_optimiser(0.0, p)

            if ll[i] ≥ 0.0
                internal_points[[ind1, ind2], i] .= p.pointa
                if !biv_opt_is_ellipse_analytical
                    variablemapping!(@view(internal_points[:, i]), p.ω_opt, θranges, ωranges)
                end
                i+=1
            else
                num_rejected+=1
            end
        end
    end

    rejection_rate = num_rejected / (num_rejected+i-1)
    ll .= ll .+ get_target_loglikelihood(model, confidence_level, EllipseApproxAnalytical(), 2)

    if biv_opt_is_ellipse_analytical
        get_ωs_bivariate_ellipse_analytical!(@view(internal_points[[ind1, ind2], :]), num_points,
            consistent, ind1, ind2,
            model.core.num_pars, initGuess,
            θranges, ωranges, internal_points)
    end

    return merge(conf_struct.internal_points, PointsAndLogLikelihood(internal_points, ll)), rejection_rate
end

"""
    sample_internal_points_single_row(model::LikelihoodModel, 
        biv_row_number::Int, 
        num_points::Int, 
        hullmethod::AbstractBivariateHullMethod)

Alternate method which uses a valid row number in `model.biv_profiles_df` to specify the bivariate profile to sample internal points using.
"""
function sample_internal_points_single_row(model::LikelihoodModel, biv_row_number::Int, num_points::Int, hullmethod::AbstractBivariateHullMethod)
    θindices = model.biv_profiles_df.θindices[biv_row_number]
    profile_type = model.biv_profiles_df.profile_type[biv_row_number]
    conf_struct = model.biv_profiles_dict[biv_row_number]
    confidence_level = model.biv_profiles_df[biv_row_number, :conf_level]
    boundary_not_ordered = model.biv_profiles_df.boundary_not_ordered[biv_row_number]

    return sample_internal_points_single_row(model, num_points, θindices,
        profile_type, conf_struct, confidence_level, boundary_not_ordered, hullmethod)
end

"""
    sample_bivariate_internal_points!(model::LikelihoodModel,
        num_points::Int;
        <keyword arguments>)

Samples `num_points` internal points in interest parameter space of existing bivariate profiles that meet the requirements of the bivariate method of [`PlaceholderLikelihood.desired_df_subset`](@ref) (see Keyword Arguments). Modifies `model` in place, with sampled internal points appended to the internal points field of each [`BivariateConfidenceStruct`](@ref).

# Arguments
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `num_points`: number of internal points to sample from within a polygon hull approximation of a bivariate boundary and append to it's array of internal points.

# Keyword Arguments
- `confidence_levels`: a vector of confidence levels. If empty, all confidence levels of bivariate profiles will be considered for finding interval points. Otherwise, only confidence levels in `confidence_levels` will be considered. Default is `Float64[]` (any confidence level).
- `profile_types`: a vector of [`AbstractProfileType`](@ref) structs. If empty, all profile types of bivariate profiles are considered. Otherwise, only profiles with matching profile types will be considered. Default is `AbstractProfileType[]` (any profile type).
- `methods`: a vector of [`AbstractBivariateMethod`](@ref) structs. If empty all methods used to find bivariate profiles are considered. Otherwise, only profiles with matching method types will be considered (struct arguments do not need to be the same). Default is `AbstractBivariateMethod[]` (any bivariate method).
- `hullmethod`: method of type [`AbstractBivariateHullMethod`](@ref) used to create a 2D polygon hull that approximates the bivariate boundary from a set of boundary points and internal points (method dependent). For available methods see [`bivariate_hull_methods()`](@ref). Default is [`MPPHullMethod()`](@ref).
- `t`: vector of timepoints to evaluate predictions at for each new sampled internal point from a bivariate boundary that has already had predictions evaluated. The vector must be the same vector used to produce these previous predictions, otherwise points will not be sampled from this boundary. Default is `missing`.
- `evaluate_predictions_for_samples`: boolean variable specifying whether to evaluate predictions for sampled points given predictions evaluated have been evaluated for the boundary they were sampled from. If `false`, then existing predictions will be forgotten by the `model` and overwritten the next time predictions are evaluated for each profile internal points were sampled from. Default is `true`.
- `proportion_of_predictions_to_keep`: The proportion of predictions from `num_points` internal points to save. Does not impact the extrema calculated from predictions. Default is `1.0`.

# Details

For each bivariate profile that meets the requirements of [`PlaceholderLikelihood.desired_df_subset`](@ref) it creates a 2D polygon hull from it's set of boundary and internal points (method dependent) using `hullmethod` and samples points from the hull homogeneously until `num_points` are found, rejecting any that are not inside the log-likelihood threshold at that `confidence_level` and `profile_type`. Nuisance parameters of each point in bivariate interest parameter space are found by maximising the log-likelihood function given by the `profile_type` of the profile.

It is highly recommended to view the docstrings of each `hullmethod` as the rejection rate of sampled points and the representation accuracy / coverage of the theoretical boundary varies between them, which can impact both computational performance and sampling coverage. For example, given the same set of boundary and internal points, [`ConvexHullMethod`](@ref) will produce a polygon hull that contains at least as much of the theoretical boundary as the other methods, but may have a higher rejection rate than other methods leading to higher computational cost.
"""
function sample_bivariate_internal_points!(model::LikelihoodModel,
                                    num_points::Int;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                    hullmethod::AbstractBivariateHullMethod=MPPHullMethod(),
                                    t::Union{AbstractVector, Missing}=missing,
                                    evaluate_predictions_for_samples::Bool=true,
                                    proportion_of_predictions_to_keep::Real=1.0)

    0 < num_points || throw(DomainError("num_points must be a strictly positive integer"))
    
    sub_df = desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, Tuple{Int,Int}[], 
                confidence_levels, profile_types, methods)

    if nrow(sub_df) < 1
        return nothing
    end

    rejection_df = DataFrame(rejection_rate=zeros(nrow(sub_df)), biv_df_row_ind=zeros(Int, nrow(sub_df)))

    for i in 1:nrow(sub_df)
        row_ind = sub_df[i, :row_ind]
        if evaluate_predictions_for_samples && !sub_df[i, :not_evaluated_predictions]

            if check_prediction_function_exists(model::LikelihoodModel) === false
                continue
            end

            if ismissing(t) 
                @warn string("t must be specified to evaluate predictions for internal points (and be the same as the t vector used to generate previous predictions with generate_prediction_bivariate!). Bivariate profile row ", row_ind, " not evaluated.")
                continue
            end

            predict_struct = model.biv_predictions_dict[row_ind]
            if size(predict_struct.predictions, 1) != length(t)
                @warn string("additional samples not evaluated for the bivariate profile at row ", row_ind, " because the length of t is not the same as the length of existing predictions")
                continue
            end
        end

        internal_points, rejection_rate = sample_internal_points_single_row(model, row_ind, num_points, hullmethod)
        update_biv_dict_internal!(model, row_ind, internal_points)
        rejection_df[i, :] .= rejection_rate, row_ind

        if evaluate_predictions_for_samples && !sub_df[i, :not_evaluated_predictions]

            predict_struct = model.biv_predictions_dict[row_ind]

            new_predict_struct = generate_prediction(model.core.predictfunction,
                                                        model.core.data, t, model.core.ymle,
                                                        internal_points.points[:, (end-num_points+1):end],
                                                        proportion_of_predictions_to_keep)

            model.biv_predictions_dict[row_ind] = merge(predict_struct, new_predict_struct)
        else
            sub_df[row_ind, :not_evaluated_predictions] = true
        end
    end

    rejection_df = rejection_df[rejection_df.biv_df_row_ind .!= 0, :]
    sub_df[:, :not_evaluated_internal_points] .= false

    return rejection_df
end