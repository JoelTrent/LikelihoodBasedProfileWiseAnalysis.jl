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

function sample_internal_points_single_row(bivariate_optimiser::Function, 
                                            model::LikelihoodModel, 
                                            num_points::Int, 
                                            θindices::Tuple{Int,Int}, 
                                            profile_type::AbstractProfileType,
                                            conf_struct::BivariateConfidenceStruct, 
                                            confidence_level::Float64,
                                            boundary_not_ordered::Bool)


    bivariate_optimiser = get_bivariate_opt_func(profile_type, RadialMLEMethod())
    biv_opt_is_ellipse_analytical = bivariate_optimiser == bivariateΨ_ellipse_analytical_vectorsearch
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)
    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]

    ind1, ind2 = θindices
    newLb, newUb, initGuess, θranges, λranges = init_bivariate_parameters(model, ind1, ind2)

    if biv_opt_is_ellipse_analytical
        p = (ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
            θranges=θranges, λranges=λranges, consistent=consistent)
    else
        p = (ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
            θranges=θranges, λranges=λranges, consistent=consistent, λ_opt=zeros(model.core.num_pars - 2))
    end

    boundary = conf_struct.confidence_boundary[[ind1, ind2], :]
    if boundary_not_ordered
        minimum_perimeter_polygon!(boundary)
    end
    boundary = permutedims(boundary)
    n = size(boundary, 1)
    
    mesh = SimpleMesh([(boundary[i, 1], boundary[i, 2]) for i in 1:n], [connect(tuple(1:n...))])
    
    internal_points = zeros(model.core.num_pars, num_points)
    ll = zeros(num_points)
    i=1
    while i ≤ num_points

        sample_points = reduce(hcat, [point.coords for point in collect(sample(mesh, HomogeneousSampling(num_points-(i-1))))])

        for j in axes(sample_points, 2)
            p.pointa .= sample_points[:,j]
            ll[i] = bivariate_optimiser(0.0, p)

            if ll[i] ≥ 0.0
                internal_points[[ind1, ind2], i] .= p.pointa
                if !biv_opt_is_ellipse_analytical
                    variablemapping2d!(@view(internal_points[:, i]), p.λ_opt, θranges, λranges)
                end
                i+=1
            end
        end
    end

    ll .= ll .+ get_target_loglikelihood(model, confidence_level, EllipseApproxAnalytical(), 2)

    if biv_opt_is_ellipse_analytical
        get_λs_bivariate_ellipse_analytical!(@view(internal_points[[ind1, ind2], :]), num_points,
            consistent, ind1, ind2,
            model.core.num_pars, initGuess,
            θranges, λranges, internal_points)
    end

    return merge(conf_struct.internal_points, PointsAndLogLikelihood(internal_points, ll))
end

function sample_internal_points_single_row(model::LikelihoodModel, biv_row_number::Int, num_points::Int)
    θindices = model.biv_profiles_df.θindices[biv_row_number]
    profile_type = model.biv_profiles_df.profile_type[biv_row_number]
    method = model.biv_profiles_df.method[biv_row_number]
    bivariate_optimiser = get_bivariate_opt_func(profile_type, method)
    conf_struct = model.biv_profiles_dict[biv_row_number]
    confidence_level = model.biv_profiles_df[biv_row_number, :conf_level]
    boundary_not_ordered = model.biv_profiles_df.boundary_not_ordered[biv_row_number]

    return sample_internal_points_single_row(bivariate_optimiser, model, num_points, θindices,
        profile_type, conf_struct, confidence_level, boundary_not_ordered)
end

"""
    sample_bivariate_internal_points!(model::LikelihoodModel,
        num_points::Int;
        confidence_levels::Vector{<:Float64}=Float64[],
        profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
        methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
        t::Union{Vector, Missing}=missing,
        evaluate_predictions_for_samples::Bool=true,
        proportion_of_predictions_to_keep::Real=1.0)


"""
function sample_bivariate_internal_points!(model::LikelihoodModel,
                                    num_points::Int;
                                    confidence_levels::Vector{<:Float64}=Float64[],
                                    profile_types::Vector{<:AbstractProfileType}=AbstractProfileType[],
                                    methods::Vector{<:AbstractBivariateMethod}=AbstractBivariateMethod[],
                                    t::Union{Vector, Missing}=missing,
                                    evaluate_predictions_for_samples::Bool=true,
                                    proportion_of_predictions_to_keep::Real=1.0)

    0 < num_points || throw(DomainError("num_points must be a strictly positive integer"))
    
    sub_df = desired_df_subset(model.biv_profiles_df, model.num_biv_profiles, Tuple{Int,Int}[], 
                confidence_levels, profile_types, methods)

    if nrow(sub_df) < 1
        return nothing
    end

    for i in 1:nrow(sub_df)

        row_ind = sub_df[i, :row_ind]
        if evaluate_predictions_for_samples && !sub_df[i, :not_evaluated_predictions]

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

        internal_points = sample_internal_points_single_row(model, row_ind, num_points)
        update_biv_dict_internal!(model, row_ind, internal_points)

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

    sub_df[:, :not_evaluated_internal_points] .= false

    return nothing
end