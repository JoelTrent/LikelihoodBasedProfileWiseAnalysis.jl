"""

Returns simulation quantile intervals for the mean coverage of the theoretical boundary by a set of boundary points that are turned into a polygon using `hullmethod`.

Tests how well the boundary polygon with a given number of points contains the theoretical boundary by testing how many valid dimensional samples from a [`AbstractSampleType`](@ref) (those within the theoretical boundary) are within the boundary polygon.



"""
function check_bivariate_boundary_coverage(data_generator::Function,
    generator_args::Union{Tuple,NamedTuple},
    model::LikelihoodModel,
    N::Int,
    num_points::Union{Int, Vector{<:Int}},
    num_points_to_sample::Union{Int, Vector{<:Int}},
    θtrue::AbstractVector{<:Real},
    θcombinations::Union{Vector{Vector{Int}}, Vector{Tuple{Int,Int}}},
    θinitialguess::AbstractVector{<:Real}=θtrue;
    confidence_level::Float64=0.95,
    profile_type::AbstractProfileType=LogLikelihood(),
    method::Union{AbstractBivariateMethod, Vector{<:AbstractBivariateMethod}}=RadialRandomMethod(3),
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    hullmethod::AbstractBivariateHullMethod=MPPHullMethod(),
    coverage_estimate_quantile_level::Float64=0.95,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false)

    length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))
    length(θinitialguess) == model.core.num_pars || throw(ArgumentError("θinitialguess must have the same length as the number of model parameters"))

    (0.0 < coverage_estimate_quantile_level && coverage_estimate_quantile_level < 1.0) || throw(DomainError("coverage_estimate_quantile_level must be in the open interval (0,1)"))
    get_target_loglikelihood(model, confidence_level, profile_type, 2)

    if num_points_to_sample isa Int
        num_points_to_sample > 0 || throw(DomainError("num_points_to_sample must be a strictly positive integer"))
    else
        minimum(num_points_to_sample) > 0 || throw(DomainError("num_points_to_sample must contain strictly positive integers"))

        sample_type isa UniformGridSamples || throw(ArgumentError(string("num_points_to_sample must be an integer for ", sample_type, " sample_type")))

        (length(num_points_to_sample) == length(θcombinations[1]) &&
         diff([extrema(length.(θcombinations))...])[1] == 0) ||
            throw(ArgumentError("num_points_to_sample must have the same length as each vector of interest parameters in num_points_to_sample"))
    end

    !xor(num_points isa Vector, method isa Vector) || throw(ArgumentError("num_points and method must both be a Vector, or both be a Int and AbstractBivariateMethod, respectively, at the same time (xnor gate)"))
    combine_methods = num_points isa Vector
    if combine_methods
        (length(num_points) == length(method)) || throw(ArgumentError("num_points must have the same length as method, each index in num_points corresponds to the number of boundary points for the corresponding index in method"))
    end

    N > 0 || throw(DomainError("N must be greater than 0"))

    if θcombinations isa Vector{Tuple{Int, Int}}
        θcombinations = [[combo...] for combo in θcombinations]
    end

    # for each combination, enforce ind1 < ind2 and make sure only unique combinations are run
    sort!.(θcombinations)
    unique!.(θcombinations)
    sort!(θcombinations)
    unique!(θcombinations)

    1 ≤ first.(θcombinations)[1] && maximum(last.(θcombinations)) ≤ model.core.num_pars || throw(DomainError("θcombinations can only contain parameter indexes between 1 and the number of model parameters"))
    extrema(length.(θcombinations)) == (2, 2) || throw(ArgumentError("θcombinations must only contain vectors of length 2"))

    len_θs = length(θcombinations)
    combo_to_index = Dict{Tuple{Int,Int},Int}(Tuple(combo) => index for (index, combo) in enumerate(θcombinations))

    coverage = zeros(len_θs, N)

    bivariate_optimiser = get_bivariate_opt_func(profile_type, RadialMLEMethod())
    biv_opt_is_ellipse_analytical = bivariate_optimiser == bivariateψ_ellipse_analytical_vectorsearch
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)
    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]
    min_num_dim_points = 20

    channel = RemoteChannel(() -> Channel{Bool}(3))
    progress = Progress(N; desc="Computing bivariate boundary coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for i in 1:N
            new_data = data_generator(θtrue, generator_args)

            m_new = initialiseLikelihoodModel(model.core.loglikefunction, new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, model.core.θmagnitudes; biv_row_preallocation_size=len_θs, show_progress=false)

            dimensional_likelihood_samples!(m_new, θcombinations, num_points_to_sample;
                confidence_level=confidence_level, sample_type=sample_type, 
                show_progress=false)

            if combine_methods
                for (j, methodj) in enumerate(method)
                    bivariate_confidenceprofiles!(m_new, θcombinations, num_points[j];
                        confidence_level=confidence_level, profile_type=profile_type, method=methodj,
                        show_progress=false)
                end
                combine_bivariate_boundaries!(m_new, confidence_level=confidence_level,
                    not_evaluated_predictions=true)
            else
                bivariate_confidenceprofiles!(m_new, θcombinations, num_points;
                    confidence_level=confidence_level, profile_type=profile_type, method=method,
                    show_progress=false)
            end

            for row_ind in 1:m_new.num_biv_profiles
                θindices_tuple = m_new.biv_profiles_df[row_ind, :θindices]
                θindices = [θindices_tuple...]

                conf_struct = m_new.biv_profiles_dict[row_ind]
                polygonhull = construct_polygon_hull(m_new, θindices, conf_struct, confidence_level,
                    m_new.biv_profiles_df[row_ind, :boundary_not_ordered], hullmethod, true)

                polygonhull_num_points = size(polygonhull, 2)
                nodes = permutedims(polygonhull)
                edges = zeros(Int, polygonhull_num_points, 2)
                for j in 1:polygonhull_num_points-1; edges[j,:] .= j, j+1 end
                edges[end,:] .= polygonhull_num_points, 1

                dim_samples_row = m_new.dim_samples_row_exists[(θindices, sample_type)][confidence_level]
                if dim_samples_row == 0; continue end

                num_dim_points = m_new.dim_samples_df[dim_samples_row, :num_points]
                if num_dim_points < min_num_dim_points; continue end

                dimensional_samples = m_new.dim_samples_dict[dim_samples_row].points[θindices, :]

                count=0
                for j in axes(dimensional_samples, 2)
                    is_inside, is_boundary = inpoly2(dimensional_samples[:,j], nodes, edges)

                    if is_inside || is_boundary
                        count += 1
                    end
                end
                coverage[combo_to_index[θindices_tuple], i] = count / num_dim_points
            end
            next!(progress)
        end

    else
        coverage_shared = SharedArray(zeros(len_θs, N))
        @sync begin
            # this task prints the progress bar
            @async while take!(channel)
                next!(progress)
            end

            # this task does the computation
            @async begin
                @distributed (+) for i in 1:N
                    new_data = data_generator(θtrue, generator_args)

                    m_new = initialiseLikelihoodModel(model.core.loglikefunction, new_data,
                        model.core.θnames, θinitialguess, model.core.θlb, model.core.θub,
                        model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false)

                    dimensional_likelihood_samples!(m_new, θcombinations, num_points_to_sample;
                        confidence_level=confidence_level, sample_type=sample_type,
                        show_progress=false)
                    
                    if combine_methods
                        for (j, methodj) in enumerate(method)
                            bivariate_confidenceprofiles!(m_new, θcombinations, num_points[j];
                                confidence_level=confidence_level, profile_type=profile_type, method=methodj,
                                show_progress=false)
                        end
                        combine_bivariate_boundaries!(m_new, confidence_level=confidence_level,
                            not_evaluated_predictions=true)
                    else
                        bivariate_confidenceprofiles!(m_new, θcombinations, num_points;
                            confidence_level=confidence_level, profile_type=profile_type, method=method,
                            show_progress=false)
                    end

                    for row_ind in 1:m_new.num_biv_profiles
                        θindices_tuple = m_new.biv_profiles_df[row_ind, :θindices]
                        θindices = [θindices_tuple...]

                        conf_struct = m_new.biv_profiles_dict[row_ind]
                        polygonhull = construct_polygon_hull(m_new, θindices, conf_struct, confidence_level,
                            m_new.biv_profiles_df[row_ind, :boundary_not_ordered], hullmethod, true)

                        polygonhull_num_points = size(polygonhull, 2)
                        nodes = permutedims(polygonhull)
                        edges = zeros(Int, polygonhull_num_points, 2)
                        for j in 1:polygonhull_num_points-1; edges[j,:] .= j, j+1 end
                        edges[end,:] .= polygonhull_num_points, 1

                        dim_samples_row = m_new.dim_samples_row_exists[(θindices, sample_type)][confidence_level]
                        if dim_samples_row == 0; continue end

                        num_dim_points = m_new.dim_samples_df[dim_samples_row, :num_points]
                        if num_dim_points < min_num_dim_points; continue end
                        
                        dimensional_samples = m_new.dim_samples_dict[dim_samples_row].points[θindices, :]

                        count = 0
                        for j in axes(dimensional_samples, 2)
                            is_inside, is_boundary = inpoly2(dimensional_samples[:,j], nodes, edges)

                            if is_inside || is_boundary
                                count += 1
                            end
                        end
                        coverage_shared[combo_to_index[θindices_tuple], i] = count / size(dimensional_samples, 2)
                    end
                    put!(channel, true); i^2
                end
                put!(channel, false)
            end
        end
        coverage .= coverage_shared
    end

    quantile_intervals = zeros(len_θs, 2)
    lower_quantile = (1.0-coverage_estimate_quantile_level) / 2.0 
    for i in 1:len_θs
        quantile_intervals[i, :] .= quantile(coverage[i, coverage[i,:] .!= 0.0], [lower_quantile, 1 - lower_quantile])
    end

    coverage_mean = [mean(coverage[i, coverage[i,:] .!= 0.0]) for i in 1:len_θs]

    return DataFrame(θnames=[model.core.θnames[[combo...]] for combo in θcombinations],
                        θindices=θcombinations, coverage=coverage_mean, 
                        coverage_lb=quantile_intervals[:, 1], coverage_ub=quantile_intervals[:, 2],
                        num_boundary_points=fill(sum(num_points), len_θs))
end