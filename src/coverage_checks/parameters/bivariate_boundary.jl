"""

Tests how well the boundary polygon with a given number of points contains the theoretical boundary by testing how many valid dimensional samples from a [`AbstractSampleType`](@ref) (those within the theoretical boundary) are within the boundary polygon.



"""
function check_bivariate_parameter_coverage(data_generator::Function,
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
    method::AbstractBivariateMethod=RadialRandomMethod(3),
    sample_type::AbstractSampleType=LatinHypercubeSamples(),
    θcombinations_is_unique::Bool=false,
    coverage_estimate_confidence_level::Float64=0.95,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false)

    length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))

    N > 0 || throw(DomainError("N must be greater than 0"))

    if num_points isa Int
        num_points > 0 || throw(DomainError("num_points must be greater than 0"))
    else
        all(num_points .> 0) || throw(DomainError("all elements of num_points must be greater than 0"))
    end

    if θcombinations isa Vector{Tuple{Int, Int}}
        θcombinations = [[combo...] for combo in θcombinations]
    end

    # for each combination, enforce ind1 < ind2 and make sure only unique combinations are run
    if !θcombinations_is_unique
        sort!.(θcombinations)
        unique!.(θcombinations)
        sort!(θcombinations)
        unique!(θcombinations)

        1 ≤ first.(θcombinations)[1] && maximum(last.(θcombinations)) ≤ model.core.num_pars || throw(DomainError("θcombinations can only contain parameter indexes between 1 and the number of model parameters"))
    end
    extrema(length.(θcombinations)) == (2, 2) || throw(ArgumentError("θcombinations must only contain vectors of length 2"))

    len_θs = length(θcombinations)
    combo_to_index = Dict{Tuple{Int,Int},Int}(Tuple(combo) => index for (index, combo) in enumerate(θcombinations))

    successes = zeros(Int, len_θs)

    bivariate_optimiser = get_bivariate_opt_func(profile_type, RadialMLEMethod())
    biv_opt_is_ellipse_analytical = bivariate_optimiser == bivariateψ_ellipse_analytical_vectorsearch
    consistent = get_consistent_tuple(model, confidence_level, profile_type, 2)
    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]

    channel = RemoteChannel(() -> Channel{Bool}(1))
    progress = Progress(N; desc="Computing bivariate parameter coverage: ",
        dt=PROGRESS__METER__DT, enabled=show_progress, showspeed=true)

    if distributed_over_parameters
        for _ in 1:N
            new_data = data_generator(θtrue, generator_args)

            m_new = initialiseLikelihoodModel(model.core.loglikefunction, new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, model.core.θmagnitudes; biv_row_preallocation_size=len_θs, show_progress=false)

            bivariate_confidenceprofiles!(m_new, θcombinations, num_points;
                confidence_level=confidence_level, profile_type=profile_type, method=method, 
                θcombinations_is_unique=true, show_progress=false)

            for row_ind in 1:m_new.num_biv_profiles
                θindices = m_new.biv_profiles_df[row_ind, :θindices]

                ind1, ind2 = θindices
                pointa .= θtrue[[ind1, ind2]]
                newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(m_new, ind1, ind2)

                if biv_opt_is_ellipse_analytical
                    p = (ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                        θranges=θranges, ωranges=ωranges, consistent=consistent)
                else
                    p = (ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                        θranges=θranges, ωranges=ωranges, consistent=consistent, ω_opt=zeros(model.core.num_pars - 2))
                end

                # first check if inside ll threshold
                if bivariate_optimiser(0.0, p) < 0.0
                    continue
                end

                # check if boundary (defined as polygon with straight edges) contains θtrue[θi]
                boundary = m_new.biv_profiles_dict[row_ind].confidence_boundary[[θindices...], :]

                if m_new.biv_profiles_df[row_ind, :boundary_not_ordered]
                    minimum_perimeter_polygon!(boundary)
                end
                nodes = permutedims(boundary)
                edges = zeros(Int, num_points, 2)
                for i in 1:num_points-1; edges[i,:] .= i, i+1 end
                edges[end,:] .= num_points, 1

                θtrue_ij = θtrue[[θindices...]]

                is_inside, is_boundary = inpoly2(θtrue_ij, nodes, edges)

                if is_inside || is_boundary
                    successes[combo_to_index[θindices]] += 1
                end
            end
            next!(progress)
        end

    else
        successes_bool = SharedArray{Bool}(len_θs, N)
        successes_bool .= false
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

                    bivariate_confidenceprofiles!(m_new, θcombinations, num_points;
                        confidence_level=confidence_level, profile_type=profile_type, method=method,
                        θcombinations_is_unique=true, show_progress=false)                    

                    for row_ind in 1:m_new.num_biv_profiles
                        θindices = m_new.biv_profiles_df[row_ind, :θindices]

                        θindices = m_new.biv_profiles_df[row_ind, :θindices]

                        ind1, ind2 = θindices
                        pointa .= θtrue[[ind1, ind2]]
                        newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(m_new, ind1, ind2)

                        if biv_opt_is_ellipse_analytical
                            p = (ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                                θranges=θranges, ωranges=ωranges, consistent=consistent)
                        else
                            p = (ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, pointa=pointa, uhat=uhat,
                                θranges=θranges, ωranges=ωranges, consistent=consistent, ω_opt=zeros(model.core.num_pars - 2))
                        end

                        # first check if inside ll threshold
                        if bivariate_optimiser(0.0, p) < 0.0
                            continue
                        end

                        # check if boundary (defined as polygon with straight edges) contains θtrue[θi]
                        boundary = m_new.biv_profiles_dict[row_ind].confidence_boundary[[θindices...], :]

                        if m_new.biv_profiles_df[row_ind, :boundary_not_ordered]
                            minimum_perimeter_polygon!(boundary)
                        end
                        nodes = permutedims(boundary)
                        edges = zeros(Int, num_points, 2)
                        for j in 1:num_points-1
                            edges[j, :] .= j, j + 1
                        end
                        edges[end, :] .= num_points, 1

                        θtrue_ij = θtrue[[θindices...]]

                        is_inside, is_boundary = inpoly2(θtrue_ij, nodes, edges)

                        if is_inside || is_boundary
                            successes_bool[combo_to_index[θindices], i] = true
                        end
                    end
                    put!(channel, true); i^2
                end
                put!(channel, false)
            end
        end
        successes .= sum(successes_bool, dims=2)
    end

    coverage = successes ./ N
    conf_ints = zeros(len_θs, 2)
    for i in 1:len_θs
        conf_ints[i, :] .= HypothesisTests.confint(HypothesisTests.BinomialTest(successes[i], N), level=coverage_estimate_confidence_level)
    end

    return DataFrame(θnames=[model.core.θnames[[combo...]] for combo in θcombinations], θindices=θcombinations, coverage=coverage, coverage_lb=conf_ints[:, 1], coverage_ub=conf_ints[:, 2])
end