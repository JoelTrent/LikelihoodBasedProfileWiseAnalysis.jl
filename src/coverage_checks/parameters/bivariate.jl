"""
    check_bivariate_parameter_coverage(data_generator::Function,
        generator_args::Union{Tuple,NamedTuple},
        model::LikelihoodModel,
        N::Int,
        num_points::Union{Int, Vector{<:Int}},
        θtrue::AbstractVector{<:Real},
        θcombinations::Union{Vector{Vector{Int}}, Vector{Tuple{Int,Int}}},
        θinitialguess::AbstractVector{<:Real}=θtrue; 
        <keyword arguments>)

Performs a simulation to estimate the coverage of bivariate confidence boundaries for two-way sets of interest parameters in `θcombinations` given a model by: 
1. Repeatedly drawing new observed data using `data_generator` for fixed true parameter values, θtrue and fitting the model. 
2. Testing if each of the true bivariate interest parameters, given nuisance parameters, have log-likelihood values within the confidence threshold. 
3. If these pass then bivariate confidence boundaries of `num_points` are found using `method` and [`MPPHullMethod`](@ref) is used to construct 2D polygon hulls of the boundary points. 
4. Finally, testing if the boundary polygons contain the true bivariate parameter values in `θtrue`. The estimated coverage is returned with a default 95% confidence interval within a DataFrame. 

# Arguments
- `data_generator`: a function with two arguments which generates data for fixed time points and true model parameters corresponding to the log-likelihood function contained in `model`. The two arguments must be the vector of true model parameters, `θtrue`, and a Tuple or NamedTuple, `generator_args`. Outputs a `data` Tuple or NamedTuple that corresponds to the log-likelihood function contained in `model`.
- `generator_args`: a Tuple or NamedTuple containing any additional information required by both the log-likelihood function and `data_generator`, such as the time points to be evaluated at. If evaluating the log-likelihood function requires more than just the simulated data, arguments for the `data` output of `data_generator` should be passed in via `generator_args`. 
- `model`: a [`LikelihoodModel`](@ref) containing model information, saved profiles and predictions.
- `N`: a positive number of coverage simulations.
- `num_points`: positive number of points to find on the boundary at the specified confidence level using a single `method`. Or a vector of positive numbers of boundary points to find for each method in `method` (if `method` is a vector of [`AbstractBivariateMethod`](@ref)). Set to at least 3 within the function as some methods need at least three points to work. 
- `θtrue`: a vector of true parameters values of the model for simulating data with. 
- `θcombinations`: a vector of pairs of parameters to profile, as a vector of vectors of model parameter indexes.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point in each iteration of the simulation. Default is `θtrue`.

# Keyword Arguments
- `confidence_level`: a number ∈ (0.0, 1.0) for the confidence level to evaluate the confidence interval coverage at. Default is `0.95` (95%).
- `profile_type`: whether to use the true log-likelihood function or an ellipse approximation of the log-likelihood function centred at the MLE (with optional use of parameter bounds). Available profile types are [`LogLikelihood`](@ref), [`EllipseApprox`](@ref) and [`EllipseApproxAnalytical`](@ref). Default is `LogLikelihood()` ([`LogLikelihood`](@ref)).
- `method`: a method of type [`AbstractBivariateMethod`](@ref) or a vector of methods of type [`AbstractBivariateMethod`](@ref) (if so `num_points` needs to be a vector of the same length). For a list of available methods use `bivariate_methods()` ([`bivariate_methods`](@ref)). Default is `RadialRandomMethod(3)` ([`RadialRandomMethod`](@ref)).
- `coverage_estimate_confidence_level`: a number ∈ (0.0, 1.0) for the level of a confidence interval of the estimated coverage. Default is `0.95` (95%).
- `show_progress`: boolean variable specifying whether to display progress bars on the percentage of simulation iterations completed and estimated time of completion. Default is `model.show_progress`.
- `distributed_over_parameters`: boolean variable specifying whether to distribute the workload of the simulation across simulation iterations (false) or across the individual bivariate boundary calculations within each iteration (true). Default is `false`.

# Details

This simulated coverage check is used to estimate the performance of bivariate parameter confidence boundaries. The simulation uses [Distributed.jl](https://docs.julialang.org/en/v1/stdlib/Distributed/) to parallelise the workload.

For a 95% confidence boundary of a pair of interest parameters `[θi, θj]` it is expected that under repeated experiments from an underlying true model (data generation) which are used to construct a 2D confidence boundary for `[θi, θj]`, 95% of the true boundaries, would contain the true value `[θi, θj]`. In the simulation where the values of the true parameters, `θtrue`, are known, this is equivalent to whether the minimum perimeter polygon of the 2d boundary points for `[θi, θj]` AND the true confidence boundary contains the value `θtrue[[θi, θj]]`.

All of the methods for constructing an approximation of the 2D boundary using [`bivariate_confidenceprofiles!`](@ref) will approach an exact representation of the 2D 95% confidence boundary, assuming bounds are not in the way, as the number of boundary points approaches infinity. Resultantly, for lower numbers of boundary points the polygon representation of the boundary will be an approximation, with straight edges that do not exactly represent the true boundary. This is why the coverage check also checks if a point is inside the true boundary, as the polygon approximation might be right by accident. This is the same logic [`sample_bivariate_internal_points!`] uses to find additional internal points within a boundary polygon.

For estimates of how well the methods approximate the true 2D boundary after turning their boundary points into a polygon hull using a [`AbstractBivariateHullMethod`](@ref), [`check_bivariate_boundary_coverage`](@ref) can be used. 

The uncertainty in estimates of the coverage under the simulated model will decrease as the number of simulations, `N`, is increased. Confidence intervals for the coverage estimate are provided to quantify this uncertainty. The confidence interval for the estimated coverage is a Clopper-Pearson interval on a binomial test generated using [HypothesisTests.jl](https://juliastats.org/HypothesisTests.jl/stable/).

!!! note "Recommended setting for distributed_over_parameters"
    - If the number of processes available to use is significantly greater than the number of model parameters or only a few pairs of model parameters are being checked for coverage, `false` is recommended.   
    - If system memory or model size in system memory is a concern, or the number of processes available is similar or less than the number of pairs of model parameters being checked, `true` will likely be more appropriate. 
    - When set to `false`, a separate [`LikelihoodModel`](@ref) struct will be used by each process, as opposed to only one when set to `true`, which could cause a memory issue for larger models. 

!!! danger "May not work correctly on bimodal confidence boundaries"
    The current implementation constructs a single polygon with minimum polygon perimeter from the set of boundary points as the confidence boundary. If there are multiple distinct boundaries represented, then there will be edges connecting the distinct boundaries which the true parameter might be inside (but not inside either of the distinct boundaries). 
"""
function check_bivariate_parameter_coverage(data_generator::Function,
    generator_args::Union{Tuple,NamedTuple},
    model::LikelihoodModel,
    N::Int,
    num_points::Union{Int, Vector{<:Int}},
    θtrue::AbstractVector{<:Real},
    θcombinations::Union{Vector{Vector{Int}}, Vector{Tuple{Int,Int}}},
    θinitialguess::AbstractVector{<:Real}=θtrue;
    confidence_level::Float64=0.95,
    profile_type::AbstractProfileType=LogLikelihood(),
    method::Union{AbstractBivariateMethod, Vector{<:AbstractBivariateMethod}}=RadialRandomMethod(3),
    coverage_estimate_confidence_level::Float64=0.95,
    show_progress::Bool=model.show_progress,
    distributed_over_parameters::Bool=false)

    function argument_handling!()
        length(θtrue) == model.core.num_pars || throw(ArgumentError("θtrue must have the same length as the number of model parameters"))
        length(θinitialguess) == model.core.num_pars || throw(ArgumentError("θinitialguess must have the same length as the number of model parameters"))

        (0.0 < coverage_estimate_confidence_level && coverage_estimate_confidence_level < 1.0) || throw(DomainError("coverage_estimate_confidence_level must be in the open interval (0,1)"))
        get_target_loglikelihood(model, confidence_level, profile_type, 2)
        
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
        return nothing
    end
    local combine_methods::Bool
    argument_handling!()

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

            m_new = initialise_LikelihoodModel(model.core.loglikefunction, new_data, model.core.θnames, θinitialguess, model.core.θlb, model.core.θub, model.core.θmagnitudes; biv_row_preallocation_size=len_θs, show_progress=false)

            if combine_methods
                for (j, methodj) in enumerate(method) 
                    bivariate_confidenceprofiles!(m_new, θcombinations, num_points[j];
                        confidence_level=confidence_level, profile_type=profile_type, method=methodj,
                        show_progress=false, use_threads=false)
                end
                combine_bivariate_boundaries!(m_new, confidence_level=confidence_level, 
                    not_evaluated_predictions=true)
            else
                bivariate_confidenceprofiles!(m_new, θcombinations, num_points;
                    confidence_level=confidence_level, profile_type=profile_type, method=method, 
                    show_progress=false, use_threads=false)
            end

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
                conf_struct = m_new.biv_profiles_dict[row_ind]
                boundary = construct_polygon_hull(m_new, [θindices...], conf_struct, confidence_level,
                    m_new.biv_profiles_df[row_ind, :boundary_not_ordered], MPPHullMethod(), true)
                
                nodes = permutedims(boundary)
                num_nodes = sum(num_points)
                edges = zeros(Int, num_nodes, 2)
                for i in 1:num_nodes-1; edges[i,:] .= i, i+1 end
                edges[end,:] .= num_nodes, 1

                θtrue_ij = θtrue[[θindices...]]

                is_inside, is_boundary = inpoly2(θtrue_ij, nodes, edges)

                if is_inside || is_boundary
                    successes[combo_to_index[θindices]] += 1
                end
            end
            next!(progress)
        end

    else
        successes_bool = SharedArray{Bool}(zeros(Bool, len_θs, N))
        @sync begin
            @async while take!(channel)
                next!(progress)
            end

            @async begin
                @distributed (+) for i in 1:N
                    new_data = data_generator(θtrue, generator_args)

                    m_new = initialise_LikelihoodModel(model.core.loglikefunction, new_data,
                        model.core.θnames, θinitialguess, model.core.θlb, model.core.θub,
                        model.core.θmagnitudes; uni_row_prealloaction_size=len_θs, show_progress=false)

                    if combine_methods
                        for (j, methodj) in enumerate(method)
                            bivariate_confidenceprofiles!(m_new, θcombinations, num_points[j];
                                confidence_level=confidence_level, profile_type=profile_type, method=methodj,
                                show_progress=false, use_distributed=false, use_threads=false)
                        end
                        combine_bivariate_boundaries!(m_new, confidence_level=confidence_level,
                            not_evaluated_predictions=true)
                    else
                        bivariate_confidenceprofiles!(m_new, θcombinations, num_points;
                            confidence_level=confidence_level, profile_type=profile_type, method=method,
                            show_progress=false, use_distributed=false, use_threads=false)
                    end

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
                        conf_struct = m_new.biv_profiles_dict[row_ind]
                        boundary = construct_polygon_hull(m_new, [θindices...], conf_struct, confidence_level,
                            m_new.biv_profiles_df[row_ind, :boundary_not_ordered], MPPHullMethod(), true)
                        
                        nodes = permutedims(boundary)
                        num_nodes = sum(num_points)
                        edges = zeros(Int, num_nodes, 2)
                        for j in 1:num_nodes-1; edges[j,:] .= j, j+1 end
                        edges[end,:] .= num_nodes, 1

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

    return DataFrame(θnames=[model.core.θnames[[combo...]] for combo in θcombinations], θindices=θcombinations, coverage=coverage, coverage_lb=conf_ints[:, 1], coverage_ub=conf_ints[:, 2], num_boundary_points=fill(sum(num_points), len_θs))
end