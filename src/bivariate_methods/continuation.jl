"""
    update_targetll!(p::NamedTuple, target_confidence_ll::Float64)

Updates `p.targetll` to `target_confidence_level`, returning p. 
"""
function update_targetll!(p::NamedTuple, target_confidence_ll::Float64)
    p = @set p.targetll=target_confidence_ll
    return p
end

"""
    normal_vector_i_2d!(gradient_i, index, points)

First order finite difference approximation of the normal vector at a point, using the location of the point on each side of it to construct a line and find the vector normal to the line, saving these in place in `gradient_i`.
"""
function normal_vector_i_2d!(gradient_i, index, points)
    if index == 1
        gradient_i .= [(points[2,2]-points[2,end]), -(points[1,2]-points[1,end])]
    elseif index == size(points, 2)
        gradient_i .= [(points[2,1]-points[2,end-1]), -(points[1,1]-points[1,end-1])]
    else
        gradient_i .= [(points[2,index+1]-points[2,index-1]), -(points[1,index+1]-points[1,index-1])]
    end
    return nothing
end

"""
    continuation_line_search!(p::NamedTuple, 
        point_is_on_bounds::BitVector,
        bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int,
        ind1::Int, 
        ind2::Int,
        target_confidence_ll::Float64, 
        search_directions::Matrix{Float64},
        start_level_set_2D::Matrix{Float64}, 
        find_zero_atol::Real,
        channel::RemoteChannel;
        start_level_set_all::Matrix{Float64}=zeros(0,0),
        level_set_not_smoothed::Bool=true,
        is_a_zero::BitVector=falses(num_points))

Implementation of the outward 'continuation' search of [`ContinuationMethod`](@ref) for getting from a lower log-likelihood threshold level set to a higher level set. 

`p`, `point_is_on_bounds` and `search_directions` are mutated by this function.
"""
function continuation_line_search!(p::NamedTuple, 
                                    point_is_on_bounds::BitVector,
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int,
                                    ind1::Int, 
                                    ind2::Int,
                                    target_confidence_ll::Float64, 
                                    search_directions::Matrix{Float64},
                                    start_level_set_2D::Matrix{Float64}, 
                                    find_zero_atol::Real,
                                    channel::RemoteChannel;
                                    start_level_set_all::Matrix{Float64}=zeros(0,0),
                                    level_set_not_smoothed::Bool=true,
                                    is_a_zero::BitVector=falses(num_points))
    
    start_have_all_pars = !isempty(start_level_set_all) 

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateψ_ellipse_analytical_continuation
    target_level_set_2D = zeros(2, num_points)
    target_level_set_all = zeros(model.core.num_pars, num_points)
    
    gradient_i = [0.0,0.0]
    # normal_i = [0.0,0.0]
    # tangent_amount=1.0
    boundpoint = [0.0,0.0]
    boundarypoint = [0.0,0.0]
    p = update_targetll!(p, target_confidence_ll)
    
    # normal_scaling = diff([extrema(start_level_set_2D[1,:])...])[1]^2 / diff([extrema(start_level_set_2D[2,:])...])[1]^2
    # println(normal_scaling)
    if isnan(model.core.θmagnitudes[ind1]) || isnan(model.core.θmagnitudes[ind2]) 
        normal_scaling = 1.0
    else
        normal_scaling = model.core.θmagnitudes[ind1]/model.core.θmagnitudes[ind2]
    end

    if !level_set_not_smoothed
        boundsmapping!(p.q.initGuess, model.core.θmle, ind1, ind2)
    end

    for i in 1:num_points
        if point_is_on_bounds[i]
            target_level_set_2D[:, i] .= start_level_set_2D[:, i]
            target_level_set_all[:, i] .= start_level_set_all[:, i]
            continue
        end
        
        # if know the optimised values of nuisance parameters at a given start point,
        # pass them to the optimiser
        if start_have_all_pars && level_set_not_smoothed
            boundsmapping!(p.q.initGuess, @view(start_level_set_all[:,i]), ind1, ind2)
        end
        
        if is_a_zero[i]
            p.pointa .= start_level_set_2D[:,i]
            bivariate_optimiser(0.0, p) # to extract nuisance parameter values
            boundarypoint .= start_level_set_2D[:,i] .* 1.0
            target_level_set_2D[:, i] .= boundarypoint
            target_level_set_all[[ind1, ind2], i] .= boundarypoint
            if !biv_opt_is_ellipse_analytical
                variablemapping!(@view(target_level_set_all[:, i]), p.ω_opt, p.q.θranges, p.q.ωranges)
            end
            continue
        end

        # calculate gradient at point i; want to go in downhill direction
        # FORWARDDIFF NOT WORKING WITH ANON FUNCTION JUST YET - likely because it is contains a mutating function, i.e. zeros()
        # 
        # gradient_i .= -ForwardDiff.gradient(f_gradient, p.pointa)

        # normal_vector_i_2d!(normal_i, i, start_level_set_2D)
        # normal_i[1] = normal_i[1] * normal_scaling
        # normal_i .= (normal_i ./ norm(normal_i, 2)) 
        # gradient_i .= tangent_amount*(search_directions[:,i] ./ norm(search_directions[:,i], 2)) + (1-tangent_amount)*normal_i

        # p.uhat .= (gradient_i ./ (norm(gradient_i, 2))) 

        boundpoint .= findpointonbounds(model, start_level_set_2D[:,i], search_directions[:,i], ind1, ind2)



        # gradient_i .= boundpoint .- p.pointa
        # v_bar_norm = norm(gradient_i, 2)
        # p.uhat .= gradient_i ./ v_bar_norm

        p.pointa .= boundpoint
        gradient_i .= start_level_set_2D[:,i] .- boundpoint
        v_bar_norm = norm(gradient_i, 2)
        p.uhat .= gradient_i ./ v_bar_norm


        # if bound point and pointa bracket a boundary, search for the boundary
        # otherwise, the bound point is used as the level set boundary (i.e. it's inside the true level set boundary)
        g = bivariate_optimiser(0.0, p) 
        if biv_opt_is_ellipse_analytical || g < 0
            
            ψ = solve(ZeroProblem(bivariate_optimiser, v_bar_norm), Roots.Order8(); atol=find_zero_atol, p=p)

            # in event Roots.Order8 fails to converge, switch to bracketing method
            if isnan(ψ) || isinf(ψ) || ψ < 0.0
                lb = isinf(g) ? 1e-8 * v_bar_norm : 0.0
                # value of v_bar_norm that satisfies the equation boundpoint = p.pointa + ψ*p.uhat
                ψ = find_zero(bivariate_optimiser, (lb, v_bar_norm), Roots.Brent(); atol=find_zero_atol, p=p)
            end

            boundarypoint .= p.pointa + ψ*p.uhat
            target_level_set_2D[:, i] .= boundarypoint
            target_level_set_all[[ind1, ind2], i] .= boundarypoint
            if !biv_opt_is_ellipse_analytical; bivariate_optimiser(ψ, p) end
        else
            point_is_on_bounds[i] = true
            target_level_set_2D[:, i] .= boundpoint
            target_level_set_all[[ind1, ind2], i] .= boundpoint
        end
        if !biv_opt_is_ellipse_analytical
            variablemapping!(@view(target_level_set_all[:, i]), p.ω_opt, p.q.θranges, p.q.ωranges)
        end
        put!(channel, true)
    end

    if biv_opt_is_ellipse_analytical
        local initGuess = zeros(model.core.num_pars-2)
        boundsmapping!(initGuess, model.core.θmle, ind1, ind2)
        target_level_set_all = get_ωs_bivariate_ellipse_analytical!(target_level_set_2D, num_points,
                                                                    p.q.consistent, ind1, ind2, 
                                                                    model.core.num_pars, initGuess,
                                                                    p.q.θranges, p.q.ωranges, 
                                                                    p.options, false, target_level_set_all)
    end


    # search_directions .= target_level_set_2D - start_level_set_2D
    # α = 0.2

    # search_directions[:,1] .= (target_level_set_2D[:,1] - start_level_set_2D[:,1]) + 
    #                                     α .* ((target_level_set_2D[:,1] - start_level_set_2D[:, end]) + 
    #                                     (target_level_set_2D[:,1] - start_level_set_2D[:, 2]))

    # search_directions[:,end] .= (target_level_set_2D[:,end] - start_level_set_2D[:,end]) + 
    #                                     α .* ((target_level_set_2D[:,end] - start_level_set_2D[:, 1]) + 
    #                                     (target_level_set_2D[:,end] - start_level_set_2D[:, end-1]))

    # for i in 2:(num_points-1)
    #     search_directions[:,i] .= (target_level_set_2D[:,i] - start_level_set_2D[:,i]) + 
    #                                     α .* ((target_level_set_2D[:,i] - start_level_set_2D[:, i-1]) + 
    #                                     (target_level_set_2D[:,i] - start_level_set_2D[:, i+1]))
    # end

    # search_directions[:, 1] .= @view(target_level_set_2D[:,1]) .- @view(start_level_set_2D[:, end])
    # search_directions[:, 2:end] .= @view(target_level_set_2D[:,2:end]) .- @view(start_level_set_2D[:, 1:end-1])

    return target_level_set_2D, target_level_set_all
end

"""
    continuation_inwards_radial_search!(p::NamedTuple, 
        bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        ind1::Int, 
        ind2::Int, 
        target_confidence_ll::Float64,
        search_directions::Matrix{Float64},
        start_level_set_2D::Matrix{Float64},
        is_a_zero::BitVector,
        find_zero_atol::Real,
        channel::RemoteChannel)

Implementation of the inwards radial search for an initial level set at `target_confidence_ll` given an initial ellipse solution for [`ContinuationMethod`](@ref). The `search_directions` for each point is a vector between the maximum likelihood estimate point in interest parameter space and the ellipse solution.
"""
function continuation_inwards_radial_search!(p::NamedTuple, 
                                                bivariate_optimiser::Function, 
                                                model::LikelihoodModel, 
                                                num_points::Int, 
                                                ind1::Int, 
                                                ind2::Int, 
                                                target_confidence_ll::Float64,
                                                search_directions::Matrix{Float64},
                                                start_level_set_2D::Matrix{Float64},
                                                is_a_zero::BitVector,
                                                find_zero_atol::Real,
                                                channel::RemoteChannel)

    mle_point = model.core.θmle[[ind1, ind2]]
    
    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateψ_ellipse_analytical_continuation
    target_level_set_2D = zeros(2, num_points)
    target_level_set_all = zeros(model.core.num_pars, num_points)
    
    boundarypoint = [0.0, 0.0]
    p = update_targetll!(p, target_confidence_ll)

    p.pointa .= mle_point
    # effectively equivalent code to vector search code 
    for i in 1:num_points
        v_bar = search_directions[:,i] # start_level_set_2D[:,i] .- mle_point

        v_bar_norm = norm(v_bar, 2)
        p.uhat .= v_bar ./ v_bar_norm

        if is_a_zero[i]
            ψ = v_bar_norm # to extract nuisance parameter values
        else
            ψ = find_zero(bivariate_optimiser, (0.0, v_bar_norm), Roots.Brent(); atol=find_zero_atol, p=p)
        end

        boundarypoint .= p.pointa + ψ*p.uhat
        target_level_set_2D[:, i] .= boundarypoint
        target_level_set_all[[ind1, ind2], i] .= boundarypoint
        if !biv_opt_is_ellipse_analytical
            bivariate_optimiser(ψ, p)
            variablemapping!(@view(target_level_set_all[:, i]), p.ω_opt, p.q.θranges, p.q.ωranges)
        end
        put!(channel, true)
    end

    if biv_opt_is_ellipse_analytical
        local initGuess = zeros(model.core.num_pars-2)
        boundsmapping!(initGuess, model.core.θmle, ind1, ind2)
        target_level_set_all = get_ωs_bivariate_ellipse_analytical!(target_level_set_2D, num_points,
                                                                    p.q.consistent, ind1, ind2, 
                                                                    model.core.num_pars, initGuess,
                                                                    p.q.θranges, p.q.ωranges, 
                                                                    p.options, false, target_level_set_all)
    end

    return target_level_set_2D, target_level_set_all
end

"""
    initial_continuation_solution!(p::NamedTuple, 
        bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        ind1::Int, 
        ind2::Int,
        profile_type::AbstractProfileType,
        ellipse_confidence_level::Float64,
        target_confidence_ll::Float64,
        ellipse_start_point_shift::Float64, 
        find_zero_atol::Real,
        channel::RemoteChannel)


Finds the initial continuation level set of [`ContinuationMethod`](@ref).

The initial ellipse solution found using [EllipseSampling.jl](https://joeltrent.github.io/EllipseSampling.jl/stable/) should be in the feasible region, contained within the bounds specified for interest parameters. A warning is raised if it is not - it may cause some unexpected behaviour if the parameter is meant to be ≥ 0, yet is allowed to start there in the initial ellipse solution.

We use the extrema of the true log likelihoods (for `profile_type`) of the initial ellipse solution to decide how we search for the first level set. We have three cases, where case one is preferred and warnings are raised for both case two and three.
1. If min ll > than target ll of the target confidence level. Then line search from initial ellipse to ll boundary defined by min ll and this is the starting continuation solution. Line search radially from the MLE point.

2. If max ll < than target ll of the target confidence level. Line search radially towards the MLE point from the ellipse to the target confidence level boundary and this is the final continuation solution.

3. If min ll and max ll bracket the target confidence level. Then line search radially towards the mle solution from initial ellipse to ll boundary defined by max ll and this is the starting continuation solution.
"""
function initial_continuation_solution!(p::NamedTuple, 
                                        bivariate_optimiser::Function, 
                                        model::LikelihoodModel, 
                                        num_points::Int, 
                                        ind1::Int, 
                                        ind2::Int,
                                        profile_type::AbstractProfileType,
                                        ellipse_confidence_level::Float64,
                                        target_confidence_ll::Float64,
                                        ellipse_start_point_shift::Float64, 
                                        find_zero_atol::Real,
                                        channel::RemoteChannel)
    
    check_ellipse_approx_exists!(model)
    
    # get initial continuation starting solution
    # internal boundary - preferably very small
    # ellipse_points = generate_N_equally_spaced_points(num_points, model.ellipse_MLE_approx.Γmle,
    #                                                     model.core.θmle, ind1, ind2,
    #                                                     confidence_level=ellipse_confidence_level, 
    #                                                     start_point_shift=ellipse_start_point_shift)
    
    ellipse_points = generate_N_clustered_points(num_points, model.ellipse_MLE_approx.Γmle,
                                                        model.core.θmle, ind1, ind2,
                                                        confidence_level=ellipse_confidence_level, 
                                                        start_point_shift=ellipse_start_point_shift, 
                                                        sqrt_distortion=0.0)

    for i in 1:num_points
        if model.core.θlb[ind1] > ellipse_points[1,i] || model.core.θub[ind1] < ellipse_points[1,i]
            @warn string("initial ellipse starting solution for 2D continuation method with variables ", model.core.θnames[ind1], " and ", model.core.θnames[ind2]," contains solutions outside specified bounds for ", model.core.θnames[ind1], ". This may cause unexpected behaviour - a smaller ellipse confidence level is recommended.")
            break
        end
        if model.core.θlb[ind2] > ellipse_points[2,i] || model.core.θub[ind2] < ellipse_points[2,i]
            @warn string("initial ellipse starting solution for 2D continuation method with variables ", model.core.θnames[ind1], " and ", model.core.θnames[ind2]," contains solutions outside specified bounds for ", model.core.θnames[ind2], ". This may cause unexpected behaviour - a smaller ellipse confidence level is recommended.")
            break
        end
    end

    # calculate true log likelihood at each point on ellipse approx
    ellipse_true_lls = zeros(num_points)
    p = update_targetll!(p, 0.0)

    for i in 1:num_points
        p.pointa .= ellipse_points[:,i]
        ellipse_true_lls[i] = bivariate_optimiser(0.0, p)
        # extract value of p.ω_opt
    end

    if profile_type isa LogLikelihood
        ellipse_true_lls .= ellipse_true_lls .- model.core.maximisedmle
    end

    min_ll, max_ll = extrema(ellipse_true_lls)
    is_a_zero = falses(num_points)

    point_is_on_bounds = falses(num_points)

    search_directions = zeros(2, num_points)
    search_directions .= ellipse_points .- model.core.θmle[[ind1, ind2]]

    if target_confidence_ll < min_ll # case 1
        corrected_ll = ll_correction(model, profile_type, min_ll)
        is_a_zero[findfirst(ellipse_true_lls .== min_ll)] = true

        a, b = continuation_line_search!(p, point_is_on_bounds, bivariate_optimiser, 
                                            model, 
                                            num_points, ind1, ind2,
                                            corrected_ll, search_directions, ellipse_points,
                                            find_zero_atol, 
                                            channel,
                                            is_a_zero=is_a_zero
                                            )
        return a, b, search_directions, min_ll, point_is_on_bounds

    elseif max_ll < target_confidence_ll # case 2
        corrected_ll = ll_correction(model, profile_type, target_confidence_ll)

        @warn string("ellipse starting point for continuation with variables ", model.core.θnames[ind1], " and ", model.core.θnames[ind2]," contains the smallest target confidence level set. Using a smaller ellipse confidence level is recommended")
        a, b = continuation_inwards_radial_search!(p, bivariate_optimiser, model, 
                                                    num_points, ind1, ind2,
                                                    corrected_ll, search_directions, ellipse_points,
                                                    is_a_zero, find_zero_atol, channel)
        return a, b, search_directions, target_confidence_ll, point_is_on_bounds
    end

    # else # case 3
    corrected_ll = ll_correction(model, profile_type, max_ll)
    is_a_zero[findfirst(ellipse_true_lls .== max_ll)] = true

    @warn string("ellipse starting point for continuation with variables ", model.core.θnames[ind1], " and ", model.core.θnames[ind2]," intersects the smallest target confidence level set. Using a smaller ellipse confidence level is recommended")
    a, b = continuation_inwards_radial_search!(p, bivariate_optimiser, model, 
                                                num_points, ind1, ind2, 
                                                corrected_ll, search_directions, ellipse_points, 
                                                is_a_zero, find_zero_atol, channel)
    return a, b, search_directions, max_ll, point_is_on_bounds
end

"""
    bivariate_confidenceprofile_continuation(bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        consistent::NamedTuple, 
        ind1::Int, 
        ind2::Int,
        profile_type::AbstractProfileType,
        θlb_nuisance::AbstractVector{<:Real},
        θub_nuisance::AbstractVector{<:Real},
        ellipse_confidence_level::Float64, 
        target_confidence_level::Float64,
        ellipse_start_point_shift::Float64,
        num_level_sets::Int,
        level_set_spacing::Symbol,
        mle_targetll::Float64,
        save_internal_points::Bool,
        find_zero_atol::Real,
        optimizationsettings::OptimizationSettings,
        channel::RemoteChannel)

Implementation of [`ContinuationMethod`](@ref).
"""
function bivariate_confidenceprofile_continuation(bivariate_optimiser::Function, 
                                                    model::LikelihoodModel, 
                                                    num_points::Int, 
                                                    consistent::NamedTuple, 
                                                    ind1::Int, 
                                                    ind2::Int,
                                                    profile_type::AbstractProfileType,
                                                    θlb_nuisance::AbstractVector{<:Real},
                                                    θub_nuisance::AbstractVector{<:Real},
                                                    ellipse_confidence_level::Float64, 
                                                    target_confidence_level::Float64,
                                                    ellipse_start_point_shift::Float64,
                                                    num_level_sets::Int,
                                                    level_set_spacing::Symbol,
                                                    mle_targetll::Float64,
                                                    save_internal_points::Bool,
                                                    find_zero_atol::Real, 
                                                    optimizationsettings::OptimizationSettings,
                                                    channel::RemoteChannel)

    newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(model, ind1, ind2, θlb_nuisance, θub_nuisance)
    
    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]

    internal_all = zeros(model.core.num_pars, 0)
    ll_values = zeros(0)


    q=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, ωranges=ωranges, consistent=consistent)
    p=(ω_opt=zeros(model.core.num_pars-2), pointa=pointa, uhat=uhat, targetll=0.0,
        q=q, options=optimizationsettings)

    initial_target_ll = get_target_loglikelihood(model, target_confidence_level,
                                                 EllipseApproxAnalytical(), 2)

    # find initial solution
    current_level_set_2D, current_level_set_all, 
        search_directions, initial_ll, point_is_on_bounds =
        initial_continuation_solution!(p, bivariate_optimiser, 
                                        model, num_points, ind1, ind2, profile_type,
                                        ellipse_confidence_level, initial_target_ll, 
                                        ellipse_start_point_shift, 
                                        find_zero_atol, channel)

    if initial_ll == initial_target_ll
        return current_level_set_all
    end

    initial_confidence_level = cdf(Chisq(2), -initial_ll*2.0)
    if level_set_spacing == :loglikelihood
        level_set_lls = collect(LinRange(get_target_loglikelihood(model, initial_confidence_level,
                                                                profile_type, 2),
                                        get_target_loglikelihood(model, target_confidence_level,
                                                                profile_type, 2), 
                                        num_level_sets+1)[2:end]
                                )
    else
        conf_level_sets = collect(LinRange(initial_confidence_level, target_confidence_level, num_level_sets+1)[2:end])
        level_set_lls = [get_target_loglikelihood(model, conf_level_sets[i], 
                            profile_type, 2) for i in 1:num_level_sets]
    end

    require_TSP_reordering=false
    for (i, level_set_ll) in enumerate(level_set_lls)
        if save_internal_points
            internal_all = hcat(internal_all, current_level_set_all[:, .!point_is_on_bounds])
            ll_values = vcat(ll_values, fill(i == 1 ? 
                                                get_target_loglikelihood(model, initial_confidence_level, profile_type, 2) : 
                                                level_set_lls[i-1], length(point_is_on_bounds) - sum(point_is_on_bounds)))
        end
        
        # perform any desired manipulation of the polygon boundary or search directions
        if false; 
            boundary_smoother!(current_level_set_2D, point_is_on_bounds) 
            level_set_not_smoothed = false
        else
            level_set_not_smoothed = true
        end

        require_TSP_reordering = refine_search_directions!(search_directions, current_level_set_2D, point_is_on_bounds)

        # find next level set
        current_level_set_2D, current_level_set_all = 
            continuation_line_search!(p, point_is_on_bounds, 
                                        bivariate_optimiser, 
                                        model, num_points, ind1, ind2, level_set_ll, search_directions,
                                        current_level_set_2D, find_zero_atol, channel,
                                        start_level_set_all=current_level_set_all,
                                        level_set_not_smoothed=level_set_not_smoothed)

        if require_TSP_reordering
            path = minimum_perimeter_polygon!(current_level_set_2D)[1:end-1]
            current_level_set_all .= current_level_set_all[:,path]
            point_is_on_bounds .= point_is_on_bounds[path]
        end
    end

    if save_internal_points; ll_values .= ll_values .- ll_correction(model, profile_type, 0.0) end

    return current_level_set_all, PointsAndLogLikelihood(internal_all, ll_values)
end