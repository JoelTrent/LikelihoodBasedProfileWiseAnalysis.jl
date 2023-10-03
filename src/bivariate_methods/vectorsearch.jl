# functions for the RadialRandomMethod, RadialMLEMethod and SimultaneousMethod methods

"""
    generatepoint(model::LikelihoodModel, ind1::Int, ind2::Int)

Generates a uniform random x and y value between the lower and upper bounds for the parameters at `ind1` and `ind2`.
"""
function generatepoint(model::LikelihoodModel, ind1::Int, ind2::Int)
    return rand(Uniform(model.core.θlb[ind1], model.core.θub[ind1])), rand(Uniform(model.core.θlb[ind2], model.core.θub[ind2]))
end

"""
    findNpointpairs_simultaneous!(p::NamedTuple, 
        bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        ind1::Int, 
        ind2::Int,
        mle_targetll::Float64,
        save_internal_points::Bool,
        biv_opt_is_ellipse_analytical::Bool,
        min_proportion_unique::Real,
        use_MLE_point::Bool)

Implementation of finding pairs of points that bracket the bivariate confidence boundary for [`SimultaneousMethod`](@ref).
"""
function findNpointpairs_simultaneous!(q::NamedTuple, 
                                        bivariate_optimiser::Function, 
                                        model::LikelihoodModel, 
                                        num_points::Int, 
                                        ind1::Int, 
                                        ind2::Int,
                                        mle_targetll::Float64,
                                        save_internal_points::Bool,
                                        biv_opt_is_ellipse_analytical::Bool,
                                        min_proportion_unique::Real,
                                        use_MLE_point::Bool,
                                        optimizationsettings::OptimizationSettings)

    internal  = zeros(2,num_points)
    internal_all = zeros(model.core.num_pars, save_internal_points ? num_points : 0)
    ll_values = zeros(save_internal_points ? num_points : 0)
    external = zeros(2,num_points)

    min_num_unique = ceil(Int, min_proportion_unique*num_points)
    
    Ninside=0; Noutside=0
    iters=0
    p = (ω_opt=zeros(model.core.num_pars - 2), pointa=zeros(2), uhat=zeros(2), q=q, options=optimizationsettings)

    if use_MLE_point
        Ninside +=1
        internal[:,Ninside] = model.core.θmle[[ind1, ind2]]
    end

    while Noutside<num_points && Ninside<num_points

        x, y = generatepoint(model, ind1, ind2)
        p.pointa .= [x,y]
        g = bivariate_optimiser(0.0, p)
        if g > 0
            Ninside+=1
            internal[:,Ninside] .= [x,y]

            if save_internal_points
                ll_values[Ninside] = g * 1.0
                internal_all[[ind1, ind2], Ninside] .= x, y
                if !biv_opt_is_ellipse_analytical
                    variablemapping!(@view(internal_all[:, Ninside]), p.ω_opt, q.θranges, q.ωranges)
                end
            end
        else
            Noutside+=1
            external[:,Noutside] .= [x,y]
        end
        iters+=1
    end

    # while Ninside < N && iters < maxIters
    while Ninside < min_num_unique
        x, y = generatepoint(model, ind1, ind2)
        p.pointa .= [x,y]
        g = bivariate_optimiser(0.0, p)
        if g > 0
            Ninside+=1
            internal[:,Ninside] .= [x,y]

            if save_internal_points
                ll_values[Ninside] = g * 1.0
                internal_all[[ind1, ind2], Ninside] .= x, y
                if !biv_opt_is_ellipse_analytical
                    variablemapping!(@view(internal_all[:, Ninside]), p.ω_opt, q.θranges, q.ωranges)
                end
            end
        end
        iters+=1
    end

    # while Noutside < N && iters < maxIters
    while Noutside < min_num_unique
        x, y = generatepoint(model, ind1, ind2)
        p.pointa .= [x,y]
        if bivariate_optimiser(0.0, p) < 0
            Noutside+=1
            external[:,Noutside] .= [x,y]
        end
        iters+=1
    end

    if Ninside < num_points
        num_unique = Ninside*1
        while Ninside < num_points
            i, j = Ninside+1, min(num_points-Ninside, num_unique)
            internal[:, i:(i+j-1)] .= internal[:,1:j]
            Ninside += num_unique
        end
        
        if save_internal_points
            ll_values = ll_values[1:num_unique]
            internal_all = internal_all[:, 1:num_unique]
        end

    elseif Noutside < num_points
        num_unique = Ninside * 1
        while Noutside < num_points
            i, j = Noutside+1, min(num_points-Noutside, num_unique)
            external[:, i:(i+j-1)] .= external[:, 1:j]
            Noutside += num_unique
        end
    end

    if use_MLE_point
        ll_values = ll_values[2:end]
        internal_all = internal_all[:, 2:end]
    end

    if save_internal_points && biv_opt_is_ellipse_analytical
        get_ωs_bivariate_ellipse_analytical!(@view(internal_all[[ind1, ind2], :]), size(internal_all, 2),
                                                    q.consistent, ind1, ind2, 
                                                    model.core.num_pars, q.initGuess,
                                                    q.θranges, q.ωranges, 
                                                    optimizationsettings, false, internal_all)
    end

    if save_internal_points; ll_values .= ll_values .+ mle_targetll end

    return internal, internal_all, ll_values, external
end

"""
    find_m_spaced_radialdirections(num_directions::Int; start_point_shift::Float64=rand())

Returns `num_directions` equally spaced anticlockwise angles between 0 and 2pi which are shifted by `start_point_shift * 2.0 / convert(Float64, num_directions)`.
"""
function find_m_spaced_radialdirections(num_directions::Int; start_point_shift::Float64=rand())
    radial_dirs = zeros(num_directions)
    radial_dirs .= (start_point_shift * 2.0 / convert(Float64, num_directions)) .+ collect(LinRange(1e-12, 2.0, num_directions+1))[1:end-1]
    return radial_dirs
end

"""
    findNpointpairs_radialrandom!(p::NamedTuple, 
        bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        num_directions::Int, 
        ind1::Int, 
        ind2::Int,
        mle_targetll::Float64,
        save_internal_points::Bool,
        biv_opt_is_ellipse_analytical::Bool, 
        use_MLE_point::Bool)

Implementation of finding pairs of points that bracket the bivariate confidence boundary for [`RadialRandomMethod`](@ref).

Distorts uniformly spaced anticlockwise angles on a circle using [`PlaceholderLikelihood.find_m_spaced_radialdirections`](@ref) to angles on an ellipse representative of the relative magnitude of each parameter. If the magnitude of a parameter is a NaN value (i.e. either bound is Inf), then the relative magnitude is set to 1.0, as no information is known about its magnitude.
"""
function findNpointpairs_radialrandom!(q::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int, 
                                    num_directions::Int, 
                                    ind1::Int, 
                                    ind2::Int,
                                    mle_targetll::Float64,
                                    save_internal_points::Bool,
                                    biv_opt_is_ellipse_analytical::Bool, 
                                    use_MLE_point::Bool,
                                    optimizationsettings::OptimizationSettings)

    internal  = zeros(2,num_points)
    internal_all = zeros(model.core.num_pars, save_internal_points ? num_points : 0)
    ll_values = zeros(save_internal_points ? num_points : 0)
    external = zeros(2,num_points)

    save_ωs = save_internal_points && !biv_opt_is_ellipse_analytical

    count = 0
    internal_count=0
    ω_opt = zeros(model.core.num_pars-2)
    g_ll = 0.0
    p = (ω_opt=zeros(model.core.num_pars-2), pointa=zeros(2), uhat=zeros(2), q=q, options=optimizationsettings)
    
    if isnan(model.core.θmagnitudes[ind1]) || isnan(model.core.θmagnitudes[ind2]) 
        relative_magnitude = 1.0
    else
        relative_magnitude = model.core.θmagnitudes[ind1]/model.core.θmagnitudes[ind2]
    end
    while count < num_points
        x, y = 0.0, 0.0
        point_is_MLE=false
        # find an internal point
        while true
            if count == 0 && use_MLE_point
                x, y = model.core.θmle[[ind1, ind2]]
                point_is_MLE=true
                break
            end
            
            x, y = generatepoint(model, ind1, ind2)
            p.pointa .= [x,y]
            g_gen = bivariate_optimiser(0.0, p)
            if g_gen > 0 
                if save_internal_points; g_ll = g_gen end
                if save_ωs; ω_opt .= p.ω_opt end
                break
            end
        end

        radial_dirs = find_m_spaced_radialdirections(num_directions)

        count_accepted=0
        for i in 1:num_directions
            dir_vector = [relative_magnitude * cospi(radial_dirs[i]), sinpi(radial_dirs[i]) ]
            boundpoint = findpointonbounds(model, [x, y], dir_vector, ind1, ind2)
            # boundpoint = findpointonbounds(model, [x, y], radial_dirs[i], ind1, ind2)

            # if bound point is a point outside the boundary, accept the point combination
            p.pointa .= boundpoint
            g = bivariate_optimiser(0.0, p)
            if g < 0
                count += 1
                count_accepted += 1
                internal[:, count] .= x, y 

                # make bracket a tiny bit smaller
                if isinf(g)
                    v_bar = boundpoint .- internal[:, count]
                    boundpoint .= internal[:, count] .+ ((1.0-1e-8) .* v_bar)
                end

                external[:, count] .= boundpoint
                if point_is_MLE; continue end

                if save_internal_points && count_accepted == 1
                    internal_count += 1
                    ll_values[internal_count] = g_ll * 1.0
                    internal_all[[ind1, ind2], internal_count] .= x, y
                    if !biv_opt_is_ellipse_analytical
                        variablemapping!(@view(internal_all[:, internal_count]), ω_opt, q.θranges, q.ωranges)
                    end
                end
            end

            if count == num_points
                break
            end
        end
    end

    if save_internal_points 
        ll_values = ll_values[1:internal_count] .+ mle_targetll
        
        internal_all = internal_all[:, 1:internal_count]
        if biv_opt_is_ellipse_analytical
            get_ωs_bivariate_ellipse_analytical!(@view(internal_all[[ind1, ind2], :]), internal_count,
                                                    q.consistent, ind1, ind2, 
                                                    model.core.num_pars, q.initGuess,
                                                    q.θranges, q.ωranges, 
                                                    optimizationsettings, false, internal_all)
        end
    end

    return internal, internal_all, ll_values, external
end

"""
    findNpointpairs_radialMLE!(p::NamedTuple, 
        bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        ind1::Int, 
        ind2::Int,
        ellipse_confidence_level::Float64,
        ellipse_start_point_shift::Float64,
        ellipse_sqrt_distortion::Float64,
        optimizationsettings::OptimizationSettings,
        use_threads::Bool)

Implementation of finding pairs of points that bracket the bivariate confidence boundary for [`RadialMLEMethod`](@ref).

Search directions from the MLE point are given by points placed on a ellipse approximation around the point using [`generate_N_clustered_points`](https://joeltrent.github.io/EllipseSampling.jl/stable/user_interface/#EllipseSampling.generate_N_clustered_points) from [EllipseSampling.jl](https://joeltrent.github.io/EllipseSampling.jl/stable). 
"""
function findNpointpairs_radialMLE!(q::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int, 
                                    ind1::Int, 
                                    ind2::Int,
                                    ellipse_confidence_level::Float64,
                                    ellipse_start_point_shift::Float64,
                                    ellipse_sqrt_distortion::Float64,
                                    optimizationsettings::OptimizationSettings,
                                    use_threads::Bool)

    mle_point = model.core.θmle[[ind1, ind2]]
    internal = zeros(2,num_points) .= mle_point
    external = zeros(2,num_points)
    point_is_on_bounds = falses(num_points)
    # warn if bound prevents reaching boundary
    bound_inds = [(0, 'a') for _ in 1:num_points]

    check_ellipse_approx_exists!(model)
    ellipse_points = generate_N_clustered_points(num_points, model.ellipse_MLE_approx.Γmle,
                                                        model.core.θmle, ind1, ind2,
                                                        confidence_level=ellipse_confidence_level, 
                                                        start_point_shift=ellipse_start_point_shift, 
                                                        sqrt_distortion=ellipse_sqrt_distortion)

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_points)
    @floop ex for i in 1:num_points
        FLoops.@init pointa = zeros(2)
        FLoops.@init uhat = zeros(2)
        FLoops.@init ω_opt = zeros(model.core.num_pars-2)
        p = (ω_opt=ω_opt, pointa=pointa, uhat=uhat, q=q, options=optimizationsettings)

        dir_vector = ellipse_points[:,i] .- mle_point
        external[:,i], bound_ind, upper_or_lower = findpointonbounds(model, mle_point, dir_vector, ind1, ind2, true)

        p.pointa .= external[:,i]
        g = bivariate_optimiser(0.0, p)
        if g ≥ 0
            point_is_on_bounds[i] = true
            bound_inds[i] = (bound_ind, upper_or_lower)
        else
            # make bracket a tiny bit smaller
            if isinf(g)
                v_bar = external[:,i] .- mle_point
                external[:,i] .= mle_point .+ ((1.0-1e-8) .* v_bar)
            end
        end
    end

    if any(point_is_on_bounds)
        _bound_ind, _upper_or_lower = bound_inds[findfirst(point_is_on_bounds)]
        _upper_or_lower = _upper_or_lower == 'U' ? "upper" : "lower"
        @warn string("The ", _upper_or_lower, " bound on variable ", model.core.θnames[_bound_ind], " is inside the confidence boundary")
    end

    internal_all = zeros(model.core.num_pars, 0)
    ll_values = zeros(0)
    return internal, internal_all, ll_values, external, point_is_on_bounds, any(point_is_on_bounds)
end

"""
    bivariate_confidenceprofile_vectorsearch(bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        consistent::NamedTuple, 
        ind1::Int, 
        ind2::Int,
        mle_targetll::Float64,
        save_internal_points::Bool,
        find_zero_atol::Real, 
        optimizationsettings::OptimizationSettings,
        use_threads::Bool,
        channel::RemoteChannel;
        num_radial_directions::Int=0,
        min_proportion_unique::Real=1.0,
        use_MLE_point::Bool=false,
        ellipse_confidence_level::Float64=-1.0,
        ellipse_start_point_shift::Float64=0.0,
        ellipse_sqrt_distortion::Float64=0.0)

Implementation of [`AbstractBivariateVectorMethod`] boundary search methods [`SimultaneousMethod`](@ref), [`RadialMLEMethod`](@ref) and [`RadialRandomMethod`](@ref).
"""
function bivariate_confidenceprofile_vectorsearch(bivariate_optimiser::Function, 
                                                    model::LikelihoodModel, 
                                                    num_points::Int, 
                                                    consistent::NamedTuple, 
                                                    ind1::Int, 
                                                    ind2::Int,
                                                    mle_targetll::Float64,
                                                    save_internal_points::Bool,
                                                    find_zero_atol::Real, 
                                                    optimizationsettings::OptimizationSettings,
                                                    use_threads::Bool,
                                                    channel::RemoteChannel;
                                                    num_radial_directions::Int=0,
                                                    min_proportion_unique::Real=1.0,
                                                    use_MLE_point::Bool=false,
                                                    ellipse_confidence_level::Float64=-1.0,
                                                    ellipse_start_point_shift::Float64=0.0,
                                                    ellipse_sqrt_distortion::Float64=0.0)

    newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(model, ind1, ind2)

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateψ_ellipse_analytical_vectorsearch
    
    boundary = zeros(model.core.num_pars, num_points)

    q=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess,
        θranges=θranges, ωranges=ωranges, consistent=consistent)


    if ellipse_confidence_level !== -1.0
        internal, internal_all, ll_values, external, point_is_on_bounds, _ = 
            findNpointpairs_radialMLE!(q, bivariate_optimiser, model, num_points, ind1, ind2, 
                                        ellipse_confidence_level, ellipse_start_point_shift, ellipse_sqrt_distortion,
                                        optimizationsettings, use_threads)

    else
        if num_radial_directions == 0
            internal, internal_all, ll_values, external = 
                findNpointpairs_simultaneous!(q, bivariate_optimiser, model, num_points, ind1, ind2,
                                                mle_targetll, save_internal_points, biv_opt_is_ellipse_analytical,
                                                min_proportion_unique, use_MLE_point,
                                                optimizationsettings)
        else
            internal, internal_all, ll_values, external = 
                findNpointpairs_radialrandom!(q, bivariate_optimiser, model, num_points, 
                                                num_radial_directions, ind1, ind2,
                                                mle_targetll, save_internal_points,
                                                biv_opt_is_ellipse_analytical, use_MLE_point,
                                                optimizationsettings)
        end
        point_is_on_bounds = falses(num_points)
    end

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=num_points)
    let internal=internal, external=external, point_is_on_bounds=point_is_on_bounds
        @floop ex for i in 1:num_points
            FLoops.@init pointa = zeros(2)
            FLoops.@init uhat = zeros(2)
            FLoops.@init ω_opt = zeros(model.core.num_pars-2)
            p = (ω_opt=ω_opt, pointa=pointa, uhat=uhat, q=q, options=optimizationsettings)

            if point_is_on_bounds[i]
                p.pointa .= external[:,i]
                bivariate_optimiser(0, p)
                boundary[[ind1, ind2], i] .= external[:,i]
            else
                p.pointa .= internal[:,i]
                v_bar = external[:,i] .- internal[:,i]

                v_bar_norm = norm(v_bar, 2)
                p.uhat .= v_bar ./ v_bar_norm

                ψ = find_zero(bivariate_optimiser, (0.0, v_bar_norm), Roots.Brent(); atol=find_zero_atol, p=p)
                
                boundary[[ind1, ind2], i] .= p.pointa + ψ*p.uhat
                if !biv_opt_is_ellipse_analytical; bivariate_optimiser(ψ, p) end
            end
            if !biv_opt_is_ellipse_analytical
                variablemapping!(@view(boundary[:, i]), p.ω_opt, θranges, ωranges)
            end
            put!(channel, true)
        end
    end

    if biv_opt_is_ellipse_analytical
        return get_ωs_bivariate_ellipse_analytical!(@view(boundary[[ind1, ind2], :]), num_points,
                    consistent, ind1, ind2, model.core.num_pars, initGuess, θranges, ωranges, 
                    optimizationsettings, use_threads, boundary), 
                    PointsAndLogLikelihood(internal_all, ll_values)
    end

    return boundary, PointsAndLogLikelihood(internal_all, ll_values)
end