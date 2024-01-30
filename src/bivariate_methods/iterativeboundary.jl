"""
    findNpointpairs_radialMLE!(p::NamedTuple, 
        bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        ind1::Int, 
        ind2::Int,
        biv_opt_is_ellipse_analytical::Bool, 
        radial_start_point_shift::Float64,
        optimizationsettings::OptimizationSettings,
        use_threads::Bool)

Implementation of finding pairs of points that bracket the bivariate confidence boundary for [`IterativeBoundaryMethod`](@ref), searching radially from the MLE point for points on the bounds, in search directions similar to those used by [`RadialRandomMethod`](@ref). If a point on the bounds is inside the confidence boundary, that point will represent the boundary in that search direction.

Search directions are given by distorting uniformly spaced anticlockwise angles on a circle to angles on an ellipse representative of the relative magnitude of each parameter. If the magnitude of a parameter is a NaN value (i.e. either bound is Inf), then the relative magnitude is set to 1.0, as no information is known about its magnitude.
"""
function findNpointpairs_radialMLE!(q::NamedTuple, 
                                    bivariate_optimiser::Function, 
                                    model::LikelihoodModel, 
                                    num_points::Int, 
                                    ind1::Int, 
                                    ind2::Int,
                                    biv_opt_is_ellipse_analytical::Bool, 
                                    radial_start_point_shift::Float64,
                                    optimizationsettings::OptimizationSettings)

    mle_point = model.core.θmle[[ind1, ind2]]
    internal = zeros(2,num_points) .= mle_point
    external = zeros(2,num_points)
    external_all = zeros(model.core.num_pars, biv_opt_is_ellipse_analytical ? 0 : num_points)
    point_is_on_bounds = falses(num_points)
    # warn if bound prevents reaching boundary
    bound_inds = [(0, 'a') for _ in 1:num_points]
    
    if isnan(model.core.θmagnitudes[ind1]) || isnan(model.core.θmagnitudes[ind2]) 
        relative_magnitude = 1.0
    else
        relative_magnitude = model.core.θmagnitudes[ind1]/model.core.θmagnitudes[ind2]
    end

    radial_dirs = find_m_spaced_radialdirections(num_points, start_point_shift=radial_start_point_shift)
    for i in 1:num_points
        pointa = zeros(2)
        uhat = zeros(2)
        ω_opt = zeros(model.core.num_pars-2)
        p = (ω_opt=ω_opt, pointa=pointa, uhat=uhat, q=q, options=optimizationsettings)

        dir_vector = [relative_magnitude * cospi(radial_dirs[i]), sinpi(radial_dirs[i]) ]
        external[:,i], bound_ind, upper_or_lower = findpointonbounds(model, mle_point, dir_vector, ind1, ind2, true)

        # if bound point is a point inside the boundary, note that this is the case
        p.pointa .= external[:,i]
        g = bivariate_optimiser(0.0, p)
        if g ≥ 0
            point_is_on_bounds[i] = true
            bound_inds[i] = (bound_ind, upper_or_lower)
            if !biv_opt_is_ellipse_analytical
                external_all[[ind1, ind2], i] .= external[:, i]
                variablemapping!(@view(external_all[:, i]), p.ω_opt .* 1.0, q.θranges, q.ωranges)
            end
        else
            # make bracket a tiny bit smaller
            if isinf(g)
                v_bar = external[:,i] .- mle_point
                external[:,i] .= mle_point .+ ((1.0-1e-12) .* v_bar)
            end
        end
    end

    if any(point_is_on_bounds)
        _bound_ind, _upper_or_lower = bound_inds[findfirst(point_is_on_bounds)]
        _upper_or_lower = _upper_or_lower == 'U' ? "upper" : "lower"
        @warn string("The ", _upper_or_lower, " bound on variable ", model.core.θnames[_bound_ind], " is inside the confidence boundary")
    end

    return internal, external, external_all, point_is_on_bounds, any(point_is_on_bounds)
end

"""
    edge_length(boundary, inds1, inds2, relative_magnitude)

Euclidean distance between two vertices (length of an edge), scaled by the relative magnitude of parameters, so that each dimension has roughly the same weight.
"""
function edge_length(boundary, inds1::Union{UnitRange, AbstractVector}, inds2::Union{UnitRange, AbstractVector}, relative_magnitude)
    return colwise(Euclidean(), 
    boundary[:, inds1] ./ SA[relative_magnitude, 1.0], 
    boundary[:, inds2] ./ SA[relative_magnitude, 1.0]) 
end

function edge_length(boundary, ind1::Int, ind2::Int, relative_magnitude)
    return evaluate(Euclidean(), 
    boundary[:, ind1] ./ SA[relative_magnitude, 1.0], 
    boundary[:, ind2] ./ SA[relative_magnitude, 1.0]) 
end

function edge_length(boundary, candidate_point, index::Int, relative_magnitude)
    return evaluate(Euclidean(), 
    candidate_point ./ SA[relative_magnitude, 1.0], 
    boundary[:, index] ./ SA[relative_magnitude, 1.0]) 
end

"""
    internal_angle_from_pi!(vertex_internal_angle_objs, 
        indexes::UnitRange, 
        boundary, 
        adjacent_vertices)

The magnitude the internal angle in radians between two adjacent edges is from pi radians - i.e. how far away the two edges are from representing a straight boundary. If a boundary is straight then the objective is 0.0 radians, whereas if the boundary has an internal angle of pi/4 radians (45 deg) the objective is pi*3/4 (135 deg). Computes this by considering the angle between the two vectors that can be used to represent the edges (using `AngleBetweenVectors.jl`).
"""
function internal_angle_from_pi!(vertex_internal_angle_objs, indexes::UnitRange, boundary, edge_clock, edge_anti, relative_magnitude)
    for i in indexes
        vertex_internal_angle_objs[i] = AngleBetweenVectors.angle((boundary[:,i] .- boundary[:, edge_clock[i]]) ./ SA[relative_magnitude, 1.0], 
                                                                    (boundary[:,edge_anti[i]] .- boundary[:,i])./ SA[relative_magnitude, 1.0])
    end
    return nothing
end

"""
    internal_angle_from_pi(index::Int, boundary, edge_clock, edge_anti, relative_magnitude)

Alternate method of [`PlaceholderLikelihood.internal_angle_from_pi`](@ref) for the internal angle at only one vertex.
"""
function internal_angle_from_pi(index::Int, boundary, edge_clock, edge_anti, relative_magnitude)
    if index == edge_clock[index] && index == edge_anti[index]
        return 0.0
    end 
    return AngleBetweenVectors.angle((boundary[:,index] .- boundary[:, edge_clock[index]])./ SA[relative_magnitude, 1.0],
                                        (boundary[:,edge_anti[index]] .- boundary[:, index])./ SA[relative_magnitude, 1.0]) 
end

"""
    iterativeboundary_init(bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        p::NamedTuple, 
        ind1::Int, 
        ind2::Int,
        initial_num_points::Int,
        biv_opt_is_ellipse_analytical::Bool,
        radial_start_point_shift::Float64,
        ellipse_sqrt_distortion::Float64,
        ellipse_confidence_level::Float64,
        use_ellipse::Bool,
        save_internal_points::Bool,
        find_zero_atol::Real, 
        optimizationsettings::OptimizationSettings,
        use_threads::Bool,
        channel::RemoteChannel)

Finds the initial boundary of [`IterativeBoundaryMethod`](@ref), containing `initial_num_points`, returning it and initialised parameter values. 

If `initial_num_points` is equal to `num_points` then the desired number of boundary points have been found. If `use_ellipse = true` the boundary will be equivalent to the boundary found by [`RadialMLEMethod`](@ref) with the same parameter settings - it's value informs the method of [`PlaceholderLikelihood.findNpointpairs_radialMLE!`](@ref) used.
"""
function iterativeboundary_init(bivariate_optimiser::Function, 
                                model::LikelihoodModel, 
                                num_points::Int, 
                                q::NamedTuple, 
                                ind1::Int, 
                                ind2::Int,
                                initial_num_points::Int,
                                biv_opt_is_ellipse_analytical::Bool,
                                radial_start_point_shift::Float64,
                                ellipse_sqrt_distortion::Float64,
                                ellipse_confidence_level::Float64,
                                dof::Int,
                                use_ellipse::Bool,
                                save_internal_points::Bool,
                                find_zero_atol::Real, 
                                optimizationsettings::OptimizationSettings,
                                use_threads::Bool,
                                channel::RemoteChannel)

    boundary = zeros(2, num_points)
    boundary_all = zeros(model.core.num_pars, num_points)
    internal_all = zeros(model.core.num_pars, save_internal_points ? num_points : 0)
    ll_values = zeros(save_internal_points ? num_points : 0)
    internal_count = 0
    point_is_on_bounds = falses(num_points)

    if use_ellipse
        internal, internal_all_init, ll_values_init, external, external_all, point_is_on_bounds_external, bound_warning = 
            findNpointpairs_radialMLE!(q, bivariate_optimiser, model, 
                                        initial_num_points, ind1, ind2, 
                                        0.0, false,
                                        biv_opt_is_ellipse_analytical,
                                        ellipse_confidence_level, 
                                        dof,
                                        radial_start_point_shift, 
                                        ellipse_sqrt_distortion,
                                        optimizationsettings)

        if save_internal_points
            internal_count = length(ll_values_init)
            ll_values[1:internal_count] .= ll_values_init
            internal_all[:, 1:internal_count] .= internal_all_init
        end
    else
        internal, external, external_all, point_is_on_bounds_external, bound_warning = 
            findNpointpairs_radialMLE!(q, bivariate_optimiser, model, 
                                        initial_num_points, ind1, ind2, 
                                        biv_opt_is_ellipse_analytical,
                                        radial_start_point_shift,
                                        optimizationsettings)
    end

    point_is_on_bounds[1:initial_num_points] .= point_is_on_bounds_external[:]

    mle_point = model.core.θmle[[ind1,ind2]]

    ex = use_threads ? ThreadedEx() : ThreadedEx(basesize=initial_num_points)
    let external=external, external_all=external_all, point_is_on_bounds=point_is_on_bounds
        @floop ex for i in 1:initial_num_points
            FLoops.@init pointa = zeros(2)
            FLoops.@init uhat = zeros(2)
            FLoops.@init ω_opt = zeros(model.core.num_pars-2)
            p = (ω_opt=ω_opt, pointa=pointa, uhat=uhat, q=q, options=optimizationsettings)

            if point_is_on_bounds[i]
                boundary[:,i] .= external[:,i]
                if biv_opt_is_ellipse_analytical
                    boundary_all[[ind1, ind2], i] .= external[:,i]
                else
                    boundary_all[:,i] .= external_all[:,i]
                end
            else
                p.pointa .= internal[:, i] .* 1.0
                v_bar = external[:,i] .- internal[:,i]

                v_bar_norm = norm(v_bar, 2)
                p.uhat .= v_bar ./ v_bar_norm

                ψ = find_zero(bivariate_optimiser, (0.0, v_bar_norm), Roots.Brent(); atol=find_zero_atol, p=p)
                
                boundary[:,i] .= p.pointa + ψ*p.uhat
                boundary_all[[ind1, ind2], i] .= boundary[:,i]
                if !biv_opt_is_ellipse_analytical
                    bivariate_optimiser(ψ, p)
                    variablemapping!(@view(boundary_all[:, i]), p.ω_opt, q.θranges, q.ωranges)
                end
            end
            put!(channel, true)
        end
    end

    if initial_num_points == num_points
        return true, boundary_all, PointsAndLogLikelihood(internal_all[:,1:internal_count], ll_values[1:internal_count])
    end 

    num_vertices = initial_num_points * 1

    if isnan(model.core.θmagnitudes[ind1]) || isnan(model.core.θmagnitudes[ind2]) 
        relative_magnitude = 1.0
    else
        relative_magnitude = model.core.θmagnitudes[ind1]/model.core.θmagnitudes[ind2]
    end

    # edge i belongs to vertex i, and gives the vertex it is connected to (clockwise)
    edge_clock = zeros(Int, num_points)
    edge_clock[2:num_vertices] .= 1:(num_vertices-1)
    edge_clock[1] = num_vertices

    # edge i belongs to vertex i, and gives the vertex it is connected to (anticlockwise)
    edge_anti = zeros(Int, num_points)
    edge_anti[1:num_vertices-1] .= 2:num_vertices
    edge_anti[num_vertices] = 1

    edge_anti_on_bounds = falses(num_points)
    edge_anti_on_bounds[1:num_vertices] .= @view(point_is_on_bounds[1:num_vertices]) .& @view(point_is_on_bounds[@view(edge_anti[1:num_vertices])])
    
    # tracked heap for length of edges (anticlockwise)
    edge_lengths = zeros(num_points)
    edge_lengths[num_vertices+1:end] .= -Inf

    edge_lengths[1:num_vertices] .= edge_length(boundary, 1:num_vertices, 
                                                @view(edge_anti[1:num_vertices]), 
                                                relative_magnitude)

    # edge_vectors = zeros(2, num_points)
    # edge_vectors[:, 1:num_vertices] .= boundary[:, edges[2,1:num_vertices]] .- boundary[:, edges[1,1:num_vertices]]


    edge_heap = TrackingHeap(Float64, S=NoTrainingWheels, O=MaxHeapOrder, N = 2,
                                init_val_coll=edge_lengths)
    
    # internal angle function
    # tracked heap for internal angles between adjacent edges (more specifically, angle away from 180 deg - i.e. a straight boundary)

    vertex_internal_angle_objs = zeros(num_points)
    internal_angle_from_pi!(vertex_internal_angle_objs, 1:num_vertices, boundary, edge_clock, edge_anti, relative_magnitude)
  
    angle_heap = TrackingHeap(Float64, S=NoTrainingWheels, O=MaxHeapOrder, N = 2, init_val_coll=vertex_internal_angle_objs)

    return false, boundary, boundary_all, internal_all, ll_values, internal_count, point_is_on_bounds, edge_anti_on_bounds, bound_warning, mle_point, num_vertices, edge_clock, edge_anti, edge_heap, angle_heap, relative_magnitude
end


"""
    newboundarypoint!(p::NamedTuple, 
        point_is_on_bounds::BitVector, 
        edge_anti_on_bounds::BitVector, 
        boundary::Matrix{Float64}, 
        boundary_all::Matrix{Float64}, 
        internal_all::Matrix{Float64}, 
        ll_values::Vector{Float64}, 
        internal_count::Int,
        bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        edge_anti::Vector{Int}, 
        num_vertices::Int, 
        ind1::Int, 
        ind2::Int,
        biv_opt_is_ellipse_analytical::Bool, 
        ve1::Int, 
        ve2::Int, 
        relative_magnitude::Float64, 
        bound_warning::Bool, 
        save_internal_points::Bool)

Method for trying to find a new boundary point for [`IterativeBoundaryMethod`](@ref) given a starting boundary polygon. Returns whether finding a new point was successful.
"""
function newboundarypoint!(p::NamedTuple,
                            point_is_on_bounds::BitVector,
                            edge_anti_on_bounds::BitVector,
                            boundary::Matrix{Float64},
                            boundary_all::Matrix{Float64},
                            internal_all::Matrix{Float64},
                            ll_values::Vector{Float64},
                            internal_count::Int,
                            bivariate_optimiser::Function, 
                            model::LikelihoodModel, 
                            edge_anti::Vector{Int},
                            num_vertices::Int,
                            ind1::Int, 
                            ind2::Int,
                            biv_opt_is_ellipse_analytical::Bool,
                            ve1::Int,
                            ve2::Int,
                            relative_magnitude::Float64,
                            bound_warning::Bool,
                            save_internal_points::Bool,
                            find_zero_atol::Real)

    failure = false

    # candidate point - midpoint of edge calculation
    candidate_midpoint = boundary[:, ve1] .+ 0.5 .* (boundary[:, ve2] - boundary[:, ve1])

    # find new boundary point, given candidate midpoint, the vertexes that describe the edge
    # use candidate point to find new vertex

    if edge_anti_on_bounds[ve1] # accept the mid point and find nuisance parameters
        p.pointa .= candidate_midpoint
        bivariate_optimiser(0.0, p)
        boundary[:, num_vertices] .= candidate_midpoint
        boundary_all[[ind1, ind2], num_vertices] .= candidate_midpoint
        edge_anti_on_bounds[num_vertices] = true
        point_is_on_bounds[num_vertices] = true
    else

        p.pointa .= candidate_midpoint
        dir_vector = SA[(boundary[2,ve2] - boundary[2,ve1]), -(boundary[1,ve2] - boundary[1,ve1])] .* SA[relative_magnitude^2, 1.0]
        g = bivariate_optimiser(0.0, p)
        
        if isapprox(0.0, g, atol=find_zero_atol) # candidate on boundary
            boundary[:, num_vertices] .= candidate_midpoint
            boundary_all[[ind1, ind2], num_vertices] .= candidate_midpoint
            
        elseif g > 0.0 # internal - push out normal to edge

            if save_internal_points
                internal_count += 1
                ll_values[internal_count] = g * 1.0
                internal_all[[ind1, ind2], internal_count] .= candidate_midpoint
                if !biv_opt_is_ellipse_analytical
                    variablemapping!(@view(internal_all[:, internal_count]), p.ω_opt, p.q.θranges, p.q.ωranges)
                end
            end

            boundpoint, bound_ind, upper_or_lower = findpointonbounds(model, candidate_midpoint, (dir_vector ./ norm(dir_vector, 2)), ind1, ind2, true)

            p.pointa .= boundpoint
            v_bar = candidate_midpoint .- boundpoint
            v_bar_norm = norm(v_bar, 2)
            p.uhat .= v_bar ./ v_bar_norm

            # if bound point and pointa bracket a boundary, search for the boundary
            # otherwise, the bound point is used as the level set boundary (i.e. it's inside the true level set boundary)
            g = bivariate_optimiser(0.0, p)
            if biv_opt_is_ellipse_analytical || g < 0.0
                # make bracket a tiny bit smaller
                
                lb = isinf(g) ? 1e-12 * v_bar_norm : 0.0

                ψ = find_zero(bivariate_optimiser, (lb, v_bar_norm), Roots.ITP(); atol=find_zero_atol, p=p)

                boundarypoint = p.pointa + ψ*p.uhat
                boundary[:, num_vertices] .= boundarypoint
                boundary_all[[ind1, ind2], num_vertices] .= boundarypoint
                if !biv_opt_is_ellipse_analytical; bivariate_optimiser(ψ, p) end
            else
                point_is_on_bounds[num_vertices] = true
                boundary[:, num_vertices] .= boundpoint
                boundary_all[[ind1, ind2], num_vertices] .= boundpoint

                if bound_warning
                    upper_or_lower = upper_or_lower == 'U' ? "upper" : "lower"
                    @warn string("The ", upper_or_lower, " bound on variable ", model.core.θnames[bound_ind], " is inside the confidence boundary")
                    bound_warning = false
                end
            end

        else # external - push inwards
            
            candidate_line = Meshes.Line(Point(candidate_midpoint...), Point((candidate_midpoint .+ dir_vector)...)) 
            # find edge that the line normal vector intersects 
            current_vertex = ve2 * 1
            while current_vertex != ve1
                edge_segment = Segment(Point(boundary[:,current_vertex]...), Point(boundary[:, edge_anti[current_vertex]]...))

                if intersection(candidate_line, edge_segment).type != IntersectionType(9)
                    break
                end
                current_vertex = edge_anti[current_vertex] * 1
            end

            # by construction/algorithm enforcement all polygons we search within must have at least three points so this is ok
            # don't want to choose an edge vertex that's on the candidate edge
            if edge_anti[current_vertex] == ve1 
                edge_vertex_index = 1
            elseif current_vertex == ve2
                edge_vertex_index = 2
            else
                edge_vertex_index = argmin(SA[edge_length(boundary, candidate_midpoint, current_vertex, relative_magnitude),
                    edge_length(boundary, candidate_midpoint, edge_anti[current_vertex], relative_magnitude)])
            end
            edge_vertex = edge_vertex_index == 1 ? current_vertex : edge_anti[current_vertex] * 1
            
            p.pointa .= boundary[:,edge_vertex] .* 1.0

            v_bar = candidate_midpoint .- p.pointa
            v_bar_norm = norm(v_bar, 2)
            p.uhat .= v_bar ./ v_bar_norm

            ψ = solve(ZeroProblem(bivariate_optimiser, v_bar_norm), Roots.Order8(); atol=find_zero_atol, p=p)

            boundarypoint = p.pointa + ψ*p.uhat

            if isnan(ψ) || isinf(ψ) || ψ < 0. || ψ > v_bar_norm || isapprox(boundarypoint, p.pointa)
                # failure=true
                f(x) = bivariate_optimiser(x, p)
                ψs = find_zeros(f, 0.0, v_bar_norm; p=p) # note no tolerance here - it can error if atol is used
                if length(ψs) == 0
                    failure=true
                elseif length(ψs) == 1
                    boundarypoint = p.pointa + ψs[1]*p.uhat
                    if isapprox(boundarypoint, p.pointa)
                        failure=true
                    else
                        ψ = ψs[1]
                    end
                else
                    boundarypoint = p.pointa + ψs[end]*p.uhat
                    ψ = ψs[end]
                end
            end
                
            if failure
                return num_vertices, internal_count, failure, bound_warning, edge_vertex
            end

            boundary[:, num_vertices] .= boundarypoint
            boundary_all[[ind1, ind2], num_vertices] .= boundarypoint
            if !biv_opt_is_ellipse_analytical; bivariate_optimiser(ψ, p) end
        end

        if !biv_opt_is_ellipse_analytical
            variablemapping!(@view(boundary_all[:, num_vertices]), p.ω_opt, p.q.θranges, p.q.ωranges)
        end
    end

    return num_vertices, internal_count, failure, bound_warning, 0
end

"""
    heapupdates_success!(edge_heap::TrackingHeap,
        angle_heap::TrackingHeap, 
        edge_clock::Vector{Int},
        edge_anti::Vector{Int},
        point_is_on_bounds::BitVector,
        edge_anti_on_bounds::BitVector,
        boundary::Matrix{Float64},
        num_vertices::Int,
        vi::Int, 
        adj_vertex::Int,
        relative_magnitude::Float64,
        clockwise_from_vi=false)

If finding a new boundary point for [`IterativeBoundaryMethod`](@ref) was successful, update the datastructures that represent the boundary as required.        
"""
function heapupdates_success!(edge_heap::TrackingHeap,
                        angle_heap::TrackingHeap, 
                        edge_clock::Vector{Int},
                        edge_anti::Vector{Int},
                        point_is_on_bounds::BitVector,
                        edge_anti_on_bounds::BitVector,
                        boundary::Matrix{Float64},
                        num_vertices::Int,
                        vi::Int, 
                        adj_vertex::Int,
                        relative_magnitude::Float64,
                        clockwise_from_vi=false)

    # perform required updates
    if clockwise_from_vi
        # adjacent vertex is clockwise from vi
        edge_clock[vi] = num_vertices*1
        edge_clock[num_vertices] = adj_vertex

        edge_anti[adj_vertex] = num_vertices
        edge_anti[num_vertices] = vi

        if point_is_on_bounds[num_vertices]
            edge_anti_on_bounds[num_vertices] = point_is_on_bounds[num_vertices] && point_is_on_bounds[vi]
            edge_anti_on_bounds[adj_vertex] = point_is_on_bounds[num_vertices] && point_is_on_bounds[adj_vertex]
        end

        # update edge length for adj_vertex and num_vertices
        for i in SA[adj_vertex, num_vertices]
            TrackingHeaps.update!(edge_heap, i, edge_length(boundary, i, edge_anti[i], relative_magnitude))
        end
    else
        # adjacent vertex is anticlockwise from vi
        edge_clock[adj_vertex] = num_vertices*1
        edge_clock[num_vertices] = vi

        edge_anti[vi] = num_vertices
        edge_anti[num_vertices] = adj_vertex

        if point_is_on_bounds[num_vertices]
            edge_anti_on_bounds[vi] = point_is_on_bounds[num_vertices] && point_is_on_bounds[vi]
            edge_anti_on_bounds[num_vertices] = point_is_on_bounds[num_vertices] && point_is_on_bounds[adj_vertex]
        end
        
        # update edge length for vi and num_vertices 
        for i in SA[vi, num_vertices]
            TrackingHeaps.update!(edge_heap, i, edge_length(boundary, i, edge_anti[i], relative_magnitude))
        end
    end

    # update angle obj for vi, adj_vertex and new vertex (num_vertices)
    for i in SA[vi, adj_vertex, num_vertices]
        if point_is_on_bounds[i]
            TrackingHeaps.update!(angle_heap, i, 0.0)
        else
            TrackingHeaps.update!(angle_heap, i, internal_angle_from_pi(i, boundary, edge_clock, edge_anti, relative_magnitude))
        end
    end

    return nothing
end

"""
    polygon_break_and_rejoin!(edge_clock::Vector{Int},
        edge_anti::Vector{Int},
        ve1::Int,
        ve2::Int,
        opposite_edge_ve1::Int,
        opposite_edge_ve2::Int,
        model::LikelihoodModel,
        ind1::Int, 
        ind2::Int)

If finding a new boundary point was not successful, breaks and rejoins the boundary polygon as required to remove the vertices from the main polygon that are likely to be on a distinct level set (boundary).

Display an info message if only one vertex was seperated, as that vertex can no longer be used to find additional points within the algorithm.
"""
function polygon_break_and_rejoin!(edge_clock::Vector{Int},
                                    edge_anti::Vector{Int},
                                    ve1::Int,
                                    ve2::Int,
                                    opposite_edge_ve1::Int,
                                    opposite_edge_ve2::Int,
                                    model::LikelihoodModel,
                                    ind1::Int, 
                                    ind2::Int)

    edge_clock[ve2] = opposite_edge_ve1 * 1
    edge_clock[opposite_edge_ve2] = ve1 * 1

    # ve1 -> edge_anti[opposite_edge]
    # opposite_edge -> ve2
    edge_anti[ve1] = opposite_edge_ve2 * 1
    edge_anti[opposite_edge_ve1] = ve2 * 1

    # In the case we have a polygon with only one vertex
    if opposite_edge_ve2 == ve1 || ve2 == opposite_edge_ve1
        @info string("there is likely to be multiple distinct level sets at this confidence level for parameters ", model.core.θnames[ind1]," and ", model.core.θnames[ind2], ". No additional points can be found on one of these level sets within this algorithm run.")
    end
    return nothing
end

"""
    heapupdates_failure!(edge_heap::TrackingHeap,
        angle_heap::TrackingHeap, 
        edge_clock::Vector{Int},
        edge_anti::Vector{Int},
        point_is_on_bounds::BitVector,
        boundary::Matrix{Float64},
        num_vertices::Int,
        ve1::Int,
        ve2::Int,
        opposite_edge_ve1::Int,
        model::LikelihoodModel,
        ind1::Int, 
        ind2::Int,
        relative_magnitude::Float64)

If finding a new boundary point for [`IterativeBoundaryMethod`](@ref) was not successful, update the datastructures that represent the boundary as required. Failure means it is likely that multiple level sets exist. If so, break the edges of the candidate point and `opposite_edge_ve1` and reconnect the vertexes such that there are now multiple boundary polygons.
		
If there are only one or two points on one of these boundary polygons, display an info message as no additional points can be found from the method directly.
		
If there are three or more points on these boundary polygons, then there should be no problems finding other parts of these polygons.

If the largest polygon has less than three points the method will display a warning message and terminate, returning the boundary found up until then. 
"""
function heapupdates_failure!(edge_heap::TrackingHeap,
                                angle_heap::TrackingHeap, 
                                edge_clock::Vector{Int},
                                edge_anti::Vector{Int},
                                point_is_on_bounds::BitVector,
                                boundary::Matrix{Float64},
                                num_vertices::Int,
                                ve1::Int,
                                ve2::Int,
                                opposite_edge_ve1::Int,
                                model::LikelihoodModel,
                                ind1::Int, 
                                ind2::Int,
                                relative_magnitude::Float64,
                                internal_count::Int,
                                boundary_all::Matrix{Float64},
                                internal_all::Matrix{Float64},
                                p::NamedTuple)

    opposite_edge_ve2 = edge_anti[opposite_edge_ve1] * 1
    polygon_break_and_rejoin!(edge_clock, edge_anti, ve1, ve2, opposite_edge_ve1, opposite_edge_ve2, model, ind1, ind2)

    # In the case we have a new polygon with two vertices, simplest way to handle is to break this two vertex polygon into two 1 vertex polygons, so long as we have another polygon with at least 3 vertices. 
    if ve1 == edge_anti[opposite_edge_ve2]
        polygon_break_and_rejoin!(edge_clock, edge_anti, ve1, opposite_edge_ve2, opposite_edge_ve2, ve1, model, ind1, ind2)
    end
    if opposite_edge_ve1 == edge_anti[ve2]
        polygon_break_and_rejoin!(edge_clock, edge_anti, ve2, opposite_edge_ve1, opposite_edge_ve1, ve2, model, ind1, ind2)
    end

    # update edge length for ve1 and opposite_edge_ve1
    for i in SA[ve1, opposite_edge_ve1]
        TrackingHeaps.update!(edge_heap, i, edge_length(boundary, i, edge_anti[i], relative_magnitude))
    end

    # update angle obj for ve1, ve2, and opposite edge vertices
    for i in SA[ve1, ve2, opposite_edge_ve1, opposite_edge_ve2]
        if point_is_on_bounds[i]
            TrackingHeaps.update!(angle_heap, i, 0.0)
        else
            TrackingHeaps.update!(angle_heap, i, internal_angle_from_pi(i, boundary, edge_clock, edge_anti, relative_magnitude))
        end
    end

    if TrackingHeaps.top(edge_heap)[2] ≤ 0.0
        @warn(string("the number of vertices on the largest level set polygon is less than three at the current step. Algorithm aborting."))

        boundary_all = boundary_all[:, 1:num_vertices]

        if biv_opt_is_ellipse_analytical
            get_ωs_bivariate_ellipse_analytical!(@view(boundary_all[[ind1, ind2], :]), num_points,
                p.q.consistent, ind1, ind2,
                model.core.num_pars, p.q.initGuess,
                p.q.θranges, p.q.ωranges,
                p.options, false, boundary_all)
        end

        if save_internal_points
            ll_values = ll_values[1:internal_count] .+ mle_targetll
            if biv_opt_is_ellipse_analytical
                internal_all = get_ωs_bivariate_ellipse_analytical!(internal_all[[ind1, ind2],1:internal_count],
                                        internal_count,
                                        p.q.consistent, ind1, ind2, 
                                        model.core.num_pars, p.q.initGuess,
                                        p.q.θranges, p.q.ωranges, p.options, false)
            else
                internal_all = internal_all[:, 1:internal_count]
            end
        end

        return true, (boundary_all, PointsAndLogLikelihood(internal_all, ll_values))
    end

    return false, nothing
end

"""
    bivariate_confidenceprofile_iterativeboundary(bivariate_optimiser::Function, 
        model::LikelihoodModel, 
        num_points::Int, 
        consistent::NamedTuple, 
        ind1::Int, 
        ind2::Int,
        dof::Int,
        θlb_nuisance::AbstractVector{<:Real},
        θub_nuisance::AbstractVector{<:Real},
        initial_num_points::Int,
        angle_points_per_iter::Int,
        edge_points_per_iter::Int,
        radial_start_point_shift::Float64,
        ellipse_sqrt_distortion::Float64,
        ellipse_confidence_level::Float64,
        use_ellipse::Bool,
        mle_targetll::Float64,
        save_internal_points::Bool,
        find_zero_atol::Real, 
        optimizationsettings::OptimizationSettings,
        use_threads::Bool,
        channel::RemoteChannel)

Implementation of [`IterativeBoundaryMethod`](@ref).
"""
function bivariate_confidenceprofile_iterativeboundary(bivariate_optimiser::Function, 
                                                model::LikelihoodModel, 
                                                num_points::Int, 
                                                consistent::NamedTuple, 
                                                ind1::Int, 
                                                ind2::Int,
                                                dof::Int,
                                                θlb_nuisance::AbstractVector{<:Real},
                                                θub_nuisance::AbstractVector{<:Real},
                                                initial_num_points::Int,
                                                angle_points_per_iter::Int,
                                                edge_points_per_iter::Int,
                                                radial_start_point_shift::Float64,
                                                ellipse_sqrt_distortion::Float64,
                                                ellipse_confidence_level::Float64,
                                                use_ellipse::Bool,
                                                mle_targetll::Float64,
                                                save_internal_points::Bool,
                                                find_zero_atol::Real, 
                                                optimizationsettings::OptimizationSettings,
                                                use_threads::Bool,
                                                channel::RemoteChannel)

    num_points ≥ initial_num_points || throw(ArgumentError("num_points must be greater than or equal to initial_num_points"))
    newLb, newUb, initGuess, θranges, ωranges = init_nuisance_parameters(model, ind1, ind2, θlb_nuisance, θub_nuisance)

    biv_opt_is_ellipse_analytical = bivariate_optimiser==bivariateψ_ellipse_analytical

    pointa = [0.0,0.0]
    uhat   = [0.0,0.0]
    ω_opt=zeros(model.core.num_pars-2)
    q=(ind1=ind1, ind2=ind2, newLb=newLb, newUb=newUb, initGuess=initGuess, 
        θranges=θranges, ωranges=ωranges, consistent=consistent)

    return_tuple = iterativeboundary_init(bivariate_optimiser, model, num_points, q, ind1, ind2,
                                            initial_num_points, biv_opt_is_ellipse_analytical,
                                            radial_start_point_shift, ellipse_sqrt_distortion,
                                            ellipse_confidence_level, dof,
                                            use_ellipse, save_internal_points, find_zero_atol, 
                                            optimizationsettings, use_threads, channel)

    if return_tuple[1]
        return return_tuple[2], return_tuple[3]
    end

    p = (ω_opt=ω_opt, pointa=pointa, uhat=uhat, q=q, options=optimizationsettings)

    _, boundary, boundary_all, internal_all, ll_values, internal_count, point_is_on_bounds, edge_anti_on_bounds, bound_warning, mle_point, num_vertices, edge_clock, edge_anti, edge_heap, angle_heap, relative_magnitude = return_tuple

    while num_vertices < num_points

        iter_max = min(num_points, num_vertices+angle_points_per_iter)
        while num_vertices < iter_max
            num_vertices += 1

            # candidate vertex
            candidate = TrackingHeaps.top(angle_heap)
            vi = candidate[1] * 1
            adjacents = SA[edge_clock[vi], edge_anti[vi]]
            adjacent_index = argmax(getindex.(Ref(angle_heap), adjacents))
            adj_vertex = adjacents[adjacent_index] * 1 # choose adjacent vertex with the biggest obj angle as candidate edge
            ve1 = adjacent_index==1 ? adj_vertex : vi
            ve2 = edge_anti[ve1] * 1

            num_vertices, internal_count, failure, bound_warning, opposite_edge_ve1 = newboundarypoint!(p, point_is_on_bounds, edge_anti_on_bounds, 
                                                                    boundary, boundary_all,
                                                                    internal_all,
                                                                    ll_values,
                                                                    internal_count,
                                                                    bivariate_optimiser, 
                                                                    model, edge_anti, num_vertices, ind1, ind2, 
                                                                    biv_opt_is_ellipse_analytical, 
                                                                    ve1, ve2, relative_magnitude,
                                                                    bound_warning,
                                                                    save_internal_points,
                                                                    find_zero_atol)

            if failure # appears we have found two distinct level sets - break the edges and join to form two separate polygons
                num_vertices -= 1
                
                termination, return_args = heapupdates_failure!(edge_heap, angle_heap, edge_clock, edge_anti,
                                                                point_is_on_bounds, boundary, num_vertices, ve1, ve2,
                                                                opposite_edge_ve1, model, ind1, ind2,
                                                                relative_magnitude, internal_count, boundary_all, 
                                                                internal_all, p)
                if termination; return return_args end
                continue
            end
            put!(channel, true)
            heapupdates_success!(edge_heap, angle_heap, edge_clock, edge_anti, point_is_on_bounds, edge_anti_on_bounds,
                            boundary, num_vertices, vi, adj_vertex, relative_magnitude, adjacent_index == 1)
            
        end
        if num_vertices == num_points; break end

        iter_max = min(num_points, num_vertices+edge_points_per_iter)
        while num_vertices < iter_max
            num_vertices += 1
            
            # candidate edge
            candidate = TrackingHeaps.top(edge_heap)
            vi = candidate[1] * 1
            adj_vertex = edge_anti[vi] * 1

            num_vertices, internal_count, failure, bound_warning, opposite_edge_ve1 = newboundarypoint!(p, point_is_on_bounds, edge_anti_on_bounds, 
                                                                    boundary, boundary_all, 
                                                                    internal_all,
                                                                    ll_values,
                                                                    internal_count, bivariate_optimiser, 
                                                                    model, edge_anti, num_vertices, ind1, ind2, 
                                                                    biv_opt_is_ellipse_analytical, 
                                                                    vi, adj_vertex, relative_magnitude,
                                                                    bound_warning,
                                                                    save_internal_points,
                                                                    find_zero_atol)
            
            if failure # appears we have found two distinct level sets - break the edges and join to form two separate polygons
                ve1 = vi
                ve2 = adj_vertex
                num_vertices -= 1
                
                termination, return_args = heapupdates_failure!(edge_heap, angle_heap, edge_clock, edge_anti,
                                                                point_is_on_bounds, boundary, num_vertices, ve1, ve2,
                                                                opposite_edge_ve1, model, ind1, ind2,
                                                                relative_magnitude, internal_count, boundary_all, 
                                                                internal_all, p)
                if termination; return return_args end
                continue
            end        
            put!(channel, true)
            heapupdates_success!(edge_heap, angle_heap, edge_clock, edge_anti, point_is_on_bounds, edge_anti_on_bounds,
                            boundary, num_vertices, vi, adj_vertex, relative_magnitude)
        end
    end

    if biv_opt_is_ellipse_analytical
        get_ωs_bivariate_ellipse_analytical!(@view(boundary_all[[ind1, ind2], :]), num_points,
                                                consistent, ind1, ind2, 
                                                model.core.num_pars, initGuess,
                                                θranges, ωranges, 
                                                optimizationsettings, false, boundary_all)
    end

    if save_internal_points
        ll_values = ll_values[1:internal_count] .+ mle_targetll
        if biv_opt_is_ellipse_analytical
            internal_all = get_ωs_bivariate_ellipse_analytical!(internal_all[[ind1, ind2],1:internal_count],
                                    internal_count, consistent, ind1, ind2, 
                                    model.core.num_pars, initGuess,
                                    θranges, ωranges, optimizationsettings, false)
        else
            internal_all = internal_all[:, 1:internal_count]
        end
    end

    return boundary_all, PointsAndLogLikelihood(internal_all, ll_values)
end