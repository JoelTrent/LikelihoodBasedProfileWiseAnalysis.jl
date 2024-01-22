"""
    construct_polygon_hull(model::LikelihoodModel,
        θindices::Vector{<:Int},
        conf_struct::BivariateConfidenceStruct,
        confidence_level::Float64,
        dof::Int,
        boundary_not_ordered::Bool,
        hullmethod::AbstractBivariateHullMethod,
        return_boundary_not_mesh::Bool)

Constructs a 2D polygon hull that represents an approximation of a true bivariate log-likelihood confidence boundary, given boundary points and saved internal points in `conf_struct`, using `hullmethod`. Optionally returns the boundary as an ordered 2*n array or as a [`SimpleMesh`](https://juliageometry.github.io/Meshes.jl/stable/domains/meshes.html#Meshes.SimpleMesh). For a description of the algorithms used for each [`AbstractBivariateHullMethod`](@ref) see their docstrings: [`ConvexHullMethod`](@ref), [`ConcaveHullMethod`](@ref) and [`MPPHullMethod`](@ref).
"""
function construct_polygon_hull(model::LikelihoodModel,
                                θindices::Vector{<:Int},
                                conf_struct::BivariateConfidenceStruct,
                                confidence_level::Float64,
                                dof::Int,
                                boundary_not_ordered::Bool,
                                hullmethod::AbstractBivariateHullMethod,
                                return_boundary_not_mesh::Bool)

    length(θindices) == 2 || throw(ArgumentError("θindices must have length 2"))

    if hullmethod isa ConvexHullMethod
        point_union = hcat(conf_struct.confidence_boundary[θindices, :], conf_struct.internal_points.points[θindices, :])
        pset = PointSet(point_union)
        mesh = convexhull(pset)

        if !return_boundary_not_mesh; return mesh end
        
        boundary = reduce(hcat, [point.coords for point in collect(mesh.outer.vertices)])
        return boundary
    end

    if hullmethod isa MPPHullMethod
        boundary = conf_struct.confidence_boundary[θindices, :]
        if boundary_not_ordered
            minimum_perimeter_polygon!(boundary)
        end

    elseif hullmethod isa ConcaveHullMethod
        point_union = hcat(conf_struct.confidence_boundary[θindices, :], conf_struct.internal_points.points[θindices, :])

        num_boundary_points = size(conf_struct.confidence_boundary, 2)
        ll_boundary = get_target_loglikelihood(model, confidence_level, EllipseApprox(), dof)
        ll_values = vcat(fill(ll_boundary, num_boundary_points), conf_struct.internal_points.ll)

        boundary = bivariate_concave_hull(point_union, ll_values, 0.8, 0.8, ll_boundary)
    end

    if return_boundary_not_mesh; return boundary end

    n = size(boundary, 2)
    mesh = SimpleMesh([(boundary[1, i], boundary[2, i]) for i in 1:n], [connect(tuple(1:n...))])
    
    return mesh
end