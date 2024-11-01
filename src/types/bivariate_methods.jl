

"""
    AbstractBivariateMethod

Supertype for bivariate boundary finding methods. Use `bivariate_methods()` for a list of available methods (see [`bivariate_methods`](@ref)).

# Subtypes

[`AbstractBivariateVectorMethod`](@ref)

[`CombinedBivariateMethod`](@ref)

[`Fix1AxisMethod`](@ref)

[`AnalyticalEllipseMethod`](@ref)
"""
abstract type AbstractBivariateMethod end

"""
    AbstractBivariateVectorMethod <: AbstractBivariateMethod

Supertype for bivariate boundary finding methods that search between two distinct points. 

# Subtypes

[`IterativeBoundaryMethod`](@ref)

[`RadialMLEMethod`](@ref)

[`RadialRandomMethod`](@ref)

[`SimultaneousMethod`](@ref)

# Supertype Hiearachy

`AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any`
"""
abstract type AbstractBivariateVectorMethod <: AbstractBivariateMethod end

"""
    CombinedBivariateMethod()

A method representing a [`BivariateConfidenceStruct`](@ref) that has been destructively merged from one or more boundaries found with a [`AbstractBivariateMethod`](@ref). Does not represent a method usable by [`bivariate_confidenceprofiles!`](@ref).
"""
struct CombinedBivariateMethod <: AbstractBivariateMethod end

"""
    IterativeBoundaryMethod(initial_num_points::Int, 
        angle_points_per_iter::Int, 
        edge_points_per_iter::Int, 
        radial_start_point_shift::Float64=rand(), 
        ellipse_sqrt_distortion::Float64=1.0;
        use_ellipse::Bool=false)

Method for iteratively improving an initial boundary of `initial_num_points`, found by pushing out from the MLE point in directions defined by either [`RadialMLEMethod`](@ref), when `use_ellipse=true`, or [`RadialRandomMethod`](@ref), when `use_ellipse=false` (see [`LikelihoodBasedProfileWiseAnalysis.findNpointpairs_radialMLE!`](@ref), [`LikelihoodBasedProfileWiseAnalysis.iterativeboundary_init`](@ref) and [`LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile_iterativeboundary`](@ref)).

# Arguments
- `initial_num_points`: a positive integer number of initial boundary points to find. 
- `angle_points_per_iter`: a integer ≥ 0 for the number of edges to explore and improve based on the angle objective before switching to the edge objective. If `angle_points_per_iter > 0` and `edge_points_per_iter > 0` the angle objective is considered first.
- `edge_points_per_iter`:  a integer ≥ 0 for the number of edges to explore and improve based on the edge objective before switching back to the angle objective. If `angle_points_per_iter > 0` and `edge_points_per_iter > 0` the angle objective is considered first.
- `radial_start_point_shift`: a number ∈ [0.0,1.0]. Default is `rand()` (defined on [0.0,1.0]), meaning that by default a different set of initial points will be found each time.
- `ellipse_sqrt_distortion`: a number ∈ [0.0,1.0]. Default is `0.01`. 

# Keyword Arguments
- `use_ellipse`: Whether to find `initial_num_points` by searching in directions defined by an ellipse approximation, as in [`RadialMLEMethod`](@ref), or as in [`RadialRandomMethod`](@ref). Default is `false`.

# Details

For additional information on the `radial_start_point_shift` and `ellipse_sqrt_distortion` arguments see the keyword arguments for [`generate_N_clustered_points`](https://joeltrent.github.io/EllipseSampling.jl/stable/user_interface/#EllipseSampling.generate_N_clustered_points) in [EllipseSampling.jl](https://joeltrent.github.io/EllipseSampling.jl/stable).

Recommended for use with the [`LogLikelihood`](@ref) profile type. Radial directions, edge length and internal angle calculations are rescaled by the relative magnitude/scale of the two interest parameters. This is so directions and regions explored and consequently the boundary found are not dominated by the parameter with the larger magnitude.

Once an initial boundary is found by pushing out from the MLE point in directions defined by either [`RadialMLEMethod`](@ref) or [`RadialRandomMethod`](@ref), the method seeks to improve this boundary by minimising an internal angle and an edge length objective, each considered sequentially, until the desired number of boundary points are found. As such, the method can be thought of as a mesh smoothing or improvement algorithm; we can consider the boundary found at a given moment in time to be a N-sided polygon with edges between adjacent boundary points (vertices). 

# Extended help

Regions we define as needing improvement are those with:

1. Internal angles between adjacent edges that are far from being straight (180 degrees or π radians). The region defined by these edges is not well explored as the real boundary edge in this region likely has some degree of smooth curvature. It may also indicate that one of these edges cuts our boundary into a enclosed region and an unexplored region on the other side of the edge. In the event that a vertex is on a user-provided bound for a parameter, this objective is set to zero, as this angle region is a byproduct of user input and not the actual log-likelihood region. This objective is defined in [`LikelihoodBasedProfileWiseAnalysis.internal_angle_from_pi!`](@ref).
2. Edges between adjacent vertices that have a large euclidean distance. The regions between these vertices is not well explored as it is unknown whether the boundary actually is straight between these vertices. On average the closer two vertices are, the more likely the edge between the two points is well approximated by a straight line, and thus our mesh is a good representation of the true log-likelihood boundary. This objective is defined in [`LikelihoodBasedProfileWiseAnalysis.edge_length`](@ref).

The method is implemented as follows:
1. Create edges between adjacent vertices on the intial boundary. Calculate angle and edge length objectives for these edges and vertices and place them into tracked heaps.
2. Until found the desired number of boundary points repeat steps 3 and 4.
3. For `angle_points_per_iter` points:
    1. Peek at the top vertex of the angle heap.
    2. Place a candidate point in the middle of the edge connected to this vertex that has the larger angle at the vertex the edge connects to. This is so we explore the worse defined region of the boundary.
    3. Use this candidate to find a new boundary point (see below).
    4. If found a new boundary point, break edge we placed candidate on and replace with edges to the new boundary point, updating angle and edge length objectives in the tracked heap (see [`LikelihoodBasedProfileWiseAnalysis.heapupdates_success!`](@ref)). Else break our polygon into multiple polygons (see [`LikelihoodBasedProfileWiseAnalysis.heapupdates_failure!`](@ref)).
4. For `edge_points_per_iter` points:
    1. Peek at the top edge of the edge heap.
    2. Place a candidate point in the middle of this edge.
    3. Same as for step 3.3.
    4. Same as for step 3.4.

!!! note "angle_points_per_iter and edge_points_per_iter"

    At least one of `angle_points_per_iter` and `edge_points_per_iter` must be non-zero.

!!! note "Using a candidate point to find a new boundary point"

    Uses [`LikelihoodBasedProfileWiseAnalysis.newboundarypoint!`](@ref).

    If a candidate point is on the log-likelihood threshold boundary, we accept the point.

    If a candidate point is inside the boundary, then we search in the normal direction to the edge until we find a boundary point or hit the parameter bound, accepting either.
	
    If a candidate point is outside the boundary we find the edge, `e_intersect` of our boundary polygon that is intersected by the line in the normal direction of the candidate edge, which passes through the candidate point. Once this edge is found, we find the vertex on `e_intersect` that is closest to our candidate point, subject to that vertex not also being on the candidate point edge. We setup a 1D line search/bracketing method between these two points. In the event that no boundary points are found between these two points it is likely that multiple boundaries exist. If so, we break the candidate point's edge and `e_intersect` and reconnect the vertexes such that we now have multiple boundary polygons.

!!! warning "Largest boundary polygon at any iteration must have at least three points"

    If the largest polygon has less than two points the method will display a warning message and terminate, returning the boundary found up until then. 

# Boundary finding method

For the initial boundary, it uses a 1D bracketing algorithm between the MLE point and the point on the user-provided bounds in the search directions to find the boundary at the desired confidence level. For boundary improvement, if the candidate point is inside the boundary, it uses a 1D bracketing algorithm between an internal point and the point on the user-provided bounds in the search direction. If the candidate point is outside the boundary, it uses a derivative-free 1D line search algorithm in the search direction. If the derivative-free approach fails, it switches to a bracketing heuristic between the candidate point and closest vertex on `e_intersect`.

This method is unlikely to find boundaries that do not contain the MLE point (if they exist), but it can find them. If a boundary that does not contain the MLE point is found, it is not guaranteed to be explored. In this case the the method will inform the user that multiple boundaries likely exist for this combination of model parameters.

## Impact of parameter bounds

If a parameter bound is in the way of reaching the boundary in a given search direction, the point on that bound will be returned as the boundary point. For the bracketing method to work, parameter values on the bounds need to be legal and return a non `Inf` value from the log-likelihood function. Interest parameter bounds that have ranges magnitudes larger than the range of the boundary may prevent the true boundary from being found. Additionally, the bracketing method will be less efficient if the interest parameter bounds are far from the true boundary. 

# Threaded implementation

This method is partially implemented with Threads parallelisation if `use_threads` is set to true when calling [`bivariate_confidenceprofiles!`](@ref). The initial boundary step is partially parallelised (the finding point pairs step is not parallelised and the finding boundary from point pairs step is parallelised) while the boundary improvement steps are not.

# Internal Points

Finds between 0 and `num_points - initial_num_points` internal points - internal points are found when the edge being considered's midpoint is inside the boundary. 

# Supertype Hiearachy

`IterativeBoundaryMethod <: AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any`
"""
struct IterativeBoundaryMethod <: AbstractBivariateVectorMethod
    initial_num_points::Int
    angle_points_per_iter::Int
    edge_points_per_iter::Int
    radial_start_point_shift::Float64
    ellipse_sqrt_distortion::Float64
    use_ellipse::Bool
    function IterativeBoundaryMethod(initial_num_points::Int, angle_points_per_iter::Int, edge_points_per_iter::Int, radial_start_point_shift::Float64=rand(), ellipse_sqrt_distortion::Float64=1.0; 
        use_ellipse::Bool=false)
    
        initial_num_points > 0 || throw(DomainError("initial_num_points must be greater than zero"))
        angle_points_per_iter ≥ 0 || throw(DomainError("angle_points_per_iter must be greater than or equal to zero"))
        edge_points_per_iter ≥ 0 || throw(DomainError("edge_points_per_iter must be greater than or equal zero"))
        angle_points_per_iter > 0 || edge_points_per_iter > 0 || throw(DomainError("at least one of angle_points_per_iter and edge_points_per_iter must be greater than zero"))
        (0.0 <= radial_start_point_shift && radial_start_point_shift <= 1.0) || throw(DomainError("radial_start_point_shift must be in the closed interval [0.0,1.0]"))
        (0.0 <= ellipse_sqrt_distortion && ellipse_sqrt_distortion <= 1.0) || throw(DomainError("ellipse_sqrt_distortion must be in the closed interval [0.0,1.0]"))
        return new(initial_num_points, angle_points_per_iter, edge_points_per_iter, radial_start_point_shift, ellipse_sqrt_distortion,
            use_ellipse)
    end
end


"""
    RadialMLEMethod(ellipse_start_point_shift::Float64=rand(), 
        ellipse_sqrt_distortion::Float64=0.01)

Method for finding the bivariate boundary of a confidence profile by bracketing between the MLE point and points on the provided bounds in directions given by points found on the boundary of a ellipse approximation of the log-likelihood function around the MLE, `e`, using [EllipseSampling.jl](https://joeltrent.github.io/EllipseSampling.jl/stable) (see [`LikelihoodBasedProfileWiseAnalysis.findNpointpairs_radialMLE!`](@ref) and [`LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile_vectorsearch`](@ref)). 

# Arguments
- `ellipse_start_point_shift`: a number ∈ [0.0,1.0]. Default is `rand()` (defined on [0.0,1.0]), meaning that by default a different set of points will be found each time.
- `ellipse_sqrt_distortion`: a number ∈ [0.0,1.0]. Default is `0.01`. 

# Details

As for [`univariate_confidenceintervals!`](@ref) with keyword argument `use_ellipse_approx_analytical_start=true`, the profile log-likelihood function will be evaluated at each ellipse point prior to creating the point pair bracket. If:

1. the ellipse point is outside of the provided bounds, the MLE point and the point on the provided bounds in the direction of the ellipse point from the MLE is the point pair for bracketing.
2. the ellipse point is between the desired boundary and the provided bounds, the MLE point and the ellipse point is the point pair for bracketing.
3. the ellipse point is between the MLE point and the desired boundary, the ellipse point and the point on the provided bounds is the point pair for bracketing.
4. the ellipse point is exactly on the desired boundary (profile log-likelihood function evaluates to exactly the confidence level threshold), the ellipse point is the boundary point and the method will (wrongfully) state there's a point on the bounds. From a numerical perspective this is incredibly unlikely so this is regarded as not a big deal. 

For additional information on arguments see the keyword arguments for [`generate_N_clustered_points`](https://joeltrent.github.io/EllipseSampling.jl/stable/user_interface/#EllipseSampling.generate_N_clustered_points) in [EllipseSampling.jl](https://joeltrent.github.io/EllipseSampling.jl/stable).

Recommended for use with the [`EllipseApprox`](@ref) profile type. Will produce reasonable results for the [`LogLikelihood`](@ref) profile type when bivariate profile boundaries are convex. Otherwise, [`IterativeBoundaryMethod`](@ref), which has an option to use a starting solution from [`RadialMLEMethod`](@ref), is preferred as it iteratively improves the quality of the boundary and can discover regions not explored by this method. 
    
# Extended help

# Boundary finding method

Uses a 1D bracketing algorithm between the MLE point and the point on the user-provided bounds in the search directions to find the boundary at the desired confidence level. Will replace one of these sides with the ellipse point depending on which side of the boundary the ellipse point is on, subject to it being between the two points. 

This method is unlikely to find boundaries that do not contain the MLE point (if they exist).

## Impact of parameter bounds

If a parameter bound is in the way of reaching the boundary in a given search direction, the point on that bound will be returned as the boundary point. For the bracketing method to work, parameter values on the bounds need to be legal and return a non `Inf` value from the log-likelihood function. Interest parameter bounds that have ranges magnitudes larger than the range of the boundary may prevent the true boundary from being found. Additionally, the bracketing method will be less efficient if the interest parameter bounds are far from the true boundary. 

# Threaded implementation

This method is partially implemented with Threads parallelisation if `use_threads` is set to true when calling [`bivariate_confidenceprofiles!`](@ref). The finding point pairs step is not parallelised and the finding boundary from point pairs step is parallelised.

# Internal Points

Finds no internal points.

# Supertype Hiearachy

`RadialMLEMethod <: AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any`
"""
struct RadialMLEMethod <: AbstractBivariateVectorMethod
    ellipse_start_point_shift::Float64
    ellipse_sqrt_distortion::Float64

    function RadialMLEMethod(ellipse_start_point_shift::Float64=rand(), ellipse_sqrt_distortion::Float64=0.01) 
        (0.0 <= ellipse_start_point_shift && ellipse_start_point_shift <= 1.0) || throw(DomainError("ellipse_start_point_shift must be in the closed interval [0.0,1.0]"))
        (0.0 <= ellipse_sqrt_distortion && ellipse_sqrt_distortion <= 1.0) || throw(DomainError("ellipse_sqrt_distortion must be in the closed interval [0.0,1.0]"))
        return new(ellipse_start_point_shift, ellipse_sqrt_distortion)
    end
end

"""
    RadialRandomMethod(num_radial_directions::Int, use_MLE_point::Bool=true)

Method for finding the bivariate boundary of a confidence profile by finding internal boundary points using a uniform random distribution on provided bounds and choosing `num_radial_directions` to explore from these points (see [`LikelihoodBasedProfileWiseAnalysis.findNpointpairs_radialrandom!`](@ref) and [`LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile_vectorsearch`](@ref)).

# Arguments
- `num_radial_directions`: an integer greater than zero. 
- `use_MLE_point`: whether to use the MLE point as the first internal point found or not. Default is true (use).

# Details

Recommended for use with the [`LogLikelihood`](@ref) profile type. Radial directions are rescaled by the relative magnitude/scale of the two interest parameters. This is so directions explored and consequently the boundary found are not dominated by the parameter with the larger magnitude.

The method first uniformly samples the region specified by the bounds for the two interest parameters until a point within the boundary is found. Then it chooses `num_radial_directions`, spaced uniformly around a circle, and rotates these directions by a random phase shift ∈ `[0.0, 2π/num_directions]` radians. These directions are then distorted by the relative magnitude/scale of the two interest parameters. Then for each of these directions, until it runs out of directions or finds the desired number of points, it finds the closest point on the bounds from the internal point in this direction. If the point on the bounds is inside the boundary it will be used in place of the boundary point. If the point on the bounds is outside the boundary then it will be used as the external point in the point pair. A bracketing method is then used to find a boundary point between the point pair (the bound point and the internal point). The method continues searching for internal points and generating directions until the desired number of boundary points is found.
    
Given a fixed number of desired boundary points, we can decrease the cost of finding internal points by increasing the number of radial directions to explore, `num_radial_directions`, at each internal point. However, it is important to balance `num_radial_directions` with the desired number of boundary points. If `num_radial_directions` is large relative to the number of boundary points, then the boundary the method finds may constitute a local search around the found internal points. Resultantly, there may be regions were the boundary is not well explored. This will be less of an issue for more 'circular' boundary regions.   

[`IterativeBoundaryMethod`](@ref) may be preferred over this method if evaluating the log-likelihood function is expensive or the bounds provided for the interest parameters are much larger than the boundary, as the uniform random sampling for internal points will become very expensive. 

# Extended help

# Boundary finding method

Uses a 1D bracketing algorithm between an internal point and the point on the user-provided bounds in the search direction to find the boundary at the desired confidence level.  

This method can find multiple boundaries (if they exist).

## Impact of parameter bounds

If a parameter bound is in the way of reaching the boundary in a given search direction, in the same way as [`RadialMLEMethod`](@ref), the point on that bound will be returned as the boundary point. For the bracketing method to work, parameter values on the bounds need to be legal and return a non `Inf` value from the log-likelihood function. Interest parameter bounds that have ranges magnitudes larger than the range of the boundary may prevent the true boundary from being found. Additionally, the bracketing method will be less efficient if the interest parameter bounds are far from the true boundary.

# Threaded implementation

This method is partially implemented with Threads parallelisation if `use_threads` is set to true when calling [`bivariate_confidenceprofiles!`](@ref). The finding point pairs step is not parallelised and the finding boundary from point pairs step is parallelised.

# Internal Points

Finds a minimum of `div(num_points, num_radial_directions, RoundUp) - 1*use_MLE_point` internal points.

# Supertype Hiearachy

`RadialRandomMethod <: AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any`
"""
struct RadialRandomMethod <: AbstractBivariateVectorMethod
    num_radial_directions::Int
    use_MLE_point::Bool
    function RadialRandomMethod(num_radial_directions::Int, use_MLE_point::Bool=true)
        num_radial_directions > 0 ||  throw(DomainError("num_radial_directions must be greater than zero")) 
        return new(num_radial_directions, use_MLE_point)
    end
end

"""
    SimultaneousMethod(min_proportion_unique::Real=0.5, include_MLE_point::Bool=true)

Method for finding the bivariate boundary of a confidence profile by finding internal and external boundary points using a uniform random distribution on provided bounds, pairing these points in the order they're found and bracketing between each pair (see [`LikelihoodBasedProfileWiseAnalysis.findNpointpairs_simultaneous!`](@ref) and [`LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile_vectorsearch`](@ref)). Values of `min_proportion_unique` lower than zero may improve performance if finding either internal points or external points is difficult given the specified bounds on interest parameters.
    

# Arguments
- `min_proportion_unique`: a proportion ∈ (0.0, 1.0] for the minimum allowed proportion of unique points in one of the internal or external point vectors. One of these vectors will be fully unique. Whichever vector is not unique, will have at least `min_proportion_unique` unique points. Default is 0.5.
- `use_MLE_point`: whether to use the MLE point as the first internal point found or not. Default is true (use).

# Details

Recommended for use with the [`LogLikelihood`](@ref) profile type. 

When `min_proportion_unique = 1.0` the method uniformly samples the region specified by the bounds for the two interest parameters until as many internal and external boundary points as the desired number of boundary points are found. These points are paired in the order they are found. 

If `min_proportion_unique < 1.0` the method uniformly samples the region specified by the bounds for the two interest parameters until either the internal or external number of boundary points found is as many as the number of desired boundary points. For whichever vector is less than the number of desired boundary points, we keep searching for that kind of point until at least `ceil(min_proportion_unique * num_points)` points have been found. Once `ceil(min_proportion_unique * num_points)` points have been found these points are used to fill the remainder of the vector, in order found. Internal and external points in each vector are then paired in the order they are found.

A bracketing method is then used to find a boundary point between each point pair (the external point and the internal point).

[`RadialRandomMethod`](@ref) and [`IterativeBoundaryMethod`](@ref) are preferred over this method from a computational performance standpoint as they require fewer log-likelihood evalutions (when [`RadialRandomMethod`](@ref) has parameter `num_radial_directions` > 1). 

# Extended help

# Boundary finding method

Uses a 1D bracketing algorithm between a valid point pair to find the boundary at the desired confidence level.  

This method can find multiple boundaries (if they exist).

## Impact of parameter bounds

If a parameter bound is in the way of reaching the boundary, points will not be put on that bound. Additionally, if the true boundary is very close to a parameter bound, the method will struggle to find this region of the boundary. This is because finding the boundary in this location requires generating a random point between the boundary and the parameter bound, which becomes more difficult the closer they are. 

Interest parameter bounds that have ranges magnitudes larger than the range of the boundary will make finding internal points difficult, requiring a lot of computational effort. Similarly, the inverse will be true if external points are hard to find. Smaller values of `min_proportion_unique` will improve performance in these cases. 

The method will fail if the interest parameter bounds are fully contained by the boundary.

# Threaded implementation

This method is partially implemented with Threads parallelisation if `use_threads` is set to true when calling [`bivariate_confidenceprofiles!`](@ref). The finding point pairs step is not parallelised and the finding boundary from point pairs step is parallelised.

# Internal Points

Finds at least `ceil(min_proportion_unique*num_points) - 1*use_MLE_point` internal points.

# Supertype Hiearachy

`SimultaneousMethod <: AbstractBivariateVectorMethod <: AbstractBivariateMethod <: Any`
"""
struct SimultaneousMethod <: AbstractBivariateVectorMethod 
    min_proportion_unique::Real
    use_MLE_point::Bool
    function SimultaneousMethod(min_proportion_unique::Real=0.5, use_MLE_point::Bool=true)
        (0.0 < min_proportion_unique && min_proportion_unique <= 1.0) || throw(DomainError("min_proportion_unique must be in the interval (0.0,1.0]"))
        return new(min_proportion_unique, use_MLE_point)
    end
end

"""
    Fix1AxisMethod()

Method for finding the bivariate boundary of a confidence profile by using uniform random distributions on provided bounds to draw a value for one interest parameter, fix it, and draw two values for the other interest parameter, using the pair to find a boundary point using a bracketing method if the pair are a valid bracket (see [`LikelihoodBasedProfileWiseAnalysis.findNpointpairs_fix1axis!`](@ref) and [`LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile_fix1axis`](@ref)).

# Details

Recommended for use with the [`LogLikelihood`](@ref) profile type. 

The method finds the desired number of boundary points by fixing the first and second interest parameters for half of these points each. It first draws a value for the fixed parameter using a uniform random distribution on provided bounds (e.g. `ψ_x`). Then it draws two values for the unfixed parameter in the same fashion (e.g. `ψ_y1` and `ψ_y2`]). If one of these points (`[ψ_x, ψ_y1]` and `[ψ_x, ψ_y2]`) is an internal point and the other an external point, the point pair is accepted as they are a valid bracket. A bracketing method is then used to find a boundary point between the point pair (the internal and external point). The method continues searching for valid point pairs until the desired number of boundary points is found.

[`RadialRandomMethod`](@ref) and [`IterativeBoundaryMethod`](@ref) are preferred over this method from a computational performance standpoint as they require fewer log-likelihood evalutions (when [`RadialRandomMethod`](@ref) has parameter `num_radial_directions` > 1). [`SimultaneousMethod`](@ref) will also likely be more efficient, even though it uses four random numbers vs three, as it doesn't unneccesarily throw away points.  

# Extended help

# Boundary finding method

Uses a 1D bracketing algorithm between a valid point pair, which are parallel to the x or y axis, to find the boundary at the desired confidence level.  

This method can find multiple boundaries (if they exist).

## Impact of parameter bounds

If a parameter bound is in the way of reaching the boundary, points will not be put on that bound. Additionally, if the true boundary is very close to a parameter bound, the method will struggle to find this region of the boundary. This is because finding the boundary in this location requires generating a random point between the boundary and the parameter bound, which becomes more difficult the closer they are. Interest parameter bounds that have ranges magnitudes larger than the range of the boundary will make finding internal points very difficult, requiring a lot of computational effort. Similarly, the inverse will be true if external points are hard to find. The method will fail if the interest parameter bounds are fully contained by the boundary.

# Threaded implementation

This method is implemented with Threads parallelisation if `use_threads` is set to true when calling [`bivariate_confidenceprofiles!`](@ref).

# Internal Points

Finds `num_points` internal points.

# Supertype Hiearachy

`Fix1AxisMethod <: AbstractBivariateMethod <: Any`
"""
struct Fix1AxisMethod <: AbstractBivariateMethod end

"""
    AnalyticalEllipseMethod(ellipse_start_point_shift::Float64, 
        ellipse_sqrt_distortion::Float64)

Method for sampling the desired number of boundary points on a ellipse approximation of the log-likelihood function centred at the maximum likelihood estimate point using [EllipseSampling.jl](https://joeltrent.github.io/EllipseSampling.jl/stable).

# Arguments
- `ellipse_start_point_shift`: a number ∈ [0.0,1.0]. Default is `rand()` (defined on [0.0,1.0]), meaning that by default a different set of points will be found each time.
- `ellipse_sqrt_distortion`: a number ∈ [0.0,1.0]. Default is `1.0`, meaning that by default points on the ellipse approximation are equally spaced with respect to arc length. 

# Details

Used for the [`EllipseApproxAnalytical`](@ref) profile type only: if this method is specified, then any user provided profile type will be overriden and replaced with [`EllipseApproxAnalytical`](@ref). This ellipse approximation ignores user provided bounds.

For additional information on arguments see the keyword arguments for [`generate_N_clustered_points`](https://joeltrent.github.io/EllipseSampling.jl/stable/user_interface/#EllipseSampling.generate_N_clustered_points) in [EllipseSampling.jl](https://joeltrent.github.io/EllipseSampling.jl/stable).
    
# Extended help

# Boundary finding method

Explicitly finds the boundary using [EllipseSampling.jl](https://joeltrent.github.io/EllipseSampling.jl/stable).

# Internal Points

Finds no internal points.

# Supertype Hiearachy

`AnalyticalEllipseMethod <: AbstractBivariateMethod <: Any`
"""
struct AnalyticalEllipseMethod <: AbstractBivariateMethod 
    ellipse_start_point_shift::Float64
    ellipse_sqrt_distortion::Float64
    function AnalyticalEllipseMethod(ellipse_start_point_shift::Float64=rand(), ellipse_sqrt_distortion::Float64=1.0)
        (0.0 <= ellipse_start_point_shift && ellipse_start_point_shift <= 1.0) || throw(DomainError("ellipse_start_point_shift must be in the closed interval [0.0,1.0]"))
        (0.0 <= ellipse_sqrt_distortion && ellipse_sqrt_distortion <= 1.0) || throw(DomainError("ellipse_sqrt_distortion must be in the closed interval [0.0,1.0]"))
        return new(ellipse_start_point_shift, ellipse_sqrt_distortion)
    end
end


"""
    bivariate_methods()

Prints a list of available bivariate methods. Available bivariate methods include [`IterativeBoundaryMethod`](@ref), [`RadialRandomMethod`](@ref), [`RadialMLEMethod`](@ref), [`SimultaneousMethod`](@ref), [`Fix1AxisMethod`](@ref) and [`AnalyticalEllipseMethod`](@ref). Note: [`CombinedBivariateMethod`](@ref) represents a destructive merge of the boundary structs of one or more methods and is not a valid method for [`bivariate_confidenceprofiles!`](@ref).
"""
function bivariate_methods()
    methods = [IterativeBoundaryMethod, RadialRandomMethod, RadialMLEMethod, SimultaneousMethod, Fix1AxisMethod, AnalyticalEllipseMethod]
    println(string("Available bivariate methods are: ", [i != length(methods) ? string(method, ", ") : string(method) for (i,method) in enumerate(methods)]...))
    return nothing
end