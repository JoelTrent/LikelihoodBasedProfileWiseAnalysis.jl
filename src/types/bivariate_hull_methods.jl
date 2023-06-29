"""
    AbstractBivariateMethod

Supertype for bivariate boundary hull methods. Use `bivariate_hull_methods()` for a list of available methods (see [`bivariate_hull_methods`](@ref)).

# Subtypes

[`ConvexHullMethod`](@ref)

[`ConcaveHullMethod`](@ref)

[`MPPHullMethod`](@ref)
"""
abstract type AbstractBivariateHullMethod end

"""
    ConvexHullMethod()

Construct a 2D polygon hull to sample internal points from by applying a convex hull algorithm from [Meshes.jl](https://docs.juliahub.com/Meshes/FuRcu/0.18.2/algorithms/hulls.html#Meshes.GrahamScan) on the collection of points given by both the boundary and saved internal points. 

# Details 

## Representation Accuracy

For convex boundaries, this method has the ability to create a more accurate representation of the theoretical boundary than [`MPPHullMethod`](@ref), if saved internal points contain information about the boundary that is not covered by a convex hull of boundary points alone. For concave boundaries, this method will not be an accurate representation of the theoretical boundary.

## Rejection Rate when Sampling Internal Points

The rejection rate when sampling internal points will be low for convex boundaries. The rejection rate may become very high for concave boundaries, if the area of the convex hull of the theoretical boundary is much larger than the area of the theoretical boundary. 

# Supertype Hiearachy

`ConvexHullMethod <: AbstractBivariateHullMethod <: Any`
"""
struct ConvexHullMethod <: AbstractBivariateHullMethod end

"""
    ConcaveHullMethod()

Construct a 2D polygon hull to sample internal points from by applying a heuristic implementation of a heuristic concave hull algorithm from [ConcaveHull.jl](https://github.com/lstagner/ConcaveHull.jl) on the collection of points given by both the boundary and saved internal points (see [`PlaceholderLikelihood.bivariate_concave_hull`](@ref)). 

# Details 

It applies the `ConcaveHull.concave_hull` algorithm twice to the union of boundary and saved internal points, with the number of neighbours for the first pass chosen using a heuristic based on the number of points in the point union. Resultantly, it may result in more accurate coverage of the theoretical boundary than [`MPPHullMethod`](@ref), for smaller numbers of boundary points, if saved internal points are in locations not enclosed by a polygon found using only boundary points. For example, if the theoretical boundary is a square and there is an internal point in the bottom left corner, but no boundary points around that corner, the boundary polygon created by this method will likely have a vertex at that internal point. However, this is not guaranteed because it is a heuristic. Bivariate methods that struggle to find boundaries close to or on the other side of a parameter bound are an example where using information on saved internal points will prove useful.

## Representation Accuracy

This method has the ability to create a more accurate representation of the boundary than [`MPPHullMethod`](@ref), but because of it's heuristic nature this is not guaranteed. However, it should be a much more accurate representation of the boundary than [`ConvexHullMethod`](@ref) for non-convex boundaries (concave boundaries).

## Rejection Rate when Sampling Internal Points

The rejection rate when sampling internal points will be low for convex boundaries. The rejection rate should be low for concave boundaries as well, but the nature of the heuristic used may cause concave sections to be treated as convex.   

# Supertype Hiearachy

`ConcaveHullMethod <: AbstractBivariateHullMethod <: Any`
"""
struct ConcaveHullMethod <: AbstractBivariateHullMethod end

"""
    MPPHullMethod()

Construct a 2D polygon hull to sample internal points from by applying a minimum perimeter polygon (MPP) traveling salesman problem algorithm to the boundary (see [`PlaceholderLikelihood.minimum_perimeter_polygon!`](@ref)). 

# Details

It does not use information on the position of internal points saved while finding a boundary. This may result in less accurate coverage of the theoretical boundary for smaller numbers of boundary points. For example, if the theoretical boundary is a square and there is an internal point in the bottom left corner, but no boundary points around that corner, the boundary polygon created by this method will not enclose the area around that corner. Bivariate methods that struggle to find boundaries close to or on the other side of a parameter bound are an example where using information on saved internal points would prove useful - [`ConcaveHullMethod`](@ref) may be more appropriate in these cases.

## Representation Accuracy

This method will create the most accurate representation of the boundary that has been found from any of the [`AbstractBivariateHullMethod`](@ref) methods, particularly for non-convex boundaries, given a sufficient number of boundary points.

## Rejection Rate when Sampling Internal Points

The rejection rate when sampling internal points will be low for convex and concave boundaries. 

# Supertype Hiearachy

`MPPHullMethod <: AbstractBivariateHullMethod <: Any`
"""
struct MPPHullMethod <: AbstractBivariateHullMethod end

"""
    bivariate_hull_methods()

Prints a list of available bivariate hull methods. Available bivariate hull methods include [`ConvexHullMethod`](@ref), [`ConcaveHullMethod`](@ref) and [`MPPHullMethod`](@ref).
"""
function bivariate_hull_methods()
    methods = [ConvexHullMethod, ConcaveHullMethod, MPPHullMethod]
    println(string("Available bivariate methods are: ", [i != length(methods) ? string(method, ", ") : string(method) for (i, method) in enumerate(methods)]...))
    return nothing
end