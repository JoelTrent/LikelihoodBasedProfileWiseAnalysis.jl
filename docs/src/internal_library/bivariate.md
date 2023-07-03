# Bivariate Functions

```@index
Pages = ["bivariate.md"]
```

## Likelihood Optimisation

```@docs
PlaceholderLikelihood.bivariateψ!
PlaceholderLikelihood.bivariateψ_vectorsearch!
PlaceholderLikelihood.bivariateψ_continuation!
PlaceholderLikelihood.bivariateψ_gradient!
PlaceholderLikelihood.bivariateψ_ellipse_analytical
PlaceholderLikelihood.bivariateψ_ellipse_analytical_vectorsearch
PlaceholderLikelihood.bivariateψ_ellipse_analytical_continuation
PlaceholderLikelihood.bivariateψ_ellipse_analytical_gradient
PlaceholderLikelihood.bivariateψ_ellipse_unbounded
```

## Finding Points on 2D bounds

```@docs
PlaceholderLikelihood.findpointonbounds
```

## Main Confidence Boundary Logic 

Note: [`AnalyticalEllipseMethod`](@ref) is calculated using [`generate_N_clustered_points`](https://joeltrent.github.io/EllipseSampling.jl/stable/user_interface/#EllipseSampling.generate_N_clustered_points) from [EllipseSampling.jl](https://github.com/JoelTrent/EllipseSampling.jl) within [`PlaceholderLikelihood.bivariate_confidenceprofile`](@ref).

```@docs
PlaceholderLikelihood.add_biv_profiles_rows!
PlaceholderLikelihood.set_biv_profiles_row!
PlaceholderLikelihood.get_bivariate_opt_func
PlaceholderLikelihood.get_ωs_bivariate_ellipse_analytical!
PlaceholderLikelihood.bivariate_confidenceprofile
Base.merge(::BivariateConfidenceStruct, ::BivariateConfidenceStruct)
```

## Minimum Perimeter Polygon

```@docs
PlaceholderLikelihood.minimum_perimeter_polygon!
```

## Iterative Boundary Method

For [`IterativeBoundaryMethod`](@ref).

```@docs
PlaceholderLikelihood.findNpointpairs_radialMLE!
PlaceholderLikelihood.edge_length
PlaceholderLikelihood.internal_angle_from_pi!
PlaceholderLikelihood.internal_angle_from_pi
PlaceholderLikelihood.iterativeboundary_init
PlaceholderLikelihood.newboundarypoint!
PlaceholderLikelihood.heapupdates_success!
PlaceholderLikelihood.heapupdates_failure!
PlaceholderLikelihood.polygon_break_and_rejoin!
PlaceholderLikelihood.bivariate_confidenceprofile_iterativeboundary
```

## Vectorsearch Methods

For the [`RadialRandomMethod`](@ref), [`RadialMLEMethod`](@ref) and [`SimultaneousMethod`](@ref).

```@docs
PlaceholderLikelihood.generatepoint
PlaceholderLikelihood.findNpointpairs_simultaneous!
PlaceholderLikelihood.find_m_spaced_radialdirections
PlaceholderLikelihood.findNpointpairs_radialrandom!
PlaceholderLikelihood.bivariate_confidenceprofile_vectorsearch
```

## Fix1Axis Method

For [`Fix1AxisMethod`](@ref).

```@docs
PlaceholderLikelihood.findNpointpairs_fix1axis!
PlaceholderLikelihood.bivariate_confidenceprofile_fix1axis!
```

## Continuation Method

For [`ContinuationMethod`](@ref).

```@docs
PlaceholderLikelihood.update_targetll!
PlaceholderLikelihood.normal_vector_i_2d!
PlaceholderLikelihood.continuation_line_search!
PlaceholderLikelihood.continuation_inwards_radial_search!
PlaceholderLikelihood.initial_continuation_solution!
PlaceholderLikelihood.bivariate_confidenceprofile_continuation
PlaceholderLikelihood.star_obj
PlaceholderLikelihood.boundary_smoother!
PlaceholderLikelihood.refine_search_directions!
```

## Sampling Internal Points From Boundaries

```@docs
PlaceholderLikelihood.construct_polygon_hull
PlaceholderLikelihood.bivariate_concave_hull
```

## Merging Boundaries From Multiple Methods

```@docs
PlaceholderLikelihood.rebuild_bivariate_datastructures!
```