```@meta
CollapsedDocStrings = true
```
# Bivariate Functions

```@index
Pages = ["bivariate.md"]
```

## Likelihood Optimisation

```@docs
LikelihoodBasedProfileWiseAnalysis.bivariateψ!
LikelihoodBasedProfileWiseAnalysis.bivariateψ_vectorsearch!
LikelihoodBasedProfileWiseAnalysis.bivariateψ_continuation!
LikelihoodBasedProfileWiseAnalysis.bivariateψ_ellipse_analytical
LikelihoodBasedProfileWiseAnalysis.bivariateψ_ellipse_analytical_vectorsearch
LikelihoodBasedProfileWiseAnalysis.bivariateψ_ellipse_analytical_continuation
LikelihoodBasedProfileWiseAnalysis.bivariateψ_ellipse_unbounded
```

## Finding Points on 2D bounds

```@docs
LikelihoodBasedProfileWiseAnalysis.findpointonbounds
```

## Main Confidence Boundary Logic 

Note: [`AnalyticalEllipseMethod`](@ref) is calculated using [`generate_N_clustered_points`](https://joeltrent.github.io/EllipseSampling.jl/stable/user_interface/#EllipseSampling.generate_N_clustered_points) from [EllipseSampling.jl](https://github.com/JoelTrent/EllipseSampling.jl) within [`LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile`](@ref).

```@docs
LikelihoodBasedProfileWiseAnalysis.add_biv_profiles_rows!
LikelihoodBasedProfileWiseAnalysis.set_biv_profiles_row!
LikelihoodBasedProfileWiseAnalysis.get_bivariate_opt_func
LikelihoodBasedProfileWiseAnalysis.get_ωs_bivariate_ellipse_analytical!
LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile
Base.merge(::BivariateConfidenceStruct, ::BivariateConfidenceStruct)
```

## Minimum Perimeter Polygon

```@docs
LikelihoodBasedProfileWiseAnalysis.minimum_perimeter_polygon!
```

## Iterative Boundary Method

For [`IterativeBoundaryMethod`](@ref).

```@docs
LikelihoodBasedProfileWiseAnalysis.findNpointpairs_radialMLE!
LikelihoodBasedProfileWiseAnalysis.edge_length
LikelihoodBasedProfileWiseAnalysis.internal_angle_from_pi!
LikelihoodBasedProfileWiseAnalysis.internal_angle_from_pi
LikelihoodBasedProfileWiseAnalysis.iterativeboundary_init
LikelihoodBasedProfileWiseAnalysis.newboundarypoint!
LikelihoodBasedProfileWiseAnalysis.heapupdates_success!
LikelihoodBasedProfileWiseAnalysis.heapupdates_failure!
LikelihoodBasedProfileWiseAnalysis.polygon_break_and_rejoin!
LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile_iterativeboundary
```

## Vectorsearch Methods

For the [`RadialRandomMethod`](@ref), [`RadialMLEMethod`](@ref) and [`SimultaneousMethod`](@ref).

```@docs
LikelihoodBasedProfileWiseAnalysis.generatepoint
LikelihoodBasedProfileWiseAnalysis.findNpointpairs_simultaneous!
LikelihoodBasedProfileWiseAnalysis.find_m_spaced_radialdirections
LikelihoodBasedProfileWiseAnalysis.findNpointpairs_radialrandom!
LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile_vectorsearch
```

## Fix1Axis Method

For [`Fix1AxisMethod`](@ref).

```@docs
LikelihoodBasedProfileWiseAnalysis.findNpointpairs_fix1axis!
LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile_fix1axis
```

## Continuation Method

For [`ContinuationMethod`](@ref).

```@docs
LikelihoodBasedProfileWiseAnalysis.update_targetll!
LikelihoodBasedProfileWiseAnalysis.normal_vector_i_2d!
LikelihoodBasedProfileWiseAnalysis.continuation_line_search!
LikelihoodBasedProfileWiseAnalysis.continuation_inwards_radial_search!
LikelihoodBasedProfileWiseAnalysis.initial_continuation_solution!
LikelihoodBasedProfileWiseAnalysis.bivariate_confidenceprofile_continuation
LikelihoodBasedProfileWiseAnalysis.star_obj
LikelihoodBasedProfileWiseAnalysis.boundary_smoother!
LikelihoodBasedProfileWiseAnalysis.refine_search_directions!
```

## Sampling Internal Points From Boundaries

```@docs
LikelihoodBasedProfileWiseAnalysis.construct_polygon_hull
LikelihoodBasedProfileWiseAnalysis.bivariate_concave_hull
LikelihoodBasedProfileWiseAnalysis.update_biv_dict_internal!
LikelihoodBasedProfileWiseAnalysis.sample_internal_points_LHC
LikelihoodBasedProfileWiseAnalysis.sample_internal_points_uniform_random
LikelihoodBasedProfileWiseAnalysis.sample_internal_points_single_row
```

## Merging Boundaries From Multiple Methods

```@docs
LikelihoodBasedProfileWiseAnalysis.predictions_can_be_merged
LikelihoodBasedProfileWiseAnalysis.rebuild_bivariate_datastructures!
```