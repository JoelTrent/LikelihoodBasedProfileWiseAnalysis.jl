module PlaceholderLikelihood

using NLopt, Roots
using ForwardDiff, LinearAlgebra
using Random, StatsBase, Combinatorics
using DataStructures, DataFrames, Accessors, StaticArrays
using EllipseSampling
using LatinHypercubeSampling
using Distributed, FLoops
using ConcaveHull
using Distances, TravelingSalesmanHeuristics
using Clustering, Meshes
using AngleBetweenVectors
using TrackingHeaps

using ProgressMeter
const global PROGRESS__METER__DT = 1.0


export AbstractEllipseMLEApprox, AbstractCoreLikelihoodModel, AbstractLikelihoodModel,
    EllipseMLEApprox, CoreLikelihoodModel, LikelihoodModel, 

    AbstractProfileType, AbstractEllipseProfileType, LogLikelihood, EllipseApprox, EllipseApproxAnalytical
    AbstractSampleType, UniformGridSamples, UniformRandomSamples, LatinHypercubeSamples

    AbstractConfidenceStruct, PointsAndLogLikelihood, 
    SampledConfidenceStruct, UnivariateConfidenceStruct, BivariateConfidenceStruct,

    AbstractBivariateMethod, AbstractBivariateVectorMethod, bivariate_methods,
    IterativeBoundaryMethod, RadialMLEMethod, RadialRandomMethod, SimultaneousMethod, Fix1AxisMethod,
    AnalyticalEllipseMethod, ContinuationMethod, 

    AbstractPredictionStruct, PredictionStruct,

    initialiseLikelihoodModel,
    getMLE_ellipse_approximation!, check_ellipse_approx_exists!
    setθmagnitudes!, setbounds!,

    transformbounds, transformbounds_NLopt,

    univariate_confidenceintervals!, get_points_in_interval!,
    bivariate_confidenceprofiles!, minimum_perimeter_polygon!,
    dimensional_likelihood_sample!, bivariate_concave_hull,
    full_likelihood_sample!,

    add_prediction_function!, check_prediction_function_exists,
    generate_predictions_bivariate!, generate_predictions_univariate!, generate_predictions_dim_samples!

include("NLopt_optimiser.jl")

# TYPES ##############################################################
include("types/bivariate_methods.jl")
include("types/levelsets.jl")
include("types/predictions.jl")
include("types/profiletype.jl")
include("types/likelihoodmodel.jl")


include("model_initialiser.jl")
# include("combination_relationships.jl")
include("transform_bounds.jl")
include("common_profile_likelihood.jl")
include("ellipse_likelihood.jl")
include("predictions.jl")
include("plotting_functions.jl")


# UNIVARIATE METHODS #################################################
include("univariate_methods/init_and_array_mapping.jl")
include("univariate_methods/loglikelihood_functions.jl")
include("univariate_methods/univariate_profile_likelihood.jl")
include("univariate_methods/points_in_interval.jl")


# BIVARIATE METHODS ##################################################
include("bivariate_methods/init_and_array_mapping.jl")
include("bivariate_methods/findpointon2Dbounds.jl")
include("bivariate_methods/loglikelihood_functions.jl")
include("bivariate_methods/fix1axis.jl")
include("bivariate_methods/vectorsearch.jl")
include("bivariate_methods/continuation_polygon_manipulation.jl")
include("bivariate_methods/continuation.jl")
include("bivariate_methods/iterativeboundary.jl")
include("bivariate_methods/bivariate_profile_likelihood.jl")
include("bivariate_methods/MPP_TSP.jl")


# SAMPLING METHODS ###################################################
include("dimensional_methods/full_likelihood_sampling.jl")
include("dimensional_methods/dimensional_likelihood_sampling.jl")
include("dimensional_methods/bivariate_concave_hull.jl")


import SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin
    1==2
end

end