module PlaceholderLikelihood

using DataStructures, DataFrames, Accessors, StaticArrays, TrackingHeaps
using NLopt, Roots
using ForwardDiff, LinearAlgebra
using EllipseSampling
using LatinHypercubeSampling
using Random, StatsBase, Combinatorics, Distributions
using Distances, TravelingSalesmanHeuristics
using Clustering, Meshes, ConcaveHull
using AngleBetweenVectors
using Distributed, FLoops
using Requires

using ProgressMeter
const global PROGRESS__METER__DT = 1.0

# TYPES ###################################################################################
export EllipseMLEApprox, CoreLikelihoodModel, LikelihoodModel, 

    AbstractProfileType, AbstractEllipseProfileType, LogLikelihood, EllipseApprox, EllipseApproxAnalytical,
    
    AbstractConfidenceStruct, PointsAndLogLikelihood, 
    UnivariateConfidenceStruct, BivariateConfidenceStruct, SampledConfidenceStruct,
    
    AbstractBivariateMethod, AbstractBivariateVectorMethod, bivariate_methods,
    IterativeBoundaryMethod, RadialMLEMethod, RadialRandomMethod, SimultaneousMethod, Fix1AxisMethod,
    AnalyticalEllipseMethod, ContinuationMethod, 
    
    AbstractSampleType, UniformGridSamples, UniformRandomSamples, LatinHypercubeSamples,

    AbstractPredictionStruct, PredictionStruct


# FUNCTIONS ###############################################################################
export initialiseLikelihoodModel,
    getMLE_ellipse_approximation!, check_ellipse_approx_exists!,
    setθmagnitudes!, setbounds!,

    optimise, optimise_unbounded,

    transformbounds, transformbounds_NLopt,

    univariate_confidenceintervals!, get_points_in_interval!,
    bivariate_confidenceprofiles!,
    dimensional_likelihood_sample!, bivariate_concave_hull,
    full_likelihood_sample!,

    add_prediction_function!, check_prediction_function_exists,
    generate_predictions_univariate!, generate_predictions_bivariate!, generate_predictions_dim_samples!

# OPTIMISER ###############################################################################
include("NLopt_optimiser.jl")

# TYPES ###################################################################################
include("types/bivariate_methods.jl")
include("types/levelsets.jl")
include("types/predictions.jl")
include("types/profiletypes.jl")
include("types/sampletypes.jl")
include("types/likelihoodmodel.jl")

# CORE FUNCTIONS ##########################################################################
include("model_initialiser.jl")
# include("combination_relationships.jl")
include("transform_bounds.jl")
include("common_profile_likelihood.jl")
include("ellipse_likelihood.jl")
include("predictions.jl")

# UNIVARIATE METHODS ######################################################################
include("univariate_methods/init_and_array_mapping.jl")
include("univariate_methods/loglikelihood_functions.jl")
include("univariate_methods/univariate_profile_likelihood.jl")
include("univariate_methods/points_in_interval.jl")

# BIVARIATE METHODS #######################################################################
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

# SAMPLING METHODS ########################################################################
include("dimensional_methods/full_likelihood_sampling.jl")
include("dimensional_methods/dimensional_likelihood_sampling.jl")
include("dimensional_methods/bivariate_concave_hull.jl")

# PLOT FUNCTIONS ##########################################################################
function plot_univariate_profiles end
function plot_univariate_profiles_comparison end
function plot_bivariate_profiles end
function plot_bivariate_profiles_comparison end
function plot_bivariate_profiles_iterativeboundary_gif end
function plot_predictions_individual end
function plot_predictions_union end

export plot_univariate_profiles, plot_univariate_profiles_comparison,
    plot_bivariate_profiles, plot_bivariate_profiles_comparison,
    plot_bivariate_profiles_iterativeboundary_gif,
    plot_predictions_individual, plot_predictions_union

function __init__()
    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin 
        using Plots
        include("plotting_functions.jl") 
    end
end

import SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin
    1==2
end

end