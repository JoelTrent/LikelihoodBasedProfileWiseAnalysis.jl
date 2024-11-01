module LikelihoodBasedProfileWiseAnalysis

using DataStructures, DataFrames, Accessors, StaticArrays, TrackingHeaps
using SciMLBase, Optimization

using Reexport
@reexport using OptimizationNLopt, ADTypes

using Roots
using ForwardDiff, LinearAlgebra, LogExpFunctions
using EllipseSampling
using LatinHypercubeSampling
using Random, StatsBase, Combinatorics, Distributions
using UnivariateUnimodalHighestDensityRegion
using Distances, TravelingSalesmanHeuristics
using Meshes, ConcaveHull, PolygonInbounds
using AngleBetweenVectors
using Distributed, SharedArrays
using FLoops

import HypothesisTests

using ProgressMeter
const global PROGRESS__METER__DT = 1.0

using TimerOutputs
const timer = TimerOutput()

# TYPES ###################################################################################
export EllipseMLEApprox, OptimizationSettings, CoreLikelihoodModel, BaseLikelihoodModel, LikelihoodModel,

    AbstractProfileType, AbstractEllipseProfileType, LogLikelihood, EllipseApprox, EllipseApproxAnalytical,
    
    AbstractConfidenceStruct, PointsAndLogLikelihood, 
    UnivariateConfidenceStruct, BivariateConfidenceStruct, SampledConfidenceStruct,
    
    AbstractBivariateMethod, AbstractBivariateVectorMethod, bivariate_methods,
    CombinedBivariateMethod, IterativeBoundaryMethod, RadialMLEMethod, RadialRandomMethod, 
    SimultaneousMethod, Fix1AxisMethod, AnalyticalEllipseMethod,

    AbstractBivariateHullMethod, bivariate_hull_methods, ConvexHullMethod, ConcaveHullMethod, MPPHullMethod,
    
    AbstractSampleType, UniformGridSamples, UniformRandomSamples, LatinHypercubeSamples,

    AbstractPredictionStruct, PredictionStruct, PredictionRealisationsStruct

# FUNCTIONS ###############################################################################
export initialise_LikelihoodModel, 
    default_OptimizationSettings, create_OptimizationSettings, set_OptimizationSettings!,
    getMLE_ellipse_approximation!, check_ellipse_approx_exists!,
    setmagnitudes!, setbounds!,

    optimise, optimise_unbounded,

    transformbounds, transformbounds_NLopt,

    univariate_confidenceintervals!, get_points_in_intervals!, 
    get_uni_confidence_interval, get_uni_confidence_intervals,
    get_uni_confidence_interval_points,
    check_univariate_parameter_coverage,

    bivariate_confidenceprofiles!, sample_bivariate_internal_points!,
    combine_bivariate_boundaries!, get_bivariate_confidence_set,
    check_bivariate_parameter_coverage, check_bivariate_boundary_coverage,
    
    dimensional_likelihood_samples!, full_likelihood_sample!,
    get_dimensional_confidence_set,

    add_prediction_function!, check_prediction_function_exists,
    add_error_function!,
    generate_predictions_univariate!, generate_predictions_bivariate!,
    generate_predictions_dim_samples!,
    get_univariate_prediction_set, get_bivariate_prediction_set,
    get_dimensional_prediction_set,

    normal_error_σ_known, normal_error_σ_estimated,
    lognormal_error_σ_known, lognormal_error_σ_estimated,
    logitnormal_error_σ_known, logitnormal_error_σ_estimated,
    poisson_error,

    check_univariate_prediction_coverage, check_bivariate_prediction_coverage,
    check_dimensional_prediction_coverage,
    check_univariate_prediction_realisations_coverage,
    check_bivariate_prediction_realisations_coverage,
    check_dimensional_prediction_realisations_coverage,
    
    trim_model_dfs!, remove_functions_from_core!, add_loglikelihood_function!

# TYPES ###################################################################################
include("types/bivariate_methods.jl")
include("types/bivariate_hull_methods.jl")
include("types/levelsets.jl")
include("types/predictions.jl")
include("types/profiletypes.jl")
include("types/sampletypes.jl")
include("types/optimizationsettings.jl")
include("types/likelihoodmodel.jl")

# OPTIMISER ###############################################################################
include("optimiser.jl")
include("optimizationsettings.jl")

# CORE FUNCTIONS ##########################################################################
include("model_initialiser.jl")
include("transform_bounds.jl")
include("common_profile_likelihood.jl")
include("ellipse_likelihood.jl")
include("predictions.jl")
include("predictions_for_realisations.jl")
include("nuisance_parameters_and_array_mapping.jl")
include("loading.jl")

# UNIVARIATE METHODS ######################################################################
include("univariate_methods/loglikelihood_functions.jl")
include("univariate_methods/univariate_profile_likelihood.jl")
include("univariate_methods/points_in_interval.jl")

# BIVARIATE METHODS #######################################################################
include("bivariate_methods/findpointon2Dbounds.jl")
include("bivariate_methods/loglikelihood_functions.jl")
include("bivariate_methods/fix1axis.jl")
include("bivariate_methods/vectorsearch.jl")
include("bivariate_methods/iterativeboundary.jl")
include("bivariate_methods/bivariate_profile_likelihood.jl")
include("bivariate_methods/MPP_TSP.jl")
include("bivariate_methods/bivariate_concave_hull.jl")
include("bivariate_methods/construct_polygon_hull.jl")
include("bivariate_methods/sample_internal_points.jl")
include("bivariate_methods/combine_bivariate_boundaries.jl")

# SAMPLING METHODS ########################################################################
include("dimensional_methods/full_likelihood_sampling.jl")
include("dimensional_methods/dimensional_likelihood_sampling.jl")

# COVERAGE CHECKS #########################################################################
include("coverage_checks/parameters/univariate.jl")
include("coverage_checks/parameters/bivariate.jl")
include("coverage_checks/parameters/bivariate_boundary.jl")
include("coverage_checks/predictions/coverage_evaluation.jl")
include("coverage_checks/predictions/univariate.jl")
include("coverage_checks/predictions/bivariate.jl")
include("coverage_checks/predictions/dimensional.jl")

# PLOT FUNCTIONS ##########################################################################
function plot_univariate_profiles end
function plot_univariate_profiles_comparison end
function plot_bivariate_profiles end
function plot_bivariate_profiles_comparison end
function plot_bivariate_profiles_iterativeboundary_gif end
function plot_predictions_individual end
function plot_predictions_union end
function plot_realisations_individual end
function plot_realisations_union end

export plot_univariate_profiles, plot_univariate_profiles_comparison,
    plot_bivariate_profiles, plot_bivariate_profiles_comparison,
    plot_bivariate_profiles_iterativeboundary_gif,
    plot_predictions_individual, plot_predictions_union,
    plot_realisations_individual, plot_realisations_union

import SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin
    
    a, b = 2.0, 1.0
    # α = 0.2 * π
    # Cx, Cy = 2.0, 2.0

    # Hw11 = (cos(α)^2 / a^2 + sin(α)^2 / b^2)
    # Hw22 = (sin(α)^2 / a^2 + cos(α)^2 / b^2)
    # Hw12 = cos(α) * sin(α) * (1 / a^2 - 1 / b^2)
    # Hw_norm = [Hw11 Hw12; Hw12 Hw22]

    # confidence_level = 0.95
    # Hw = Hw_norm ./ (0.5 ./ (quantile(Chisq(2), confidence_level) * 0.5))

    # data = (θmle=[Cx, Cy], Hmle=Hw)

    # θnames = [:x, :y]
    # θG = [2.0, 2.0]
    # lb = [0.0, 0.0]
    # ub = [4.0, 4.0]
    # par_magnitudes = [1, 1]
    # function loglike(θ::AbstractVector, data); ellipse_loglike(θ, data); end
    # function loglike(θ::Tuple, data); ellipse_loglike([θ...], data); end

    # m = initialiseLikelihoodModel(loglike, data, θnames, θG, lb, ub, par_magnitudes)
    # getMLE_ellipse_approximation!(m)

    # N=8
    # for profile_type in [LogLikelihood(), EllipseApprox(), EllipseApproxAnalytical()]
    #     univariate_confidenceintervals!(m, profile_type=profile_type)
    #     for method in [IterativeBoundaryMethod(4, 2, 2), RadialRandomMethod(3), SimultaneousMethod(), Fix1AxisMethod(), ContinuationMethod(2, 0.1, 0.0)]
    #         bivariate_confidenceprofiles!(m, N, method=method, profile_type=profile_type)
    #     end
    # end

    # univariate_confidenceintervals!(m, use_existing_profiles=true, confidence_level=0.9, num_points_in_interval=10)

    # dimensional_likelihood_sample!(m, 1, 100, sample_type=UniformGridSamples())
    # dimensional_likelihood_sample!(m, 1, 100, sample_type=UniformRandomSamples())
    # dimensional_likelihood_sample!(m, 1, 100, sample_type=LatinHypercubeSamples())

    # dimensional_likelihood_sample!(m, 2, 10, sample_type=UniformGridSamples())
    # dimensional_likelihood_sample!(m, 2, 100, sample_type=UniformRandomSamples())
    # dimensional_likelihood_sample!(m, 2, 100, sample_type=LatinHypercubeSamples())


end

end