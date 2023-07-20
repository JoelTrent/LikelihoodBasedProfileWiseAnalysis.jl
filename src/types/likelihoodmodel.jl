"""
    EllipseMLEApprox(Hmle::Matrix{<:Float64}, Γmle::Matrix{<:Float64})

Struct containing two n*n arrays representing the ellipse approximation of the log-likelihood function around the MLE point. See [`getMLE_ellipse_approximation!`](@ref)

# Fields
- `Hmle`: a n*n array, where n is the number of model parameters, containing the negative Hessian of the log-likelihood function evaluated at the MLE point. 
- `Γmle`: a n*n array, where n is the number of model parameters, containing the inverse of `Hmle`.

# Supertype Hiearachy

`EllipseMLEApprox <: Any`
"""
struct EllipseMLEApprox
    Hmle::Matrix{<:Float64}
    Γmle::Matrix{<:Float64}
end

"""
    OptimizationSettings(adtype::SciMLBase.AbstractADType,
        solve_alg,
        solve_kwargs::NamedTuple)

Contains the optimization settings used to solve for nuisance parameters of the log-likelihood function using [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). Note: Optimization.jl uses a minimisation objective, while the log-likelihood functions are setup with a maximisation objective in mind. This is handled inside the package. Defaults can be seen in [`defaultOptimizationSettings`](@ref).

# Fields
- `adtype`: a method for automatically generating derivative functions of the log-likelihood function being optimised. For available settings see [Automatic Differentiation Construction Choice Recommendations](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#Automatic-Differentiation-Construction-Choice-Recommendations). Note: the corresponding package to `adtype` needs to be loaded with `using`. e.g. setting `adtype = AutoFiniteDiff()` requires `using FiniteDiff`. Derivative-based algorithms in `solve_alg` will require an `adtype` to be specified. 
    - `AutoFiniteDiff()`, `AutoForwardDiff()`, `AutoReverseDiff()`, `AutoZygote()` and `AutoTracker()` have been tested to work for finding the maximum likelihood estimate with regular model parameters. 
    - For profiling, `AutoFiniteDiff()`, `AutoForwardDiff()`, `AutoReverseDiff()` and `AutoTracker()` have been tested to work with regular model parameters. 
    - If the variance of the error distribution is included as a parameter to be estimated, `AutoFiniteDiff()`, `AutoForwardDiff()` and `AutoTracker()` have been tested to work for finding the MLE and profiling. 
    - `AutoFiniteDiff()` will always work, regardless of model specification, but may be less optimal than other methods.
- `solve_alg`: an algorithm to use to solve for the nuisance parameters of the log-likelihood function defined within [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). The package is loaded with the Optimization integration of NLopt, so any of the NLopt algorithms are available without having to load another package (see [OptimizationNLopt](https://docs.sciml.ai/Optimization/stable/optimization_packages/nlopt/)). Good starting methods may be `NLopt.LN_BOBYQA()`, `NLopt.LN_NELDERMEAD()` and `NLopt.LD_LBFGS`. Other packages can be used as well - see [Overview of the Optimizers](https://docs.sciml.ai/Optimization/stable/#Overview-of-the-Optimizers).
- `solve_kwargs`: a `NamedTuple` of keyword arguments used to set solver options like `maxiters` and `maxtime`. For a list of common solver arguments see: [Common Solver Options](https://docs.sciml.ai/Optimization/stable/API/solve/#Common-Solver-Options-(Solve-Keyword-Arguments)). Other specific package arguments may also be available. For NLopt see [Methods](https://docs.sciml.ai/Optimization/stable/optimization_packages/nlopt/#Methods).

# Supertype Hiearachy

`OptimizationSettings <: Any`
"""
struct OptimizationSettings 
    adtype::SciMLBase.AbstractADType
    solve_alg
    solve_kwargs::NamedTuple
end

"""
    CoreLikelihoodModel(loglikefunction::Function, 
        predictfunction::Union{Function, Missing}, 
        data::Union{Tuple, NamedTuple}, 
        θnames::AbstractVector
        θname_to_index::Dict{Symbol, Int}, 
        θlb::AbstractVector{<:Real}, 
        θub::AbstractVector{<:Real}, 
        θmagnitudes::AbstractVector{<:Real}, 
        θmle::Vector{<:Float64}, 
        ymle::Array{<:Real}, 
        maximisedmle::Float64, 
        num_pars::Int)

Struct containing the core information required to define a [`LikelihoodModel`](@ref). For additional information on parameters (where repeated), see [`initialiseLikelihoodModel`](@ref).

# Fields
- `loglikefunction`: a log-likelihood function which takes two arguments, `θ` and `data`, in that order. Set up to be used in a maximisation objective.
- `predictfunction`: a prediction function to generate model predictions from that is paired with the `loglikefunction`. 
- `optimizationsettings`: a [`OptimizationSettings`](@ref) struct of settings to use for optimisation unless others are specified.
- `data`: a Tuple or a NamedTuple containing any additional information required by the log-likelihood function, such as the time points to be evaluated at.
- `θnames`: a vector of symbols containing the names of each parameter, e.g. `[:λ, :K, :C0]`.
- `θname_to_index`: a dictionary with keys of type Symbol and values of type Int, with the key being an element of `θnames` and the value being the corresponding index of the key in `θnames`.
- `θlb`: a vector of lower bounds on parameters. 
- `θub`: a vector of upper bounds on parameters. 
- `θmagnitudes`: a vector of the relative magnitude of each parameter. 
- `θmle`: a vector containing the maximum likelihood estimate for each parameter.
- `ymle`: an array containing the output of the prediction function at `θmle` and `data`.
- `maximisedmle`: the value of the log-likelihood function evaluated at `θmle`.
- `num_pars`: the number of model parameters, `length(θnames)`.

# Supertype Hiearachy

`CoreLikelihoodModel <: Any`
"""
struct CoreLikelihoodModel
    loglikefunction::Function
    predictfunction::Union{Function, Missing}
    optimizationsettings::OptimizationSettings
    data::Union{Tuple, NamedTuple}
    θnames::AbstractVector
    θname_to_index::Dict{Symbol, Int}
    θlb::AbstractVector{<:Real}
    θub::AbstractVector{<:Real}
    θmagnitudes::AbstractVector{<:Real}
    θmle::Vector{<:Float64}
    ymle::Array{<:Real}
    maximisedmle::Float64
    num_pars::Int
end

"""
    BaseLikelihoodModel(data::Union{Tuple, NamedTuple}, 
        θnames::AbstractVector,
        θname_to_index::Dict{Symbol, Int}, 
        θlb::AbstractVector{<:Real}, 
        θub::AbstractVector{<:Real}, 
        θmagnitudes::AbstractVector{<:Real}, 
        θmle::Vector{<:Float64}, 
        ymle::Array{<:Real}, 
        maximisedmle::Float64, 
        num_pars::Int)

Version of [`CoreLikelihoodModel`](@ref) without functions, allowing loading of the model struct without those functions defined.

# Supertype Hiearachy

`BaseLikelihoodModel <: Any`
"""
struct BaseLikelihoodModel
    data::Union{Tuple, NamedTuple}
    θnames::AbstractVector
    θname_to_index::Dict{Symbol, Int}
    θlb::AbstractVector{<:Real}
    θub::AbstractVector{<:Real}
    θmagnitudes::AbstractVector{<:Real}
    θmle::Vector{<:Float64}
    ymle::Array{<:Real}
    maximisedmle::Float64
    num_pars::Int
end

"""
    LikelihoodModel(core::Union{CoreLikelihoodModel, BaseLikelihoodModel}, 
        ellipse_MLE_approx::Union{Missing, EllipseMLEApprox},
        num_uni_profiles::Int, 
        num_biv_profiles::Int, 
        num_dim_samples::Int, 
        uni_profiles_df::DataFrame, 
        biv_profiles_df::DataFrame, 
        dim_samples_df::DataFrame, 
        uni_profile_row_exists::Dict{Tuple{Int, AbstractProfileType}, DefaultDict{Float64, Int}},
        biv_profile_row_exists::Dict{Tuple{Tuple{Int, Int}, AbstractProfileType, AbstractBivariateMethod}, DefaultDict{Float64, Int}},
        dim_samples_row_exists::Dict{Union{AbstractSampleType, Tuple{Vector{Int}, AbstractSampleType}}, DefaultDict{Float64, Int}},
        uni_profiles_dict::Dict{Int, UnivariateConfidenceStruct}, 
        biv_profiles_dict::Dict{Int, BivariateConfidenceStruct}, 
        dim_samples_dict::Dict{Int, SampledConfidenceStruct},
        uni_predictions_dict::Dict{Int, AbstractPredictionStruct}, 
        biv_predictions_dict::Dict{Int, AbstractPredictionStruct}, 
        dim_predictions_dict::Dict{Int, AbstractPredictionStruct},
        show_progress::Bool)

Mutable struct containing all the information required to compute profiles, samples and predictions. Created by [`initialiseLikelihoodModel`](@ref).

# Fields
- `core`: a [`CoreLikelihoodModel`](@ref) or [`BaseLikelihoodModel`](@ref) struct.
- `ellipse_MLE_approx`: a [`EllipseMLEApprox`](@ref) struct OR a missing value if the ellipse approximation of the log-likelihood at the MLE point has not been evaluated yet. 
- `num_uni_profiles`: the number of different univariate profiles that have been evaluated (distinct combinations of different confidence levels, [`AbstractProfileType`](@ref) structs and single interest parameters). Specifies the number of valid rows in `uni_profiles_df`.  
- `num_biv_profiles`: the number of different bivariate profiles that have been evaluated (distinct combinations of different confidence levels, [`AbstractProfileType`](@ref) structs, [`AbstractBivariateMethod`](@ref) structs and two interest parameters). Specifies the number of valid rows in `biv_profiles_df`.  
- `num_dim_samples`: the number of different dimensional profiles that have been evaluated (distinct combinations of different confidence levels, [`AbstractProfileType`](@ref) structs, [`AbstractSampleType`](@ref) structs and sets of interest parameters). Specifies the number of valid rows in `dim_samples_df`.  
- `uni_profiles_df`: a DataFrame with each row containing information on each univariate profile evaluated, where the row index is the key for that profile in `uni_profiles_dict` and `uni_predictions_dict`.
- `biv_profiles_df`: a DataFrame with each row containing information on each bivariate profile evaluated, where the row index is the key for that profile in `biv_profiles_dict` and `biv_predictions_dict`.
- `dim_samples_df`: a DataFrame with each row containing information on each dimensional sample evaluated, where the row index is the key for that sample in `dim_samples_dict` and `dim_predictions_dict`.
- `uni_profile_row_exists`: a dictionary containing information on whether a row in `uni_profiles_df` exists for a given combination of interest parameter, [`AbstractProfileType`](@ref) and confidence level. If it does exist, it's value will be the row index in `uni_profiles_df` otherwise it will be `0`.
- `biv_profile_row_exists`: a dictionary containing information on whether a row in `biv_profiles_df` exists for a given combination of two interest parameters, [`AbstractProfileType`](@ref), [`AbstractBivariateMethod`](@ref) and confidence level. If it does exist, it's value will be the row index in `biv_profiles_df` otherwise it will be `0`.
- `dim_samples_row_exists`: a dictionary containing information on whether a row in `dim_samples_df` exists for a given combination of interest parameter, [`AbstractProfileType`](@ref), [`AbstractSampleType`](@ref) and confidence level. If it does exist, it's value will be the row index in `dim_samples_df` otherwise it will be `0`.
- `uni_profiles_dict`: a dictionary with keys of type Integer and values of type [`UnivariateConfidenceStruct`](@ref) containing the profile for each valid row in `uni_profiles_df`. The row index of `uni_profiles_df` is the key for the corresponding profile.
- `biv_profiles_dict`: a dictionary with keys of type Integer and values of type [`BivariateConfidenceStruct`](@ref) containing the profile for each valid row in `biv_profiles_df`. The row index of `biv_profiles_df` is the key for the corresponding profile.
- `dim_samples_dict`: a dictionary with keys of type Integer and values of type [`SampledConfidenceStruct`](@ref) containing the profile for each valid row in `dim_samples_df`. The row index of `dim_samples_df` is the key for the corresponding profile.
- `uni_predictions_dict`: a dictionary with keys of type Integer and values of type [`PredictionStruct`](@ref) containing the predictions from the profiles in `uni_profiles_dict` for each valid row in `uni_profiles_df`. The row index of `uni_profiles_df` is the key for the corresponding prediction, if that prediction has been calculated using [`generate_predictions_univariate!`](@ref). 
- `biv_predictions_dict`: a dictionary with keys of type Integer and values of type [`PredictionStruct`](@ref) containing the predictions from the profiles in `biv_profiles_dict` for each valid row in `biv_profiles_df`. The row index of `biv_profiles_df` is the key for the corresponding prediction, if that prediction has been calculated using [`generate_predictions_bivariate!`](@ref). 
- `dim_predictions_dict`: a dictionary with keys of type Integer and values of type [`PredictionStruct`](@ref) containing the predictions from the profiles in `dim_samples_dict` for each valid row in `dim_samples_df`. The row index of `dim_samples_df` is the key for the corresponding prediction, if that prediction has been calculated using [`generate_predictions_dim_samples!`](@ref). 
- `find_zero_atol`: a `Real` number greater than zero for the absolute tolerance of the log-likelihood function value from the target value to be used when searching for confidence intervals/boundaries.
- `show_progress`: a boolean specifying whether to show the progress of profile methods with respect to sets of interest parameter(s).

# Supertype Hiearachy

`LikelihoodModel <: Any`
"""
mutable struct LikelihoodModel
    core::Union{CoreLikelihoodModel,BaseLikelihoodModel}
    ellipse_MLE_approx::Union{Missing, EllipseMLEApprox}

    num_uni_profiles::Int
    num_biv_profiles::Int
    num_dim_samples::Int

    uni_profiles_df::DataFrame
    biv_profiles_df::DataFrame
    dim_samples_df::DataFrame

    uni_profile_row_exists::Dict{Tuple{Int, AbstractProfileType}, DefaultDict{Float64, Int}}
    biv_profile_row_exists::Dict{Tuple{Tuple{Int, Int}, AbstractProfileType, AbstractBivariateMethod}, DefaultDict{Float64, Int}}
    dim_samples_row_exists::Dict{Union{AbstractSampleType, Tuple{Vector{Int}, AbstractSampleType}}, DefaultDict{Float64, Int}}

    uni_profiles_dict::Dict{Int, UnivariateConfidenceStruct}
    biv_profiles_dict::Dict{Int, BivariateConfidenceStruct}
    dim_samples_dict::Dict{Int, SampledConfidenceStruct}

    uni_predictions_dict::Dict{Int, AbstractPredictionStruct}
    biv_predictions_dict::Dict{Int, AbstractPredictionStruct}
    dim_predictions_dict::Dict{Int, AbstractPredictionStruct}

    # misc arguments
    find_zero_atol::Real
    show_progress::Bool
end