"""
    init_uni_profile_row_exists!(model::LikelihoodModel, 
        θs_to_profile::Vector{<:Int}, 
        dof::Int,
        profile_type::AbstractProfileType)

Initialises the dictionary entry in `model.uni_profile_row_exists` for the key `(θi, dof, profile_type)`, where `θi` is an element of `θs_to_profile` and `dof` is the degrees of freedom used to define the asymptotic threshold, with a `DefaultDict` with key of type `Float64` (a confidence level) and default value of 0.
"""
function init_uni_profile_row_exists!(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Int},
                                        dof::Int,
                                        profile_type::AbstractProfileType)
    for θi in θs_to_profile
        if !haskey(model.uni_profile_row_exists, (θi, dof, profile_type))
            model.uni_profile_row_exists[(θi, dof, profile_type)] = DefaultDict{Float64, Int}(0)
        end
    end
    return nothing
end

"""
    init_biv_profile_row_exists!(model::LikelihoodModel, 
        θcombinations::Vector{Vector{Int}}, 
        dof::Int,
        profile_type::AbstractProfileType, 
        method::AbstractBivariateMethod)

Initialises the dictionary entry in `model.biv_profile_row_exists` for the key `((ind1, ind2), dof, profile_type, method)`, where `(ind1, ind2)` is a combination in `θcombinations` and `dof` is the degrees of freedom used to define the asymptotic threshold, with a `DefaultDict` with key of type `Float64` (a confidence level) and default value of 0.
"""
function init_biv_profile_row_exists!(model::LikelihoodModel, 
                                        θcombinations::Union{Vector{Vector{Int}},Vector{Tuple{Int,Int}}},
                                        dof::Int,
                                        profile_type::AbstractProfileType,
                                        method::AbstractBivariateMethod)
    for (ind1, ind2) in θcombinations
        if !haskey(model.biv_profile_row_exists, ((ind1, ind2), dof, profile_type, method))
            model.biv_profile_row_exists[((ind1, ind2), dof, profile_type, method)] = DefaultDict{Float64, Int}(0)
        end
    end
    return nothing
end

"""
    init_dim_samples_row_exists!(model::LikelihoodModel, 
        sample_type::AbstractSampleType)

Initialises the dictionary entry in `model.dim_samples_row_exists` for the key `(sample_type)` with a `DefaultDict` with key of type `Float64` (a confidence level) and default value of 0. For a full likelihood sample (dimension equal to the number of model parameters).
"""
function init_dim_samples_row_exists!(model::LikelihoodModel, 
                                        sample_type::AbstractSampleType)
    if !haskey(model.dim_samples_row_exists, (sample_type))
        model.dim_samples_row_exists[sample_type] = DefaultDict{Float64, Int}(0)
    end
    return nothing
end

"""
    init_dim_samples_row_exists!(model::LikelihoodModel, 
        θindices::Vector{Vector{Int}},
        sample_type::AbstractSampleType)

Initialises the dictionary entry in `model.dim_samples_row_exists` for the key `(θvec, dof, sample_type)`, where `θvec` is a vector in `θindices` and `dof=length(θvec)` is the degrees of freedom used to define the asymptotic threshold, with a `DefaultDict` with key of type `Float64` (a confidence level) and default value of 0. For a non-full likelihood sample (dimension less than the number of model parameters).
"""
function init_dim_samples_row_exists!(model::LikelihoodModel, 
                                        θindices::Vector{Vector{Int}},
                                        sample_type::AbstractSampleType)
    for θvec in θindices
        if !haskey(model.dim_samples_row_exists, (θvec, length(θvec), sample_type))
            model.dim_samples_row_exists[(θvec, length(θvec), sample_type)] = DefaultDict{Float64,Int}(0)
        end
    end
    return nothing
end

"""
    init_uni_profiles_df(num_rows::Int; existing_largest_row::Int=0)

Initialises the DataFrame of `model.uni_profiles_df` with `num_rows` initial rows. In the event that the DataFrame already exists and more rows are being added, keyword argument, `existing_largest_row`, will be the number of rows in the existing dataframe, so that values of `row_ind` when concatenating the DataFrames will increase in steps of 1.
"""
function init_uni_profiles_df(num_rows::Int; existing_largest_row::Int=0)
   
    uni_profiles_df = DataFrame()
    uni_profiles_df.row_ind = collect(1:num_rows) .+ existing_largest_row
    uni_profiles_df.θindex = zeros(Int, num_rows)
    uni_profiles_df.not_evaluated_internal_points = trues(num_rows)
    uni_profiles_df.not_evaluated_predictions = trues(num_rows)
    uni_profiles_df.conf_level = zeros(num_rows)
    uni_profiles_df.dof = zeros(Int, num_rows)
    uni_profiles_df.profile_type = Vector{AbstractProfileType}(undef, num_rows)
    uni_profiles_df.num_points = zeros(Int, num_rows)
    uni_profiles_df.additional_width = zeros(num_rows)
    uni_profiles_df.region = zeros(num_rows)

    return uni_profiles_df
end

"""
    init_biv_profiles_df(num_rows::Int; existing_largest_row::Int=0)

Initialises the DataFrame of `model.biv_profiles_df` with `num_rows` initial rows. In the event that the DataFrame already exists and more rows are being added, keyword argument, `existing_largest_row`, will be the number of rows in the existing dataframe, so that values of `row_ind` when concatenating the DataFrames will increase in steps of 1.
"""
function init_biv_profiles_df(num_rows::Int; existing_largest_row::Int=0)
   
    biv_profiles_df = DataFrame()
    biv_profiles_df.row_ind = collect(1:num_rows) .+ existing_largest_row
    biv_profiles_df.θindices = fill((0,0), num_rows)
    biv_profiles_df.not_evaluated_internal_points = trues(num_rows)
    biv_profiles_df.not_evaluated_predictions = trues(num_rows)
    biv_profiles_df.boundary_not_ordered = trues(num_rows)
    biv_profiles_df.conf_level = zeros(num_rows)
    biv_profiles_df.dof = zeros(Int, num_rows)
    biv_profiles_df.profile_type = Vector{AbstractProfileType}(undef, num_rows)
    biv_profiles_df.method = Vector{AbstractBivariateMethod}(undef, num_rows)
    biv_profiles_df.num_points = zeros(Int, num_rows)
    biv_profiles_df.region = zeros(num_rows)

    return biv_profiles_df
end

"""
    init_dim_samples_df(num_rows::Int; existing_largest_row::Int=0)

Initialises the DataFrame of `model.dim_samples_df` with `num_rows` initial rows. In the event that the DataFrame already exists and more rows are being added, keyword argument, `existing_largest_row`, will be the number of rows in the existing dataframe, so that values of `row_ind` when concatenating the DataFrames will increase in steps of 1.
"""
function init_dim_samples_df(num_rows::Int; existing_largest_row::Int=0)
   
    dim_samples_df = DataFrame()
    dim_samples_df.row_ind = collect(1:num_rows) .+ existing_largest_row
    dim_samples_df.θindices = [Int[] for _ in 1:num_rows]
    dim_samples_df.dimension = zeros(Int, num_rows)
    dim_samples_df.not_evaluated_predictions = trues(num_rows)
    dim_samples_df.conf_level = zeros(num_rows)
    dim_samples_df.dof = zeros(Int, num_rows)
    dim_samples_df.sample_type = Vector{AbstractSampleType}(undef, num_rows)
    dim_samples_df.num_points = zeros(Int, num_rows)
    dim_samples_df.region = zeros(num_rows)

    return dim_samples_df
end

"""
    calculate_θmagnitudes(θlb::Vector{<:Float64}, θub::Vector{<:Float64})

Estimates the magnitude for each parameter using the difference between parameter bounds. If a bound is an `Inf`, the value is set to `NaN`. Values are divided by the minimum estimated magnitude such that the returned magnitudes have a lowest value of 1.0.
"""
function calculate_θmagnitudes(θlb::Vector{<:Float64}, θub::Vector{<:Float64})

    θmagnitudes = zeros(length(θub))
    for i in eachindex(θmagnitudes)
        θmagnitudes[i] = isinf(θub[i]) || isinf(θlb[i]) ? NaN : θub[i] - θlb[i]
    end

    NaN_θmagnitudes = isnan.(θmagnitudes)

    if sum(NaN_θmagnitudes) < length(NaN_θmagnitudes)
        θmagnitudes .= θmagnitudes ./ minimum(θmagnitudes[.!NaN_θmagnitudes])
    end

    return θmagnitudes
end

"""
    initialise_LikelihoodModel(loglikefunction::Function,
        predictfunction::Union{Function, Missing},
        errorfunction::Union{Function, Missing}
        data::Union{Tuple, NamedTuple},
        θnams::Vector{<:Symbol},
        θinitialguess::AbstractVector{<:Real},
        θlb::AbstractVector{<:Real},
        θub::AbstractVector{<:Real},
        θmagnitudes::AbstractVector{<:Real}=Float64[];
        <keyword arguments>)

Initialises a [`LikelihoodModel`](@ref) struct, which contains all model information, profiles, samples and predictions. Solves for the maximum likelihood estimate of `loglikefunction`.

# Arguments
- `loglikefunction`: a log-likelihood function to maximise which takes two arguments, `θ` and `data`, in that order, where θ is a vector containing the values of each parameter in `θnames` and `data` is a Tuple or NamedTuple - see `data` below. Set up to be used in a maximisation objective.
- `predictfunction`: a prediction function to generate model predictions from that is paired with the `loglikefunction`. Requirements for the prediction function can be seen in [`add_prediction_function!`](@ref). It can also be `missing` if no function is provided to [`initialise_LikelihoodModel`](@ref), because predictions are not required when evaluating parameter profiles. The function can be added at a later point using [`add_prediction_function!`](@ref).
- `errorfunction`: an error function used to predict realisations from predictions generated with `predictfunction`. Requirements for the error function can be seen in [`add_error_function!`](@ref). It can also be `missing` if no function is provided to [`initialise_LikelihoodModel`](@ref), because predictions are not required when evaluating parameter profiles. The function can be added at a later point using [`add_error_function!`](@ref).
- `data`: a Tuple or a NamedTuple containing any additional information required by the log-likelihood function, such as the time points to be evaluated at.
- `θnames`: a vector of symbols containing the names of each parameter, e.g. `[:λ, :K, :C0]`.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point.
- `θlb`: a vector of lower bounds on parameters. 
- `θub`: a vector of upper bounds on parameters. 
- `θmagnitudes`: a vector of the relative magnitude of each parameter. If not provided, it will be estimated using the difference of `θlb` and `θub` with [`LikelihoodBasedProfileWiseAnalysis.calculate_θmagnitudes`](@ref). Can be updated after initialisation using [`setmagnitudes!`](@ref).

# Keyword Arguments
- `optimizationsettings`: optimization settings used to optimize the log-likelihood function using [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). Default is `default_OptimizationSettings()` (see [`default_OptimizationSettings`](@ref)).
- `uni_row_prealloaction_size`: number of rows of `uni_profiles_df` to preallocate. Default is `missing` (a single row).
- `biv_row_preallocation_size`: number of rows of `biv_profiles_df` to preallocate. Default is `missing` (a single row).
- `dim_row_preallocation_size`: number of rows of `dim_samples_df` to preallocate. Default is `missing` (a single row).
- `find_zero_atol`: a `Real` number greater than zero for the absolute tolerance of the log-likelihood function value from the target value to be used when searching for confidence intervals/boundaries. Default is `0.001`.
- `show_progress`: Whether to show the progress of profiling and predictions. 

!!! note "Array initialisation within a log-likelihood function"
    If you initialise an array within the provided log-likelihood function, e.g. using `zeros`, then for automatic differentiation methods to work you need to also initialise the type of the array to be based on the type of the input values of θ. Otherwise, zeros will by default create an array with element types `Float64` which will likely return errors. For example, use:
    ```julia
    my_new_array = zeros(eltype(θ), dimensions)
    ``` 
    where `eltype` passes the element type of the θ vector into the `zeros` function.
"""
function initialise_LikelihoodModel(loglikefunction::Function,
    predictfunction::Union{Function, Missing},
    errorfunction::Union{Function, Missing},
    data::Union{Tuple, NamedTuple},
    θnames::Vector{<:Symbol},
    θinitialguess::AbstractVector{<:Real},
    θlb::AbstractVector{<:Real},
    θub::AbstractVector{<:Real},
    θmagnitudes::AbstractVector{<:Real}=Float64[];
    optimizationsettings::OptimizationSettings=default_OptimizationSettings(),
    uni_row_prealloaction_size::Union{Missing,Int}=missing,
    biv_row_preallocation_size::Union{Missing,Int}=missing,
    dim_row_preallocation_size::Union{Missing,Int}=missing,
    find_zero_atol::Real=0.001,
    show_progress::Bool=true)

    # Initialise CoreLikelihoodModel, finding the MLE solution
    θnameToIndex = Dict{Symbol,Int}(name=>i for (i, name) in enumerate(θnames))
    num_pars = length(θnames)

    function argument_handling()
        num_pars == length(θinitialguess) || throw(ArgumentError("The length of θinitialguess must be the same as the length of θnames"))
        num_pars == length(θlb) || throw(ArgumentError("The length of θlb must be the same as the length of θnames"))
        num_pars == length(θub) || throw(ArgumentError("The length of θub must be the same as the length of θnames"))

        if !isempty(θmagnitudes)
            num_pars == length(θmagnitudes) || throw(ArgumentError("The length of θmagnitudes must be the same as the length of θnames"))
        end
        return nothing
    end

    argument_handling()

    function negloglikefunction(θ, data); return -loglikefunction(θ, data) end
    (θmle, maximisedmle) = optimise(negloglikefunction, data, optimizationsettings, θinitialguess, θlb, θub)

    ymle=zeros(0,0)
    if !ismissing(predictfunction)
        ymle = predictfunction(θmle, data)

        if !ismissing(errorfunction)
            errorfunction(ymle, θmle, 0.95) # test to see if it works
        end
    end

    if isempty(θmagnitudes)
        θmagnitudes = calculate_θmagnitudes(θlb, θub)
    end

    corelikelihoodmodel = CoreLikelihoodModel(loglikefunction, predictfunction, errorfunction, optimizationsettings, data, θnames,
                            θnameToIndex, θlb.*1.0, θub.*1.0, θmagnitudes.*1.0, θmle, ymle, maximisedmle, num_pars)

    # conf_levels_evaluated = DefaultDict{Float64, Bool}(false)
    # When initialising a new confidence level, the first line should be written as: 
    # conf_ints_evaluated[conflevel] = DefaultDict{Union{Int, Symbol}, Bool}(false)
    # conf_ints_evaluated = Dict{Float64, DefaultDict{Union{Int, Symbol}, Bool}}()

    num_uni_profiles = 0
    num_biv_profiles = 0
    num_dim_samples = 0

    uni_profiles_df = ismissing(uni_row_prealloaction_size) ? init_uni_profiles_df(1) : init_uni_profiles_df(uni_row_prealloaction_size)
    # if zero, is invalid row
    uni_profile_row_exists = Dict{Tuple{Int, Int, AbstractProfileType}, DefaultDict{Float64, Int}}()    
    # uni_profile_row_exists = DefaultDict{Tuple{Int, Float64, AbstractProfileType}, Int}(0)
    uni_profiles_dict = Dict{Int, UnivariateConfidenceStruct}()

    num_combinations = binomial(num_pars, 2)
    biv_profiles_df = ismissing(biv_row_preallocation_size) ? init_biv_profiles_df(1) : init_biv_profiles_df(biv_row_preallocation_size)
    # if zero, is invalid row
    biv_profile_row_exists = Dict{Tuple{Tuple{Int, Int}, Int, AbstractProfileType, AbstractBivariateMethod}, DefaultDict{Float64, Int}}()
    biv_profiles_dict = Dict{Int, BivariateConfidenceStruct}()

    dim_samples_row_exists = Dict{Union{AbstractSampleType, Tuple{Vector{Int}, Int, AbstractSampleType}}, DefaultDict{Float64, Int}}()
    dim_samples_df = ismissing(dim_row_preallocation_size) ? init_dim_samples_df(1) : init_dim_samples_df(dim_row_preallocation_size)
    dim_samples_dict = Dict{Int, SampledConfidenceStruct}()

    uni_predictions_dict = Dict{Int, AbstractPredictionStruct}()
    biv_predictions_dict = Dict{Int, AbstractPredictionStruct}()
    dim_predictions_dict = Dict{Int, AbstractPredictionStruct}()

    likelihoodmodel = LikelihoodModel(corelikelihoodmodel,
                                    missing, 
                                    num_uni_profiles, num_biv_profiles, num_dim_samples,
                                    uni_profiles_df, biv_profiles_df, dim_samples_df,
                                    uni_profile_row_exists, biv_profile_row_exists,
                                    dim_samples_row_exists,
                                    uni_profiles_dict, biv_profiles_dict, dim_samples_dict,
                                    uni_predictions_dict, biv_predictions_dict,
                                    dim_predictions_dict, find_zero_atol,
                                    show_progress)

    return likelihoodmodel
end

"""
    initialise_LikelihoodModel(loglikefunction::Function,
        predictfunction::Function,
        data::Union{Tuple, NamedTuple},
        θnames::Vector{<:Symbol},
        θinitialGuess::Vector{<:Float64},
        θlb::Vector{<:Float64},
        θub::Vector{<:Float64},
        θmagnitudes::Vector{<:Real}=zeros(0);
        <keyword arguments>)

Alternate version of [`initialise_LikelihoodModel`](@ref) that can be called without a error function. The function can be added at a later point using [`add_error_function!`](@ref).
"""
function initialise_LikelihoodModel(loglikefunction::Function,
    predictfunction::Function,
    data::Union{Tuple,NamedTuple},
    θnames::Vector{<:Symbol},
    θinitialGuess::Vector{<:Float64},
    θlb::Vector{<:Float64},
    θub::Vector{<:Float64},
    θmagnitudes::Vector{<:Real}=zeros(0);
    optimizationsettings::OptimizationSettings=default_OptimizationSettings(),
    uni_row_prealloaction_size::Union{Missing,Int}=missing,
    biv_row_preallocation_size::Union{Missing,Int}=missing,
    dim_row_preallocation_size::Union{Missing,Int}=missing,
    find_zero_atol::Real=0.001,
    show_progress::Bool=true)

    return initialise_LikelihoodModel(loglikefunction, predictfunction, missing, data, θnames,
        θinitialGuess, θlb, θub, θmagnitudes,
        optimizationsettings=optimizationsettings,
        uni_row_prealloaction_size=uni_row_prealloaction_size,
        biv_row_preallocation_size=biv_row_preallocation_size,
        dim_row_preallocation_size=dim_row_preallocation_size,
        find_zero_atol=find_zero_atol,
        show_progress=show_progress)
end

"""
    initialise_LikelihoodModel(loglikefunction::Function,
        data::Union{Tuple, NamedTuple},
        θnames::Vector{<:Symbol},
        θinitialGuess::Vector{<:Float64},
        θlb::Vector{<:Float64},
        θub::Vector{<:Float64},
        θmagnitudes::Vector{<:Real}=zeros(0);
        <keyword arguments>)

Alternate version of [`initialise_LikelihoodModel`](@ref) that can be called without a prediction and error function. The functions can be added at a later point using [`add_prediction_function!`](@ref) and [`add_error_function!`](@ref).
"""
function initialise_LikelihoodModel(loglikefunction::Function,
    data::Union{Tuple, NamedTuple},
    θnames::Vector{<:Symbol},
    θinitialGuess::Vector{<:Float64},
    θlb::Vector{<:Float64},
    θub::Vector{<:Float64},
    θmagnitudes::Vector{<:Real}=zeros(0);
    optimizationsettings::OptimizationSettings=default_OptimizationSettings(),
    uni_row_prealloaction_size::Union{Missing,Int}=missing,
    biv_row_preallocation_size::Union{Missing,Int}=missing,
    dim_row_preallocation_size::Union{Missing,Int}=missing,
    find_zero_atol::Real=0.001,
    show_progress::Bool=true)

    return initialise_LikelihoodModel(loglikefunction, missing, missing, data, θnames,
                                        θinitialGuess, θlb, θub, θmagnitudes,
                                        optimizationsettings=optimizationsettings,
                                        uni_row_prealloaction_size=uni_row_prealloaction_size,
                                        biv_row_preallocation_size=biv_row_preallocation_size,
                                        dim_row_preallocation_size=dim_row_preallocation_size, 
                                        find_zero_atol=find_zero_atol,
                                        show_progress=show_progress)
end