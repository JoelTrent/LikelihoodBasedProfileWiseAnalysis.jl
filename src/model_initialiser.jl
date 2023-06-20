"""
    init_uni_profile_row_exists!(model::LikelihoodModel, 
        θs_to_profile::Vector{<:Int}, 
        profile_type::AbstractProfileType)

Initialises the dictionary entry in `model.uni_profile_row_exists` for the key `(θi, profile_type)`, where `θi` is an element of `θs_to_profile`, with a `DefaultDict` with key of type `Float64` (a confidence level) and default value of 0.
"""
function init_uni_profile_row_exists!(model::LikelihoodModel, 
                                        θs_to_profile::Vector{<:Int},
                                        profile_type::AbstractProfileType)
    for θi in θs_to_profile
        if !haskey(model.uni_profile_row_exists, (θi, profile_type))
            model.uni_profile_row_exists[(θi, profile_type)] = DefaultDict{Float64, Int}(0)
        end
    end
    return nothing
end

"""
    init_biv_profile_row_exists!(model::LikelihoodModel, 
        θcombinations::Vector{Vector{Int}}, 
        profile_type::AbstractProfileType, 
        method::AbstractBivariateMethod)

Initialises the dictionary entry in `model.biv_profile_row_exists` for the key `((ind1, ind2), profile_type, method)`, where `(ind1, ind2)` is a combination in `θcombinations`, with a `DefaultDict` with key of type `Float64` (a confidence level) and default value of 0.
"""
function init_biv_profile_row_exists!(model::LikelihoodModel, 
                                        θcombinations::Vector{Vector{Int}},
                                        profile_type::AbstractProfileType,
                                        method::AbstractBivariateMethod)
    for (ind1, ind2) in θcombinations
        if !haskey(model.biv_profile_row_exists, ((ind1, ind2), profile_type, method))
            model.biv_profile_row_exists[((ind1, ind2), profile_type, method)] = DefaultDict{Float64, Int}(0)
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

Initialises the dictionary entry in `model.dim_samples_row_exists` for the key `(θvec, sample_type)`, where `θvec` is a vector in `θindices`, with a `DefaultDict` with key of type `Float64` (a confidence level) and default value of 0. For a non-full likelihood sample (dimension less than the number of model parameters).
"""
function init_dim_samples_row_exists!(model::LikelihoodModel, 
                                        θindices::Vector{Vector{Int}},
                                        sample_type::AbstractSampleType)
    for θvec in θindices
        if !haskey(model.dim_samples_row_exists, (θvec, sample_type))
            model.dim_samples_row_exists[(θvec, sample_type)] = DefaultDict{Float64, Int}(0)
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
    uni_profiles_df.profile_type = Vector{AbstractProfileType}(undef, num_rows)
    uni_profiles_df.num_points = zeros(Int, num_rows)
    uni_profiles_df.additional_width = zeros(num_rows)

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
    biv_profiles_df.profile_type = Vector{AbstractProfileType}(undef, num_rows)
    biv_profiles_df.method = Vector{AbstractBivariateMethod}(undef, num_rows)
    biv_profiles_df.num_points = zeros(Int, num_rows)

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
    dim_samples_df.sample_type = Vector{AbstractSampleType}(undef, num_rows)
    dim_samples_df.num_points = zeros(Int, num_rows)

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
        θmagnitudes .= θmagnitudes ./ min(θmagnitudes[.!NaN_θmagnitudes])
    end

    return θmagnitudes
end

"""
    initialiseLikelihoodModel(loglikefunction::Function,
        predictfunction::Union{Function, Missing},
        data::Union{Tuple, NamedTuple},
        θnames::Vector{<:Symbol},
        θinitialGuess::AbstractVector{<:Real},
        θlb::AbstractVector{<:Real},
        θub::AbstractVector{<:Real},
        θmagnitudes::AbstractVector{<:Real}=Float64[];
        uni_row_prealloaction_size=NaN,
        biv_row_preallocation_size=NaN,
        dim_row_preallocation_size=NaN,
        show_progress=true)

Initialises a [`LikelihoodModel`](@ref) struct, which contains all model information, profiles, samples and predictions.

# Arguments
- `loglikefunction`: a loglikelihood function which takes two arguments, `θ` and `data`, in that order, where θ is a vector containing the values of each parameter in `θnames` and `data` is a Tuple or NamedTuple - see `data` below.
- `predictfunction`: a prediction function to generate model predictions from that is paired with the `loglikefunction`. Takes three arguments, `θ`, `data` and `t`, in that order, where `θ` and `data` are the same as for `loglikefunction` and `t` needs to be an optional third argument. When `t` is not specified, the prediction function should be evaluated for the same time points/independent variable as the data. When `t` is specified, the prediction function should be evaluated for those specified time points/independent variable. It can also be `missing` if no function is provided to [`initialiseLikelihoodModel`](@ref), because predictions are not required when evaluating parameter profiles. The function can be added at a later point using [`add_prediction_function!`](@ref).
- `data`: a Tuple or a NamedTuple containing any additional information required by the log-likelihood function, such as the time points to be evaluated at.
- `θnames`: a vector of symbols containing the names of each parameter, e.g. `[:λ, :K, :C0]`.
- `θinitialguess`: a vector containing the initial guess for the values of each parameter. Used to find the MLE point.
- `θlb`: a vector of lower bounds on parameters. 
- `θub`: a vector of upper bounds on parameters. 
- `θmagnitudes`: a vector of the relative magnitude of each parameter. If not provided, it will be estimated using the difference of `θlb` and `θub` with [`PlaceholderLikelihood.calculate_θmagnitudes`](@ref). Can be updated after initialisation using [`setθmagnitudes!`](@ref).

# Keyword Arguments
- `uni_row_prealloaction_size`: number of rows of `uni_profiles_df` to preallocate. Default is NaN (a single row).
- `biv_row_preallocation_size`:number of rows of `biv_profiles_df` to preallocate. Default is NaN (a single row).
- `dim_row_preallocation_size`: number of rows of `dim_samples_df` to preallocate. Default is NaN (a single row).
- `show_progress`: Whether to show the progress of profiling across sets of interest parameters. 
"""
function initialiseLikelihoodModel(loglikefunction::Function,
    predictfunction::Union{Function, Missing},
    data::Union{Tuple, NamedTuple},
    θnames::Vector{<:Symbol},
    θinitialguess::AbstractVector{<:Real},
    θlb::AbstractVector{<:Real},
    θub::AbstractVector{<:Real},
    θmagnitudes::AbstractVector{<:Real}=Float64[];
    uni_row_prealloaction_size=NaN,
    biv_row_preallocation_size=NaN,
    dim_row_preallocation_size=NaN,
    show_progress=true)

    # Initialise CoreLikelihoodModel, finding the MLE solution
    θnameToIndex = Dict{Symbol,Int}(name=>i for (i, name) in enumerate(θnames))
    num_pars = length(θnames)

    function funmle(θ); return loglikefunction(θ, data) end
    (θmle, maximisedmle) = optimise(funmle, θinitialguess, θlb, θub)

    ymle=zeros(0,0)
    if !ismissing(predictfunction)
        ymle = predictfunction(θmle, data)
    end

    if isempty(θmagnitudes)
        θmagnitudes = calculate_θmagnitudes(θlb, θub)
    end

    corelikelihoodmodel = CoreLikelihoodModel(loglikefunction, predictfunction, data, θnames, θnameToIndex,
                                        θlb.*1.0, θub.*1.0, θmagnitudes.*1.0, θmle, ymle, maximisedmle, num_pars)


    # conf_levels_evaluated = DefaultDict{Float64, Bool}(false)
    # When initialising a new confidence level, the first line should be written as: 
    # conf_ints_evaluated[conflevel] = DefaultDict{Union{Int, Symbol}, Bool}(false)
    # conf_ints_evaluated = Dict{Float64, DefaultDict{Union{Int, Symbol}, Bool}}()

    num_uni_profiles = 0
    num_biv_profiles = 0
    num_dim_samples = 0

    uni_profiles_df = isnan(uni_row_prealloaction_size) ? init_uni_profiles_df(num_pars) : init_uni_profiles_df(uni_row_prealloaction_size)
    # if zero, is invalid row
    uni_profile_row_exists = Dict{Tuple{Int, AbstractProfileType}, DefaultDict{Float64, Int}}()    
    # uni_profile_row_exists = DefaultDict{Tuple{Int, Float64, AbstractProfileType}, Int}(0)
    uni_profiles_dict = Dict{Int, UnivariateConfidenceStruct}()

    num_combinations = binomial(num_pars, 2)
    biv_profiles_df = isnan(biv_row_preallocation_size) ? init_biv_profiles_df(num_combinations) : init_biv_profiles_df(biv_row_preallocation_size)
    # if zero, is invalid row
    biv_profile_row_exists = Dict{Tuple{Tuple{Int, Int}, AbstractProfileType, AbstractBivariateMethod}, DefaultDict{Float64, Int}}()
    biv_profiles_dict = Dict{Int, BivariateConfidenceStruct}()

    dim_samples_row_exists = Dict{Union{AbstractSampleType, Tuple{Vector{Int}, AbstractSampleType}}, DefaultDict{Float64, Int}}()
    dim_samples_df = isnan(dim_row_preallocation_size) ? init_dim_samples_df(1) : init_dim_samples_df(dim_row_preallocation_size)
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
                                    dim_predictions_dict, 
                                    show_progress)

    return likelihoodmodel
end

"""
    initialiseLikelihoodModel(loglikefunction::Function,
        data::Union{Tuple, NamedTuple},
        θnames::Vector{<:Symbol},
        θinitialGuess::Vector{<:Float64},
        θlb::Vector{<:Float64},
        θub::Vector{<:Float64},
        θmagnitudes::Vector{<:Real}=zeros(0);
        uni_row_prealloaction_size=NaN,
        biv_row_preallocation_size=NaN,
        dim_row_preallocation_size=NaN,
        show_progress=true)

Alternate version of [`initialiseLikelihoodModel`](@ref) that can be called without a prediction function. The function can be added at a later point using [`add_prediction_function!`](@ref).
"""
function initialiseLikelihoodModel(loglikefunction::Function,
    data::Union{Tuple, NamedTuple},
    θnames::Vector{<:Symbol},
    θinitialGuess::Vector{<:Float64},
    θlb::Vector{<:Float64},
    θub::Vector{<:Float64},
    θmagnitudes::Vector{<:Real}=zeros(0);
    uni_row_prealloaction_size=NaN,
    biv_row_preallocation_size=NaN,
    dim_row_preallocation_size=NaN,
    show_progress=true)

    return initialiseLikelihoodModel(loglikefunction, missing, data, θnames,
                                        θinitialGuess, θlb, θub, θmagnitudes,
                                        uni_row_prealloaction_size=uni_row_prealloaction_size,
                                        biv_row_preallocation_size=biv_row_preallocation_size,
                                        dim_row_preallocation_size=dim_row_preallocation_size, 
                                        show_progress=show_progress)
end