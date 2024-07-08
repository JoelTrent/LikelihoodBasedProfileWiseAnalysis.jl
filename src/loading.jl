"""
    trim_model_dfs!(model::LikelihoodModel)

Removes any unitialised rows of `model.uni_profiles_df`, `model.biv_profiles_df` and `model.dim_samples_df` in place, which contain undefined references and will prevent saving using [BSON.jl](https://github.com/JuliaIO/BSON.jl/tree/master).
"""
function trim_model_dfs!(model::LikelihoodModel)
    if nrow(model.uni_profiles_df) > model.num_uni_profiles
        model.uni_profiles_df = model.uni_profiles_df[1:model.num_uni_profiles, :]
    end
    if nrow(model.biv_profiles_df) > model.num_biv_profiles
        model.biv_profiles_df = model.biv_profiles_df[1:model.num_biv_profiles, :]
    end
    if nrow(model.dim_samples_df) > model.num_dim_samples
        model.dim_samples_df = model.dim_samples_df[1:model.num_dim_samples, :]
    end
    return nothing
end

"""
    remove_functions_from_core!(model::LikelihoodModel)

Removes the functions from `model.core` by replacing the [`CoreLikelihoodModel`](@ref) at `model.core` with a [`BaseLikelihoodModel`](@ref), returning the [`CoreLikelihoodModel`](@ref).
"""
function remove_functions_from_core!(model::LikelihoodModel)

    corelikelihoodmodel = model.core
    model.core = BaseLikelihoodModel([getfield(corelikelihoodmodel, k) 
                                    for k ∈ fieldnames(BaseLikelihoodModel)]...)
    return corelikelihoodmodel
end

"""
    add_loglikelihood_function!(model::LikelihoodModel, loglikefunction::Function; 
        optimizationsettings::OptimizationSettings=default_OptimizationSettings())

Adds a log-likelihood function, `loglikefunction`, to `model` as well as optimization settings, `optimizationsettings`, using [`default_OptimizationSettings`](@ref).
    
Requirements for `loglikefunction`: loglikelihood function which takes two arguments, `θ` and `data`, in that order, where θ is a vector containing the values of each parameter in `θnames` and `data` is a Tuple or NamedTuple containing any additional information required by the log-likelihood function, such as the time points to be evaluated at.
"""
function add_loglikelihood_function!(model::LikelihoodModel, loglikefunction::Function;
        optimizationsettings::OptimizationSettings=default_OptimizationSettings())

    if model.core isa CoreLikelihoodModel
        return nothing
    end
    corelikelihoodmodel = model.core
    model.core = CoreLikelihoodModel(vcat([loglikefunction, missing, optimizationsettings], 
                                            [getfield(corelikelihoodmodel, k)
                                            for k ∈ fieldnames(BaseLikelihoodModel)])...)
    return nothing
end