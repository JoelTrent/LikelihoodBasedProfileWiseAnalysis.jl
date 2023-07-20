"""
    default_OptimizationSettings()

Creates a [`OptimizationSettings`](@ref) struct with defaults of:
- `adtype`: `SciMLBase.NoAD()` (no automatic differentiation). 
- `solve_alg`: [`NLopt.LN_BOBYQA()`](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#bobyqa).
- `solve_kwargs`: `(maxtime=15, xtol_rel=1e-9)`.

If this function causes an error then `PlaceholderLikelihood` needs to be loaded. Alternatively, the packages `SciMLBase`, `Optimization` and `OptimizationNLopt` need to be loaded.
"""
function default_OptimizationSettings()
    return OptimizationSettings(SciMLBase.NoAD(), NLopt.LN_BOBYQA(), (maxtime=15, xtol_rel=1e-9))
end

"""
    create_OptimizationSettings(model::LikelihoodModel;
        adtype::Union{SciMLBase.AbstractADType, Missing}=missing, 
        solve_alg=missing, 
        solve_kwargs::Union{NamedTuple, Missing}=missing)

Method for creating a [`OptimizationSettings`](@ref) struct with each field of the struct as a keyword argument. If a keyword argument is not provided, then the setting in `model.core.optimizationsettings` is used (the currently set optimization settings).
"""
function create_OptimizationSettings(model::LikelihoodModel;
    adtype::Union{SciMLBase.AbstractADType,Missing}=missing,
    solve_alg=missing,
    solve_kwargs::Union{NamedTuple,Missing}=missing)

    if model.core isa CoreLikelihoodModel
        defaults = model.core.optimizationsettings
    else
        defaults = defaultOptimizationSettings()
    end
    adtype = ismissing(adtype) ? defaults.adtype : adtype
    solve_alg = ismissing(solve_alg) ? defaults.solve_alg : solve_alg
    solve_kwargs = ismissing(solve_kwargs) ? defaults.solve_kwargs : solve_kwargs

    return OptimizationSettings(adtype, solve_alg, solve_kwargs)
end

"""
    create_OptimizationSettings(;
        adtype::Union{SciMLBase.AbstractADType, Missing}=missing, 
        solve_alg=missing, 
        solve_kwargs::Union{NamedTuple, Missing}=missing)

Method for creating a [`OptimizationSettings`](@ref) struct with each field of the struct as a keyword argument. If a keyword argument is not provided, then the default setting in [`default_OptimizationSettings`](@ref) is used.
"""
function create_OptimizationSettings(;
    adtype::Union{SciMLBase.AbstractADType,Missing}=missing,
    solve_alg=missing,
    solve_kwargs::Union{NamedTuple,Missing}=missing)

    defaults = defaultOptimizationSettings()
    adtype = ismissing(adtype) ? defaults.adtype : adtype
    solve_alg = ismissing(solve_alg) ? defaults.solve_alg : solve_alg
    solve_kwargs = ismissing(solve_kwargs) ? defaults.solve_kwargs : solve_kwargs

    return OptimizationSettings(adtype, solve_alg, solve_kwargs)
end

"""
    set_OptimizationSettings!(model::LikelihoodModel, optimizationsettings::OptimizationSettings)

Updates the optimization settings contained with `model` with `optimizationsettings`.
"""
function set_OptimizationSettings!(model::LikelihoodModel, 
    optimizationsettings::OptimizationSettings)

    corelikelihoodmodel = model.core
    model.core = @set corelikelihoodmodel.optimizationsettings = optimizationsettings
    return nothing
end
