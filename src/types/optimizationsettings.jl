"""
    OptimizationSettings(adtype::SciMLBase.AbstractADType,
        solve_alg,
        solve_kwargs::NamedTuple)

Contains the optimization settings used to solve for nuisance parameters of the log-likelihood function using [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). Note: Optimization.jl uses a minimisation objective, while the log-likelihood functions are setup with a maximisation objective in mind. This is handled inside the package. Defaults can be seen in [`default_OptimizationSettings`](@ref).

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