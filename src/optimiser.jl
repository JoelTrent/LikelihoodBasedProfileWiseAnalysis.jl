"""
    optimise(fun, q, options::OptimizationSettings, θ₀, lb, ub)

Optimization.jl optimiser used for calculating the values of nuisance parameters. Default values of options use NLopt.jl algorithms (see[`default_OptimizationSettings`](@ref)).
"""
function optimise(fun, q, options::OptimizationSettings, θ₀, lb, ub)
    fopt = OptimizationFunction(fun, options.adtype)
    prob = OptimizationProblem(fopt, θ₀, q; lb=lb, ub=ub, options.solve_kwargs...)
    sol = solve(prob, options.solve_alg)
    return sol.u, -sol.objective
end

"""
    optimise_unbounded(fun, q, options::OptimizationSettings, θ₀)

Alternative version of [`optimise`](@ref) without nuisance parameter bounds. Used for computing the nuisance parameters of [`EllipseApproxAnalytical`](@ref) profiles. Default values of options use NLopt.jl algorithms (see[`default_OptimizationSettings`](@ref)).
"""
function optimise_unbounded(fun, q, options::OptimizationSettings, θ₀)
    fopt = OptimizationFunction(fun, options.adtype)
    prob = OptimizationProblem(fopt, θ₀, q; options.solve_kwargs...)
    sol = solve(prob, options.solve_alg)
    return sol.u, -sol.objective
end

"""
    optimise(fun, options::OptimizationSettings, θ₀, lb, ub)

Optimization.jl optimiser used for calculating the bound transformations. Default values of options use NLopt.jl algorithms (see[`default_OptimizationSettings`](@ref)).
"""
function optimise(fun, options::OptimizationSettings, θ₀, lb, ub)
    fopt = OptimizationFunction(fun, options.adtype)
    prob = OptimizationProblem(fopt, θ₀; lb=lb, ub=ub, options.solve_kwargs...)
    sol = solve(prob, options.solve_alg)
    return sol.u, -sol.objective
end