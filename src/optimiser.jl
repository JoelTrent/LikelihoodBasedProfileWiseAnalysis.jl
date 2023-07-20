"""
    optimise(fun, p, options::OptimizationSettings, θ₀, lb, ub)

Optimization.jl optimiser used for calculating the values of nuisance parameters. Default values of options use NLopt.jl algorithms (see[`default_OptimizationSettings`](@ref)).
"""
function optimise(fun, q, options::OptimizationSettings, θ₀, lb, ub)
    fopt = OptimizationFunction(fun, options.adtype)
    prob = OptimizationProblem(fopt, θ₀, q; lb=lb, ub=ub, options.solve_kwargs...)
    sol = solve(prob, options.solve_alg)
    return sol.u, -sol.objective
end

"""
    optimise_unbounded(fun, p, options::OptimizationSettings, θ₀)

Alternative version of [`optimise`](@ref) without nuisance parameter bounds. Used for computing the nuisance parameters of [`EllipseApproxAnalytical`](@ref) profiles. Default values of options use NLopt.jl algorithms (see[`default_OptimizationSettings`](@ref)).
"""
function optimise_unbounded(fun, q, options::OptimizationSettings, θ₀)
    fopt = OptimizationFunction(fun, options.adtype)
    prob = OptimizationProblem(fopt, θ₀, q; options.solve_kwargs...)
    sol = solve(prob, options.solve_alg)
    return sol.u, -sol.objective
end

"""
    optimise(fun, θ₀, lb, ub;
        dv = false, method = dv ? :LD_LBFGS : :LN_BOBYQA)

NLopt optimiser used for calculating the values of nuisance parameters.
"""
function optimise(fun, θ₀, lb, ub;
    dv = false,
    method = dv ? :LD_LBFGS : :LN_BOBYQA
    )

    if dv || String(method)[2] == 'D'
        tomax = fun
    else
        tomax = (θ,∂θ) -> fun(θ)
    end

    opt = Opt(method,length(θ₀))
    opt.max_objective = tomax
    opt.lower_bounds = lb       # Lower bound
    opt.upper_bounds = ub       # Upper bound
    # opt.local_optimizer = Opt(:LN_NELDERMEAD, length(θ₀))
    opt.xtol_rel=1e-9
    # opt.maxeval=4000
    opt.maxtime=15
    res = NLopt.optimize(opt, θ₀)
    return res[[2,1]]
end

"""
    optimise_unbounded(fun, θ₀;
        dv = false, method = dv ? :LD_LBFGS : :LN_BOBYQA)

Alternative version of [`optimise`](@ref) without nuisance parameter bounds. Used for computing the nuisance parameters of [`EllipseApproxAnalytical`](@ref) profiles.
"""
function optimise_unbounded(fun, θ₀;
    dv = false,
    method = dv ? :LD_LBFGS : :LN_BOBYQA,
    )

    if dv || String(method)[2] == 'D'
        tomax = fun
    else
        tomax = (θ,∂θ) -> fun(θ)
    end

    opt = Opt(method,length(θ₀))
    opt.max_objective = tomax
    # opt.local_optimizer = Opt(:LN_NELDERMEAD, length(θ₀))
    res = NLopt.optimize(opt, θ₀)
    return res[[2,1]]
end