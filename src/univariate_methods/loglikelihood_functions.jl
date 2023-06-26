# function univariateψ_ellipse_analytical(ψ, p)
#     return analytic_ellipse_loglike([ψ], [p.ind], p.consistent.data) - p.consistent.targetll
# end
"""
    univariateψ_ellipse_unbounded(ψ::Real, p::NamedTuple)

Given an ellipse approximation of a log-likelihood function ([`PlaceholderLikelihood.ellipse_loglike`](@ref)) which is unbounded in parameter space, this function finds the values of the nuisance parameters ω that optimise the function at fixed values of the interest parameter ψ and returns the approximated log-likelihood value minus the confidence interval target threshold. The returned function value will be zero at the locations of the approximate confidence interval for ψ, which correspond to the locations found by [`PlaceholderLikelihood.analytic_ellipse_loglike_1D_soln`](@ref). Nuisance parameter values are stored in the NamedTuple `p`. 
"""
function univariateψ_ellipse_unbounded(ψ::Real, p::NamedTuple)
    θs=zeros(p.consistent.num_pars)

    function fun(ω)
        θs[p.ind] = ψ
        return ellipse_loglike(variablemapping!(θs, ω, p.θranges, p.ωranges), p.consistent.data) 
    end

    (xopt,fopt)=optimise_unbounded(fun, p.initGuess)
    llb=fopt-p.consistent.targetll
    p.ω_opt .= xopt
    return llb
end

"""
    univariateψ(ψ::Real, p::NamedTuple)

Given a log-likelihood function (`p.consistent.loglikefunction`) which is bounded in parameter space and may be an ellipse approximation, this function finds the values of the nuisance parameters ω that optimise the function fixed values of the interest parameter ψ and returns the log-likelihood value minus the confidence interval target threshold. The returned function value will be zero at the locations of the approximate confidence interval for ψ. Nuisance parameter values are stored in the NamedTuple `p` at `p.ω_opt`. 
"""
function univariateψ(ψ::Real, p::NamedTuple)
    θs=zeros(p.consistent.num_pars)

    function fun(ω)
        θs[p.ind] = ψ
        return p.consistent.loglikefunction(variablemapping!(θs, ω, p.θranges, p.ωranges), p.consistent.data) 
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    p.ω_opt .= xopt
    return llb
end