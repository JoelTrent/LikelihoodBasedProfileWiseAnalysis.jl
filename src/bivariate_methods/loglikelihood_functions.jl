"""
    bivariateψ!(ψ::Real, p::NamedTuple)

Given a log-likelihood function (`p.consistent.loglikefunction`) which is bounded in parameter space and may be an ellipse approximation, this function finds the values of the nuisance parameters ω that optimise the function fixed values of the interest parameters `p.ψ_x[1]` and `ψ` and returns the log-likelihood value minus the confidence boundary target threshold. The returned function value will be zero at the locations of the approximate confidence boundary for `p.ψ_x[1]` and `ψ`. Nuisance parameter values are stored in the NamedTuple `p` at `p.ω_opt`. Used by [`Fix1AxisMethod`](@ref).
"""
function bivariateψ!(ψ::Real, p::NamedTuple)
    θs=zeros(p.consistent.num_pars)
    
    function fun(ω)
        θs[p.ind1] = p.ψ_x[1]
        θs[p.ind2] = ψ
        return p.consistent.loglikefunction(variablemapping!(θs, ω, p.θranges, p.ωranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    p.ω_opt .= xopt
    return llb
end

"""
    bivariateψ_vectorsearch!(ψ::Real, p::NamedTuple)

Given a log-likelihood function (`p.consistent.loglikefunction`) which is bounded in parameter space and may be an ellipse approximation, this function finds the values of the nuisance parameters ω that optimise the function fixed values of the interest parameters `ψxy = p.pointa + ψ*p.uhat` and returns the log-likelihood value minus the confidence boundary target threshold. The returned function value will be zero at the locations of the approximate confidence boundary for `ψxy`. Nuisance parameter values are stored in the NamedTuple `p` at `p.ω_opt`. Used by [`AbstractBivariateVectorMethod`](@ref).
"""
function bivariateψ_vectorsearch!(ψ::Real, p::NamedTuple)
    θs=zeros(p.consistent.num_pars)
    ψxy = p.pointa + ψ*p.uhat
    
    function fun(ω)
        θs[p.ind1], θs[p.ind2] = ψxy
        return p.consistent.loglikefunction(variablemapping!(θs, ω, p.θranges, p.ωranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    p.ω_opt .= xopt
    return llb
end

"""
    bivariateψ_continuation!(ψ::Real, p::NamedTuple)

Given a log-likelihood function (`p.consistent.loglikefunction`) which is bounded in parameter space and may be an ellipse approximation, this function finds the values of the nuisance parameters ω that optimise the function fixed values of the interest parameters `ψxy = p.pointa + ψ*p.uhat` and returns the log-likelihood value minus the confidence boundary target threshold. The returned function value will be zero at the locations of the approximate confidence boundary for `ψxy`. Nuisance parameter values are stored in the NamedTuple `p` at `p.ω_opt`. Used by [`ContinuationMethod`](@ref).
"""
function bivariateψ_continuation!(ψ::Real, p::NamedTuple)
    θs=zeros(p.consistent.num_pars)
    ψxy = p.pointa + ψ*p.uhat
    
    function fun(ω)
        θs[p.ind1], θs[p.ind2] = ψxy
        return p.consistent.loglikefunction(variablemapping!(θs, ω, p.θranges, p.ωranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.targetll
    p.ω_opt .= xopt
    return llb
end

"""
    bivariateψ_ellipse_analytical(ψ::Real, p::NamedTuple)

Returns the approximated log-likelihood value minus the confidence boundary target threshold, given an analytic ellipse approximation of a log-likelihood function ([`PlaceholderLikelihood.analytic_ellipse_loglike`](@ref)) which is unbounded in parameter space, for values of the interest parameters `p.ψ_x[1]` and `ψ`. The returned function value will be zero at the locations of the approximate confidence boundary for `p.ψ_x[1]` and `ψ`, which correspond to locations found by [`PlaceholderLikelihood.AnalyticalEllipseMethod`](@ref). Used by [`Fix1AxisMethod`](@ref). 
"""
function bivariateψ_ellipse_analytical(ψ::Real, p::NamedTuple)
    return analytic_ellipse_loglike([p.ψ_x[1], ψ], [p.ind1, p.ind2], p.consistent.data_analytic) - p.consistent.targetll
end

"""
    bivariateψ_ellipse_analytical_vectorsearch(ψ::Real, p::NamedTuple)

Returns the approximated log-likelihood value minus the confidence boundary target threshold, given an analytic ellipse approximation of a log-likelihood function ([`PlaceholderLikelihood.analytic_ellipse_loglike`](@ref)) which is unbounded in parameter space, for values of the interest parameters `ψxy = p.pointa + ψ*p.uhat`. The returned function value will be zero at the locations of the approximate confidence boundary for `ψxy = p.pointa + ψ*p.uhat`, which correspond to locations found by [`PlaceholderLikelihood.AnalyticalEllipseMethod`](@ref). Used by [`AbstractBivariateVectorMethod`](@ref).
"""
function bivariateψ_ellipse_analytical_vectorsearch(ψ::Real, p::NamedTuple)
    return analytic_ellipse_loglike(p.pointa + ψ*p.uhat, [p.ind1, p.ind2], p.consistent.data_analytic) - p.consistent.targetll
end

"""
    bivariateψ_ellipse_analytical_continuation(ψ::Real, p::NamedTuple)

Returns the approximated log-likelihood value minus the confidence boundary target threshold, given an analytic ellipse approximation of a log-likelihood function ([`PlaceholderLikelihood.analytic_ellipse_loglike`](@ref)) which is unbounded in parameter space, for values of the interest parameters `ψxy = p.pointa + ψ*p.uhat`. The returned function value will be zero at the locations of the approximate confidence boundary for `ψxy = p.pointa + ψ*p.uhat`, which correspond to locations found by [`PlaceholderLikelihood.AnalyticalEllipseMethod`](@ref). Used by [`ContinuationMethod`](@ref).
"""
function bivariateψ_ellipse_analytical_continuation(ψ::Real, p::NamedTuple)
    return analytic_ellipse_loglike(p.pointa + ψ*p.uhat, [p.ind1, p.ind2], p.consistent.data_analytic) - p.targetll
end

"""
    bivariateψ_ellipse_unbounded(ψ::Vector, p::NamedTuple)

Given an ellipse approximation of a log-likelihood function ([`PlaceholderLikelihood.ellipse_loglike`](@ref)) which is unbounded in parameter space, this function finds the values of the nuisance parameters ω that optimise the function at fixed values of the two interest parameters in `ψ` and returns the approximated log-likelihood value minus the confidence boundary target threshold. The returned function value will be zero at the locations of the approximate confidence boundary for `ψ`, which correspond to the locations found by [`PlaceholderLikelihood.AnalyticalEllipseMethod`](@ref). Nuisance parameter values are stored in the NamedTuple `p`. 
"""
function bivariateψ_ellipse_unbounded(ψ::Vector, p::NamedTuple)
    θs=zeros(p.consistent.num_pars)
    θs[p.ind1], θs[p.ind2] = ψ

    function fun(ω)
        return ellipse_loglike(variablemapping!(θs, ω, p.θranges, p.ωranges), p.consistent.data) 
    end

    (xopt,_)=optimise_unbounded(fun, p.initGuess)
    return xopt
end