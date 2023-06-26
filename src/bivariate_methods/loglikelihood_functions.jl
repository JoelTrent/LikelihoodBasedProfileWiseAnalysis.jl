function bivariateψ!(ψ::Real, p)
    θs=zeros(p.consistent.num_pars)
    
    function fun(ω)
        θs[p.ind1] = p.ψ_x[1]
        θs[p.ind2] = ψ
        return p.consistent.loglikefunction(variablemapping2d!(θs, ω, p.θranges, p.ωranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    p.ω_opt .= xopt
    return llb
end

function bivariateψ_vectorsearch!(ψ, p)
    θs=zeros(p.consistent.num_pars)
    ψxy = p.pointa + ψ*p.uhat
    
    function fun(ω)
        θs[p.ind1], θs[p.ind2] = ψxy
        return p.consistent.loglikefunction(variablemapping2d!(θs, ω, p.θranges, p.ωranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    p.ω_opt .= xopt
    return llb
end

function bivariateψ_continuation!(ψ, p)
    θs=zeros(p.consistent.num_pars)
    ψxy = p.pointa + ψ*p.uhat
    
    function fun(ω)
        θs[p.ind1], θs[p.ind2] = ψxy
        return p.consistent.loglikefunction(variablemapping2d!(θs, ω, p.θranges, p.ωranges), p.consistent.data)
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.targetll
    p.ω_opt .= xopt
    return llb
end

"""
Requires optimal values of nuisance parameters at point ψ to be contained in p.ω_opt
"""
function bivariateψ_gradient!(ψ::Vector, p)
    θs=zeros(eltype(ψ), p.consistent.num_pars)

    θs[p.ind1], θs[p.ind2] = ψ
    variablemapping2d!(θs, p.ω_opt, p.θranges, p.ωranges)
    return p.consistent.loglikefunction(θs, p.consistent.data)
end

function bivariateψ_ellipse_analytical(ψ, p)
    return analytic_ellipse_loglike([p.ψ_x[1], ψ], [p.ind1, p.ind2], p.consistent.data_analytic) - p.consistent.targetll
end

function bivariateψ_ellipse_analytical_vectorsearch(ψ, p)
    return analytic_ellipse_loglike(p.pointa + ψ*p.uhat, [p.ind1, p.ind2], p.consistent.data_analytic) - p.consistent.targetll
end

function bivariateψ_ellipse_analytical_continuation(ψ, p)
    return analytic_ellipse_loglike(p.pointa + ψ*p.uhat, [p.ind1, p.ind2], p.consistent.data_analytic) - p.targetll
end

function bivariateψ_ellipse_analytical_gradient(ψ::Vector, p)
    return analytic_ellipse_loglike(ψ, [p.ind1, p.ind2], p.consistent.data_analytic) - p.consistent.targetll
end

function bivariateψ_ellipse_unbounded(ψ::Vector, p)
    θs=zeros(p.consistent.num_pars)
    θs[p.ind1], θs[p.ind2] = ψ

    function fun(ω)
        return ellipse_loglike(variablemapping2d!(θs, ω, p.θranges, p.ωranges), p.consistent.data) 
    end

    (xopt,_)=optimise_unbounded(fun, p.initGuess)
    return xopt
end