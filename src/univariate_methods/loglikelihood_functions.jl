# function univariateψ_ellipse_analytical(ψ, p)
#     return analytic_ellipse_loglike([ψ], [p.ind], p.consistent.data) - p.consistent.targetll
# end
"""
    univariateψ_ellipse_unbounded(ψ, p)

Unbounded ellipse approximation of a log-likelihood function, with univariate interest parameter ψ, which finds the optimal values of the  
"""
function univariateψ_ellipse_unbounded(ψ, p)
    θs=zeros(p.consistent.num_pars)

    function fun(ω)
        θs[p.ind] = ψ
        return ellipse_loglike(variablemapping1d!(θs, ω, p.θranges, p.ωranges), p.consistent.data) 
    end

    (xopt,fopt)=optimise_unbounded(fun, p.initGuess)
    llb=fopt-p.consistent.targetll
    p.ω_opt .= xopt
    return llb
end

"""
    univariateψ(ψ, p)



"""
function univariateψ(ψ, p)
    θs=zeros(p.consistent.num_pars)

    function fun(ω)
        θs[p.ind] = ψ
        return p.consistent.loglikefunction(variablemapping1d!(θs, ω, p.θranges, p.ωranges), p.consistent.data) 
    end

    (xopt,fopt)=optimise(fun, p.initGuess, p.newLb, p.newUb)
    llb=fopt-p.consistent.targetll
    p.ω_opt .= xopt
    return llb
end