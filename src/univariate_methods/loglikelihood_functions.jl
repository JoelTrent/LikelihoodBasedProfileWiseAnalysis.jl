# function univariateψ_ellipse_analytical(ψ, p)
#     return analytic_ellipse_loglike([ψ], [p.ind], p.consistent.data) - p.consistent.targetll
# end
"""
    univariateψ_ellipse_unbounded(ψ::Real, p::NamedTuple)

Given an ellipse approximation of a log-likelihood function ([`LikelihoodBasedProfileWiseAnalysis.ellipse_loglike`](@ref)) which is unbounded in parameter space, this function finds the values of the nuisance parameters ω that optimise the function at fixed values of the interest parameter ψ and returns the approximated log-likelihood value minus the confidence interval target threshold. The returned function value will be zero at the locations of the approximate confidence interval for ψ, which correspond to the locations found by [`LikelihoodBasedProfileWiseAnalysis.analytic_ellipse_loglike_1D_soln`](@ref). Nuisance parameter values are stored in the NamedTuple `p`. 
"""
function univariateψ_ellipse_unbounded(ψ::Real, p::NamedTuple)

    function fun(ω, q)
        θs = zeros(eltype(ω), q.consistent.num_pars)
        θs[q.ind] = ψ
        θs[q.θranges[1]] .= ω[q.ωranges[1]]
        θs[q.θranges[2]] .= ω[q.ωranges[2]]
        @timeit_debug timer "Likelihood evaluation" begin
            return -ellipse_loglike(θs, q.consistent.data)
        end
    end

    @timeit_debug timer "Likelihood nuisance parameter optimisation" begin
        (xopt,fopt)=optimise_unbounded(fun, p.q, p.options, p.q.initGuess)
    end
    llb=fopt-p.q.consistent.targetll
    p.ω_opt .= xopt
    return llb
end

"""
    univariateψ(ψ::Real, p::NamedTuple)

Given a log-likelihood function (`p.consistent.loglikefunction`) which is bounded in parameter space and may be an ellipse approximation, this function finds the values of the nuisance parameters ω that optimise the function fixed values of the interest parameter ψ and returns the log-likelihood value minus the confidence interval target threshold. The returned function value will be zero at the locations of the approximate confidence interval for ψ. Nuisance parameter values are stored in the NamedTuple `p` at `p.ω_opt`. 
"""
function univariateψ(ψ::Real, p::NamedTuple)
    
    function fun(ω, q)
        θs = zeros(eltype(ω), q.consistent.num_pars)
        θs[q.ind] = ψ
        θs[q.θranges[1]] .= ω[q.ωranges[1]]
        θs[q.θranges[2]] .= ω[q.ωranges[2]]
        @timeit_debug timer "Likelihood evaluation" begin
            return -q.consistent.loglikefunction(θs, q.consistent.data)
        end
    end

    @timeit_debug timer "Likelihood nuisance parameter optimisation" begin
        (xopt,fopt)=optimise(fun, p.q, p.options, p.q.initGuess, p.q.newLb, p.q.newUb)
    end
    llb=fopt-p.q.consistent.targetll
    p.ω_opt .= xopt
    return llb
end