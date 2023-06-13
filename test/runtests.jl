using PlaceholderLikelihood
using Test
using Distributions, Random
using EllipseSampling

@testset "PlaceholderLikelihood.jl" begin

    # ELLIPSE LIKELIHOOD
    begin
        a, b = 2.0, 1.0
        α = 0.2 * π
        Cx, Cy = 2.0, 2.0
    
        Hw11 = (cos(α)^2 / a^2 + sin(α)^2 / b^2)
        Hw22 = (sin(α)^2 / a^2 + cos(α)^2 / b^2)
        Hw12 = cos(α) * sin(α) * (1 / a^2 - 1 / b^2)
        Hw_norm = [Hw11 Hw12; Hw12 Hw22]
    
        confidence_level = 0.95
        Hw = Hw_norm ./ (0.5 ./ (quantile(Chisq(2), confidence_level) * 0.5))
    
        data = (θmle=[Cx, Cy], Hmle=Hw)
    
        θnames = [:x, :y]
        θG = [2.0, 2.0]
        lb = [0.0, 0.0]
        ub = [4.0, 4.0]
        par_magnitudes = [1, 1]
        
        @testset "ExactPointsOn_EllipseLikelihood" begin
            N = 8
            expected_points = generate_N_equally_spaced_points(N, a, b, α, Cx, Cy, start_point_shift=0.0)
    
            m = initialiseLikelihoodModel(PlaceholderLikelihood.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes)
            getMLE_ellipse_approximation!(m)
        
            # should find exactly the same points as `expected_points`
            bivariate_confidenceprofiles!(m, N, method=RadialMLEMethod(0.0, 1.0), profile_type=LogLikelihood())
            bivariate_confidenceprofiles!(m, N, method=RadialMLEMethod(0.0, 1.0), profile_type=EllipseApprox())
            bivariate_confidenceprofiles!(m, N, method=RadialMLEMethod(0.0, 1.0), profile_type=EllipseApproxAnalytical())
            bivariate_confidenceprofiles!(m, N, method=AnalyticalEllipseMethod(0.0, 1.0), profile_type=EllipseApproxAnalytical())
    
            for i in 1:3
               @test isapprox(m.biv_profiles_dict[i].confidence_boundary, expected_points, atol=1e-14)
            end
        end

        @testset "GetIntervalPoints_EllipseLikelihood" begin
            m1 = initialiseLikelihoodModel(PlaceholderLikelihood.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes)

            N=20

            univariate_confidenceintervals!(m1, [1], num_points_in_interval=N, additional_width=0.2)

            m2 = initialiseLikelihoodModel(PlaceholderLikelihood.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes)

            univariate_confidenceintervals!(m2, [1])
            get_points_in_interval!(m2, N, additional_width=0.2)

            @test isapprox(m1.uni_profiles_dict[1].interval_points.boundary_col_indices, m2.uni_profiles_dict[1].interval_points.boundary_col_indices)

            targetll = PlaceholderLikelihood.get_target_loglikelihood(m1, 0.95, EllipseApprox(), 1)

            for m0 in (m1, m2)
                boundary_col_indices = m0.uni_profiles_dict[1].interval_points.boundary_col_indices
                @test diff(boundary_col_indices .- [0, 1])[1] == N

                @test diff(boundary_col_indices .- [1, 0])[1] == m0.uni_profiles_df[1, :num_points]

                lls = [PlaceholderLikelihood.ellipse_loglike(m0.uni_profiles_dict[1].interval_points.points[:, j], m0.core.data) for j in boundary_col_indices]
                @test isapprox(lls .- targetll, zeros(2), atol=1e-14)

                @test isapprox(m0.uni_profiles_dict[1].interval_points.ll[boundary_col_indices] .- targetll, zeros(2), atol=1e-14)
            end
        end
        
        @testset "BoundaryIsAZero_EllipseLikelihood" begin        
            N = 50
    
            m = initialiseLikelihoodModel(PlaceholderLikelihood.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes)
            getMLE_ellipse_approximation!(m)

            # UNIVARIATE
            for profile_type in [LogLikelihood(), EllipseApprox(), EllipseApproxAnalytical()]
                univariate_confidenceintervals!(m, profile_type=profile_type)
            end

            targetll = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, EllipseApprox(), 1)

            for i in 1:3
                lls = [PlaceholderLikelihood.ellipse_loglike(m.uni_profiles_dict[i].interval_points.points[:, j], m.core.data) for j in 1:2]
                @test isapprox(lls .- targetll, zeros(2), atol=1e-14)

                @test isapprox(m.uni_profiles_dict[i].interval_points.ll .- targetll, zeros(2), atol=1e-14)
            end
    
            # BIVARIATE
            for method in [IterativeBoundaryMethod(4, 2, 2), RadialRandomMethod(3), SimultaneousMethod(), Fix1AxisMethod(), ContinuationMethod(2, 0.1, 0.0)]
                for profile_type in [LogLikelihood(), EllipseApprox(), EllipseApproxAnalytical()]
                    bivariate_confidenceprofiles!(m, N, method=method, profile_type=profile_type)
                end
            end
    
            targetll = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, EllipseApprox(), 2)
    
            for i in 1:15
                lls = [PlaceholderLikelihood.ellipse_loglike(m.biv_profiles_dict[i].confidence_boundary[:,j], m.core.data) for j in 1:N] 
                @test isapprox(lls .- targetll, zeros(N), atol=1e-14)
            end
        end

        function PlaceholderLikelihood.ellipse_loglike(θ::Tuple{T,T}, mleTuple::@NamedTuple{θmle::Vector{T}, Hmle::Matrix{T}}) where {T<:Float64}
            return PlaceholderLikelihood.ellipse_loglike([θ...], mleTuple)
        end

        @testset "ValidDimensionalPoints_EllipseLikelihood" begin
            m = initialiseLikelihoodModel(PlaceholderLikelihood.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes)

            # UNIVARIATE
            dimensional_likelihood_sample!(m, 1, 100, sample_type=UniformGridSamples())
            dimensional_likelihood_sample!(m, 1, 100, sample_type=UniformRandomSamples())
            dimensional_likelihood_sample!(m, 1, 100, sample_type=LatinHypercubeSamples())

            targetll_standardised = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, EllipseApprox(), 1)

            for i in 1:6
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end

            # BIVARIATE / FULL
            dimensional_likelihood_sample!(m, 2, 10, sample_type=UniformGridSamples())
            dimensional_likelihood_sample!(m, 2, 100, sample_type=UniformRandomSamples())
            dimensional_likelihood_sample!(m, 2, 100, sample_type=LatinHypercubeSamples())

            targetll_standardised = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, EllipseApprox(), 2)

            for i in 7:9
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end            
        end
    end
    
    # REAL LIKELIHOOD
    begin
        function solvedmodel(θ, t)
            return (θ[2] * θ[3]) ./ ((θ[2] - θ[3]) .* (exp.(-θ[1] .* t)) .+ θ[3])
        end

        function loglhood(θ, data)
            y=solvedmodel(θ, data.t)
            e=sum(loglikelihood(data.dist, data.yobs .- y))
            return e
        end

        λ, K, C0 = 0.01, 100.0, 10.0
        t = 0:100:1000
        σ = 10.0

        λmin, λmax = 0.00, 0.05
        Kmin, Kmax = 50.0, 150.0
        C0min, C0max = 0.001, 50.0

        θnames = [:λ, :K, :C0]
        θG = [λ, K, C0]
        lb = [λmin, Kmin, C0min]
        ub = [λmax, Kmax, C0max]
        par_magnitudes = [0.005, 10, 10]

        ytrue = solvedmodel(θG, t)
        Random.seed!(12348)
        yobs = ytrue + σ * randn(length(t))
        data = (yobs=yobs, σ=σ, t=t, dist=Normal(0, σ))

        @testset "BoundaryIsAZeroRealLikelihood" begin
            N = 50

            m = initialiseLikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes)
            getMLE_ellipse_approximation!(m)

            # UNIVARIATE
            univariate_confidenceintervals!(m)

            targetll = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, LogLikelihood(), 1)
            targetll_standardised = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, EllipseApprox(), 1)

            for i in 1:3
                lls = [loglhood(m.uni_profiles_dict[i].interval_points.points[:, j], m.core.data) for j in 1:2]
                @test isapprox(lls .- targetll, zeros(2), atol=1e-14)

                @test isapprox(m.uni_profiles_dict[i].interval_points.ll .-  targetll_standardised, zeros(2), atol=1e-14)
            end

            # BIVARIATE
            for method in [IterativeBoundaryMethod(4, 2, 2), RadialRandomMethod(3), RadialMLEMethod(0.0), SimultaneousMethod(), Fix1AxisMethod(), ContinuationMethod(2, 0.1, 0.0)]
                bivariate_confidenceprofiles!(m, N, method=method)
            end

            targetll = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, LogLikelihood(), 2)

            for i in 1:6
                lls = [loglhood(m.biv_profiles_dict[i].confidence_boundary[:,j], m.core.data) for j in 1:N]
                @test isapprox(lls .- targetll, zeros(N), atol=1e-12)
            end
        end

        @testset "ValidDimensionalPoints_RealLikelihood" begin
            m = initialiseLikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes)

            # UNIVARIATE
            dimensional_likelihood_sample!(m, 1, 100, sample_type=UniformGridSamples())
            dimensional_likelihood_sample!(m, 1, 100, sample_type=UniformRandomSamples())
            dimensional_likelihood_sample!(m, 1, 100, sample_type=LatinHypercubeSamples())

            targetll_standardised = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, EllipseApprox(), 1)

            for i in 1:9
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end

            # BIVARIATE
            dimensional_likelihood_sample!(m, 2, 10, sample_type=UniformGridSamples())
            dimensional_likelihood_sample!(m, 2, 100, sample_type=UniformRandomSamples())
            dimensional_likelihood_sample!(m, 2, 100, sample_type=LatinHypercubeSamples())

            targetll_standardised = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, EllipseApprox(), 2)

            for i in 10:18
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end

            # FULL
            dimensional_likelihood_sample!(m, 3, 10, sample_type=UniformGridSamples())
            dimensional_likelihood_sample!(m, 3, 1000, sample_type=UniformRandomSamples())
            dimensional_likelihood_sample!(m, 3, 1000, sample_type=LatinHypercubeSamples())

            targetll_standardised = PlaceholderLikelihood.get_target_loglikelihood(m, 0.95, EllipseApprox(), 3)

            for i in 19:21
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end
        end
    end
end
