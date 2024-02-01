using LikelihoodBasedProfileWiseAnalysis
using Test
using Distributions, Random
using EllipseSampling

@testset "LikelihoodBasedProfileWiseAnalysis.jl" begin

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
    
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub,  par_magnitudes, show_progress=false, 
                find_zero_atol=0.0)
            getMLE_ellipse_approximation!(m)
        
            # should find exactly the same points as `expected_points`
            bivariate_confidenceprofiles!(m, N, method=RadialMLEMethod(0.0, 1.0), profile_type=LogLikelihood())
            bivariate_confidenceprofiles!(m, N, method=RadialMLEMethod(0.0, 1.0), profile_type=EllipseApprox())
            bivariate_confidenceprofiles!(m, N, method=RadialMLEMethod(0.0, 1.0), profile_type=EllipseApproxAnalytical())
            bivariate_confidenceprofiles!(m, N, method=AnalyticalEllipseMethod(0.0, 1.0), profile_type=EllipseApproxAnalytical())
    
            for i in 1:4
               @test isapprox(m.biv_profiles_dict[i].confidence_boundary, expected_points, atol=1e-14)
            end
        end

        @testset "GetIntervalPoints_EllipseLikelihood" begin
            m1 = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false, 
                find_zero_atol=0.0)

            N=20
            univariate_confidenceintervals!(m1, [1], num_points_in_interval=N, additional_width=0.2)

            m2 = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, 
                find_zero_atol=0.0)

            univariate_confidenceintervals!(m2, [1])
            get_points_in_intervals!(m2, N, additional_width=0.2)

            @test isapprox(m1.uni_profiles_dict[1].interval_points.boundary_col_indices, m2.uni_profiles_dict[1].interval_points.boundary_col_indices)

            targetll = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m1, 0.95, EllipseApprox(), 1)

            for m0 in (m1, m2)
                boundary_col_indices = m0.uni_profiles_dict[1].interval_points.boundary_col_indices
                @test diff(boundary_col_indices .- [0, 1])[1] == N

                @test diff(boundary_col_indices .- [1, 0])[1] == m0.uni_profiles_df[1, :num_points]

                lls = [LikelihoodBasedProfileWiseAnalysis.ellipse_loglike(m0.uni_profiles_dict[1].interval_points.points[:, j], m0.core.data) for j in boundary_col_indices]
                @test isapprox(lls .- targetll, zeros(2), atol=1e-14)

                @test isapprox(m0.uni_profiles_dict[1].interval_points.ll[boundary_col_indices] .- targetll, zeros(2), atol=1e-14)
            end
        end
        
        @testset "BoundaryIsAZero_EllipseLikelihood" begin        
            N = 50
    
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false, 
                find_zero_atol=0.0)
            getMLE_ellipse_approximation!(m)

            # UNIVARIATE
            for profile_type in [LogLikelihood(), EllipseApprox(), EllipseApproxAnalytical()]
                univariate_confidenceintervals!(m, profile_type=profile_type)
            end

            targetll = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 1)

            for i in 1:6
                lls = [LikelihoodBasedProfileWiseAnalysis.ellipse_loglike(m.uni_profiles_dict[i].interval_points.points[:, j], m.core.data) for j in 1:2]
                @test isapprox(lls .- targetll, zeros(2), atol=1e-14)

                @test isapprox(m.uni_profiles_dict[i].interval_points.ll .- targetll, zeros(2), atol=1e-14)
            end
    
            # BIVARIATE
            for method in [IterativeBoundaryMethod(4, 2, 2), RadialRandomMethod(3), SimultaneousMethod(), Fix1AxisMethod(), ContinuationMethod(2, 0.1, 0.0)]
                for profile_type in [LogLikelihood(), EllipseApprox(), EllipseApproxAnalytical()]
                    bivariate_confidenceprofiles!(m, N, method=method, profile_type=profile_type)
                end
            end
    
            targetll = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 2)
    
            for i in 1:15
                lls = [LikelihoodBasedProfileWiseAnalysis.ellipse_loglike(m.biv_profiles_dict[i].confidence_boundary[:,j], m.core.data) for j in 1:N] 
                @test isapprox(lls .- targetll, zeros(N), atol=1e-12)
            end
        end

        function LikelihoodBasedProfileWiseAnalysis.ellipse_loglike(θ::Tuple{T,T}, mleTuple::@NamedTuple{θmle::Vector{T}, Hmle::Matrix{T}}) where {T<:Float64}
            return LikelihoodBasedProfileWiseAnalysis.ellipse_loglike([θ...], mleTuple)
        end

        @testset "ValidDimensionalPoints_EllipseLikelihood" begin
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false)

            # UNIVARIATE
            dimensional_likelihood_samples!(m, 1, 100, sample_type=UniformGridSamples())
            dimensional_likelihood_samples!(m, 1, 100, sample_type=UniformRandomSamples())
            dimensional_likelihood_samples!(m, 1, 100, sample_type=LatinHypercubeSamples())

            targetll_standardised = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 1)

            for i in 1:6
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end

            # BIVARIATE / FULL
            dimensional_likelihood_samples!(m, 2, 10, sample_type=UniformGridSamples())
            dimensional_likelihood_samples!(m, 2, 100, sample_type=UniformRandomSamples())
            dimensional_likelihood_samples!(m, 2, 100, sample_type=LatinHypercubeSamples())

            targetll_standardised = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 2)

            for i in 7:9
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end            
        end

        @testset "BoundaryIsAZeroMethodExtensions_EllipseLikelihood" begin        
            N = 50
    
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false, 
                find_zero_atol=0.0)

            # UNIVARIATE
            targetll = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 1)
            univariate_confidenceintervals!(m, [:x])

            for i in 1:1
                lls = [LikelihoodBasedProfileWiseAnalysis.ellipse_loglike(m.uni_profiles_dict[i].interval_points.points[:, j], m.core.data) for j in 1:2]
                @test isapprox(lls .- targetll, zeros(2), atol=1e-14)
                @test isapprox(m.uni_profiles_dict[i].interval_points.ll .- targetll, zeros(2), atol=1e-14)
                @test m.num_uni_profiles == 1
            end

            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false, 
                find_zero_atol=0.0)
            univariate_confidenceintervals!(m, 1)

            for i in 1:1
                lls = [LikelihoodBasedProfileWiseAnalysis.ellipse_loglike(m.uni_profiles_dict[i].interval_points.points[:, j], m.core.data) for j in 1:2]
                @test isapprox(lls .- targetll, zeros(2), atol=1e-14)
                @test isapprox(m.uni_profiles_dict[i].interval_points.ll .- targetll, zeros(2), atol=1e-14)
                @test m.num_uni_profiles == 1
            end
    
            # BIVARIATE
            targetll = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 2)
            bivariate_confidenceprofiles!(m, [[:x, :y]], N)
    
            for i in 1:1
                lls = [LikelihoodBasedProfileWiseAnalysis.ellipse_loglike(m.biv_profiles_dict[i].confidence_boundary[:,j], m.core.data) for j in 1:N] 
                @test isapprox(lls .- targetll, zeros(N), atol=1e-14)
                @test m.num_biv_profiles == 1
            end

            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false, 
                find_zero_atol=0.0)
            bivariate_confidenceprofiles!(m, 2, N) # will get set to 1

            for i in 1:1
                lls = [LikelihoodBasedProfileWiseAnalysis.ellipse_loglike(m.biv_profiles_dict[i].confidence_boundary[:, j], m.core.data) for j in 1:N]
                @test isapprox(lls .- targetll, zeros(N), atol=1e-14)
                @test m.num_biv_profiles == 1
            end
        end

        @testset "UseExistingProfilesBehaviour_EllipseLikelihood" begin
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false)

            setbounds!(m, lb=[-10000000.0, -10000000.0], ub=[10000000.0, 10000000.0])
            dof=1

            univariate_confidenceintervals!(m, [:x], confidence_level=0.90, dof=dof)
            lb1, ub1 = LikelihoodBasedProfileWiseAnalysis.get_interval_brackets(m, 1, 0.9, dof, LogLikelihood())
            @test isempty(lb1) && isempty(ub1)

            t1 = @elapsed univariate_confidenceintervals!(m, [:x], confidence_level=0.90, dof=dof, existing_profiles=:overwrite)

            univariate_confidenceintervals!(m, [:x], confidence_level=0.95, dof=dof)
            lb2, ub2 = LikelihoodBasedProfileWiseAnalysis.get_interval_brackets(m, 1, 0.9, dof, LogLikelihood())
            @test isapprox(lb2[2], m.core.θmle[1]) && isapprox(ub2[1], m.core.θmle[2])
            @test lb2[1] > m.core.θlb[1] && ub2[2] < m.core.θub[2]

            univariate_confidenceintervals!(m, [:x], confidence_level=0.70, dof=dof)
            lb3, ub3 = LikelihoodBasedProfileWiseAnalysis.get_interval_brackets(m, 1, 0.9, dof, LogLikelihood())
            @test lb3[1] < m.core.θmle[1] && ub3[1] > m.core.θmle[2]

            univariate_confidenceintervals!(m, [:x], confidence_level=0.90, dof=dof, use_existing_profiles=true, existing_profiles=:overwrite)
            t2 = @elapsed univariate_confidenceintervals!(m, [:x], confidence_level=0.90, dof=dof, use_existing_profiles=true, existing_profiles=:overwrite)
            # @test t2 < t1
        end

        @testset "ExistingProfilesBehaviour_EllipseLikelihood" begin 
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false)

            # UNIVARIATE
            univariate_confidenceintervals!(m, [:x])
            get_points_in_intervals!(m, 10; confidence_levels=[0.95], profile_types=[LogLikelihood()])

            @test m.uni_profiles_df[1, :num_points] == 12

            univariate_confidenceintervals!(m, [:x], existing_profiles=:ignore)
            @test m.uni_profiles_df[1, :num_points] == 12
            @test length(m.uni_profiles_dict[1].interval_points.ll) == 12
            
            univariate_confidenceintervals!(m, [:x], existing_profiles=:overwrite)
            @test m.uni_profiles_df[1, :num_points] == 2
            @test length(m.uni_profiles_dict[1].interval_points.ll) == 2

            # BIVARIATE
            bivariate_confidenceprofiles!(m, 10)
            bivariate_confidenceprofiles!(m, 20, existing_profiles=:ignore)
            @test m.biv_profiles_df[1, :num_points] == 10
            @test size(m.biv_profiles_dict[1].confidence_boundary, 2) == 10

            existing_points = m.biv_profiles_dict[1].confidence_boundary .* 1.0

            bivariate_confidenceprofiles!(m, 20, existing_profiles=:merge)
            @test m.biv_profiles_df[1, :num_points] == 20
            @test size(m.biv_profiles_dict[1].confidence_boundary, 2) == 20
            @test isapprox(m.biv_profiles_dict[1].confidence_boundary[:, 1:10], existing_points)

            existing_points = m.biv_profiles_dict[1].confidence_boundary .* 1.0

            bivariate_confidenceprofiles!(m, 20, existing_profiles=:overwrite)
            @test m.biv_profiles_df[1, :num_points] == 20
            @test size(m.biv_profiles_dict[1].confidence_boundary, 2) == 20
            @test !isapprox(m.biv_profiles_dict[1].confidence_boundary, existing_points)

            # DIMENSIONAL
            dimensional_likelihood_samples!(m, 2, 1000)
            existing_points = m.dim_samples_dict[1].points[:,1] .* 1.0
            num_points = m.dim_samples_df[1, :num_points] .* 1

            dimensional_likelihood_samples!(m, 2, 1000, existing_profiles=:ignore)
            @test m.dim_samples_df[1, :num_points] == num_points
            @test isapprox(m.dim_samples_dict[1].points[:, 1], existing_points)

            dimensional_likelihood_samples!(m, 2, 100, existing_profiles=:overwrite)
            @test m.dim_samples_df[1, :num_points] != num_points
            @test (m.dim_samples_df[1, :num_points] == 0) || !isapprox(m.dim_samples_dict[1].points[:, 1], existing_points)
        end

        @testset "SetMagnitudes_EllipseLikelihood" begin
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes)

            @test isapprox(m.core.θmagnitudes, par_magnitudes)

            new_par_magnitudes = [20.0, 20.0]
            setmagnitudes!(m, [20.0, 20.0])
            @test isapprox(m.core.θmagnitudes, new_par_magnitudes)
        end
        
        @testset "SetBounds_EllipseLikelihood" begin
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes)

            setbounds!(m, lb=[-100.0, -50.0], ub=[2.0, 4.0])

            @test isapprox(m.core.θlb, [-100.0, -50.0])
            @test isapprox(m.core.θub, [2.0, 4.0])
        end

        @testset "combine_bivariate_boundaries_EllipseLikelihood" begin
            function predict_func(θ, data, t=[1.5]); return sum(θ) .* t end # exact output is not important here
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, predict_func, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false)

            bivariate_confidenceprofiles!(m, 10)
            bivariate_confidenceprofiles!(m, 10, method=SimultaneousMethod())
            combine_bivariate_boundaries!(m)
            @test m.num_biv_profiles == 1
            @test length(m.biv_profiles_df.row_ind) == 1
            @test length(keys(m.biv_profiles_dict)) == 1
            @test haskey(m.biv_profiles_dict, 1) && size(m.biv_profiles_dict[1].confidence_boundary, 2) == 20

            bivariate_confidenceprofiles!(m, 10, method=SimultaneousMethod())
            @test length(m.biv_profiles_df.row_ind) == 2
            combine_bivariate_boundaries!(m)
            @test length(m.biv_profiles_df.row_ind) == 1
            @test m.biv_profiles_df[1, :num_points] == 30
            @test m.biv_profiles_df[1, :row_ind] == 1

            generate_predictions_bivariate!(m, [1.2], 1.0)
            bivariate_confidenceprofiles!(m, 10, method=SimultaneousMethod())
            combine_bivariate_boundaries!(m, not_evaluated_predictions=true)
            @test length(m.biv_profiles_df.row_ind) == 2
            combine_bivariate_boundaries!(m, not_evaluated_predictions=false)
            @test length(m.biv_profiles_df.row_ind) == 2

            generate_predictions_bivariate!(m, [1.2, 2.4], 1.0)
            combine_bivariate_boundaries!(m, not_evaluated_predictions=false)
            @test length(m.biv_profiles_df.row_ind) == 2
            generate_predictions_bivariate!(m, [1.2], 1.0, overwrite_predictions=true, methods=[SimultaneousMethod()])
            combine_bivariate_boundaries!(m, not_evaluated_predictions=false)
            @test length(m.biv_profiles_df.row_ind) == 1
        end

        @testset "error_handling_EllipseLikelihood" begin
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false)

            @test_throws DomainError   univariate_confidenceintervals!(m, 0)
            @test_throws DomainError   univariate_confidenceintervals!(m, confidence_level=-0.1)
            @test_throws DomainError   univariate_confidenceintervals!(m, confidence_level=1.0)
            @test_throws DomainError   univariate_confidenceintervals!(m, find_zero_atol=-0.1)
            @test_throws DomainError   univariate_confidenceintervals!(m, num_points_in_interval=-1)
            @test_throws DomainError   univariate_confidenceintervals!(m, additional_width=-1.0)
            @test_throws DomainError   univariate_confidenceintervals!(m, [1,4,2,3])
            @test_throws ArgumentError univariate_confidenceintervals!(m, existing_profiles=:merge)
            @test_throws ArgumentError univariate_confidenceintervals!(m, θlb_nuisance=[1.])
            @test_throws ArgumentError univariate_confidenceintervals!(m, θlb_nuisance=[1.])
            @test_throws DomainError   univariate_confidenceintervals!(m, θlb_nuisance=m.core.θmle .+ 1.0)
            @test_throws DomainError   univariate_confidenceintervals!(m, θub_nuisance=m.core.θmle .- 1.0)

            @test_throws DomainError   get_points_in_intervals!(m, 0)
            @test_throws DomainError   get_points_in_intervals!(m, 1, additional_width=-1.0)

            @test_throws DomainError   bivariate_confidenceprofiles!(m, 0, 10)
            @test_throws DomainError   bivariate_confidenceprofiles!(m, 10, confidence_level=-0.1)
            @test_throws DomainError   bivariate_confidenceprofiles!(m, 10, confidence_level=1.0)
            @test_throws DomainError   bivariate_confidenceprofiles!(m, 10, find_zero_atol=-0.1)
            @test_throws DomainError   bivariate_confidenceprofiles!(m, [[1,2],[2,4],[5,1]], 10)
            @test_throws ArgumentError bivariate_confidenceprofiles!(m, [[1,2], [1]], 10)
            @test_throws ArgumentError bivariate_confidenceprofiles!(m, 10, existing_profiles=:something)
            @test_throws ArgumentError bivariate_confidenceprofiles!(m, 10, method=CombinedBivariateMethod())
            @test_throws ArgumentError bivariate_confidenceprofiles!(m, 10, θlb_nuisance=[1.0])
            @test_throws ArgumentError bivariate_confidenceprofiles!(m, 10, θlb_nuisance=[1.0])
            @test_throws DomainError   bivariate_confidenceprofiles!(m, 10, θlb_nuisance=m.core.θmle .+ 1.0)
            @test_throws DomainError   bivariate_confidenceprofiles!(m, 10, θub_nuisance=m.core.θmle .- 1.0)

            @test_throws DomainError   sample_bivariate_internal_points!(m, 0)

            @test_throws DomainError   dimensional_likelihood_samples!(m, 1, 0, 10)
            @test_throws DomainError   dimensional_likelihood_samples!(m, 1, 10, confidence_level=-0.1)
            @test_throws DomainError   dimensional_likelihood_samples!(m, 1, 10, confidence_level=1.0)
            @test_throws DomainError   dimensional_likelihood_samples!(m, 1, 0)
            @test_throws DomainError   dimensional_likelihood_samples!(m, 2, [0, 1], sample_type=UniformGridSamples())
            @test_throws DomainError   dimensional_likelihood_samples!(m, [[1,2],[2,4],[5,1]], 10)
            @test_throws ArgumentError dimensional_likelihood_samples!(m, 2, [1, 1], sample_type=LatinHypercubeSamples())
            @test_throws ArgumentError dimensional_likelihood_samples!(m, [[1],[1,2]], [1,1], sample_type=UniformGridSamples())
            @test_throws ArgumentError dimensional_likelihood_samples!(m, 1, 10, existing_profiles=:merge)
            @test_throws ArgumentError dimensional_likelihood_samples!(m, 1, 10, lb=[1,1], ub=[2])
            @test_throws ArgumentError dimensional_likelihood_samples!(m, 1, 10, lb=[1], ub=[2, 2])
            @test_throws ArgumentError dimensional_likelihood_samples!(m, 1, 10, θlb_nuisance=[1.0])
            @test_throws ArgumentError dimensional_likelihood_samples!(m, 1, 10, θlb_nuisance=[1.0])
            @test_throws DomainError   dimensional_likelihood_samples!(m, 1, 10, θlb_nuisance=m.core.θmle .+ 1.0)
            @test_throws DomainError   dimensional_likelihood_samples!(m, 1, 10, θub_nuisance=m.core.θmle .- 1.0)
            
            @test_throws DomainError   full_likelihood_sample!(m, 0)
            @test_throws DomainError   full_likelihood_sample!(m, 10, confidence_level=-0.1)
            @test_throws DomainError   full_likelihood_sample!(m, 10, confidence_level=1.0)
            @test_throws DomainError   full_likelihood_sample!(m, [ 0, 10], sample_type=UniformGridSamples())
            @test_throws ArgumentError full_likelihood_sample!(m, [10, 10], sample_type=LatinHypercubeSamples())
            @test_throws ArgumentError full_likelihood_sample!(m, 10, existing_profiles=:merge)
            @test_throws ArgumentError full_likelihood_sample!(m, 10, lb=[1,1], ub=[2])
            @test_throws ArgumentError full_likelihood_sample!(m, 10, lb=[1], ub=[2,2])

            function predict_func(θ, data, t=[1.5]); return sum(θ) .* t end # exact output is not important here
            m = initialise_LikelihoodModel(LikelihoodBasedProfileWiseAnalysis.ellipse_loglike, predict_func, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false)
            
            @test_throws DomainError generate_predictions_univariate!(m,  [1, 2], -0.1)
            @test_throws DomainError generate_predictions_bivariate!(m,   [1, 2], -0.1)
            @test_throws DomainError generate_predictions_dim_samples!(m, [1, 2], -0.1)

            function data_gen(θtrue, generator_args); return sum(θtrue) end
            @test_throws ArgumentError check_univariate_parameter_coverage(data_gen, data, m, 10, [2.0], [1], [1.0, 2.0])
            @test_throws ArgumentError check_univariate_parameter_coverage(data_gen, data, m, 10, [1.0, 2.0], [1], [2.0])
            @test_throws DomainError   check_univariate_parameter_coverage(data_gen, data, m, 10, [1.0, 2.0], [1]; coverage_estimate_confidence_level=0.0)
            @test_throws DomainError   check_univariate_parameter_coverage(data_gen, data, m, 10, [1.0, 2.0], [1]; coverage_estimate_confidence_level=1.0)
            @test_throws DomainError   check_univariate_parameter_coverage(data_gen, data, m, 10, [1.0, 2.0], [1]; confidence_level=-0.1)
            @test_throws DomainError   check_univariate_parameter_coverage(data_gen, data, m, 10, [1.0, 2.0], [1]; confidence_level=1.0)
            @test_throws DomainError   check_univariate_parameter_coverage(data_gen, data, m, 0, [1.0, 2.0], [1])
            @test_throws DomainError   check_univariate_parameter_coverage(data_gen, data, m, 0, [1.0, 2.0], [1, 3])

            @test_throws ArgumentError check_bivariate_parameter_coverage(data_gen, data, m, 10, 3, [2.0], [[1, 2]])
            @test_throws ArgumentError check_bivariate_parameter_coverage(data_gen, data, m, 10, 3, [1.0, 2.0], [[1, 2]], [2.0])
            @test_throws DomainError   check_bivariate_parameter_coverage(data_gen, data, m, 10, 3, [1.0, 2.0], [[1, 2]]; coverage_estimate_confidence_level=0.0)
            @test_throws DomainError   check_bivariate_parameter_coverage(data_gen, data, m, 10, 3, [1.0, 2.0], [[1, 2]]; coverage_estimate_confidence_level=1.0)
            @test_throws DomainError   check_bivariate_parameter_coverage(data_gen, data, m, 10, 3, [1.0, 2.0], [[1, 2]]; confidence_level=-0.1)
            @test_throws DomainError   check_bivariate_parameter_coverage(data_gen, data, m, 10, 3, [1.0, 2.0], [[1, 2]]; confidence_level=1.0)
            @test_throws ArgumentError check_bivariate_parameter_coverage(data_gen, data, m, 10, [3,3], [1.0, 2.0], [[1, 2]])
            @test_throws ArgumentError check_bivariate_parameter_coverage(data_gen, data, m, 10, 3, [1.0, 2.0], [[1, 2]]; method=[RadialRandomMethod(3), RadialMLEMethod()])
            @test_throws ArgumentError check_bivariate_parameter_coverage(data_gen, data, m, 10, [3], [1.0, 2.0], [[1, 2]]; method=[RadialRandomMethod(3), RadialMLEMethod()])
            @test_throws DomainError   check_bivariate_parameter_coverage(data_gen, data, m, 0, 3, [1.0, 2.0], [[1, 2]])
            @test_throws DomainError   check_bivariate_parameter_coverage(data_gen, data, m, 10, 3, [1.0, 2.0], [[1, 3]])

            @test_throws ArgumentError check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, 10, [2.0], [[1, 2]])
            @test_throws ArgumentError check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, 10, [1.0, 2.0], [[1, 2]], [2.0])
            @test_throws DomainError   check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, 10, [1.0, 2.0], [[1, 2]]; coverage_estimate_quantile_level=0.0)
            @test_throws DomainError   check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, 10, [1.0, 2.0], [[1, 2]]; coverage_estimate_quantile_level=1.0)
            @test_throws DomainError   check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, 10, [1.0, 2.0], [[1, 2]]; confidence_level=-0.1)
            @test_throws DomainError   check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, 10, [1.0, 2.0], [[1, 2]]; confidence_level=1.0)
            @test_throws ArgumentError check_bivariate_boundary_coverage(data_gen, data, m, 10, [3,3], 10, [1.0, 2.0], [[1, 2]])
            @test_throws ArgumentError check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, 10, [1.0, 2.0], [[1, 2]]; method=[RadialRandomMethod(3), RadialMLEMethod()])
            @test_throws ArgumentError check_bivariate_boundary_coverage(data_gen, data, m, 10, [3], 10, [1.0, 2.0], [[1, 2]]; method=[RadialRandomMethod(3), RadialMLEMethod()])
            @test_throws DomainError   check_bivariate_boundary_coverage(data_gen, data, m, 0, 3, 10, [1.0, 2.0], [[1, 2]])
            @test_throws DomainError   check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, 10, [1.0, 2.0], [[1, 3]])
            @test_throws DomainError   check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, 0, [1.0, 2.0], [[1, 2]])
            @test_throws DomainError   check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, [0, 10], [1.0, 2.0], [[1, 2]])
            @test_throws ArgumentError check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, [10, 10], [1.0, 2.0], [[1, 2]], sample_type=LatinHypercubeSamples())
            @test_throws ArgumentError check_bivariate_boundary_coverage(data_gen, data, m, 10, 3, [10], [1.0, 2.0], [[1, 2]], sample_type=UniformGridSamples())

            LikelihoodBasedProfileWiseAnalysis.TimerOutputs.enable_debug_timings(LikelihoodBasedProfileWiseAnalysis)
            @test_throws ArgumentError univariate_confidenceintervals!(m; use_distributed=false, use_threads=true)
            @test_throws ArgumentError get_points_in_intervals!(m, 1; use_threads=true)
            @test_throws ArgumentError bivariate_confidenceprofiles!(m, [[1, 2]], 10; use_distributed=false, use_threads=true)
            @test_throws ArgumentError sample_bivariate_internal_points!(m, 10; use_distributed=false, use_threads=true)
            @test_throws ArgumentError dimensional_likelihood_samples!(m, 1, 10; use_distributed=false, use_threads=true)
            @test_throws ArgumentError full_likelihood_sample!(m, 10; use_distributed=false, use_threads=true)
            LikelihoodBasedProfileWiseAnalysis.TimerOutputs.disable_debug_timings(LikelihoodBasedProfileWiseAnalysis)

            univariate_confidenceintervals!(m)
            @test_throws ArgumentError get_points_in_intervals!(m, 1, θlb_nuisance=[1.0])
            @test_throws ArgumentError get_points_in_intervals!(m, 1, θlb_nuisance=[1.0])
            @test_throws DomainError   get_points_in_intervals!(m, 1, θlb_nuisance=m.core.θmle .+ 1.0)
            @test_throws DomainError   get_points_in_intervals!(m, 1, θub_nuisance=m.core.θmle .- 1.0)

            @test remove_functions_from_core!(m) isa CoreLikelihoodModel
            @test_throws ArgumentError univariate_confidenceintervals!(m)
            @test_throws ArgumentError get_points_in_intervals!(m, 1)
            @test_throws ArgumentError bivariate_confidenceprofiles!(m, 10)
            @test_throws ArgumentError sample_bivariate_internal_points!(m, 1)
            @test_throws ArgumentError dimensional_likelihood_samples!(m, 1, 10)
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

            m = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false, 
                find_zero_atol=0.0)
            getMLE_ellipse_approximation!(m)

            # UNIVARIATE
            univariate_confidenceintervals!(m)

            targetll = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, LogLikelihood(), 1)
            targetll_standardised = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 1)

            for i in 1:3
                lls = [loglhood(m.uni_profiles_dict[i].interval_points.points[:, j], m.core.data) for j in 1:2]
                @test isapprox(lls .- targetll, zeros(2), atol=1e-14)

                @test isapprox(m.uni_profiles_dict[i].interval_points.ll .-  targetll_standardised, zeros(2), atol=1e-14)
            end

            # BIVARIATE
            for method in [IterativeBoundaryMethod(4, 2, 2), RadialRandomMethod(3), RadialMLEMethod(0.0), SimultaneousMethod(), Fix1AxisMethod(), ContinuationMethod(2, 0.1, 0.0)]
                bivariate_confidenceprofiles!(m, N, method=method)
            end

            targetll = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, LogLikelihood(), 2)

            for i in 1:6
                lls = [loglhood(m.biv_profiles_dict[i].confidence_boundary[:,j], m.core.data) for j in 1:N]
                @test isapprox(lls .- targetll, zeros(N), atol=1e-12)
            end
        end

        @testset "ValidDimensionalPoints_RealLikelihood" begin
            m = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, show_progress=false)

            # UNIVARIATE
            dimensional_likelihood_samples!(m, 1, 100, sample_type=UniformGridSamples())
            dimensional_likelihood_samples!(m, 1, 100, sample_type=UniformRandomSamples())
            dimensional_likelihood_samples!(m, 1, 100, sample_type=LatinHypercubeSamples())

            targetll_standardised = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 1)

            for i in 1:9
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end

            # BIVARIATE
            dimensional_likelihood_samples!(m, 2, 10, sample_type=UniformGridSamples())
            dimensional_likelihood_samples!(m, 2, 100, sample_type=UniformRandomSamples())
            dimensional_likelihood_samples!(m, 2, 100, sample_type=LatinHypercubeSamples())

            targetll_standardised = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 2)

            for i in 10:18
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end

            # FULL
            dimensional_likelihood_samples!(m, 3, 10, sample_type=UniformGridSamples())
            dimensional_likelihood_samples!(m, 3, 1000, sample_type=UniformRandomSamples())
            dimensional_likelihood_samples!(m, 3, 1000, sample_type=LatinHypercubeSamples())

            targetll_standardised = LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood(m, 0.95, EllipseApprox(), 3)

            for i in 19:21
                @test all(m.dim_samples_dict[i].ll .≥ targetll_standardised)
            end
        end
    end
end
