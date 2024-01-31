##############################################################################
######################### ONE SPECIES LOGISTIC MODEL #########################
########### ORIGINAL FROM https://github.com/ProfMJSimpson/Workflow ##########
##############################################################################

# Initial Setup
##############################################################################
using Distributed
if nprocs()==1; addprocs(3, env=["JULIA_NUM_THREADS"=>"1"]) end
@everywhere using Random, Distributions
@everywhere using PlaceholderLikelihood
using Combinatorics

## Model and Likelihood Function Definition
##############################################################################
@everywhere function solvedmodel(t, θ)
    return (θ[2]*θ[3]) ./ ((θ[2]-θ[3]) .* (exp.(-θ[1] .* t)) .+ θ[3])
end

@everywhere function loglhood(θ, data)
    y=solvedmodel(data.t, θ)
    e=sum(loglikelihood(data.dist, data.y_obs .- y))
    return e
end

## Initial Data and Parameter Definition
##############################################################################
# true parameters
λ_true=0.01; K_true=100.0; C0_true=10.0; t=0:100:1000; 
@everywhere global σ=10.0;
θ_true=[λ_true, K_true, C0_true]
y_true = solvedmodel(t, θ_true)
y_obs = [19.27, 20.14, 37.23, 74.87, 88.51, 82.91, 123.88, 103.25, 78.89, 87.87, 113.0]

# Named tuple of all data required within the log-likelihood function
data = (y_obs=y_obs, t=t, dist=Normal(0, σ))

# Bounds on model parameters 
λ_min, λ_max = (0.00, 0.05)
K_min, K_max = (50., 150.)
C0_min, C0_max = (0.0, 50.)
lb = [λ_min, K_min, C0_min]
ub = [λ_max, K_max, C0_max]

θnames = [:λ, :K, :C0]
θG = θ_true
par_magnitudes = [0.005, 10, 10]

## LikelihoodModel Initialisation
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

## Full Parameter Vector Confidence Set Evaluation
##############################################################################
full_likelihood_sample!(model, 30000, use_distributed=true)

## Profiling
##############################################################################

### Univariate Profiles
##############################################################################
univariate_confidenceintervals!(model)
univariate_confidenceintervals!(model, confidence_level=0.99)
univariate_confidenceintervals!(model, dof=model.core.num_pars) # model.core.num_pars=3
univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
univariate_confidenceintervals!(model, profile_type=EllipseApprox())
get_points_in_intervals!(model, 20, additional_width=0.2)

#### Initial Guesses
##############################################################################
univariate_confidenceintervals!(model, confidence_level=0.99)
univariate_confidenceintervals!(model, confidence_level=0.95, use_existing_profiles=true, 
    existing_profiles=:overwrite, num_points_in_interval=20, additional_width=0.2)

univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
univariate_confidenceintervals!(model, use_ellipse_approx_analytical_start=true, 
    existing_profiles=:overwrite, num_points_in_interval=20, additional_width=0.2)

### Bivariate Profiles
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
bivariate_confidenceprofiles!(model, 50, 
    method=IterativeBoundaryMethod(20, 5, 5, 0.15, 1.0, use_ellipse=true), 
    optimizationsettings=opt_settings)

bivariate_confidenceprofiles!(model, 50, 
    method=IterativeBoundaryMethod(20, 5, 5, 0.15, 1.0, use_ellipse=true), 
    dof=model.core.num_pars,
    optimizationsettings=opt_settings)

bivariate_confidenceprofiles!(model, 50, 
    profile_type=EllipseApproxAnalytical(), method=AnalyticalEllipseMethod(0.15, 1.0))

sample_bivariate_internal_points!(model, 100)

### Plots of Profiles
##############################################################################
using Plots, Plots.PlotMeasures; gr()
Plots.reset_defaults(); Plots.scalefontsizes(0.75)

plts = plot_univariate_profiles_comparison(model, 0.1, 0.1, confidence_levels=[0.95], dofs=[1])
plt = plot(plts..., layout=(1,3),
    legend=:outertop, title="", dpi=150, size=(550,300), margin=1mm)
display(plt)
# savefig(plt, joinpath("docs", "src", "assets", "figures", "logistic", "logistic_univariate_plots.png"))

plts = plot_bivariate_profiles_comparison(model, 0.1, 0.1, confidence_levels=[0.95], dofs=[2])
plt = plot(plts..., layout=(1,3),
    legend=:outertop, title="", dpi=150, size=(550,300), margin=1mm)
display(plt)
# savefig(plt, joinpath("docs", "src", "assets", "figures", "logistic", "logistic_bivariate_plots.png"))

## Predictions
##############################################################################
@everywhere function predictfunction(θ, data, t=data.t); solvedmodel(t, θ) end
@everywhere function errorfunction(predictions, θ, region); normal_error_σ_known(predictions, θ, region, σ) end
add_prediction_function!(model, predictfunction)
add_error_function!(model, errorfunction)

t_pred=0:5:1000
generate_predictions_univariate!(model, t_pred)
generate_predictions_bivariate!(model, t_pred)
generate_predictions_dim_samples!(model, t_pred) # for the full likelihood sample

### Plotting Predictions
##############################################################################

#### Model Trajectory
##############################################################################
plt = plot_predictions_union(model, t_pred, 1, dof=model.core.num_pars,
    compare_to_full_sample_type=LatinHypercubeSamples(), 
    xlims=(t_pred[1], t_pred[end]), linealpha=0.7, title="") # univariate profiles

plot!(plt, t_pred, solvedmodel(t_pred, θ_true), 
    label="True model trajectory", lw=3, color=:turquoise4, linestyle=:dash,
    dpi=150, size=(450,300), rightmargin=3mm)
savefig(plt, joinpath("docs", "src", "assets", "figures", "logistic", "logistic_univariate_trajectory.png"))

plt = plot_predictions_union(model, t_pred, 2, dof=model.core.num_pars,
    compare_to_full_sample_type=LatinHypercubeSamples(), xlims=(t_pred[1], t_pred[end]), linealpha=0.2, title="") # bivariate profiles

plot!(plt, t_pred, solvedmodel(t_pred, θ_true), 
    label="True model trajectory", lw=3, color=:turquoise4, linestyle=:dash,
    dpi=150, size=(450,300), rightmargin=3mm)
# savefig(plt, joinpath("docs", "src", "assets", "figures", "logistic", "logistic_bivariate_trajectory.png"))

#### 1-δ Population Reference Set 
##############################################################################
plt = plot_realisations_union(model, t_pred, 1, dof=model.core.num_pars,
    compare_to_full_sample_type=LatinHypercubeSamples(), title="") # univariate profiles

lq, uq = errorfunction(solvedmodel(t_pred, θ_true), θ_true, 0.95)
plot!(plt, t_pred, lq, fillrange=uq, fillalpha=0.3, linealpha=0,
    label="95% population reference set", color=palette(:Paired)[1])
scatter!(plt, data.t, data.y_obs, label="Observations", msw=0, ms=7,color=palette(:Paired)[3], 
    xlims=(t_pred[1], t_pred[end]),dpi=150, size=(450,300), rightmargin=3mm)
# savefig(plt, joinpath("docs", "src", "assets", "figures", "logistic", "logistic_univariate_reference_tolerance.png"))

plt = plot_realisations_union(model, t_pred, 2, dof=model.core.num_pars, 
    compare_to_full_sample_type=LatinHypercubeSamples(), title="") # bivariate profiles

plot!(plt, t_pred, lq, fillrange=uq, fillalpha=0.3, linealpha=0,
    label="95% population reference set", color=palette(:Paired)[1])
scatter!(plt, data.t, data.y_obs, label="Observations", msw=0, ms=7, color=palette(:Paired)[3], 
    xlims=(t_pred[1], t_pred[end]), dpi=150, size=(450,300), rightmargin=3mm)
# savefig(plt, joinpath("docs", "src", "assets", "figures", "logistic", "logistic_bivariate_reference_tolerance.png"))

## Coverage Testing
##############################################################################

### Data Generation
##############################################################################
@everywhere function data_generator(θ_true, generator_args::NamedTuple)
    y_obs = generator_args.y_true .+ rand(generator_args.dist, length(generator_args.t))
    if generator_args.is_test_set; return y_obs end

    data = (y_obs=y_obs, generator_args...)
    return data
end

@everywhere function reference_set_generator(θ_true, generator_args::NamedTuple, region::Float64)
    lq, uq = errorfunction(generator_args.y_true, θ_true, region)
    return (lq, uq)
end

training_gen_args = (y_true=y_true, t=t, dist=Normal(0, σ), is_test_set=false)
testing_gen_args = (y_true=solvedmodel(t_pred, θ_true), t=t_pred, dist=Normal(0, σ), is_test_set=true)

### Parameter Coverage
##############################################################################

#### Parameter Confidence Intervals
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

uni_coverage_df = check_univariate_parameter_coverage(data_generator,
    training_gen_args, model, 1000, θ_true, collect(1:model.core.num_pars),
    optimizationsettings=opt_settings)

#### Bivariate Profiles
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

biv_coverage_df = check_bivariate_parameter_coverage(data_generator,
    training_gen_args, model, 1000, 50, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    method = IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), 
    optimizationsettings=opt_settings)

biv_boundary_coverage_df = check_bivariate_boundary_coverage(data_generator,
    training_gen_args, model, 200, 50, 4000, θ_true,
    collect(combinations(1:model.core.num_pars, 2)); 
    method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true), 
    coverage_estimate_quantile_level=0.9,
    optimizationsettings=opt_settings)

### Prediction Coverage
##############################################################################

# On versions of Julia earlier than 1.10, we recommend setting the kwarg, 
# `manual_GC_calls`, to true in each of the coverage functions. Otherwise 
# the garbage collector may not successfully free memory every iteration 
# leading to out of memory errors.

#### Model Trajectory
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

full_trajectory_coverage_df = check_dimensional_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 30000, 
    θ_true, [collect(1:model.core.num_pars)])

uni_trajectory_coverage_df = check_univariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 
    θ_true, collect(1:model.core.num_pars), 
    num_points_in_interval=20, 
    optimizationsettings=opt_settings)

biv_trajectory_coverage_df = check_bivariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 20, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
    optimizationsettings=opt_settings)

##### Profile Path
##############################################################################
uni_trajectory_coverage_df = check_univariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 
    θ_true, collect(1:model.core.num_pars), 
    dof=model.core.num_pars,
    num_points_in_interval=20, 
    optimizationsettings=opt_settings)

biv_trajectory_coverage_df = check_bivariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 20, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    dof=model.core.num_pars,
    method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
    optimizationsettings=opt_settings)

#### 1-δ Population Reference Set and Observations
##############################################################################
full_reference_coverage_df = check_dimensional_prediction_realisations_coverage(data_generator,
    reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 30000, 
    θ_true, [collect(1:model.core.num_pars)])

uni_reference_coverage_df = check_univariate_prediction_realisations_coverage(data_generator,
    reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 
    θ_true, collect(1:model.core.num_pars), 
    dof=model.core.num_pars,
    num_points_in_interval=20, 
    optimizationsettings=opt_settings)

biv_reference_coverage_df = check_bivariate_prediction_realisations_coverage(data_generator,
    reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 20, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    dof=model.core.num_pars,
    method=IterativeBoundaryMethod(10, 5, 5, 0.15, 0.1, use_ellipse=true),
    optimizationsettings=opt_settings)
```
