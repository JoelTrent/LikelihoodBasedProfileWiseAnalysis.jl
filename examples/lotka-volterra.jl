##############################################################################
############################ LOTKA VOLTERRA MODEL ############################
########### ORIGINAL FROM https://github.com/ProfMJSimpson/Workflow ##########
##############################################################################

# Initial Setup
##############################################################################
using Distributed
# if nprocs()==1; addprocs(6, env=["JULIA_NUM_THREADS"=>"1"]) end
@everywhere using Random, Distributions, DifferentialEquations, StaticArrays
@everywhere using PlaceholderLikelihood
using Combinatorics

## Model and Likelihood Function Definition
##############################################################################
@everywhere function lotka_static(C,p,t)
    dC_1=p[1]*C[1] - C[1]*C[2];
    dC_2=p[2]*C[1]*C[2] - C[2];
    SA[dC_1, dC_2]
end

@everywhere function odesolver(t,α,β,C01,C02)
    p=SA[α,β]
    C0=SA[C01,C02]
    tspan=(0.0,t[end])
    prob=ODEProblem(lotka_static,C0,tspan,p)
    sol=solve(prob, AutoTsit5(Rosenbrock23()), saveat=t);
    return sol[1,:], sol[2,:]
end

@everywhere function ODEmodel(t,θ)
    return odesolver(t,θ[1],θ[2],θ[3],θ[4])
end

@everywhere function loglhood(θ, data)
    (y1, y2) = ODEmodel(data.t, θ)
    e=loglikelihood(data.dist, data.y_obs[:, 1] .- y1)  
    f=loglikelihood(data.dist, data.y_obs[:, 2] .- y2)
    return e+f
end

## Initial Data and Parameter Definition
##############################################################################
# true parameters
α_true=0.9; β_true=1.1; x0_true=0.8; y0_true=0.3
@everywhere global σ=0.2
θ_true=[α_true, β_true, x0_true, y0_true]

t=LinRange(0,7,15)
y_true = hcat(ODEmodel(t, θ_true)...)
y_obs = [0.99 0.22; 1.02 0.26; 1.28 0.38; 1.92 0.36; 2.03 0.80; 1.41 1.78;
         1.54 2.04; 0.67 1.63; 0.18 1.45; 0.44 1.13; 0.74 0.94; 0.37 0.86;
         0.01 0.16; 0.65 0.52; 0.54 0.32]

# Named tuple of all data required within the log-likelihood function
data = (y_obs=y_obs, t=t, dist=Normal(0, σ))

# Bounds on model parameters 
αmin, αmax = (0.4, 1.5)
βmin, βmax = (0.7, 1.8)
x0min, x0max = (0.4, 1.3)
y0min, y0max = (0.02, 0.8)
lb = [αmin,βmin,x0min,y0min]
ub = [αmax,βmax,x0max,y0max]

θG = θ_true
θnames = [:α, :β, :x0, :y0]
par_magnitudes = [1,1,1,1]

## LikelihoodModel Initialisation
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

## Full Parameter Vector Confidence Set Evaluation
##############################################################################
full_likelihood_sample!(model, 500000, use_distributed=true)

## Profiling
##############################################################################

### Univariate Profiles
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
univariate_confidenceintervals!(model, optimizationsettings=opt_settings)
univariate_confidenceintervals!(model, dof=model.core.num_pars, # model.core.num_pars=4
    optimizationsettings=opt_settings) 

### Bivariate Profiles
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
bivariate_confidenceprofiles!(model, 30, 
    method=IterativeBoundaryMethod(20, 5, 5, 0.15, 1.0, use_ellipse=true), 
    optimizationsettings=opt_settings)

bivariate_confidenceprofiles!(model, 30, 
    method=IterativeBoundaryMethod(20, 5, 5, 0.15, 1.0, use_ellipse=true), 
    dof=model.core.num_pars,
    optimizationsettings=opt_settings)

### Plots of Profiles
##############################################################################
using Plots; gr()

plts = plot_univariate_profiles(model, confidence_level=0.95, dof=1)
plt = plot(plts..., layout=(1,4))
display(plt)

plts = plot_bivariate_profiles(model, confidence_level=0.95, dof=model.core.num_pars)
plt = plot(plts..., layout=(2,3))
display(plt)


## Predictions
##############################################################################
@everywhere function predictfunction(θ, data, t=data.t)
    y1, y2 = ODEmodel(t, θ) 
    y = hcat(y1,y2)
    return y
end

@everywhere function errorfunction(predictions, θ, region); normal_error_σ_known(predictions, θ, region, σ) end

add_prediction_function!(model, predictfunction)
add_error_function!(model, errorfunction)

t_pred=LinRange(0,10,201)
generate_predictions_univariate!(model, t_pred)
generate_predictions_bivariate!(model, t_pred)
generate_predictions_dim_samples!(model, t_pred) # for the full likelihood sample

### Plotting Predictions
##############################################################################

#### Model Trajectory
##############################################################################
plot_predictions_union(model, t_pred, 1, dof=model.core.num_pars,
    compare_to_full_sample_type=LatinHypercubeSamples()) # univariate profiles

plot_predictions_union(model, t_pred, 2, dof=model.core.num_pars,
    compare_to_full_sample_type=LatinHypercubeSamples()) # bivariate profiles

#### 1-δ  Population Reference Set 
##############################################################################
plot_realisations_union(model, t_pred, 1, dof=model.core.num_pars,
    compare_to_full_sample_type=LatinHypercubeSamples()) # univariate profiles

plot_realisations_union(model, t_pred, 2, dof=model.core.num_pars, 
    compare_to_full_sample_type=LatinHypercubeSamples()) # bivariate profiles

## Coverage Testing
##############################################################################

### Data Generation
##############################################################################
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    y_obs = generator_args.y_true .+ rand(generator_args.dist, length(generator_args.t), 2)
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

@everywhere function reference_set_generator(θtrue, generator_args::NamedTuple, region::Float64)
    lq, uq = errorfunction(generator_args.y_true, θtrue, region)
    return (lq, uq)
end

training_gen_args = (y_true=y_true, t=t, dist=Normal(0, σ), is_test_set=false)
testing_gen_args = (y_true=hcat(ODEmodel(t_pred, θ_true)...), t=t_pred, dist=Normal(0, σ), is_test_set=true)

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
    training_gen_args, model, 1000, 30, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    method = IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true), 
    optimizationsettings=opt_settings)

biv_boundary_coverage_df = check_bivariate_boundary_coverage(data_generator,
    training_gen_args, model, 100, 30, 5000, θ_true,
    collect(combinations(1:model.core.num_pars, 2)); 
    method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true), 
    coverage_estimate_quantile_level=0.9,
    optimizationsettings=opt_settings)

### Prediction Coverage
##############################################################################

#### Model Trajectory
##############################################################################

# On versions of Julia earlier than 1.10, we recommend setting the kwarg, 
# `manual_GC_calls`, to true in each of the coverage functions. Otherwise 
# the garbage collector may not successfully free memory every iteration 
# leading to out of memory errors.

opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

full_trajectory_coverage_df = check_dimensional_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 500000, 
    θ_true, [collect(1:model.core.num_pars)])

uni_trajectory_coverage_df = check_univariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 
    θ_true, collect(1:model.core.num_pars), 
    num_points_in_interval=20, 
    optimizationsettings=opt_settings)

biv_trajectory_coverage_df = check_bivariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 30, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
    optimizationsettings=opt_settings)

##### Profile path
##############################################################################
uni_trajectory_coverage_df = check_univariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 
    θ_true, collect(1:model.core.num_pars), 
    dof=model.core.num_pars,
    num_points_in_interval=20, 
    optimizationsettings=opt_settings)

biv_trajectory_coverage_df = check_bivariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 30, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    dof=model.core.num_pars,
    method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
    optimizationsettings=opt_settings)

#### 1-δ Population Reference Set and Observations
##############################################################################
full_reference_coverage_df = check_dimensional_prediction_realisations_coverage(data_generator,
    reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 500000, 
    θ_true, [collect(1:model.core.num_pars)])

uni_reference_coverage_df = check_univariate_prediction_realisations_coverage(data_generator,
    reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 
    θ_true, collect(1:model.core.num_pars), 
    dof=model.core.num_pars,
    num_points_in_interval=20, 
    optimizationsettings=opt_settings)

biv_reference_coverage_df = check_bivariate_prediction_realisations_coverage(data_generator,
    reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 30, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    dof=model.core.num_pars,
    method=IterativeBoundaryMethod(20, 5, 5, 0.15, 0.1, use_ellipse=true),
    optimizationsettings=opt_settings)

