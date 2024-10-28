##############################################################################
######################### TWO SPECIES LOGISTIC MODEL #########################
##### ORIGINAL FROM https://github.com/ProfMJSimpson/profile_predictions #####
##############################################################################

# Initial Setup
##############################################################################
using Distributed
if nprocs()==1; addprocs(10, env=["JULIA_NUM_THREADS"=>"1"]) end
@everywhere using Random, Distributions, DifferentialEquations
@everywhere using LogExpFunctions
@everywhere using LikelihoodBasedProfileWiseAnalysis
using Combinatorics

## Model and Likelihood Function Definition
##############################################################################
# USING LOGIT-NORMAL DATA DISTRIBUTION
@everywhere function DE!(dC, C, p, t)
    λ1, λ2, δ, KK = p
    S = C[1] + C[2]
    dC[1] = λ1 * C[1] * (1.0 - S/KK)
    dC[2] = λ2 * C[2] * (1.0 - S/KK) - δ*C[2]*C[1]/KK
end

@everywhere function odesolver(t, λ1, λ2, δ, KK, C01, C02)
    p=(λ1, λ2, δ, KK)
    C0=[C01, C02]
    tspan=eltype(p).((0.0, maximum(t))) # need to define with eltype for A.D. to work
    prob=ODEProblem(DE!, C0, tspan, p)
    sol=solve(prob, saveat=t)
    return sol[1,:], sol[2,:]
end

@everywhere function ODEmodel(t, θ)
    (y1, y2) = odesolver(t, θ[1], θ[2], θ[3], θ[4], θ[5], θ[6])
    return y1, y2
end

@everywhere function loglhood(θ, data)
    (y1, y2) = ODEmodel(data.t, θ)
    e=0.0
    for i in axes(data.y_obs,1)
        e += (loglikelihood(LogitNormal(logit(y1[i]/100.), θ[7]), data.y_obs[i,1]/100.) + 
                loglikelihood(LogitNormal(logit(y2[i]/100.), θ[7]), data.y_obs[i,2]/100.))
    end
    return e
end


## Initial Data and Parameter Definition
##############################################################################
# true data
t=[0, 769, 1140, 1488, 1876, 2233, 2602, 2889, 3213, 3621, 4028]
data11=[0.748717949, 0.97235023, 5.490243902, 17.89100529, 35, 56.38256703, 64.55087666, 66.61940299, 71.67362453, 80.47179487, 79.88291457]
data12=[1.927065527, 0.782795699, 1.080487805, 2.113227513, 3.6, 2.74790376, 2.38089652, 1.8, 0.604574153, 1.305128205, 1.700502513]

# simulated data
data11_simulated = [0.493044, 3.45615, 9.48014, 22.4058, 42.6299, 62.1176, 73.7073, 76.4812, 77.8276, 79.3753, 78.4457]
data12_simulated = [1.05671, 1.69669, 1.8453, 2.38854, 2.28005, 2.36482, 2.06503, 1.46763, 1.51971, 0.836532, 1.07154]

# true parameters used for coverage testing
θ_true = [0.003, 0.0004, 0.0004, 80.0, 0.4, 1.2, 0.1]
y_true = hcat(ODEmodel(t, θ_true)...)

# Named tuple of all data required within the log-likelihood function
data = (y_obs=hcat(data11_simulated, data12_simulated), t=t)

# Bounds on model parameters 
lb_sample = [0.0022, 0.00001, 0.0001, 73.0, 0.25, 0.7, 0.03]
ub_sample = [0.0036,  0.001, 0.0009, 85., 0.65, 2.0, 0.2]

lb = [0.0005, 0.00001, 0.00001, 60.0, 0.01, 0.1, 0.01]
ub = [0.01, 0.005, 0.005, 98.0, 2.0, 3.0, 1.0]

λ1g=0.002; λ2g=0.002; δg=0.001; KKg=80.0; C0g=[1.0, 1.0]; σg=0.5
θG = [λ1g, λ2g, δg, KKg, C0g[1], C0g[2], σg]

θnames = [:λ1, :λ2, :δ, :K, :C01, :C02, :σ]
par_magnitudes = [0.001, 0.001, 0.001, 10, 1, 1, 1]

## LikelihoodModel Initialisation
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5,))
model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

## Full Parameter Vector Confidence Set Evaluation
##############################################################################
full_likelihood_sample!(model, Int(1e7), lb=lb_sample, ub=ub_sample, use_distributed=false)

## Profiling
##############################################################################

### Univariate Profiles
##############################################################################
univariate_confidenceintervals!(model)
univariate_confidenceintervals!(model, dof=model.core.num_pars) # model.core.num_pars=7

get_points_in_intervals!(model, 20, additional_width=0.2)

### Bivariate Profiles
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
bivariate_confidenceprofiles!(model, 20, 
    method=RadialMLEMethod(0.15, 0.01),
    optimizationsettings=opt_settings)

bivariate_confidenceprofiles!(model, 20, 
    method=RadialMLEMethod(0.15, 0.01), 
    dof=model.core.num_pars,
    optimizationsettings=opt_settings)


### Plots of Profiles
##############################################################################
using Plots, Plots.PlotMeasures; gr()
Plots.reset_defaults(); Plots.scalefontsizes(0.75)

plts = plot_univariate_profiles(model, θs_to_plot=[1,2],
    confidence_levels=[0.95], dofs=[1])

plt = plot(plts..., layout=(1,2),
    legend=:outertop, title="", dpi=150, size=(550,300), margin=1mm)
display(plt)
# savefig(plt, joinpath("docs", "src", "assets", "figures", "two-species_logistic", "two-species_logistic_univariate_plots.png"))

plts = plot_bivariate_profiles(model, θcombinations_to_plot=[[1,2], [1,3]],
    confidence_levels=[0.95], dofs=[model.core.num_pars])

plt = plot(plts..., layout=(1,2),
    legend=:outertop, title="", dpi=150, size=(550,300), margin=1mm)
display(plt)
# savefig(plt, joinpath("docs", "src", "assets", "figures", "two-species_logistic", "two-species_logistic_bivariate_plots.png"))

## Predictions
##############################################################################
@everywhere function predictfunction(θ, data, t=data.t)
    y1, y2 = ODEmodel(t, θ) 
    y = hcat(y1,y2)
    return y
end

@everywhere function errorfunction(predictions, θ, region)
    lq, uq = logitnormal_error_σ_estimated(predictions ./ 100, θ, region, 7)
    lq .= lq .* 100
    uq .= uq .* 100
    return lq, uq
end

add_prediction_function!(model, predictfunction)
add_error_function!(model, errorfunction)

t_pred=LinRange(t[1], t[end], 201)

generate_predictions_univariate!(model, t_pred)
generate_predictions_bivariate!(model, t_pred)
generate_predictions_dim_samples!(model, t_pred) # for the full likelihood sample

### Plotting Predictions
##############################################################################

#### Model Trajectory
##############################################################################
model_trajectory = ODEmodel(t_pred, θ_true)

plt = plot_predictions_union(model, t_pred, 1, dof=model.core.num_pars,
    compare_to_full_sample_type=LatinHypercubeSamples(), plot_title="") # univariate profiles

plot!(plt; dpi=150, size=(450, 300), xlims=(t_pred[1], t_pred[end]))
plot!(plt[1], t_pred, model_trajectory[1],
    lw=3, color=:turquoise4, linestyle=:dash)
plot!(plt[2], t_pred, model_trajectory[2],
    label="True model trajectory", lw=3, color=:turquoise4, linestyle=:dash)
# savefig(plt, joinpath("docs", "src", "assets", "figures", "two-species_logistic", "two-species_logistic_univariate_trajectory.png"))

plt = plot_predictions_union(model, t_pred, 2, dof=model.core.num_pars,
    compare_to_full_sample_type=LatinHypercubeSamples(), plot_title="") # bivariate profiles

plot!(plt; dpi=150, size=(450, 300), xlims=(t_pred[1], t_pred[end]))
plot!(plt[1], t_pred, model_trajectory[1],
    lw=3, color=:turquoise4, linestyle=:dash)
plot!(plt[2], t_pred, model_trajectory[2],
    label="True model trajectory", lw=3, color=:turquoise4, linestyle=:dash)
# savefig(plt, joinpath("docs", "src", "assets", "figures", "two-species_logistic", "two-species_logistic_bivariate_trajectory.png"))

#### ``1-\delta`` Population Reference Set 
##############################################################################
lq, uq = errorfunction(hcat(ODEmodel(t_pred, θ_true)...), θ_true, 0.95)

plt = plot_realisations_union(model, t_pred, 1, dof=model.core.num_pars,
    compare_to_full_sample_type=LatinHypercubeSamples(), plot_title="") # univariate profiles

plot!(plt, t_pred, lq, fillrange=uq, fillalpha=0.3, linealpha=0,
    label="95% population reference set", color=palette(:Paired)[1])
scatter!(plt, data.t, data.y_obs, label="Observations", msw=0, ms=7, color=palette(:Paired)[3],
    xlims=(t_pred[1], t_pred[end]), dpi=150, size=(450, 300))
# savefig(plt, joinpath("docs", "src", "assets", "figures", "two-species_logistic", "two-species_logistic_univariate_reference_tolerance.png")) # univariate profiles

plt = plot_realisations_union(model, t_pred, 2, dof=model.core.num_pars, 
    compare_to_full_sample_type=LatinHypercubeSamples(), plot_title="") # bivariate profiles

plot!(plt, t_pred, lq, fillrange=uq, fillalpha=0.3, linealpha=0,
    label="95% population reference set", color=palette(:Paired)[1])
scatter!(plt, data.t, data.y_obs, label="Observations", msw=0, ms=7, color=palette(:Paired)[3],
    xlims=(t_pred[1], t_pred[end]), dpi=150, size=(450, 300))
# savefig(plt, joinpath("docs", "src", "assets", "figures", "two-species_logistic", "two-species_logistic_bivariate_reference_tolerance.png"))

## Coverage Testing
##############################################################################
# Computational time of these tests
# The computational time of several of the below tests is up to 4 hrs with 10 
# worker processes on the author's pc.

### Data Generation
##############################################################################
@everywhere function data_generator(θtrue, generator_args::NamedTuple)
    y_obs = zeros(size(generator_args.y_true))
    for i in eachindex(generator_args.y_true)
        y_obs[i] = rand(LogitNormal(logit(generator_args.y_true[i]/100.), θtrue[7]))
    end
    y_obs .= y_obs .* 100
    if generator_args.is_test_set; return y_obs end
    data = (y_obs=y_obs, generator_args...)
    return data
end

@everywhere function reference_set_generator(θtrue, generator_args::NamedTuple, region::Float64)
    lq, uq = errorfunction(generator_args.y_true, θtrue, region)
    return (lq, uq)
end

training_gen_args = (y_true=y_true, t=t, is_test_set=false)
testing_gen_args = (y_true=predictfunction(θ_true, data, t_pred), t=t_pred, is_test_set=true)

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
    method = RadialMLEMethod(0.15, 0.1), 
    optimizationsettings=opt_settings)

biv_boundary_coverage_df = check_bivariate_boundary_coverage(data_generator,
    training_gen_args, model, 100, 30, 2000, θ_true,
    collect(combinations(1:model.core.num_pars, 2)); 
    method = RadialMLEMethod(0.15, 0.1), 
    coverage_estimate_quantile_level=0.9,
    optimizationsettings=opt_settings)

### Prediction Coverage
##############################################################################

#### Model Trajectory
##############################################################################

# On versions of Julia earlier than 1.10, we recommend setting the kwarg, 
# `manual_GC_calls`, to true in each of the coverage functions. Otherwise 
# the garbage collector may not successfully free memory every iteration 
# leading to out of memory errors. This may still be important in Julia 1.10 
# onwards

opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

full_trajectory_coverage_df = check_dimensional_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 200, Int(1e7), 
    θ_true, [collect(1:model.core.num_pars)], lb=lb_sample, ub=ub_sample)

uni_trajectory_coverage_df = check_univariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 
    θ_true, collect(1:model.core.num_pars),
    optimizationsettings=opt_settings)

biv_trajectory_coverage_df = check_bivariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 30, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    method=RadialMLEMethod(0.15, 0.1),
    optimizationsettings=opt_settings)

##### Profile Path
##############################################################################
uni_trajectory_coverage_df = check_univariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 
    θ_true, collect(1:model.core.num_pars), 
    dof=model.core.num_pars,
    optimizationsettings=opt_settings)

biv_trajectory_coverage_df = check_bivariate_prediction_coverage(data_generator, 
    training_gen_args, t_pred, model, 1000, 30, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    dof=model.core.num_pars,
    method=RadialMLEMethod(0.15, 0.1),
    optimizationsettings=opt_settings)

#### ``1-\delta`` Population Reference Set and Observations
##############################################################################
full_reference_coverage_df = check_dimensional_prediction_realisations_coverage(data_generator,
    reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 200, Int(1e7), 
    θ_true, [collect(1:model.core.num_pars)], lb=lb_sample, ub=ub_sample)

uni_reference_coverage_df = check_univariate_prediction_realisations_coverage(data_generator,
    reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 
    θ_true, collect(1:model.core.num_pars), 
    dof=model.core.num_pars, 
    optimizationsettings=opt_settings)

biv_reference_coverage_df = check_bivariate_prediction_realisations_coverage(data_generator,
    reference_set_generator, training_gen_args, testing_gen_args, t_pred, model, 1000, 30, θ_true, 
    collect(combinations(1:model.core.num_pars, 2)),
    dof=model.core.num_pars,
    method=RadialMLEMethod(0.15, 0.1),
    optimizationsettings=opt_settings)