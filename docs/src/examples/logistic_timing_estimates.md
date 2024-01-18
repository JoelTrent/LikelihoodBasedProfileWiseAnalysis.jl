# Function Evaluation Timing - Logistic Model

Recording the number of function evaluations (likelihood and optimisation functions) that a particular method requires to evaluate a profile is implemented only in a singly threaded environment as we use an external package ([TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl/tree/master)) for recording; it's not inherently built in to our package. It is likely possible to implement this in a distributed environment, but given limited time this has not been implemented. When [TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl/tree/master) is not being used, no timing overhead is introduced. 

To demonstrate how to use this, we will use the [Logistic Model](@ref) example that we previously discussed.

## Initial Setup, Data and Parameter Definition

```julia
using Random, Distributions
using DataFrames
using Combinatorics
using TimerOutputs
using PlaceholderLikelihood
using PlaceholderLikelihood.TimerOutputs: TimerOutputs as TO

@everywhere function solvedmodel(t, θ)
    return (θ[2]*θ[3]) ./ ((θ[2]-θ[3]) .* (exp.(-θ[1] .* t)) .+ θ[3])
end

@everywhere function loglhood(θ, data)
    y=solvedmodel(data.t, θ)
    e=sum(loglikelihood(data.dist, data.y_obs .- y))
    return e
end

# DATA GENERATION FUNCTION 
@everywhere function data_generator(θ_true, generator_args::NamedTuple)
    y_obs = generator_args.y_true .+ rand(generator_args.dist, length(generator_args.t))
    if generator_args.is_test_set; return y_obs end

    data = (y_obs=y_obs, generator_args...)
    return data
end

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

training_gen_args = (y_true=y_true, t=t, dist=Normal(0, σ), is_test_set=false)
```

## LikelihoodModel Initialisation

```julia
opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5))
model = initialise_LikelihoodModel(loglhood, data, θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)
```

## Profiling

### Univariate Profile Timing

To time the number of likelihood and optimisation function evaluations required to find each 95% parameter confidence interval and find 20 points within the interval we use:

```julia
TO.enable_debug_timings(PlaceholderLikelihood)
TO.reset_timer!(PlaceholderLikelihood.timer)
timer_df = DataFrame(parameter=zeros(Int, model.core.num_pars), 
                        optimisation_calls=zeros(Int, model.core.num_pars),
                        likelihood_calls=zeros(Int, model.core.num_pars))


opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

for i in 1:model.core.num_pars
    univariate_confidenceintervals!(model, [i], num_points_in_interval=20, optimizationsettings=opt_settings)
    timer_df[i, :] .= i, TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

    TO.reset_timer!(PlaceholderLikelihood.timer)
end

TO.disable_debug_timings(PlaceholderLikelihood)
```

You can also print the timer results, which can be useful for identifying the strings that represent the different sections timed (e.g. `"Univariate confidence interval"`) as well as providing the amount of time taken.

```julia
print_timer(PlaceholderLikelihood.timer)
```

Similarly, if we wish to use asymptotic confidence intervals as the starting guess for the parameter confidence intervals we can first evaluate them using the [`EllipseApproxAnalytical`](@ref) profile type and set the keyword argument `use_ellipse_approx_analytical_start` to true.

```julia
TO.enable_debug_timings(PlaceholderLikelihood)
TO.reset_timer!(PlaceholderLikelihood.timer)
timer_df = DataFrame(parameter=zeros(Int, model.core.num_pars), 
                        optimisation_calls=zeros(Int, model.core.num_pars),
                        likelihood_calls=zeros(Int, model.core.num_pars))


opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

univariate_confidenceintervals!(model, profile_type=EllipseApproxAnalytical())
for i in 1:model.core.num_pars
    TO.reset_timer!(PlaceholderLikelihood.timer)

    univariate_confidenceintervals!(model, [i], num_points_in_interval=20, 
        use_ellipse_approx_analytical_start=true, optimizationsettings=opt_settings)
    timer_df[i, :] .= i, TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

    TO.reset_timer!(PlaceholderLikelihood.timer)
end

TO.disable_debug_timings(PlaceholderLikelihood)
```

If we wish to evaluate the average (mean) number of function evaluations required during a simulation (such as our coverage simulations) we need to define additional functions. For comparison to coverage simulations it is recommended that a random seed is set prior to training data generation here and prior to calling e.g. [`check_univariate_parameter_coverage`](@ref). This ensures that the training data used is consistent between the two simulations. 

The previous procedure could also just be used for each parameter in turn with [`check_univariate_parameter_coverage`](@ref), with the mean figure obtained by dividing through by the number of simulations. However, this would prevent using Distributed with the simulation which could make it infeasible. We are less concerned with the exact accuracy of the number of function evaluations in a coverage simulation and more concerned with the general magnitude so here we only evaluate the average number of function evaluations across 100 iterations. Note, we need to use [`initialise_LikelihoodModel`](@ref) for each new data set.

```julia
function record_CI_LL_evaluations!(N)
    timer_df = DataFrame(parameter=zeros(Int, model.core.num_pars), 
                        optimisation_calls=zeros(Int, model.core.num_pars),
                        likelihood_calls=zeros(Int, model.core.num_pars))
    
    Random.seed!(1234)
    training_data = [data_generator(θ_true, training_gen_args) for _ in 1:N]
    total_opt_calls = zeros(Int, model.core.num_pars)
    total_ll_calls = zeros(Int, model.core.num_pars)

    for j in 1:N
        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5))
        model = initialise_LikelihoodModel(loglhood, training_data[j], θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

        opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))
        for i in 1:model.core.num_pars
            TO.reset_timer!(PlaceholderLikelihood.timer)

            univariate_confidenceintervals!(model, [i], existing_profiles=:overwrite, optimizationsettings=opt_settings)

            total_opt_calls[i] += TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"])

            total_ll_calls[i] += TO.ncalls(
                PlaceholderLikelihood.timer["Univariate confidence interval"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])
        end
    end
    
    timer_df = DataFrame(parameter=zeros(Int, model.core.num_pars),
        mean_optimisation_calls=zeros(model.core.num_pars),
        mean_likelihood_calls=zeros(model.core.num_pars))

    timer_df[:, 1] .= 1:model.core.num_pars
    timer_df[:, 2] .= total_opt_calls ./ N
    timer_df[:, 3] .= total_ll_calls ./ N

    return timer_df
end

TO.enable_debug_timings(PlaceholderLikelihood)
TO.reset_timer!(PlaceholderLikelihood.timer)

timer_df = record_CI_LL_evaluations!(N)

TO.disable_debug_timings(PlaceholderLikelihood)
```

### Bivariate Profile Timing

We can do the same with bivariate profiles by modifying the string used to access the relevant timer section.

```julia
TO.enable_debug_timings(PlaceholderLikelihood)
TO.reset_timer!(PlaceholderLikelihood.timer)

len = length(combinations(1:model.core.num_pars, 2))
timer_df = DataFrame(parameter=zeros(Int, len), 
                        optimisation_calls=zeros(Int, len),
                        likelihood_calls=zeros(Int, len))

opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
    bivariate_confidenceprofiles!(model, [pars], 50, method=IterativeBoundaryMethod(20, 5, 5, 0.15, 1.0, use_ellipse=true), optimizationsettings=opt_settings)
    timer_df[i, :] .= i, TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Bivariate confidence boundary"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

    TO.reset_timer!(PlaceholderLikelihood.timer)
end

TO.disable_debug_timings(PlaceholderLikelihood)
```

### Dimensional Samples

And similarly for dimensional samples. Note, these are just sampled versions of profiles with interest parameter dimension ``\in [1, |\theta|]`` where ``|\theta|`` is the total number of model parameters, `model.core.num_pars`. For example, for bivariate dimensional samples (2D):

```julia
TO.enable_debug_timings(PlaceholderLikelihood)
TO.reset_timer!(PlaceholderLikelihood.timer)

len = length(combinations(1:model.core.num_pars, 2))
timer_df = DataFrame(parameter=zeros(Int, len), 
                        optimisation_calls=zeros(Int, len),
                        likelihood_calls=zeros(Int, len))

opt_settings = create_OptimizationSettings(solve_kwargs=(maxtime=5, xtol_rel=1e-12))

for (i, pars) in enumerate(collect(combinations(1:model.core.num_pars, 2)))
    dimensional_likelihood_samples!(model, [pars], 1000, optimizationsettings=opt_settings)
    timer_df[i, :] .= i, TO.ncalls(
                PlaceholderLikelihood.timer["Dimensional likelihood sample"]["Likelihood nuisance parameter optimisation"]),
            TO.ncalls(
                PlaceholderLikelihood.timer["Dimensional likelihood sample"]["Likelihood nuisance parameter optimisation"]["Likelihood evaluation"])

    TO.reset_timer!(PlaceholderLikelihood.timer)
end

TO.disable_debug_timings(PlaceholderLikelihood)
```