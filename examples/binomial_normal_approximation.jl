##############################################################################
############# Gaussian Approximation of a Binomial Distribution ##############
########################## ORIGINAL FROM UNRELEASED ##########################
##############################################################################

# Initial Setup 
##############################################################################
using Random, Distributions, StaticArrays
using LikelihoodBasedProfileWiseAnalysis

## Model and Likelihood Function Definition
##############################################################################
distrib(θ) = Normal(θ[1] * θ[2], sqrt(θ[1] * θ[2] * (1 - θ[2])))

function loglhood(θ, data)
    return sum(logpdf.(distrib(θ), data.samples))
end

function predictfunction(θ, data, t=["n*p"]); [prod(θ)] end

## Initial Data and Parameter Definition
##############################################################################
# true parameters
θ_true = [100.0, 0.2]

# Named tuple of all data required within the log-likelihood function
data = (;samples=SA[21.9, 22.3, 12.8, 16.4, 16.4, 20.3, 16.2, 20.0, 19.7, 24.4])

# Bounds on model parameters
lb = [0.0001, 0.0001]
ub = [500.0, 1.0]

θnames = [:n, :p]
θG = [50, 0.3]
par_magnitudes = [100, 1]

## LikelihoodModel Initialisation
##############################################################################
opt_settings = create_OptimizationSettings(solve_kwargs=(;))
model = initialise_LikelihoodModel(loglhood, data, 
    θnames, θG, lb, ub, par_magnitudes, optimizationsettings=opt_settings)

## Evaluating a Concave Boundary
##############################################################################
bivariate_confidenceprofiles!(model, 200, 
    method=IterativeBoundaryMethod(20, 20, 20, 0.5, 0.01, use_ellipse=true))

## Visualising the Progress of the IterativeBoundaryMethod
##############################################################################
using Plots, Plots.PlotMeasures; gr()
Plots.reset_defaults(); Plots.scalefontsizes(0.75)

format = (size=(450, 300), dpi=150, title="",
    legend_position=:topright, palette=:Paired)
plt = plot_bivariate_profiles_iterativeboundary_gif(model, 0.2, 0.2; 
    markeralpha=0.5, color=2, save_as_separate_plots=false, save_folder=joinpath("docs", "src", "assets", "figures", "binomial"), format...)

## Coordinate Transformation
##############################################################################

### Redefining Functions
##############################################################################
function loglhood_Θ(Θ, data)
    return loglhood(exp.(Θ), data)
end

function predictfunctions_Θ(Θ, data, t=["n*p"]); [prod(exp.(Θ))] end

function forward_parameter_transformLog(θ)
    return log.(θ)
end

### Transforming Parameter Definitions
##############################################################################
lb_Θ, ub_Θ = transformbounds_NLopt(forward_parameter_transformLog, lb, ub)

Θnames = [:ln_n, :ln_p]
ΘG = forward_parameter_transformLog(θG)
par_magnitudes = [2, 1]

### LikelihoodModel Initialisation
##############################################################################
model = initialise_LikelihoodModel(loglhood_Θ, data, Θnames, ΘG, lb_Θ, ub_Θ, par_magnitudes)

### Re-evaluating the Bivariate Boundary
##############################################################################
bivariate_confidenceprofiles!(model, 40, method=RadialMLEMethod(0.15, 1.))

using Plots; gr()

plts = plot_bivariate_profiles(model, 0.2, 0.2; include_internal_points=true, markeralpha=0.9, format...)
display(plts[1])
savefig(plts[1], joinpath("docs", "src", "assets", "figures", "binomial", "binomial_bivariate_plot.png"))
