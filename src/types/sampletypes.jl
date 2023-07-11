"""
    AbstractSampleType

Supertype for sample types.

# Subtypes

[`UniformGridSamples`](@ref)

[`UniformRandomSamples`](@ref)

[`LatinHypercubeSamples`](@ref)
"""
abstract type AbstractSampleType end

"""
    UniformGridSamples()

Evaluate the interest parameter bounds space on a uniform grid, determining the optimised log-likelihood value at each grid point. Keep samples which are inside the confidence threshold boundary. 

# Supertype Hiearachy

`UniformGridSamples <: AbstractSampleType <: Any`
"""
struct UniformGridSamples <: AbstractSampleType end

"""
    UniformRandomSamples()

Take uniform random samples of interest parameter bounds space, determining the optimised log-likelihood value at each point. Keep samples which are inside the confidence threshold boundary. 

# Supertype Hiearachy

`UniformRandomSamples <: AbstractSampleType <: Any`
"""
struct UniformRandomSamples <: AbstractSampleType end

"""
    LatinHypercubeSamples()

Create a Latin Hypercube sampling plan in interest parameter bounds space, determining the optimised log-likelihood value at each point in the plan. Keep samples which are inside the confidence threshold boundary. Uses [LatinHypercubeSampling.jl](https://github.com/MrUrq/LatinHypercubeSampling.jl).

# Supertype Hiearachy

`LatinHypercubeSamples <: AbstractSampleType <: Any`
"""
struct LatinHypercubeSamples <: AbstractSampleType end
