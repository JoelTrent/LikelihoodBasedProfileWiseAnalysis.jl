# Dimensional Samples

We can generate samples within the log-likelihood boundary at any dimension of model interest parameters. Nuisance parameters will be set to the values that maximise the log-likelihood function, found using an optimisation scheme. Samples are only implemented for the true log-likelihood function (the [`LogLikelihood`](@ref) profile type).

## Sample Types

Three methods are implemented for rejection sampling of parameter confidence sets within the supplied parameter bounds. These will be inefficient if the bounds are not well-specified (are not close to the boundary of the desired parameter confidence set), and as both the model parameter dimension increases and the interest parameter dimension of the profile increases. By default these are the bounds contained with [`CoreLikelihoodModel`](@ref), however, seperate bounds can be supplied for the purposes of sampling. Uniform grid sampling, uniform random sampling and sampling from a random Latin Hypercube scheme (the default) are supported.

```@docs
AbstractSampleType
UniformGridSamples
UniformRandomSamples
LatinHypercubeSamples
```

## Full Likelihood Sampling

To sample a confidence set for the full parameter vector, a full parameter confidence set, we use [`full_likelihood_sample!`](@ref). The sampled confidence set will be contained within a [`SampledConfidenceStruct`](@ref) that is stored in the [`LikelihoodModel`](@ref).

```@docs
full_likelihood_sample!
```

## Dimensional Likelihood Sampling

Similarly, to sample a confidence set for an interest subset of the parameter vector, a 'dimensional profile', we use [`dimensional_likelihood_samples!`](@ref). The sampled confidence set(s) will be contained within a [`SampledConfidenceStruct`](@ref) that is stored in the [`LikelihoodModel`](@ref).

Note: dimensional likelihood samples can be 'full' likelihood samples as well. These will be computed before any other dimensional samples.

```@docs
dimensional_likelihood_samples!
get_dimensional_confidence_set
```

## Index

```@index
Pages = ["dimensional.md"]
```