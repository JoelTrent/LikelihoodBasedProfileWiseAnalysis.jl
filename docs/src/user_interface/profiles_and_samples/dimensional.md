# Dimensional Samples

```@index
Pages = ["dimensional.md"]
```

We can generate samples within the log-likelihood boundary at any dimension of model interest parameters. Nuisance parameters will be set to the values that maximise the log-likelihood function, found using an optimisation scheme. Samples are only implemented for the true log-likelihood function (the [`LogLikelihood`](@ref) profile type).

## Sample Types

```@docs
AbstractSampleType
UniformGridSamples
UniformRandomSamples
LatinHypercubeSamples
```

## Full Likelihood Sampling

```@docs
full_likelihood_sample!
```

## Dimensional Likelihood Sampling

Note: dimensional likelihood samples can be 'full' likelihood samples as well.

```@docs
dimensional_likelihood_samples!
```