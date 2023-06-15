# Dimensional Samples

```@index
Pages = ["dimensional.md"]
```

We can generate samples within the loglikelihood boundary at any dimension of model interest parameters.

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
dimensional_likelihood_sample!
bivariate_concave_hull
```