# LikelihoodBasedProfileWiseAnalysis

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JoelTrent.github.io/LikelihoodBasedProfileWiseAnalysis.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JoelTrent.github.io/LikelihoodBasedProfileWiseAnalysis.jl/dev/)
[![Build Status](https://github.com/JoelTrent/LikelihoodBasedProfileWiseAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoelTrent/LikelihoodBasedProfileWiseAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JoelTrent/LikelihoodBasedProfileWiseAnalysis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JoelTrent/LikelihoodBasedProfileWiseAnalysis.jl)

__Warning: [ForwardDiff use on ODE problems is currently broken on this package](https://github.com/JoelTrent/LikelihoodBasedProfileWiseAnalysis.jl/issues/55). A fix will be worked on but requires spare time outside of work.__

This package is an implementation and exploration of the [likelihood-based Profile-Wise Analysis (PWA) workflow](https://doi.org/10.1371/journal.pcbi.1011515) from Matthew Simpson and Oliver Maclaren. It provides methods for:

- Maximum Likelihood Estimation.
- Calculation of the observed Fisher information matrix (FIM) and associated approximation of the log-likelihood function.
- Parameter identifiability analysis.
- Parameter confidence intervals.
- Evaluating univariate profiles.
- Evaluating the boundaries of bivariate profiles and sampling points within these boundaries.
- Rejection sampling of full parameter vector confidence sets and profiles.
- Simultaneous prediction of model solutions/trajectories using approximate profile-wise confidence trajectory sets.
- Simultaneous prediction of population reference sets using approximate profile-wise reference tolerance sets.

Additionally, to assist with evaluating the frequentist coverage properties of intervals and sets within the PWA workflow on new models it provides methods for the coverage testing of:
- Parameter confidence intervals.
- Bivariate confidence profiles.
- Profile-wise confidence trajectory sets.
- Profile-wise reference tolerance sets.

To understand the background of the workflow and how it can be used see the Motivation section of the documention. For implementation examples see the examples section, such as on a Logistic Model. To better understand how to interact with the user interface and in particular the `LikelihoodModel`, which holds all the information on computed profiles and predictions, check out the user interface starting with the Initialisation section.

A package developed to fulfil the requirements of a Masters of Engineering at The University of Auckland by Joel Trent between March 2023 and February 2024. 

Supervised by Oliver Maclaren, Ruanui Nicholson and Matthew Simpson.

## Getting Started: Installation

To install the package, use the following command inside the Julia REPL. In the future this will be able to be installed directly rather than via the url.

```julia
using Pkg
Pkg.add(url="https://github.com/JoelTrent/LikelihoodBasedProfileWiseAnalysis.jl")
```

To load the package, use the command:

```julia
using LikelihoodBasedProfileWiseAnalysis
```

## Alternatives for Likelihood-Based Uncertainty Quantification

If you are solely interested in parameter identifiability analysis and computing parameter confidence intervals we recommend [LikelihoodProfiler](https://insysbio.github.io/LikelihoodProfiler.jl/stable/), which is generally more stable and faster than the implementation in this package on the models we've tested.

[ProfileLikelihood](https://danielvandh.github.io/ProfileLikelihood.jl/stable/) is another excellent package which implements the PWA workflow from Simpson and Maclaren - it has a different interface and its own set of heuristics for computing profiles. 

[InformationGeometry](https://rafaelarutjunjan.github.io/InformationGeometry.jl/stable/) can compute the exact confidence regions/boundaries of models using differential geometry. It would be an interesting approach which could potentially evaluate confidence set boundaries more efficiently than the heuristics implemented in this package. Resultantly, its use within the PWA workflow may be worth investigating.