# Common Functions

```@index
Pages = ["common.md"]
```

## Utility and Log-likelihood Thresholds

```@docs
LikelihoodBasedProfileWiseAnalysis.convertθnames_toindices
LikelihoodBasedProfileWiseAnalysis.ll_correction
LikelihoodBasedProfileWiseAnalysis.get_target_loglikelihood
LikelihoodBasedProfileWiseAnalysis.get_consistent_tuple
```

## DataFrame Subsets

```@docs
LikelihoodBasedProfileWiseAnalysis.desired_df_subset
```

## Nuisance Parameters and Array Mapping

```@docs
LikelihoodBasedProfileWiseAnalysis.variablemappingranges
LikelihoodBasedProfileWiseAnalysis.variablemapping!
LikelihoodBasedProfileWiseAnalysis.boundsmapping!
LikelihoodBasedProfileWiseAnalysis.init_nuisance_parameters
LikelihoodBasedProfileWiseAnalysis.correct_θbounds_nuisance
```