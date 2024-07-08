```@meta
CollapsedDocStrings = true
```
# Univariate Functions

```@index
Pages = ["univariate.md"]
```

## Likelihood Optimisation

```@docs
LikelihoodBasedProfileWiseAnalysis.univariateψ_ellipse_unbounded
LikelihoodBasedProfileWiseAnalysis.univariateψ
```

## Get Points in Confidence Interval

```@docs
LikelihoodBasedProfileWiseAnalysis.update_uni_dict_internal!
LikelihoodBasedProfileWiseAnalysis.get_points_in_interval_single_row
```

## Main Confidence Interval Logic 

```@docs
LikelihoodBasedProfileWiseAnalysis.get_interval_brackets
LikelihoodBasedProfileWiseAnalysis.add_uni_profiles_rows!
LikelihoodBasedProfileWiseAnalysis.set_uni_profiles_row!
LikelihoodBasedProfileWiseAnalysis.get_univariate_opt_func
LikelihoodBasedProfileWiseAnalysis.univariate_confidenceinterval
```