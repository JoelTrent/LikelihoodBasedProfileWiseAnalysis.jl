```@index
Pages = ["univariate.md"]
```

# Univariate Functions

## Initialisation and Array Mapping

```@docs
PlaceholderLikelihood.variablemapping1dranges
PlaceholderLikelihood.variablemapping1d!
PlaceholderLikelihood.boundsmapping1d!
PlaceholderLikelihood.init_univariate_parameters
```

## Likelihood Optimisation

```@docs
PlaceholderLikelihood.univariateΨ_ellipse_unbounded
PlaceholderLikelihood.univariateΨ
```

## Get Points in Confidence Interval

```@docs
PlaceholderLikelihood.update_uni_dict_internal!
PlaceholderLikelihood.get_points_in_interval_single_row
```

## Main Confidence Interval Logic 

```@docs
PlaceholderLikelihood.get_interval_brackets
PlaceholderLikelihood.add_uni_profiles_rows!
PlaceholderLikelihood.set_uni_profiles_row!
PlaceholderLikelihood.get_univariate_opt_func
PlaceholderLikelihood.univariate_confidenceinterval
PlaceholderLikelihood.univariate_confidenceinterval_master
```