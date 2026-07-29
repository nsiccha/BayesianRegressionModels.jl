# API

## Frontend macros

```@docs
BayesianRegressionModels.@brm
BayesianRegressionModels.@n
BayesianRegressionModels.@x
BayesianRegressionModels.@getproperty
BayesianRegressionModels._brm
BayesianRegressionModels.parse!
```

## Intermediate representation

```@docs
BayesianRegressionModels.BRMI
BayesianRegressionModels.AbstractColumn
BayesianRegressionModels.MissingColumn
BayesianRegressionModels.DataColumn
BayesianRegressionModels.NamedColumn
BayesianRegressionModels.ExprColumn
BayesianRegressionModels.LikelihoodColumn
BayesianRegressionModels.MaterializedColumn
BayesianRegressionModels.Data
BayesianRegressionModels.MaybeData
BayesianRegressionModels.maybedata
```

## Column-tree accessors

```@docs
BayesianRegressionModels.name
BayesianRegressionModels.getf
BayesianRegressionModels.getargs
BayesianRegressionModels.getkwargs
BayesianRegressionModels.getop
BayesianRegressionModels.getbroadcast
```

## Formula-term markers

```@docs
BayesianRegressionModels.assign
BayesianRegressionModels.doublepipe
BayesianRegressionModels.gr
BayesianRegressionModels.gp
BayesianRegressionModels.hsgp
BayesianRegressionModels.offset
BayesianRegressionModels.zscale
BayesianRegressionModels.center
BayesianRegressionModels.standardize
BayesianRegressionModels.protect
BayesianRegressionModels.factor
BayesianRegressionModels.mi
BayesianRegressionModels.me
BayesianRegressionModels.s
BayesianRegressionModels.t2
BayesianRegressionModels.ar
BayesianRegressionModels.mo
BayesianRegressionModels.mo1
```

## Likelihood distributions / prior markers

```@docs
BayesianRegressionModels.OrderedLogistic
BayesianRegressionModels.CategoricalLogit
BayesianRegressionModels.Horseshoe
BayesianRegressionModels.ZeroInflatedPoisson
BayesianRegressionModels.NegativeBinomial2
BayesianRegressionModels.BetaBinomial
BayesianRegressionModels.BetaBinomial2
BayesianRegressionModels.BinomialLogit
BayesianRegressionModels.CircularVonMises
```

## Backends

```@docs
BayesianRegressionModels.VBRMI
BayesianRegressionModels.SBBRMI
BayesianRegressionModels.stan_code
BayesianRegressionModels.GenerativeDeclaration
BayesianRegressionModels.GenerativePlan
BayesianRegressionModels.generative_plan
BayesianRegressionModels.reprocess
BayesianRegressionModels.restan_data
```

## Introspection

```@docs
BayesianRegressionModels.outcomes
BayesianRegressionModels.linear_predictor_op
BayesianRegressionModels.linear_predictors
BayesianRegressionModels.predictors
BayesianRegressionModels.grouping_factors
BayesianRegressionModels.column_data
BayesianRegressionModels.data_columns
BayesianRegressionModels.dependencies
BayesianRegressionModels.hierarchical_outcomes
BayesianRegressionModels.linear_predictor_args
BayesianRegressionModels.data_args
BayesianRegressionModels.primary_lp
```

## Extension API

For downstream packages adding their own formula terms (e.g.
`bruno-ext`):

```@docs
BayesianRegressionModels.Part
BayesianRegressionModels.push_parts!!
BayesianRegressionModels.nparams
BayesianRegressionModels.lprior!
BayesianRegressionModels.vbroadcasted
BayesianRegressionModels.vmeta_sampling_rhs
BayesianRegressionModels._sb_submodel_rhs!
```
