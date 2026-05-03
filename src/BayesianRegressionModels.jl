module BayesianRegressionModels

# @brm macro — parses a formula block, produces a BRMI (BRM
# intermediate: expression tree + column metadata).
using OrderedCollections
include("macro.jl")

# Introspection helpers on a BRMI -- outcomes / linear predictors /
# grouping factors / continuous & categorical predictors / RE terms.
# Used by web-macro's auto-PPC detector (and any downstream code that
# wants the model shape without re-walking the operations dict).
include("introspection.jl")

# VBRMI — vectorized implementation. Materializes predictors and
# likelihood into a LogDensityProblems-compatible object.
using LogExpFunctions, InverseFunctions, Distributions, ElasticArrays,
      LogDensityProblems, LinearAlgebra, SpecialFunctions
import CategoricalArrays as CA
include("vimpl.jl")

# SBBRMI — StanBlocks backend. Lowers a BRMI into a StanBlocks SlicModel
# so it can be compiled by BridgeStan / fit via Stan.
using StanBlocks
include("sbimpl.jl")

# Public surface. The macros and value types everything downstream
# (web-macro, bruno, tests) reaches for.
export @brm, @n, @x, @getproperty
export assign, doublepipe, gr, gp, offset, zscale, center, standardize, protect, factor
export me, mi, s, ar, mo, mo1, hsgp, OrderedLogistic, Horseshoe, ZeroInflatedPoisson
# Logit-form Bernoulli/Binomial -- prefer these over `Bernoulli(logistic(eta))`
# / `Binomial(N, logistic(eta))`. Both backends lower to a logit-native log-pmf
# (Stan's `bernoulli_logit_lpmf` / `binomial_logit_lpmf`; LogExpFunctions'
# `loglogistic` / `log1mlogistic` on the Julia side), avoiding the `inv_logit`
# round-trip and staying numerically stable for large |eta|. `BernoulliLogit`
# is re-exported from Distributions; `BinomialLogit` is defined in vimpl.jl.
export BernoulliLogit, BinomialLogit
export Data, MaybeData, maybedata
export AbstractColumn, MissingColumn, DataColumn, NamedColumn,
       ExprColumn, LikelihoodColumn, MaterializedColumn
export BRMI, VBRMI, SBBRMI

# Accessor helpers for column types — used unqualified by html renderers,
# sbimpl dispatch logic, and downstream extension hooks like bruno-ext.
export name, getf, getargs, getkwargs, getbroadcast, getop

# Macro plumbing + compiled-output accessors used by downstream code
# (web-macro's Formula struct, bruno's direct pipeline calls).
export parse!, _brm, stan_code

# Introspection -- model-shape questions answered without re-walking
# the operations dict.
export outcomes, linear_predictor_op, linear_predictors, predictors,
       grouping_factors, column_data, data_columns, dependencies,
       hierarchical_outcomes,
       linear_predictor_args, data_args, primary_lp

# Extension API. Downstream packages (bruno) add their own formula terms
# by defining methods on `vmeta_sampling_rhs` + `_sb_submodel_rhs!` and
# pushing `Part`s via `push_parts!!`; `nparams` + `lprior!` complete the
# vimpl side. `vbroadcasted` is the materializer they call to resolve
# column args inside those method bodies.
export Part, push_parts!!, nparams, lprior!
export vbroadcasted, vmeta_sampling_rhs, _sb_submodel_rhs!

# Internal @slic submodels from sbimpl.jl — exported so downstream
# modules (bruno-ext via web-macro) that build their own SBBRMI-style
# models with `StanBlocks.SlicModel(body, data, mod)` where `mod` is the
# caller's namespace can still resolve the BRM built-in submodel names.
export popefs, cdirichlet, c0dirichlet, c01dirichlet,
       ranef_intercept, ranef_correlated, ranef_correlated_by,
       ranef_correlated_draws, ranef_correlated_by_draws,
       _sb_mo, _sb_cat, _sb_ar1, _sb_s, _sb_me, _sb_hsgp, _sb_horseshoe,
       zero_inflated_poisson, zero_inflated_poisson_lpmf,
       zero_inflated_poisson_lpmfs, zero_inflated_poisson_rng

end # module
