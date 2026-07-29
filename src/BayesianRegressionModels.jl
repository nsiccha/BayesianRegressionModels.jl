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

# BRMDescriptor — ONE authoritative executable semantic model descriptor.
# Collapses the GenerativePlan (what BRM emitted), introspection.jl (the
# formula shape), the preproc record (dataframe provenance) and StanBlocks'
# own `stan_descriptor` (the executable half) into one reflectable value with
# derived operations, so a consumer mounts a declaration without keeping any
# parallel registry of its own.
include("descriptor.jl")

# Post-fit prediction modes — population-level ("no random effects") and
# transported ("same draws, new covariates / new group levels") prediction.
# Both are unconstrained-draw-matrix operations whose correctness depends on
# the emitted random-effect parameterization, so they live next to the emitter
# rather than being re-derived (differently) in every consumer.
using Random
include("prediction.jl")
include("adaptive_centering.jl")

# Public surface. The macros and value types everything downstream
# (web-macro, bruno, tests) reaches for.
export @brm, @n, @x, @getproperty
export assign, doublepipe, gr, gp, offset, zscale, center, standardize, protect, factor
export me, mi, s, ar, mo, mo1, hsgp, OrderedLogistic, Horseshoe,
       CategoricalLogit, ZeroInflatedPoisson, NegativeBinomial2,
       BetaBinomial, BetaBinomial2,
       sb_group_demo, addprop
# Bordet model family — formula-surface markers for custom likelihood + submodel
export TruncatedNormal, bordet_hierarchical_parametric, kernel
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
export BRMI, VBRMI, SBBRMI, GenerativeDeclaration, GenerativePlan
export BRMDescriptor, BRMInput, BRMOutput, BRMOperation
export brm_descriptor, brm_operation, brm_execute, brm_columns,
       required_brm_inputs

# Accessor helpers for column types — used unqualified by html renderers,
# sbimpl dispatch logic, and downstream extension hooks like bruno-ext.
export name, getf, getargs, getkwargs, getbroadcast, getop

# Macro plumbing + compiled-output accessors used by downstream code
# (web-macro's Formula struct, bruno's direct pipeline calls).
export parse!, _brm, stan_code, reprocess, restan_data, generative_plan

# Post-fit prediction — the population-level ("nore") and transported
# ("recov") modes, plus the random-effect block description both stand on.
export RanefBlock, ranef_blocks, ranef_coordinates,
       population_draws, transport_draws
export AdaptiveCenteringBlock, adaptive_centering_blocks,
       adaptive_centering_problem

# Introspection -- model-shape questions answered without re-walking
# the operations dict.
export outcomes, linear_predictor_op, linear_predictors, predictors,
       grouping_factors, column_data, data_columns, dependencies,
       hierarchical_outcomes,
       linear_predictor_args, data_args, primary_lp, popcoefnames

# Extension API. Downstream packages (bruno) add their own formula terms
# by defining methods on `vmeta_sampling_rhs` + `_sb_submodel_rhs!` and
# pushing `Part`s via `push_parts!!`; `nparams` + `lprior!` complete the
# vimpl side. `vbroadcasted` is the materializer they call to resolve
# column args inside those method bodies.
export Part, push_parts!!, nparams, lprior!
export vbroadcasted, vmeta_sampling_rhs, _sb_submodel_rhs!
export _sb_term_group_block, _sb_emit_group_block_term!

# Internal @slic submodels from sbimpl.jl — exported so downstream
# modules (bruno-ext via web-macro) that build their own SBBRMI-style
# models with `StanBlocks.SlicModel(body, data, mod)` where `mod` is the
# caller's namespace can still resolve the BRM built-in submodel names.
export popefs, cdirichlet, c0dirichlet, c01dirichlet,
       ranef_intercept, ranef_correlated, ranef_correlated_by,
       ranef_correlated_draws, ranef_correlated_by_draws,
       ranef_intercept_centered, ranef_correlated_centered,
       ranef_correlated_draws_centered,
       _sb_mo, _sb_cat, _sb_ar1, _sb_s, _sb_me,
       _sb_gp, _sb_gp_aniso, _sb_hsgp, _sb_hsgp_aniso,
       _sb_hsgp_by, _sb_hsgp_by_aniso,
       brm_exp_quad_cov, brm_hsgp_sqrt_spd, _sb_horseshoe,
       _sb_mi_normal, mi_merge,
       zero_inflated_poisson, zero_inflated_poisson_lpmf,
       zero_inflated_poisson_lpmfs, zero_inflated_poisson_rng,
       sb_group_demo_slic, sb_group_clamped_demo, sb_group_clamped_demo_slic

end # module
