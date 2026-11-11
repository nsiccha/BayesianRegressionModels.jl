# Public API

```@meta
CurrentModule = BayesianRegressionModels
```

## Declaring models

Build a model from a declaration and bind its data. See [Model syntax](model-syntax.md).

```@docs
@brm
@getproperty
@n
@x
AbstractColumn
BRMBuilder
BRMComponent
BRMI
Data
DataColumn
ExprColumn
LikelihoodColumn
MaybeData
MissingColumn
NamedColumn
brm
brm_components
getargs
getf
getkwargs
getop
maybedata
name
```

## Fitting

Choose a backend, construct an inference problem and retain fitted state. See [Backends](backends.md) and [Adaptive centering](adaptive-centering.md).

```@docs
AbstractAdaptiveCenteringBlock
AdaptiveCenteringBlock
BRMCoordinatePair
BRMCoordinateTransport
BRMCoordinateTransportError
BRMCutFamily
BRMFilter
BRMFilterSMC
BRMFittedState
BRMLaplacePlan
BRMLaplaceProblem
BRMNLMEConditionalProblem
BRMNLMEJointMap
BRMNLMEUnitProblem
BRMNLMEView
BRMNLMEViewError
BRMSMCCarriedReference
BRMSMCFit
BRMSMCWindowFit
BRMSVILaplaceState
BRMSVINonfiniteError
BRMSVIOptimizerState
BRMSVIStart
BRMSVIState
BRMSparseLaplaceProblem
BRMStructuredLaplaceProblem
GaussianBlockLaplace
LearnedCentering
MatrixFreeLaplace
R2D2AdaptiveCenteringBlock
RKBRMI
S2ZEffectBlock
SBBRMI
adaptive_centering_blocks
adaptive_centering_controls
adaptive_centering_problem
brm_check_coordinate_transport
brm_coordinate_transport
brm_cut_family
brm_cut_fit
brm_cut_point
brm_cut_propagate
brm_execute
brm_filter
brm_filter_checkpoint
brm_filter_step
brm_filter_stream
brm_filter_tune
brm_fitted_state
brm_laplace_mode
brm_laplace_plan
brm_nlme_conditional_point
brm_nlme_estimates
brm_nlme_fit
brm_nlme_joint_start
brm_nlme_joint_values
brm_nlme_loglikelihood
brm_nlme_loglikelihood_and_gradient
brm_nlme_loglikelihoods
brm_nlme_loglikelihoods_and_gradients
brm_nlme_model
brm_nlme_start
brm_nlme_unit_eta
brm_nlme_unit_problem
brm_nlme_view
brm_rk_point_to_stan
brm_rk_to_stan
brm_smc_carry
brm_smc_checkpoint
brm_smc_rescore
brm_smc_compare
brm_smc_coordinates
brm_smc_fit
brm_smc_recover
brm_smc_tune
brm_smc_update
brm_stan_gradient_to_rk
brm_stan_logdensity_to_rk
brm_start
brm_svi_centeredness
brm_svi_combine
brm_svi_fit
brm_svi_optimizer_checkpoint
brm_svi_resume
rk_logdensity_problem
s2z_effect_blocks
select_ranef_centeredness
select_s2z_centeredness
select_s2z_rho
select_total_centeredness
total_effect_blocks
```

Additional exports in this task group:

| Origin | Names |
|:--|:--|
| BayesianRegressionModels | `LowRankGlobal`, `MeanFieldGlobal`, `RKDesignBinder`, `brm_filter_resume`, `brm_nlme_conditional_problem`, `brm_nlme_joint_map`, `brm_smc_resume`, `brm_smc_runs` |

### [`BRMNLMEModel`](@id BRMNLMEModel)

Prepared population model returned by [`brm_nlme_model`](@ref). It retains the
RK backend, unit partition and conditional likelihood workspaces used by NLME
and variational fitting.

### [`brm_laplace_problem`](@id brm_laplace_problem)

Construct the marginal population target described by [`BRMLaplaceProblem`](@ref).
The type's docstring lists the constructor keywords and approximation contract.

### [`brm_svi_checkpoint`](@id brm_svi_checkpoint)

Extract the data-only streaming state described by [`BRMSVIState`](@ref).
It carries a fitted approximation into a new window. Use
[`brm_svi_optimizer_checkpoint`](@ref) to resume interrupted optimization on the
same model and data.

### [`brm_stan_to_rk`](@id brm_stan_to_rk)

Apply the reverse coordinate permutation in [`brm_rk_to_stan`](@ref)'s shared
contract. Transports with simplex or centered blocks need the point-specific
operation below.

### [`brm_stan_point_to_rk`](@id brm_stan_point_to_rk)

Transform a Stan unconstrained point into RK coordinates, including the
transport's nonlinear blocks. Its contract is documented with the inverse,
[`brm_rk_point_to_stan`](@ref).

## Draws and summaries

Read fitted coordinates, predict responses and summarize conditional quantities. See [Warfarin PK/PD](warfarin.md).

```@docs
BRMPowerscaleSensitivity
brm_conditional_draws
brm_conditional_tree
brm_contrast_draws
brm_laplace_draws
brm_nlme_draws
brm_output_draws
brm_output_tree
brm_powerscale_inputs
brm_powerscale_sensitivity
brm_prediction_grid
brm_predictive_draws
brm_predictive_tree
brm_ranef_draws
brm_ranef_tree
brm_slope_draws
brm_summarize_draws
linear_predictor_args
linear_predictor_op
linear_predictors
modeled_covariates
parameter_draws
population_draws
population_model
predictors
prune_responses
recover_population_draws
recover_s2z_draws
rk_draw_tree
rk_draw_values
rk_restore_draws
term_draws
transport_draws
```

Additional exports in this task group:

| Origin | Names |
|:--|:--|
| BayesianRegressionModels | `brm_centerednessplot`, `brm_centering_lossplot`, `brm_comparisonsplot`, `brm_gradientplot`, `brm_pairplot`, `brm_posteriorplot`, `brm_ppcplot`, `brm_predictionsplot`, `brm_slopesplot` |

## Descriptions and export

Inspect declared structure and emit model descriptions or backend programs. See [Backends](backends.md).

```@docs
BRMDescription
BRMDescriptionComponent
BRMDescriptionContext
BRMDescriptionFragment
BRMDescriptionReference
BRMDescriptionTexts
BRMDescriptor
BRMDocumentation
BRMHighlight
BRMHighlightSpec
BRMInput
BRMOperation
BRMOutput
BRMPriorDescription
BRMSubmodel
GenerativeDeclaration
GenerativePlan
RanefBlock
TotalEffectBlock
brm_columns
brm_describe_component
brm_description
brm_description_binding
brm_description_components
brm_description_markdown
brm_description_math
brm_description_prior
brm_description_prior_references
brm_description_record
brm_description_symbol
brm_description_texts
brm_descriptor
brm_distribution_type
brm_documentation
brm_documentation_markdown
brm_highlight
brm_ladder_fit
brm_length_scale_floor
brm_model_view
brm_operation
brm_output
brm_output_coordinates
brm_outputs
brm_population_effect_coordinates
brm_ranef_sd_coordinates
brm_source
brm_term_coordinates
brm_transform_constants
brm_two_stage_fit
column_data
compiles
data_args
data_columns
dependencies
description_texts
effect_priors
generative_plan
grouping_factors
hierarchical_outcomes
outcomes
popcoefnames
primary_lp
priors_of
r2d2_priors
ranef_blocks
ranef_coordinates
ranef_effect_priors
ranefcoefnames
reprocess
required_brm_inputs
restan_data
rk_artifact_inputs
rk_bound_data
rk_bound_plan
rk_design_binder
rk_model
rkppl_model
stan_data
stan_instantiate
stan_model
structure_of
term_priors
transpiles
```

Additional exports in this task group:

| Origin | Names |
|:--|:--|
| BayesianRegressionModels | `BRMDescriptionCoverage`, `brm_description_prior_anchor`, `rk_translate_artifact` |

### [`stan_code`](@id stan_code)

Re-exported from StanBlocks. `stan_code(sb::SBBRMI)` returns the emitted Stan
program for the bound model; [`stan_data`](@ref) returns its corresponding data.

## Response families

Construct response distributions and prior families. See [Response families](response-families.md).

```@docs
AR1Normal
BetaBinomial2
BinomialLogit
CategoricalLogit
CircularVonMises
CloglogLink
Cumulative
DoubleIntervalCensored
Flat
Horseshoe
LKJCovarianceFactor
LogitLink
MvNormalCholesky
NegativeBinomial2
OrderedLogistic
Ordinal
OrdinalLink
OrdinalStructure
PrimaryCensored
ProbitLink
SkewDoubleExponential
StoppingRatio
TruncatedNormal
ZeroInflatedPoisson
double_interval_censored
interval_censored
primary_censored
```

Additional exports in this task group:

| Origin | Names |
|:--|:--|
| Distributions | `Arcsine`, `Bernoulli`, `BernoulliLogit`, `Beta`, `BetaBinomial`, `Binomial`, `Categorical`, `Cauchy`, `Chisq`, `Dirichlet`, `DirichletMultinomial`, `Erlang`, `Exponential`, `Frechet`, `Gamma`, `Gumbel`, `InverseGamma`, `InverseGaussian`, `LKJCholesky`, `Laplace`, `LocationScale`, `LogNormal`, `Logistic`, `MixtureModel`, `Multinomial`, `MvNormal`, `NegativeBinomial`, `Normal`, `NormalCanon`, `Pareto`, `Poisson`, `Rayleigh`, `SkewNormal`, `SkewedExponentialPower`, `TDist`, `Uniform`, `VonMises`, `Weibull`, `censored`, `logpdf`, `product_distribution` |
| BayesianRegressionModels | `HurdlePoisson` |
| LogExpFunctions | `log1mexp`, `log1pexp`, `logexpm1`, `logistic`, `logit`, `logsumexp` |

### [`truncated`](@id truncated)

Re-exported from Distributions. `truncated(distribution; lower, upper)`
normalizes the density over the retained interval. See
[bounds and vectors](terms-and-priors.md#Bounds-and-vectors) for the difference
from [`restricted`](@ref).
| StatsFuns | `normcdf`, `norminvcdf` |

## Terms and priors

Declare structured effects, preprocessing and formula priors. See the [term catalogue](terms-reference.md).

```@docs
assign
center
constant
doublepipe
effect
factor
gr
groups
kernel
me
mi
mm
mo
mo1
offset
protect
r2d2
ragged
restricted
s
standardize
t2
weighted
zscale
```

Additional exports in this task group:

| Origin | Names |
|:--|:--|
| StatsBase | `AbstractWeights`, `AnalyticWeights`, `FrequencyWeights`, `ProbabilityWeights`, `UnitWeights`, `Weights`, `aweights`, `fweights`, `pweights`, `uweights` |
| StatsAPI | `weights` |

## Fields

Declare spatial and temporal fields, then recover or predict their values. See [Space-time fields](space-time-fields.md).

```@docs
ar
brm_gmrf_predict
brm_multigroup_coherence
brm_multigroup_field
car
cdar
dar
gmrf
gp
hsgp
hsgp_boundary_check
hsgp_coordinate_draws
hsgp_population_curve
hsgp_transform_draws
multigroup
rw
select_hsgp_centeredness
```

## Exported for backends (not a user API)

These names are exported so the StanBlocks and ReactiveKernels backends can
reach them; they will be un-exported in a separate source change.

```@autodocs
Modules = [BayesianRegressionModels]
Public = true
Private = false
Filter = value -> any(name -> isdefined(BayesianRegressionModels, name) && value === getfield(BayesianRegressionModels, name), Main.BRM_DOCS_BACKEND_EXPORTS)
```

The following internal definitions are linked from the exported docstrings.

```@docs
BayesianRegressionModels._adaptive_cdar_centering_blocks
BayesianRegressionModels._adaptive_hsgp_centering_blocks
BayesianRegressionModels._adaptive_hsgp_level_centering_blocks
BayesianRegressionModels._s2z_fisher_raw
BayesianRegressionModels._s2z_rescale_rho
```
