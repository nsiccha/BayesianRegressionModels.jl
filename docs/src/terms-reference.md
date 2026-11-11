# [Terms, priors and families](@id Formula-terms)

This reference gives the contracts behind the [model syntax](model-syntax.md),
[priors](terms-and-priors.md), [response families](response-families.md) and
[fields](space-time-fields.md) guides. Each executable table binds small
synthetic data, constructs a `BRMI`, and lowers it through `SBBRMI`. Its last
column is extracted from the emitted model density, including the likelihood.
It is generated evidence of the declaration, not a second handwritten model.

```@eval
Base.include(Main, joinpath(Main.BRMDocsComparisons.REPOSITORY_ROOT, "docs", "terms_reference.jl"))
nothing
```

## Columns and formula operators

| Form | Contract |
| --- | --- |
| `1`, `0` | Explicit intercept or no intercept; there is no implicit intercept |
| `x + z`, `x & z`, `x * z` | Add terms, interaction only, or main effects plus interaction |
| `offset(x)` | Add a data or sampled expression with coefficient one; no population coefficient and no random-effect offset |
| `protect(expr)` | Materialize a raw data expression as one column with a coefficient; ordinary raw expressions such as `log(x)` use this interpretation |
| `center(x)` | Subtract the fitted mean |
| `zscale(x)`, `standardize(x)` | Subtract the fitted mean and divide by fitted SD |
| `factor(c; ref=k)` | Treatment contrasts with chosen reference level |
| `factor(c; cmc=false)` | Retain treatment contrasts even without a predictor intercept |

Bare integer and categorical columns are factors. Under an intercept they
have K−1 contrasts. Without an intercept, the first categorical term uses K
cell means unless `cmc=false`. Fitted level order and transformation constants
are preserved by replay. Transformations take no tuning keywords; a term
computed on a group-level table uses that table's weighting, not observation
counts. Nested `@brm(...)` marks one likelihood argument as a formula; it
does not insert a link and is not a standalone formula value.

```@eval
Main.BRMTermsReference.table(:columns)
```

## Smooths and Gaussian processes

| Term | Representation and boundaries | Addressed quantities |
| --- | --- | --- |
| `s(x)` | Rank-10 thin-plate spline; at least 10 distinct finite values; centered linear null space plus 8 whitened penalized columns; the intercept owns the constant | `sd(lp, s(x))` |
| `t2(x,z; k=(5,5), basis=(:cr,:cr), full=false)` | Two cubic regression margins; each `k > 2`, with at least `k` distinct values; default has 3 null columns and `rr`, `rn`, `nr` penalty blocks | `sd(lp, t2(x,z), rr)` and the other penalty components |
| `gp(x...; cov=:exp_quad, iso=true, jitter=1e-9)` | Exact noncentered latent GP; a direct predictor contribution | `sd`, `length_scale` |
| `hsgp(x...; k=20, c=1.5, iso=true)` | Hilbert basis with `prod(k)` weights; approximate squared-exponential covariance | `sd`, `length_scale` |
| `gp(x; cov=:periodic, period, jitter=1e-9)` | Exact one-axis periodic covariance | `sd`, `length_scale` |
| `hsgp(x; k=20, cov=:periodic, period)` | `k` harmonics, `2k` sine/cosine weights; excludes the constant | `sd`, `length_scale` |

`s` and `t2` work on Stan and RK. Stan accepts explicit smoothing-SD
priors `Exponential(scale)`; RK currently keeps the default half-standard
normal scales and refuses overrides. They freeze the
fitted basis, knots and decomposition during replay. `s` is not a B-spline
`bs(...)` spelling and accepts neither `k` nor supplied knots.

GP/HSGP terms supply the function itself, without a second population
multiplier. `iso=false` gives one length per axis; the default shares one.
`by=g` gives group-specific HSGP weights with shared hyperparameters.
`cor=Omega` correlates those weights across group levels. `jitter` belongs
to exact GP; HSGP uses `k`, `c` or an explicit `domain` instead.

For a periodic kernel the covariance is
``\sigma^2\exp[-2\sin^2(\pi(x-x')/p)/\rho^2]``. Its length is in dimensionless
chord units. Periodic HSGP rejects `c`, `domain`, `by`, anisotropy, latent
inputs and orthogonalization. Exact periodic replay evaluates the kernel
at the new coordinates.

```@eval
Main.BRMTermsReference.table(:smooths)
```

### HSGP support and coordinates

An HSGP over a sampled predictor needs an explicit `domain`, not `c`.
The domain sets the basis; it does not truncate the predictor. The supported
latent-input form is one-dimensional, ungrouped and isotropic.
`orthogonal_to=:linear` projects out the current input's intercept and linear
direction. [`hsgp_population_curve`](@ref) reads the fitted population curve
on a grid using that fitted projection; it excludes other predictor terms
and does not extrapolate beyond the domain. Include transformed parameters
when obtaining the draws it needs.

The default squared-exponential length prior enforces an approximation floor.
For a one-axis half-domain width `L` and `k > 1`, it is
``(4L/\pi)\sqrt{\log(100)/(k^2-1)}``; an isotropic multi-axis basis uses the
strictest axis. With `k=1` this bound is unbounded. The periodic floor solves
``I_k(1/\rho^2)/I_1(1/\rho^2)=100^{-2}``.
An explicit length prior replaces the default support. Use its supported
truncation form with `length_scale_floor` when retaining that bound.
`hsgp_boundary_check` reports posterior proximity to it. Increasing
`k` lowers the floor; widening the domain without increasing `k` raises it.
A fitted curve alone cannot establish that length hyperparameters are resolved.

Stan HSGP hyper-predictors such as
`log(length_scale(hsgp(x))) ~ 1 + (1 | g)` predict group-specific lengths.
The analogous `log(sd(hsgp(x)))` predicts amplitudes. They support an
intercept and the term's own grouping, isotropic lengths, log-scale
intercept priors `Normal(0,1)` and half-normal group SDs. The length remains
at least its approximation floor. Constant level correlation `cor=Omega`
can combine with predicted amplitudes and lengths. A frequency-dependent
`multigroup(...)` correlation requires one shared length, so it accepts
predicted amplitudes but refuses predicted lengths.

Fixed `centeredness=c` controls physical basis weights through
``w=s^{1-c}u``, ``u\sim N(0,s^c)``. A scalar or one value per basis weight is
accepted for raw, ungrouped squared-exponential HSGP; nonzero values are
refused for periodic, latent-input, grouped or projected variants. This
keyword has a narrower surface than the runtime
[adaptive-centering wrapper](adaptive-centering.md).

## Latent covariates and ordered effects

| Term | Contract | Prior address |
| --- | --- | --- |
| `mo(c)` | Ordered factor effect via simplex increments, with a free coefficient | `simplex(lp, mo(c))` |
| `mo1(c)` | The same simplex contrast with coefficient fixed to one | `simplex(lp, mo1(c))` |
| `me(x, sd)` | Latent true covariate with known observation SD; the same latent covariate can feed fixed and random slopes | `latent(lp, me(x))` |
| `interval_censored(x; upper=lloq, lower=0)` | Rows equal to `lloq` have bounded latent covariates; larger values remain observed | `latent(lp, interval_censored(x))` |

The monotonic default is symmetric `Dirichlet(1)` over K−1 increments.
Here `Dirichlet(a)` means one concentration per increment; explicit
`Dirichlet(a1,...,aKminus1)` supplies each concentration. This addressed form
does not use the standalone simplex's dimension/concentration constructor.
Latent-covariate priors accept `Normal(location,scale)`; measurement-error
observation SDs are not prior targets. Censored predictors require finite
values at or above the limit and at least one censored row. This Stan surface
rebuilds the observed/censored split on replay.

```@eval
Main.BRMTermsReference.table(:latent)
```

## Trajectories and areal fields

| Term | Process | Prior and replay contract |
| --- | --- | --- |
| `ar(t; p=1)` | First-order autoregressive noise | Only order 1 is emitted; its transformed persistence has no addressed prior |
| `dar(t; p=1)` | Integrated AR(1) increments, starting at zero | Persistence in `[0,1]`; increasing unique time orders the increments, independent of gap size |
| `rw(t)` | `dar` with persistence fixed to zero | One shared state per distinct sorted time; append times for forecasts |
| `cdar(t; by=g, cor=C)` | Stationary AR(1) deviations per group, with optional known cross-group correlation | First SD is `sigma`; transition innovation SD is `sigma*sqrt(1-rho^2)` |
| `car(area; edges=(from,to), type=:icar)` | Intrinsic CAR on the undirected graph | Sum-to-zero constraint per connected component; isolated nodes have zero spatial effect |
| `car(area; edges=(from,to), type=:bym2)` | Scaled CAR plus independent area effects | `sd` is overall scale; `mixing` allocates structured variance; islands have independent effects |

`dar` defaults to a bounded `Normal(0.5,0.2)` persistence and
half-`Normal(0,0.2)` innovation SD. Its persistence address accepts Normal,
Beta or Uniform. `rw` exposes only its SD. `cdar` uses the same persistence
and marginal-SD defaults, freezes group order and the known correlation
factor, and accepts future appended steps; an unseen interior time is refused.
Its independent-group computation is linear in groups; dense `cor` couples
them. `resample_groups` redraws selected groups conditionally on the others.

CAR uses nodes in the graph and rows, with `nodes=` for additional areas.
ICAR applies a soft component sum constraint of SD `0.001 * component_size`.
BYM2 scales each connected component by the geometric mean of the diagonal
of its generalized inverse; defaults are half-`Normal(0,1)` SD and
`Beta(0.5,0.5)` mixing. Replay keeps the graph, including islands.
The [fields guide](space-time-fields.md) covers `gmrf` mesh, neighbor and
multiresolution representations and their prediction support.

```@eval
Main.BRMTermsReference.table(:trajectories)
```

## Group effects and coordinate choices

| Form | Contract |
| --- | --- |
| `(1 + x | p | g)` | Group intercept/slope block; the ID `p` can join margins from several predictors |
| `(1 + x || g)` | Independent margins |
| `gr(g; by=b)` | Separate group distributions by stratum |
| `gr(g; cor=Omega)` | Correlation across levels, giving covariance `Omega ⊗ Sigma` |
| `mm(g1,g2,...; weights=(w1,w2,...), normalize=true)` | Shared block across multiple memberships; equal weights when omitted |

`cor(:,p) ~ LKJCholesky(K,eta)` correlates the K margins within a group.
In contrast, `cor=Omega` accepts a known level-correlation matrix or a
separately declared `Omega ~ LKJCholesky(K,eta)`. The same declaration can
be shared with grouped HSGP weights. The dimension is the number of levels,
and `eta` is finite and positive. For ordinary random-effect blocks,
Student-t/R2D2, `by`, S2Z, adaptive centering and group resampling do not
combine with level correlation.

Multiple membership needs at least two group columns. Supplied weights must
be finite, nonnegative and have positive row sums. It does not combine with
a shared block ID, `cv_groups` or `centered_groups`.

```@eval
Main.BRMTermsReference.table(:groups)
```

Exact total coefficients and posterior-preserving sum-to-zero coordinates
are preparation options, not different likelihoods:

| Option | Applicability and interpretation |
| --- | --- |
| `total_groups=()` | Conventional coordinates; the default |
| `total_groups=:auto` or selected group/predictor pairs | Analytically integrate eligible population Normal, Student-t or Flat margins into group totals; partial selection is supported |
| `s2z_groups=(...)`, `s2z_rho=...` | Integrate the group mean and represent J−1 contrasts, preserving the original posterior |

Exact totals require eligible independent population margins, a compatible
shared group structure and supported scale priors. They can absorb frozen
numeric group covariates; offsets remain ordinary contributions. Known
level-correlation blocks remain conventional. Recovering population draws
adds their conditional uncertainty; transporting to new groups uses that
recovery rather than treating total coordinates as deviations.

S2Z supports Normal or Flat population priors, at least two groups and one
selected grouping per predictor; other groupings may remain conventional.
`s2z_rho` accepts a scalar, a margin vector or a group-by-margin matrix in
`[0,1]`. The Fisher-based rule supports identity Normal, logit Bernoulli or
Binomial, and log Poisson observations. Unobserved blocks keep conventional
predictive treatment. Use the current contrast coordinates and the named
recovery functions in the API; historical alternate coordinate experiments
are archived under research.

## Prior selection and scalar/vector parameters

| Prior target | Scope |
| --- | --- |
| `parameter ~ Distribution(...)` | Authored scalar, vector or simplex |
| `effect(lp, column) ~ Prior(...)` | Existing population coefficient; `Intercept` names the intercept |
| `sd(lp, group) ~ Prior(...)` | Group margin SD; a shared ID addresses its block |
| `cor(:, group) ~ LKJCholesky(K,eta)` | Within-level group correlation |
| `quantity(lp, term[,component]) ~ Prior(...)` | A model-scale parameter owned by a term |
| `quantity(lp, term[,component]) = value` | Fix a supported term parameter |

Colons are wildcard defaults; more concrete slots win. Equal-specificity
ties and unmatched targets fail during lowering. `optional(address)` permits
an absent target in a reusable prior fragment. Term addresses omit numeric
and keyword arguments, and tensor penalties require their component name.
Linked predictors use the underlying name (`sigma`, not `log(sigma)`).
Coefficient labels come from `popcoefnames`; `site_lvl_k` selects a fitted
cell-mean level. `effect_priors`, `term_priors`, `ranef_effect_priors` and
`r2d2_priors` expose selectors. `priors_of` selects addressed priors;
authored scalar/vector declarations remain in `structure_of`.

Population coefficients default to independent standard normals. For GP/HSGP
positive parameters, addressed families are LogNormal, InverseGamma, Gamma,
Exponential, Normal and Uniform with numeric hyperparameters. Positive support
is retained; Uniform also contributes its finite bounds. Both GP parameters
default to `LogNormal(0,1)`, subject to HSGP's length floor. An anisotropic
length prior applies to each axis. Raw standardized innovations remain
standard normal; they are not configurable prior targets.

| Declaration | Contract |
| --- | --- |
| `x ~ restricted(D; lower,upper)` | Original density kernel on bounded support; no interval-mass normalizer |
| `x ~ truncated(D; lower,upper)` | Conditional law with its interval-mass normalizer |
| `x ~ Horseshoe(...)` | Scalar local/global shrinkage construction; not an implicit shared scale across independent declarations |
| `share ~ Dirichlet(K,a)` or `Dirichlet(alpha)` | K-dimensional simplex, K≥2, positive finite data concentrations, K−1 free coordinates |
| `x ~ MvNormal(mean,covariance)` | Joint vector; matrix and `Diagonal` arguments are covariance; scalar/vector SD constructors retain their Julia convention |
| `x[1:n] .~ Family.(args...)` | Declared vector of independent elements; n is an integer or data expression |

Standalone Dirichlet rejects `Dirichlet(K)` and sampled concentrations; it
is not a response family. MvNormal supports literal/data means and covariance,
including expressions and sampled scalar scale; a sampled covariance matrix
is outside this prior surface. Bounds do not wrap a joint MvNormal prior.
Declared independent vectors support scalar broadcasts, length-n data
arguments, sampled scalar arguments and elementwise restricted/truncated or
LocationScale laws. Their role stays a parameter even if unobserved.
An undeclared vector-valued response is different. Typed LHS declarations
such as `x::vector[n]` are not formula syntax.

Stan accepts supported sampled arguments and bounds in scalar restricted
and truncated priors. RK accepts literal or observed scalar bounds, but refuses sampled bounds. Legacy
`lower`/`upper` distribution keywords express a restricted kernel; use the
explicit wrapper in new code. A normalizing mass that depends on a sampled
hyperparameter cannot be discarded without changing the model.

```@eval
Main.BRMTermsReference.table(:parameters)
```

### R² shrinkage

| Address | Variance allocation |
| --- | --- |
| `effect(lp,:) ~ r2d2(R2=Beta(a,b), tau_bsv=..., alpha=1)` | Predictor's non-intercept coefficient budget and one random-intercept residual |
| `sd(:,p) ~ r2d2(reference_scale=..., R2=Beta(a,b), alpha=1)` | Shared random-effect block budget |
| `sd(lp,p) ~ r2d2(...)` without a block-wide declaration | Independent per-margin ICC construction |
| `sd(:,p) ~ r2d2(include=(:population,:contrasts), ...)` | One joint budget over random effects and selected coefficient groups |

For the population construction, coefficient variance is
``\phi_j R^2\tau_{bsv}^2/\operatorname{Var}(x_j)`` and residual group SD is
``\sqrt{1-R^2}\tau_{bsv}``. A fixed `tau_bsv` or its default half-normal
prior is supported. Explicit column priors exclude columns; excluding all
columns is refused except an inherently intercept-only model. Shared
correlated intercept blocks must opt in on all margins or none.

For a random-effect block, margin SD is
``s_j\sqrt{\phi_jR^2/(1-R^2)}``. A more-specific margin selector can change
its reference scale, not create a second block budget. `mean_R2`/`prec_R2`
are aliases for Beta parameters and `concentration` aliases `alpha`.
Do not mix direct SD priors into the same budget. Shared block and joint
R2D2 budgets are available on Stan and RK.
The joint `include` form always includes random effects, excludes intercepts,
orders random margins before selected predictors and rejects conflicting
per-column priors. Without an explicit reference scale it introduces a
half-normal scale, so its latent R² is not an identified observed-scale R².

```@eval
Main.BRMTermsReference.table(:shrinkage)
```

## Response families

Only explicitly mapped constructors are accepted; subtyping a
Distributions.jl family is insufficient. These tables describe Stan coverage.
The guides and API identify narrower RK prediction/composition surfaces.

| Group | Constructors |
| --- | --- |
| Continuous, direct | `Normal`, `NormalCanon`, `Cauchy`, `TDist`, `Logistic`, `Gumbel`, `Chisq`, `Exponential`, `Gamma`, `Erlang`, `Beta`, `Uniform`, `LogNormal`, `Laplace`, `Frechet`, `Rayleigh`, `SkewNormal`, `Pareto`, `Weibull`, `InverseGamma`, `InverseGaussian`, `VonMises` |
| Continuous, restricted parameterization | Standard `Arcsine()`; `SkewedExponentialPower(mu,sigma,1,alpha)` with literal shape 1 |
| Discrete, direct | `Bernoulli`, `BernoulliLogit`, `Binomial`, `BinomialLogit`, `BetaBinomial`, `Poisson`, `NegativeBinomial` |
| Regression parameterizations | `LocationScale(mu,sigma,TDist(nu))`, `NegativeBinomial2(mu,phi)`, `BetaBinomial2(n,mean,precision)`, `SkewDoubleExponential(mu,sigma,tau)` |
| Extra-zero counts | `HurdlePoisson(lambda,p_zero)`, `ZeroInflatedPoisson(lambda,zi)` |
| Unordered/ordered | `CategoricalLogit(eta2,...)`, `Ordinal(structure,link,eta;...)`; legacy `OrderedLogistic(eta)` |
| Fixed circular interval | `CircularVonMises(mu,kappa; interval=(-pi,pi))` |
| Joint Gaussian | `[y1,y2,...] ~ MvNormalCholesky([mu1,mu2,...], L)` with declared `LKJCovarianceFactor` |

Scale/rate conversions preserve Julia distributions: Gamma/Erlang/Exponential
scales become Stan rates; `Pareto(shape,scale)` reorders to minimum/shape;
`NormalCanon(eta,lambda)` has mean `eta/lambda` and SD `1/sqrt(lambda)`;
`TDist(nu)` is standard Student-t. `Arcsine()` is `Beta(1/2,1/2)`; shifted
Arcsine forms are rejected. `InverseGaussian(mu,lambda)` has variance
`mu^3/lambda`, with all three quantities positive.

`SkewDoubleExponential(mu,sigma,tau)` uses Stan's scale: its CDF at `mu` is
`tau`, and `tau=0.5` is `Laplace(mu,sigma)`. A brms check-loss scale `s`
corresponds to `sigma=2s`. The supported SEPD shape-1 spelling translates to
`sigma=4*sigma_sepd*tau*(1-tau)`; other shapes are refused.

`VonMises(mu,kappa)` preserves moving support `[mu-pi,mu+pi]`.
`CircularVonMises` uses a fixed half-open interval of width `2pi` and positive
concentration. Use it for observations already encoded on a principal interval.

Categorical logits use the first fitted category as reference. Ordinal
structures are Cumulative and StoppingRatio, each with Logit, Probit or Cloglog
link tags. Threshold/stage priors are elementwise `Normal(0,1)`; the common
predictor must omit its intercept. Discrimination is positive, defaults to
one and needs scale identification when predicted (`cmc=false` retains a
reference). `per_threshold` supports numeric data terms for StoppingRatio.

`LKJCovarianceFactor(K; scale_prior=Exponential(1),shape=1)` combines K scales
with an LKJ Cholesky correlation factor. Joint observations need complete,
finite, equal-length rows. This surface does not implement non-Gaussian
residual copulas or partly missing vectors.

```@eval
Main.BRMTermsReference.table(:responses)
```

### Observation compositions

| Composition | Supported contract |
| --- | --- |
| `truncated(d; lower,upper)` | Conditional sampling law on the interval |
| `censored(d; lower,upper)` | Clamp the base draw to the observation limits |
| `interval_censored(d; upper)` | Probability of `(response,upper]`; prediction returns an uncoarsened base draw |
| `weighted(d,aweights(w))` | Normal precision weights: SD divided by `sqrt(w)` in density and prediction |
| `weighted(d,fweights(w))`, `weighted(d,weights(w))` | Replication/power weights multiply direct mapped densities; prediction stays the base law |
| `mi(y) ~ d` | Impute partly missing continuous responses |
| `MixtureModel(components,share)` | Marginalize scalar component membership |

The three bound wrappers support Normal, LogNormal, Exponential, Weibull and
Poisson, with literal or observed row-wise bounds and nondegenerate intervals,
on Stan. Legacy `TruncatedNormal` actually denotes censored Normal and should
be replaced by `censored`. Probability/survey weights are not implemented.

Mixtures accept same-family mapped components except Uniform, Pareto,
LocationScale, ordinal and categorical surfaces. Heterogeneous components
must have full-real continuous support. Binomial components share their
trial count. Weights are finite, nonnegative and sum to one; they may be
fixed, a standalone simplex, or a valid per-row expression. Wrappers apply
to the entire mixture. Stan supplies density, prediction and membership
probabilities; RK supplies density, gradient and pointwise likelihood but
not mixture prediction. `brm_output(d,:y; role=:membership)` identifies the
row-by-component output; draw vectors reshape in Julia column-major order.

```@eval
Main.BRMTermsReference.table(:observation)
```

## Plates, recurrences and extension points

`groups(observation_group => group_table)` maps observation rows onto a
separate group axis. `groups(g; order=t)` supplies ordered row indices to
`@plate` cells and scans. `AR1Normal` represents Gaussian residual AR(1)
within a group: a gap of k steps uses correlation `phi^k` and conditional
SD `sigma*sqrt((1-phi^(2k))/(1-phi^2))`; the first state has stationary SD
`sigma/sqrt(1-phi^2)`. Higher-order and non-Gaussian recurrences are authored
with `@scan`, not an implicit ARMA family.

A Julia helper may be called from a plate; its backend translation must be
available. Function-valued arguments, lambdas and `do` blocks can capture
sampled and data values under StanBlocks' `@deffun` surface. Names resolve
in the defining module. Reusable `@brm` components keep their own priors and
logical namespaces; see the [model-syntax guide](model-syntax.md).
Developer recipes for adding downstream grouped terms are retained in the
[research notes](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/term_extensions).
