# Complete-PLATE blueprint

!!! warning "Design catalog, not deployed API"
    This page separates the hardened PLATE implementation deployed on
    StanBlocks `devibe` from the complete target contract. Constrained matrix
    and ragged-constrained values per cell, general ragged results, arbitrary
    result-shape inference, CV/replay propagation, and optional parallel
    likelihood lowering remain target capabilities.

The purpose of this catalog is to give StanBlocks a concrete consumer contract
and give BRM one migration target. It is not a proposal to replace already-good
vectorized Stan with loops everywhere.

## Current deployment baseline

As of 2026-07-17, StanBlocks `devibe` at `73ebc3f` contains the hardened PLATE
and ragged constrained-parameter line (principally merged through `a5dc1d5`),
plus an in-source statement of the current consumer contract:

- the public one-dimensional do-block PLATE with scalar-per-cell fresh values,
  direct top-level sampling/assignment multiplexing, scalar results, and Stan
  block routing (`5de175c`);
- vector-per-cell fresh parameters and inferred scalar or vector cell results.
  A one-dimensional `vector[K]` cell is collected by column as
  `matrix[K,N]`, so cell `i` is addressed as `value[:, i]`; additional outer
  axes use array-wrapped matrices;
- trace-then-promote lowering for ordinary called `@slic` submodels, including
  inferred scalar and vector results from the traced trailing expression;
- plate-scoped slicing of one-dimensional ragged data inputs into typed vector
  cells (`3a47a63`), including typed called-cell dispatch and BridgeStan runtime;
- integer or tuple-valued N-dimensional `outer` shapes with nested Stan-block
  routing;
- hygienic fresh cell-locals across independent plates (`73ebc3f`): repeated
  source names such as `z` are namespaced by each plate result and remain
  distinct parameter carriers;
- dense native-constrained vector cells (`simplex`, `ordered`, and
  `positive_ordered`) for fixed `K` and any outer rank (`921afb1`). They emit
  native Stan constrained arrays, preserving each cell's transform and
  Jacobian;
- ragged `simplex`, `ordered`, `positive_ordered`, and Cholesky parameter
  carriers. The integration uses `devibe`'s bare free-parameter declarations
  for both ragged vector and matrix carriers. Using these carriers at runtime
  requires BridgeStan 2.9 / Stan 2.39. BRM's web and app environments resolve
  BridgeStan 2.9.0, with the requirement recorded by BRM commit `a085ecb`;
- corrected typed-LHS constraint propagation (`c672062`): `lower`, `upper`,
  `offset`, and `multiplier` on the sampling RHS now reach typed declarations.
  Earlier StanBlocks revisions could silently turn a typed half-normal into an
  unconstrained Normal.

The merge retains the public, called-submodel, N-dimensional, ragged, and
block-routing hardening coverage, and the combined tree loads cleanly. Thus
scalar-only, typed-result-only, called-submodel-unsupported, and
one-dimensional-only are no longer deployment limitations. The six-vector
Bordet hierarchy is BRM's first vector-cell migration target. General ragged
iteration/results, arbitrary shape-polymorphic collection, CV/replay taint,
and parallel likelihood lowering remain part of the complete target below.

### BRM stress verdict

The executable acceptance matrix is
[`test/plate_stress.jl`](../../test/plate_stress.jl). Run the full compiler,
`stanc`, and BridgeStan 2.9 log-density/gradient gate with:

```sh
julia --project=web-macro --startup-file=no test/plate_stress.jl
```

It is green for scalar likelihood cells, fixed correlated vector cells, dense
N-dimensional vector cells, crossed independent factors with either distinct
or reused cell-local names, one-dimensional ragged input slices, and the
heterogeneous `K[g]` result composed with top-level informative ragged Cholesky
factors and a called submodel. Dense simplex cells also pass BridgeStan with
the expected sum of per-cell free coordinates.

Expected-failure probes preserve the current boundary: matrix-valued cells,
constrained matrix and ragged-constrained cells, vararg do-block parameters,
and automatic `reduce_sum` lowering. Unsupported constrained families must
continue to fail loudly, but that guard remains only an interim safety
property: each capability closes only when its constraint-preserving carrier
transpiles, passes `stanc`, and runs with correct transforms and Jacobians in
BridgeStan.

## Assumed complete contract

All examples use the existing surface:

```julia
result ~ plate(iterable_1, iterable_2; outer=(n_cells,)) do cell_1, cell_2
    # sampling statements, observations, transformations, and submodel calls
    cell_value
end
```

A complete implementation has the following semantics. The deployed line
implements a substantial subset of items 3--6, but this list remains the
acceptance contract rather than a statement about deployed behavior.

1. Positional iterables are sliced by cell. A dense vector yields a scalar;
   an array-of-vectors or offset-backed ragged value yields one vector slice.
2. Lexically captured values are shared. This is how a cell uses population
   means, covariance factors, GP hyperparameters, or common noise parameters.
3. A fresh `~` or `=` in the body is cell-local and collected outside the
   loop with its full type. A fixed `vector[K]` cell variable becomes
   `matrix[K,N]`, with one cell per column; constrained values retain their
   constraint.
4. The trailing value is one **shape-polymorphic result**. StanBlocks infers
   its complete type from the traced expression; the user provides no output
   type, shape keyword, or declaration. Scalars, vectors, matrices,
   constrained values, and ragged vectors are all collectable, and the
   compiler chooses the corresponding outer representation.
5. A called `@slic` submodel is hygienically inlined before cell-local names
   are discovered. A called `@deffun` remains an opaque Stan function and may
   contain loops, conditionals, ODE calls, and local vector calculations.
6. The compiler splits one logical plate into the required Stan blocks:
   declarations, transformed fills, likelihood contributions, and generated
   quantities. Parameter-only plates are valid.
7. Cells are conditionally independent given shared captures. A recurrence
   across time belongs in an opaque function inside one subject/series cell;
   dependence between different cells is not a plate.
8. Likelihood-carrying plates may be lowered to `reduce_sum` without changing
   the model surface.

The shape-polymorphic trailing value is the resolved collection contract.
Named or explicitly declared multiple outputs are deliberately outside this
blueprint; the examples compose related quantities into one shaped result.

## A small cell-model vocabulary

The central simplification is that BRM can define small models for **one** cell
and let `plate` own replication, naming, declaration shapes, and Stan loops.
The names below are illustrative internal building blocks.

### Scalar latent cell

```julia
StanBlocks.@slic brm_scaled_normal_cell(log_scale) = begin
    xi ~ std_normal()
    return exp(log_scale) * xi
end
```

This is enough for random intercepts, group-specific distributional
parameters, assay-specific noise terms, and scalar latent-class parameters.

### Correlated vector cell

```julia
StanBlocks.@slic brm_correlated_cell(L, tau, n_terms) = begin
    z::vector[n_terms] ~ std_normal()
    return diag_pre_multiply(tau, L) * z
end
```

`L` and `tau` are normally shared captures. Replicating this cell yields the
row-wise coefficient matrix for arbitrary correlated random slopes and
cross-formula `|ID|` blocks.

### Positive observation-scale cell

```julia
StanBlocks.@slic brm_addprop_cell(rate) = begin
    add  ~ exponential(rate; lower=0.)
    prop ~ exponential(rate; lower=0.)
    return [add, prop]'
end
```

One vector-valued result replaces the special structured-latent path built for
Bruno's `obs_scale` matrix.

### Per-stratum constrained cells

```julia
StanBlocks.@slic brm_cholesky_cell(eta, n_terms) = begin
    L ~ lkj_corr_cholesky(eta; n=n_terms)
    return L
end

StanBlocks.@slic brm_scale_vector_cell(n_terms) = begin
    tau::vector[n_terms] ~ std_normal(; lower=0.)
    return tau
end
```

These are the constrained-value cases needed by `gr(g, by=stratum)`. Because
the resolved PLATE contract has one result, heterogeneous Cholesky and scale
values are collected by two plates and combined by the group cell that uses
them. No tuple or manual output type is required.

### Ragged subject/series cell

```julia
StanBlocks.@slic brm_subject_pk_cell(times, doses, y, L, tau, sigma) = begin
    theta ~ brm_correlated_cell(L, tau, num_elements(tau))
    amount = twocmt_superposition(times, doses, theta)
    y ~ normal(amount, addprop(amount, sigma[1], sigma[2]))
    return amount
end
```

The PLATE boundary is ragged, while the event recurrence or ODE remains inside
`twocmt_superposition`. No recursive plate is required.

## New BRM building blocks and formula functionality

| Building block | BRM surface or consumer | What complete PLATE removes |
|---|---|---|
| Generic group-local submodel | `kernel(args...; by=g)` extension seam | One emitter hook and parameter allocator per custom term |
| Scalar varying parameter | `(1 | g)`, group-specific `sigma`, `phi`, `nu`, `zi` | Bespoke gather vectors and distribution-specific group code |
| Correlated vector by group | `(1 + x + z | g)` and `(e | ID | g)` | `ranef_correlated*`, flat draw/reshape code, cross-formula draw variants |
| Stratified covariance | `(1 + x | gr(g, by=s))` | `multi_lkj_corr_cholesky_lpdf`, `multi_std_normal_lpdf`, `stratified_correlated_b` |
| Independent structured fields | Bruno `obs_scale`, per-assay add/prop noise | The general `ez6anl` structured-latent allocation floor for fixed fields |
| Grouped constrained parameter | `mo(x; by=g)`, group-specific thresholds and mixture weights | Manual flat unconstrained storage, offset tables, and constrain loops in BRM |
| Grouped basis model | `gp(x; by=g)`, grouped splines | `_sb_hsgp_by` and term-specific basis-weight floors |
| Inferred-observation cell | `me`, missing values, calibration and censoring cells | One SLIC model per observation family merely to vectorize scalar latent draws |
| Ragged longitudinal kernel | Bruno PK/PKPD and Bordet biomarker series | Prefix/gather scaffolding and one custom vector wrapper for every kernel |
| Per-cell likelihood | mixture, hurdle, zero-inflated, censored, custom user likelihoods | PLATE owns the per-cell density (`lpdf`/`lpmf`) loop today; observation posterior-predictive RNG synthesis remains target work |
| Parallel subject/series likelihood | expensive PK/PD, ODE, GP, and longitudinal models | Separate threaded model variants; the compiler may choose `reduce_sum` |
| Crossed independent plates | subject and item effects, multiple group factors | A combined monolithic random-effect allocator |

Most of these do **not** require new user-facing formula syntax. Existing BRM
terms can emit the generic cells. The important genuinely new extension is a
user-supplied group-local kernel:

```julia
@brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V  ~ 1 +          (1 | p | subject)
    pred ~ kernel(time, dose, dv, log_CL, log_V) do ts, d, yy, lCL, lV
        mu = <prediction from exp(lCL), exp(lV) over ts, d>
        yy ~ normal(mu, sigma)
        mu
    end
end
```

The public spelling has since been decided and shipped: per-subject quantities
are ordinary formula linear predictors, the cell is an inline `do`-block, and
observations are `~` statements inside it. The retired `by=` / `n_eta=` /
`model=` / `obs=` keywords are rejected loudly. The backend
contract is simply “inline this cell model under a plate keyed by the grouping
derived from those linear predictors.”

## Rewriting shipped components

The snippets below show the desired end state, not code that can be pasted into
the currently deployed StanBlocks version.

### Random intercepts

The shipped model allocates a vector explicitly and gathers it:

```julia
ranef_intercept = StanBlocks.@slic begin
    log_scale ~ std_normal()
    xi ~ std_normal(; n=n_groups)
    return exp(log_scale) * xi[group_idx]
end
```

The plate version describes one group:

```julia
ranef_intercept = StanBlocks.@slic begin
    log_scale ~ std_normal()
    b ~ plate(; outer=(n_groups,)) do g
        bg ~ brm_scaled_normal_cell(log_scale)
        bg
    end
    return b[group_idx]
end
```

The CV form changes only the outer size:

```julia
b ~ plate(; outer=(maximum(group_idx),)) do g
    bg ~ brm_scaled_normal_cell(log_scale)
    bg
end
```

A complete implementation must propagate CV taint through the plate's outer
shape so the standardized group draws move to generated quantities exactly as
`ranef_intercept_cv` does today.

### Correlated random slopes and cross-formula draws

Today `ranef_correlated` and `ranef_correlated_draws` allocate one flat draw,
reshape it, transform it, and differ only in whether they return contributions
or the coefficient matrix.

```julia
correlated_group_draws = StanBlocks.@slic begin
    L   ~ lkj_corr_cholesky(1.; n=n_terms)
    tau ~ std_normal(; n=n_terms, lower=0.)
    b ~ plate(; outer=(n_groups,)) do g
        bg ~ brm_correlated_cell(L, tau, n_terms)
        bg
    end
    return b
end

ranef_correlated = StanBlocks.@slic begin
    b ~ correlated_group_draws(; n_groups, n_terms)
    return rows_dot_product(Z, b[:, group_idx]')
end
```

One collected `matrix[n_terms,n_groups]` now serves ordinary random slopes,
`|ID|` cross-formula buckets, Bordet parameter blocks, and any future custom
group kernel. `ranef_correlated_cv` again changes only the cell count and relies
on taint propagation.

### Stratified correlated random effects

The current implementation needs three custom loop helpers: arrays of
Cholesky factors, arrays of scale vectors, and a per-group transformation.
Complete PLATE makes each level explicit:

```julia
L_by_stratum ~ plate(; outer=(n_strata,)) do s
    Ls ~ lkj_corr_cholesky(1.; n=n_terms)
    Ls
end

tau_by_stratum ~ plate(; outer=(n_strata,)) do s
    taus::vector[n_terms] ~ std_normal(; lower=0.)
    taus
end

b ~ plate(stratum_idx; outer=(n_groups,)) do s
    z::vector[n_terms] ~ std_normal()
    diag_pre_multiply(tau_by_stratum[s], L_by_stratum[s]) * z
end
```

This replaces `multi_lkj_corr_cholesky_lpdf`, `multi_std_normal_lpdf`, and
`stratified_correlated_b`, as well as both `ranef_correlated_by` return forms.

### Structured latent fields and `obs_scale`

The shipped floor converts an arbitrary elementwise prior to a flat parameter,
then reshapes it into `n_groups × n_terms`. For the concrete two-field use case:

```julia
obs_scale ~ plate(; outer=(n_assays,)) do assay
    pair ~ brm_addprop_cell(obs_scale_rate)
    pair
end
```

The logical result is one two-vector per assay, lowered as
`matrix[2,n_assays]`. General iid structured fields use the same cell with a
`vector[K]` draw. Correlated fields use
`brm_correlated_cell`. The demo consumers `sb_group_demo_slic` and
`sb_group_clamped_demo_slic` reduce to ordinary indexing of the collected
result and no longer need special emitter hooks.

### Monotonic effects and cumulative Dirichlet helpers

The global `_sb_mo`, `cdirichlet`, `c0dirichlet`, and `c01dirichlet` models are
already concise. PLATE matters when their constrained vectors vary by group:

```julia
contrasts_by_group ~ plate(alpha_by_group; outer=(n_groups,)) do alpha_g
    increments::simplex[num_elements(alpha_g)] ~ dirichlet(alpha_g)
    cumulative_sum(append_row(0., increments))
end

mo_value = ragged_gather(contrasts_by_group, group_idx, level_idx)
```

`alpha_by_group` and the collected contrasts may be ragged. The same building
block yields group-specific ordered cutpoints, positive-ordered event times,
and mixture weights by changing the constrained cell type.

### Measurement error

The shipped `_sb_me` is already only three lines, but it is the clearest routing
test for a parameter, an observation, and a returned value in every cell:

```julia
x_true ~ plate(x_obs, sd_x; outer=(num_elements(x_obs),)) do xo, sdo
    xt ~ std_normal()
    xo ~ normal(xt, sdo)
    xt
end
```

This pattern generalizes to calibration curves, interval-censored latent
predictors, and inferred covariates with non-Normal observation models.

### Horseshoe coefficients

The shipped `_sb_horseshoe` creates a scalar raw/local/global triple at every
call site. A plate can express a conventional shared global scale while keeping
coefficient-local shrinkage:

```julia
tau ~ cauchy(0., 1.; lower=0.)
beta ~ plate(; outer=(n_coefs,)) do j
    raw    ~ std_normal()
    lambda ~ cauchy(0., 1.; lower=0.)
    raw * lambda * tau
end
```

To reproduce the current behavior byte-for-byte, move `tau ~ ...` into the
body. The new version is useful because the plate makes “shared” versus
“per-coefficient” structure visible instead of relying on call-site naming.

### Zero-inflated and custom scalar likelihoods

The scalar conditional `zero_inflated_poisson_lpmf` remains a useful opaque
`@deffun`. Its hand-written vector `lpmf`, `lpmfs`, and RNG loops become
compiler-owned:

```julia
zip_loc ~ plate(y, lambda, zi; outer=(num_elements(y),)) do yi, li, pii
    yi ~ zero_inflated_poisson(li, pii)
    li
end
```

The same pattern covers hurdle families, censored observations, mixtures, and
user-defined scalar likelihoods. StanBlocks can derive the likelihood and RNG
loop from one cell statement.

### Missing-data parameters

`mi_merge` remains the right opaque scatter operation, but the missing draws in
`_sb_mi_normal` no longer require one family-specific vector parameter
declaration:

```julia
y_mis ~ plate(loc[Jmis], scale[Jmis]; outer=(num_elements(Jmis),)) do mu, sigma
    ym ~ normal(mu, sigma)
    ym
end

y_obs ~ normal(loc[Jobs], scale[Jobs])
y_complete = mi_merge(y_obs, y_mis, Jobs, Jmis, num_elements(loc))
```

BRM can generate the same cell for Poisson, binomial, ordinal, or custom
families instead of maintaining one `_sb_mi_<family>` model per argument shape.

### Grouped HSGP and splines

The population `_sb_hsgp` and `_sb_s` should remain vectorized. Their grouped
forms become one basis model per group:

```julia
f_by_group ~ plate(PHI_by_group; outer=(n_groups,)) do PHI_g
    beta_g::vector[n_basis] ~ std_normal()
    PHI_g * (sqrt_spd .* beta_g)
end

f = ragged_ungroup(f_by_group, group_idx, within_group_idx)
```

`PHI_by_group` may be dense with equal row counts or ragged. Shared `rho` and
`sigma` remain lexical captures; group-specific hyperparameters are simply
additional cell-local draws. This replaces `_sb_hsgp_by` and extends naturally
to grouped spline/tensor-product bases.

### Grouped autoregressive and state-space terms

The recurrence in `_sb_ar1` cannot itself be a plate because time points depend
on previous time points. The useful plate is one level higher:

```julia
u_by_series ~ plate(time_by_series; outer=(n_series,)) do times
    phi_raw ~ std_normal()
    epsilon::vector[num_elements(times)] ~ std_normal()
    ar1_recurse(tanh(phi_raw), epsilon, num_elements(times))
end
```

Each series is independent; the sequential recurrence remains in the opaque
`@deffun`. The same structure supports state-space, survival, and event-history
kernels with different-length series.

## Components that should not be rewritten

Complete PLATE is not a reason to de-vectorize concise global models.

| Shipped component | Keep it because |
|---|---|
| `popefs` | A single vector coefficient draw and matrix multiply are the natural Stan representation |
| `cdirichlet`, `c0dirichlet`, `c01dirichlet` | A single global simplex is already one constrained draw; PLATE only helps grouped variants |
| `_sb_mo` | The population monotonic effect is already vectorized; PLATE enables `mo(...; by=...)` |
| `_sb_cat` | Treatment coding is one coefficient vector plus a gather |
| `_sb_s` | One population spline basis is a matrix-vector multiply |
| `_sb_hsgp` | One population GP is a vector basis model; only grouped GP benefits |
| `_sb_ar1` | The time recurrence is sequential; PLATE belongs around independent series |
| `addprop` | Pure elementwise deterministic vector algebra is already concise |
| `mi_merge` | Scatter/gather mutation belongs in an opaque function, not the model loop |

## Bordet family

The shipped hierarchical-parametric representative currently obtains one
correlated six-vector for every `(biomarker, person)` series from the structured
floor, gathers six columns back to observations, then evaluates the response.

### Minimal migration: replace only the floor

```julia
L     ~ lkj_corr_cholesky(1.; n=6)
tau   ~ std_normal(; n=6, lower=0.)
theta ~ plate(; outer=(n_series,)) do series
    z::vector[6] ~ std_normal()
    diag_pre_multiply(tau, L) * z
end

# Existing vectorized Bordet response and likelihood remain unchanged.
```

The deployed `bd06936` line infers this vector result from the trailing
expression, so the untyped spelling above is now valid on `devibe`. Older
deployments required
`theta::vector[6] ~ plate(; outer=(n_series,)) do series`.

This is the first real migration unlocked by fixed-vector cell results; it does
not require ragged inputs.

### Complete migration: one ragged series cell

```julia
log_y_by_series ~ plate(
    y_by_series, time_by_series, dose_by_series, biomarker_by_series;
    outer=(n_series,),
) do ys, ts, ds, biomarker
    theta_s ~ brm_correlated_cell(L, tau, 6)
    mu_s = bordet_response(ts, ds, theta_s)
    ys ~ normal(mu_s, sigma[biomarker])
    mu_s
end
```

This eliminates `series_idxs`, repeated gathers for six parameter columns, and
prefix bookkeeping. The semi- and non-parametric variants substitute a
vector-valued basis cell. Student-t and CV variants change the likelihood or
outer taint, not the plate structure. Full correlation *between* series remains
outside the independent-cell abstraction.

## Bruno PK/PD family

Bruno's near-replicates expose three increasingly useful plate levels.

### Scalar observation maps

`_sb_gamma_time`, `_sb_biexponential_time`, and `_sb_triexponential_time`
currently share the vararg `unit_dose_effects` wrapper. A scalar-output plate
can replace its outer observation loop while the response UDF retains the
dose-event loop:

```julia
effect ~ plate(subject, time; outer=(num_elements(time),)) do s, t
    gamma_response(t, dose_times[s], shape, scale, magnitude)
end
```

The bi- and triexponential variants call their corresponding scalar response.

### Per-assay observation scales

```julia
obs_scale ~ plate(obs_scale_rate; outer=(n_assays,)) do rate
    scale_pair ~ brm_addprop_cell(rate)
    scale_pair
end
```

This replaces the special clamped `matrix[n_assays,2]` floor; consumers use the
new column-oriented result as `obs_scale[:, assay]`.

### Full subject-level PK/PD

```julia
amount_by_subject ~ plate(
    y_by_subject, times_by_subject, doses_by_subject, assay_by_subject;
    outer=(n_subjects,),
) do ys, ts, doses, assay
    theta_s ~ brm_correlated_cell(L_subject, tau_subject, 13)
    amount_s = twocmt_superposition(ts, doses, theta_s)
    ys ~ normal(amount_s,
                addprop(amount_s,
                        obs_scale[1, assay],
                        obs_scale[2, assay]))
    amount_s
end
```

The same cell can call a PD ODE or indirect-response model. Different numbers
of observations and dosing events are represented by ragged inputs, not by a
different model. QT's per-subject extension changes the cell kernel while
reusing the same plate and correlated parameter block.

## Multi-response and cross-formula models

A shared `|ID|` block becomes one vector result per group. Each regression
slices the columns it owns:

```julia
b_shared ~ plate(; outer=(n_subjects,)) do subject
    bs ~ brm_correlated_cell(L_shared, tau_shared, n_shared_terms)
    bs
end

mu1 = X1 * beta1 + rows_dot_product(Z1, b_shared[cols1, g1]')
mu2 = X2 * beta2 + rows_dot_product(Z2, b_shared[cols2, g2]')
```

This removes the need for a separate matrix-returning `*_draws` submodel. The
same collected value serves any number of linked regressions, including
Bruno's multi-assay outcomes.

## What PLATE still should not model

Even a complete implementation has a useful boundary.

- CAR, spatial, phylogenetic, and full cross-series covariance couple cells;
  they should use one global multivariate model.
- A time recurrence couples positions *inside* a series. Put it in `@deffun`
  and plate the independent series.
- Population fixed effects, single global GPs/splines, and matrix algebra
  remain vectorized.
- Data preprocessing still owns stable group coding, ragged offsets, and replay
  onto new data. PLATE consumes those structures; it does not define their
  statistical meaning.

## Migration and acceptance sequence

1. **Validate the deployed PLATE line in BRM (acceptance suite landed):** exercise untyped scalar/vector
   results, called-submodel promotion, integer and N-dimensional outer shapes,
   nested block routing, and the one-dimensional versus extra-axis layouts in
   BRM-shaped models. `test/plate_stress.jl` now gates SLIC, `stanc`, and
   BridgeStan log-density/gradient behavior and retains unsupported forms as
   expected failures.
2. **Fixed vector BRM migration:** rewrite the six-vector Bordet floor and
   generic `ranef_correlated_draws`; compare emitted parameter shapes and log
   density on the deployed implementation.
3. **Hygienic submodel acceptance:** express the same cell as a named `@slic`
   building block and verify per-call names, captures, and return substitution.
4. **Constrained cell values:** replace `ranef_correlated_by*` and the `multi_*`
   helpers with per-stratum Cholesky/scale plates.
5. **Ragged cells:** migrate grouped HSGP, Bordet series likelihoods, and Bruno
   subject kernels; verify offset/gather order under `reprocess`.
6. **Scalar likelihood families:** remove vector wrapper UDFs for
   zero-inflated, hurdle, censored, and missing-data models.
7. **Parallel lowering:** compare serial and `reduce_sum` log density for
   likelihood-carrying Bruno/Bordet plates.
8. **CV and replay:** prove that outer-size taint, generated-quantity redraws,
   new groups, and ragged offsets survive `reprocess`/`restan_data`.

The first acceptance model should be Bordet's shipped
`bordet_hierarchical_parametric`: it exercises a real correlated vector cell
without depending on the later ragged or parallel layers.
