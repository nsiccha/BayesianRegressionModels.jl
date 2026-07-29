using StanBlocks


# ==============================================================================
# SlicModel helpers (ported verbatim from /home/niko/github/nsiccha/bruno/src/qt.jl,
# `popefs`/`ranefs`/`popranefs`/`cdirichlet` family, lines 303-344). Kept here
# as module-local bindings so the walker can emit calls to them by name without
# depending on bruno. Duplication is intentional for now.
# ==============================================================================

# Formula-term stubs needed so the @brm macro can parse example formulas
# whose parser side is not yet owned by macro.jl. Empty function bindings only;
# the actual backend is implemented below (`_sb_predictor_term!`). These may
# migrate to macro.jl once the frontend agent adds shared stubs.

"""
    me(x_obs, sd_x)

brms-style measurement-error predictor marker. `me(x_obs, sd_x)` declares
that the observed `x_obs` has Gaussian measurement error with SD `sd_x`;
the backend allocates a latent `x_true` and emits the observation
likelihood `x_obs ~ Normal(x_true, sd_x)`. Dispatch tag — see `_sb_me`.
"""
function me end

"""
    s(x)

Add a penalized one-dimensional thin-plate regression spline to an `SBBRMI`
linear predictor. The fixed rank-10 basis contains the
unpenalized null space `{1, x}` and eight penalty-whitened range columns whose
shared smoothing standard deviation has a standard half-normal prior.

The public syntax is exactly one finite numeric predictor, `s(x)`. At least ten
unique training values are required; keyword arguments such as `k=` and
`knots=` are not supported. The term owns its complete smooth contribution and
does not receive an additional population coefficient.

Prediction and replay through [`reprocess`](@ref) or [`restan_data`](@ref) use
the frozen training basis by default. This marker is implemented only by the
StanBlocks backend; it is not a deterministic B-spline expansion and is not
available to `VBRMI`.

See [Formula terms](@ref) for an example and a comparison with
`bs(...)` formulas. Dispatch tag — see `_sb_s`.
"""
function s end

"""
    t2(x, z; k=(5, 5), basis=(:cr, :cr), full=false)

Tensor-product smooth marker. The StanBlocks backend builds cubic-regression-
spline margins, separates their null and penalized spaces, and gives the RR,
RN, and NR tensor blocks independent smoothing scales. The current surface is
two-dimensional, supports only `basis=(:cr, :cr)`, and requires `full=false`.
Prediction/replay freezes the training knots and penalty decomposition by
default. See [Formula terms](@ref) for the full contract. Dispatch tag — see
`_sb_t2`.
"""
function t2 end

"""
    ar(time; p=1)

Autoregressive-noise predictor marker. Adds an AR(p) noise process
ordered by `time`. Only `p=1` is supported in the current sbimpl
emitter. Dispatch tag — see `_sb_ar1`.
"""
function ar end

"""
    Horseshoe

Carvalho-Polson-Scott horseshoe shrinkage prior marker. Use as a prior
on a coefficient: `coef ~ Horseshoe()`. sbimpl emits the standard
reparameterised hierarchy `beta = raw * lambda * tau`. Marker struct
only — the `@brm` parser never constructs an instance.
"""
struct Horseshoe end

_sb_interval_literal(x::Real) = Float64(x)
_sb_interval_literal(x::NamedColumn) = begin
    parent(x) isa MissingColumn && name(x) in (:pi, :π) || error(
        "CircularVonMises: `interval` must be a compile-time numeric pair; " *
        "got formula symbol `$(name(x))`")
    Float64(pi)
end
function _sb_interval_literal(x::ExprColumn)
    args = getargs(x)
    length(args) == 1 && getf(x) === (-) && return -_sb_interval_literal(args[1])
    length(args) == 1 && getf(x) === (+) && return _sb_interval_literal(args[1])
    error("CircularVonMises: `interval` must be a compile-time numeric pair; got $(x)")
end
_sb_interval_literal(x) = error(
    "CircularVonMises: `interval` must be a compile-time numeric pair; got $(typeof(x))")

function _sb_circular_interval(kwargs)
    interval = get(kwargs, :interval, (-Float64(pi), Float64(pi)))
    interval isa Tuple && length(interval) == 2 || error(
        "CircularVonMises: `interval` must be a 2-tuple `(lo, hi)`, got $(repr(interval))")
    lo, hi = map(_sb_interval_literal, interval)
    all(isfinite, (lo, hi)) && lo < hi || error(
        "CircularVonMises: `interval` endpoints must be finite with lo < hi, got $(repr((lo, hi)))")
    isapprox(hi - lo, 2 * Float64(pi); rtol=8eps(Float64), atol=8eps(Float64)) || error(
        "CircularVonMises: `interval` must have length 2pi, got $(hi - lo)")
    (lo, hi)
end

function _check_term_kwargs(::Type{<:CircularVonMises}, kwargs)
    unknown = filter(!=(:interval), keys(kwargs))
    isempty(unknown) || error(
        "CircularVonMises: unsupported keyword(s): $(join(unknown, ", ")); " *
        "the only supported keyword is `interval`")
    _sb_circular_interval(kwargs)
    nothing
end

function _check_term_kwargs(::Type{<:Ordinal}, kwargs)
    allowed = (:discrimination, :per_threshold)
    unknown = filter(k -> k ∉ allowed, keys(kwargs))
    isempty(unknown) || error(
        "Ordinal: unsupported keyword(s): $(join(unknown, ", ")); " *
        "supported keywords are `discrimination` and `per_threshold`")
    nothing
end

"""
    interval_censored(base; upper)

Formula-only RHS wrapper for genuine interval evidence. The observed response
column supplies each interval's lower endpoint and `upper` supplies its upper
endpoint:

```julia
y_lower ~ interval_censored(Normal(mu, sigma); upper=y_upper)
```

Each row contributes the base family's probability over `(y_lower, y_upper]`,
exactly `CDF(y_upper) - CDF(y_lower)`. The lower endpoint is open for discrete
families too; unlike inclusive truncation it receives no predecessor shift.
Posterior prediction remains on the uncoarsened base-response scale. This is
separate from Distributions.jl's `censored`, which is the distribution
of `clamp(X, lower, upper)` and therefore has atoms at its thresholds.

The marker is intentionally formula-local: unlike `truncated` and `censored`,
there is no existing Distributions.jl value with these per-row evidence
semantics for BRM to construct outside `@brm`.
"""
function interval_censored end

popefs = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ std_normal(; n=n_covariates)
    return X * beta_pop
end

# Population effects with coefficient-specific Normal priors. Kept as a
# sibling of `popefs` so the default submodel (and every downstream direct use
# of it) remains byte-for-byte unchanged. The caller supplies one location and
# scale per design column; StanBlocks keeps the same prefixed `beta_pop` vector
# parameter name as the default submodel.
_popefs_normal = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ normal(beta_loc, beta_scale; n=n_covariates)
    return X * beta_pop
end

cdirichlet = StanBlocks.@slic begin
    increments ~ dirichlet(alpha)
    return cumulative_sum(increments)
end

c0dirichlet = StanBlocks.@slic begin
    increments ~ dirichlet(alpha)
    return cumulative_sum(increments) - increments[1]
end

c01dirichlet = StanBlocks.@slic begin
    increments ~ dirichlet(alpha)
    return append_row(0., cumulative_sum(increments))
end

# Monotonic effect contrast (Buerkner & Charpentier 2018). Returns the per-obs
# contrast vector; the walker hcat's it as one column of X_pop so popefs
# supplies the free beta (matches vimpl's free-beta `mo` variant, not `mo1`).
# Named `_sb_mo` to avoid clashing with vimpl's marker function `mo`.
_sb_mo = StanBlocks.@slic begin
    n_levels = maximum(x)
    simplex_incr ~ dirichlet(rep_vector(1., n_levels - 1))
    return cumulative_sum(append_row(0., simplex_incr))[x]
end

# (1 | g) random intercept. Mirrors vimpl's scalar grouped_normal + chol(n=1)
# collapse: Part{chol}(1x1) -> log_scale ~ N(0,1), L[1,1] = exp(log_scale);
# Part{grouped_normal}(n_groups, 1) -> xi ~ N(0,1), values = L[1,1] * xi;
# per-obs contribution is values[group_idx]. No LKJ needed at n=1.
#
# `n_groups` is an ordinary kwarg, so the CALLER picks the size expression and
# any cv taint rides in with it -- see the cv-contagion note below
# `ranef_correlated_draws`. That is why there is no `_cv` sibling.
ranef_intercept = StanBlocks.@slic begin
    log_scale ~ std_normal()
    xi ~ std_normal(; n=n_groups)
    return exp(log_scale) * xi[group_idx]
end

# Draw-returning sibling used by multi-membership intercepts. The caller keeps
# `group_idx=` in the declaration metadata (for ranef_blocks) but performs the
# many-to-one weighted gather with `multi_membership_intercept` below.
ranef_intercept_draws = StanBlocks.@slic begin
    log_scale ~ std_normal()
    xi ~ std_normal(; n=n_groups)
    return exp(log_scale) * xi
end

# Correlated random effects for K terms x G groups. brms-style (1 + x + y | g).
# Non-centered parameterization:
#   L      ~ lkj_corr_cholesky(1, K)         # K x K Cholesky factor
#   tau    ~ half-std_normal(; n=K)          # per-term marginal scales
#   z_flat ~ std_normal(; n=K*n_groups)      # one standardised vector
#   z      = reshape(z_flat, K, n_groups)
#   b      = (diag_pre_multiply(tau, L) * z)'   # n_groups x K correlated draws
# Per-row contribution = Z[i, :] . b[group_idx[i], :], returned as a length-n
# vector via rows_dot_product. Note: `(1 | g) + (0 + x | g)` and `(1 + x | g)`
# are equivalent -- the walker merges everything sharing a group symbol into
# one correlated block.
#
# FLAT, not a plate, and `n_groups` is an ordinary kwarg -- the same shape
# `ranef_correlated_draws` below uses, for the same two reasons (see the
# cv-contagion note there): the CALLER picks the size expression so one submodel
# serves both the ordinary and the cv-contagious case, and the flat spelling
# samples `z` in one vectorised `std_normal()` call rather than `n_groups`
# per-cell calls in a loop.
#
# This path WAS a plate (StanBlocks devibe 9210b05). It is not any more. The
# plate could not carry cv sizing -- StanBlocks' plate branch returns before the
# `cv ? :quantities : :parameter` decision, so a plate-internal fresh parameter
# never consults cv at all and fails SILENTLY into a model that still samples
# per-group effects nothing informs -- which forced a duplicate
# `ranef_correlated_cv` whose body differed from this one in exactly one size
# expression. Collapsing to flat deletes that duplicate. It also costs nothing:
# StanBlocks now hoists loop-invariants out of plate bodies itself (snag
# `benchmarked-brm-20aa0361` item 1, landed `94a71a0`), which closed about half
# of the measured plate gap, and the residual is the per-call Cholesky logdet
# that a plate cannot avoid at all (item 2, still open).
#
# EMITTED-STAN CHANGE: the sampled parameter is now `<binding>_z_flat`
# (`vector[K*n_groups]`) where it was `<binding>_b_cols_z`
# (`matrix[K, n_groups]`). Models on this path recompile, and their unconstrained
# coordinate NAMES change from `<p>.<t>.<g>` to `<p>.<i>`, i = t + (g-1)*K.
# Consumers that resolve coordinates through `ranef_blocks` / `ranef_coordinates`
# (src/prediction.jl) follow automatically; one that hardcoded the parameter name
# does not. The column-major layout is unchanged, so the VALUES are the same.
ranef_correlated = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(1.; n=n_terms)
    tau    ~ std_normal(; n=n_terms, lower=0.)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    b = (diag_pre_multiply(tau, L) * z)'   # n_groups x n_terms
    return rows_dot_product(Z, b[group_idx, :])
end

# Cross-formula correlated ranef draws for brms-style `(e | ID | g)` buckets.
# Same parameterization as `ranef_correlated` but returns the raw per-group
# matrix `b` (n_groups x n_terms) so multiple sub-formulas can each slice out
# their own column(s) and apply their own Z separately.
#
# FLAT, not a plate, and there is deliberately NO `_cv` sibling: `n_groups` is
# an ordinary kwarg, so the CALLER picks the size expression and the cv taint
# rides in with it. `_sb_emit_id_bucket_sampling!` passes the data scalar
# `n_<g>` by default and `maximum(<g>_idx)` for a group in `cv_groups`; in the
# latter case a `maybecv(:<g>_idx)` mark reaches the declared size and the whole
# block flips to a generated-quantities re-draw. One submodel serves both.
#
# A plate CANNOT serve the cv case, which is why this one is flat. StanBlocks'
# plate branch (`forward.jl`, `forward!(::SamplingExpr{Symbol,<:StanExpr})`)
# returns before the `cv ? :quantities : :parameter` decision, so a plate-
# internal fresh parameter never consults cv -- even when the plate's OUTER size
# is itself cv-tainted the promoted parameter stays in `parameters` while the
# likelihood is dropped, i.e. it fails SILENTLY into a model that still samples
# per-group effects nothing informs. Measured, not assumed. Flat is also the
# faster Stan here: one vectorised `z_flat ~ std_normal()` instead of n_groups
# per-cell calls in a loop (1.6x fewer us/gradient at n_groups=200, 3.3x at
# n_groups=1000; identical log-density and gradients to ~1e-14).
ranef_correlated_draws = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(1.; n=n_terms)
    tau    ~ std_normal(; n=n_terms, lower=0.)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    return (diag_pre_multiply(tau, L) * z)'   # n_groups x n_terms
end

# Heterogeneous marginal-SD prior used only when an `effect(sd, ID, ...)`
# statement targets a shared `|ID|` bucket. `family[i] == 0` retains BRM's
# historical half-standard-normal density for that margin; `family[i] == 1`
# selects an Exponential whose `rate[i]` is already in Stan's rate
# parameterization. A single vector density lets block defaults and
# margin-specific overrides compose without double-prioring any element.
StanBlocks.@deffun begin
    @lhs @lpxf brm_ranef_sd_lpdf(tau::vector[n], family::vector[n],
                                  rate::vector[n])::real = begin
        rv = 0.
        for i in 1:n
            if family[i] == 0
                rv += std_normal_lpdf(tau[i])::real
            else
                rv += exponential_lpdf(tau[i], rate[i])::real
            end
        end
        rv
    end
end

ranef_correlated_draws_effect = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(lkj_eta; n=n_terms)
    tau    ~ brm_ranef_sd(sd_family, sd_rate; n=n_terms, lower=0.)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    return (diag_pre_multiply(tau, L) * z)'
end

# Multi-membership gathers. Membership indices and weights are row-major flat
# vectors: entries `(i-1)*n_memberships + m` describe observation `i`, member
# slot `m`. Keeping indices flat avoids relying on a Stan array-of-int matrix
# spelling while retaining a single validated preprocessing record in Julia.
StanBlocks.@deffun begin
    multi_membership_intercept(b::vector[n_groups],
                               group_idx::int[n_mm],
                               weights::vector[n_mm],
                               n_obs::int,
                               n_memberships::int)::vector[n_obs] = begin
        rv = rep_vector(0., n_obs)
        for i in 1:n_obs
            for m in 1:n_memberships
                j = (i - 1) * n_memberships + m
                rv[i] += weights[j] * b[group_idx[j]]
            end
        end
        rv
    end
    multi_membership_correlated(Z::matrix[n_obs, n_terms],
                                b::matrix[n_groups, n_terms],
                                group_idx::int[n_mm],
                                weights::vector[n_mm],
                                n_obs::int,
                                n_memberships::int)::vector[n_obs] = begin
        rv = rep_vector(0., n_obs)
        for i in 1:n_obs
            for m in 1:n_memberships
                j = (i - 1) * n_memberships + m
                for k in 1:n_terms
                    rv[i] += weights[j] * Z[i, k] * b[group_idx[j], k]
                end
            end
        end
        rv
    end
end

# ---- cv-contagious sizing (opt-in; for out-of-sample / CV models) ------------
#
# There are NO `_cv` submodels. There used to be three; all three are gone. The
# behaviour is not: it is now a property of the CALL SITE, because every
# non-centered submodel above takes `n_groups` as an ordinary kwarg.
#
# StanBlocks' cv-contagion (see StanBlocks forward.jl:321 + types.jl:215):
# a parameter taints to a `:quantities` (generated-quantities re-draw) qual iff
# its TYPE is cv, and a type is cv iff its own flag OR *any element of its size*
# is cv. So a random effect flips to a GQ population re-draw exactly when its
# SIZE traces from a cv-marked input.
#
# The emitters therefore pass, for a group `g`:
#
#   n_groups = n_<g>              (a standalone data scalar -- no taint; the RE
#                                  stays a fitted parameter: the ordinary build)
#   n_groups = maximum(<g>_idx)   (for `g in cv_groups`; `maximum` propagates the
#                                  taint from `maybecv(:<g>_idx)` into the
#                                  declared size via passes.jl:23, flipping the
#                                  draw to a generated-quantities re-draw)
#
# `L`/`tau` keep their `n_terms` (untainted) sizing either way, so under marking
# ONLY the standardised draw is re-drawn -- a leave-all-out population re-draw
# from the fitted covariance, the semantics confirmed for QT's `source` knob.
# It stays OPT-IN via `SBBRMI(...; cv_groups=...)`; a build without it emits the
# `n_<g>` form.
#
# WHY THE DUPLICATES EXISTED, AND WHY THEY DO NOT NOW. A `_cv` sibling was only
# ever needed where the size expression could not be supplied by the caller --
# i.e. where the submodel computed it internally because it was a plate. A plate
# cannot carry cv sizing for a sharper reason than "the taint does not arrive":
# StanBlocks' `forward!(::SamplingExpr{Symbol,<:StanExpr})` (forward.jl)
# dispatches to the plate promotion and RETURNS before reaching the
# `cv ? :quantities : :parameter` decision, so a plate-internal fresh parameter
# never consults cv AT ALL. Making the plate's OUTER size cv-tainted does not
# help: the promoted parameter stays in `parameters` while the likelihood is
# dropped -- a model that still samples per-group effects nothing informs, with
# no error. Measured, not inferred. Once the non-centered paths stopped being
# plates, each duplicate collapsed into its sibling.
#
# STILL UNSUPPORTED: the typed-LHS `_by`/stratified path
# (`z::vector[n_groups, n_terms] ~ multi_std_normal()`). StanBlocks' typed-LHS
# forward (forward.jl:355) derives cv from the RHS call-args, not the declared
# size, so a cv size there yields a cv *parameter*, not a `:quantities` re-draw.
# `gr(g, by=b)` in `cv_groups` is rejected loudly rather than emitted wrong.
# Centered emission is likewise not combinable with cv sizing -- see below.

# ---- centered ranef variants (opt-in; for strong per-group likelihoods) ------
#
# The default (and `_cv`) submodels above are NON-CENTERED: the sampled
# parameter is a standardised `z ~ std_normal()` and the scale is applied
# downstream (`diag_pre_multiply(tau, L) * z`). That geometry is the right
# default for weak per-group likelihoods (few observations per level), which is
# brms' default and the shape every existing BRM model was fitted under.
#
# It is the WRONG choice when the per-group likelihood is strong -- dense
# repeated measurement per level, e.g. a PK model with many samples per subject.
# There the non-centered funnel inverts and the centered parameterization, in
# which the per-group effect IS the sampled parameter with the covariance as its
# prior, samples better. These variants emit that form:
#
#   b_cols ~ plate(; outer=(n_groups,)) do g
#       bc::vector[n_terms] ~ multi_normal_cholesky(0, diag_pre_multiply(tau, L))
#       bc
#   end
#
# They are OPT-IN per grouping factor via `SBBRMI(...; centered_groups=[:g])`:
# a group not named here is emitted by the non-centered path above, untouched by
# anything in this section. The returned VALUE is identical in distribution to
# the non-centered sibling's -- only the parameterization (and hence the
# unconstrained coordinate system) differs, so fitted draws are NOT
# interchangeable between the two.
#
# Shape: the per-group effect is declared typed-LHS as `array[n_groups]
# vector[n_terms]` and sampled with ONE vectorised `multi_normal_cholesky` call,
# rather than as a plate over a per-cell `multi_normal_cholesky`. Both transpile
# and both are stanc-clean. The reason to prefer the typed-LHS spelling is the
# Cholesky LOG-DETERMINANT: Stan evaluates `log(diagonal(L))` once per
# `multi_normal_cholesky` CALL, so `n_groups` per-cell calls pay it `n_groups`
# times where one array-vectorised call pays it once. A plate cannot express a
# vectorised MVN, so this is a floor the plate spelling cannot reach.
#
# Measured (BridgeStan us/gradient, n_terms=5, identical posterior -- gradients
# agree to 2.6e-13, log-densities differ by exactly the dropped normalisation
# constant n_groups*K/2*log(2pi)):
#
#                                     n_groups=200   n_groups=1000
#   plate, cell body written inline    198.3          859.4
#   plate, invariants hoisted          138.7          537.4
#   typed-LHS (this file)               85.0          348.7
#
# Note what the middle row says: MOST of the naive plate's disadvantage was not
# the plate concept, it was that `diag_pre_multiply(tau, L)` and `rep_vector(0.,
# n_terms)` were emitted INSIDE the per-cell loop where they are loop-invariant.
# That is FIXED UPSTREAM as of StanBlocks `94a71a0` (snag
# `benchmarked-brm-20aa0361` item 1): the emitter now hoists loop-invariants out
# of plate bodies itself, so the top row collapses onto the middle one and the
# rows above are kept only as the record of how the gap was decomposed.
# StanBlocks re-measured it at n_groups=1000 as 728.8 -> 476.9 us/grad centered,
# 626.3 -> 308.4 non-centered, both matching a hand-hoisted control exactly.
#
# The residual 1.5-1.6x is the logdet, and it is the whole justification for
# spelling this one differently from the rest of the file. It is a real cost in
# idiom uniformity, accepted deliberately, not an oversight. Closing it needs
# StanBlocks to collapse a plate whose cell body is a single `~` over a
# multivariate distribution with cell-invariant parameters into one
# array-vectorised call -- item 2 of that same snag, still open and not yet
# filed as its own row. If it lands, this submodel can go back to a plate and
# this whole comment can go with it.
#
# Scope: centered emission is NOT combinable with cv-contagious sizing. A
# centered block's sampled parameter is the per-group effect itself, which under
# either available spelling (plate cell, or the typed-LHS form below) sits behind
# the same StanBlocks gap the `_cv` comment above describes -- so there is no
# centered shape whose size can carry a cv taint. `SBBRMI` rejects the
# combination explicitly rather than silently emitting an in-sample block.

# `@stanonly` because `vector[m, n]` (an `array[m] vector[n]`) has no Julia
# emission; both helpers exist only to be called from the SLIC bodies below,
# so they are never invoked from Julia. `multi_normal_cholesky0` is the zero-mean
# array-vectorised MVN-Cholesky the typed-LHS `~` routes to (a distinctly-named
# `@lhs @lpxf` UDF, exactly as `multi_std_normal` is for `ranef_correlated_by`);
# `ranef_b_matrix` rebuilds the `n_groups × n_terms` matrix the public contract
# promises. The loop is why it is a `@deffun` -- Stan's `to_matrix` has no
# `array[] vector` overload, and `@slic` bodies cannot contain control flow.
StanBlocks.@deffun begin
    @lhs @lpxf multi_normal_cholesky0_lpdf(x::vector[m, n], scale::matrix[n, n])::real =
        multi_normal_cholesky_lpdf(x, rep_vector(0., n), scale)
    @stanonly ranef_b_matrix(b::vector[m, n])::matrix[m, n] = begin
        rv::matrix[m, n]
        for i in 1:m
            rv[i, :] = b[i]'
        end
        rv
    end
end

ranef_intercept_centered = StanBlocks.@slic begin
    log_scale ~ std_normal()
    xi ~ normal(0., exp(log_scale); n=n_groups)
    return xi[group_idx]
end

ranef_correlated_centered = StanBlocks.@slic begin
    L   ~ lkj_corr_cholesky(1.; n=n_terms)
    tau ~ std_normal(; n=n_terms, lower=0.)
    b::vector[n_groups, n_terms] ~ multi_normal_cholesky0(diag_pre_multiply(tau, L))
    bm = ranef_b_matrix(b)   # n_groups x n_terms
    return rows_dot_product(Z, bm[group_idx, :])
end

# Centered sibling of `ranef_correlated_draws` for `(e | ID | g)` buckets.
ranef_correlated_draws_centered = StanBlocks.@slic begin
    L   ~ lkj_corr_cholesky(1.; n=n_terms)
    tau ~ std_normal(; n=n_terms, lower=0.)
    b::vector[n_groups, n_terms] ~ multi_normal_cholesky0(diag_pre_multiply(tau, L))
    return ranef_b_matrix(b)   # n_groups x n_terms
end

# Configured centered blocks use a plate of native MVN-Cholesky draws. With the
# heterogeneous custom `tau` density in the same submodel, the array-vector
# `multi_normal_cholesky0` tracetype cannot recover its free outer dimension;
# the plate carries that dimension explicitly and emits the model-scale effect
# parameter as `<binding>_b_cols_bc`. Keep prediction.jl's family table aligned.
ranef_correlated_draws_centered_effect = StanBlocks.@slic begin
    L   ~ lkj_corr_cholesky(lkj_eta; n=n_terms)
    tau ~ brm_ranef_sd(sd_family, sd_rate; n=n_terms, lower=0.)
    b_cols ~ plate(; outer=(n_groups,)) do g
        bc::vector[n_terms] ~ multi_normal_cholesky(
            rep_vector(0., n_terms), diag_pre_multiply(tau, L))
        bc
    end
    return b_cols'
end

# Stratified gather + Cholesky scale, kept as a Stan function (loops are not
# allowed in @slic bodies, but they are allowed in @deffun bodies). For each
# group g, pick the stratum s = stratum_idx[g] and compute
#   b[g, :] = (diag_pre_multiply(tau[s, :], L[s, :, :]) * z[g, :])'.

# ---- group-block toy demo term -----------------------------------------------
#
# sb_group_demo: demonstrates the inverted-control group-block mechanism.
# Allocates 2 correlated per-group params via ranef_correlated_draws (proper
# 2x2 LKJ block), receives them as `group_block` (n_groups x 2 matrix), and
# returns the per-obs sum of both per-group params.
# No docstring (docstring → @deffun AssertionError gotcha; see primer).
function sb_group_demo end

sb_group_demo_slic = StanBlocks.@slic begin
    return group_block[group_idx, 1] + group_block[group_idx, 2]
end

# sb_group_clamped_demo: proves the general structured-latent floor's
# clamped / non-normal element-wise prior path (decision 10uz10q, the obs_scale
# shape). Declares a `matrix<lower=0>[n_groups, 2]` latent with an element-wise
# Exponential prior — exactly varyingsource2's `matrix<lower=0>[n_assays,2]`
# obs_scale. Not a wired consumer; it exists so a transpile probe can confirm
# the floor emits a valid positively-constrained matrix param with a non-normal
# prior. Returns the per-obs sum of both clamped per-group params.
# No docstring (docstring → @deffun AssertionError gotcha; see primer).
function sb_group_clamped_demo end

sb_group_clamped_demo_slic = StanBlocks.@slic begin
    return group_block[group_idx, 1] + group_block[group_idx, 2]
end

# Zero-inflated Poisson lpmf (per-element + vectorised). Per element:
#   y == 0 -> log_sum_exp(log(zi),  log1m(zi) + poisson_lpmf(0  | lambda))
#   y >  0 ->                       log1m(zi) + poisson_lpmf(y  | lambda)
# Defined here (not in StanBlocks/builtin.jl) so the BRM frontend can ship
# zero-inflated Poisson without an upstream addition; @deffun bodies are
# allowed control flow, which the per-element conditional needs. The
# `@lpxf` annotation on the first definition wires the SLIC sampling
# dispatch (lpxf_expr / rng_expr / likelihood_expr hooks) so
# `y ~ zero_inflated_poisson(lambda, zi)` resolves to the lpmf.
StanBlocks.@deffun begin
    @lpxf zero_inflated_poisson_lpmf(y::int, lambda::real, zi::real)::real = begin
        if y == 0
            log_sum_exp(log(zi), log1m(zi) + poisson_lpmf(0::int, lambda))
        else
            log1m(zi) + poisson_lpmf(y, lambda)
        end
    end
    zero_inflated_poisson_lpmf(y::int[n], lambda::vector[n], zi::vector[n])::real = begin
        rv = 0.
        for i in 1:n
            rv += zero_inflated_poisson_lpmf(y[i], lambda[i], zi[i])::real
        end
        rv
    end
    zero_inflated_poisson_lpmf(y::int[n], lambda::vector[n], zi::real)::real = begin
        zero_inflated_poisson_lpmf(y, lambda, rep_vector(zi, n))
    end
    zero_inflated_poisson_lpmf(y::int[n], lambda::real, zi::vector[n])::real = begin
        zero_inflated_poisson_lpmf(y, rep_vector(lambda, n), zi)
    end
    zero_inflated_poisson_lpmf(y::int[n], lambda::real, zi::real)::real = begin
        zero_inflated_poisson_lpmf(y, rep_vector(lambda, n), rep_vector(zi, n))
    end
    # Hand-rolled per-element loop returning vector[n]. Mirrors the
    # `ordered_logistic_lpmfs` pattern in StanBlocks/builtin.jl --
    # avoids `jbroadcasted` which lives in `StanBlocks.builtin` and may
    # not be reachable from a user-side @deffun's symbol resolver.
    zero_inflated_poisson_lpmfs(args...) = begin
        zero_inflated_poisson_lpmf(args...)
    end
    zero_inflated_poisson_lpmfs(y::int[n], lambda::vector[n], zi::vector[n]) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = zero_inflated_poisson_lpmf(y[i], lambda[i], zi[i])
        end
        rv
    end
    zero_inflated_poisson_lpmfs(y::int[n], lambda::vector[n], zi::real)::vector[n] = begin
        zero_inflated_poisson_lpmfs(y, lambda, rep_vector(zi, n))
    end
    zero_inflated_poisson_lpmfs(y::int[n], lambda::real, zi::vector[n])::vector[n] = begin
        zero_inflated_poisson_lpmfs(y, rep_vector(lambda, n), zi)
    end
    zero_inflated_poisson_lpmfs(y::int[n], lambda::real, zi::real)::vector[n] = begin
        zero_inflated_poisson_lpmfs(y, rep_vector(lambda, n), rep_vector(zi, n))
    end
    # Synthetic-data RNG used by SLIC's generated_quantities. Mirrors the
    # `binomial_logit_rng(int[n], …)` token-path pattern from
    # StanBlocks/builtin.jl: takes a sized int[n] token + the params,
    # writes per-element draws into the output vector.
    zero_inflated_poisson_rng(int[n], lambda::vector[n], zi::vector[n])::int[n] = begin
        rv::int[n]
        for i in 1:n
            if bernoulli_rng(zi[i]) == 1
                rv[i] = 0
            else
                rv[i] = poisson_rng(lambda[i])
            end
        end
        rv
    end
    zero_inflated_poisson_rng(int[n], lambda::vector[n], zi::real)::int[n] = begin
        rv::int[n]
        for i in 1:n
            if bernoulli_rng(zi) == 1
                rv[i] = 0
            else
                rv[i] = poisson_rng(lambda[i])
            end
        end
        rv
    end
    zero_inflated_poisson_rng(int[n], lambda::real, zi::vector[n])::int[n] = begin
        rv::int[n]
        for i in 1:n
            if bernoulli_rng(zi[i]) == 1
                rv[i] = 0
            else
                rv[i] = poisson_rng(lambda)
            end
        end
        rv
    end
    zero_inflated_poisson_rng(int[n], lambda::real, zi::real)::int[n] = begin
        rv::int[n]
        for i in 1:n
            if bernoulli_rng(zi) == 1
                rv[i] = 0
            else
                rv[i] = poisson_rng(lambda)
            end
        end
        rv
    end
end

# One internal Stan family serves the public typed ordinal composition. The
# integer tags are emitted compile-time literals (structure: cumulative=1,
# stopping-ratio=2; link: logit=1, probit=2, cloglog=3). Keeping the tags inside
# one private UDF avoids a Cartesian product of public pair names while still
# giving StanBlocks the complete lpmf / pointwise-lpmf / RNG triad.
#
# For cumulative-logit the scalar density delegates to Stan's native
# ordered_logistic_lpmf after applying discrimination to both eta and the
# cutpoints. Probit and cloglog use their exact native CDF/log-CDF primitives.
# Stopping-ratio has no native Stan distribution, so its sequential hazards are
# accumulated directly in log space.
StanBlocks.@deffun begin
    @stanonly brm_ordinal_logcdf(z::real, link::int)::real = begin
        if link == 1
            log_inv_logit(z)
        else
            if link == 2
                normal_lcdf(z, 0., 1.)
            else
                log1m_exp(-exp(z))
            end
        end
    end
    @stanonly brm_ordinal_logccdf(z::real, link::int)::real = begin
        if link == 1
            log_inv_logit(-z)
        else
            if link == 2
                normal_lccdf(z, 0., 1.)
            else
                -exp(z)
            end
        end
    end
    @stanonly brm_ordinal_cdf(z::real, link::int)::real = begin
        if link == 1
            inv_logit(z)
        else
            if link == 2
                Phi(z)
            else
                -expm1(-exp(z))
            end
        end
    end

    @stanonly @lpxf brm_ordinal_lpmf(y::int, eta::real, thresholds::vector[k],
                           discrimination::real, structure::int, link::int,
                           threshold_effect::vector[k])::real = begin
        K = k + 1
        if discrimination <= 0.
            negative_infinity()
        else
            if y < 1
                negative_infinity()
            else
                if y > K
                    negative_infinity()
                else
                    if structure == 1
                        if link == 1
                            ordered_logistic_lpmf(
                                y,
                                discrimination * eta,
                                discrimination .* thresholds,
                            )
                        else
                            if y == 1
                                z_first = discrimination * (thresholds[1] - eta)
                                brm_ordinal_logcdf(z_first, link)
                            else
                                if y == K
                                    z_last = discrimination * (thresholds[k] - eta)
                                    brm_ordinal_logccdf(z_last, link)
                                else
                                    z_hi = discrimination * (thresholds[y] - eta)
                                    z_lo = discrimination * (thresholds[y - 1] - eta)
                                    log_diff_exp(
                                        brm_ordinal_logcdf(z_hi, link),
                                        brm_ordinal_logcdf(z_lo, link),
                                    )
                                end
                            end
                        end
                    else
                        rv = 0.
                        for j in 1:k
                            z_stage = discrimination *
                                (thresholds[j] - eta - threshold_effect[j])
                            if j < y
                                rv += brm_ordinal_logccdf(z_stage, link)
                            else
                                if j == y
                                    rv += brm_ordinal_logcdf(z_stage, link)
                                end
                            end
                        end
                        rv
                    end
                end
            end
        end
    end

    @stanonly brm_ordinal_lpmf(y::int[n], eta::vector[n], thresholds::vector[k],
                     discrimination::vector[n], structure::int, link::int,
                     threshold_effect::matrix[n,k])::real = begin
        rv = 0.
        for i in 1:n
            rv += brm_ordinal_lpmf(
                y[i], eta[i], thresholds, discrimination[i], structure, link,
                to_vector(threshold_effect[i, :]),
            )
        end
        rv
    end
    @stanonly brm_ordinal_lpmf(y::int[n], eta::vector[n], thresholds::vector[k],
                     discrimination::real, structure::int, link::int,
                     threshold_effect::matrix[n,k])::real = begin
        brm_ordinal_lpmf(
            y, eta, thresholds, rep_vector(discrimination, n), structure, link,
            threshold_effect,
        )
    end
    @stanonly brm_ordinal_lpmf(y::int[n], eta::real, thresholds::vector[k],
                     discrimination::vector[n], structure::int, link::int,
                     threshold_effect::matrix[n,k])::real = begin
        brm_ordinal_lpmf(
            y, rep_vector(eta, n), thresholds, discrimination, structure, link,
            threshold_effect,
        )
    end
    @stanonly brm_ordinal_lpmf(y::int[n], eta::real, thresholds::vector[k],
                     discrimination::real, structure::int, link::int,
                     threshold_effect::matrix[n,k])::real = begin
        brm_ordinal_lpmf(
            y, rep_vector(eta, n), thresholds, rep_vector(discrimination, n),
            structure, link, threshold_effect,
        )
    end

    @stanonly brm_ordinal_lpmfs(args...) = begin
        brm_ordinal_lpmf(args...)
    end
    @stanonly brm_ordinal_lpmfs(y::int[n], eta::vector[n], thresholds::vector[k],
                      discrimination::vector[n], structure::int, link::int,
                      threshold_effect::matrix[n,k])::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_ordinal_lpmf(
                y[i], eta[i], thresholds, discrimination[i], structure, link,
                to_vector(threshold_effect[i, :]),
            )
        end
        rv
    end
    @stanonly brm_ordinal_lpmfs(y::int[n], eta::vector[n], thresholds::vector[k],
                      discrimination::real, structure::int, link::int,
                      threshold_effect::matrix[n,k])::vector[n] = begin
        brm_ordinal_lpmfs(
            y, eta, thresholds, rep_vector(discrimination, n), structure, link,
            threshold_effect,
        )
    end
    @stanonly brm_ordinal_lpmfs(y::int[n], eta::real, thresholds::vector[k],
                      discrimination::vector[n], structure::int, link::int,
                      threshold_effect::matrix[n,k])::vector[n] = begin
        brm_ordinal_lpmfs(
            y, rep_vector(eta, n), thresholds, discrimination, structure, link,
            threshold_effect,
        )
    end
    @stanonly brm_ordinal_lpmfs(y::int[n], eta::real, thresholds::vector[k],
                      discrimination::real, structure::int, link::int,
                      threshold_effect::matrix[n,k])::vector[n] = begin
        brm_ordinal_lpmfs(
            y, rep_vector(eta, n), thresholds, rep_vector(discrimination, n),
            structure, link, threshold_effect,
        )
    end

    @stanonly brm_ordinal_rng(eta::real, thresholds::vector[k], discrimination::real,
                    structure::int, link::int,
                    threshold_effect::vector[k])::int = begin
        K = k + 1
        rv = K
        if structure == 1
            u = uniform_rng(0., 1.)
            for j in 1:k
                if rv == K
                    z_cumulative = discrimination * (thresholds[j] - eta)
                    if u <= brm_ordinal_cdf(z_cumulative, link)
                        rv += j - rv
                    end
                end
            end
        else
            for j in 1:k
                if rv == K
                    z_stopping = discrimination *
                        (thresholds[j] - eta - threshold_effect[j])
                    if bernoulli_rng(brm_ordinal_cdf(z_stopping, link)) == 1
                        rv += j - rv
                    end
                end
            end
        end
        rv
    end
    @stanonly brm_ordinal_rng(int[n], eta::vector[n], thresholds::vector[k],
                    discrimination::vector[n], structure::int, link::int,
                    threshold_effect::matrix[n,k])::int[n] = begin
        rv::int[n]
        for i in 1:n
            rv[i] = brm_ordinal_rng(
                eta[i], thresholds, discrimination[i], structure, link,
                to_vector(threshold_effect[i, :]),
            )
        end
        rv
    end
    @stanonly brm_ordinal_rng(int[n], eta::vector[n], thresholds::vector[k],
                    discrimination::real, structure::int, link::int,
                    threshold_effect::matrix[n,k])::int[n] = begin
        brm_ordinal_rng(
            int[n], eta, thresholds, rep_vector(discrimination, n), structure,
            link, threshold_effect,
        )
    end
    @stanonly brm_ordinal_rng(int[n], eta::real, thresholds::vector[k],
                    discrimination::vector[n], structure::int, link::int,
                    threshold_effect::matrix[n,k])::int[n] = begin
        brm_ordinal_rng(
            int[n], rep_vector(eta, n), thresholds, discrimination, structure,
            link, threshold_effect,
        )
    end
    @stanonly brm_ordinal_rng(int[n], eta::real, thresholds::vector[k],
                    discrimination::real, structure::int, link::int,
                    threshold_effect::matrix[n,k])::int[n] = begin
        brm_ordinal_rng(
            int[n], rep_vector(eta, n), thresholds,
            rep_vector(discrimination, n), structure, link, threshold_effect,
        )
    end
end

# Native-Stan von-Mises density with the two public BRM support contracts made
# explicit. `principal == 0` is Distributions.jl's `VonMises`: moving inclusive
# support `[mu - pi, mu + pi]`. `principal == 1` is `CircularVonMises`: fixed
# half-open support `[lo, hi)`, with `mu` and generated draws wrapped into it.
# The scalar density always delegates its in-support value to Stan's native
# `von_mises_lpdf`; the wrappers only add support/domain semantics.
StanBlocks.@deffun begin
    @lpxf brm_von_mises_lpdf(y::real, mu::real, kappa::real,
                             lo::real, hi::real, principal::int)::real = begin
        if kappa <= 0.
            negative_infinity()
        else
            if principal == 1
                if y < lo
                    negative_infinity()
                else
                    if y >= hi
                        negative_infinity()
                    else
                        wrapped_mu = lo + fmod(fmod(mu - lo, hi - lo) + hi - lo, hi - lo)
                        von_mises_lpdf(y, wrapped_mu, kappa)
                    end
                end
            else
                if y < mu - 3.141592653589793
                    negative_infinity()
                else
                    if y > mu + 3.141592653589793
                        negative_infinity()
                    else
                        von_mises_lpdf(y, mu, kappa)
                    end
                end
            end
        end
    end
    brm_von_mises_lpdf(y::vector[n], mu::vector[n], kappa::vector[n],
                       lo::real, hi::real, principal::int)::real = begin
        rv = 0.
        for i in 1:n
            rv += brm_von_mises_lpdf(y[i], mu[i], kappa[i], lo, hi, principal)::real
        end
        rv
    end
    brm_von_mises_lpdf(y::vector[n], mu::vector[n], kappa::real,
                       lo::real, hi::real, principal::int)::real = begin
        brm_von_mises_lpdf(y, mu, rep_vector(kappa, n), lo, hi, principal)
    end
    brm_von_mises_lpdf(y::vector[n], mu::real, kappa::vector[n],
                       lo::real, hi::real, principal::int)::real = begin
        brm_von_mises_lpdf(y, rep_vector(mu, n), kappa, lo, hi, principal)
    end
    brm_von_mises_lpdf(y::vector[n], mu::real, kappa::real,
                       lo::real, hi::real, principal::int)::real = begin
        brm_von_mises_lpdf(y, rep_vector(mu, n), rep_vector(kappa, n), lo, hi, principal)
    end

    brm_von_mises_lpdfs(args...) = begin
        brm_von_mises_lpdf(args...)
    end
    brm_von_mises_lpdfs(y::vector[n], mu::vector[n], kappa::vector[n],
                        lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_lpdf(y[i], mu[i], kappa[i], lo, hi, principal)
        end
        rv
    end
    brm_von_mises_lpdfs(y::vector[n], mu::vector[n], kappa::real,
                        lo::real, hi::real, principal::int)::vector[n] = begin
        brm_von_mises_lpdfs(y, mu, rep_vector(kappa, n), lo, hi, principal)
    end
    brm_von_mises_lpdfs(y::vector[n], mu::real, kappa::vector[n],
                        lo::real, hi::real, principal::int)::vector[n] = begin
        brm_von_mises_lpdfs(y, rep_vector(mu, n), kappa, lo, hi, principal)
    end
    brm_von_mises_lpdfs(y::vector[n], mu::real, kappa::real,
                        lo::real, hi::real, principal::int)::vector[n] = begin
        brm_von_mises_lpdfs(y, rep_vector(mu, n), rep_vector(kappa, n), lo, hi, principal)
    end

    brm_von_mises_rng(mu::real, kappa::real,
                      lo::real, hi::real, principal::int)::real = begin
        if kappa <= 0.
            reject("brm_von_mises_rng: kappa must be strictly positive")
            0.
        else
            draw = von_mises_rng(mu, kappa)
            if principal == 1
                lo + fmod(fmod(draw - lo, hi - lo) + hi - lo, hi - lo)
            else
                support_lo = mu - 3.141592653589793
                support_lo + fmod(fmod(draw - support_lo, 6.283185307179586) +
                                  6.283185307179586, 6.283185307179586)
            end
        end
    end
    brm_von_mises_rng(vector[n], mu::vector[n], kappa::vector[n],
                      lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_rng(mu[i], kappa[i], lo, hi, principal)
        end
        rv
    end
    brm_von_mises_rng(vector[n], mu::vector[n], kappa::real,
                      lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_rng(mu[i], kappa, lo, hi, principal)
        end
        rv
    end
    brm_von_mises_rng(vector[n], mu::real, kappa::vector[n],
                      lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_rng(mu, kappa[i], lo, hi, principal)
        end
        rv
    end
    brm_von_mises_rng(vector[n], mu::real, kappa::real,
                      lo::real, hi::real, principal::int)::vector[n] = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = brm_von_mises_rng(mu, kappa, lo, hi, principal)
        end
        rv
    end
end

function addprop end

StanBlocks.@deffun begin
    addprop(loc::vector[n], add::real, prop::real)::vector[n] = begin
        sqrt(add^2 .+ (loc .* prop).^2)
    end
end

StanBlocks.@deffun begin
    stratified_correlated_b(L, tau, z, stratum_idx::int[n_groups],
                            n_groups::int, n_terms::int) = begin
        b = rep_matrix(0., n_groups, n_terms)
        for g in 1:n_groups
            b[g, :] = (diag_pre_multiply(tau[stratum_idx[g], :],
                                         L[stratum_idx[g], :, :]) * z[g, :])'
        end
        b
    end
end

# `(expr | gr(g, by=b))` stratified random effects: independent LKJ-Cholesky +
# tau per level of `b`, so each stratum has its own full covariance structure.
# `stratum_idx[g]` maps each group-level to its stratum (walker pre-computes it
# and errors if any group straddles strata).
#   L   :: array[n_strata] cholesky_factor_corr[n_terms]
#   tau :: array[n_strata] vector<lower=0>[n_terms]
#   z   :: array[n_groups] vector[n_terms]
# Per-group contribution: b[g, :] = (diag_pre_multiply(tau[s], L[s]) * z[g])'
# where s = stratum_idx[g]. The per-group loop lives in
# `stratified_correlated_b` (a @deffun helper) because @slic bodies cannot
# contain control flow.
# Distinctly-named 2-arg `@lhs @lpxf` UDFs so typed-LHS sampling routes to a
# user-defined Stan function without clashing with Stan's scalar-only 
# built-ins. `@lpxf` creates the base stub + `lpxf_expr` hook so
# `L ~ multi_lkj_corr_cholesky(1.)` resolves; `@lhs` registers the base
# tracetype so the 2-arg call dispatches to this lpdf with `m, n` bound from
# the declared LHS shape.
StanBlocks.@deffun begin
    @lhs @lpxf multi_lkj_corr_cholesky_lpdf(L::cholesky_factor_corr[m, n], x::real)::real = begin
        rv = 0.
        for i in 1:m
            rv += lkj_corr_cholesky_lpdf(L[i, :, :], x)::real
        end 
        rv
    end
    @lhs @lpxf multi_std_normal_lpdf(x::vector[m, n])::real = begin
        rv = 0.
        for i in 1:m
            rv += std_normal_lpdf(x[i, :])::real
        end
        rv
    end
end

ranef_correlated_by = StanBlocks.@slic begin
    L::cholesky_factor_corr[n_strata, n_terms] ~ multi_lkj_corr_cholesky(1.)
    tau::vector[n_strata, n_terms] ~ multi_std_normal(; lower=0.)
    # z is the standardised per-GROUP draw (n_groups rows), not per-stratum --
    # `stratified_correlated_b` indexes z[g, :] for g in 1:n_groups.
    z::vector[n_groups, n_terms] ~ multi_std_normal()
    b = stratified_correlated_b(L, tau, z, stratum_idx, n_groups, n_terms)
    return rows_dot_product(Z, b[group_idx, :])
end

# Cross-formula stratified correlated ranef draws for brms-style
# `(e | ID | gr(g, by=b))` buckets. Matrix-returning variant of
# `ranef_correlated_by` so each sub-formula can slice its own column(s).
ranef_correlated_by_draws = StanBlocks.@slic begin
    L::cholesky_factor_corr[n_strata, n_terms] ~ multi_lkj_corr_cholesky(1.)
    tau::vector[n_strata, n_terms] ~ multi_std_normal(; lower=0.)
    z::vector[n_groups, n_terms] ~ multi_std_normal()
    return stratified_correlated_b(L, tau, z, stratum_idx, n_groups, n_terms)
end

# Treatment-coded categorical predictor. Allocates K-1 free betas; reference
# level 1 contributes 0. Mirrors vimpl's `AbstractVector{<:Integer}` dispatch.
# `x` is the per-row 1-based level index, `n_levels = K`.
_sb_cat = StanBlocks.@slic begin
    beta ~ std_normal(; n=n_levels - 1)
    return append_row(0., beta)[x]
end

# Minimal `ar(time, p=1)` autoregressive submodel. Adds an AR(1) noise process
# `u[t] = phi * u[t-1] + epsilon[t]` (with `u[1] = epsilon[1]`; no stationary
# init) to the linear predictor. `phi` is parameterized via `tanh(phi_raw)` so
# it stays in (-1, 1) under a `std_normal` prior on `phi_raw`. Rows are
# assumed to already be in time order; the `time` arg is used only as a
# length probe (explicit sort by time is a follow-up). Only `p=1` is
# supported for the first pass.
StanBlocks.@deffun begin
    ar1_recurse(phi::real, epsilon::vector[n], n::int)::vector[n] = begin
        u = rep_vector(0., n)
        u[1] = epsilon[1]
        for t in 2:n
            u[t] = phi * u[t-1] + epsilon[t]
        end
        u
    end
end

_sb_ar1 = StanBlocks.@slic begin
    n_obs = num_elements(time)
    phi_raw ~ std_normal()
    phi = tanh(phi_raw)
    epsilon ~ std_normal(; n=n_obs)
    return ar1_recurse(phi, epsilon, n_obs)
end

# Penalized 1-D thin-plate regression spline. `Xnull` contains the unpenalized
# polynomial null space {1, x}; `Zpen` is the range-space basis after the
# wiggliness penalty has been diagonalized and absorbed into the columns. The
# standardized range coefficients therefore have one iid Gaussian scale `sds`,
# matching the mixed-model parameterization used by mgcv/brms. The caller adds
# the resulting length-N contribution directly to the linear predictor.
_sb_s = StanBlocks.@slic begin
    n_pen = dims(Zpen)[2]
    b_fixed::vector[2]
    sds ~ std_normal(; lower=0.)
    b_pen_raw ~ std_normal(; n=n_pen)
    b_pen = sds * b_pen_raw
    return Xnull * b_fixed + Zpen * b_pen
end

# Two-margin tensor-product smooth. With cubic-regression-spline margins each
# null space has dimension two. Removing the tensor intercept leaves three
# unpenalized NN columns. The remaining RR, RN, and NR blocks correspond to the
# three penalties used by mgcv/brms `t2(..., full=FALSE)` and deliberately get
# distinct smoothing scales.
_sb_t2 = StanBlocks.@slic begin
    n_rr = dims(Zrr)[2]
    n_rn = dims(Zrn)[2]
    n_nr = dims(Znr)[2]
    b_fixed::vector[3]
    sd_rr ~ std_normal(; lower=0.)
    sd_rn ~ std_normal(; lower=0.)
    sd_nr ~ std_normal(; lower=0.)
    b_rr_raw ~ std_normal(; n=n_rr)
    b_rn_raw ~ std_normal(; n=n_rn)
    b_nr_raw ~ std_normal(; n=n_nr)
    b_rr = sd_rr * b_rr_raw
    b_rn = sd_rn * b_rn_raw
    b_nr = sd_nr * b_nr_raw
    return Xfixed * b_fixed + Zrr * b_rr + Zrn * b_rn + Znr * b_nr
end

# Fit/apply split for Wood's rank-k thin-plate regression spline (d=1, m=2).
# The radial kernel is eta(r)=r^3/12 (Wood 2003, eq. 7). We keep the k largest-
# magnitude eigenvectors of the full kernel matrix, impose the T' * delta = 0
# side constraint, and diagonalize the resulting range-space penalty. Applying
# the fitted object to new x values needs only the frozen training centers,
# shift, and penalty-whitened range projection.
function _sb_tps_kernel(x::AbstractVector{<:Real}, centers::AbstractVector{<:Real})
    E = Matrix{Float64}(undef, length(x), length(centers))
    for j in eachindex(centers), i in eachindex(x)
        E[i, j] = abs(Float64(x[i]) - Float64(centers[j]))^3 / 12
    end
    E
end

function _sb_fit_spline(x::AbstractVector{<:Real}; k::Int=10)
    k > 2 || error("sbimpl: `s(x)` needs basis dimension k > 2 (got $k)")
    xs = collect(Float64, x)
    all(isfinite, xs) || error("sbimpl: `s(x)` requires finite numeric data")
    length(unique(xs)) >= k || error(
        "sbimpl: `s(x)` needs at least $k unique x values for the default ",
        "thin-plate basis (got $(length(unique(xs))))")

    shift = sum(xs) / length(xs)
    centers = xs .- shift
    E = _sb_tps_kernel(centers, centers)
    eig_E = eigen(Symmetric(E))
    keep = sortperm(abs.(eig_E.values); rev=true)[1:k]
    U = eig_E.vectors[:, keep]
    D = eig_E.values[keep]

    T = hcat(ones(Float64, length(xs)), centers)
    Z = nullspace(transpose(T) * U)
    size(Z, 2) == k - 2 || error(
        "sbimpl: `s(x)` could not isolate the two-dimensional TPS null space")

    S = Symmetric(transpose(Z) * Diagonal(D) * Z)
    eig_S = eigen(S)
    penalty_scale = maximum(abs, eig_S.values)
    penalty_scale > 0 || error("sbimpl: `s(x)` produced a zero range-space penalty")
    tol = penalty_scale * eps(Float64) * 100
    minimum(eig_S.values) >= -tol || error(
        "sbimpl: `s(x)` produced a non-positive range-space penalty")
    penalty_values = max.(eig_S.values, tol)
    penalty_whitener = eig_S.vectors * Diagonal(inv.(sqrt.(penalty_values)))
    range_projection = U * Z * penalty_whitener

    (; shift, centers, range_projection, k)
end

function _sb_apply_spline(fit, x::AbstractVector{<:Real})
    xs = collect(Float64, x)
    all(isfinite, xs) || error("sbimpl: `s(x)` requires finite numeric data")
    centered = xs .- fit.shift
    Xnull = hcat(ones(Float64, length(xs)), centered)
    Zpen = _sb_tps_kernel(centered, fit.centers) * fit.range_projection
    Xnull, Zpen
end

_sb_spline_basis_tps(x::AbstractVector{<:Real}; k::Int=10) =
    _sb_apply_spline(_sb_fit_spline(x; k), x)

# Cubic regression spline margin used by `t2`. Knots follow R's default
# quantile algorithm (type 7) over the sorted unique training values. `F` maps
# knot values to the natural cubic spline's second derivatives, while `S` is
# the integrated-squared-second-derivative penalty. The positive eigenspace of
# `S` is penalty-whitened without mixing in the null space, so tensoring the
# marginal range/null pieces preserves the three `t2(full=false)` penalties.
function _sb_type7_knots(x::AbstractVector{<:Real}, k::Int)
    values = sort!(unique(collect(Float64, x)))
    length(values) >= k || error(
        "sbimpl: `t2` margin needs at least $k unique values (got $(length(values)))")
    n = length(values)
    knots = Vector{Float64}(undef, k)
    for i in 1:k
        pos = 1 + (n - 1) * (i - 1) / (k - 1)
        lo = clamp(floor(Int, pos), 1, n)
        hi = clamp(ceil(Int, pos), 1, n)
        weight = pos - lo
        knots[i] = (1 - weight) * values[lo] + weight * values[hi]
    end
    all(diff(knots) .> 0) || error(
        "sbimpl: `t2` margin produced non-distinct cubic-regression-spline knots")
    knots
end

function _sb_cr_second_derivative_map(knots::AbstractVector{<:Real})
    k = length(knots)
    h = diff(knots)
    all(h .> 0) || error("sbimpl: `t2` cubic-regression-spline knots must increase")
    D = zeros(Float64, k - 2, k)
    B = zeros(Float64, k - 2, k - 2)
    for i in 1:(k - 2)
        D[i, i] = inv(h[i])
        D[i, i + 1] = -inv(h[i]) - inv(h[i + 1])
        D[i, i + 2] = inv(h[i + 1])
        B[i, i] = (h[i] + h[i + 1]) / 3
        if i < k - 2
            B[i, i + 1] = h[i + 1] / 6
            B[i + 1, i] = B[i, i + 1]
        end
    end
    interior = B \ D
    F = zeros(Float64, k, k)
    F[2:(k - 1), :] .= interior
    F, transpose(D) * interior
end

function _sb_cr_basis(knots, F, x::AbstractVector{<:Real})
    k = length(knots)
    X = zeros(Float64, length(x), k)
    for (i, raw_x) in enumerate(x)
        xi = Float64(raw_x)
        if xi < knots[1]
            h = knots[2] - knots[1]
            xik = xi - knots[1]
            cjm = -xik * h / 3
            cjp = -xik * h / 6
            for q in 1:k
                X[i, q] = cjm * F[1, q] + cjp * F[2, q]
            end
            X[i, 1] += 1 - xik / h
            X[i, 2] += xik / h
        elseif xi > knots[k]
            h = knots[k] - knots[k - 1]
            xik = xi - knots[k]
            cjm = xik * h / 6
            cjp = xik * h / 3
            for q in 1:k
                X[i, q] = cjm * F[k - 1, q] + cjp * F[k, q]
            end
            X[i, k - 1] -= xik / h
            X[i, k] += 1 + xik / h
        else
            j = clamp(searchsortedlast(knots, xi), 1, k - 1)
            h = knots[j + 1] - knots[j]
            ajm = knots[j + 1] - xi
            ajp = xi - knots[j]
            cjm = ajm * (ajm * ajm / h - h) / 6
            cjp = ajp * (ajp * ajp / h - h) / 6
            for q in 1:k
                X[i, q] = cjm * F[j, q] + cjp * F[j + 1, q]
            end
            X[i, j] += ajm / h
            X[i, j + 1] += ajp / h
        end
    end
    X
end

function _sb_fit_cr_spline(x::AbstractVector{<:Real}; k::Int=5)
    k > 2 || error("sbimpl: `t2` basis dimensions must be integers greater than 2 (got $k)")
    xs = collect(Float64, x)
    isempty(xs) && error("sbimpl: `t2` cannot use an empty margin")
    all(isfinite, xs) || error("sbimpl: `t2` margins require finite numeric data")
    shift = sum(xs) / length(xs)
    scale = maximum(xs) - minimum(xs)
    scale > 0 || error("sbimpl: `t2` margin is degenerate (all values equal)")
    normalized = (xs .- shift) ./ scale
    knots = _sb_type7_knots(normalized, k)
    F, penalty = _sb_cr_second_derivative_map(knots)

    eig_penalty = eigen(Symmetric(penalty))
    order = sortperm(eig_penalty.values; rev=true)
    keep = order[1:(k - 2)]
    penalty_scale = maximum(abs, eig_penalty.values)
    tol = penalty_scale * eps(Float64) * 100
    minimum(eig_penalty.values) >= -tol || error(
        "sbimpl: `t2` cubic-regression-spline penalty is not positive semidefinite")
    minimum(eig_penalty.values[keep]) > tol || error(
        "sbimpl: `t2` could not isolate the two-dimensional marginal null space")
    range_projection = eig_penalty.vectors[:, keep] *
                       Diagonal(inv.(sqrt.(eig_penalty.values[keep])))

    null_const_scale = inv(sqrt(length(xs)))
    slope_norm = norm(normalized)
    slope_norm > 0 || error("sbimpl: `t2` margin has a zero linear null-space norm")

    (; shift, scale, knots, F, range_projection, null_const_scale, slope_norm, k)
end

function _sb_apply_cr_spline(fit, x::AbstractVector{<:Real})
    xs = collect(Float64, x)
    all(isfinite, xs) || error("sbimpl: `t2` margins require finite numeric data")
    normalized = (xs .- fit.shift) ./ fit.scale
    Xnull = hcat(fill(fit.null_const_scale, length(xs)),
                 normalized ./ fit.slope_norm)
    range = _sb_cr_basis(fit.knots, fit.F, normalized) * fit.range_projection
    Xnull, range
end

function _sb_row_tensor(A::AbstractMatrix, B::AbstractMatrix)
    size(A, 1) == size(B, 1) || error(
        "sbimpl: `t2` marginal basis row counts differ ($(size(A, 1)) vs $(size(B, 1)))")
    out = Matrix{Float64}(undef, size(A, 1), size(A, 2) * size(B, 2))
    for i in axes(out, 1), a in axes(A, 2), b in axes(B, 2)
        out[i, (a - 1) * size(B, 2) + b] = A[i, a] * B[i, b]
    end
    out
end

function _sb_t2_raw_blocks(margins, x, z)
    N1, R1 = _sb_apply_cr_spline(margins[1], x)
    N2, R2 = _sb_apply_cr_spline(margins[2], z)
    NN = _sb_row_tensor(N1, N2)
    (fixed=Matrix(NN[:, 2:end]), rr=_sb_row_tensor(R1, R2),
     rn=_sb_row_tensor(R1, N2), nr=_sb_row_tensor(N1, R2))
end

_sb_block_center(A::AbstractMatrix) = vec(sum(A; dims=1)) ./ size(A, 1)
_sb_center_block(A::AbstractMatrix, center) = A .- reshape(center, 1, :)

function _sb_fit_t2(x::AbstractVector{<:Real}, z::AbstractVector{<:Real};
                    k::Tuple{Int,Int}=(5, 5))
    length(x) == length(z) || error(
        "sbimpl: `t2(x, z)` margins must have equal lengths ($(length(x)) vs $(length(z)))")
    margins = (_sb_fit_cr_spline(x; k=k[1]), _sb_fit_cr_spline(z; k=k[2]))
    raw = _sb_t2_raw_blocks(margins, x, z)
    fixed_center = _sb_block_center(raw.fixed)
    (; margins, fixed_center, k)
end

function _sb_apply_t2(fit, x::AbstractVector{<:Real}, z::AbstractVector{<:Real})
    length(x) == length(z) || error(
        "sbimpl: `t2(x, z)` margins must have equal lengths ($(length(x)) vs $(length(z)))")
    raw = _sb_t2_raw_blocks(fit.margins, x, z)
    (_sb_center_block(raw.fixed, fit.fixed_center), raw.rr, raw.rn, raw.nr)
end

# brms-style `me(x_obs, sd_x)` measurement-error predictor. The submodel
# allocates a length-N latent `x_true` vector with prior `std_normal` and
# emits the observation likelihood `x_obs ~ normal(x_true, sd_x)` directly.
# (Earlier StanBlocks versions silently dropped data-LHS `~` inside submodel
# bodies; that's fixed, so we keep the likelihood self-contained.)
# The linear predictor uses `x_true` via popefs's free beta, so `me` behaves
# like a regular continuous covariate except the predictor values themselves
# are parameters.
_sb_me = StanBlocks.@slic begin
    x_true ~ std_normal(; n=num_elements(x_obs))
    x_obs ~ normal(x_true, sd_x)
    return x_true
end

# Carvalho-Polson-Scott horseshoe prior, scalar form. Standard
# reparameterisation: beta = raw * lambda * tau with raw ~ N(0,1) and
# half-Cauchy(0,1) local + global scales. Each `coef ~ Horseshoe()`
# call site gets its own (raw, lambda, tau) triple via SLIC's per-call
# scoping.
_sb_horseshoe = StanBlocks.@slic begin
    raw    ~ std_normal()
    lambda ~ cauchy(0., 1.; lower=0.)
    tau    ~ cauchy(0., 1.; lower=0.)
    return raw * lambda * tau
end

# Missing-data scatter. Builds a length-`n` vector by placing the observed
# values at positions `Jobs` and the imputed parameters at positions `Jmis`.
# Mutation is allowed inside `@deffun` bodies (top-level @slic blocks are
# single-assignment), which is why the merge lives here rather than inline.
StanBlocks.@deffun begin
    mi_merge(y_obs::vector[n_obs], y_mis::vector[n_mis],
             Jobs::int[n_obs], Jmis::int[n_mis], n::int)::vector[n] = begin
        rv = rep_vector(0., n)
        for i in 1:n_obs; rv[Jobs[i]] = y_obs[i]; end
        for i in 1:n_mis; rv[Jmis[i]] = y_mis[i]; end
        return rv
    end
end

# Missing-data response submodel for the Normal family. Caller passes
# `loc`, `scale`, `y_obs`, `Jobs`, `Jmis` as kwargs (all data-qualified
# in the SLIC sense -- caller-provided). The two `~` lines split the
# joint likelihood:
#   - `y_mis ~ Normal(loc[Jmis], scale[Jmis])` introduces y_mis as a
#     parameter (LHS not yet bound) and contributes the missing-row
#     log-density. The conditional shape is exactly the family at those
#     positions, so no informative prior bias is introduced.
#   - `y_obs ~ Normal(loc[Jobs], scale[Jobs])` is the observed-row
#     likelihood (y_obs is :data-qualified via the kwarg, so SLIC routes
#     this to the model block).
# `mi_merge` then assembles the merged response vector for cross-formula
# references (e.g. `loc2 = a + b * y` in another formula).
# Per-family submodels (one each for Normal / BinomialLogit / Poisson / ...)
# rather than HOF-generic because each family's arg list shape differs and
# needs to be sliced per-arg at `[Jobs]` / `[Jmis]`.
_sb_mi_normal = StanBlocks.@slic begin
    # Typed-LHS sampling form: explicit `vector[n_mis]` so SLIC declares
    # y_mis as a vector parameter rather than inferring scalar from the
    # bare `normal(...)` call (which has no size-bearing kwarg).
    n_mis = num_elements(Jmis)
    y_mis :: vector[n_mis] ~ normal(loc[Jmis], scale[Jmis])
    y_obs ~ normal(loc[Jobs], scale[Jobs])
    return mi_merge(y_obs, y_mis, Jobs, Jmis,
                    num_elements(Jobs) + n_mis)
end

# Squared-exponential GP helpers. The data-layout conversion loop lives in a
# Stan function because top-level @slic bodies are deliberately control-flow
# free.
#
# BRM records exact-GP inputs as an N x d matrix so replay and descriptors keep
# their ordinary dense-data shape. `brm_gp_locations` converts its rows to
# Stan's native `array[N] vector[d]` GP carrier once in transformed data.
# `brm_exp_quad_cov` then delegates the whole multidimensional covariance and
# diagonal jitter to Stan's `gp_exp_quad_cov` / `add_diag` built-ins.
#
# `brm_hsgp_sqrt_spd` evaluates the separable d-dimensional squared-exponential
# spectral density at every tensor-product HSGP frequency. `omega2[b, j]` is
# the squared angular frequency for basis row b and predictor axis j.
StanBlocks.@deffun begin
    @stanonly brm_gp_locations(X::matrix[n, d])::vector[n, d] = begin
        locations::vector[n, d]
        for i in 1:n
            locations[i] = to_vector(X[i, :])
        end
        return locations
    end

    @stanonly brm_exp_quad_cov(X::vector[n, d], sigma::real,
                               rho::real, jitter::real)::matrix[n, n] = begin
        return add_diag(gp_exp_quad_cov(X, sigma, rho), jitter)
    end

    @stanonly brm_exp_quad_cov(X::vector[n, d], sigma::real,
                               rho::vector[d], jitter::real)::matrix[n, n] = begin
        return add_diag(gp_exp_quad_cov(X, sigma, rho), jitter)
    end

    @stanonly brm_hsgp_sqrt_spd(omega2::matrix[m, d], sigma::real,
                                 rho::vector[d])::vector[m] = begin
        rv::vector[m]
        scale = sigma
        for axis in 1:d
            scale *= sqrt(rho[axis] * 2.5066282746310002)
        end
        for b in 1:m
            exponent = 0.
            for axis in 1:d
                exponent += rho[axis] * rho[axis] * omega2[b, axis]
            end
            rv[b] = scale * exp(-0.25 * exponent)
        end
        return rv
    end
end

# Exact latent squared-exponential GP. The non-centred draw keeps the geometry
# explicit: f = cholesky(K(X, X)) * z. These are direct predictor summands, so
# there is no redundant population beta multiplying the returned draw.
_sb_gp = StanBlocks.@slic begin
    n_obs = dims(X)[1]
    X_gp = brm_gp_locations(X)
    log_rho   ~ std_normal()
    log_sigma ~ std_normal()
    z         ~ std_normal(; n=n_obs)
    K = brm_exp_quad_cov(X_gp, exp(log_sigma), exp(log_rho), jitter)
    return cholesky_decompose(K) * z
end

_sb_gp_aniso = StanBlocks.@slic begin
    n_obs = dims(X)[1]
    n_axes = dims(X)[2]
    X_gp = brm_gp_locations(X)
    log_rho :: vector[n_axes] ~ std_normal()
    log_sigma ~ std_normal()
    z         ~ std_normal(; n=n_obs)
    rho = exp(log_rho)
    K = brm_exp_quad_cov(X_gp, exp(log_sigma), rho, jitter)
    return cholesky_decompose(K) * z
end

# Hilbert-space approximate GP (Riutort-Mayol et al. 2022). `PHI` and
# `omega2` are tensor-product basis data precomputed by Julia. Isotropic and
# anisotropic variants differ only in whether one or d log length scales are
# sampled. As with exact GP, the returned draw is a direct predictor summand.
_sb_hsgp = StanBlocks.@slic begin
    n_basis = dims(omega2)[1]
    n_axes = dims(omega2)[2]
    log_rho   ~ std_normal()
    log_sigma ~ std_normal()
    beta_raw  ~ std_normal(; n=n_basis)
    rho = rep_vector(exp(log_rho), n_axes)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, exp(log_sigma), rho)
    return PHI * (sqrt_spd .* beta_raw)
end

_sb_hsgp_aniso = StanBlocks.@slic begin
    n_basis = dims(omega2)[1]
    n_axes = dims(omega2)[2]
    log_rho :: vector[n_axes] ~ std_normal()
    log_sigma ~ std_normal()
    beta_raw  ~ std_normal(; n=n_basis)
    rho = exp(log_rho)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, exp(log_sigma), rho)
    return PHI * (sqrt_spd .* beta_raw)
end

# Per-group HSGP. Length-scale/marginal-SD hyperparameters are shared across
# groups (decision 7p44fo); only tensor-basis weights vary by group.
_sb_hsgp_by = StanBlocks.@slic begin
    n_axes = dims(omega2)[2]
    log_rho   ~ std_normal()
    log_sigma ~ std_normal()
    rho = rep_vector(exp(log_rho), n_axes)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, exp(log_sigma), rho)
    PHI_scaled = diag_post_multiply(PHI, sqrt_spd)
    return rows_dot_product(PHI_scaled, beta[group_idx, :])
end

_sb_hsgp_by_aniso = StanBlocks.@slic begin
    n_axes = dims(omega2)[2]
    log_rho :: vector[n_axes] ~ std_normal()
    log_sigma ~ std_normal()
    rho = exp(log_rho)
    sqrt_spd = brm_hsgp_sqrt_spd(omega2, exp(log_sigma), rho)
    PHI_scaled = diag_post_multiply(PHI, sqrt_spd)
    return rows_dot_product(PHI_scaled, beta[group_idx, :])
end

# Categorical -> (n_levels::Int, per-row level index::Vector{Int}). Mirrors
# vimpl._level_index so the integer indices the walker stashes in `data`
# agree with what the cimpl-side uses.
_sb_level_index(raw::CA.CategoricalVector) = length(CA.levels(raw)), Int.(CA.levelcode.(raw))
_sb_level_index(raw::AbstractVector) = begin
    lvls = sort(unique(raw))
    lm = Dict(l => i for (i, l) in enumerate(lvls))
    length(lvls), [lm[l] for l in raw]
end

# Fit/apply split for categorical level coding (factor / mo). `_sb_fit_levels`
# returns the ordered level set (the frozen constant); `_sb_apply_levels` maps a
# raw column to 1-based codes against a (possibly frozen) level set, erroring on
# an unseen level (the dimension-coupled guard — brm-use §4 constraint 8). On
# the SAME training column these reproduce `_sb_level_index`'s codes exactly:
# for a CategoricalVector the level position == `CA.levelcode`; for a plain
# vector `sort(unique)` gives the same ordering. `_sb_level_index` (the
# construct-time entry) is unchanged.
_brm_fit_levels(raw::CA.CategoricalVector) = CA.levels(raw)
_sb_fit_levels(raw::AbstractVector) = _brm_fit_levels(raw)
_sb_apply_levels(levels, raw::CA.CategoricalVector) = _sb_apply_levels(levels, CA.unwrap.(raw))
_sb_apply_levels(levels, raw::AbstractVector) = begin
    lm = Dict(l => i for (i, l) in enumerate(levels))
    idx = Vector{Int}(undef, length(raw))
    for (j, l) in enumerate(raw)
        haskey(lm, l) || error(
            "sbimpl: reprocess: value `$l` is not a training level for this factor ",
            "(training levels: $(collect(levels))). The trained model has no ",
            "parameter for an unseen level. Re-fit with `freeze_constants=false` to ",
            "re-derive levels from the new data, or drop unseen categories first.")
        idx[j] = lm[l]
    end
    idx
end

# GP input helpers. Both public terms accept one-or-more raw real-valued axes
# and lower them to an N x d matrix. Keeping the raw column names in the
# preprocessing record lets `reprocess` rebuild that matrix on new data.
function _sb_gp_axes(label::Symbol, args::Tuple)
    isempty(args) && error("sbimpl: `$label(x...)` expects at least one positional axis")
    names = Symbol[]
    axes = Vector{Float64}[]
    for a in args
        n, raw = _sb_inner_data(label, a)
        v = collect(Float64, _sb_real_vec(label, n, raw))
        isempty(v) && error("sbimpl: `$label($n)` cannot use an empty axis")
        all(isfinite, v) || error("sbimpl: `$label($n)` requires finite values")
        push!(names, n)
        push!(axes, v)
    end
    n = length(first(axes))
    all(v -> length(v) == n, axes) || error(
        "sbimpl: `$label(x...)` axes must have equal lengths (got $(length.(axes)))")
    Tuple(names), Tuple(axes)
end

_sb_gp_matrix(axes::Tuple) = Matrix{Float64}(hcat(axes...))

function _sb_axis_option(label::Symbol, key::Symbol, value, n_axes::Int, pred, expectation::String)
    values = value isa Tuple || value isa AbstractVector ? Tuple(value) : ntuple(_ -> value, n_axes)
    length(values) == n_axes || error(
        "sbimpl: `$label(...; $key=...)` needs one value per axis ($n_axes), got $(length(values))")
    all(pred, values) || error(
        "sbimpl: `$label(...; $key=...)` expects $expectation, got $values")
    values
end

_sb_hsgp_options(kw, n_axes::Int) = begin
    K = _sb_axis_option(:hsgp, :k, get(kw, :k, 20), n_axes,
        x -> x isa Integer && !(x isa Bool) && x >= 1, "positive integers")
    c = _sb_axis_option(:hsgp, :c, get(kw, :c, 1.5), n_axes,
        x -> x isa Real && isfinite(x) && x > 1, "finite real values greater than 1")
    Tuple(Int(x) for x in K), Tuple(Float64(x) for x in c)
end

_sb_gp_iso(kw, label::Symbol) = begin
    iso = get(kw, :iso, true)
    iso isa Bool || error("sbimpl: `$label(...; iso=...)` expects Bool, got $(typeof(iso))")
    iso
end

_sb_gp_cov(kw, label::Symbol) = begin
    cov = get(kw, :cov, :exp_quad)
    cov === :exp_quad || error(
        "sbimpl: `$label(...; cov=...)` currently supports only `:exp_quad`, got $(repr(cov))")
    cov
end

function _check_term_kwargs(::typeof(gp), kw)
    allowed = (:cov, :iso, :jitter)
    unknown = filter(k -> k ∉ allowed, keys(kw))
    isempty(unknown) || error(
        "gp: exact GP accepts only `cov`, `iso`, and `jitter`; unsupported keyword(s): $(join(unknown, ", ")). " *
        "Use `hsgp(...; k=..., c=..., by=...)` for the Hilbert-space approximation.")
    _sb_gp_cov(kw, :gp)
    _sb_gp_iso(kw, :gp)
    jitter = get(kw, :jitter, 1e-9)
    jitter isa Real && isfinite(jitter) && jitter > 0 || error(
        "gp: `jitter` must be a finite positive real, got $(repr(jitter))")
    nothing
end

function _check_term_kwargs(::typeof(hsgp), kw)
    allowed = (:cov, :iso, :k, :c, :by)
    unknown = filter(k -> k ∉ allowed, keys(kw))
    isempty(unknown) || error(
        "hsgp: unsupported keyword(s): $(join(unknown, ", ")); " *
        "supported keywords are `cov`, `iso`, `k`, `c`, and `by`")
    _sb_gp_cov(kw, :hsgp)
    _sb_gp_iso(kw, :hsgp)
    nothing
end

function _sb_t2_options(kw)
    kval = get(kw, :k, (5, 5))
    kval isa Tuple && length(kval) == 2 || error(
        "t2: `k` must be a 2-tuple of integers greater than 2, got $(repr(kval))")
    all(x -> x isa Integer && !(x isa Bool) && x > 2, kval) || error(
        "t2: `k` must be a 2-tuple of integers greater than 2, got $(repr(kval))")

    basis = get(kw, :basis, (:cr, :cr))
    basis isa Tuple && length(basis) == 2 || error(
        "t2: `basis` must be a 2-tuple; only `(:cr, :cr)` is currently supported")
    basis == (:cr, :cr) || error(
        "t2: only cubic-regression-spline margins `basis=(:cr, :cr)` are currently supported, got $(repr(basis))")

    full = get(kw, :full, false)
    full isa Bool || error("t2: `full` must be Bool, got $(typeof(full))")
    full && error("t2: `full=true` is not supported yet; use `full=false`")
    (Tuple(Int(x) for x in kval), basis, full)
end

function _check_term_kwargs(::typeof(t2), kw)
    allowed = (:k, :basis, :full)
    unknown = filter(k -> k ∉ allowed, keys(kw))
    isempty(unknown) || error(
        "t2: unsupported keyword(s): $(join(unknown, ", ")); " *
        "supported keywords are `k`, `basis`, and `full`")
    _sb_t2_options(kw)
    nothing
end

_sb_mm_values(raw::CA.CategoricalVector) = CA.unwrap.(raw)
_sb_mm_values(raw::AbstractVector) = raw
_sb_mm_level_pool(raw::CA.CategoricalVector) = collect(CA.levels(raw))
_sb_mm_level_pool(raw::AbstractVector) = collect(unique(raw))

function _sb_mm_fit_levels(raw_groups)
    pooled = Any[]
    for raw in raw_groups
        append!(pooled, _sb_mm_level_pool(raw))
    end
    unique!(pooled)
    try
        sort!(pooled)
    catch
        error("sbimpl: `mm(...)` grouping levels must be mutually orderable so ",
              "one shared level index can be fitted (got $(repr(pooled)))")
    end
    pooled
end

function _sb_prepare_mm(raw_groups::Tuple, raw_weights, normalize::Bool;
                        levels=nothing,
                        group_names=ntuple(i -> Symbol(:group_, i), length(raw_groups)),
                        weight_names=nothing)
    M = length(raw_groups)
    M >= 2 || error("sbimpl: internal — multi-membership preprocessing received $M groups")
    all(v -> v isa AbstractVector, raw_groups) || error(
        "sbimpl: every `mm(...)` group must resolve to a vector column")
    N = length(first(raw_groups))
    group_values = map(_sb_mm_values, raw_groups)
    for m in 1:M
        v = group_values[m]
        length(v) == N || error(
            "sbimpl: `mm(...)` group column `$(group_names[m])` has $(length(v)) rows; ",
            "expected $N to match `$(group_names[1])`")
        bad = findfirst(ismissing, v)
        isnothing(bad) || error(
            "sbimpl: `mm(...)` group column `$(group_names[m])` is missing at row $bad")
    end

    fitted_levels = isnothing(levels) ? _sb_mm_fit_levels(raw_groups) : collect(levels)
    isempty(fitted_levels) && error("sbimpl: `mm(...)` grouping columns contain no levels")
    level_map = Dict(l => i for (i, l) in enumerate(fitted_levels))
    group_idx = Vector{Int}(undef, N * M)
    for i in 1:N, m in 1:M
        l = group_values[m][i]
        haskey(level_map, l) || error(
            "sbimpl: reprocess: `mm(...)` group column `$(group_names[m])` has ",
            "unseen level `$(l)` at row $i (training levels: $(fitted_levels)). ",
            "Re-fit with `freeze_constants=false` to derive a new shared level set.")
        group_idx[(i - 1) * M + m] = level_map[l]
    end

    weights = Vector{Float64}(undef, N * M)
    if isnothing(raw_weights)
        fill!(weights, inv(Float64(M)))
    else
        length(raw_weights) == M || error(
            "sbimpl: internal — `mm(...)` has $M groups but $(length(raw_weights)) weights")
        names = isnothing(weight_names) ? ntuple(i -> Symbol(:weight_, i), M) : weight_names
        for m in 1:M
            v = raw_weights[m]
            v isa AbstractVector || error(
                "sbimpl: `mm(...)` weight `$(names[m])` must resolve to a vector column")
            length(v) == N || error(
                "sbimpl: `mm(...)` weight column `$(names[m])` has $(length(v)) rows; expected $N")
            for i in 1:N
                x = v[i]
                x isa Real || error(
                    "sbimpl: `mm(...)` weight `$(names[m])` at row $i must be real; got $(repr(x))")
                isfinite(x) || error(
                    "sbimpl: `mm(...)` weight `$(names[m])` at row $i must be finite; got $(repr(x))")
                x >= 0 || error(
                    "sbimpl: `mm(...)` weight `$(names[m])` at row $i must be nonnegative; got $(repr(x))")
                xf = Float64(x)
                isfinite(xf) || error(
                    "sbimpl: `mm(...)` weight `$(names[m])` at row $i cannot be represented as finite Float64")
                weights[(i - 1) * M + m] = xf
            end
        end
        for i in 1:N
            row = ((i - 1) * M + 1):(i * M)
            total = sum(@view weights[row])
            isfinite(total) || error("sbimpl: `mm(...)` weights have a non-finite total at row $i")
            total > 0 || error("sbimpl: `mm(...)` weights must have a positive total at row $i")
            normalize && (@views weights[row] ./= total)
        end
    end
    (; levels=fitted_levels, group_idx, weights, n_obs=N, n_memberships=M)
end

# HSGP fit/apply split. The scalar methods reproduce the historical 1D basis;
# the tuple methods form its tensor product for variadic `hsgp(x...)`.
_sb_fit_hsgp(raw::AbstractVector{<:Real}, K::Integer, c::Real) = begin
    K >= 1 || error("hsgp: k must be >= 1 (got $K)")
    c > 1  || error("hsgp: c must be > 1 (got $c)")
    mu = sum(raw) / length(raw)
    L = c * maximum(abs, raw .- mu)
    L > 0 || error("hsgp: degenerate input (all x equal)")
    (mu, L)
end
_sb_apply_hsgp(c::Tuple, raw::AbstractVector{<:Real}, K::Integer) = begin
    mu, L = c
    x_c = raw .- mu
    lambda = [(k * pi / (2 * L))^2 for k in 1:K]
    PHI = zeros(length(raw), K)
    inv_sqrt_L = 1 / sqrt(L)
    for k in 1:K, i in eachindex(x_c)
        PHI[i, k] = inv_sqrt_L * sin(sqrt(lambda[k]) * (x_c[i] + L))
    end
    PHI, lambda
end

_sb_fit_hsgp(axes::Tuple, K::Tuple, c::Tuple) =
    ntuple(j -> _sb_fit_hsgp(axes[j], K[j], c[j]), length(axes))

function _sb_apply_hsgp(fits::Tuple, axes::Tuple, K::Tuple)
    n_axes = length(axes)
    length(fits) == n_axes == length(K) || error("hsgp: internal axis-count mismatch")
    axis_basis = ntuple(j -> _sb_apply_hsgp(fits[j], axes[j], K[j]), n_axes)
    n_obs = length(first(axes))
    n_basis = prod(K)
    PHI = Matrix{Float64}(undef, n_obs, n_basis)
    omega2 = Matrix{Float64}(undef, n_basis, n_axes)
    for (b, I) in enumerate(CartesianIndices(K))
        for i in 1:n_obs
            value = 1.0
            for axis in 1:n_axes
                value *= axis_basis[axis][1][i, I[axis]]
            end
            PHI[i, b] = value
        end
        for axis in 1:n_axes
            omega2[b, axis] = axis_basis[axis][2][I[axis]]
        end
    end
    PHI, omega2
end


# ==============================================================================
# SBBRMI: walk a BRMI, emit a SlicModel whose body references raw data columns
# by name. Scope for phase 1: population fixed effects + Normal likelihood.
# Ranefs / categorical / non-Normal likelihoods error out clearly.
#
# Design contract: data columns are referenced in the emitted expression by
# their formula names. `hcat(rep_vector(1., n), a, c1)` builds the design
# matrix inside Stan at runtime, so changing the data size (= length of the
# vectors) does not require recompiling the Stan model. StanBlocks' activity
# analysis routes each statement to the right Stan block (data / transformed
# data / parameters / model).
# ==============================================================================

"""
    SBBRMI(brmi::BRMI; mod=@__MODULE__, cv_groups=Set{Symbol}(),
           centered_groups=Set{Symbol}()) -> SBBRMI

StanBlocks backend: walks `brmi`, emits a `StanBlocks.SlicModel`, and
materialises the data dict. Pass `mod` if you're constructing the model
from a module other than `BayesianRegressionModels` so SLIC's symbol
resolver finds your locally-defined submodels.

`cv_groups` is an opt-in set of grouping-factor names (e.g. `[:subject]`)
whose per-group random effect should be emitted with **cv-contagious
sizing** -- the std-normal draw is sized from `maximum(<g>_idx)` instead of
the standalone data scalar `n_<g>`, so marking the group index with
`maybecv(:<g>_idx)` at trace time flips that RE to a generated-quantities
population re-draw (leave-all-out / out-of-sample). This is used when
generating a CV model artifact; the default (empty `cv_groups`) sizes the
same submodel from `n_<g>` and leaves the RE a fitted parameter. There are no
separate `_cv` submodels — the size expression passed at the call site is the
entire difference. Plain `(… | g)` ranefs and cross-formula `(… |ID| g)`
buckets are both supported; stratified `gr(g, by=b)` errors if opted-in.

`centered_groups` is an opt-in set of grouping-factor names whose per-group
random effect should be emitted in the **centered** parameterization -- the
per-group effect itself is the sampled parameter, with the covariance as its
prior (`bc ~ multi_normal_cholesky(0, diag_pre_multiply(tau, L))`), instead of
the default non-centered standardised draw plus downstream scaling. Centered
is the better geometry when the per-group likelihood is strong (dense repeated
measurement per level, e.g. many PK samples per subject); non-centered stays
the default, and a group not named here is untouched by this kwarg. Plain
`(… | g)` ranefs and `(… |ID| g)` buckets are supported; stratified
`gr(g, by=b)` is not.

The two sets are **mutually exclusive per group**: a centered block's sampled
parameter is the per-group effect itself, and no available spelling of that
can carry a cv taint in its size, so `SBBRMI` rejects a group named in both
rather than silently emitting an in-sample block. Note also that centered and
non-centered emissions use different unconstrained coordinates, so fitted
draws are not interchangeable between them.

Formula statements `effect(sd, ID) ~ Exponential(scale)` and
`effect(cor, ID) ~ LKJCholesky(K, eta)` configure a shared `|ID|` block.
An SD statement can instead select one emitted margin with
`effect(sd, ID, predictor, coefficient)`, or use the three-argument shorthand
when that predictor contributes exactly one margin. See [`ranefcoefnames`](@ref)
for the authoritative ordered addresses. Omitted statements retain historical
defaults; Julia's Exponential scale is converted to Stan's rate.

Use [`stan_code`](@ref) to extract the transpiled Stan source. For
sampling, load `StanLogDensityProblems` + `BridgeStan` and wrap the
emitted `SlicModel` in a `StanProblem`.

```julia
brmi  = @brm df (y ~ 1 + a + (1|g))
sbbrmi = SBBRMI(brmi)
src   = stan_code(sbbrmi)
```
"""
# ---- preprocessing-constant provenance (decision nr3v8n A) ------------------
# Each Category-A transform (zscale/standardize/center/factor/mo/s/t2/gp/hsgp) and the
# element-wise `protect`/implicit-fn fallback compute a data-derived constant in
# Julia at construct-time and land only the TRANSFORMED result in `data`. To
# support `reprocess`/`restan_data` on a new DataFrame we record, per EMITTED
# data key, how to regenerate that key from a new df:
#   kind         -- which transform (:zscale/:standardize/:center/:factor/:mo/
#                   :spline/:tensor_spline/:gp/:hsgp/:protect/:interaction/
#                   :categorical_outcome/:ordinal_outcome/
#                   :ordinal_threshold_predictor)
#   const_       -- the fitted constant: (μ,σ) / μ / level-vector / TPS basis /
#                   tensor-spline margins/centers / exact-GP axis metadata /
#                   HSGP (μ,L,K,c) / categorical
#                   outcome levels / nothing (protect)
#   raw_ref      -- the source: a column-node tree (zscale/center/standardize/
#                   protect, re-materialised via `_sb_rematerialize_vec`) or a
#                   column NAME Symbol (factor/mo/spline) or axis-name Tuple
#                   (gp/hsgp)
#   dim_coupled  -- true when a fitted level set drives parameter dimension
struct PreprocEntry
    kind::Symbol
    const_::Any
    raw_ref::Any
    dim_coupled::Bool
end

# Reserved side-channel key: during construction the emitters record into
# `data[_SB_PREPROC_KEY]`; the constructor pops it BEFORE building the SlicModel
# so it never reaches Stan's data dict. Filtered in `_sb_any_data_symbol`'s
# last-resort fallback so a data-iterating helper can never mistake it for a
# column while present.
const _SB_PREPROC_KEY = :__preproc__

_sb_record_preproc!(data, key::Symbol, entry::PreprocEntry) = begin
    pp = get(data, _SB_PREPROC_KEY, nothing)
    pp === nothing && return nothing   # recording disabled (defensive)
    pp[key] = entry
    nothing
end

# Re-materialise a column-node tree (NamedColumn / ExprColumn) against a fresh
# DataFrame `df`, mirroring `_sb_materialize_vec` but resolving raw-data leaves
# from `df` instead of the BRMI's training-bound DataColumns. Used by `reprocess`
# for zscale/center/standardize inners and the protect/implicit-fn fallback.
_sb_rematerialize_vec(x::Number, _df) = x
_sb_rematerialize_vec(x::NamedColumn, df) = _sb_df_column(df, name(x))
_sb_rematerialize_vec(x::ExprColumn, df) =
    broadcast(getf(x), map(a -> _sb_rematerialize_vec(a, df), getargs(x))...)
_sb_rematerialize_vec(x, _df) = error(
    "sbimpl: reprocess: cannot re-materialise $(typeof(x)) against the new DataFrame")

# Fetch a raw column from a new df via the BRM `Data` wrapper (the same accessor
# the `@brm` builder uses), unwrapped to a plain vector. Errors if absent.
_sb_df_column(df, col::Symbol) = begin
    nc = getproperty(Data(df), col)
    d = _as_data_column(parent(nc))
    isnothing(d) && error("sbimpl: reprocess: new DataFrame has no column `$col`")
    parent(d)
end

function _sb_gp_axes_from_df(df, raw_ref, label::Symbol)
    names = raw_ref isa Symbol ? (raw_ref,) : Tuple(raw_ref)
    axes = ntuple(length(names)) do j
        v = collect(Float64, _sb_df_column(df, names[j]))
        isempty(v) && error("sbimpl: reprocess: `$label($(names[j]))` cannot use an empty axis")
        all(isfinite, v) || error(
            "sbimpl: reprocess: `$label($(names[j]))` requires finite values")
        v
    end
    n = length(first(axes))
    all(v -> length(v) == n, axes) || error(
        "sbimpl: reprocess: `$label(x...)` axes must have equal lengths (got $(length.(axes)))")
    axes
end

# `mm`-only source columns are consumed entirely by Julia preprocessing. Find
# them before the generic data collector so raw labels (including strings) do
# not become unused Stan data. A source ALSO referenced outside `mm` is retained
# normally (e.g. `w1 + (1 | mm(...; weights=(w1,w2)))`).
_sb_collect_mm_sources!(_, _) = nothing
_sb_collect_mm_sources!(out, x::NamedColumn) = _sb_collect_mm_sources!(out, parent(x))
_sb_collect_mm_sources!(out, x::ExprColumn) = begin
    foreach(a -> _sb_collect_mm_sources!(out, a), getargs(x))
    foreach(v -> _sb_collect_mm_sources!(out, v), values(getkwargs(x)))
end
_sb_collect_mm_sources!(out, x::MultiMembershipTerm) = begin
    foreach(g -> push!(out, name(g)), getargs(x))
    weights = getfield(x, :weights)
    isnothing(weights) || foreach(w -> push!(out, name(w)), weights)
end

_sb_collect_non_mm_sources!(_, _) = nothing
_sb_collect_non_mm_sources!(out, x::NamedColumn) = begin
    parent(x) isa DataColumn && push!(out, name(x))
    nothing
end
_sb_collect_non_mm_sources!(out, x::ExprColumn) = begin
    foreach(a -> _sb_collect_non_mm_sources!(out, a), getargs(x))
    foreach(v -> _sb_collect_non_mm_sources!(out, v), values(getkwargs(x)))
end
_sb_collect_non_mm_sources!(_, ::MultiMembershipTerm) = nothing

struct SBBRMI{P<:BRMI, M, D<:AbstractDict, PP<:AbstractDict}
    parent::P
    model::M
    data::D
    preproc::PP
end

# Resolve formula-level `effect(...) ~ Normal(...)` statements against the
# authoritative population-column labels before emission. The returned vector
# for each LP is aligned 1:1 with `popcoefnames`; `nothing` means retain the
# default Normal(0, 1) for that column.
function _sb_effect_prior_overrides(brmi::BRMI)
    specs = effect_priors(brmi)
    isempty(specs) && return Dict{Symbol,Vector{Any}}()

    lp_names = Symbol[x.name for x in linear_predictors(brmi)]
    labels_by_lp = Dict{Symbol,Vector{Symbol}}()
    for lp in lp_names
        labels = try
            popcoefnames(brmi, lp)
        catch err
            error("sbimpl: cannot resolve population-effect labels for predictor `$lp` ",
                  "while applying `effect(...)`: $(sprint(showerror, err))")
        end
        isnothing(labels) || (labels_by_lp[lp] = labels)
    end

    overrides = Dict{Symbol,Vector{Any}}()
    for spec in specs
        T = _as_distribution_type(spec.family)
        (!isnothing(T) && T <: Normal) || error(
            "sbimpl: population `effect(...)` overrides currently support only " *
            "`Normal(location, scale)`; got `$(spec.family)`. Ordinary scalar " *
            "parameter priors remain available for other supported families.")
        isempty(spec.keywords) || error(
            "sbimpl: `effect(...) ~ Normal(...)` does not accept distribution " *
            "keywords; put bounds on an explicitly declared scalar parameter instead")

        target = spec.predictor
        if isnothing(target)
            candidates = Symbol[lp for lp in lp_names
                                if spec.coefficient in get(labels_by_lp, lp, Symbol[])]
            isempty(candidates) && error(
                "sbimpl: `effect($(spec.coefficient))` matches no population " *
                "coefficient. Inspect `popcoefnames(brmi, lp)` for valid labels.")
            length(candidates) == 1 || error(
                "sbimpl: `effect($(spec.coefficient))` is ambiguous across linear " *
                "predictors $(join(candidates, ", ")); use " *
                "`effect(linear_predictor, $(spec.coefficient))`.")
            target = only(candidates)
        end

        haskey(labels_by_lp, target) || error(
            "sbimpl: `effect($target, $(spec.coefficient))` names no linear " *
            "predictor with population coefficients. Available predictors: " *
            "$(join(sort!(collect(keys(labels_by_lp))), ", ")).")
        labels = labels_by_lp[target]
        idx = findfirst(==(spec.coefficient), labels)
        isnothing(idx) && error(
            "sbimpl: `$(spec.coefficient)` is not a population coefficient of " *
            "`$target`. Available labels: $(join(labels, ", ")).")
        target_overrides = get!(overrides, target) do
            Any[nothing for _ in labels]
        end
        isnothing(target_overrides[idx]) || error(
            "sbimpl: duplicate prior override for " *
            "`effect($target, $(spec.coefficient))`")
        target_overrides[idx] = spec.expression
    end
    overrides
end

function _sb_effect_normal_args(rhs::ExprColumn)
    args = _sb_stan_dist_args(getf(rhs), map(_sb_effect_prior_arg, getargs(rhs)))
    length(args) == 2 || error(
        "sbimpl: `effect(...) ~ Normal(...)` must lower to exactly location and " *
        "scale arguments, got $(length(args))")
    args
end

_sb_effect_prior_arg(x) = _sb_prior_arg(x)
function _sb_effect_prior_arg(x::ExprColumn)
    args = map(_sb_effect_prior_arg, getargs(x))
    # Formula parsing captures literal arithmetic as ExprColumns. Evaluate an
    # all-numeric effect-prior hyperparameter in Julia so `log(1 / 8)` retains
    # Julia's floating division semantics instead of becoming Stan integer
    # division. Symbol-bearing expressions stay as Stan expressions.
    if all(a -> a isa Real, args)
        value = try
            getf(x)(args...)
        catch
            nothing
        end
        value isa Real && return value
    end
    Expr(:call, getf(x), args...)
end

SBBRMI(brmi::BRMI; mod::Module=@__MODULE__, cv_groups=Set{Symbol}(),
       centered_groups=Set{Symbol}()) = begin
    cv_groups = cv_groups isa Set ? cv_groups : Set{Symbol}(cv_groups)
    centered_groups = centered_groups isa Set ? centered_groups : Set{Symbol}(centered_groups)
    both = intersect(cv_groups, centered_groups)
    isempty(both) || error(
        "sbimpl: group(s) $(join(sort!(collect(both)), ", ")) named in BOTH ",
        "`cv_groups` and `centered_groups`. A centered block's sampled ",
        "parameter is the per-group effect itself, whose size cannot carry a ",
        "cv taint (StanBlocks: cv does not reach a plate-internal fresh ",
        "parameter, and typed-LHS derives cv from RHS call-args rather than ",
        "the declared size). Emit the CV artifact non-centered, or drop the ",
        "group from `cv_groups`.")
    stmts = Any[]
    data = Dict{Symbol,Any}()
    # Side-channel: transform emitters record their fit-time constant + raw
    # reference here (via `_sb_record_preproc!`); popped before `SlicModel` below
    # so it never reaches Stan's data dict. See `PreprocEntry` / `reprocess`.
    data[_SB_PREPROC_KEY] = Dict{Symbol,PreprocEntry}()
    # Prepass 0: collect every column wrapped in `mi(...)` somewhere in the
    # model. Those columns are NOT materialised as plain data -- the
    # `_sb_emit_mi!` handler later splits them into observed values + missing
    # parameters. Any other formula that references the same name (e.g.
    # `loc2 = a + b * y`) will see the merged response, not the raw column.
    # Generic prepass: each registered LHS-decorator method writes into its
    # own subkey of `prepass`. The constructor only knows the subkey at the
    # consumer site below.
    prepass = Dict{Symbol, Any}()
    for (_, op) in pairs(brmi.operations)
        _sb_visit_op!(prepass, op)
    end
    effect_overrides = _sb_effect_prior_overrides(brmi)
    # Prepass 1: stash every data-backed NamedColumn so later intercept-only
    # predictors have a length probe to hang `rep_vector(1., num_elements(...))`
    # off, regardless of iteration order. Decorators that materialise their
    # own columns (e.g. `mi`'s `_sb_emit_mi!`) claim them via the prepass'
    # `:skip_data` bucket; the data-collection pass honours that.
    skip_data = get(prepass, :skip_data, Set{Symbol}())
    mm_sources = Set{Symbol}()
    non_mm_sources = Set{Symbol}()
    for (_, op) in pairs(brmi.operations)
        p = op isa NamedColumn ? parent(op) : op
        _sb_collect_mm_sources!(mm_sources, p)
        _sb_collect_non_mm_sources!(non_mm_sources, p)
    end
    union!(skip_data, setdiff(mm_sources, non_mm_sources))
    for (_, op) in pairs(brmi.operations)
        _sb_collect_data!(data, op; skip=skip_data)
    end
    # Prepass 2: collect brms-style `|ID|` ranef buckets across all sub-formulas,
    # emit one shared ranef_correlated_draws per bucket, and build a lookup
    # `(brmi_key, (id_sym, group_key)) => (bucket_name, col_range, idx_name, suffix)`
    # for per-sub-formula emission below.
    id_buckets = _sb_collect_id_buckets(brmi)
    ranef_effect_overrides = _sb_ranef_effect_overrides(brmi, id_buckets)
    id_lookup = _sb_emit_id_buckets!(stmts, data, id_buckets;
        cv_groups, centered_groups, ranef_effect_overrides)
    # Prepass 2.5: group-block terms. For each `mu ~ f(...)` where f has a
    # _sb_term_group_block declaration, allocate one ranef_correlated_draws
    # block per (f, group-column) pair. The lookup is threaded into _sb_emit!
    # so _sb_sampling_backed! can route declaring terms to their emit hook
    # with the pre-allocated block name and group-index name in hand.
    gb_terms = _sb_collect_group_block_terms(brmi)
    group_block_lookup = _sb_emit_group_blocks!(stmts, data, gb_terms)
    # Prepass 3: target -> observation-source map. Lets purely-intercept
    # linear predictors (`loc ~ 1`, `log(y_scale) ~ 1`) borrow N from the
    # observed `~` target that consumes them, instead of the hash-order
    # `_sb_any_data_symbol(data)` fallback. See `_sb_collect_target_obs`.
    target_obs = _sb_collect_target_obs(brmi)
    for (key, op) in pairs(brmi.operations)
        nc = _as_named_column(op)
        isnothing(nc) && error("sbimpl: top-level op `$key` is not a NamedColumn")
        obs_n = get(target_obs, key, nothing)
        _sb_emit!(stmts, data, key, parent(nc); id_lookup, obs_n, cv_groups,
                  centered_groups, group_block_lookup, effect_overrides)
    end
    # Pop the preproc side-channel BEFORE building the SlicModel so it never
    # pollutes Stan's data dict.
    preproc = pop!(data, _SB_PREPROC_KEY, Dict{Symbol,PreprocEntry}())
    body = Expr(:block, stmts...)
    model = StanBlocks.SlicModel(body, data, mod)
    SBBRMI(brmi, model, data, preproc)
end

# Materialise a user data vector into Stan's `data` dict. Refuses
# vectors containing `missing` -- BRM does not silently drop NA rows.
# To model NAs as parameters wrap the response in `mi(...)`; for NAs
# in predictors, drop or impute upstream before passing the dataframe.
_sb_data_vec(col_name::Symbol, raw) = begin
    if eltype(raw) >: Missing
        any(ismissing, raw) && error(
            "sbimpl: data column `$col_name` contains `missing` values. ",
            "BRM never silently drops rows. Either (a) drop / impute the ",
            "missing values in your dataframe before passing it, or ",
            "(b) for the response, wrap the LHS in `mi(...)` to model ",
            "the missing values as parameters (see TODO for support).")
        # Statically known to have no missings -- coerce the eltype so the
        # downstream Stan typer doesn't see a `Union{Missing,Float64}` it
        # can't translate.
        return collect(nonmissingtype(eltype(raw)), raw)
    end
    raw
end

# Generic prepass walker. Visits every operation; per-`typeof(f)` overrides
# of `_sb_register_sampling_lhs!` write into `ctx` under their own subkey.
# `ctx` is a plain Dict — each decorator's extension method picks its own
# key (e.g. `:skip_data`) and lazily creates the bucket via `get!`. The
# walker has no knowledge of `mi` or any specific decorator. Adding one is
# a new method on `_sb_register_sampling_lhs!`, no walker edit.
_sb_visit_op!(_, _) = nothing
_sb_visit_op!(ctx, op::NamedColumn) = _sb_visit_op!(ctx, parent(op))
_sb_visit_op!(ctx, p::ExprColumn{typeof(~)}) =
    _sb_register_sampling_lhs!(ctx, getargs(p, 2)[1])

# Default: no LHS shape claims anything.
_sb_register_sampling_lhs!(_, _) = nothing

# `mi(NamedColumn) ~ rhs` — claim the inner column under `:skip_data`,
# which the data-collection pass honours by not materialising the column
# (the merged response is emitted later by `_sb_emit_mi!`).
_sb_register_sampling_lhs!(ctx::AbstractDict, lhs::ExprColumn{typeof(mi)}) =
    _sb_register_mi_inner!(ctx, only(getargs(lhs)))
_sb_register_mi_inner!(_, _) = nothing
_sb_register_mi_inner!(ctx::AbstractDict, inner::NamedColumn) =
    (push!(get!(ctx, :skip_data, Set{Symbol}()), name(inner)); nothing)


_as_data_column(x::DataColumn) = x
_as_data_column(_) = nothing

_as_missing_column(x::MissingColumn) = x
_as_missing_column(_) = nothing

_as_distribution_type(::Type{T}) where {T<:Distribution} = T
_as_distribution_type(_) = nothing

_as_symbol(s::Symbol) = s
_as_symbol(_) = nothing

_as_int_vec(v::AbstractVector{<:Integer}) = v
_as_int_vec(_) = nothing

_as_real_vec(v::AbstractVector{<:Real}) = v
_as_real_vec(_) = nothing

_as_integer(x::Integer) = x
_as_integer(_) = nothing

_as_real(x::Real) = x
_as_real(_) = nothing

_as_error_exception(e::ErrorException) = e
_as_error_exception(_) = nothing

_scalar_or_lift(::Real, scalar_v, other_v) = :(rep_vector($scalar_v, num_elements($other_v)))
_scalar_or_lift(_, scalar_v, _other_v) = scalar_v

_sb_collect_data!(data, x; skip=Set{Symbol}()) = nothing
function _sb_record_data!(data, x::NamedColumn, d::DataColumn, skip)
    !(name(x) in skip) && (data[name(x)] = _sb_data_vec(name(x), parent(d)))
    nothing
end
_sb_record_data!(args...) = nothing

_sb_collect_data!(data, x::NamedColumn; skip=Set{Symbol}()) = begin
    d = parent(x)
    _sb_record_data!(data, x, d, skip)
    _sb_collect_data!(data, d; skip)
end
_sb_collect_data!(data, x::ExprColumn; skip=Set{Symbol}()) = begin
    foreach(a -> _sb_collect_data!(data, a; skip), getargs(x))
    foreach(v -> _sb_collect_data!(data, v; skip), values(getkwargs(x)))
end
_sb_collect_data!(data, x::MultiMembershipTerm; skip=Set{Symbol}()) = begin
    foreach(a -> _sb_collect_data!(data, a; skip), getargs(x))
    weights = getfield(x, :weights)
    isnothing(weights) || foreach(a -> _sb_collect_data!(data, a; skip), weights)
end

"""
    stan_code(sb::SBBRMI) -> String

Return the transpiled Stan source generated from `sb.model`. Forwards
to `StanBlocks.stan_code`. Useful for inspecting what the sbimpl walker
emitted before compiling.
"""
stan_code(sb::SBBRMI) = StanBlocks.stan_code(sb.model)

Base.show(io::IO, sb::SBBRMI) = begin
    print(io, "SBBRMI with data keys = ", sort(collect(keys(sb.data))), "\n")
    print(io, "emitted @slic body:\n")
    print(io, sb.model.model)
end


# ---- declaration-driven generative plans -----------------------------------

"""
    GenerativeDeclaration

One emitted sampling declaration in a [`GenerativePlan`](@ref).

- `role` is `:prior` or `:observation`. A `:prior` includes structured latent
  submodels such as `popefs`, `ranef_correlated`, and `plate`, not only scalar
  distribution calls.
- `target` and `family` are the emitted SLIC binding and RHS head.
- `data_source` is the original data key for an observation (including a
  plate-local alias such as `kernel_y => dv`), otherwise `nothing`.
- `draw` is the canonical binding a generative executor should use for an
  observation, otherwise `nothing`. For top-level observations this matches
  StanBlocks' current posterior-predictive `*_gen` name; nested plate
  observations are inventoried even though StanBlocks does not yet emit their
  generated quantity.
- `context` names enclosing plate results, outermost first.
- `expression` is an exact snapshot of the emitted `~` expression.

The remaining fields decompose that RHS call so an executor never has to parse
the snapshot itself:

- `arguments` are the RHS call's positional arguments, in order (the rate of
  `exponential(1)`, the eta of `lkj_corr_cholesky(1.)`).
- `keywords` is a `NamedTuple` of every RHS keyword argument, verbatim.
- `annotation` is the LHS type annotation (`:(vector[3])` for
  `kernel_z::vector[3] ~ ...`), or `nothing` for a bare LHS.
- `dimension` normalises the *two* ways a declaration can spell its size into
  one tuple: the `m`/`n`/`o` (or `size`) keyword that StanBlocks' `autotype`
  reads, and the LHS `::T[s...]` annotation, which wins when both are present.
  `()` means the declaration spells no size of its own — it is a scalar, or its
  extent comes from the data or from a submodel's internals (`popefs`,
  `ranef_correlated`) rather than from this declaration.
- `constraints` is the subset of `keywords` StanBlocks folds into the declared
  type: `lower`, `upper`, `offset`, `multiplier`. **This is the field that makes
  a half-normal visible**: `std_normal(; n=3, lower=0.)` and `std_normal(; n=3)`
  differ only here.

Entries of `arguments`, `dimension`, and `constraints` are the emitted
expressions, not evaluated values: an entry may be a literal, or a `Symbol` that
is a key of the plan's `data`, or a larger `Expr`. Resolve symbolic sizes
against `plan.data`.

`constraints` reports only the constraints spelled **on this declaration**.
Families carry their own implied support (`exponential` is positive,
`beta` lives on `[0, 1]`) — that is a property of `family`, held in StanBlocks'
`autokwargs` table, and BRM deliberately does not duplicate it here.

The declaration is intentionally backend-level: it describes what BRM really
emitted after formula terms introduced their latent blocks, rather than a
second model-specific interpretation of the `@brm` source.
"""
struct GenerativeDeclaration
    role::Symbol
    target::Symbol
    family::Any
    data_source::Union{Nothing,Symbol}
    draw::Union{Nothing,Symbol}
    context::Tuple{Vararg{Symbol}}
    expression::Expr
    arguments::Tuple
    keywords::NamedTuple
    annotation::Union{Nothing,Symbol,Expr}
    dimension::Tuple
    constraints::NamedTuple
end

"""
    GenerativePlan

An introspectable, replayable snapshot of an `SBBRMI`'s actual emitted
declarations. `model`, `data`, and `preproc` are copied together at plan
construction, while `declarations` inventories every emitted sampling site,
including priors introduced inside `kernel(...)` plates and every observation
in a multi-output model.

Construct with [`generative_plan`](@ref). [`stan_code`](@ref) accepts a plan,
and [`reprocess`](@ref) preserves the existing SBBRMI replay contract (including
its correct-or-loud unsupported cases).
Plans constructed from a reusable `@brm` builder also retain that builder so
`generative_plan(plan, new_df)` can rebuild the same declarations for genuinely
new groups.

This is a declaration plan, not an RNG executor: it does not claim that a
component-wise consumer draw is prior-predictive. Its purpose is to expose the
one authoritative program and provenance an executor must consume.
"""
struct GenerativePlan{P,M,D,PP,DS,B,CV}
    parent::P
    model::M
    data::D
    preproc::PP
    declarations::DS
    builder::B
    cv_groups::CV
end

_sb_plan_lhs_name(x::Symbol) = x
_sb_plan_lhs_name(x::Expr) =
    x.head === :(::) && !isempty(x.args) ? _sb_plan_lhs_name(x.args[1]) : nothing
_sb_plan_lhs_name(_) = nothing

_sb_plan_family(x::Expr) = begin
    if x.head === :call && !isempty(x.args)
        x.args[1]
    elseif x.head === :do && !isempty(x.args)
        _sb_plan_family(x.args[1])
    else
        x.head
    end
end
_sb_plan_family(x) = x

_sb_plan_generated(context, target) =
    Symbol(join((context..., target), "_"), "_gen")

# The LHS type annotation, if any: `z::vector[3] ~ rhs` -> `:(vector[3])`.
_sb_plan_annotation(x::Expr) =
    x.head === :(::) && length(x.args) == 2 ? deepcopy(x.args[2]) : nothing
_sb_plan_annotation(_) = nothing

# Split the emitted RHS call into positional args and keyword args. `do`-block
# RHSs (`plate(...) do ...`) split on the underlying call, matching `family`.
_sb_plan_as_kw(x::Expr) = x.head === :kw && length(x.args) == 2 ?
    (x.args[1]::Symbol => deepcopy(x.args[2])) : nothing
_sb_plan_as_kw(_) = nothing

_sb_plan_call_parts(x) = ((), NamedTuple())
function _sb_plan_call_parts(x::Expr)
    x.head === :do && !isempty(x.args) && return _sb_plan_call_parts(x.args[1])
    x.head === :call && !isempty(x.args) || return ((), NamedTuple())
    args, kws = Any[], Pair{Symbol,Any}[]
    for a in x.args[2:end]
        if a isa Expr && a.head === :parameters
            for p in a.args
                kw = _sb_plan_as_kw(p)
                isnothing(kw) || push!(kws, kw)
            end
            continue
        end
        kw = _sb_plan_as_kw(a)
        isnothing(kw) ? push!(args, deepcopy(a)) : push!(kws, kw)
    end
    (Tuple(args), (; kws...))
end

# StanBlocks' `autotype` reads sizes from `m`/`n`/`o` in that order, else `size`
# (functions.jl `autotype`), and folds these four keywords into the declared
# type's constraints (forward.jl typed-LHS path).
_sb_plan_size_kwargs() = (:m, :n, :o)
_sb_plan_constraint_kwargs() = (:lower, :upper, :offset, :multiplier)

# One tuple for both spellings of a declared size. The LHS `::T[s...]`
# annotation is the declared type's own size, so it wins over the keyword form.
function _sb_plan_dimension(annotation, kwargs)
    if annotation isa Expr && annotation.head === :ref && length(annotation.args) >= 2
        return Tuple(deepcopy.(annotation.args[2:end]))
    end
    sizes = Any[kwargs[k] for k in _sb_plan_size_kwargs() if haskey(kwargs, k)]
    isempty(sizes) || return Tuple(sizes)
    haskey(kwargs, :size) || return ()
    s = kwargs[:size]
    s isa Expr && s.head === :tuple ? Tuple(deepcopy.(s.args)) : (deepcopy(s),)
end

_sb_plan_constraints(kwargs) = (;
    (k => kwargs[k] for k in _sb_plan_constraint_kwargs() if haskey(kwargs, k))...)

_sb_plan_param_names(x::Symbol) = (x,)
_sb_plan_param_names(x::Expr) = if x.head === :tuple
    Tuple(filter(!isnothing, map(_sb_plan_lhs_name, x.args)))
else
    name = _sb_plan_lhs_name(x)
    isnothing(name) ? () : (name,)
end
_sb_plan_param_names(_) = ()

_sb_plan_plate_parts(x) = nothing
function _sb_plan_plate_parts(x::Expr)
    x.head === :do || return nothing
    length(x.args) == 2 || return nothing
    call, lambda = x.args
    call isa Expr && call.head === :call && !isempty(call.args) &&
        call.args[1] === :plate || return nothing
    lambda isa Expr && lambda.head === :-> && length(lambda.args) == 2 ||
        return nothing
    iterables = filter(a -> !(a isa Expr && a.head === :parameters), call.args[2:end])
    params = _sb_plan_param_names(lambda.args[1])
    (; iterables=Tuple(iterables), params, body=lambda.args[2])
end

function _sb_plan_collect!(declarations, x, data_scope, context=())
    x isa Expr || return nothing
    if x.head === :block
        foreach(stmt -> _sb_plan_collect!(declarations, stmt, data_scope, context), x.args)
        return nothing
    end
    if x.head === :call && length(x.args) >= 3 && x.args[1] === :~
        target = _sb_plan_lhs_name(x.args[2])
        isnothing(target) && error(
            "generative_plan: cannot identify emitted sampling LHS `$(x.args[2])`")
        rhs = x.args[3]
        data_source = get(data_scope, target, nothing)
        role = isnothing(data_source) ? :prior : :observation
        draw = role === :observation ? _sb_plan_generated(context, target) : nothing
        annotation = _sb_plan_annotation(x.args[2])
        arguments, keywords = _sb_plan_call_parts(rhs)
        push!(declarations, GenerativeDeclaration(
            role, target, _sb_plan_family(rhs), data_source, draw,
            Tuple(context), deepcopy(x), arguments, keywords, annotation,
            _sb_plan_dimension(annotation, keywords),
            _sb_plan_constraints(keywords)))

        plate = _sb_plan_plate_parts(rhs)
        if !isnothing(plate)
            nested_scope = copy(data_scope)
            for (param, iterable) in zip(plate.params, plate.iterables)
                iterable isa Symbol || continue
                source = get(data_scope, iterable, nothing)
                isnothing(source) || (nested_scope[param] = source)
            end
            _sb_plan_collect!(declarations, plate.body, nested_scope, (context..., target))
        end
        return nothing
    end
    nothing
end

function _generative_plan(sb::SBBRMI, builder, cv_groups)
    parent = deepcopy(sb.parent)
    data = deepcopy(sb.data)
    preproc = deepcopy(sb.preproc)
    body = deepcopy(sb.model.model)
    model = StanBlocks.SlicModel(body, data, sb.model.mod)
    declarations = GenerativeDeclaration[]
    data_scope = Dict{Symbol,Union{Nothing,Symbol}}(k => k for k in keys(data))
    _sb_plan_collect!(declarations, body, data_scope)
    GenerativePlan(parent, model, data, preproc, Tuple(declarations), builder,
                   copy(cv_groups))
end

"""
    generative_plan(sb::SBBRMI) -> GenerativePlan
    generative_plan(builder::Function, df; mod=@__MODULE__, cv_groups=Set()) -> GenerativePlan
    generative_plan(plan::GenerativePlan, new_df; cv_groups=Set()) -> GenerativePlan

Snapshot the declarations BRM actually emitted. The inventory is derived from
`sb.model.model`, so auto-introduced population coefficients, random-effect
blocks, `kernel(...)` eta blocks, observation families, and multiple outputs
cannot drift from the fitted model.

Use the reusable-builder form when future schedules may contain new groups:

```julia
builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    y ~ Normal(mu, sigma)
end
plan = generative_plan(builder, schedule; mod=@__MODULE__)
new_population_plan = generative_plan(plan, new_schedule)
```

The `SBBRMI` form has no reusable formula builder to apply to genuinely new
groups; use [`reprocess`](@ref) on that plan for the existing frozen-constant
replay semantics instead.
"""
generative_plan(sb::SBBRMI) = _generative_plan(sb, nothing, Set{Symbol}())

function generative_plan(builder::Function, df;
                         mod::Module=@__MODULE__, cv_groups=Set{Symbol}())
    brmi = Base.invokelatest(builder, df)
    brmi isa BRMI || error(
        "generative_plan: builder returned $(typeof(brmi)); expected a BRMI from `@brm begin ... end`")
    cv_groups = cv_groups isa Set ? cv_groups : Set{Symbol}(cv_groups)
    _generative_plan(SBBRMI(brmi; mod, cv_groups), builder, cv_groups)
end

function generative_plan(plan::GenerativePlan, new_df;
                         cv_groups=plan.cv_groups)
    isnothing(plan.builder) && error(
        "generative_plan: this plan was built from an SBBRMI and has no reusable `@brm` builder. " *
        "Construct it with `generative_plan(builder, df)` to rebuild the same declarations for new groups.")
    generative_plan(plan.builder, new_df; mod=plan.model.mod, cv_groups)
end

stan_code(plan::GenerativePlan) = StanBlocks.stan_code(plan.model)

Base.show(io::IO, plan::GenerativePlan) = begin
    nprior = count(d -> d.role === :prior, plan.declarations)
    nobs = count(d -> d.role === :observation, plan.declarations)
    print(io, "GenerativePlan with $nprior prior/latent and $nobs observation declarations")
end


# ---- reprocess / restan_data (decision nr3v8n A) ----------------------------

_sb_df_has_column(df, k::Symbol) = hasproperty(df, k)

# Recompute one transform-output data key against `df`. `freeze=true` applies
# the stored training constant (prediction-replay); `freeze=false` re-derives
# the constant from `df`, then applies (fresh-fit semantics). Writes the
# regenerated key(s) into `new_data`, the (possibly re-derived) record into
# `new_preproc`, and marks every key it owns in `handled`.
function _sb_reprocess_entry!(new_data, new_preproc, handled, key::Symbol, e::PreprocEntry, df, freeze::Bool)
    if e.kind === :zscale || e.kind === :standardize
        v = collect(Float64, _sb_rematerialize_vec(e.raw_ref, df))
        c = freeze ? e.const_ : _sb_fit_zscale(v)
        new_data[key] = _sb_apply_zscale(c, v)
        new_preproc[key] = PreprocEntry(e.kind, c, e.raw_ref, false)
    elseif e.kind === :center
        v = collect(Float64, _sb_rematerialize_vec(e.raw_ref, df))
        c = freeze ? e.const_ : _sb_fit_center(v)
        new_data[key] = _sb_apply_center(c, v)
        new_preproc[key] = PreprocEntry(:center, c, e.raw_ref, false)
    elseif e.kind === :protect
        # No fitted constant — re-materialise the same expr tree on `df`.
        new_data[key] = collect(Float64, _sb_rematerialize_vec(e.raw_ref, df))
        new_preproc[key] = e
    elseif e.kind === :interaction
        left_key, right_key = e.raw_ref
        haskey(new_data, left_key) || error(
            "sbimpl: reprocess: interaction `$key` is missing regenerated operand `$left_key`")
        haskey(new_data, right_key) || error(
            "sbimpl: reprocess: interaction `$key` is missing regenerated operand `$right_key`")
        left = new_data[left_key]
        right = new_data[right_key]
        length(left) == length(right) || error(
            "sbimpl: reprocess: interaction `$key` operand lengths mismatch ",
            "($(length(left)) vs $(length(right)))")
        new_data[key] = collect(Float64, left .* right)
        new_preproc[key] = e
    elseif e.kind === :multi_membership
        group_names = e.raw_ref.groups
        weight_names = e.raw_ref.weights
        raw_groups = Tuple(_sb_df_column(df, n) for n in group_names)
        raw_weights = isnothing(weight_names) ? nothing :
                      Tuple(_sb_df_column(df, n) for n in weight_names)
        levels = freeze ? e.const_.levels : nothing
        prepared = _sb_prepare_mm(raw_groups, raw_weights, e.const_.normalize;
                                  levels, group_names, weight_names)
        new_data[key] = prepared.group_idx
        new_data[e.const_.weight_key] = prepared.weights
        new_data[e.const_.n_groups_key] = length(prepared.levels)
        new_data[e.const_.n_obs_key] = prepared.n_obs
        new_data[e.const_.n_memberships_key] = prepared.n_memberships
        for owned in (e.const_.weight_key, e.const_.n_groups_key,
                      e.const_.n_obs_key, e.const_.n_memberships_key)
            push!(handled, owned)
        end
        const_ = merge(e.const_, (; levels=prepared.levels))
        new_preproc[key] = PreprocEntry(:multi_membership, const_, e.raw_ref, true)
    elseif e.kind === :spline
        v = collect(Float64, _sb_df_column(df, e.raw_ref))
        old_fit = e.const_.fit
        fit = freeze ? old_fit : _sb_fit_spline(v; k=old_fit.k)
        Xnull, Zpen = _sb_apply_spline(fit, v)
        new_data[key] = Xnull
        new_data[e.const_.zpen_key] = Zpen
        push!(handled, e.const_.zpen_key)
        new_preproc[key] = PreprocEntry(
            :spline, (; fit, zpen_key=e.const_.zpen_key), e.raw_ref, false)
    elseif e.kind === :tensor_spline
        axes = _sb_gp_axes_from_df(df, e.raw_ref, :t2)
        length(axes) == 2 || error("sbimpl: reprocess: `t2` needs exactly two margins")
        old_fit = e.const_.fit
        fit = freeze ? old_fit : _sb_fit_t2(axes[1], axes[2]; k=old_fit.k)
        Xfixed, Zrr, Zrn, Znr = _sb_apply_t2(fit, axes[1], axes[2])
        new_data[key] = Xfixed
        new_data[e.const_.zrr_key] = Zrr
        new_data[e.const_.zrn_key] = Zrn
        new_data[e.const_.znr_key] = Znr
        push!(handled, e.const_.zrr_key)
        push!(handled, e.const_.zrn_key)
        push!(handled, e.const_.znr_key)
        new_preproc[key] = PreprocEntry(:tensor_spline,
            (; fit, zrr_key=e.const_.zrr_key, zrn_key=e.const_.zrn_key,
             znr_key=e.const_.znr_key), e.raw_ref, false)
    elseif e.kind === :gp
        axes = _sb_gp_axes_from_df(df, e.raw_ref, :gp)
        new_data[key] = _sb_gp_matrix(axes)
        new_preproc[key] = PreprocEntry(:gp, e.const_, e.raw_ref, false)
    elseif e.kind === :hsgp
        axes = _sb_gp_axes_from_df(df, e.raw_ref, :hsgp)
        K = e.const_.K
        fits = freeze ? e.const_.fits : _sb_fit_hsgp(axes, K, e.const_.c)
        PHI, omega2 = _sb_apply_hsgp(fits, axes, K)
        new_data[key] = PHI
        new_data[e.const_.omega2_key] = omega2
        push!(handled, e.const_.omega2_key)
        new_preproc[key] = PreprocEntry(:hsgp,
            (; fits, K, c=e.const_.c, omega2_key=e.const_.omega2_key), e.raw_ref, false)
    elseif e.kind === :categorical_outcome
        v = _sb_df_column(df, e.raw_ref)
        fitted_levels = e.const_.levels
        expected_n_levels = e.const_.n_levels
        levels = freeze ? fitted_levels : _sb_fit_levels(v)
        length(levels) == expected_n_levels || error(
            "sbimpl: reprocess: categorical outcome `$key` has $(length(levels)) levels, " *
            "but the fitted `CategoricalLogit` has $expected_n_levels. " *
            "Preserve the fitted level set/order or rebuild the model with one " *
            "linear predictor per non-reference class.")
        new_data[key] = _sb_apply_levels(levels, v)
        new_preproc[key] = PreprocEntry(
            :categorical_outcome, (; levels, n_levels=expected_n_levels),
            e.raw_ref, true)
    elseif e.kind === :ordinal_outcome
        v = _sb_df_column(df, e.raw_ref)
        fitted_levels = e.const_.levels
        expected_n_levels = e.const_.n_levels
        levels = freeze ? fitted_levels : _sb_fit_levels(v)
        length(levels) == expected_n_levels || error(
            "sbimpl: reprocess: ordinal outcome `$key` has $(length(levels)) levels, " *
            "but the fitted `Ordinal` model has $expected_n_levels. Preserve the " *
            "fitted ordered level set or rebuild the model.")
        new_data[key] = _sb_apply_levels(levels, v)
        new_preproc[key] = PreprocEntry(
            :ordinal_outcome, (; levels, n_levels=expected_n_levels),
            e.raw_ref, true)
    elseif e.kind === :ordinal_threshold_predictor
        v = _sb_df_column(df, e.raw_ref)
        v isa AbstractVector{<:Real} || error(
            "sbimpl: reprocess: ordinal threshold predictor `$key` must be numeric, " *
            "got $(typeof(v))")
        all(isfinite, v) || error(
            "sbimpl: reprocess: ordinal threshold predictor `$key` contains " *
            "non-finite values")
        new_data[key] = collect(Float64, v)
        new_preproc[key] = e
    elseif e.kind === :observation_weight
        raw = _sb_df_column(df, e.raw_ref)
        response = _sb_df_column(df, e.const_.response)
        response isa AbstractVector || error(
            "sbimpl: reprocess: weighted response `$(e.const_.response)` must " *
            "be a vector, got $(typeof(response))")
        new_data[key] = _sb_prepare_weight_values(
            e.const_.kind, raw, length(response), e.const_.response, e.raw_ref)
        new_preproc[key] = e
    elseif e.kind === :factor || e.kind === :mo
        v = _sb_df_column(df, e.raw_ref)
        levels = freeze ? e.const_ : _sb_fit_levels(v)
        new_data[key] = _sb_apply_levels(levels, v)   # errors loudly on an unseen level
        new_preproc[key] = PreprocEntry(e.kind, levels, e.raw_ref, true)
        if e.kind === :factor
            # The `<x>_n_levels` count key co-emitted by `_sb_emit_cat!`: frozen
            # training count under freeze=true, re-counted under freeze=false.
            n_name = Symbol(e.raw_ref, :_n_levels)
            new_data[n_name] = length(levels)
            push!(handled, n_name)
        end
    else
        error("sbimpl: reprocess: unknown preproc kind `$(e.kind)` for key `$key`")
    end
    push!(handled, key)
    nothing
end

"""
    reprocess(sb::SBBRMI, new_df; freeze_constants=true) -> SBBRMI

Re-materialise the SBBRMI's Stan data dict against `new_df`, **re-running** the
Julia-side preprocessing (decision nr3v8n A) — so the silent-stale-constant bug
of a naive per-column `sb.model(; col=…)` rebind is avoided. Returns a NEW
`SBBRMI` that REUSES the already-transpiled Stan model (byte-identical
`stan_code` when shapes are stable) with the new data dict.

- `freeze_constants=true` (default — prediction / replay): apply the **training**
  constant to `new_df` (z-score with training mean/sd, map factor codes via the
  training level set, rebuild the spline/HSGP basis on the training
  eigenbasis/(mean, L)). This is the operation Bruno hand-rolls outside BRM.
- `freeze_constants=false` (fresh-fit semantics): re-derive each constant from
  `new_df`, then apply.

Covered: the Julia-side transforms (`zscale`/`standardize`/`center`/`factor`/
`mo`/`s`/`t2`/`gp`/`hsgp`), `protect`/implicit-fn columns (re-materialised on
`new_df`), typed `mm(...)` group indices and weights, continuous × continuous
interaction columns, typed observation weights, categorical outcomes, and
pass-through raw columns (plain data, `me` obs values, `ar` time). Errors loudly
on a `factor`/`mo`/`mm` **unseen level** and on any derived data key it cannot
account for (ordinary `(1|g)` indices remain outside this replay path) rather
than silently copying a stale vector — rebuild from `new_df` for those models.
"""
function reprocess(sb::SBBRMI, new_df; freeze_constants::Bool=true)
    new_data = Dict{Symbol,Any}()
    new_preproc = Dict{Symbol,PreprocEntry}()
    handled = Set{Symbol}()
    interaction_keys = Set(k for (k, e) in sb.preproc if e.kind === :interaction)
    # 1. Regenerate every independent transform-output key from its preproc
    #    record. Interactions wait until their raw/transformed operands exist.
    for (key, e) in sb.preproc
        e.kind === :interaction && continue
        _sb_reprocess_entry!(new_data, new_preproc, handled, key, e, new_df, freeze_constants)
    end
    # 2. Account for every remaining old data key: pass-through raw columns, or
    #    frozen structural scalars; ERROR on any unaccounted derived structure.
    for (k, v) in sb.data
        k in handled && continue
        k in interaction_keys && continue
        if v isa AbstractVector
            if _sb_df_has_column(new_df, k)
                new_data[k] = _sb_df_column(new_df, k)   # pass-through (plain / me obs / ar time)
            else
                error(
                    "sbimpl: reprocess: data key `$k` is a derived vector with no ",
                    "preprocessing record and is not a column of the new DataFrame. ",
                    "reprocess covers the Julia-side predictor transforms ",
                    "(zscale/standardize/center/factor/mo/s/t2/gp/hsgp), protect/implicit-fn ",
                    "columns, typed `mm(...)`, and pass-through raw columns; ordinary ",
                    "random-effects group indices (e.g. `(1|g)`) are not yet supported — ",
                    "rebuild the SBBRMI from the new DataFrame instead.")
            end
        elseif v isa Number || v isa AbstractString || v isa Bool
            new_data[k] = v   # frozen structural scalar / formula literal (e.g. me `sd_<x>`)
        else
            error(
                "sbimpl: reprocess: data key `$k` (::$(typeof(v))) is a derived ",
                "structure with no preprocessing record. reprocess cannot safely ",
                "regenerate it on the new DataFrame — rebuild the SBBRMI instead.")
        end
    end
    # 3. Derived interactions depend on operands regenerated in steps 1-2.
    for (key, e) in sb.preproc
        e.kind === :interaction || continue
        _sb_reprocess_entry!(new_data, new_preproc, handled, key, e, new_df, freeze_constants)
    end
    new_model = StanBlocks.SlicModel(sb.model.model, new_data, sb.model.mod)
    SBBRMI(sb.parent, new_model, new_data, new_preproc)
end

function reprocess(plan::GenerativePlan, new_df; freeze_constants::Bool=true)
    sb = SBBRMI(plan.parent, plan.model, plan.data, plan.preproc)
    _generative_plan(reprocess(sb, new_df; freeze_constants), plan.builder,
                     plan.cv_groups)
end

"""
    restan_data(sb::SBBRMI, new_df; freeze_constants=true) -> Dict

Thin convenience over [`reprocess`](@ref): the prepared Stan **data dict** for
`new_df`, ready for a `param_constrain!` replay. Equivalent to
`StanBlocks.stan_data(reprocess(sb, new_df; freeze_constants).model)`. Same
`freeze_constants` semantics (default `true` = training constants applied to new
data). See [`reprocess`](@ref) for the covered-terms list and error cases.
"""
restan_data(sb::SBBRMI, new_df; freeze_constants::Bool=true) =
    StanBlocks.stan_data(reprocess(sb, new_df; freeze_constants).model)

# ---- top-level op dispatch ---------------------------------------------------

_sb_emit!(stmts, data, key, op::ExprColumn; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Vector{Any}}()) =
    _sb_emit_expr!(stmts, data, key, getf(op), op; id_lookup, obs_n, cv_groups, centered_groups, group_block_lookup, effect_overrides)
# Raw data / missing columns appear as top-level ops when the formula mentions
# them as bare references (e.g. `c2` in `loc ~ 1 + c2`). Nothing to emit — the
# prepass already stashed data columns in `data`.
_sb_emit!(stmts, data, key, ::DataColumn; kwargs...) = nothing
_sb_emit!(stmts, data, key, ::MissingColumn; kwargs...) = nothing
_sb_emit!(stmts, data, key, op; kwargs...) = error("sbimpl: top-level op for `$key` not an ExprColumn (got $(typeof(op)))")

_sb_emit_expr!(stmts, data, key, ::typeof(~), op; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Vector{Any}}()) = begin
    lhs, rhs = getargs(op, 2)
    _sb_sampling!(stmts, data, key, lhs, rhs; id_lookup, obs_n, cv_groups,
                  centered_groups, group_block_lookup, effect_overrides)
end
_sb_emit_expr!(stmts, data, key, ::typeof(assign), op; id_lookup=_sb_empty_id_lookup(), kwargs...) = begin
    _, rhs = getargs(op, 2)
    target_expr = _sb_scalar_expr(rhs, data)
    push!(stmts, :($key = $target_expr))
end
_sb_emit_expr!(stmts, data, key, f, op; kwargs...) = error("sbimpl: unsupported top-level op `$f` for `$key`")

_sb_empty_id_lookup() = Dict{Tuple{Symbol,Tuple{Symbol,Any}}, Any}()


# ---- sampling: likelihood vs linear-predictor split --------------------------

"""
    _sb_submodel_rhs!(stmts, data, target, f, rhs)

sbimpl extension hook. Override (e.g. in `bruno-ext.jl`) to route
`target ~ f(...)` where `f` is a known SLIC submodel family
(`logistic_dr`, `gamma_time`, …) straight to `target ~ <slic>(; kwargs)`,
bypassing the population-linear-predictor wrap (which would otherwise
multiply the submodel output by a fresh β).

Return anything non-`nothing` to claim the binding; return `nothing` to
fall through to the default linear-predictor path. The default method
(this one) returns `nothing`.

**Use `import BayesianRegressionModels: _sb_submodel_rhs!`** (not
`using`) when adding methods from a downstream module so the binding is
extended rather than shadowed.
"""
_sb_submodel_rhs!(stmts, data, target, f, rhs) = nothing

# ============================================================================
# kernel(...) — general group-local submodel HoF term (decision `1vwycxb`).
#
#   pred ~ kernel(data..., per_subject_lps...) do slices..., lp_values...
#       ...
#   end
#
# The per-subject LP formulas own population effects, covariates, links and
# random-effect buckets. `kernel` derives one shared grouping from those LPs and
# only broadcasts the inline cell. The legacy `model=`/`obs=` spelling and its
# anonymous `n_eta` block were removed by user decision `130c904`.
function kernel end

# Reject the retired keyword surface at CONSTRUCTION — i.e. at the `@brm` /
# `kernel(...)` call the consumer actually wrote — not merely when the model is
# lowered.
#
# The loud rejections further down (`_sb_kernel_doblock!`, `_sb_submodel_rhs!`)
# run inside `SBBRMI(...)`, and `@brm` is a pure parser that captured `by=` /
# `n_eta=` / `model=` / `obs=` into the term's `getkwargs()` without looking at
# them. So a model written in the retired v1 spelling BUILT cleanly and objected
# only once someone lowered it. A consumer whose compatibility gate stops at
# BRMI construction — a reasonable gate, since it needs no Stan toolchain — saw
# retired syntax keep passing, and carried `by=`/`n_eta=` across 13 executable
# kernel sites for eight days after the removal landed while `brm-use` told them
# the keywords were rejected loudly (snag `by-and-n-eta-are-3625f645`).
#
# Checked in the same order the lowering-time guards use, so the reported
# keyword does not change when several are present. Keys off the `kernel`
# function object, so `hsgp(x; by=g)` — where `by=` is live — is unaffected, and
# an aliased `kernel` is still caught. The lowering-time guards stay as the
# backstop for a BRMI assembled without the macro.
_check_term_kwargs(::typeof(kernel), kwargs) = for (k, replacement) in (
        (:by, "Grouping is DERIVED from the ranef bucket of the per-subject \
               linear-predictor positional args."),
        (:n_eta, "Declare per-subject linear predictors with `(1 | ID | group)` \
                  terms and pass those LPs positionally."),
        (:model, "Write the per-subject cell INLINE as a `do`-block."),
        (:obs, "Write observation likelihoods as ordinary `~` statements inside \
                the cell body."),
    )
    haskey(kwargs, k) && error(
        "@brm: kernel(...) no longer accepts `$k=`. ", replacement,
        "\nThe surface is formula linear predictors plus one inline do-block:\n",
        "    log_CL ~ 1 + weight + (1 | p | subject)\n",
        "    log_V  ~ 1 +          (1 | p | subject)\n",
        "    pred ~ kernel(t, dose, dv, log_CL, log_V) do ts, d, yy, lCL, lV\n",
        "        mu = <prediction from exp(lCL), exp(lV) over ts, d>\n",
        "        yy ~ normal(mu, sigma)\n",
        "        mu\n",
        "    end\n",
        "See the `brm-use` skill, `kernel(...)` section.")
end

# do-block kernel surface (decision z9vkkf, User chose B): the consumer writes the
# per-subject cell body INLINE as a plate-style do-block, with the obs likelihood as
# ordinary `~` statements in the body (no `obs=` family DSL). Per-subject LPs own
# their random-effect blocks; the kernel only broadcasts the cell over their shared
# grouping.
#
# Multi-output (e.g. joint PK-PD, two correlated LP buckets). Give EACH output its
# OWN ragged observation column + time grid and observe it as a WHOLE column:
#
#   log_CL   ~ 1 + (1 | p | subject)
#   log_V    ~ 1 + (1 | p | subject)
#   log_Emax ~ 1 + (1 | q | subject)
#   log_EC50 ~ 1 + (1 | q | subject)
#   pred ~ kernel(
#       t_pk, t_pd, dose, dv_pk, dv_pd, log_CL, log_V, log_Emax, log_EC50,
#   ) do tpk, tpd, d, ypk, ypd, lCL, lV, lEmax, lEC50
#       conc = <PK prediction from lCL/lV over tpk>
#       resp = <PD prediction from lEmax/lEC50 (+ conc at tpd) over tpd>
#       ypk ~ normal(conc, sd)   # obs are just statements, one per output
#       ypd ~ normal(resp, sd)
#       conc                      # last expr = collected result
#   end
#
# Do-params bind to sliced positional data/LP args one-for-one, in order. Outer
# params (sd, covariates) are reached by lexical scope. No `by=`/`n_eta=`/
# `model=`/`obs=`.
#
# A custom `@deffun`-backed distribution must match one of its declared
# signatures exactly. For `truncated_normal`, a vector observation and location
# therefore require vector scale/bounds too; scalar captures must be lifted:
#
#   y ~ truncated_normal(mu,
#       rep_vector(sigma, dims(mu)[1]),
#       rep_vector(lloq, dims(mu)[1]),
#       rep_vector(uloq, dims(mu)[1]))
#
# Keep the `dims(mu)[1]` query inline: assigning it to an untyped plate-local
# name currently loses the integer type needed by `rep_vector`.
#
# NOT YET SUPPORTED — in-cell cmt/compartment masking. Keying ONE interleaved obs
# column by a cmt column INSIDE the cell —
#     y[c .== 1] ~ normal(conc[c .== 1], sd)   # <- does NOT transpile
# — is blocked in the StanBlocks substrate two ways: (1) a real ragged column slices
# to a native Stan `vector`, and `.==` on a native vector yields a real vector, not
# the `array[] int` mask boolean-mask indexing needs; (2) with an integer cmt column
# the mask + `findall` work, but the plate cannot hold the resulting `array[] int`
# index as a per-cell local ("scalar or vector[K] only (MVP)", StanBlocks
# `_plate_cell_shape`). Until that substrate lands, use separate per-output columns
# (above), or sub-select OUTSIDE the plate with a top-level integer index column over
# the collected result (`conc_pred[obs_idx] ~ ...`). This is NOT pending work: decision
# z9vkkf ("what @brm surface for cmt-keyed PK-PD obs?") was RESOLVED 2026-07-22 as
# "None of the above" — the ByCmt/obs-family extension, the defer, and the coupled
# `driver=` terms were ALL declined. So no cmt-keyed obs surface is being built; the
# per-output-column do-block form above is the answer.
#
# Walk a per-subject linear predictor to the `(id, group)` ranef bucket that
# defines its grouping — what lets `kernel(...)` DERIVE its plate grouping from
# the LPs handed in instead of being told via `by=` (v2, GO `0dnesv9`). `by=`
# only ever contributed the subject COUNT, restating a fact the ranef already
# knows, and was checked against nothing.
#
# Returns `(id_sym, group_key, group_col)`.
function _sb_kernel_lp_bucket(lp_col)
    decl = parent(lp_col)
    (decl isa ExprColumn && getf(decl) === ~) || error(
        "sbimpl: kernel(...) positional arg `$(name(lp_col))` is not a formula ",
        "statement; expected `$(name(lp_col)) ~ <terms>` in this @brm block.")
    rhs = getargs(decl)[2]
    pop_terms, ran_terms, direct_terms = Any[], Any[], Any[]
    for t in _sb_terms(rhs)
        _sb_classify_term!(t, pop_terms, ran_terms, direct_terms)
    end
    isempty(ran_terms) && error(
        "sbimpl: kernel(...) per-subject linear predictor `$(name(lp_col))` has no ",
        "random-effect term, so it defines no grouping. Give it one, e.g. ",
        "`$(name(lp_col)) ~ 1 + (1 | p | subject)`.")
    parts = map(_sb_ranef_parts, ran_terms)
    buckets = unique([(p[1], _sb_group_key(p[3])) for p in parts])
    length(buckets) == 1 || error(
        "sbimpl: kernel(...) per-subject linear predictor `$(name(lp_col))` spans ",
        "several ranef buckets ($(buckets)); a kernel cell needs exactly one grouping.")
    id_sym, gkey = only(buckets)
    desc = parts[1][3]
    (id_sym, gkey, desc isa NamedColumn ? desc : first(desc))
end

function _sb_kernel_doblock!(stmts, data, target::Symbol, dcols, kw)
    haskey(kw, :by) && error(
        "sbimpl: kernel(...) do-block form no longer accepts `by=`; grouping is ",
        "derived from its per-subject linear-predictor positional args.")
    haskey(kw, :n_eta) && error(
        "sbimpl: kernel(...) do-block form no longer accepts `n_eta=`; declare ",
        "per-subject linear predictors with `(1 | ID | group)` terms and pass ",
        "those LPs positionally.")
    haskey(kw, :model) &&
        error("sbimpl: kernel(...) do-block form takes an inline do-block, not `model=`")
    haskey(kw, :obs) &&
        error("sbimpl: kernel(...) do-block form takes ordinary `~` statements, not `obs=`")

    lam = first(dcols)
    ptuple = lam.args[1]
    params = ptuple isa Symbol ? Symbol[ptuple] :
        (Meta.isexpr(ptuple, :tuple) && all(p -> p isa Symbol, ptuple.args) ?
            Symbol[ptuple.args...] :
            error("sbimpl: kernel(...) do-block params must be plain names (no types/defaults)"))
    body = lam.args[2]
    body_stmts = Meta.isexpr(body, :block) ? body.args : Any[body]

    # positional args (everything after the do-block); sliced in the plate.
    # Two admissible kinds, both `NamedColumn` and both length-n_subjects under
    # the pre-grouped contract below, so `plate` slices either one identically:
    #
    #   - a RAW DATA column   -> register its vector in `data`;
    #   - a LATENT per-subject LINEAR PREDICTOR declared by an earlier formula
    #     statement (`log_CL ~ 1 + weight + (1|p|subject)`) -> emit NOTHING here.
    #     `_sb_linear_predictor!` has already assigned that name in the SLIC body,
    #     so the plate can slice it by name. Registering it as data would shadow
    #     the parameter with a constant (v2, decision `0dnesv9`).
    #
    # The per-subject LP needs no reshaping: the kernel contract is one row per
    # subject, so an ordinary LP over that frame is ALREADY length n_subjects.
    dcol_names = Symbol[]
    lp_cols    = Any[]
    for c in dcols[2:end]
        c isa NamedColumn || error(
            "sbimpl: kernel(...) positional args (after the do-block) must be a data ",
            "column or a per-subject linear predictor declared in this @brm block; ",
            "got a bare $(typeof(c)).")
        k = name(c)
        if parent(c) isa DataColumn
            data[k] = parent(parent(c))
        else
            push!(lp_cols, c)
        end
        push!(dcol_names, k)
    end
    ndata = length(dcol_names)

    length(params) == ndata || error(
        "sbimpl: kernel(...) do-block has $(length(params)) params but expects ",
        "$ndata — exactly one per positional data/LP arg.")
    slice_params = params

    # n_subjects + long-format guard (pre-grouped: one row per subject).
    #
    # v2 (`0xuaz0k`): the GROUPING IS DERIVED from the per-subject LPs' ranef
    # bucket. `by=` only ever restated a fact the ranef already knew (the subject
    # count), and a kernel with no per-subject LP now fails loudly because there
    # is no authoritative grouping to derive.
    #
    # ORDER: cells stay in ROW order, deliberately. `_sb_linear_predictor!`
    # returns `popefs(X) + rows_dot_product(Z, b[group_idx,:])`, i.e. a
    # ROW-ordered vector of length n_rows — it has already mapped level -> row.
    # Reordering cells to the ranef's LEVEL order (which `_sb_level_index` sorts)
    # would therefore MIS-align the LP against its own subjects whenever the
    # labels are not sorted. The only thing that must hold is the bijection
    # below: one level per row.
    isempty(lp_cols) && error(
        "sbimpl: kernel(...) needs at least one per-subject linear-predictor ",
        "positional arg with a random-effect term; without one there is no grouping ",
        "to derive.")
    lp_buckets = [_sb_kernel_lp_bucket(c) for c in lp_cols]
    groups = unique([b[2] for b in lp_buckets])
    length(groups) == 1 || error(
        "sbimpl: kernel(...) per-subject linear predictors disagree on their ",
        "grouping — got groups $(groups) across $(Tuple(name(c) for c in lp_cols)). ",
        "LPs may use distinct `|ID|` buckets, but every LP handed to one kernel ",
        "must describe the same subjects.")
    group_col = first(lp_buckets)[3]

    # The labels identify groups to Julia callers but never enter the emitted
    # Stan program: only their count and row order do. Accept arbitrary unique
    # labels so a reusable generative-plan builder can rebuild on genuinely new
    # subject ids without app-local recoding to 1:n.
    g_vals = collect(parent(parent(group_col)))
    nsub, g_idx = _sb_level_index(g_vals)
    (!isempty(g_vals) && !any(ismissing, g_vals) && nsub == length(g_idx)) || error(
        "sbimpl: kernel(...) needs pre-grouped per-subject data — `$(name(group_col))` ",
        "must list one non-missing unique subject per row; repeated levels indicate ",
        "long-format data. Got $(g_vals).")
    # Arbitrary labels identify rows on the Julia side; the emitted Stan program
    # consumes only their integer index/count. The generic data prepass has
    # already materialised the raw column, so discard it when Stan cannot type it
    # (e.g. `Vector{String}`). Numeric group labels remain available in case the
    # consumer also passed that column to the cell as ordinary numeric data.
    all(v -> v isa Real, g_vals) || pop!(data, name(group_col), nothing)
    nsub_sym = Symbol("kernel_nsub_", target); data[nsub_sym] = nsub

    # Per-subject plate: slice params bind the data columns and already-emitted LPs;
    # the user's inline body (obs `~` statements and all) runs inside; its last
    # expression is collected.
    plate_body = Expr(:block, body_stmts...)
    plate_call = Expr(:call, :plate,
        Expr(:parameters, Expr(:kw, :outer, Expr(:tuple, nsub_sym))),
        dcol_names...)
    plate_do = Expr(:do, plate_call,
        Expr(:->, Expr(:tuple, slice_params...), plate_body))
    push!(stmts, :($target ~ $plate_do))
    :done
end

function _sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(kernel), rhs)
    dcols = getargs(rhs)
    kw = getkwargs(rhs)
    # do-block form (decision z9vkkf, User chose B):
    # `kernel(datacols..., per_subject_lps...) do slices..., lps... <body> end`.
    # @brm captures the do-block as a verbatim lambda first-arg (macro.jl `_x`);
    # dispatch to the inline-body emitter.
    if !isempty(dcols) && first(dcols) isa Expr && first(dcols).head == :->
        return _sb_kernel_doblock!(stmts, data, target, dcols, kw)
    end
    error(
        "sbimpl: kernel(...) only accepts the inline do-block form; the legacy ",
        "`model=`/`obs=` surface was removed. Declare per-subject linear predictors, ",
        "pass them positionally, and write observation `~` statements in the cell.")
end

# ---- structured-latent (group-block) declaration + emit API -----------------
#
# This is BRM's GENERAL structured-latent floor (KB todo ez6anl; user resolved
# `1o7se36` to build the general mechanism, not a one-off). A term author
# declares one-or-more STRUCTURED latent fields — matrix-valued per-group blocks,
# each with its OWN prior (correlated-normal, iid-normal, or an element-wise
# possibly-clamped / non-normal distribution) — by defining a
# `_sb_term_group_block` method. Prepass 2.5 reads the declaration, allocates one
# block per field, and threads the un-expanded n_groups×K matrices into the term
# at emit time via `_sb_emit_group_block_term!`. Basis evaluation stays in-term.
#
# Three consumers shape this API: hsgp(x, by=g) (per-group iid-normal HSGP weights,
# the deliverable), obs_scale (matrix<lower=0> element-wise Exponential, decision
# 10uz10q — proven via `sb_group_clamped_demo`, not yet wired), and future
# splines-by-group (per-group normal spline coefficients).
#
# Declaration shape — the 2-arg `_sb_term_group_block(f, call)` form receives the
# term CALL so a term can declare fields conditional on its actual arguments
# (hsgp only declares a block when called with `by=`). It returns EITHER:
#   (a) legacy single correlated-normal block:
#       (; n_per_group::Int, group_arg_pos|group_fn, group_fn_name)
#   (b) general field list:
#       (; fields = [ (; name, n_per_group, group, prior), ... ])
# where, per field:
#   name         — Symbol; unique per term; names the emitted block `b_<name>_<g>`
#   n_per_group  — K columns per group (the matrix is n_groups × K)
#   group        — (; arg_pos=Int) | (; kwarg=Symbol) | (; fn, fn_name=Symbol)
#                    how to find the grouping column (positional arg / kwarg /
#                    a synthesised computed column)
#   prior        — :correlated_normal (LKJ across columns, std_normal across
#                    groups; via ranef_correlated_draws)
#                | :iid_normal        (iid std_normal weights; via std_normal)
#                | (; dist, args, [lower], [upper])  element-wise prior over the
#                    matrix, reusing the scalar-prior dist-name table
#                    (`_sb_stan_dist_name`); `lower`/`upper` clamp the matrix.
#
# Use `import BayesianRegressionModels: _sb_term_group_block, _sb_emit_group_block_term!`
# when adding methods from a downstream module so the binding is extended.

# Default: no group block. Override for declaring terms. The 2-arg form defaults
# to the legacy type-only 1-arg form so existing declarations keep working.
_sb_term_group_block(_) = nothing
_sb_term_group_block(f, _call) = _sb_term_group_block(f)

# Toy term declaration: 2 correlated params per group, first arg is the group.
_sb_term_group_block(::typeof(sb_group_demo)) = (; n_per_group=2, group_arg_pos=1)

# Clamped / non-normal demo (the obs_scale shape, decision 10uz10q): a
# matrix<lower=0>[n_groups, 2] field with an element-wise Exponential prior.
_sb_term_group_block(::typeof(sb_group_clamped_demo)) = (; fields=[
    (; name=:obs_scale_demo, n_per_group=2, group=(; arg_pos=1),
       prior=(; dist=Exponential, args=(1.0,), lower=0.)),
])

# Normalize a term's declaration (legacy single-block NT or general `fields` NT)
# into a uniform Vector of field specs, or `nothing` if the term declares none.
_sb_structured_fields(::Nothing, _f) = nothing
function _sb_structured_fields(decl::NamedTuple, f)
    haskey(decl, :fields) && return decl.fields
    # Legacy single correlated-normal block -> one normalized field. The field
    # name is `nameof(f)` so the emitted block name (`b_<name>_<g>`) is
    # byte-for-byte identical to the pre-generalization floor.
    group = haskey(decl, :group_fn) ?
        (; fn=decl.group_fn, fn_name=decl.group_fn_name) :
        (; arg_pos=get(decl, :group_arg_pos, 1))
    [(; name=nameof(f), n_per_group=decl.n_per_group, group, prior=:correlated_normal)]
end

# Emit hook for structured-latent terms. `block_info` carries a `fields` map
# (field-name => (; block_name, idx_name, n_per_group)); for single-field terms
# the lone field's keys are ALSO spliced at top level so legacy consumers that
# destructure `(; block_name, idx_name)` keep working unchanged.
# Default errors so a term with a declaration but no emit method is caught early.
_sb_emit_group_block_term!(stmts, data, target, f, rhs_e, block_info) =
    error("sbimpl: `$(nameof(f))` declared a structured-latent block but has no ",
          "`_sb_emit_group_block_term!` method — define one.")

# Toy term emit: thread group_block + group_idx into sb_group_demo_slic.
function _sb_emit_group_block_term!(stmts, data, target, ::typeof(sb_group_demo),
                                     rhs_e, block_info)
    (; block_name, idx_name) = block_info
    push!(stmts, :($target ~ sb_group_demo_slic(;
        group_block=$block_name, group_idx=$idx_name)))
end

# Clamped-demo emit: thread the matrix<lower=0> exponential block into the slic.
function _sb_emit_group_block_term!(stmts, data, target, ::typeof(sb_group_clamped_demo),
                                     rhs_e, block_info)
    info = block_info.fields[:obs_scale_demo]
    push!(stmts, :($target ~ sb_group_clamped_demo_slic(;
        group_block=$(info.block_name), group_idx=$(info.idx_name))))
end

# Built-in prior families on a missing-LHS sampling statement (e.g.
# `coef_a ~ Horseshoe()`). Returns `true` if it consumed the binding,
# `false` otherwise (then `_sb_linear_predictor!` runs).
_sb_emit_prior!(stmts, target, ::Type{<:Horseshoe}, _) = begin
    push!(stmts, :($target ~ _sb_horseshoe()))
    true
end

# A Distributions.jl `VonMises(mu, kappa)` prior has moving parameter support
# `[mu - pi, mu + pi]`. Stan cannot faithfully declare that support when `mu`
# is itself a parameter, so do not silently lower it to an unconstrained native
# `von_mises` prior. Observation likelihoods are handled separately below.
_sb_emit_prior!(_, target, ::Type{<:VonMises}, _) = error(
    "sbimpl: `VonMises` is supported as an observation likelihood, but not as " *
    "a parameter prior: preserving Distributions.jl's moving support " *
    "`[mu - pi, mu + pi]` requires a Stan parameterization chosen explicitly. " *
    "Use a real-valued latent parameter and transform/wrap it deliberately.")
# Generic scalar prior via a Distributions.jl constructor on the RHS
# (e.g. `coef_a ~ Normal(0, 0.1)`). Reuses the same family -> Stan-name
# table the likelihood path uses (`_sb_stan_dist_name`), so adding a
# new family extends both paths in one place. Args are literals or
# already-bound parameter symbols (no data materialisation in prior
# context), so we walk them directly without dragging in the full
# `_sb_scalar_expr` reducer.
function _sb_emit_prior!(stmts, target, ::Type{D}, op) where {D <: Distribution}
    stan_name = _sb_stan_dist_name(D)
    isnothing(stan_name) && return false
    # Lower Julia-side formula nodes first, then normalize constructor defaults
    # and parameterizations.  Some translations (scale -> inverse scale,
    # probability -> odds) create Stan expressions, so applying `_sb_prior_arg`
    # afterwards would mistake those already-lowered Exprs for formula nodes.
    arg_exprs = _sb_stan_dist_args(D, map(_sb_prior_arg, getargs(op)))
    push!(stmts, Expr(:call, :~, target, Expr(:call, stan_name, arg_exprs...)))
    true
end
_sb_emit_prior!(_, _, _, _) = false

# `LocationScale(mu, sigma, base)` is Distributions.jl's generic
# location-scale wrapper (alias for `AffineDistribution`). Useful
# specifically for distributions that don't take location/scale args
# natively -- in BRM today that's `TDist`. Compose:
#   coef ~ LocationScale(0, 0.1, TDist(3))  =>  coef ~ student_t(3, 0, 1) re-args'd to (3, 0, 0.1)
# i.e. the same effect as if Stan's `student_t` took (nu, mu, sigma).
# Unwraps the base distribution and reuses the existing per-family
# dispatch with the LocationScale's mu/sigma threaded in.
function _sb_emit_prior!(stmts, target, ::Type{<:LocationScale}, op)
    mu, sigma, base = getargs(op, 3)
    base_e = _as_expr_column(base)
    isnothing(base_e) && error(
        "sbimpl: `LocationScale(...)` third arg must be a distribution call, got $(typeof(base))")
    base_fam = getf(base_e)
    isnothing(_as_distribution_type(base_fam)) && error(
        "sbimpl: `LocationScale` base must be a Distribution type, got $(base_fam)")
    base = base_e
    stan_name = _sb_stan_dist_name(base_fam)
    isnothing(stan_name) && error(
        "sbimpl: `LocationScale` over `$(base_fam)` -- no Stan-name mapping ",
        "for the base. Add a `_sb_stan_dist_name(::Type{<:$(base_fam)})` entry.")
    base_args = _sb_stan_dist_args(base_fam, map(_sb_prior_arg, getargs(base)))
    # Stan's standard univariate dists put (location, scale) at
    # positions 2-3 (e.g. `student_t(nu, mu, sigma)`). The arg-shape
    # transform on the base already pads to that layout (TDist(nu) ->
    # (nu, 0, 1)). Replace those defaults with the LocationScale wrapper's
    # values; if the base call already supplied non-default location/scale,
    # composing them is the user's job (LocationScale(0.1, 2.0, Normal(5, 0.3))
    # would otherwise silently overwrite the inner 5 / 0.3).
    length(base_args) >= 3 || error(
        "sbimpl: `LocationScale` over `$(base_fam)`: base lowers to ",
        "$(length(base_args)) Stan args, need >= 3 for location/scale slots.")
    composed = (base_args[1], _sb_prior_arg(mu), _sb_prior_arg(sigma),
                base_args[4:end]...)
    arg_exprs = composed
    push!(stmts, Expr(:call, :~, target, Expr(:call, stan_name, arg_exprs...)))
    true
end

# Prior-arg lowering. Literals pass through; bare-Symbol references
# (already-bound parameter names) pass through; nested expressions
# recursively lower. Anything else (e.g. a data-column NamedColumn) is
# rejected -- prior args should be literal hyperparameters or
# already-declared scalar params.
_sb_prior_arg(x::Real) = x
_sb_prior_arg(x::Symbol) = x
_sb_prior_arg(x::NamedColumn) = _sb_prior_arg_named(x, parent(x))
_sb_prior_arg_named(x, ::MissingColumn) = name(x)
_sb_prior_arg_named(x, d) = error(
    "sbimpl: prior arg `$(name(x))` is backed by $(typeof(d)); ",
    "prior args must be literals or already-declared scalar parameters.")
_sb_prior_arg(x::ExprColumn) = Expr(:call, getf(x), map(_sb_prior_arg, getargs(x))...)
_sb_prior_arg(x) = error("sbimpl: unsupported prior-arg shape $(typeof(x))")

# LHS backed by real data => this is a likelihood. Record the observed values
# under the formula name in `data` and emit `key ~ dist(args...)`.
_sb_sampling!(stmts, data, key, lhs::NamedColumn, rhs; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Vector{Any}}()) =
    _sb_sampling_backed!(stmts, data, key, parent(lhs), rhs; id_lookup, obs_n,
                         cv_groups, centered_groups, group_block_lookup,
                         effect_overrides)

# `effect(...) ~ Distribution(...)` is metadata consumed by constructor prepasses;
# it deliberately emits no independent parameter or likelihood statement.
_sb_sampling!(_stmts, _data, _key, _lhs::ExprColumn{typeof(effect)}, _rhs;
              kwargs...) = nothing

_sb_sampling_backed!(stmts, data, key, backing::DataColumn, rhs; id_lookup, kwargs...) = begin
    data[key] = _sb_data_vec(key, parent(backing))
    _sb_likelihood!(stmts, key, rhs, data)
end

_sb_sampling_backed!(stmts, data, key, backing::MissingColumn, rhs;
                     id_lookup, obs_n=nothing, cv_groups=Set{Symbol}(),
                     centered_groups=Set{Symbol}(),
                     group_block_lookup=Dict(),
                     effect_overrides=Dict{Symbol,Vector{Any}}()) = begin
    rhs_e = _as_expr_column(rhs)
    if !isnothing(rhs_e)
        f = getf(rhs_e)
        # Group-block terms are claimed before the generic submodel/prior hooks.
        if !isempty(group_block_lookup)
            block_info = _sb_find_group_block(f, rhs_e, group_block_lookup)
            if !isnothing(block_info)
                _sb_emit_group_block_term!(stmts, data, key, f, rhs_e, block_info)
                return
            end
        end
        _sb_submodel_rhs!(stmts, data, key, f, rhs_e) !== nothing && return
        _sb_emit_prior!(stmts, key, f, rhs_e) && return
    end
    _sb_linear_predictor!(stmts, data, key, rhs; id_lookup, brmi_key=key, obs_n,
                          cv_groups, centered_groups, group_block_lookup,
                          effect_overrides)
end

_sb_sampling_backed!(stmts, data, key, backing, rhs; id_lookup, kwargs...) =
    error("sbimpl: unsupported LHS backing for `$key` ($(typeof(backing)))")

# Link-transformed LHS: `log(err) ~ 1 + d`, `logit(p) ~ 1 + x`, etc.
# Sample the linear predictor on the linked scale, then invert to recover the
# response. Mirrors vimpl's `inverse(getf(lhs))` path — any link whose Julia
# `inverse` is a function with a Stan-known name (log/exp/logit/logistic/
# sqrt/square, ...) works; unknown links error at transpile time.
#
# Generic LHS-of-tilde call `f(NamedColumn) ~ rhs`: link-function path. `f`
# is treated as a link whose `InverseFunctions.inverse(f)` we can map to a
# Stan name. Per-`typeof(f)` overrides (`mi`, future `cens`/`trunc`, …) live
# as separate methods of `_sb_sampling!`, mirroring vimpl's
# `vbroadcasted(::ExprColumn{typeof(F)})` extension idiom.
_sb_sampling!(stmts, data, key, lhs::ExprColumn, rhs; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Vector{Any}}()) =
    _sb_sampling_through_link!(stmts, data, key, getf(lhs), only(getargs(lhs)), rhs;
                               id_lookup, obs_n, cv_groups, centered_groups,
                               group_block_lookup, effect_overrides)

_sb_sampling_through_link!(stmts, data, key, f, inner, rhs; kwargs...) =
    error("sbimpl: expected NamedColumn inside link `$f(...)`, got $(typeof(inner))")
function _sb_sampling_through_link!(stmts, data, key, f, inner::NamedColumn, rhs; id_lookup, obs_n=nothing, cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(), group_block_lookup=Dict(), effect_overrides=Dict{Symbol,Vector{Any}}())
    inv_f = InverseFunctions.inverse(f)
    inner_name = name(inner)
    pre_name = Symbol(nameof(f), :_, inner_name)
    _sb_linear_predictor!(stmts, data, pre_name, rhs; id_lookup, brmi_key=key,
                          obs_n, cv_groups, centered_groups, group_block_lookup,
                          effect_overrides)
    push!(stmts, :($inner_name = $(_sb_julia_to_stan_fn(inv_f))($pre_name)))
end

# `mi(y) ~ <family>(args...)` — response with missing values modelled
# jointly. The inner symbol stays as the merged response so other formulas
# can reference it (e.g. `loc2 = ... + b * y`). One method on the same
# `_sb_sampling!` dispatch surface — same shape as `vbroadcasted(::ExprColumn{typeof(protect)})`
# and friends in vimpl.
_sb_sampling!(stmts, data, key, lhs::ExprColumn{typeof(mi)}, rhs; id_lookup=_sb_empty_id_lookup(), kwargs...) =
    _sb_emit_mi!(stmts, data, key, lhs, rhs)

# DRAFT: missing-data response handler. Triggered by `mi(y) ~ <family>(args)`.
# Splits the response into observed values (data) + missing parameters,
# routes both to the per-family `_sb_mi_<family>` submodel, and binds the
# merged vector back to the original symbol for cross-formula reference.
#
# The submodel does the actual SLIC work: declares y_mis via a real `~`,
# adds the observed-row likelihood, and returns the merged vector via the
# `mi_merge` UDF (mutation lives there since top-level slic is single-assign).
# Per-family rather than HOF-generic because family arg lists differ in
# shape and need per-arg `[Jobs]` / `[Jmis]` slicing.
#
# Today only the Normal family is wired (`_sb_mi_normal`); other families
# error with a clear "not yet implemented" message until their submodel
# lands. Predictor-side `mi(x)` (NAs in covariates needing a paired model
# formula) is also not drafted yet.
_SB_MI_FAMILIES = Dict{Any,Symbol}(
    Normal => :_sb_mi_normal,
)
function _sb_emit_mi!(stmts, data, key, lhs::ExprColumn, rhs)
    inner_raw = only(getargs(lhs))
    inner = _as_named_column(inner_raw)
    isnothing(inner) && error("sbimpl: `mi(...)` expects a NamedColumn inside, got $(typeof(inner_raw))")
    inner_name = name(inner)
    backing_raw = parent(inner)
    backing = _as_data_column(backing_raw)
    isnothing(backing) && error(
        "sbimpl: `mi($inner_name)` requires a real data column with missings; ",
        "got backing $(typeof(backing_raw)).")
    raw = parent(backing)
    eltype(raw) >: Missing || error(
        "sbimpl: `mi($inner_name)` requires a column whose eltype admits ",
        "`missing` (got $(eltype(raw))). If `$inner_name` has no NAs, drop ",
        "the `mi(...)` wrapper.")
    obs_mask = .!ismissing.(raw)
    Jobs = findall(obs_mask)
    Jmis = findall(.!obs_mask)
    isempty(Jmis) && error(
        "sbimpl: `mi($inner_name)` invoked on a column with NO missing values. ",
        "Drop the `mi(...)` wrapper, or check your data.")
    elT = nonmissingtype(eltype(raw))
    y_obs = collect(elT, raw[obs_mask])

    rhs_e = _as_expr_column(rhs)
    isnothing(rhs_e) && error("sbimpl: `mi($inner_name) ~ <family>(args)` requires the RHS to be a family call, got $(typeof(rhs))")
    fam = getf(rhs_e)
    rhs = rhs_e
    submodel = get(_SB_MI_FAMILIES, fam, nothing)
    isnothing(submodel) && error(
        "sbimpl: `mi(...)` for family `$fam` is not yet implemented. ",
        "Currently supported: $(sort(collect(keys(_SB_MI_FAMILIES)); by=string)). ",
        "Add a `_sb_mi_<family>` @slic submodel + a `_SB_MI_FAMILIES` entry.")

    # Split-data keys, scoped to the response name so two `mi()`-bearing
    # responses in the same model don't collide.
    obs_key  = Symbol(inner_name, :_obs)
    Jobs_key = Symbol(:Jobs_, inner_name)
    Jmis_key = Symbol(:Jmis_, inner_name)
    data[obs_key]  = y_obs
    data[Jobs_key] = Jobs
    data[Jmis_key] = Jmis

    # Family-arg kwargs: inspect the family RHS to pick out the per-arg
    # source bindings the submodel expects (e.g. Normal -> loc, scale).
    # Keep the kwarg names matching the submodel's body symbols.
    fam_kwargs = _sb_mi_family_kwargs(fam, rhs, data)

    # Single submodel call. SLIC binds `inner_name` to the merged vector
    # via the submodel's `return` (lowered to `<inner_name> = <return>`
    # in the parent model scope), so cross-formula references (e.g.
    # `loc2 = a + b * y` elsewhere) see the imputed-merged response.
    call_kwargs = Expr(:parameters,
        Expr(:kw, :y_obs, obs_key),
        Expr(:kw, :Jobs, Jobs_key),
        Expr(:kw, :Jmis, Jmis_key),
        fam_kwargs...)
    push!(stmts, Expr(:call, :~, inner_name, Expr(:call, submodel, call_kwargs)))
end

# Per-family unpacking of the family RHS into the kwargs the matching
# `_sb_mi_<family>` submodel body expects. Each family's arg list is
# fixed (Normal: loc, scale; later: BinomialLogit: n_trials, eta; etc.).
# Materialises any data columns referenced by family args into `data`.
function _sb_mi_family_kwargs(::Type{Normal}, rhs::ExprColumn, data)
    args = getargs(rhs, 2)
    loc, scale = args
    loc_v   = _sb_mi_kwarg_value(loc,   data)
    scale_v = _sb_mi_kwarg_value(scale, data)
    # Submodel body unconditionally subscripts (`scale[Jmis]` etc.), so a
    # scalar arg has to be lifted to a vector at the call site. Use the
    # other arg's length probe -- LP names are always full-length.
    scale_expr = _scalar_or_lift(scale, scale_v, loc_v)
    loc_expr   = _scalar_or_lift(loc,   loc_v,   scale_v)
    [Expr(:kw, :loc,   loc_expr),
     Expr(:kw, :scale, scale_expr)]
end

# Resolve a family-arg into the symbol the submodel body should reference.
# Plain LP names (NamedColumn over an ExprColumn) pass through as the
# bound symbol. Data columns get materialised into `data` (if not already)
# and reference by name. Numeric literals turn into themselves.
_sb_mi_kwarg_value(x::Real, _) = x
_sb_mi_kwarg_value(x::NamedColumn, data) = begin
    _maybe_record_data!(data, x, parent(x))
    name(x)
end

_maybe_record_data!(data, x, d::DataColumn) = (data[name(x)] = _sb_data_vec(name(x), parent(d)); nothing)
_maybe_record_data!(args...) = nothing
_sb_mi_kwarg_value(x, _) = error("sbimpl: unsupported `mi(...)` family arg shape $(typeof(x))")

# Map a Julia function (typically the result of `InverseFunctions.inverse(...)`
# for a link transform) to the Stan-side function name. Stan ships
# `inv_logit` rather than `logistic`, so the bare `nameof` would emit a
# call stanc rejects. Add other rename cases here as they come up.
_sb_julia_to_stan_fn(f) = f === LogExpFunctions.logistic ? :inv_logit : Symbol(nameof(f))

# Inner-arg unwrap helpers shared by `_sb_predictor_term!` overloads (mo, me,
# s, t2, gp, hsgp, ar, mo1) and a few `mi`-style sites. Each step is a tiny dispatch
# pair so a wrong call shape errors with the wrapping function's name in the
# message rather than a generic `isa` failure.
_sb_named_inner(label::Symbol, x::NamedColumn) = x
_sb_named_inner(label::Symbol, x) =
    error("sbimpl: `$label(...)` expects a NamedColumn, got $(typeof(x))")

_sb_data_backing(label::Symbol, n::Symbol, d::DataColumn) = parent(d)
_sb_data_backing(label::Symbol, n::Symbol, d) =
    error("sbimpl: `$label($n)` expects a raw data column, got $(typeof(d))")

_sb_inner_data(label::Symbol, x) = let inner = _sb_named_inner(label, x)
    (name(inner), _sb_data_backing(label, name(inner), parent(inner)))
end

_sb_real_vec(label::Symbol, n::Symbol, v::AbstractVector{<:Real}) = v
_sb_real_vec(label::Symbol, n::Symbol, v) =
    error("sbimpl: `$label($n)` expects numeric data, got $(typeof(v))")


# ---- linear predictor: emit `X_<name> = hcat(...); <name> ~ popefs(; X=X_<name>)` --

_sb_classify_term!(t::ExprColumn, pop_terms, ran_terms, direct_terms) = begin
    f = getf(t)
    f === (|) && (push!(ran_terms, t); return)
    (f === offset || f === mo1 || f === s || f === t2 || f === gp || f === hsgp) &&
        (push!(direct_terms, t); return)
    push!(pop_terms, t)
end
_sb_classify_term!(t, pop_terms, ran_terms, direct_terms) =
    isnothing(_sb_cat_levels(t)) ? push!(pop_terms, t) : push!(direct_terms, t)

function _sb_linear_predictor!(stmts, data, target::Symbol, rhs;
                                id_lookup=_sb_empty_id_lookup(),
                                brmi_key::Symbol=target,
                                obs_n::Union{Symbol,Nothing}=nothing,
                                cv_groups=Set{Symbol}(),
                                centered_groups=Set{Symbol}(),
                                group_block_lookup=Dict(),
                                effect_overrides=Dict{Symbol,Vector{Any}}())
    terms = _sb_terms(rhs)
    pop_terms    = Any[]
    ran_terms    = Any[]  # `(expr | group)` -> collected per-group below
    direct_terms = Any[]  # e.g. `mo1(c)` / `s(x)` / `t2(x,z)` -> direct summand
    for t in terms
        _sb_classify_term!(t, pop_terms, ran_terms, direct_terms)
    end
    isempty(pop_terms) && isempty(ran_terms) && isempty(direct_terms) &&
        error("sbimpl: empty RHS for `$target` — no predictor terms")

    # Direct fixed-slope terms such as `offset(log(exposure))` may contribute
    # an expression rather than a named emitted column. Keep the summand
    # carrier expression-capable; sampled/direct submodels still push Symbols.
    summands = Any[]

    if !isempty(pop_terms)
        col_exprs = Any[]
        for t in pop_terms
            _sb_pop_cols!(col_exprs, t, data, stmts, pop_terms; obs_n, group_block_lookup)
        end
        X_name = Symbol(:X_, target)
        pop_name = Symbol(:pop_, target)
        # StanBlocks `hcat` promotes a lone vector to matrix[n,1] and folds to
        # append_col for two-or-more columns, so we can always just emit hcat.
        push!(stmts, :($X_name = $(Expr(:call, :hcat, col_exprs...))))
        overrides = get(effect_overrides, brmi_key, nothing)
        if isnothing(overrides)
            push!(stmts, :($pop_name ~ popefs(; X=$X_name)))
        else
            length(overrides) == length(col_exprs) || error(
                "sbimpl: internal effect-prior alignment error for `$brmi_key`: " *
                "$(length(overrides)) priors for $(length(col_exprs)) columns")
            beta_loc = Any[0.0 for _ in overrides]
            beta_scale = Any[1.0 for _ in overrides]
            for i in eachindex(overrides)
                isnothing(overrides[i]) && continue
                beta_loc[i], beta_scale[i] = _sb_effect_normal_args(overrides[i])
            end
            loc_expr = Expr(:vect, beta_loc...)
            scale_expr = Expr(:vect, beta_scale...)
            push!(stmts, :($pop_name ~ _popefs_normal(;
                X=$X_name, beta_loc=$loc_expr, beta_scale=$scale_expr)))
        end
        push!(summands, pop_name)
    end

    for dt in direct_terms
        _sb_emit_direct!(stmts, data, target, dt, summands; group_block_lookup)
    end

    _sb_emit_ranefs!(stmts, data, target, ran_terms, summands; id_lookup, brmi_key, cv_groups, centered_groups)

    if length(summands) == 1
        push!(stmts, :($target = $(only(summands))))
    else
        push!(stmts, :($target = $(Expr(:call, :+, summands...))))
    end
end

# ---- consumer helper: name the `beta_pop` columns -------------------------
#
# `popcoefnames(brmi, lhs)` lets a downstream consumer relabel raw
# `pop_<lhs>_beta_pop.N` posterior columns WITHOUT re-parsing the formula and
# WITHOUT re-deriving the (data-dependent) pop-vs-cat-vs-ranef classification.
# It mirrors the emitter EXACTLY by driving the SAME `_sb_classify_term!` +
# `_sb_pop_cols!` used to build the design matrix `X` — so the returned labels
# can never drift from what `popefs` actually multiplies. Standard fixed-effect
# terms only (covers the whole regression-covariate surface); group-structured
# terms whose columns need prepass context (e.g. `hsgp(x, by=g)`) are out of
# scope in v1 and raise a clear, actionable error rather than mis-counting.

"""
    popcoefnames(brmi::BRMI, lhs::Symbol) -> Union{Vector{Symbol}, Nothing}

Ordered labels of the population-level `beta_pop` coefficient columns for
linear predictor `lhs`, as emitted by the sbimpl backend (`SBBRMI`). The
k-th returned symbol labels the parameter column `pop_<lhs>_beta_pop.k`
1:1, so a consumer can turn raw `pop_<lhs>_beta_pop.N` posterior columns
into human-readable names without re-parsing the formula.

Only terms that sbimpl folds into the `popefs` design matrix appear, in
formula (left-to-right) order:

- the intercept `1` is INCLUDED (labelled `:Intercept`), at its formula
  position — conventionally first, `1 + …`; there is NO separate
  `pop_<lhs>_Intercept` parameter;
- plain continuous (`Real`, non-integer) predictors — one column each,
  labelled by the column name;
- single-`beta` wrapped terms (`mo`, `me`, `protect`,
  `log(x)`, `x^2`, …) — one column each, labelled by the emitted design key;
- an interaction `a & b` — one label per expanded treatment-contrast column.

EXCLUDED — emitted as their OWN parameters, NOT `beta_pop`, so they never
appear in `pop_<lhs>_beta_pop`:

- integer- or `CategoricalVector`-typed bare predictors → `cat_<name>`
  (K−1 treatment contrasts);
- `offset(x)` → `x` itself, with fixed coefficient one;
- `mo1(c)` → `mo1_<c>`; `s(x)` and `t2(x,z)` → their own fixed/range
  coefficients and smoothing scales;
- explicit-coefficient `coef * a` (own scalar);
- random-effects blocks `(… | g)` → per-group ranef parameters.

Needs a data-bound `BRMI` (any fitted model has one): the population-vs-
categorical split reads each predictor's element type. Returns `Symbol[]`
when `lhs` has no population columns (e.g. `loc ~ (1 | g)`), and `nothing`
when `lhs` is not a linear predictor of `brmi`.
"""
function popcoefnames(brmi::BRMI, lhs::Symbol)
    op = linear_predictor_op(brmi, lhs)
    isnothing(op) && return nothing
    _, rhs = getargs(op, 2)
    pop_terms = Any[]; ran_terms = Any[]; direct_terms = Any[]
    for t in _sb_terms(rhs)
        _sb_classify_term!(t, pop_terms, ran_terms, direct_terms)
    end
    labels = Symbol[]
    # Scratch data/stmts: `_sb_pop_cols!` materialises columns into these as a
    # side effect; we only read back the pushed column reference(s) as labels.
    scratch = Dict{Symbol,Any}(); scratch_stmts = Any[]
    for t in pop_terms
        # The intercept is always exactly one all-ones column; skip the
        # emitter's length probe (irrelevant to naming) and label it directly.
        if t isa Integer
            push!(labels, :Intercept)
            continue
        end
        cols = Any[]
        try
            _sb_pop_cols!(cols, t, scratch, scratch_stmts, pop_terms)
        catch err
            error("popcoefnames: cannot resolve the `beta_pop` column(s) for term ",
                  "`$(_popcoef_show(t))` in predictor `$lhs` without full model context ",
                  "($(sprint(showerror, err))). This helper covers the standard ",
                  "fixed-effect terms; for group-structured terms (e.g. `hsgp(x, by=g)`) ",
                  "read the parameter names off the transpiled `SBBRMI` instead.")
        end
        for c in cols
            # Non-intercept pop columns are Symbol references; a stray Expr
            # (only the intercept probe emits one) maps back to :Intercept.
            push!(labels, c isa Symbol ? c : :Intercept)
        end
    end
    labels
end

_popcoef_show(t::ExprColumn) = string(getf(t))
_popcoef_show(t::NamedColumn) = string(name(t))
_popcoef_show(t) = string(t)

# Classify a predictor term as "direct" (allocates its own parameters, no
# popefs multiplication). Matches vimpl: integer-backed NamedColumns are
# treated as treatment-coded categoricals; floats go through popefs.
# Value-narrowing chain: returns the underlying integer/categorical level
# vector if `t` is a NamedColumn over a DataColumn over a categorical-shaped
# vector, else `nothing`. Replaces the old Bool-predicate trio
# (`_sb_is_categorical` / `_is_cat_data` / `_is_cat_vec`); call sites compose
# with `isnothing(_sb_cat_levels(t))` instead of carrying the predicate.
_sb_cat_levels(t::NamedColumn) = _sb_cat_levels_data(parent(t))
_sb_cat_levels(_t) = nothing

_sb_cat_levels_data(d::DataColumn) = _sb_cat_levels_vec(parent(d))
_sb_cat_levels_data(_d) = nothing

_sb_cat_levels_vec(v::AbstractVector{<:Integer}) = v
_sb_cat_levels_vec(v::CA.CategoricalVector) = v
_sb_cat_levels_vec(_v) = nothing

# Free-summand terms (no popefs beta): `mo1(c)`, `s(x)`, `t2(x,z)`, categoricals.
# Categoricals emit `cat_<c> ~ _sb_cat(; x=<c>_idx, n_levels=<c>_n_levels)`.
# `mo1(c)` reuses `_sb_mo`; smooths own their complete fixed + penalized bases.
_sb_emit_direct!(stmts, data, target::Symbol, t::NamedColumn, summands; kwargs...) =
    _sb_emit_cat!(stmts, data, t, summands)
function _sb_emit_direct!(stmts, data, target::Symbol, t::ExprColumn, summands;
                          group_block_lookup=Dict())
    f = getf(t)
    if f === gp || f === hsgp
        push!(summands, _sb_predictor_term!(stmts, data, f, t; group_block_lookup))
        return
    end
    _sb_emit_direct_expr!(stmts, data, target, getf(t), t, summands)
end
function _sb_emit_direct_expr!(_stmts, data, _target::Symbol,
                               ::typeof(offset), t, summands)
    args = getargs(t)
    length(args) == 1 || error(
        "sbimpl: `offset(x)` expects exactly one positional argument, got $(length(args))")
    isempty(getkwargs(t)) || error("sbimpl: `offset(x)` does not accept keyword arguments")
    # Unlike `protect`, offset is a direct model expression: it may reference
    # raw data, an already-declared sampled scalar, or a composition of either.
    # `_sb_scalar_expr` preserves that provenance in the emitted Stan expression
    # and, critically, introduces no `popefs` coefficient.
    push!(summands, _sb_scalar_expr(only(args), data))
end
function _sb_emit_direct_expr!(stmts, data, target::Symbol, ::typeof(mo1), t, summands)
    inner_name, raw = _sb_inner_data(:mo1, only(getargs(t)))
    n_levels, idx = _sb_level_index(raw)
    n_levels >= 2 || error("sbimpl: `mo1($inner_name)` needs >= 2 levels (got $n_levels)")
    idx_name = Symbol(inner_name, :_idx)
    col_name = Symbol(:mo1_, inner_name)
    data[idx_name] = idx
    push!(stmts, :($col_name ~ _sb_mo(; x=$idx_name)))
    push!(summands, col_name)
end
function _sb_emit_direct_expr!(stmts, data, target::Symbol, ::typeof(s), t, summands)
    push!(summands, _sb_predictor_term!(stmts, data, s, t))
end
function _sb_emit_direct_expr!(stmts, data, target::Symbol, ::typeof(t2), t, summands)
    push!(summands, _sb_predictor_term!(stmts, data, t2, t; target))
end
_sb_emit_direct_expr!(_stmts, _data, _target::Symbol, f, _t, _summands) =
    error("sbimpl: unsupported direct-summand term `$f`")

# Categorical population-level predictor. Allocates K-1 betas via `_sb_cat`
# and pushes the per-row contribution column into `summands`. K == 1 (single
# level) degenerates to a zero column instead of erroring — see the in-body note.
function _sb_emit_cat!(stmts, data, t::NamedColumn, summands)
    backing = parent(t)
    n_levels, idx = _sb_level_index(parent(backing))
    col_name = Symbol(:cat_, name(t))
    if n_levels < 2
        # Single-level categorical: K-1 = 0 treatment contrasts, so the term
        # degenerates to a zero contribution — the lone level is absorbed by the
        # intercept (or, for an intercept-less predictor, vanishes). Emit a literal
        # zero column instead of erroring, so callers never need an
        # `n_levels > 1 ? " + c" : ""` conditional: `y ~ 1 + c` ≡ `y ~ 1`, and a
        # no-intercept `b ~ c` ≡ 0, at K == 1. (`_sb_cat` already degenerates via
        # `std_normal(; n=0)`, but the assembler has no zero-summand branch —
        # `length(summands)==0` would emit `+()` — so we contribute a column, not skip.)
        data[col_name] = zeros(Float64, length(idx))
        push!(summands, col_name)
        return
    end
    idx_name = Symbol(name(t), :_idx)
    n_name   = Symbol(name(t), :_n_levels)
    data[idx_name] = idx
    data[n_name]   = n_levels
    # Frozen level set drives the K-1 treatment-contrast betas; reprocess
    # re-codes a new df against it and updates the `<x>_n_levels` count key
    # (derived from raw_ref). Dimension-coupled (unseen level / changed count).
    _sb_record_preproc!(data, idx_name,
        PreprocEntry(:factor, _sb_fit_levels(parent(backing)), name(t), true))
    push!(stmts, :($col_name ~ _sb_cat(; x=$idx_name, n_levels=$n_name)))
    push!(summands, col_name)
end

# Expand a ranef LHS term into one-or-more design-matrix column references.
# Continuous/intercept terms produce a single column; categorical NamedColumns
# expand to K-1 treatment-coded dummy columns (level 1 is reference), matching
# the design-matrix that brms / lme4 build for `(1 + c | g)`.
# `gterms` is the full LHS-terms list of the current ranef block, threaded so
# an intercept term (`t === 1`) can probe peer terms in the same block for a
# deterministic length probe (analogous to how `pop_terms` is threaded on the
# population path; see `_sb_pop_cols!`).
function _sb_ranef_cols!(cols, data, stmts, t, gterms=())
    _sb_ranef_cols_dispatch!(cols, data, stmts, t, _sb_cat_levels(t), gterms)
end
_sb_ranef_cols!(cols, data, stmts, t::ExprColumn{typeof(offset)}, gterms=()) =
    error("sbimpl: `offset(...)` is a population-level fixed contribution and cannot appear inside a random-effects term")
_sb_ranef_cols_dispatch!(cols, data, stmts, t, ::Nothing, gterms=()) =
    push!(cols, _sb_predictor_col(t, data, stmts, gterms))
function _sb_ranef_cols_dispatch!(cols, data, _stmts, t, levels, _gterms=())
    n_levels, idx = _sb_level_index(levels)
    n_levels >= 2 || error("sbimpl: categorical ranef term `$(name(t))` needs >= 2 levels (got $n_levels)")
    for lvl in 2:n_levels
        col_name = Symbol(name(t), :_dummy_, lvl)
        data[col_name] = Float64[l == lvl ? 1.0 : 0.0 for l in idx]
        push!(cols, col_name)
    end
end

# Collected ranef handling. Terms that share a grouping symbol are merged into
# a single correlated block (matches vimpl / brms: `(1 | g) + (x | g)` is the
# same as `(1 + x | g)`; `(1 | g) + (0 + x | g)` also collapses -- the two
# terms' LKJ and tau parameters are shared by design). Per-group we build:
#   Z_<target>_<g> = hcat(<col_1>, <col_2>, ...)   # n x K
# and emit one `~ ranef_correlated(...)` (K >= 2) or `~ ranef_intercept(...)`
# (K == 1, intercept-only) call. Scalar-slope-only blocks (K == 1 but not an
# intercept) also go through ranef_correlated -- the math degenerates gracefully.
# `(... | rhs)` -> walker-side group descriptor. Bare NamedColumn and `gr(g)`
# (no kwargs) both collapse to the inner NamedColumn (plain correlated block);
# `gr(g; by=b)` returns `(group, by)` so the emitter allocates the stratified
# `ranef_correlated_by` block. Mirrors vimpl's `_normalize_group`. Also extracts
# brms's `gr(g, id=<sym|str>)`, producing an id Symbol which the caller carries
# alongside the group descriptor to drive cross-formula bucket coalescing.
_sb_id_sym(::Nothing) = nothing
_sb_id_sym(s::Symbol) = s
_sb_id_sym(s::AbstractString) = Symbol(s)
_sb_id_sym(x) = error("sbimpl: `gr(...; id=...)` expects a Symbol or String, got $(typeof(x))")

_sb_normalize_group(g::NamedColumn) = (g, nothing)
_sb_normalize_group(g::MultiMembershipTerm) = (g, nothing)
_sb_normalize_group(g::ExprColumn) = begin
    getf(g) === gr || error("sbimpl: expected NamedColumn or `gr(...)` on RHS of `|`, got `$(getf(g))`")
    args = getargs(g); kw = getkwargs(g)
    length(args) == 1 || error("sbimpl: `gr(...)` expects exactly one positional group, got $(length(args))")
    group_raw = args[1]
    group = _as_named_column(group_raw)
    isnothing(group) && error("sbimpl: `gr(...)` expects a NamedColumn group, got $(typeof(group_raw))")
    by_raw = get(kw, :by, nothing)
    id_sym = _sb_id_sym(get(kw, :id, nothing))
    by_raw === nothing && return (group, id_sym)
    by = _as_named_column(by_raw)
    isnothing(by) && error("sbimpl: `gr(...; by=...)` expects a NamedColumn for `by`, got $(typeof(by_raw))")
    ((group, by), id_sym)
end
_sb_normalize_group(g) = error("sbimpl: expected NamedColumn or `gr(...)` on RHS of `|`, got $(typeof(g))")

# Walker-side key used to coalesce ran_terms. Plain group -> `Symbol`; stratified
# `gr(g, by=b)` -> `(Symbol, Symbol)` so the two don't accidentally merge.
_sb_group_key(g::NamedColumn) = name(g)
_sb_group_key(g::Tuple{NamedColumn,NamedColumn}) = (name(g[1]), name(g[2]))
_sb_group_key(g::MultiMembershipTerm) =
    (:mm, Tuple(name(x) for x in getargs(g)),
     isnothing(getfield(g, :weights)) ? nothing : Tuple(name(x) for x in getfield(g, :weights)),
     getfield(g, :normalize))

_sb_mm_group_names(g::MultiMembershipTerm) = Tuple(name(x) for x in getargs(g))
_sb_mm_weight_names(g::MultiMembershipTerm) = begin
    weights = getfield(g, :weights)
    isnothing(weights) ? nothing : Tuple(name(x) for x in weights)
end
function _sb_mm_suffix(g::MultiMembershipTerm)
    groups = join(String.(_sb_mm_group_names(g)), "__")
    weights = _sb_mm_weight_names(g)
    weighted = isnothing(weights) ? "" : "__w__$(join(String.(weights), "__"))"
    raw = getfield(g, :normalize) ? "" : "__raw"
    Symbol("mm__", groups, weighted, raw)
end

# Normalize `(expr | group)` vs `(expr | id | group)` into (id_sym, lhs, descriptor)
# where id_sym === nothing signals the plain (non-ID'd) case. Surface-level
# `gr(g, id=...)` with a plain `|` likewise produces a non-nothing id_sym.
# `|` rewritten by macro.jl: ExprColumn{|}(lhs, id_sym::Symbol, group) for `|ID|`.
function _sb_ranef_parts(rt::ExprColumn)
    getf(rt) === (|) || error("sbimpl: expected `|` ExprColumn, got `$(getf(rt))`")
    args = getargs(rt)
    if length(args) == 2
        lhs, raw_group = args
        desc, id_sym = _sb_normalize_group(raw_group)
        desc isa MultiMembershipTerm && id_sym !== nothing && error(
            "sbimpl: `mm(...)` cannot be combined with `gr(...; id=...)`; ",
            "multi-membership terms already define one shared coefficient block")
        (id_sym, lhs, desc)
    elseif length(args) == 3
        lhs, id_sym_raw, raw_group = args
        id_sym = _as_symbol(id_sym_raw)
        isnothing(id_sym) && error("sbimpl: `(e | ID | g)` middle must be a Symbol, got $(typeof(id_sym_raw))")
        desc, gr_id_sym = _sb_normalize_group(raw_group)
        desc isa MultiMembershipTerm && error(
            "sbimpl: `(e | ID | mm(...))` is not supported; `mm(...)` already ",
            "defines one shared coefficient block across its membership columns")
        gr_id_sym === nothing ||
            error("sbimpl: `(e | ID | g)` cannot also carry `gr(g, id=...)` (got `$gr_id_sym`)")
        (id_sym, lhs, desc)
    else
        error("sbimpl: malformed ranef term, expected 2 or 3 args, got $(length(args))")
    end
end

# Per-group level of `g` -> stratum level of `by`. Errors if any group level
# straddles multiple strata. Ported from vimpl's `_stratum_idx`.
_sb_stratum_idx(g_idx::AbstractVector{Int}, b_idx::AbstractVector{Int}, gname, bname) = begin
    m_groups = maximum(g_idx)
    mapping = zeros(Int, m_groups)
    for (gi, bi) in zip(g_idx, b_idx)
        if mapping[gi] == 0
            mapping[gi] = bi
        elseif mapping[gi] != bi
            error("sbimpl: gr($gname, by=$bname): group level $gi straddles multiple strata ($(mapping[gi]) vs $bi)")
        end
    end
    mapping
end

function _sb_emit_ranefs!(stmts, data, target::Symbol, ran_terms, summands;
                           id_lookup=_sb_empty_id_lookup(),
                           brmi_key::Symbol=target,
                           cv_groups=Set{Symbol}(),
                           centered_groups=Set{Symbol}())
    isempty(ran_terms) && return
    # Partition: ID'd terms route to the pre-emitted shared bucket; plain terms
    # coalesce per-target via the existing `ranef_correlated` block. Bare-
    # NamedColumn and gr-by groups key differently so they never coalesce.
    plain_keys_seen = Any[]
    plain_by_group = Dict{Any, Vector{Any}}()
    plain_descs = Dict{Any, Any}()
    id_keys_seen = Any[]
    id_terms_by_bucket = Dict{Tuple{Symbol,Any}, Vector{Any}}()
    for rt in ran_terms
        id_sym, lhs, desc = _sb_ranef_parts(rt)
        if id_sym === nothing
            k = _sb_group_key(desc)
            haskey(plain_by_group, k) || (push!(plain_keys_seen, k); plain_by_group[k] = Any[]; plain_descs[k] = desc)
            append!(plain_by_group[k], _sb_terms(lhs))
        else
            k = (id_sym, _sb_group_key(desc))
            haskey(id_terms_by_bucket, k) || (push!(id_keys_seen, k); id_terms_by_bucket[k] = Any[])
            append!(id_terms_by_bucket[k], _sb_terms(lhs))
        end
    end
    for k in plain_keys_seen
        gterms = plain_by_group[k]
        desc = plain_descs[k]
        isempty(gterms) && error("sbimpl: ranef `(… | $k)` has no terms after dropping `0`")
        _sb_emit_ranef_block!(stmts, data, target, desc, gterms, summands; cv_groups, centered_groups)
    end
    for k in id_keys_seen
        gterms = id_terms_by_bucket[k]
        isempty(gterms) && error("sbimpl: ranef `(… | $(k[1]) | $(k[2]))` has no terms after dropping `0`")
        # The `|ID|` bucket's shared draws block is emitted in prepass 2, which
        # receives `cv_groups` / `centered_groups` and picks the matching
        # `ranef_correlated_draws{,_cv,_centered}` variant there. Nothing to do
        # here beyond slicing it per sub-formula.
        info = get(id_lookup, (brmi_key, k), nothing)
        info === nothing && error("sbimpl: internal — no pre-emitted bucket for (target=$brmi_key, id=$(k[1]), group=$(k[2]))")
        _sb_emit_id_ranef_block!(stmts, data, target, info, gterms, summands)
    end
end

# Emit a single ranef block for one normalized group descriptor.
# `cv_groups`: groups whose RE should be sized cv-contagiously (opt-in, for CV
# model artifacts). There is no separate `_cv` submodel -- the SIZE EXPRESSION
# passed here is the whole mechanism: `maximum(<g>_idx)` instead of the data
# scalar `n_<g>` carries the taint from `maybecv(:<g>_idx)` into the declared
# size and flips the RE to a generated-quantities population re-draw. Empty by
# default. See the cv-contagion note above `ranef_intercept` in this file.
function _sb_emit_ranef_block!(stmts, data, target::Symbol, group::NamedColumn, gterms, summands;
                                cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}())
    g_backing = _as_data_column(parent(group))
    isnothing(g_backing) && error("sbimpl: group `$(name(group))` must be a raw data column")
    g = name(group)
    is_cv = g in cv_groups
    is_centered = g in centered_groups
    n_levels, g_idx = _sb_level_index(parent(g_backing))
    idx_name = Symbol(g, :_idx)
    n_name   = Symbol(:n_, g)
    data[idx_name] = g_idx
    data[n_name]   = n_levels
    r_name = Symbol(:r_, target, :_, g)
    # The size expression the non-centered submodels are given. Bound to a named
    # local in the cv case so it appears once as `int <r>_n_g = max(<g>_idx);`
    # rather than being inlined into every declaration -- same shape as
    # `_sb_emit_id_bucket_sampling!`.
    n_groups_expr = n_name
    if is_cv
        n_groups_expr = Symbol(r_name, :_n_g)
        push!(stmts, :($n_groups_expr = maximum($idx_name)))
    end
    if length(gterms) == 1 && gterms[1] === 1
        # (1 | g) fast/equivalent path -- stays on ranef_intercept so the
        # emitted Stan matches existing sb.3 smoke tests bit for bit.
        if is_centered
            push!(stmts, :($r_name ~ ranef_intercept_centered(; group_idx=$idx_name, n_groups=$n_name)))
        else
            push!(stmts, :($r_name ~ ranef_intercept(; group_idx=$idx_name, n_groups=$n_groups_expr)))
        end
    else
        col_exprs = Any[]
        for t in gterms
            _sb_ranef_cols!(col_exprs, data, stmts, t, gterms)
        end
        Z_name = Symbol(:Z_, target, :_, g)
        k_name = Symbol(:n_terms_, target, :_, g)
        data[k_name] = length(col_exprs)
        push!(stmts, :($Z_name = $(Expr(:call, :hcat, col_exprs...))))
        if is_centered
            push!(stmts, :($r_name ~ ranef_correlated_centered(;
                Z=$Z_name, group_idx=$idx_name,
                n_groups=$n_name, n_terms=$k_name)))
        else
            push!(stmts, :($r_name ~ ranef_correlated(;
                Z=$Z_name, group_idx=$idx_name,
                n_groups=$n_groups_expr, n_terms=$k_name)))
        end
    end
    push!(summands, r_name)
end

function _sb_emit_ranef_block!(stmts, data, target::Symbol,
                                term::MultiMembershipTerm, gterms, summands;
                                cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}())
    group_names = _sb_mm_group_names(term)
    weight_names = _sb_mm_weight_names(term)
    cv_hit = intersect(Set(group_names), cv_groups)
    isempty(cv_hit) || error(
        "sbimpl: cv-contagious sizing for `mm(...)` is not yet supported ",
        "(requested membership columns: $(collect(cv_hit)))")
    centered_hit = intersect(Set(group_names), centered_groups)
    isempty(centered_hit) || error(
        "sbimpl: centered parameterization for `mm(...)` is not supported ",
        "(requested membership columns: $(collect(centered_hit))); leave this ",
        "shared block non-centered")

    raw_groups = Tuple(begin
        backing = _as_data_column(parent(g))
        isnothing(backing) && error(
            "sbimpl: `mm(...)` group `$(name(g))` must be a raw data column")
        parent(backing)
    end for g in getargs(term))
    raw_weights = if isnothing(getfield(term, :weights))
        nothing
    else
        Tuple(begin
            backing = _as_data_column(parent(w))
            isnothing(backing) && error(
                "sbimpl: `mm(...)` weight `$(name(w))` must be a raw data column")
            parent(backing)
        end for w in getfield(term, :weights))
    end
    prepared = _sb_prepare_mm(raw_groups, raw_weights, getfield(term, :normalize);
                              group_names, weight_names)

    suffix = _sb_mm_suffix(term)
    idx_name = Symbol(suffix, :_idx)
    weight_name = Symbol(suffix, :_weights)
    n_name = Symbol(:n_, suffix)
    n_obs_name = Symbol(:n_obs_, suffix)
    n_memberships_name = Symbol(:n_memberships_, suffix)
    data[idx_name] = prepared.group_idx
    data[weight_name] = prepared.weights
    data[n_name] = length(prepared.levels)
    data[n_obs_name] = prepared.n_obs
    data[n_memberships_name] = prepared.n_memberships
    const_ = (; levels=prepared.levels, weight_key=weight_name,
              n_groups_key=n_name, n_obs_key=n_obs_name,
              n_memberships_key=n_memberships_name,
              normalize=getfield(term, :normalize))
    raw_ref = (; groups=group_names, weights=weight_names)
    _sb_record_preproc!(data, idx_name,
        PreprocEntry(:multi_membership, const_, raw_ref, true))

    b_name = Symbol(:b_, target, :_, suffix)
    r_name = Symbol(:r_, target, :_, suffix)
    if length(gterms) == 1 && gterms[1] === 1
        push!(stmts, :($b_name ~ ranef_intercept_draws(;
            group_idx=$idx_name, n_groups=$n_name)))
        push!(stmts, :($r_name = multi_membership_intercept(
            $b_name, $idx_name, $weight_name, $n_obs_name, $n_memberships_name)))
    else
        col_exprs = Any[]
        for t in gterms
            _sb_ranef_cols!(col_exprs, data, stmts, t, gterms)
        end
        Z_name = Symbol(:Z_, target, :_, suffix)
        k_name = Symbol(:n_terms_, target, :_, suffix)
        data[k_name] = length(col_exprs)
        push!(stmts, :($Z_name = $(Expr(:call, :hcat, col_exprs...))))
        push!(stmts, :($b_name ~ ranef_correlated_draws(;
            Z=$Z_name, group_idx=$idx_name,
            n_groups=$n_name, n_terms=$k_name)))
        push!(stmts, :($r_name = multi_membership_correlated(
            $Z_name, $b_name, $idx_name, $weight_name,
            $n_obs_name, $n_memberships_name)))
    end
    push!(summands, r_name)
end

function _sb_emit_ranef_block!(stmts, data, target::Symbol, group::Tuple{NamedColumn,NamedColumn}, gterms, summands;
                                cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}())
    gcol, bcol = group
    g_backing = _as_data_column(parent(gcol))
    b_backing = _as_data_column(parent(bcol))
    isnothing(g_backing) && error("sbimpl: group `$(name(gcol))` must be a raw data column")
    isnothing(b_backing) && error("sbimpl: `by=$(name(bcol))` must be a raw data column")
    g, b = name(gcol), name(bcol)
    g in cv_groups && error(
        "sbimpl: cv-contagious sizing requested for group `$g`, but `$g` is a ",
        "stratified `gr($g, by=$b)` ranef whose `z` is declared via typed-LHS. ",
        "StanBlocks' typed-LHS forward derives cv from RHS call-args, not the ",
        "declared size, so a cv size yields a cv *parameter*, not a generated-",
        "quantities re-draw. Making `by=` blocks cv-contagious needs a bare-`z` ",
        "rewrite or a StanBlocks typed-LHS fix -- not yet supported.")
    g in centered_groups && error(
        "sbimpl: centered parameterization requested for group `$g`, but `$g` is ",
        "a stratified `gr($g, by=$b)` ranef whose per-group draw goes through the ",
        "typed-LHS `ranef_correlated_by` path (one Cholesky per stratum). The ",
        "centered variants emit a plate over a single shared covariance and have ",
        "no per-stratum form -- not yet supported. Use a plain `(… | $g)` or ",
        "`(… |ID| $g)` ranef, or leave `$g` non-centered.")
    n_groups, g_idx = _sb_level_index(parent(g_backing))
    n_strata, b_idx = _sb_level_index(parent(b_backing))
    stratum_idx = _sb_stratum_idx(g_idx, b_idx, g, b)
    # Block-local names include both `g` and `b` so this block never clashes
    # with a plain `(… | g)` block against the same group column.
    suffix = Symbol(g, :__by__, b)
    idx_name     = Symbol(suffix, :_idx)
    n_name       = Symbol(:n_, suffix)
    s_idx_name   = Symbol(suffix, :_stratum_idx)
    n_strata_nm  = Symbol(:n_strata_, suffix)
    data[idx_name]    = g_idx
    data[n_name]      = n_groups
    data[s_idx_name]  = stratum_idx
    data[n_strata_nm] = n_strata
    r_name = Symbol(:r_, target, :_, suffix)
    col_exprs = Any[]
    for t in gterms
        _sb_ranef_cols!(col_exprs, data, stmts, t, gterms)
    end
    Z_name = Symbol(:Z_, target, :_, suffix)
    k_name = Symbol(:n_terms_, target, :_, suffix)
    data[k_name] = length(col_exprs)
    push!(stmts, :($Z_name = $(Expr(:call, :hcat, col_exprs...))))
    push!(stmts, :($r_name ~ ranef_correlated_by(;
        Z=$Z_name, group_idx=$idx_name,
        n_groups=$n_name, n_terms=$k_name,
        stratum_idx=$s_idx_name, n_strata=$n_strata_nm)))
    push!(summands, r_name)
end

# Pre-pass: harvest every `|ID|` ranef across all sub-formulas and bucket by
# (id_sym, group_key). Returns an OrderedDict keyed by bucket, carrying
# `(group_desc, per_target::Vector{Pair{Symbol, Vector}})` in appearance order.
# Non-ID'd ranef terms are left alone for the existing per-target emitter.
function _sb_collect_id_buckets(brmi::BRMI)
    buckets = OrderedCollections.OrderedDict{Tuple{Symbol,Any}, Any}()
    for (brmi_key, op_nc) in pairs(brmi.operations)
        op = _as_expr_column(parent(op_nc)); isnothing(op) && continue
        getf(op) === (~) || continue
        _, rhs_raw = getargs(op, 2)
        rhs = _as_expr_column(rhs_raw); isnothing(rhs) && continue
        for t_raw in _sb_terms(rhs)
            t = _as_expr_column(t_raw); isnothing(t) && continue
            getf(t) === (|) || continue
            id_sym, lhs, desc = _sb_ranef_parts(t)
            id_sym === nothing && continue
            k = (id_sym, _sb_group_key(desc))
            if !haskey(buckets, k)
                buckets[k] = (group_desc=desc, per_target=Pair{Symbol,Vector{Any}}[])
            else
                # Consistency check: same id must always pair with the same group.
                _sb_group_desc_matches(buckets[k].group_desc, desc) ||
                    error("sbimpl: `|$id_sym|` sees conflicting grouping factors ($(buckets[k].group_desc) vs $desc)")
            end
            push!(buckets[k].per_target, brmi_key => _sb_terms(lhs))
        end
    end
    buckets
end

# Expand one collected `|ID|` bucket with the exact ranef-column emitter used
# by Stan lowering. This is the single source of truth for public margin
# addresses: categorical terms therefore expose their emitted dummy-column
# labels, and formula order is preserved across predictors and terms.
function _sb_id_bucket_margins(bucket)
    out = NamedTuple[]
    scratch_data = Dict{Symbol,Any}()
    scratch_stmts = Any[]
    for (predictor, terms) in bucket.per_target
        for t in terms
            if t isa Integer
                t == 1 || error(
                    "ranefcoefnames: unsupported integer random-effect term `$t`")
                push!(out, (; predictor, coefficient=:Intercept))
                continue
            end
            cols = Any[]
            try
                _sb_ranef_cols!(cols, scratch_data, scratch_stmts, t, terms)
            catch err
                error("ranefcoefnames: cannot resolve random-effect column(s) for " *
                      "predictor `$predictor`: $(sprint(showerror, err))")
            end
            for col in cols
                col isa Symbol || error(
                    "ranefcoefnames: random-effect column for predictor " *
                    "`$predictor` is not symbol-addressable: $(repr(col))")
                push!(out, (; predictor, coefficient=col))
            end
        end
    end
    out
end

"""
    ranefcoefnames(brmi::BRMI, id::Symbol) -> Union{Vector{NamedTuple},Nothing}

Ordered `(predictor, coefficient)` addresses of the marginal SDs in the
shared random-effect block selected by public `|ID|` symbol `id`. The k-th
entry labels the k-th `tau` element emitted by the SBBRMI backend. Categorical
random slopes use the exact treatment-contrast column symbols emitted into the
random-effect design matrix.

Returns `nothing` when `id` is absent. Reusing one ID with multiple grouping
factors is ambiguous on the public ID-only surface and raises.
"""
function ranefcoefnames(brmi::BRMI, id::Symbol)
    buckets = _sb_collect_id_buckets(brmi)
    matches = Pair[k => bucket for (k, bucket) in pairs(buckets) if first(k) === id]
    isempty(matches) && return nothing
    length(matches) == 1 || error(
        "ranefcoefnames: `|$id|` identifies $(length(matches)) random-effect " *
        "blocks with different grouping factors; the ID-only address is ambiguous")
    _sb_id_bucket_margins(last(only(matches)))
end

function _sb_ranef_sd_rate(spec)
    T = _as_distribution_type(spec.family)
    (!isnothing(T) && T <: Exponential) || error(
        "sbimpl: `effect(sd, ...)` currently supports `Exponential(scale)`; " *
        "got `$(spec.family)`. Unmentioned margins retain the historical " *
        "half-standard-normal prior.")
    isempty(spec.keywords) || error(
        "sbimpl: `effect(sd, ...) ~ Exponential(...)` does not accept keywords")
    args = map(_sb_effect_prior_arg, spec.arguments)
    length(args) in (0, 1) || error(
        "sbimpl: `Exponential` SD priors expect zero or one Julia scale argument")
    scale = isempty(args) ? 1.0 : only(args)
    scale isa Real || error(
        "sbimpl: random-effect SD prior scale must be a numeric formula " *
        "constant, got $(repr(scale))")
    isfinite(scale) && scale > 0 || error(
        "sbimpl: random-effect SD prior scale must be finite and strictly " *
        "positive, got $scale")
    # Distributions.Exponential uses scale; Stan's exponential_lpdf uses rate.
    Float64(1.0 / scale)
end

function _sb_ranef_lkj(spec, n_terms::Int)
    T = _as_distribution_type(spec.family)
    (!isnothing(T) && T <: LKJCholesky) || error(
        "sbimpl: `effect(cor, ID)` expects `LKJCholesky(K, eta)`; " *
        "got `$(spec.family)`")
    isempty(spec.keywords) || error(
        "sbimpl: `effect(cor, ID) ~ LKJCholesky(...)` does not accept keywords")
    args = map(_sb_effect_prior_arg, spec.arguments)
    length(args) == 2 || error(
        "sbimpl: `LKJCholesky` correlation priors require dimension K and eta")
    k, eta = args
    k isa Integer || error(
        "sbimpl: `LKJCholesky(K, eta)` dimension K must be an integer " *
        "formula constant, got $(repr(k))")
    k == n_terms || error(
        "sbimpl: `LKJCholesky($k, ...)` does not match the addressed " *
        "random-effect block width $n_terms")
    eta isa Real || error(
        "sbimpl: LKJ eta must be a numeric formula constant, got $(repr(eta))")
    isfinite(eta) && eta > 0 || error(
        "sbimpl: LKJ eta must be finite and strictly positive, got $eta")
    Float64(eta)
end

function _sb_ranef_margin_index(spec, margins)
    if isnothing(spec.predictor)
        return nothing
    elseif isnothing(spec.coefficient)
        hits = findall(m -> m.predictor === spec.predictor, margins)
        isempty(hits) && error(
            "sbimpl: `effect(sd, $(spec.id), $(spec.predictor))` matches no " *
            "random-effect margin. Inspect `ranefcoefnames(brmi, :$(spec.id))`.")
        length(hits) == 1 || error(
            "sbimpl: `effect(sd, $(spec.id), $(spec.predictor))` is ambiguous " *
            "because that predictor contributes $(length(hits)) margins; use " *
            "`effect(sd, $(spec.id), $(spec.predictor), coefficient)`.")
        return only(hits)
    else
        hits = findall(m -> m.predictor === spec.predictor &&
                            m.coefficient === spec.coefficient, margins)
        isempty(hits) && error(
            "sbimpl: `effect(sd, $(spec.id), $(spec.predictor), " *
            "$(spec.coefficient))` matches no random-effect margin. Inspect " *
            "`ranefcoefnames(brmi, :$(spec.id))` for valid addresses.")
        length(hits) == 1 || error(
            "sbimpl: `effect(sd, $(spec.id), $(spec.predictor), " *
            "$(spec.coefficient))` matches $(length(hits)) margins and is " *
            "therefore ambiguous")
        return only(hits)
    end
end

# Resolve formula-level random-effect prior statements onto collected ID
# buckets. The result is keyed exactly like `id_buckets`; entries are present
# only for explicitly configured buckets, preserving default emission byte for
# byte when the formula contains no ranef effect statements.
function _sb_ranef_effect_overrides(brmi::BRMI, id_buckets)
    specs = ranef_effect_priors(brmi)
    isempty(specs) && return Dict{Tuple{Symbol,Any},NamedTuple}()

    states = Dict{Tuple{Symbol,Any},Dict{Symbol,Any}}()
    for spec in specs
        matches = Any[k for k in keys(id_buckets) if first(k) === spec.id]
        isempty(matches) && error(
            "sbimpl: `effect($(spec.class), $(spec.id), ...)` matches no shared " *
            "`|$(spec.id)|` random-effect block")
        length(matches) == 1 || error(
            "sbimpl: public `|$(spec.id)|` addresses $(length(matches)) blocks " *
            "with different grouping factors; use a unique ID")
        key = only(matches)
        bucket = id_buckets[key]
        bucket.group_desc isa Tuple && error(
            "sbimpl: covariance-prior effects for stratified " *
            "`|$(spec.id)| gr(..., by=...)` buckets are not yet supported")
        margins = _sb_id_bucket_margins(bucket)
        state = get!(states, key) do
            Dict{Symbol,Any}(
                :margins => margins,
                :sd_default => nothing,
                :sd_overrides => Dict{Int,Float64}(),
                :lkj_eta => nothing,
            )
        end

        if spec.class === :cor
            state[:lkj_eta] === nothing || error(
                "sbimpl: duplicate correlation prior for `effect(cor, $(spec.id))`")
            state[:lkj_eta] = _sb_ranef_lkj(spec, length(margins))
            continue
        end

        rate = _sb_ranef_sd_rate(spec)
        idx = _sb_ranef_margin_index(spec, margins)
        if isnothing(idx)
            state[:sd_default] === nothing || error(
                "sbimpl: duplicate block SD prior for `effect(sd, $(spec.id))`")
            state[:sd_default] = rate
        else
            haskey(state[:sd_overrides], idx) && error(
                "sbimpl: multiple SD prior statements resolve to margin " *
                "$(margins[idx]) of `|$(spec.id)|`")
            state[:sd_overrides][idx] = rate
        end
    end

    out = Dict{Tuple{Symbol,Any},NamedTuple}()
    for (key, state) in states
        margins = state[:margins]
        default_rate = state[:sd_default]
        sd_family = fill(isnothing(default_rate) ? 0 : 1, length(margins))
        sd_rate = fill(isnothing(default_rate) ? 1.0 : default_rate, length(margins))
        for (idx, rate) in state[:sd_overrides]
            sd_family[idx] = 1
            sd_rate[idx] = rate
        end
        out[key] = (; sd_family, sd_rate,
                    lkj_eta=isnothing(state[:lkj_eta]) ? 1.0 : state[:lkj_eta],
                    margins)
    end
    out
end

# ---- group-block prepass (Prepass 2.5) ---------------------------------------
#
# Scan brmi.operations for declaring terms anywhere in the model: a term `f`
# with a `_sb_term_group_block` declaration, appearing EITHER as a whole-RHS
# parameter submodel (`mu ~ f(...)`, like bordet) OR as a predictor summand
# (`y ~ 1 + hsgp(t, by=g)`). We walk every `~` op's RHS summands (`_sb_terms`),
# which covers both: bordet is a single summand of its RHS, hsgp is one of several.
# Every declaring summand is its own term INSTANCE — there is no `(key, f)`
# dedup, so N instances of the same term (e.g. `hsgp(t, by=g) + hsgp(s, by=g)`)
# each get collected and allocated their own block. Identical instances (same
# block name) are coalesced later, at emit time. A subsequent emit step
# allocates one block per declared field and builds the lookup.
function _sb_collect_group_block_terms(brmi::BRMI)
    result = Any[]
    for (key, op_nc) in pairs(brmi.operations)
        op = _as_expr_column(parent(op_nc)); isnothing(op) && continue
        getf(op) === (~) || continue
        _, rhs = getargs(op, 2)
        for t in _sb_terms(rhs)
            te = _as_expr_column(t); isnothing(te) && continue
            f = getf(te)
            fields = _sb_structured_fields(_sb_term_group_block(f, te), f)
            isnothing(fields) && continue
            push!(result, (; key, f, rhs_e=te, fields))
        end
    end
    result
end

# Resolve a field's grouping column from its group spec + the term call.
function _sb_resolve_group_col(gspec, rhs_e, data)
    haskey(gspec, :fn) && return gspec.fn(rhs_e, data)
    if haskey(gspec, :kwarg)
        kw = getkwargs(rhs_e)
        haskey(kw, gspec.kwarg) || error(
            "sbimpl: structured-latent group kwarg `$(gspec.kwarg)=` missing in `$(nameof(getf(rhs_e)))` call")
        nc = _as_named_column(kw[gspec.kwarg])
        isnothing(nc) && error(
            "sbimpl: `$(gspec.kwarg)=` must be a NamedColumn group, got $(typeof(kw[gspec.kwarg]))")
        return nc
    end
    pos = gspec.arg_pos
    args = getargs(rhs_e)
    pos <= length(args) || error(
        "sbimpl: structured-latent group_arg_pos=$pos but `$(nameof(getf(rhs_e)))` has $(length(args)) args")
    nc = _as_named_column(args[pos])
    isnothing(nc) && error(
        "sbimpl: structured-latent group arg $pos must be a NamedColumn, got $(typeof(args[pos]))")
    nc
end

# The grouping column's NAME, without materialising data. Used to build the
# disambiguating block name (`b_<field>_<gname>`) in BOTH the emit pass and the
# find pass so they agree on the per-instance block. For a `group_fn` spec this
# is `fn_name` by construction (the synthesised column is named `fn_name`), so
# `data` is never needed here — only `_sb_resolve_group_col` (emit) needs it.
_sb_group_name(gspec, rhs_e) =
    haskey(gspec, :fn)    ? gspec.fn_name :
    haskey(gspec, :kwarg) ? name(_as_named_column(getkwargs(rhs_e)[gspec.kwarg])) :
                            name(_as_named_column(getargs(rhs_e)[gspec.arg_pos]))

# Emit the per-group block draw for one field, dispatching on its prior spec.
# Each block is an n_groups × n_per_group matrix referenced by row = group.
_sb_emit_block_draw!(stmts, prior::Symbol, block_name, idx_name, n_name, n_terms_name, suffix) = begin
    if prior === :correlated_normal
        push!(stmts, :($block_name ~ ranef_correlated_draws(;
            group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name)))
    elseif prior === :iid_normal
        flat_name = Symbol(:zflat_, suffix)
        push!(stmts, :($flat_name ~ std_normal(; n=$n_name * $n_terms_name)))
        push!(stmts, :($block_name = reshape($flat_name, $n_terms_name, $n_name)'))
    else
        error("sbimpl: unknown structured-latent prior symbol `:$prior` ",
              "(expected :correlated_normal or :iid_normal)")
    end
end
# Element-wise prior over the matrix, reusing the scalar-prior dist-name table.
function _sb_emit_block_draw!(stmts, prior::NamedTuple, block_name, idx_name, n_name, n_terms_name, suffix)
    stan_name = _sb_stan_dist_name(prior.dist)
    isnothing(stan_name) && error(
        "sbimpl: structured-latent prior dist `$(prior.dist)` has no `_sb_stan_dist_name` mapping")
    pos_args = _sb_stan_dist_args(
        prior.dist, map(_sb_prior_arg, get(prior, :args, ())))
    flat_name = Symbol(:zflat_, suffix)
    kw = Any[Expr(:kw, :n, :($n_name * $n_terms_name))]
    haskey(prior, :lower) && push!(kw, Expr(:kw, :lower, prior.lower))
    haskey(prior, :upper) && push!(kw, Expr(:kw, :upper, prior.upper))
    rhs_call = Expr(:call, stan_name, pos_args..., Expr(:parameters, kw...))
    push!(stmts, Expr(:call, :~, flat_name, rhs_call))
    push!(stmts, :($block_name = reshape($flat_name, $n_terms_name, $n_name)'))
end

# Allocate one block per declared field and return a lookup
# Dict{block_name::Symbol => (; block_name, idx_name, n_per_group)}.
# Keyed by the fully-disambiguated block NAME (`b_<field>_<gname>`), so N
# distinct instances (different x and/or group) each get their own block and
# only genuinely identical instances (same block name) are coalesced.
function _sb_emit_group_blocks!(stmts, data, gb_terms)
    lookup = Dict{Symbol, NamedTuple}()
    for (; f, rhs_e, fields) in gb_terms
        for fld in fields
            group_col = _sb_resolve_group_col(fld.group, rhs_e, data)
            gname = name(group_col)
            # Block name `b_<field>_<gname>`; for legacy single-field terms
            # field == nameof(f), so this is byte-for-byte the old name. For hsgp
            # the field name embeds x, so two HSGPs on the same group never collide.
            suffix = Symbol(fld.name, :_, gname)
            block_name = Symbol(:b_, suffix)
            haskey(lookup, block_name) && continue   # identical instance already allocated
            n_terms_name = Symbol(:n_terms_, suffix)
            data[n_terms_name] = fld.n_per_group
            idx_name, n_name = _sb_ensure_group_data!(data, group_col)
            _sb_emit_block_draw!(stmts, fld.prior, block_name, idx_name, n_name, n_terms_name, suffix)
            lookup[block_name] = (; block_name, idx_name, n_per_group=fld.n_per_group)
        end
    end
    lookup
end

# Look up block_info for a declaring term call at emit time, or nothing. Rebuilds
# each field's block name (via `_sb_group_name`, matching the emit pass) and
# resolves it against the block-name-keyed lookup. Returns a NamedTuple carrying
# a `fields` map (field-name => per-field info); for single-field terms the lone
# field's keys are also spliced at top level so legacy consumers that destructure
# `(; block_name, idx_name)` keep working.
function _sb_find_group_block(f, rhs_e, group_block_lookup)
    fields = _sb_structured_fields(_sb_term_group_block(f, rhs_e), f)
    isnothing(fields) && return nothing
    fmap = Dict{Symbol,NamedTuple}()
    for fld in fields
        block_name = Symbol(:b_, fld.name, :_, _sb_group_name(fld.group, rhs_e))
        info = get(group_block_lookup, block_name, nothing)
        isnothing(info) && return nothing
        fmap[fld.name] = info
    end
    isempty(fmap) && return nothing
    first_info = fmap[first(fields).name]
    (; first_info..., fields=fmap)
end

_sb_group_desc_matches(a::NamedColumn, b::NamedColumn) = name(a) === name(b)
_sb_group_desc_matches(a::Tuple{NamedColumn,NamedColumn}, b::Tuple{NamedColumn,NamedColumn}) =
    name(a[1]) === name(b[1]) && name(a[2]) === name(b[2])
_sb_group_desc_matches(_, _) = false

# Column count for a ranef term (without emitting). `1` -> 1 (intercept);
# scalar NamedColumn -> 1; categorical NamedColumn -> (n_levels-1); ExprColumn
# submodel terms (mo/s/ar/me) -> 1.
_sb_ranef_term_ncols(t::Int, _) = t == 0 ? 0 : 1
_sb_ranef_term_ncols(t::NamedColumn, _data) = _sb_ranef_named_ncols(_sb_cat_levels(t))
_sb_ranef_named_ncols(::Nothing) = 1
_sb_ranef_named_ncols(levels) = _sb_level_index(levels)[1] - 1
_sb_ranef_term_ncols(::ExprColumn, _) = 1
_sb_ranef_term_ncols(t, _) = error("sbimpl: unsupported ranef term $(typeof(t)): $t")

# Emit one shared `b_<id>_<g> ~ ranef_correlated_draws(...)` per bucket, compute
# per-target column ranges, stash `group_idx` / `n_groups` / `n_terms_<id>_<g>`
# in `data`, and return the lookup table consumed by `_sb_emit_ranefs!`.
#
# `cv_groups` / `centered_groups` select the `_cv` / `_centered` draws variant
# for a bucket whose grouping factor is opted in. Both are known at `SBBRMI`
# construction, i.e. BEFORE this prepass runs, so there is no ordering obstacle
# to threading them here -- only the plain-group spelling has a variant, and the
# stratified `gr(g, by=b)` bucket still errors (see `_sb_emit_id_bucket_sampling!`).
function _sb_emit_id_buckets!(stmts, data, buckets;
                              cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(),
                              ranef_effect_overrides=Dict{Tuple{Symbol,Any},NamedTuple}())
    lookup = _sb_empty_id_lookup()
    for (k, bucket) in pairs(buckets)
        id_sym, _ = k
        desc = bucket.group_desc
        suffix = _sb_id_bucket_suffix(id_sym, desc)
        bucket_name = Symbol(:b_, suffix)
        n_terms_name = Symbol(:n_terms_, suffix)
        cursor = 0
        per_target_ranges = Pair{Symbol,UnitRange{Int}}[]
        for (brmi_key, terms) in bucket.per_target
            ncols = sum(_sb_ranef_term_ncols(t, data) for t in terms; init=0)
            ncols > 0 || error("sbimpl: `|$id_sym|` bucket sees empty term list for target `$brmi_key`")
            push!(per_target_ranges, brmi_key => (cursor+1):(cursor+ncols))
            cursor += ncols
        end
        n_terms_total = cursor
        n_terms_total >= 1 || error("sbimpl: `|$id_sym|` bucket has zero terms")
        data[n_terms_name] = n_terms_total
        ranef_effect = get(ranef_effect_overrides, k, nothing)
        idx_name = _sb_emit_id_bucket_sampling!(stmts, data, bucket_name, n_terms_name, desc;
                                                cv_groups, centered_groups, id_sym,
                                                ranef_effect)
        for (brmi_key, cols) in per_target_ranges
            lookup[(brmi_key, k)] = (; bucket_name, cols, idx_name, suffix)
        end
    end
    lookup
end

_sb_id_bucket_suffix(id_sym, g::NamedColumn) = Symbol(id_sym, :_, name(g))
_sb_id_bucket_suffix(id_sym, g::Tuple{NamedColumn,NamedColumn}) =
    Symbol(id_sym, :_, name(g[1]), :__by__, name(g[2]))

# Emit the shared `b_<suffix> ~ …_draws(...)` statement for one ID bucket.
# Plain group -> `ranef_correlated_draws` (or its `_cv` / `_centered` variant
# when the group is opted in); `gr(g, by=b)` group -> stratified
# `ranef_correlated_by_draws`, which has neither variant. Returns the idx_name
# callers use to slice the draw matrix per sub-formula.
function _sb_emit_id_bucket_sampling!(stmts, data, bucket_name, n_terms_name, g::NamedColumn;
                                      cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(),
                                      id_sym=nothing, ranef_effect=nothing)
    idx_name, n_name = _sb_ensure_group_data!(data, g)
    gname = name(g)
    sd_family = isnothing(ranef_effect) ? nothing : Expr(:vect, ranef_effect.sd_family...)
    sd_rate = isnothing(ranef_effect) ? nothing : Expr(:vect, ranef_effect.sd_rate...)
    lkj_eta = isnothing(ranef_effect) ? nothing : ranef_effect.lkj_eta
    if gname in cv_groups
        # Same submodel as the default branch; only the SIZE EXPRESSION differs.
        # Tracing `maximum(<g>_idx)` at the CALL SITE carries the cv taint on
        # `<g>_idx` into the submodel's declared size, so a `maybecv(:<g>_idx)`
        # mark flips the whole block to a generated-quantities re-draw. Value and
        # column-major layout are unchanged. Bound to a named local first so the
        # size appears once as `int <b>_n_g = max(<g>_idx);` instead of being
        # inlined into every declaration.
        n_cv_name = Symbol(bucket_name, :_n_g)
        push!(stmts, :($n_cv_name = maximum($idx_name)))
        if isnothing(ranef_effect)
            push!(stmts, :($bucket_name ~ ranef_correlated_draws(;
                group_idx=$idx_name, n_groups=$n_cv_name, n_terms=$n_terms_name)))
        else
            push!(stmts, :($bucket_name ~ ranef_correlated_draws_effect(;
                group_idx=$idx_name, n_groups=$n_cv_name, n_terms=$n_terms_name,
                sd_family=$sd_family, sd_rate=$sd_rate, lkj_eta=$lkj_eta)))
        end
    elseif gname in centered_groups
        if isnothing(ranef_effect)
            push!(stmts, :($bucket_name ~ ranef_correlated_draws_centered(;
                group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name)))
        else
            push!(stmts, :($bucket_name ~ ranef_correlated_draws_centered_effect(;
                group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name,
                sd_family=$sd_family, sd_rate=$sd_rate, lkj_eta=$lkj_eta)))
        end
    else
        if isnothing(ranef_effect)
            push!(stmts, :($bucket_name ~ ranef_correlated_draws(;
                group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name)))
        else
            push!(stmts, :($bucket_name ~ ranef_correlated_draws_effect(;
                group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name,
                sd_family=$sd_family, sd_rate=$sd_rate, lkj_eta=$lkj_eta)))
        end
    end
    idx_name
end
function _sb_emit_id_bucket_sampling!(stmts, data, bucket_name, n_terms_name,
                                       g::Tuple{NamedColumn,NamedColumn};
                                       cv_groups=Set{Symbol}(), centered_groups=Set{Symbol}(),
                                       id_sym=nothing, ranef_effect=nothing)
    gname, bname = name(g[1]), name(g[2])
    id_str = isnothing(id_sym) ? "ID" : String(id_sym)
    isnothing(ranef_effect) || error(
        "sbimpl: covariance-prior effects for stratified `|$id_str| " *
        "gr($gname, by=$bname)` buckets are not yet supported")
    gname in cv_groups && error(
        "sbimpl: cv-contagious sizing requested for group `$gname`, but it ",
        "appears in a `(… |$id_str| gr($gname, by=$bname))` stratified ID bucket, ",
        "whose shared draws block goes through the typed-LHS ",
        "`ranef_correlated_by_draws` path. StanBlocks' typed-LHS forward derives ",
        "cv from RHS call-args, not the declared size, so a cv size there yields ",
        "a cv *parameter*, not a generated-quantities re-draw. Use a plain ",
        "`(… |$id_str| $gname)` bucket for the cv group -- `by=` is not yet supported.")
    gname in centered_groups && error(
        "sbimpl: centered parameterization requested for group `$gname`, but it ",
        "appears in a `(… |$id_str| gr($gname, by=$bname))` stratified ID bucket ",
        "(one Cholesky per stratum). The centered variants emit a plate over a ",
        "single shared covariance and have no per-stratum form -- not yet ",
        "supported. Use a plain `(… |$id_str| $gname)` bucket, or leave ",
        "`$gname` non-centered.")
    info = _sb_ensure_group_data!(data, g)
    push!(stmts, :($bucket_name ~ ranef_correlated_by_draws(;
        group_idx=$(info.idx_name), n_groups=$(info.n_name),
        n_terms=$n_terms_name,
        stratum_idx=$(info.s_idx_name), n_strata=$(info.n_strata_name))))
    info.idx_name
end

# Ensure `group_idx` / `n_groups` for a plain-group ranef descriptor are stashed
# in `data`. Idempotent — safe to call from both the ID pre-pass and the per-
# target plain-block emitter. Returns the (idx_name, n_name) pair used in stmts.
function _sb_ensure_group_data!(data, g::NamedColumn)
    g_backing = _as_data_column(parent(g))
    isnothing(g_backing) && error("sbimpl: group `$(name(g))` must be a raw data column")
    gname = name(g)
    idx_name = Symbol(gname, :_idx)
    n_name   = Symbol(:n_, gname)
    n_levels, g_idx = _sb_level_index(parent(g_backing))
    # The generic data prepass sees grouping columns too. Stan consumes only
    # their dense integer code, never the raw labels (which may be strings and
    # therefore are not valid Stan data at all).
    delete!(data, gname)
    data[idx_name] = g_idx
    data[n_name]   = n_levels
    idx_name, n_name
end

# Stratified variant for `gr(g, by=b)`: stashes g_idx / n_groups as well as
# stratum_idx / n_strata under block-local names (suffixed with `__by__`) so
# this block never clashes with a plain `(... | g)` bucket on the same group.
function _sb_ensure_group_data!(data, g::Tuple{NamedColumn,NamedColumn})
    gcol, bcol = g
    g_backing = _as_data_column(parent(gcol))
    b_backing = _as_data_column(parent(bcol))
    isnothing(g_backing) && error("sbimpl: group `$(name(gcol))` must be a raw data column")
    isnothing(b_backing) && error("sbimpl: `by=$(name(bcol))` must be a raw data column")
    gname, bname = name(gcol), name(bcol)
    n_groups, g_idx = _sb_level_index(parent(g_backing))
    n_strata, b_idx = _sb_level_index(parent(b_backing))
    delete!(data, gname)
    delete!(data, bname)
    stratum_idx = _sb_stratum_idx(g_idx, b_idx, gname, bname)
    suffix = Symbol(gname, :__by__, bname)
    idx_name       = Symbol(suffix, :_idx)
    n_name         = Symbol(:n_, suffix)
    s_idx_name     = Symbol(suffix, :_stratum_idx)
    n_strata_name  = Symbol(:n_strata_, suffix)
    data[idx_name]      = g_idx
    data[n_name]        = n_groups
    data[s_idx_name]    = stratum_idx
    data[n_strata_name] = n_strata
    (; idx_name, n_name, s_idx_name, n_strata_name)
end

# Emit the per-sub-formula reference to a pre-emitted ID bucket: slice the
# bucket's draw matrix at this target's column range, apply this target's Z,
# and append the resulting per-row contribution to `summands`.
function _sb_emit_id_ranef_block!(stmts, data, target::Symbol, info, gterms, summands)
    (; bucket_name, cols, idx_name, suffix) = info
    r_name = Symbol(:r_, target, :_, suffix)
    col_exprs = Any[]
    for t in gterms
        _sb_ranef_cols!(col_exprs, data, stmts, t, gterms)
    end
    length(col_exprs) == length(cols) ||
        error("sbimpl: id-bucket `$suffix` for target `$target`: expanded $(length(col_exprs)) columns but reserved $(length(cols)) — internal mismatch")
    if length(cols) == 1
        col_idx = first(cols)
        if length(gterms) == 1 && gterms[1] === 1
            # Intercept fast path: Z is all-ones, skip the elementwise multiply.
            push!(stmts, :($r_name = $bucket_name[$idx_name, $col_idx]))
        else
            push!(stmts, :($r_name = $(col_exprs[1]) .* $bucket_name[$idx_name, $col_idx]))
        end
    else
        Z_name       = Symbol(:Z_, target, :_, suffix)
        col_idx_name = Symbol(:col_idx_, target, :_, suffix)
        data[col_idx_name] = collect(Int, cols)
        push!(stmts, :($Z_name = $(Expr(:call, :hcat, col_exprs...))))
        push!(stmts, :($r_name = rows_dot_product($Z_name, $bucket_name[$idx_name, $col_idx_name])))
    end
    push!(summands, r_name)
end

# Extract pop terms from `1 + a + c1 [+ (...|g)]`. `0` is the standard
# formula-language drop-intercept marker (e.g. `loc ~ 0 + ftime`) and
# contributes no predictor column, so it's filtered out here.
_sb_terms(x) = (acc = Any[]; _sb_collect_terms!(acc, x); acc)
_sb_collect_terms!(acc, x::ExprColumn) = _sb_collect_terms_expr!(acc, getf(x), x)
_sb_collect_terms!(acc, x::Int) = x == 0 ? nothing : push!(acc, x)
_sb_collect_terms!(acc, x) = push!(acc, x)
_sb_collect_terms_expr!(acc, ::typeof(+), x) = foreach(a -> _sb_collect_terms!(acc, a), getargs(x))
# `(expr | group)` is kept as-is; `_sb_linear_predictor!` splits it off into
# the ranef side of the additive linear predictor.
_sb_collect_terms_expr!(acc, ::typeof(|), x) = push!(acc, x)
# `coef * a` -- sampled scalar times a data column. Kept whole here so the
# classifier in `_sb_linear_predictor!` can route it into direct_terms
# (the user supplies their own coefficient via the scalar; no popefs beta).
_sb_collect_terms_expr!(acc, ::typeof(*), x) = push!(acc, x)
# `factor(c, ref=k)`: configurable reference level for a categorical
# column. Re-encode at term-collection time (swap level k <-> level 1)
# and inject a synthetic NamedColumn with the recoded data, so the
# downstream categorical pipeline (treatment-coded dummies relative to
# level 1) reuses unchanged.
_sb_collect_terms_expr!(acc, ::typeof(factor), x::ExprColumn) = begin
    args = getargs(x); kw = getkwargs(x)
    length(args) == 1 || error("sbimpl: `factor(...)` expects 1 positional arg, got $(length(args))")
    inner_raw = only(args)
    inner = _as_named_column(inner_raw)
    isnothing(inner) && error("sbimpl: `factor(...)` expects a NamedColumn, got $(typeof(inner_raw))")
    backing = _as_data_column(parent(inner))
    isnothing(backing) && error("sbimpl: `factor($(name(inner)))` expects a raw data column")
    raw = _as_int_vec(parent(backing))
    isnothing(raw) && error("sbimpl: `factor($(name(inner)))` expects integer-coded categorical data, got $(typeof(parent(backing)))")
    ref_raw = get(kw, :ref, 1)
    ref = _as_integer(ref_raw)
    isnothing(ref) && error("sbimpl: `factor(...; ref=k)` expects an integer level, got $(typeof(ref_raw))")
    1 <= ref <= maximum(raw) || error("sbimpl: `factor($(name(inner)); ref=$ref)` ref out of range (max level $(maximum(raw)))")
    new_name = ref == 1 ? name(inner) : Symbol(name(inner), :__ref_, ref)
    new_backing = ref == 1 ? backing :
        DataColumn(Int[r == ref ? 1 : r == 1 ? ref : r for r in raw])
    push!(acc, NamedColumn(new_name, new_backing))
end
# `(t1 + t2 + ... || g)` zerocorr ranefs: independent variances per term,
# no shared correlation. Expand into N separate `(t_i | g_nocor_i)` ran
# terms with synthetic group names so the ran-term coalescer in
# `_sb_emit_ranefs!` keeps them as separate (degenerate K=1) blocks.
# Mirrors vimpl's `vmeta_sampling_rhs(::ExprColumn{typeof(doublepipe)})`.
_sb_collect_terms_expr!(acc, ::typeof(doublepipe), x) = begin
    args = getargs(x)
    length(args) == 2 || error("sbimpl: `||` zerocorr expects 2 args, got $(length(args))")
    lhs, rhs = args
    rhs_nc = _as_named_column(rhs)
    isnothing(rhs_nc) && error("sbimpl: `||` zerocorr RHS must be a NamedColumn group, got $(typeof(rhs))")
    inner = _zerocorr_inner(lhs)
    for (i, term) in enumerate(inner)
        nocor = NamedColumn(Symbol(name(rhs_nc), :__nocor__, i), parent(rhs_nc))
        push!(acc, ExprColumn(|, term, nocor))
    end
end

_zerocorr_inner(lhs::ExprColumn) = getf(lhs) === (+) ? collect(getargs(lhs)) : Any[lhs]
_zerocorr_inner(lhs) = Any[lhs]
# `a & b` is the interaction operator (parallels StatsModels.jl). `&` has
# higher precedence than `+` in Julia, so `1 + a + b + a&b` naturally parses
# as `+(1, a, b, a&b)` and the normal `+`-flatten path applies. We deliberately
# chose `&` over R's `:` because Julia parses `:` as lower-precedence than
# `+`, which forces a precedence-peel hack that breaks chained interactions.
_sb_collect_terms_expr!(acc, _, x) = push!(acc, x)

# Pop-term column accumulator. Most terms produce a single column via
# `_sb_predictor_col`; interactions (`a:b`) can produce multiple columns
# depending on operand types, so we push into a caller-owned vector.
# `pop_terms` is threaded so the intercept emitter can borrow N from a
# data-backed peer in the same formula (deterministic) rather than
# probing the shared `data` dict in hash order.
_sb_pop_cols!(cols, t, data, stmts, pop_terms=(); obs_n=nothing, group_block_lookup=Dict()) =
    push!(cols, _sb_predictor_col(t, data, stmts, pop_terms; obs_n, group_block_lookup))
_sb_pop_cols!(cols, t::ExprColumn, data, stmts, pop_terms=(); obs_n=nothing, group_block_lookup=Dict()) =
    _sb_pop_cols_expr!(cols, getf(t), t, data, stmts, pop_terms; obs_n, group_block_lookup)
_sb_pop_cols_expr!(cols, ::Any, t, data, stmts, pop_terms=(); obs_n=nothing, group_block_lookup=Dict()) =
    push!(cols, _sb_predictor_col(t, data, stmts, pop_terms; obs_n, group_block_lookup))
_sb_pop_cols_expr!(cols, ::typeof(&), t, data, stmts, _pop_terms=(); kwargs...) =
    _sb_interaction_cols!(cols, t, data, stmts)

# `a & b` interaction expansion. Three supported operand-type combinations:
#   cont x cont -> 1 column (elementwise product)
#   cont x cat  -> K-1 columns (a .* (c == k ? 1 : 0) for k=2..K)
#   cat  x cat  -> (K1-1)*(K2-1) columns (product of level-k1, level-k2 dummies)
# Reference level is always 1 (treatment coding; matches brms / vimpl).
# Columns are materialised at walker time (indicator math only uses data
# columns), stashed in `data` under a `int_<name...>` key, and pushed as
# Symbol refs for hcat downstream.
function _sb_interaction_cols!(cols, t::ExprColumn, data, stmts)
    args = getargs(t)
    length(args) == 2 ||
        error("sbimpl: interaction `&` expects exactly 2 operands, got $(length(args))")
    l = _sb_interaction_operand(args[1], data, stmts)
    r = _sb_interaction_operand(args[2], data, stmts)
    _sb_interaction_expand!(cols, data, l, r)
end

_sb_interaction_operand(t::NamedColumn, _data, _stmts) = begin
    d_raw = parent(t)
    d = _as_data_column(d_raw)
    isnothing(d) && error(
        "sbimpl: interaction operand `$(name(t))` must be a raw data column, got $(typeof(d_raw))"
    )
    v = parent(d)
    _sb_interaction_operand_kind(t, v, _sb_cat_levels(t))
end
# A transformed raw-data term (for example `zscale(math)`) first uses the
# ordinary predictor-term lowering. Pure/data-derived terms leave a concrete
# vector in `data`; parameter-owning terms (`mo`, `me`, `s`, `gp`, `ar`, ...)
# return a Stan variable name instead and are rejected below. This keeps
# transformed interactions on the same fit/reprocess constants as their
# standalone term without ever snapshotting a latent parameter as data.
_sb_interaction_operand(t::ExprColumn, data, stmts) = begin
    col_name = _sb_predictor_col(t, data, stmts)
    haskey(data, col_name) || error(
        "sbimpl: interaction operand `$(getf(t))(...)` is parameter-owning, not a data-materialized transform; ",
        "supported transformed operands include `zscale`, `standardize`, `center`, `protect`, ",
        "and pure expressions in raw data columns"
    )
    v = data[col_name]
    rv = _as_real_vec(v)
    isnothing(rv) && error(
        "sbimpl: transformed interaction operand `$col_name` has unsupported eltype $(eltype(v))"
    )
    (; kind=:cont, name=col_name, vec=collect(Float64, rv))
end
_sb_interaction_operand_kind(t, v, levels) = begin
    n_levels, idx = _sb_level_index(levels)
    n_levels >= 2 || error(
        "sbimpl: categorical interaction operand `$(name(t))` needs >= 2 levels (got $n_levels)"
    )
    (; kind=:cat, name=name(t), n_levels, idx)
end
_sb_interaction_operand_kind(t, v, ::Nothing) = begin
    rv = _as_real_vec(v)
    isnothing(rv) && error("sbimpl: interaction operand `$(name(t))` has unsupported eltype $(eltype(v))")
    (; kind=:cont, name=name(t), vec=collect(Float64, rv))
end
_sb_interaction_operand(t, _data, _stmts) = error(
    "sbimpl: interaction operand must be a raw-data NamedColumn or data-materialized ExprColumn, got $(typeof(t)); ",
    "interactions with parameter-owning terms such as `mo` / `me` / `s` / `gp` / `ar` are not supported"
)

# cont x cont
_sb_interaction_expand!(cols, data, l::NamedTuple{<:Any,<:Tuple}, r::NamedTuple{<:Any,<:Tuple}) = begin
    if l.kind === :cont && r.kind === :cont
        col_name = Symbol(:int_, l.name, :_x_, r.name)
        data[col_name] = l.vec .* r.vec
        _sb_record_preproc!(data, col_name,
            PreprocEntry(:interaction, nothing, (l.name, r.name), false))
        push!(cols, col_name)
    elseif l.kind === :cont && r.kind === :cat
        for lvl in 2:r.n_levels
            col_name = Symbol(:int_, l.name, :_x_, r.name, :_lvl_, lvl)
            data[col_name] = Float64[l.vec[i] * (r.idx[i] == lvl ? 1.0 : 0.0) for i in eachindex(l.vec)]
            push!(cols, col_name)
        end
    elseif l.kind === :cat && r.kind === :cont
        # Symmetric: reuse the :cont × :cat branch with swapped operands so
        # column names consistently put the cont term first.
        _sb_interaction_expand!(cols, data, r, l)
    elseif l.kind === :cat && r.kind === :cat
        n = length(l.idx)
        length(r.idx) == n || error(
            "sbimpl: interaction `$(l.name):$(r.name)`: operand lengths mismatch ($n vs $(length(r.idx)))"
        )
        for lvl1 in 2:l.n_levels, lvl2 in 2:r.n_levels
            col_name = Symbol(:int_, l.name, :_lvl_, lvl1, :_x_, r.name, :_lvl_, lvl2)
            data[col_name] = Float64[(l.idx[i] == lvl1 && r.idx[i] == lvl2) ? 1.0 : 0.0 for i in 1:n]
            push!(cols, col_name)
        end
    else
        error("sbimpl: unsupported interaction operand combination (`$(l.kind)` x `$(r.kind)`)")
    end
end

# Predictor column emitter. `stmts` is threaded in so terms that need their own
# `~` statement (e.g. `mo(c)`) can push before returning their column symbol.
# Integer `1` -> intercept, NamedColumn -> reference by name, ExprColumn(mo, c)
# -> submodel-sampled contrast column.
_sb_predictor_col(t::Int, data, _stmts, pop_terms=(); obs_n::Union{Symbol,Nothing}=nothing, kwargs...) = begin
    t == 1 || error("sbimpl: integer term must be `1` for intercept, got `$t`")
    # Three-tier length probe, in priority order:
    #   1. A data-backed peer in the same formula's terms (`_sb_n_obs_probe`).
    #      Deterministic for any mixed-intercept formula like `y ~ 1 + x`.
    #   2. The observation column threaded from the likelihood walker
    #      (`obs_n`). Covers purely-intercept formulas like `loc ~ 1` whose
    #      length matches the observed `~` target consuming `loc`.
    #   3. Hash-order fallback (`_sb_any_data_symbol`). Last resort; lossy
    #      for composite models with multi-length data and reachable only
    #      when neither (1) nor (2) yields a name (e.g. a `~ 1` formula
    #      whose target isn't referenced by any observed likelihood).
    probe = _sb_n_obs_probe(pop_terms)
    isnothing(probe) && !isnothing(obs_n) && (probe = obs_n)
    isnothing(probe) && (probe = _sb_any_data_symbol(data))
    :(rep_vector(1., num_elements($probe)))
end
_sb_predictor_col(t::NamedColumn, data, _stmts, _pop_terms=(); kwargs...) = _predictor_col_for(t, parent(t), data)

# If the named column was already bound earlier in the walker (e.g.
# `ftime ~ gamma_time(...)` emitted a `ftime ~ _sb_gamma_time(...)` stmt),
# its parent is the sampling ExprColumn rather than a raw data column --
# just reference the Stan variable by name.
_predictor_col_for(t, ::ExprColumn, _) = name(t)
function _predictor_col_for(t, d::DataColumn, data)
    v = parent(d)
    rv = _as_real_vec(v)
    isnothing(rv) && error("sbimpl: non-numeric predictor `$(name(t))` not supported yet (wrap in `categorical(…)` once we add it)")
    data[name(t)] = collect(Float64, rv)
    name(t)
end
_predictor_col_for(t, d, _) = error("sbimpl: expected data-backed NamedColumn for `$(name(t))`, got $(typeof(d))")
_sb_predictor_col(t::ExprColumn, data, stmts, _pop_terms=(); kwargs...) = _sb_predictor_term!(stmts, data, getf(t), t; kwargs...)
_sb_predictor_col(t, _data, _stmts, _pop_terms=(); kwargs...) = error("sbimpl: unsupported predictor term $(typeof(t)): $t")

# Monotonic-effect predictor: emit `mo_<c> ~ _sb_mo(; x=<c>_idx)` and return
# `mo_<c>` as the column. Scope: single NamedColumn inner arg backed by raw
# data. Other wrapped terms dispatch to their own methods below.
_sb_predictor_term!(stmts, data, ::typeof(mo), t; kwargs...) = begin
    inner_name, raw = _sb_inner_data(:mo, only(getargs(t)))
    n_levels, idx = _sb_level_index(raw)
    n_levels >= 2 || error("sbimpl: `mo($inner_name)` needs >= 2 levels (got $n_levels)")
    idx_name = Symbol(inner_name, :_idx)
    col_name = Symbol(:mo_, inner_name)
    data[idx_name] = idx
    # Frozen level set drives the monotonic-effect simplex dimension; re-coding
    # a new df against it is dimension-coupled (unseen level / changed count).
    _sb_record_preproc!(data, idx_name, PreprocEntry(:mo, _sb_fit_levels(raw), inner_name, true))
    push!(stmts, :($col_name ~ _sb_mo(; x=$idx_name)))
    col_name
end
# Measurement-error predictor `me(x_obs, sd_x)`: emit a submodel that allocates
# a length-N latent `me_<x>` with prior std_normal and an observation
# likelihood `x_obs ~ normal(me_<x>, sd_x)`. Returns `me_<x>` as the predictor
# column so popefs supplies a free beta. `sd_x` must be a positive constant;
# per-row error sizes would require a vector kwarg and a tweaked submodel.
_sb_predictor_term!(stmts, data, ::typeof(me), t; kwargs...) = begin
    args = getargs(t)
    length(args) == 2 || error("sbimpl: `me(x, sd)` expects 2 args, got $(length(args))")
    inner, sd_arg = args
    xname, raw = _sb_inner_data(:me, inner)
    v = _sb_real_vec(:me, xname, raw)
    sd_real = _as_real(sd_arg)
    isnothing(sd_real) && error("sbimpl: `me(x, sd)` expects a numeric constant `sd`, got $(typeof(sd_arg))")
    sd_real > 0 || error("sbimpl: `me(x, sd)` expects sd > 0 (got $sd_real)")
    sd_arg = sd_real
    data[xname] = collect(Float64, v)
    sd_name = Symbol(:sd_, xname)
    data[sd_name] = Float64(sd_arg)
    col_name = Symbol(:me_, xname)
    push!(stmts, :($col_name ~ _sb_me(; x_obs=$xname, sd_x=$sd_name)))
    col_name
end
# Penalized thin-plate predictor `s(x)`. Fits a frozen rank-10 TPS eigenbasis
# from the raw training column, then stashes its two-column null-space matrix
# and eight-column penalty-whitened range matrix as Stan data. `_sb_s` owns the
# flat null-space coefficients, penalized coefficients, and smoothing SD; the
# returned contribution is a direct summand (no extra `popefs` beta). Only
# the default basis is supported -- `bs` and `k=`/`knots=` are follow-ons.
_sb_predictor_term!(stmts, data, ::typeof(s), t; kwargs...) = begin
    args = getargs(t)
    length(args) == 1 || error("sbimpl: `s(x)` expects 1 positional arg, got $(length(args))")
    isempty(getkwargs(t)) || error("sbimpl: `s(x)` does not support keyword arguments yet")
    xname, raw = _sb_inner_data(:s, only(args))
    v = _sb_real_vec(:s, xname, raw)
    fit = _sb_fit_spline(v)
    Xnull, Zpen = _sb_apply_spline(fit, v)
    Xnull_name = Symbol(:Xnull_, xname)
    Zpen_name = Symbol(:Zpen_, xname)
    data[Xnull_name] = Xnull
    data[Zpen_name] = Zpen
    # Frozen training centers/eigenbasis → fixed dimension. Reprocess evaluates
    # both matrices at new x values against these constants.
    _sb_record_preproc!(data, Xnull_name,
        PreprocEntry(:spline, (; fit, zpen_key=Zpen_name), xname, false))
    col_name = Symbol(:s_, xname)
    push!(stmts, :($col_name ~ _sb_s(; Xnull=$Xnull_name, Zpen=$Zpen_name)))
    col_name
end

# Two-margin tensor-product cubic-regression spline. The Julia-side fit records
# the marginal knot/penalty decomposition and training centering constants;
# `_sb_t2` owns the three unpenalized NN coefficients plus independent RR/RN/NR
# smoothing scales and standardized range coefficients. It is therefore a
# direct summand, never multiplied by an additional `popefs` beta.
_sb_predictor_term!(stmts, data, ::typeof(t2), t;
                    target::Union{Symbol,Nothing}=nothing, kwargs...) = begin
    args = getargs(t)
    length(args) == 2 || error(
        "sbimpl: `t2(x, z)` expects exactly 2 positional margins, got $(length(args))")
    kw = getkwargs(t)
    _check_term_kwargs(t2, kw)
    k, _, _ = _sb_t2_options(kw)
    names, axes = _sb_gp_axes(:t2, args)
    fit = _sb_fit_t2(axes[1], axes[2]; k)
    Xfixed, Zrr, Zrn, Znr = _sb_apply_t2(fit, axes[1], axes[2])
    axes_suffix = join(string.(names), "_")
    suffix = isnothing(target) ? axes_suffix : string(target, "_", axes_suffix)
    Xfixed_name = Symbol(:Xfixed_t2_, suffix)
    Zrr_name = Symbol(:Zrr_t2_, suffix)
    Zrn_name = Symbol(:Zrn_t2_, suffix)
    Znr_name = Symbol(:Znr_t2_, suffix)
    data[Xfixed_name] = Xfixed
    data[Zrr_name] = Zrr
    data[Zrn_name] = Zrn
    data[Znr_name] = Znr
    _sb_record_preproc!(data, Xfixed_name, PreprocEntry(:tensor_spline,
        (; fit, zrr_key=Zrr_name, zrn_key=Zrn_name, znr_key=Znr_name),
        names, false))
    col_name = Symbol(:t2_, suffix)
    push!(stmts, :($col_name ~ _sb_t2(;
        Xfixed=$Xfixed_name, Zrr=$Zrr_name, Zrn=$Zrn_name, Znr=$Znr_name)))
    col_name
end

# `gp(x...)` is the exact GP term. It records an N x d predictor matrix and
# delegates covariance construction + non-centred sampling to `_sb_gp` (one
# shared length scale) or `_sb_gp_aniso` (one per axis).
_sb_predictor_term!(stmts, data, ::typeof(gp), t; group_block_lookup=Dict(), kwargs...) = begin
    args = getargs(t); kw = getkwargs(t)
    _check_term_kwargs(gp, kw)
    names, axes = _sb_gp_axes(:gp, args)
    suffix = join(string.(names), "_")
    X_name = Symbol(:X_gp_, suffix)
    col_name = Symbol(:gp_, suffix)
    data[X_name] = _sb_gp_matrix(axes)
    _sb_record_preproc!(data, X_name, PreprocEntry(:gp, nothing, names, false))
    submodel = _sb_gp_iso(kw, :gp) ? :_sb_gp : :_sb_gp_aniso
    jitter = Float64(get(kw, :jitter, 1e-9))
    push!(stmts, :($col_name ~ $submodel(; X=$X_name, jitter=$jitter)))
    col_name
end

# `hsgp(x...; k, c, by, iso)` is the Hilbert-space approximation. Scalar `k`
# and `c` broadcast across axes; tuples/vectors specify one value per axis.
# The tensor basis has `prod(k)` columns. With `by=`, only those basis weights
# vary per group; length-scale and marginal-SD hyperparameters stay shared.
_sb_predictor_term!(stmts, data, ::typeof(hsgp), t; group_block_lookup=Dict(), kwargs...) = begin
    args = getargs(t); kw = getkwargs(t)
    _check_term_kwargs(hsgp, kw)
    names, axes = _sb_gp_axes(:hsgp, args)
    K, c = _sb_hsgp_options(kw, length(axes))
    fits = _sb_fit_hsgp(axes, K, c)
    PHI, omega2 = _sb_apply_hsgp(fits, axes, K)
    suffix = join(string.(names), "_")
    iso = _sb_gp_iso(kw, :hsgp)

    if haskey(kw, :by)
        block_info = _sb_find_group_block(hsgp, t, group_block_lookup)
        isnothing(block_info) && error(
            "sbimpl: `hsgp($suffix, by=...)` found no allocated per-group weight ",
            "block — prepass 2.5 should have allocated it")
        info = block_info
        gname = name(_sb_resolve_group_col((; kwarg=:by), t, data))
        PHI_name = Symbol(:PHI_hsgp_, suffix, :_by_, gname)
        omega2_name = Symbol(:omega2_hsgp_, suffix, :_by_, gname)
        data[PHI_name] = PHI
        data[omega2_name] = omega2
        _sb_record_preproc!(data, PHI_name, PreprocEntry(:hsgp,
            (; fits, K, c, omega2_key=omega2_name), names, false))
        col_name = Symbol(:hsgp_, suffix, :_by_, gname)
        submodel = iso ? :_sb_hsgp_by : :_sb_hsgp_by_aniso
        push!(stmts, :($col_name ~ $submodel(; PHI=$PHI_name, omega2=$omega2_name,
            beta=$(info.block_name), group_idx=$(info.idx_name))))
        return col_name
    end

    PHI_name = Symbol(:PHI_hsgp_, suffix)
    omega2_name = Symbol(:omega2_hsgp_, suffix)
    data[PHI_name] = PHI
    data[omega2_name] = omega2
    _sb_record_preproc!(data, PHI_name, PreprocEntry(:hsgp,
        (; fits, K, c, omega2_key=omega2_name), names, false))
    col_name = Symbol(:hsgp_, suffix)
    submodel = iso ? :_sb_hsgp : :_sb_hsgp_aniso
    push!(stmts, :($col_name ~ $submodel(; PHI=$PHI_name, omega2=$omega2_name)))
    col_name
end

function _sb_term_group_block(::typeof(hsgp), call)
    kw = getkwargs(call)
    haskey(kw, :by) || return nothing
    _check_term_kwargs(hsgp, kw)
    args = getargs(call)
    isempty(args) && error("sbimpl: `hsgp(x...; by=...)` expects at least one positional axis")
    names = Tuple(name(_sb_named_inner(:hsgp, a)) for a in args)
    K, _ = _sb_hsgp_options(kw, length(args))
    fname = Symbol(:hsgpw_, join(string.(names), "_"))
    (; fields=[(; name=fname, n_per_group=prod(K), group=(; kwarg=:by), prior=:iid_normal)])
end

# `ar(time; p=1)` AR(1) residual submodel. Routes to `_sb_ar1`, which owns the
# phi / epsilon parameters and returns the per-row u[t] as a single length-N
# column. popefs multiplies by an overall beta -- harmless, but a direct-
# summand variant would skip it.
_sb_predictor_term!(stmts, data, ::typeof(ar), t; kwargs...) = begin
    args = getargs(t)
    kw = getkwargs(t)
    p = get(kw, :p, 1)
    p == 1 || error("sbimpl: `ar(time; p=$p)` only supports p=1 so far")
    length(args) == 1 || error("sbimpl: `ar(time; p=1)` expects 1 positional arg, got $(length(args))")
    xname, raw = _sb_inner_data(:ar, only(args))
    # Ensure the time column lands in `data`. The prepass already handles this
    # for named data columns, but be defensive -- the submodel uses it as a
    # length probe via `num_elements(time)`.
    data[xname] = collect(Float64, raw)
    col_name = Symbol(:ar_, xname)
    push!(stmts, :($col_name ~ _sb_ar1(; time=$xname)))
    col_name
end
# Vector-wise wrapper predictors: `zscale`, `standardize`, and `center`
# need the whole inner column to compute (mean / sd are not element-wise),
# so the generic broadcast-based fallback in `_sb_materialize_vec` won't
# do. Materialize the inner separately, apply the transform, stash. (The
# brms-style `protect(...)` no-op is handled by the generic fallback once
# `protect(x::Real) = x` is defined in macro.jl.)
for (fn, kind, fitf, applyf) in (
        (:zscale,      :zscale,      :_sb_fit_zscale, :_sb_apply_zscale),
        (:standardize, :standardize, :_sb_fit_zscale, :_sb_apply_zscale),
        (:center,      :center,      :_sb_fit_center, :_sb_apply_center),
    )
    # fit → apply keeps the construct-time column byte-identical to the old
    # one-pass while exposing the fitted constant `c` for the preproc record
    # (so `reprocess` can re-apply it, frozen, to a new df). `raw_ref` is the
    # inner column-node tree, re-materialised on the new df at reprocess time.
    @eval function _sb_predictor_term!(stmts, data, ::typeof($fn), t; kwargs...)
        inner = only(getargs(t))
        v = collect(Float64, _sb_materialize_vec(inner))
        c = $fitf(v)
        v_t = $applyf(c, v)
        cn = _sb_wrapper_col_name(Symbol($(QuoteNode(fn))), inner)
        data[cn] = v_t
        _sb_record_preproc!(data, cn, PreprocEntry($(QuoteNode(kind)), c, inner, false))
        cn
    end
end

# Fallback: a "plain" expression like `log(exposure)` or `a^2` reaching this
# point is treated as an implicit `protect(...)` -- materialize the whole
# subtree to a Stan data vector and let popefs supply the beta. Errors out
# if any leaf isn't a raw data column (e.g. references a sampled parameter
# directly), preserving the old "unsupported" diagnostic.
function _sb_predictor_term!(stmts, data, f::Function, t; kwargs...)
    try
        v = collect(Float64, _sb_materialize_vec(t))
        cn = _sb_wrapper_col_name(Symbol(f), t)
        data[cn] = v
        # Element-wise pure transform: no fitted constant, just re-materialise
        # the same expr tree against the new df (freeze-agnostic).
        _sb_record_preproc!(data, cn, PreprocEntry(:protect, nothing, t, false))
        return cn
    catch err
        _ee = _as_error_exception(err); isnothing(_ee) && rethrow()
        error("sbimpl: unsupported predictor-term function `$f` (supported: `mo`, `mo1`, `me`, `s`, `ar`, `protect`, `zscale`, `center`, `standardize`, or any expression in raw data columns) -- materialization failed: $(_ee.msg)")
    end
end

# Recursively materialize an ExprColumn / NamedColumn tree into a plain
# vector, walking only data-backed leaves. Used by the wrapper predictors
# (protect / zscale / etc) to compute their column at transpile time.
_sb_materialize_vec(x::Number) = x
_sb_materialize_vec(x::NamedColumn) = _materialize_named(x, parent(x))
_materialize_named(_, d::DataColumn) = parent(d)
_materialize_named(x, _) = error(
    "sbimpl: cannot materialize NamedColumn `$(name(x))` -- only raw data columns supported inside `protect` / `zscale` / `center` / `standardize`")
_sb_materialize_vec(x::ExprColumn) = broadcast(getf(x), map(_sb_materialize_vec, getargs(x))...)
_sb_materialize_vec(x) = error("sbimpl: cannot materialize $(typeof(x)) inside wrapper predictor")

# Fit/apply split for the vector-wise standardisers. `_sb_fit_*` computes the
# data-derived constant; `_sb_apply_*` applies a (possibly frozen) constant to
# a vector. The fused `_sb_zscale`/`_sb_center` keep construct-time behaviour
# byte-identical (fit∘apply == the old one-pass), and `reprocess` reuses the
# apply half with a frozen constant for prediction-replay (decision nr3v8n A).
_sb_fit_zscale(v::AbstractVector{<:Real}) = let mu = sum(v) / length(v)
    sd = sqrt(sum((x - mu)^2 for x in v) / length(v))
    sd > 0 || error("sbimpl: zscale: zero variance")
    (mu, sd)
end
_sb_apply_zscale(c::Tuple, v::AbstractVector{<:Real}) = (v .- c[1]) ./ c[2]
_sb_zscale(v::AbstractVector{<:Real}) = _sb_apply_zscale(_sb_fit_zscale(v), v)

_sb_fit_center(v::AbstractVector{<:Real}) = sum(v) / length(v)
_sb_apply_center(mu::Real, v::AbstractVector{<:Real}) = v .- mu
_sb_center(v::AbstractVector{<:Real}) = _sb_apply_center(_sb_fit_center(v), v)

# Stable, human-readable column name for a wrapped predictor. When the
# inner is a single NamedColumn we tag with its name; otherwise we hash
# the expr structure so multiple `protect(...)` summands don't collide.
_sb_wrapper_col_name(prefix::Symbol, inner::NamedColumn) =
    Symbol(prefix, :_, name(inner))
_sb_wrapper_col_name(prefix::Symbol, inner) =
    Symbol(prefix, :_expr_, string(hash(inner); base=16)[1:8])

_n_obs_name(t::NamedColumn) = _n_obs_named_data(t, parent(t))
_n_obs_name(_) = nothing
_n_obs_named_data(t, ::DataColumn) = name(t)
_n_obs_named_data(args...) = nothing

_sb_n_obs_probe(terms) = begin
    for t in terms
        n = _n_obs_name(t)
        isnothing(n) || return n
    end
    nothing
end

# Prepass: build a `target -> observation` map. For each `~` op whose LHS is
# observed (a data-backed NamedColumn directly, or wrapped in a link like
# `log(y)` over a data-backed NamedColumn), walk the RHS recursively and
# record `target_obs[ref_name] = obs_name` for every NamedColumn reference.
# The observation column itself maps to itself. First write wins -- if a
# target is referenced by multiple likelihoods, the first encountered op
# (BRMI iteration order) supplies the source N. The intercept emitter
# consults this map for purely-intercept formulas (`y ~ 1`, `loc ~ 1`) where
# no in-formula data-backed peer exists; the threaded `obs_n` becomes the
# length probe before falling through to `_sb_any_data_symbol`.
function _sb_collect_target_obs(brmi::BRMI)
    target_obs = Dict{Symbol,Symbol}()
    for (_, op_nc) in pairs(brmi.operations)
        op = _as_expr_column(parent(op_nc))
        isnothing(op) && continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        obs_name = _sb_observation_name(lhs)
        isnothing(obs_name) && continue
        get!(target_obs, obs_name, obs_name)
        _sb_collect_rhs_refs!(target_obs, rhs, obs_name)
    end
    target_obs
end

# Resolve a `~` op's LHS to the observed data-column name (the source of N),
# or `nothing` if the LHS isn't an observation. Handles direct
# data-backed NamedColumn and one-arg link wrappers (`log(y)`, `mi(y)`, etc.).
_sb_observation_name(_) = nothing
_sb_observation_name(lhs::NamedColumn) = _n_obs_named_data(lhs, parent(lhs))
_sb_observation_name(lhs::ExprColumn) = begin
    args = getargs(lhs)
    length(args) == 1 ? _sb_observation_name(args[1]) : nothing
end

# Walk a RHS expression tree, recording every NamedColumn reference's name
# into `target_obs` keyed back to `obs_name`. `get!` ensures first-write
# wins so the BRMI's natural iteration order picks the source.
_sb_collect_rhs_refs!(_target_obs, _x, _obs_name) = nothing
_sb_is_nothing_column(x::NamedColumn) =
    name(x) === :nothing && parent(x) isa MissingColumn
_sb_collect_rhs_refs!(target_obs, x::NamedColumn, obs_name) = begin
    !_sb_is_nothing_column(x) && get!(target_obs, name(x), obs_name)
    nothing
end
_sb_collect_rhs_refs!(target_obs, x::ExprColumn, obs_name) = begin
    foreach(a -> _sb_collect_rhs_refs!(target_obs, a, obs_name), getargs(x))
    foreach(v -> _sb_collect_rhs_refs!(target_obs, v, obs_name), values(getkwargs(x)))
end
_sb_any_data_symbol(data) = begin
    isempty(data) && error("sbimpl: can't emit `rep_vector(1., n)` — no data column seen yet. Make sure an observed `~` comes before the intercept-only predictor, or add a concrete covariate.")
    # Prefer a flat length-N vector (numeric / integer) so `num_elements(...)` in
    # Stan resolves to an int. Skip ragged `Vector{<:AbstractVector}` layouts
    # (bruno-ext's `dose_times`) which StanBlocks serializes as a
    # `tuple(vector, array[] int)` that Stan's `num_elements` rejects.
    for (k, v) in data
        k === _SB_PREPROC_KEY && continue
        hit = _flat_vec_key(k, v)
        isnothing(hit) || return hit
    end
    first(k for k in keys(data) if k !== _SB_PREPROC_KEY)
end

# Return `k` if `v` is a flat (non-ragged) vector, else `nothing` — replaces
# the old `_is_flat_vec` Bool predicate so the caller composes via the
# returned value rather than a branch on the test.
_flat_vec_key(k, ::AbstractVector{<:AbstractVector}) = nothing
_flat_vec_key(k, ::AbstractVector) = k
_flat_vec_key(_k, _v) = nothing


# ---- likelihood emitters: `y ~ Normal(loc, sigma)` etc. ----------------------

function _sb_likelihood!(stmts, target::Symbol, rhs::ExprColumn, data)
    f = getf(rhs)
    _sb_lik_family!(stmts, target, f, getargs(rhs), getkwargs(rhs), data)
end

_sb_weight_kind(f) =
    f === aweights || f === AnalyticWeights ? :analytic :
    f === fweights || f === FrequencyWeights ? :frequency :
    f === weights || f === Weights ? :power :
    f === pweights || f === ProbabilityWeights ? :probability :
    f === uweights || f === UnitWeights ? :unit : nothing

function _sb_weight_source(target::Symbol, weight::ExprColumn)
    isempty(getkwargs(weight)) || error(
        "sbimpl: `weighted(..., $(getf(weight))(...))` does not accept weight " *
        "constructor keywords")
    args = getargs(weight)
    length(args) == 1 || error(
        "sbimpl: `weighted` expects a one-column StatsBase weight constructor " *
        "such as `aweights(k)`, `fweights(n)`, or `weights(w)`; got " *
        "$(length(args)) arguments for response `$target`")
    source = only(args)
    source isa NamedColumn && parent(source) isa DataColumn || error(
        "sbimpl: weights for response `$target` must be built from one raw " *
        "dataframe column, got $(typeof(source))")
    name(source), parent(parent(source))
end

function _sb_prepare_weight_values(kind::Symbol, raw, nobs::Int, target::Symbol,
                                   source::Symbol)
    raw isa AbstractVector{<:Real} || error(
        "sbimpl: weight column `$source` for response `$target` must be a real " *
        "vector, got $(typeof(raw))")
    length(raw) == nobs || error(
        "sbimpl: weight column `$source` has length $(length(raw)) but response " *
        "`$target` has length $nobs")
    values = collect(Float64, raw)
    all(isfinite, values) || error(
        "sbimpl: weight column `$source` for response `$target` contains " *
        "non-finite values")
    if kind === :analytic
        all(>(0), values) || error(
            "sbimpl: analytic/precision weights for response `$target` must be " *
            "strictly positive")
    elseif kind === :frequency
        all(x -> x >= 0 && isinteger(x), values) || error(
            "sbimpl: frequency weights for response `$target` must be " *
            "nonnegative integer-valued counts")
    elseif kind === :power
        all(>=(0), values) || error(
            "sbimpl: power-likelihood weights for response `$target` must be " *
            "nonnegative")
    else
        error("sbimpl: internal unsupported observation-weight kind `$kind`")
    end
    values
end

_sb_weight_data_key(target::Symbol) = Symbol(:brm_weight_, target)

function _sb_weight_data!(data, target::Symbol, kind::Symbol,
                          weight::ExprColumn)
    source, raw = _sb_weight_source(target, weight)
    response = get(data, target, nothing)
    response isa AbstractVector || error(
        "sbimpl: weighted response `$target` must be an observed vector, got " *
        "$(typeof(response))")
    key = _sb_weight_data_key(target)
    haskey(data, key) && error(
        "sbimpl: reserved derived weight key `$key` collides with a model/data " *
        "column; rename that column")
    data[key] = _sb_prepare_weight_values(kind, raw, length(response), target, source)
    _sb_record_preproc!(data, key, PreprocEntry(
        :observation_weight, (; kind, response=target), source, false))
    key
end

function _sb_weighted_distribution(rhs, target::Symbol)
    rhs isa ExprColumn || error(
        "sbimpl: first argument of `weighted` for response `$target` must be a " *
        "distribution call, got $(typeof(rhs))")
    isempty(getkwargs(rhs)) || error(
        "sbimpl: weighted distribution `$target` does not currently support " *
        "distribution constructor keywords")
    rhs
end

function _sb_analytic_weighted_likelihood!(stmts, target::Symbol,
                                           distribution::ExprColumn,
                                           weight_key::Symbol, data)
    family = getf(distribution)
    family isa Type && family <: Normal || error(
        "sbimpl: `AnalyticWeights` currently support only `Normal` observations; " *
        "response `$target` uses `$family`")
    args = map(a -> _sb_scalar_expr(a, data), getargs(distribution))
    normal_args = _sb_stan_dist_args(family, args)
    length(normal_args) == 2 || error(
        "sbimpl: internal Normal lowering for analytic weights expected two " *
        "arguments, got $(length(normal_args))")
    location, scale = normal_args
    weighted_scale = Expr(:call, Symbol("./"), scale,
                          Expr(:call, :sqrt, weight_key))
    _sb_lik_stan_exprs!(stmts, target, :normal, (location, weighted_scale))
end

function _sb_objective_weighted_likelihood!(stmts, target::Symbol,
                                            distribution::ExprColumn,
                                            weight_key::Symbol, data)
    family = getf(distribution)
    family isa Type && family <: Distribution || error(
        "sbimpl: frequency/power weights currently require a Distributions.jl " *
        "family call for response `$target`, got `$family`")
    stan_name = _sb_stan_dist_name(family)
    isnothing(stan_name) && error(
        "sbimpl: weighted likelihood family `$family` is not supported yet")
    args = map(a -> _sb_scalar_expr(a, data), getargs(distribution))
    stan_args = _sb_stan_dist_args(family, args)
    weighted_rhs = Expr(:call, :weighted, stan_name, weight_key, stan_args...)
    push!(stmts, Expr(:call, :~, target, weighted_rhs))
end

function _sb_likelihood!(stmts, target::Symbol,
                         rhs::ExprColumn{typeof(weighted)}, data)
    isempty(getkwargs(rhs)) || error(
        "sbimpl: `weighted(distribution, weights)` accepts no keywords")
    distribution_raw, weight_raw = getargs(rhs, 2)
    distribution = _sb_weighted_distribution(distribution_raw, target)
    weight_raw isa ExprColumn || error(
        "sbimpl: second argument of `weighted` must be a StatsBase weight " *
        "constructor, got $(typeof(weight_raw))")
    kind = _sb_weight_kind(getf(weight_raw))
    isnothing(kind) && error(
        "sbimpl: unsupported weight constructor `$(getf(weight_raw))`; use " *
        "`aweights`, `fweights`, or `weights`")
    kind === :probability && error(
        "sbimpl: `ProbabilityWeights` sampling-weight semantics are not " *
        "implemented; they are not interchangeable with likelihood weights")
    kind === :unit && error(
        "sbimpl: omit `weighted(...)` for unit weights; write the base " *
        "distribution directly")
    weight_key = _sb_weight_data!(data, target, kind, weight_raw)
    if kind === :analytic
        _sb_analytic_weighted_likelihood!(
            stmts, target, distribution, weight_key, data)
    else
        _sb_objective_weighted_likelihood!(
            stmts, target, distribution, weight_key, data)
    end
end
_sb_likelihood!(stmts, target, rhs, _) =
    error("sbimpl: likelihood RHS for `$target` must be an ExprColumn (got $(typeof(rhs)))")

# Existing ordinary families remain positional and retain their prior behavior.
# Wrapper families override this six-argument seam below so only the new public
# compositions interpret formula keywords.
_sb_lik_family!(stmts, target, fam, args, ::NamedTuple, data) =
    _sb_lik_family!(stmts, target, fam, args, data)

# One dispatch per likelihood family. Each method states the Stan name and
# implicitly the arity (by destructuring `args`). Julia constructor arguments
# are normalized centrally by `_sb_stan_dist_args` before they reach the Stan
# call, so model density, predictive RNG and pointwise log-likelihood all see
# the same exact parameterization.
_sb_lik_stan_exprs!(stmts, target, name::Symbol, arg_exprs) =
    push!(stmts, Expr(:call, :~, target, Expr(:call, name, arg_exprs...)))

_sb_lik_stan!(stmts, target, name::Symbol, args, data) =
    _sb_lik_stan_exprs!(
        stmts, target, name, map(a -> _sb_scalar_expr(a, data), args))

# `y ~ OrderedLogistic(eta)`: cumulative-link ordinal likelihood. Allocates
# `ordered[K-1]` cutpoints (K = max(y)) with a std_normal prior via typed-LHS,
# then emits `y ~ ordered_logistic(eta, cutpoints)` — dispatches against
# Stan's built-in `ordered_logistic_lpmf` (surfaced into SB's @builtin_module).
function _sb_lik_family!(stmts, target, ::Type{<:OrderedLogistic}, args::Tuple{Any}, data)
    y_raw = data[target]
    y = _as_int_vec(y_raw)
    isnothing(y) && error(
        "sbimpl: `OrderedLogistic` expects integer outcome data for `$target`, got $(typeof(y_raw))"
    )
    n_levels = maximum(y)
    n_levels >= 2 || error("sbimpl: `OrderedLogistic($target)` needs >= 2 levels (got $n_levels)")
    n_cut = n_levels - 1
    cut_name = Symbol(target, :_cutpoints)
    push!(stmts, :($cut_name::ordered[$n_cut] ~ std_normal()))
    eta_expr = _sb_scalar_expr(args[1], data)
    push!(stmts, :($target ~ ordered_logistic($eta_expr, $cut_name)))
end

_sb_ordinal_structure_code(::Cumulative) = 1
_sb_ordinal_structure_code(::StoppingRatio) = 2
_sb_ordinal_structure_code(::Type{Cumulative}) = 1
_sb_ordinal_structure_code(::Type{StoppingRatio}) = 2
function _sb_ordinal_structure_code(x::ExprColumn)
    isempty(getargs(x)) && isempty(getkwargs(x)) || error(
        "sbimpl: ordinal structure tags take no arguments; use `Cumulative()` " *
        "or `StoppingRatio()`")
    f = getf(x)
    f === Cumulative && return 1
    f === StoppingRatio && return 2
    error("sbimpl: unsupported ordinal structure `$f`; use `Cumulative()` or " *
          "`StoppingRatio()`")
end
_sb_ordinal_structure_code(x) = error(
    "sbimpl: ordinal structure must be `Cumulative()` or `StoppingRatio()`, " *
    "got $(typeof(x))")

_sb_ordinal_link_code(::LogitLink) = 1
_sb_ordinal_link_code(::ProbitLink) = 2
_sb_ordinal_link_code(::CloglogLink) = 3
_sb_ordinal_link_code(::Type{LogitLink}) = 1
_sb_ordinal_link_code(::Type{ProbitLink}) = 2
_sb_ordinal_link_code(::Type{CloglogLink}) = 3
function _sb_ordinal_link_code(x::ExprColumn)
    isempty(getargs(x)) && isempty(getkwargs(x)) || error(
        "sbimpl: ordinal link tags take no arguments; use `LogitLink()`, " *
        "`ProbitLink()`, or `CloglogLink()`")
    f = getf(x)
    f === LogitLink && return 1
    f === ProbitLink && return 2
    f === CloglogLink && return 3
    error("sbimpl: unsupported ordinal link `$f`; use `LogitLink()`, " *
          "`ProbitLink()`, or `CloglogLink()`")
end
_sb_ordinal_link_code(x) = error(
    "sbimpl: ordinal link must be `LogitLink()`, `ProbitLink()`, or " *
    "`CloglogLink()`, got $(typeof(x))")

_sb_ordinal_has_fixed_intercept(x::Real) = !iszero(x)
_sb_ordinal_has_fixed_intercept(x::NamedColumn) =
    _sb_ordinal_has_fixed_intercept_parent(parent(x))
_sb_ordinal_has_fixed_intercept(_) = false
_sb_ordinal_has_fixed_intercept_parent(p::ExprColumn) = begin
    getf(p) === (~) || return false
    _, rhs = getargs(p, 2)
    any(t -> t isa Integer && t == 1, _sb_terms(rhs))
end
_sb_ordinal_has_fixed_intercept_parent(_) = false

function _sb_ordinal_threshold_predictors!(data, target, raw, n_obs)
    raw isa Tuple || error(
        "sbimpl: `Ordinal(...; per_threshold=...)` expects a tuple of raw " *
        "numeric columns, for example `per_threshold=(treat,)`")
    names = Symbol[]
    for term in raw
        term isa NamedColumn && parent(term) isa DataColumn || error(
            "sbimpl: `Ordinal(...; per_threshold=...)` currently accepts only " *
            "raw numeric data columns; got $(typeof(term))")
        key = name(term)
        values = parent(parent(term))
        values isa AbstractVector{<:Real} || error(
            "sbimpl: ordinal threshold predictor `$key` must be numeric, got " *
            "$(typeof(values))")
        length(values) == n_obs || error(
            "sbimpl: ordinal threshold predictor `$key` has $(length(values)) " *
            "rows; outcome `$target` has $n_obs")
        all(isfinite, values) || error(
            "sbimpl: ordinal threshold predictor `$key` contains non-finite values")
        data[key] = collect(Float64, values)
        _sb_record_preproc!(data, key, PreprocEntry(
            :ordinal_threshold_predictor, nothing, key, false))
        push!(names, key)
    end
    Tuple(names)
end

function _sb_ordinal_discrimination_expr(raw, data)
    if raw isa Real
        isfinite(raw) && raw > 0 || error(
            "sbimpl: ordinal discrimination must be finite and strictly " *
            "positive, got $(repr(raw))")
    elseif raw isa NamedColumn && parent(raw) isa DataColumn
        values = parent(parent(raw))
        values isa AbstractVector{<:Real} && all(x -> isfinite(x) && x > 0, values) ||
            error("sbimpl: ordinal discrimination data `$(name(raw))` must " *
                  "contain only finite positive values")
    end
    _sb_scalar_expr(raw, data)
end

# Typed structure/link composition. Cumulative thresholds are ordered;
# stopping-ratio stage intercepts are deliberately unconstrained by ordering.
# A fixed eta intercept is non-identifiable with either threshold vector, so
# the new surface rejects it (the legacy OrderedLogistic shorthand remains
# unchanged for compatibility).
function _sb_lik_family!(stmts, target, ::Type{<:Ordinal},
                         args::Tuple{Any,Any,Any}, kwargs, data)
    structure_raw, link_raw, eta_raw = args
    structure = _sb_ordinal_structure_code(structure_raw)
    link = _sb_ordinal_link_code(link_raw)
    _sb_ordinal_has_fixed_intercept(eta_raw) && error(
        "sbimpl: `Ordinal($target)` cannot include a fixed intercept in `eta`; " *
        "the estimated thresholds already supply the location. Use `eta ~ 0 + ...`.")

    raw = get(data, target, nothing)
    raw isa AbstractVector || error(
        "sbimpl: `Ordinal` expects an observed outcome vector for `$target`, " *
        "got $(typeof(raw))")
    levels = _sb_fit_levels(raw)
    n_levels = length(levels)
    n_levels >= 2 || error(
        "sbimpl: `Ordinal($target)` needs at least two observed levels " *
        "(got $n_levels)")
    n_cut = n_levels - 1
    data[target] = _sb_apply_levels(levels, raw)
    _sb_record_preproc!(data, target, PreprocEntry(
        :ordinal_outcome, (; levels, n_levels), target, true))

    cut_name = Symbol(target, :_thresholds)
    if structure == 1
        push!(stmts, :($cut_name::ordered[$n_cut] ~ std_normal()))
    else
        push!(stmts, :($cut_name::vector[$n_cut] ~ std_normal()))
    end

    per_threshold_raw = get(kwargs, :per_threshold, ())
    per_threshold = _sb_ordinal_threshold_predictors!(
        data, target, per_threshold_raw, length(raw))
    structure == 1 && !isempty(per_threshold) && error(
        "sbimpl: `per_threshold` is currently supported for " *
        "`StoppingRatio()` only; unrestricted cumulative category-specific " *
        "effects can make cumulative probabilities non-monotone")

    effect_name = Symbol(target, :_threshold_effect)
    if isempty(per_threshold)
        push!(stmts, :($effect_name = rep_matrix(
            0., num_elements($target), $n_cut)))
    else
        X_name = Symbol(target, :_threshold_X)
        beta_name = Symbol(target, :_threshold_beta)
        n_terms = length(per_threshold)
        push!(stmts, Expr(:(=), X_name, Expr(:call, :hcat, per_threshold...)))
        # Stan's std_normal_lpdf is not matrix-vectorised. Reuse BRM's
        # array-of-vectors standard-normal prior, then rebuild a K-1 by p matrix
        # through the measured ranef_b_matrix helper for the design multiply.
        push!(stmts, :($beta_name::vector[$n_cut,$n_terms] ~ multi_std_normal()))
        beta_matrix = Expr(:call, :adjoint,
            Expr(:call, :ranef_b_matrix, beta_name))
        push!(stmts, Expr(:(=), effect_name,
            Expr(:call, :*, X_name, beta_matrix)))
    end

    eta = _sb_scalar_expr(eta_raw, data)
    discrimination = _sb_ordinal_discrimination_expr(
        get(kwargs, :discrimination, 1.0), data)
    _sb_lik_stan_exprs!(stmts, target, :brm_ordinal,
        (eta, cut_name, discrimination, structure, link, effect_name))
end
_sb_lik_family!(_, target, ::Type{<:Ordinal}, args, _, _) = error(
    "sbimpl: `Ordinal($target)` expects exactly three positional arguments " *
    "`(structure, link, eta)`, got $(length(args))")

_sb_lik_family!(stmts, target, ::Type{<:ZeroInflatedPoisson},
                args::Tuple{Any,Any}, data) =
    _sb_lik_stan!(stmts, target, :zero_inflated_poisson, args, data)

_sb_lik_family!(stmts, target, ::Type{<:NegativeBinomial2},
                args::Tuple{Any,Any}, data) =
    _sb_lik_stan!(stmts, target, :neg_binomial_2, args, data)

_sb_von_mises_observations(data, target) = begin
    raw = get(data, target, nothing)
    raw isa AbstractVector || error(
        "sbimpl: von-Mises likelihood expects an observed vector for `$target`, " *
        "got $(typeof(raw))")
    all(y -> y isa Real && isfinite(y), raw) || error(
        "sbimpl: von-Mises outcome `$target` must contain only finite real values")
    raw
end

function _sb_validate_von_mises!(data, target, mu, kappa;
                                 interval=nothing, principal=false)
    raw = _sb_von_mises_observations(data, target)
    if kappa isa Real
        isfinite(kappa) && kappa > 0 || error(
            "sbimpl: von-Mises concentration `kappa` must be finite and strictly " *
            "positive, got $(repr(kappa))")
    end
    if principal
        lo, hi = interval
        bad = findfirst(y -> !(lo <= y < hi), raw)
        isnothing(bad) || error(
            "sbimpl: `CircularVonMises($target)` observation $(repr(raw[bad])) " *
            "at index $bad is outside the half-open interval " *
            "[$lo, $hi)")
    elseif mu isa Real
        lo, hi = mu - Float64(pi), mu + Float64(pi)
        bad = findfirst(y -> !(lo <= y <= hi), raw)
        isnothing(bad) || error(
            "sbimpl: `VonMises($target)` observation $(repr(raw[bad])) at index " *
            "$bad is outside Distributions.jl support [$lo, $hi] for mu=$mu")
    end
    nothing
end

# Distributions.jl's exact constructor semantics. One positional argument is
# kappa (mu defaults to zero), while the two-argument form is `(mu, kappa)`.
# The custom lpxf adds the moving-support/strict-domain guards before calling
# native Stan `von_mises_lpdf`, and supplies matching pointwise/RNG hooks.
function _sb_lik_family!(stmts, target, ::Type{<:VonMises}, args, data)
    length(args) in (1, 2) || error(
        "sbimpl: `VonMises` expects `VonMises(kappa)` or " *
        "`VonMises(mu, kappa)`, got $(length(args)) positional arguments")
    arg_exprs = map(a -> _sb_scalar_expr(a, data), args)
    mu, kappa = _sb_stan_dist_args(VonMises, arg_exprs)
    _sb_validate_von_mises!(data, target, mu, kappa)
    _sb_lik_stan_exprs!(
        stmts, target, :brm_von_mises, (mu, kappa, 0.0, 0.0, 0))
end

function _sb_lik_family!(stmts, target, ::Type{<:CircularVonMises},
                         args::Tuple{Any,Any}, kwargs, data)
    mu, kappa = map(a -> _sb_scalar_expr(a, data), args)
    interval = _sb_circular_interval(kwargs)
    _sb_validate_von_mises!(data, target, mu, kappa;
                            interval, principal=true)
    lo, hi = interval
    _sb_lik_stan_exprs!(
        stmts, target, :brm_von_mises, (mu, kappa, lo, hi, 1))
end
_sb_lik_family!(_, target, ::Type{<:CircularVonMises}, args, _, _) = error(
    "sbimpl: `CircularVonMises($target)` expects exactly two positional " *
    "arguments `(mu, kappa)`, got $(length(args))")

# Reference-class categorical regression. The user supplies one named scalar
# LP per non-reference class; the fitted outcome level order determines which
# class each argument owns. A leading all-zero row fixes class 1 as the
# reference. StanBlocks' categorical-logit contract is matrix[K, N], one
# observation's K logits per column, so transpose the row-wise hcat carrier.
function _sb_lik_family!(stmts, target, ::Type{<:CategoricalLogit},
                         args::Tuple, data)
    isempty(args) && error(
        "sbimpl: `CategoricalLogit($target)` needs at least one non-reference " *
        "linear predictor")
    all(a -> a isa NamedColumn && !(parent(a) isa DataColumn), args) || error(
        "sbimpl: `CategoricalLogit($target)` expects one existing scalar linear " *
        "predictor per non-reference class, got $(args)")

    raw = get(data, target, nothing)
    raw isa AbstractVector || error(
        "sbimpl: `CategoricalLogit` expects an observed outcome vector for " *
        "`$target`, got $(typeof(raw))")
    levels = _sb_fit_levels(raw)
    n_levels = length(levels)
    n_levels >= 2 || error(
        "sbimpl: `CategoricalLogit($target)` needs >= 2 outcome levels " *
        "(got $n_levels)")
    expected_n_levels = length(args) + 1
    n_levels == expected_n_levels || error(
        "sbimpl: `CategoricalLogit($target)` observed $n_levels outcome levels " *
        "but received $(length(args)) non-reference predictors; expected " *
        "$(n_levels - 1). Outcome level order is $(collect(levels)).")

    data[target] = _sb_apply_levels(levels, raw)
    _sb_record_preproc!(data, target, PreprocEntry(
        :categorical_outcome, (; levels, n_levels), target, true))

    logits_name = Symbol(target, :_categorical_logits)
    zero_reference = :(rep_vector(0., num_elements($target)))
    eta_exprs = map(a -> _sb_scalar_expr(a, data), args)
    row_logits = Expr(:call, :hcat, zero_reference, eta_exprs...)
    push!(stmts, Expr(:(=), logits_name, Expr(:call, :adjoint, row_logits)))
    push!(stmts, :($target ~ categorical_logit($logits_name)))
end

# Mean/precision Beta-binomial convenience surface. Shape lowering stays in
# the emitted expression so scalar, vector, and linked-predictor arguments all
# share one method and StanBlocks can synthesize matching lpmfs/RNG paths.
function _sb_lik_family!(stmts, target, ::Type{<:BetaBinomial2},
                         args::Tuple{Any,Any,Any}, data)
    trials, mean, precision = map(a -> _sb_scalar_expr(a, data), args)
    alpha = Expr(:call, Symbol(".*"), mean, precision)
    beta = Expr(:call, Symbol(".*"), Expr(:call, :-, 1, mean), precision)
    push!(stmts, Expr(:call, :~, target,
        Expr(:call, :beta_binomial, trials, alpha, beta)))
end

# Distributions.jl expresses a location-scale Student-t as
# `LocationScale(loc, scale, TDist(nu))`. The prior path already supports this
# composition; likelihoods use the same lowering to Stan's
# `student_t(nu, loc, scale)` rather than rejecting the wrapper family.
function _sb_lik_family!(stmts, target, ::Type{<:LocationScale},
                         args::Tuple{Any,Any,Any}, data)
    loc, scale, base = args
    base_e = _as_expr_column(base)
    isnothing(base_e) && error(
        "sbimpl: `LocationScale(...)` third arg must be a distribution call, got $(typeof(base))")
    base_fam = getf(base_e)
    isnothing(_as_distribution_type(base_fam)) && error(
        "sbimpl: `LocationScale` base must be a Distribution type, got $(base_fam)")
    stan_name = _sb_stan_dist_name(base_fam)
    isnothing(stan_name) && error(
        "sbimpl: `LocationScale` over `$(base_fam)` -- no Stan-name mapping for the base.")
    base_args = _sb_stan_dist_args(
        base_fam, map(a -> _sb_scalar_expr(a, data), getargs(base_e)))
    length(base_args) >= 3 || error(
        "sbimpl: `LocationScale` over `$(base_fam)`: base lowers to ",
        "$(length(base_args)) Stan args, need >= 3 for location/scale slots.")
    composed = (base_args[1], _sb_scalar_expr(loc, data),
                _sb_scalar_expr(scale, data), base_args[4:end]...)
    _sb_lik_stan_exprs!(stmts, target, stan_name, composed)
end
# Single source of truth: Julia Distribution type -> Stan distribution
# function name. Both the likelihood path (`_sb_lik_family!` below) and
# the scalar-prior path (`_sb_emit_prior!` above) consult this. Adding
# a new family adds one entry here and both routes pick it up.
_sb_stan_dist_name(::Type{<:Normal})              = :normal
_sb_stan_dist_name(::Type{<:Cauchy})              = :cauchy
# Distributions.jl's `TDist(nu)` is 1-arg (df only); Stan's `student_t`
# is 3-arg (nu, mu, sigma). The name maps directly; the arg-shape
# adjustment is the `_sb_stan_dist_args` override below.
_sb_stan_dist_name(::Type{<:TDist})               = :student_t
_sb_stan_dist_name(::Type{<:Exponential})         = :exponential
_sb_stan_dist_name(::Type{<:Gamma})               = :gamma
_sb_stan_dist_name(::Type{<:Beta})                = :beta
_sb_stan_dist_name(::Type{<:Uniform})             = :uniform
_sb_stan_dist_name(::Type{<:LogNormal})           = :lognormal
_sb_stan_dist_name(::Type{<:Laplace})             = :double_exponential
_sb_stan_dist_name(::Type{<:SkewDoubleExponential}) = :skew_double_exponential
_sb_stan_dist_name(::Type{<:SkewedExponentialPower}) = :skew_double_exponential
_sb_stan_dist_name(::Type{<:Weibull})             = :weibull
_sb_stan_dist_name(::Type{<:InverseGamma})        = :inv_gamma
_sb_stan_dist_name(::Type{<:Bernoulli})           = :bernoulli
_sb_stan_dist_name(::Type{<:BernoulliLogit})      = :bernoulli_logit
_sb_stan_dist_name(::Type{<:Binomial})            = :binomial
_sb_stan_dist_name(::Type{<:BinomialLogit})       = :binomial_logit
_sb_stan_dist_name(::Type{<:BetaBinomial})        = :beta_binomial
_sb_stan_dist_name(::Type{<:Poisson})             = :poisson
_sb_stan_dist_name(::Type{<:NegativeBinomial})    = :neg_binomial
_sb_stan_dist_name(::Type) = nothing

# Per-family argument normalization between Julia constructors and native Stan
# distributions.  Inputs here are already-lowered Stan expressions.  Besides
# parameterization changes, preserve Distributions.jl's shorter constructor
# forms rather than emitting an invalid native-Stan arity.
_sb_stan_dist_args(::Type, args) = args

_sb_stan_reciprocal(x) = Expr(:call, Symbol("./"), 1.0, x)
_sb_stan_success_odds(p) =
    Expr(:call, Symbol("./"), p, Expr(:call, :-, 1.0, p))

_sb_stan_dist_args(::Type{<:Normal}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:Normal}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Cauchy}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:Cauchy}, args::Tuple{Any}) = (args[1], 1.0)

# `TDist(nu)` is standard Student-t; Stan requires explicit location/scale.
_sb_stan_dist_args(::Type{<:TDist}, args::Tuple{Any}) = (args[1], 0, 1)

# Explicit capability gate for Julia distribution composition. A Stan density
# name alone is insufficient: generic truncation/censoring additionally needs
# lcdf/lccdf companions with the same parameterization. Keep this list honest
# and executable rather than implicitly advertising every name-table entry.
_sb_cdf_family_kind(::Type) = nothing
_sb_cdf_family_kind(::Type{<:Normal})       = :continuous
_sb_cdf_family_kind(::Type{<:Exponential})  = :continuous
_sb_cdf_family_kind(::Type{<:LogNormal})    = :continuous
_sb_cdf_family_kind(::Type{<:Weibull})      = :continuous
_sb_cdf_family_kind(::Type{<:Poisson})      = :discrete

function _sb_composed_family(wrapper, args)
    length(args) in (1, 3) || error(
        "sbimpl: `$wrapper` expects a base distribution and either keyword ",
        "bounds or positional `(lower, upper)` bounds, got $(length(args)) arguments")
    base = _as_expr_column(first(args))
    isnothing(base) && error(
        "sbimpl: `$wrapper` first argument must be a distribution call, got ",
        "$(typeof(first(args)))")
    isempty(getkwargs(base)) || error(
        "sbimpl: `$wrapper` base distribution `$(getf(base))` cannot use formula keywords")
    family = getf(base)
    D = _as_distribution_type(family)
    isnothing(D) && error(
        "sbimpl: `$wrapper` base must be a Distributions.jl distribution type, ",
        "got `$family`")
    stan_name = _sb_stan_dist_name(D)
    isnothing(stan_name) && error(
        "sbimpl: `$wrapper` base family `$D` has no Stan distribution-name mapping")
    kind = _sb_cdf_family_kind(D)
    isnothing(kind) && error(
        "sbimpl: `$wrapper` base family `$D` has no generic CDF/CCDF composition ",
        "capability; add and test its density, pointwise, predictive, lcdf and ",
        "lccdf paths before advertising it")
    (; family=D, stan_name, stan_args=getargs(base), kind)
end

function _sb_wrapper_bounds(wrapper, args, kwargs::NamedTuple)
    if length(args) == 3
        isempty(kwargs) || error(
            "sbimpl: `$wrapper` cannot mix positional bounds with keyword bounds")
        return _sb_normalize_bound(args[2]), _sb_normalize_bound(args[3])
    end
    unknown = setdiff(collect(keys(kwargs)), [:lower, :upper])
    isempty(unknown) || error(
        "sbimpl: `$wrapper` accepts only `lower` and `upper` keywords, got $unknown")
    _sb_normalize_bound(get(kwargs, :lower, nothing)),
        _sb_normalize_bound(get(kwargs, :upper, nothing))
end

_sb_normalize_bound(::Nothing) = nothing
_sb_normalize_bound(x::NamedColumn) = _sb_is_nothing_column(x) ? nothing : x
_sb_normalize_bound(x) = x

_sb_bound_data(x::Real) = x
_sb_bound_data(x::AbstractVector{<:Real}) = x
_sb_bound_data(x::NamedColumn) = _sb_bound_data_named(x, parent(x))
_sb_bound_data_named(_x, d::DataColumn) = parent(d)
_sb_bound_data_named(x, backing) = error(
    "sbimpl: bound `$(name(x))` must be backed by observed data, got $(typeof(backing))")
_sb_bound_data(x) = error(
    "sbimpl: bounds must be numeric literals or observed data columns, got $(typeof(x))")

function _sb_validate_bounds(wrapper, target, lower, upper, data; check_order=true)
    y = data[target]
    for (label, bound) in ((:lower, lower), (:upper, upper))
        isnothing(bound) && continue
        b = _sb_bound_data(bound)
        b isa AbstractVector && length(b) != length(y) && error(
            "sbimpl: `$wrapper` $label bound has $(length(b)) rows but response ",
            "`$target` has $(length(y))")
    end
    if check_order && !isnothing(lower) && !isnothing(upper)
        lo, hi = _sb_bound_data(lower), _sb_bound_data(upper)
        ok = if lo isa AbstractVector || hi isa AbstractVector
            all(eachindex(y)) do i
                (lo isa AbstractVector ? lo[i] : lo) <=
                    (hi isa AbstractVector ? hi[i] : hi)
            end
        else
            lo <= hi
        end
        ok || error("sbimpl: `$wrapper` lower bounds must not exceed upper bounds")
    end
    nothing
end

function _sb_validate_composed_support(wrapper, target, lower, upper, kind, data)
    y = data[target]
    lo = isnothing(lower) ? nothing : _sb_bound_data(lower)
    hi = isnothing(upper) ? nothing : _sb_bound_data(upper)
    if kind === :discrete
        (eltype(y) <: Integer && !(eltype(y) <: Bool)) || error(
            "sbimpl: `$wrapper` discrete base family requires an integer response, ",
            "got $(eltype(y)) for `$target`")
        for (label, bound) in ((:lower, lower), (:upper, upper))
            isnothing(bound) && continue
            b = _sb_bound_data(bound)
            all(v -> v isa Integer && !(v isa Bool), b isa AbstractVector ? b : (b,)) ||
                error("sbimpl: `$wrapper` discrete $label bounds must be integers")
        end
    end
    all(eachindex(y)) do i
        lov = lo isa AbstractVector ? lo[i] : lo
        hiv = hi isa AbstractVector ? hi[i] : hi
        (isnothing(lov) || lov <= y[i]) && (isnothing(hiv) || y[i] <= hiv)
    end || error(
        "sbimpl: `$wrapper` response `$target` contains values outside its bounds")
    nothing
end

# StanBlocks decision 1wd43wt: one base-family token plus compile-time optional
# `lower` / `upper` kwargs. BRM always spells both; an absent bound is the Julia
# value `nothing`, which the HOF consumes before Stan name/type resolution.
_sb_composed_stan_args(base, data) = _sb_stan_dist_args(
    base.family, map(a -> _sb_scalar_expr(a, data), base.stan_args))

function _sb_emit_optional_family!(stmts, target, producer, base, lower, upper, data)
    family_args = _sb_composed_stan_args(base, data)
    lower_expr = isnothing(lower) ? nothing : _sb_scalar_expr(lower, data)
    upper_expr = isnothing(upper) ? nothing : _sb_scalar_expr(upper, data)
    rhs = Expr(:call, producer,
        Expr(:parameters,
            Expr(:kw, :lower, lower_expr),
            Expr(:kw, :upper, upper_expr)),
        base.stan_name, family_args...)
    push!(stmts, Expr(:call, :~, target, rhs))
end

function _sb_lik_composed!(stmts, target, wrapper, producer,
                           args, kwargs::NamedTuple, data)
    base = _sb_composed_family(wrapper, args)
    lower, upper = _sb_wrapper_bounds(wrapper, args, kwargs)

    isnothing(lower) && isnothing(upper) && error(
        "sbimpl: `$wrapper` needs at least one non-`nothing` bound")

    _sb_validate_bounds(wrapper, target, lower, upper, data)
    _sb_validate_composed_support(wrapper, target, lower, upper, base.kind, data)
    _sb_emit_optional_family!(stmts, target, producer, base, lower, upper, data)
end

_sb_lik_family!(stmts, target, ::typeof(truncated), args, kwargs::NamedTuple, data) =
    _sb_lik_composed!(stmts, target, :truncated, :conditioned, args, kwargs, data)

_sb_lik_family!(stmts, target, ::typeof(censored), args, kwargs::NamedTuple, data) =
    _sb_lik_composed!(stmts, target, :censored, :clamped, args, kwargs, data)

# Genuine interval evidence uses the observed response as the lower endpoint.
# Its producer call has no optional-bound encoding, so it is independent of the
# one-sided truncation/censoring producer decision.
function _sb_lik_family!(stmts, target, ::typeof(interval_censored),
                         args, kwargs::NamedTuple, data)
    length(args) == 1 || error(
        "sbimpl: `interval_censored` expects one base distribution argument")
    keys(kwargs) == (:upper,) || error(
        "sbimpl: `interval_censored` requires exactly the `upper` keyword; ",
        "the response column is the interval lower endpoint")
    base = _sb_composed_family(:interval_censored, args)
    upper = kwargs.upper
    _sb_validate_bounds(:interval_censored, target, data[target], upper, data;
                        check_order=false)
    lo, hi = data[target], _sb_bound_data(upper)
    all(eachindex(lo)) do i
        lo[i] < (hi isa AbstractVector ? hi[i] : hi)
    end || error(
        "sbimpl: `interval_censored` lower endpoints must be strictly below upper endpoints")
    _sb_validate_composed_support(:interval_censored, target, data[target],
                                  upper, base.kind, data)
    family_args = _sb_composed_stan_args(base, data)
    upper_expr = _sb_scalar_expr(upper, data)
    _sb_lik_stan_exprs!(stmts, target, :interval_evidence,
                        (base.stan_name, target, upper_expr, family_args...))
end

# Distributions.jl uses scale `theta`; Stan uses inverse scale (rate) `beta`.
_sb_stan_dist_args(::Type{<:Exponential}, ::Tuple{}) = (1.0,)
_sb_stan_dist_args(::Type{<:Exponential}, args::Tuple{Any}) =
    (_sb_stan_reciprocal(args[1]),)
_sb_stan_dist_args(::Type{<:Gamma}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:Gamma}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Gamma}, args::Tuple{Any,Any}) =
    (args[1], _sb_stan_reciprocal(args[2]))

_sb_stan_dist_args(::Type{<:Beta}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:Beta}, args::Tuple{Any}) = (args[1], args[1])
_sb_stan_dist_args(::Type{<:Uniform}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:LogNormal}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:LogNormal}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Laplace}, ::Tuple{}) = (0.0, 1.0)
_sb_stan_dist_args(::Type{<:Laplace}, args::Tuple{Any}) = (args[1], 1.0)

# Distributions.jl's asymmetric-Laplace special case keeps its own scale.
# Only an explicit literal p=1 is accepted: the general SEPD has no faithful
# native Stan analogue. All three Stan paths consume this one translation.
function _sb_stan_dist_args(
    ::Type{<:SkewedExponentialPower},
    args::Tuple{Any,Any,Any,Any},
)
    mu, sigma_sepd, p, alpha = args
    p isa Real && p == one(p) || throw(ArgumentError(
        "sbimpl: `SkewedExponentialPower` is supported only with the explicit " *
        "literal shape `p = 1`; got $(repr(p))"))
    one_minus_alpha = Expr(:call, :-, 1.0, alpha)
    stan_scale = Expr(:call, Symbol(".*"),
        Expr(:call, Symbol(".*"),
            Expr(:call, Symbol(".*"), 4.0, sigma_sepd), alpha),
        one_minus_alpha)
    (mu, stan_scale, alpha)
end
_sb_stan_dist_args(::Type{<:SkewedExponentialPower}, args) =
    throw(ArgumentError(
        "sbimpl: `SkewedExponentialPower` requires four explicit arguments " *
        "`(mu, sigma, 1, alpha)`; got $(length(args))"))

_sb_stan_dist_args(::Type{<:Weibull}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:Weibull}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:InverseGamma}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:InverseGamma}, args::Tuple{Any}) = (args[1], 1.0)
_sb_stan_dist_args(::Type{<:Bernoulli}, ::Tuple{}) = (0.5,)
_sb_stan_dist_args(::Type{<:BernoulliLogit}, ::Tuple{}) = (0.0,)
_sb_stan_dist_args(::Type{<:Binomial}, ::Tuple{}) = (1, 0.5)
_sb_stan_dist_args(::Type{<:Binomial}, args::Tuple{Any}) = (args[1], 0.5)
_sb_stan_dist_args(::Type{<:Poisson}, ::Tuple{}) = (1.0,)

_sb_stan_dist_args(::Type{<:VonMises}, args::Tuple{Any}) = (0.0, args[1])

# Distributions.jl `NegativeBinomial(r, p)` counts failures before `r`
# successes.  Native Stan `neg_binomial(alpha, beta)` uses shape and inverse
# scale, with the exact translation alpha=r, beta=p/(1-p).
_sb_stan_dist_args(::Type{<:NegativeBinomial}, ::Tuple{}) = (1.0, 1.0)
_sb_stan_dist_args(::Type{<:NegativeBinomial}, args::Tuple{Any}) =
    (args[1], 1.0)
_sb_stan_dist_args(::Type{<:NegativeBinomial}, args::Tuple{Any,Any}) =
    (args[1], _sb_stan_success_odds(args[2]))

# Default: look up the Stan name from the table and emit
# `target ~ <stan-name>(<lowered-args>...)` via `_sb_lik_stan!`. Families
# that need bespoke handling (custom args, side effects, mangled
# arities) override on a more-specific signature -- see OrderedLogistic
# above.
_sb_lik_family!(stmts, target, ::Type{D}, args, data) where {D <: Distribution} =
    let stan_name = _sb_stan_dist_name(D)
        isnothing(stan_name) && error(
            "sbimpl: likelihood family `$(D)` not supported yet -- ",
            "no `_sb_stan_dist_name` entry. Add one (and a `_sb_stan_dist_args` ",
            "override if its args don't lower verbatim).")
        arg_exprs = map(a -> _sb_scalar_expr(a, data), args)
        _sb_lik_stan_exprs!(
            stmts, target, stan_name, _sb_stan_dist_args(D, arg_exprs))
    end

_sb_lik_family!(stmts, target, fam, args, _) =
    error("sbimpl: likelihood family `$fam` (arity $(length(args))) not supported yet")


# ---- scalar-expression reducer (unwraps NamedColumn references etc.) --------

_sb_scalar_expr(x::Symbol, _) = x
_sb_scalar_expr(x::Real, _) = x
_sb_scalar_expr(x::NamedColumn, data) = begin
    _record_scalar_data!(data, name(x), parent(x))
    name(x)
end
_record_scalar_data!(data, sym, d::DataColumn) = (data[sym] = parent(d); nothing)
_record_scalar_data!(args...) = nothing
# Formula arithmetic is element-wise by intent -- `loc = loc_loc + loc_slope * cdslope`
# on two length-n vectors means Stan's `.*`, not matrix/dot product. Translate `*`
# and `/` to their dotted variants so Stan's typechecker accepts vector-vector
# operands (and scalar operands broadcast correctly in either form). Addition /
# subtraction already element-wise-broadcast in Stan between vectors, no change
# needed there.
_sb_scalar_expr(x::ExprColumn, data) = begin
    f = getf(x)
    op = f === (*) ? Symbol(".*") :
         f === (/) ? Symbol("./") :
         f
    Expr(:call, op, (_sb_scalar_expr(a, data) for a in getargs(x))...)
end
_sb_scalar_expr(x, _) = error("sbimpl: cannot lift to Stan expression: $(typeof(x)): $x")

# ==============================================================================
# Bordet model family — BRM-side composition surface.
#
# Representative model formula (log_y MUST precede log_obs so sigma is in scope):
#
#   brmi = @brm df begin
#       log_y ~ bordet_hierarchical_parametric(;
#           time, dose, person_idxs, biomarker_idxs,
#           sigma_rate, scale_rate, hierarchical_centeredness
#       )
#       log_obs ~ TruncatedNormal(log_y, sigma, lloq, uloq)
#   end
#
# Cross-tree contract (StanBlocks:bordet, decision 1lmystf):
#   - `truncated_normal` lpxf triad: censored normal (LLOQ→lcdf, ULOQ→lccdf)
#   - `bordet_time_response(log_time, loc, log_slope, mag)::vector` bell kernel
#   - `bordet_dose_response(log_dose, loc, log_slope)::vector` — returns LOG
#   - Transdata builtins: `linear_idxs`, `broadcasted_max`, `broadcasted_gt`
#   All registered in StanBlocks' builtin module; reachable with no import.
# ==============================================================================

"""
    bordet_hierarchical_parametric

BRM formula marker for the bordet hierarchical parametric mean submodel.
All data columns and hyperparameters are passed as keyword arguments.
Emits (in order): sizes (n_biomarkers, n_persons, n_series), sigma prior
(biomarker-level exponential), transdata (series_idxs, log_time, log_dose,
affectable), 6 per-series parameter priors, and log_y via StanBlocks:bordet
kernel builtins (bordet_time_response + bordet_dose_response).
"""
function bordet_hierarchical_parametric end

# TruncatedNormal custom likelihood.
# Emits: target ~ truncated_normal(mean, sigma[biomarker_idxs], lloq[biomarker_idxs], uloq[biomarker_idxs])
# biomarker_idxs is in scope because bordet_hierarchical_parametric emits it as data.
function _sb_lik_family!(stmts, target, ::typeof(TruncatedNormal), args, data)
    length(args) == 4 || error(
        "sbimpl: TruncatedNormal likelihood expects 4 positional args ",
        "(mean, sigma, lloq, uloq), got $(length(args))")
    mean_arg, sigma_arg, lloq_arg, uloq_arg = args
    mean_sym  = _sb_scalar_expr(mean_arg, data)
    sigma_sym = _sb_scalar_expr(sigma_arg, data)
    lloq_sym  = _sb_scalar_expr(lloq_arg, data)
    uloq_sym  = _sb_scalar_expr(uloq_arg, data)
    push!(stmts, :($target ~ truncated_normal(
        $mean_sym,
        $(sigma_sym)[biomarker_idxs],
        $(lloq_sym)[biomarker_idxs],
        $(uloq_sym)[biomarker_idxs])))
end

# Group-block declaration for bordet_hierarchical_parametric.
# The 6 per-(biomarker×person) params (baseline, time_loc, time_log_slope,
# time_mag, dose_loc, dose_log_slope) are correlated normals via BRM's
# ranef_correlated_draws floor. The group is series_idxs = linear_idxs(bm, pn)
# — a computed column, so group_fn synthesises it from the kwargs in data.
_sb_term_group_block(::typeof(bordet_hierarchical_parametric)) = (;
    n_per_group  = 6,
    group_fn     = (rhs_e, data) -> begin
        bm   = data[:biomarker_idxs]
        pn   = data[:person_idxs]
        n_bm = maximum(bm)
        series = bm .+ (pn .- 1) .* n_bm
        NamedColumn(:series_idxs, DataColumn(series))
    end,
    group_fn_name = :series_idxs,
)

# Emit hook: allocates sigma, transdata, and log_y from the group-block matrix.
# block_info = (; block_name, idx_name, n_per_group)
#   block_name  — n_series × 6 matrix from ranef_correlated_draws (col order
#                 matches the param order below)
#   idx_name    — per-obs integer index into block rows (= series_idxs_idx)
function _sb_emit_group_block_term!(stmts, data, target,
                                     ::typeof(bordet_hierarchical_parametric),
                                     rhs_e, block_info)
    (; block_name, idx_name) = block_info
    # Sizes: deterministic transdata from int[] index arrays
    push!(stmts, :(n_biomarkers = max(biomarker_idxs)))
    push!(stmts, :(n_persons    = max(person_idxs)))
    push!(stmts, :(n_series     = n_biomarkers * n_persons))
    # sigma: biomarker-level exponential prior
    push!(stmts, :(sigma ~ exponential(sigma_rate; n=n_biomarkers, lower=0.)))
    # Derived transdata: obs-level index + input transforms
    push!(stmts, :(series_idxs = linear_idxs(biomarker_idxs, person_idxs)))
    push!(stmts, :(log_time    = log(broadcasted_max(time, 0.001))))
    push!(stmts, :(log_dose    = log(broadcasted_max(dose, 0.1))))
    push!(stmts, :(affectable  = broadcasted_gt(time .* dose, 0.0)))
    # Extract 6 per-series params from the group-block matrix (row = series, col = param)
    push!(stmts, :(baseline       = $(block_name)[$(idx_name), 1]))
    push!(stmts, :(time_loc       = $(block_name)[$(idx_name), 2]))
    push!(stmts, :(time_log_slope = $(block_name)[$(idx_name), 3]))
    push!(stmts, :(time_mag       = $(block_name)[$(idx_name), 4]))
    push!(stmts, :(dose_loc       = $(block_name)[$(idx_name), 5]))
    push!(stmts, :(dose_log_slope = $(block_name)[$(idx_name), 6]))
    # log_y via StanBlocks:bordet kernel builtins
    push!(stmts, :(time_response = bordet_time_response(log_time, time_loc, time_log_slope, time_mag)))
    push!(stmts, :(dose_response = exp(bordet_dose_response(log_dose, dose_loc, dose_log_slope))))
    push!(stmts, :($target = baseline + affectable .* time_response .* dose_response))
end
