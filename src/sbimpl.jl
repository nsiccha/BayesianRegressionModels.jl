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

Smooth-term marker (cubic-spline predictor). brms-style `s(x)` — the
backend builds a natural cubic-spline basis with default interior knots
and emits a linear-in-basis predictor. Dispatch tag — see `_sb_s`.
"""
function s end

"""
    ar(time; p=1)

Autoregressive-noise predictor marker. Adds an AR(p) noise process
ordered by `time`. Only `p=1` is supported in the current sbimpl
emitter. Dispatch tag — see `_sb_ar1`.
"""
function ar end

"""
    OrderedLogistic

Cumulative-link ordinal-likelihood marker. Use as a family on the RHS:
`y ~ OrderedLogistic(eta)`. The sbimpl backend lowers to Stan's
`ordered_logistic_lpmf`. Marker struct only — Distributions.jl does not
ship an `OrderedLogistic`, and the `@brm` parser never constructs an
instance, so the empty struct is sufficient.
"""
struct OrderedLogistic end

"""
    Horseshoe

Carvalho-Polson-Scott horseshoe shrinkage prior marker. Use as a prior
on a coefficient: `coef ~ Horseshoe()`. sbimpl emits the standard
reparameterised hierarchy `beta = raw * lambda * tau`. Marker struct
only — the `@brm` parser never constructs an instance.
"""
struct Horseshoe end

"""
    ZeroInflatedPoisson(lambda, zi)

Zero-inflated Poisson likelihood marker — a mixture of a point-mass at
zero (with probability `zi`) and `Poisson(lambda)`. Surfaces as
`y ~ ZeroInflatedPoisson(lambda, zi)`; sbimpl routes through
`zero_inflated_poisson_lpmf`. Marker struct only.
"""
struct ZeroInflatedPoisson end

popefs = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ std_normal(; n=n_covariates)
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
ranef_intercept = StanBlocks.@slic begin
    log_scale ~ std_normal()
    xi ~ std_normal(; n=n_groups)
    return exp(log_scale) * xi[group_idx]
end

# Correlated random effects for K terms x G groups. brms-style (1 + x + y | g).
# Non-centered parameterization:
#   L   ~ lkj_corr_cholesky(1, K)            # K x K Cholesky factor
#   tau ~ half-std_normal(; n=K)             # per-term marginal scales
#   z   ~ std_normal(; n=K, m=n_groups)      # K x n_groups std normal
#   b   = (diag_pre_multiply(tau, L) * z)'   # n_groups x K correlated draws
# Per-row contribution = Z[i, :] . b[group_idx[i], :], returned as a length-n
# vector via rows_dot_product. Note: `(1 | g) + (0 + x | g)` and `(1 + x | g)`
# are equivalent -- the walker merges everything sharing a group symbol into
# one correlated block.
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
ranef_correlated_draws = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(1.; n=n_terms)
    tau    ~ std_normal(; n=n_terms, lower=0.)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    return (diag_pre_multiply(tau, L) * z)'   # n_groups x n_terms
end

# ---- cv-contagious ranef variants (opt-in; for out-of-sample / CV models) ----
#
# StanBlocks' cv-contagion (see StanBlocks forward.jl:321 + types.jl:215):
# a parameter taints to a `:quantities` (generated-quantities re-draw) qual iff
# its TYPE is cv, and a type is cv iff its own flag OR *any element of its size*
# is cv. So a random effect flips to a GQ population re-draw exactly when its
# SIZE traces from a cv-marked input. The default `ranef_*` submodels above size
# the std-normal draw from the standalone data scalar `n_groups`, which carries
# no taint -- so `maybecv(:<group>_idx)` reaches `group_idx` but never the RE's
# size, and the RE stays a fitted parameter (the QT out-of-sample FAIL).
#
# These `_cv` variants are identical in *value* but size the std-normal draw
# from `maximum(group_idx)` instead, computed INSIDE the body so the cv taint on
# `group_idx` propagates into the size (`maximum` -> `cv` via passes.jl:23) and
# flips the draw to a generated-quantities re-draw. `L`/`tau` keep their
# `n_terms` (untainted) sizing, so under marking ONLY the standardised draw `z`
# is re-drawn -- a leave-all-out population re-draw from the fitted covariance,
# the semantics confirmed for QT's `source` knob.
#
# They are OPT-IN: emitted only for groups passed in `SBBRMI(...; cv_groups=...)`
# (used when generating a CV model artifact, e.g. qt_cv.stan). The default build
# keeps the `n_groups`-sized submodels above, so committed models are emitted
# byte-for-byte unchanged and never recompile.
#
# Scope: only the clean `std_normal(; n=...)` forms below can flip via a size
# change. The typed-LHS `_by`/stratified path (`z::vector[n_groups, n_terms] ~
# multi_std_normal()`) cannot -- StanBlocks' typed-LHS forward (forward.jl:355)
# derives cv from the RHS call-args, not the declared-size, so a cv size there
# yields a cv *parameter*, not a `:quantities` re-draw. Making `gr(g, by=b)`
# blocks and `(e | ID | g)` buckets cv-contagious needs either a StanBlocks
# typed-LHS fix or a bare-`z` rewrite, and neither is in QT's current use case.
ranef_intercept_cv = StanBlocks.@slic begin
    log_scale ~ std_normal()
    xi ~ std_normal(; n=maximum(group_idx))
    return exp(log_scale) * xi[group_idx]
end

ranef_correlated_cv = StanBlocks.@slic begin
    n_g    = maximum(group_idx)
    L      ~ lkj_corr_cholesky(1.; n=n_terms)
    tau    ~ std_normal(; n=n_terms, lower=0.)
    z_flat ~ std_normal(; n=n_terms * n_g)
    z = reshape(z_flat, n_terms, n_g)
    b = (diag_pre_multiply(tau, L) * z)'   # n_groups x n_terms
    return rows_dot_product(Z, b[group_idx, :])
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
    # Hand-rolled per-element loop returning vector[n]. Mirrors the
    # `ordered_logistic_lpmfs` pattern in StanBlocks/builtin.jl --
    # avoids `jbroadcasted` which lives in `StanBlocks.builtin` and may
    # not be reachable from a user-side @deffun's symbol resolver.
    zero_inflated_poisson_lpmfs(args...) = begin
        zero_inflated_poisson_lpmf(args...)
    end
    zero_inflated_poisson_lpmfs(y::int[n], lambda, zi) = begin
        rv::vector[n]
        for i in 1:n
            rv[i] = zero_inflated_poisson_lpmf(y[i], lambda[i], zi[i])
        end
        rv
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

# Minimal `s(x)` cubic-spline predictor. Uses a truncated-power basis for a
# natural cubic spline with n_interior interior knots placed at equally-spaced
# quantiles of `x`. The submodel takes the precomputed N x n_basis matrix
# `X_basis` (raw columns: x, x^2, x^3, (x - k_j)^3_+ for each interior knot)
# and returns `X_basis * coefs` -- a length-N smooth contribution. The caller
# emits `s_<x> ~ _sb_s(; X_basis=..., n_basis=...)` and hands `s_<x>` to
# popefs as a single design-matrix column (so there is one extra overall beta
# multiplying the smooth; a direct-summand variant would drop that beta).
# This is a first-pass implementation: no penalty / smoothness prior beyond
# std_normal on the basis coefficients, no tensor products, no bs/t2/gp.
_sb_s = StanBlocks.@slic begin
    n_basis = dims(X_basis)[2]
    coefs ~ std_normal(; n=n_basis)
    return X_basis * coefs
end

function _sb_spline_basis_ncs(x::AbstractVector{<:Real}; n_interior::Int=2)
    n_interior >= 0 || error("sbimpl: `s(x)` needs n_interior >= 0 (got $n_interior)")
    xs = collect(Float64, x)
    n = length(xs)
    sorted = sort(xs)
    # Equally-spaced quantiles at 1/(n_interior+1), ..., n_interior/(n_interior+1)
    knots = Float64[]
    for j in 1:n_interior
        q = j / (n_interior + 1)
        pos = 1 + q * (n - 1)
        lo  = floor(Int, pos); hi = ceil(Int, pos)
        val = lo == hi ? sorted[lo] : sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo])
        push!(knots, val)
    end
    n_basis = 3 + n_interior
    M = zeros(Float64, n, n_basis)
    for i in 1:n
        xi = xs[i]
        M[i, 1] = xi
        M[i, 2] = xi * xi
        M[i, 3] = xi * xi * xi
        for (j, k) in enumerate(knots)
            d = xi - k
            M[i, 3 + j] = d > 0 ? d^3 : 0.0
        end
    end
    M, n_basis
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

# Hilbert-space approximate Gaussian process (Riutort-Mayol et al. 2022),
# 1D squared-exponential kernel. Inputs precomputed by the caller:
#   PHI    : N x K eigen-basis matrix (sin terms)
#   lambda : length-K squared eigenvalues
# Parameters:
#   log_rho   -- log length-scale
#   log_sigma -- log marginal sd
#   beta_raw  -- length-K standard-normal basis weights
# Spectral-density-scaled basis weights `sqrt_spd .* beta_raw` give the
# usual GP draw f(x) = PHI * (sqrt_spd .* beta_raw). Returns f as a
# length-N column the caller (popefs) multiplies by an overall beta.
_sb_hsgp = StanBlocks.@slic begin
    n_basis = num_elements(lambda)
    log_rho   ~ std_normal()
    log_sigma ~ std_normal()
    beta_raw  ~ std_normal(; n=n_basis)
    rho   = exp(log_rho)
    sigma = exp(log_sigma)
    # sqrt(spectral density of squared-exp kernel) at omega = sqrt(lambda).
    # = sigma * sqrt(rho * sqrt(2*pi)) * exp(-0.25 * rho^2 * lambda).
    # Inlined sqrt(2*pi) constant since SLIC has no `pi` builtin.
    sqrt_spd = sigma * sqrt(rho * 2.5066282746310002) * exp(-0.25 * rho * rho * lambda)
    return PHI * (sqrt_spd .* beta_raw)
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
    SBBRMI(brmi::BRMI; mod=@__MODULE__, cv_groups=Set{Symbol}()) -> SBBRMI

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
generating a CV model artifact; the default (empty `cv_groups`) emits the
group RE byte-for-byte as before, so committed models never recompile. Only
plain `(… | g)` ranefs are supported; stratified `gr(g, by=b)` and
cross-formula `(… |ID| g)` buckets error if opted-in (see `ranef_*_cv`).

Use [`stan_code`](@ref) to extract the transpiled Stan source. For
sampling, load `StanLogDensityProblems` + `BridgeStan` and wrap the
emitted `SlicModel` in a `StanProblem`.

```julia
brmi  = @brm df (y ~ 1 + a + (1|g))
sbbrmi = SBBRMI(brmi)
src   = stan_code(sbbrmi)
```
"""
struct SBBRMI{P<:BRMI, M, D<:AbstractDict}
    parent::P
    model::M
    data::D
end

SBBRMI(brmi::BRMI; mod::Module=@__MODULE__, cv_groups=Set{Symbol}()) = begin
    cv_groups = cv_groups isa Set ? cv_groups : Set{Symbol}(cv_groups)
    stmts = Any[]
    data = Dict{Symbol,Any}()
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
    # Prepass 1: stash every data-backed NamedColumn so later intercept-only
    # predictors have a length probe to hang `rep_vector(1., num_elements(...))`
    # off, regardless of iteration order. Decorators that materialise their
    # own columns (e.g. `mi`'s `_sb_emit_mi!`) claim them via the prepass'
    # `:skip_data` bucket; the data-collection pass honours that.
    skip_data = get(prepass, :skip_data, Set{Symbol}())
    for (_, op) in pairs(brmi.operations)
        _sb_collect_data!(data, op; skip=skip_data)
    end
    # Prepass 2: collect brms-style `|ID|` ranef buckets across all sub-formulas,
    # emit one shared ranef_correlated_draws per bucket, and build a lookup
    # `(brmi_key, (id_sym, group_key)) => (bucket_name, col_range, idx_name, suffix)`
    # for per-sub-formula emission below.
    id_buckets = _sb_collect_id_buckets(brmi)
    id_lookup = _sb_emit_id_buckets!(stmts, data, id_buckets)
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
        _sb_emit!(stmts, data, key, parent(nc); id_lookup, obs_n, cv_groups, group_block_lookup)
    end
    body = Expr(:block, stmts...)
    model = StanBlocks.SlicModel(body, data, mod)
    SBBRMI(brmi, model, data)
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


# ---- top-level op dispatch ---------------------------------------------------

_sb_emit!(stmts, data, key, op::ExprColumn; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), group_block_lookup=Dict()) =
    _sb_emit_expr!(stmts, data, key, getf(op), op; id_lookup, obs_n, cv_groups, group_block_lookup)
# Raw data / missing columns appear as top-level ops when the formula mentions
# them as bare references (e.g. `c2` in `loc ~ 1 + c2`). Nothing to emit — the
# prepass already stashed data columns in `data`.
_sb_emit!(stmts, data, key, ::DataColumn; kwargs...) = nothing
_sb_emit!(stmts, data, key, ::MissingColumn; kwargs...) = nothing
_sb_emit!(stmts, data, key, op; kwargs...) = error("sbimpl: top-level op for `$key` not an ExprColumn (got $(typeof(op)))")

_sb_emit_expr!(stmts, data, key, ::typeof(~), op; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), group_block_lookup=Dict()) = begin
    lhs, rhs = getargs(op, 2)
    _sb_sampling!(stmts, data, key, lhs, rhs; id_lookup, obs_n, cv_groups, group_block_lookup)
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

# ---- group-block declaration + emit API -------------------------------------
#
# A submodel term author declares K normally-distributed correlated per-group
# params by defining a `_sb_term_group_block(::typeof(term))` method. BRM's
# Prepass 2.5 reads the declaration, allocates one ranef_correlated_draws
# block per (term-function, group-column) pair, and threads the un-expanded
# n_groups×K matrix into the term at emit time via `_sb_emit_group_block_term!`.
#
# Declaration shape: (; n_per_group::Int, group_arg_pos::Int)
#   n_per_group   — K params per group (fixed by the term, not user-adjustable)
#   group_arg_pos — which positional arg of the term call is the grouping column
#                   (defaults to 1 when reading via `get(decl, :group_arg_pos, 1)`)
#
# Use `import BayesianRegressionModels: _sb_emit_group_block_term!` when adding
# methods from a downstream module so the binding is extended, not shadowed.

# Default: no group block. Override for declaring terms.
_sb_term_group_block(_) = nothing

# Toy term declaration: 2 correlated params per group, first arg is the group.
_sb_term_group_block(::typeof(sb_group_demo)) = (; n_per_group=2, group_arg_pos=1)

# Emit hook for group-block terms. block_info = (; block_name, idx_name, n_per_group).
# Default errors so a term with a declaration but no emit method is caught early.
_sb_emit_group_block_term!(stmts, data, target, f, rhs_e, block_info) =
    error("sbimpl: `$(nameof(f))` declared a group block but has no ",
          "`_sb_emit_group_block_term!` method — define one.")

# Toy term emit: thread group_block + group_idx into sb_group_demo_slic.
function _sb_emit_group_block_term!(stmts, data, target, ::typeof(sb_group_demo),
                                     rhs_e, block_info)
    (; block_name, idx_name) = block_info
    push!(stmts, :($target ~ sb_group_demo_slic(;
        group_block=$block_name, group_idx=$idx_name)))
end

# Built-in prior families on a missing-LHS sampling statement (e.g.
# `coef_a ~ Horseshoe()`). Returns `true` if it consumed the binding,
# `false` otherwise (then `_sb_linear_predictor!` runs).
_sb_emit_prior!(stmts, target, ::Type{<:Horseshoe}, _) = begin
    push!(stmts, :($target ~ _sb_horseshoe()))
    true
end
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
    arg_exprs = map(_sb_prior_arg, _sb_stan_dist_args(D, getargs(op)))
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
    base_args = _sb_stan_dist_args(base_fam, getargs(base))
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
    composed = (base_args[1], mu, sigma, base_args[4:end]...)
    arg_exprs = map(_sb_prior_arg, composed)
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
_sb_sampling!(stmts, data, key, lhs::NamedColumn, rhs; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), group_block_lookup=Dict()) =
    _sb_sampling_backed!(stmts, data, key, parent(lhs), rhs; id_lookup, obs_n, cv_groups, group_block_lookup)

_sb_sampling_backed!(stmts, data, key, backing::DataColumn, rhs; id_lookup, kwargs...) = begin
    data[key] = _sb_data_vec(key, parent(backing))
    _sb_likelihood!(stmts, key, rhs, data)
end

_sb_sampling_backed!(stmts, data, key, backing::MissingColumn, rhs;
                     id_lookup, obs_n=nothing, cv_groups=Set{Symbol}(),
                     group_block_lookup=Dict()) = begin
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
    _sb_linear_predictor!(stmts, data, key, rhs; id_lookup, brmi_key=key, obs_n, cv_groups)
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
_sb_sampling!(stmts, data, key, lhs::ExprColumn, rhs; id_lookup=_sb_empty_id_lookup(), obs_n=nothing, cv_groups=Set{Symbol}(), group_block_lookup=Dict()) =
    _sb_sampling_through_link!(stmts, data, key, getf(lhs), only(getargs(lhs)), rhs; id_lookup, obs_n, cv_groups)

_sb_sampling_through_link!(stmts, data, key, f, inner, rhs; kwargs...) =
    error("sbimpl: expected NamedColumn inside link `$f(...)`, got $(typeof(inner))")
function _sb_sampling_through_link!(stmts, data, key, f, inner::NamedColumn, rhs; id_lookup, obs_n=nothing, cv_groups=Set{Symbol}())
    inv_f = InverseFunctions.inverse(f)
    inner_name = name(inner)
    pre_name = Symbol(nameof(f), :_, inner_name)
    _sb_linear_predictor!(stmts, data, pre_name, rhs; id_lookup, brmi_key=key, obs_n, cv_groups)
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
# s, gp, ar, mo1) and a few `mi`-style sites. Each step is a tiny dispatch
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
    f === mo1 && (push!(direct_terms, t); return)
    push!(pop_terms, t)
end
_sb_classify_term!(t, pop_terms, ran_terms, direct_terms) =
    isnothing(_sb_cat_levels(t)) ? push!(pop_terms, t) : push!(direct_terms, t)

function _sb_linear_predictor!(stmts, data, target::Symbol, rhs;
                                id_lookup=_sb_empty_id_lookup(),
                                brmi_key::Symbol=target,
                                obs_n::Union{Symbol,Nothing}=nothing,
                                cv_groups=Set{Symbol}())
    terms = _sb_terms(rhs)
    pop_terms    = Any[]
    ran_terms    = Any[]  # `(expr | group)` -> collected per-group below
    direct_terms = Any[]  # e.g. `mo1(c)` -> direct summand, no popefs beta
    for t in terms
        _sb_classify_term!(t, pop_terms, ran_terms, direct_terms)
    end
    isempty(pop_terms) && isempty(ran_terms) && isempty(direct_terms) &&
        error("sbimpl: empty RHS for `$target` — no predictor terms")

    summands = Symbol[]

    if !isempty(pop_terms)
        col_exprs = Any[]
        for t in pop_terms
            _sb_pop_cols!(col_exprs, t, data, stmts, pop_terms; obs_n)
        end
        X_name = Symbol(:X_, target)
        pop_name = Symbol(:pop_, target)
        # StanBlocks `hcat` promotes a lone vector to matrix[n,1] and folds to
        # append_col for two-or-more columns, so we can always just emit hcat.
        push!(stmts, :($X_name = $(Expr(:call, :hcat, col_exprs...))))
        push!(stmts, :($pop_name ~ popefs(; X=$X_name)))
        push!(summands, pop_name)
    end

    for dt in direct_terms
        _sb_emit_direct!(stmts, data, target, dt, summands)
    end

    _sb_emit_ranefs!(stmts, data, target, ran_terms, summands; id_lookup, brmi_key, cv_groups)

    if length(summands) == 1
        push!(stmts, :($target = $(only(summands))))
    else
        push!(stmts, :($target = $(Expr(:call, :+, summands...))))
    end
end

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

# Free-summand terms (no popefs beta): `mo1(c)`, categorical NamedColumns.
# Categoricals emit `cat_<c> ~ _sb_cat(; x=<c>_idx, n_levels=<c>_n_levels)`.
# `mo1(c)` reuses the `_sb_mo` submodel already used for free-beta `mo(c)`.
_sb_emit_direct!(stmts, data, target::Symbol, t::NamedColumn, summands) =
    _sb_emit_cat!(stmts, data, t, summands)
function _sb_emit_direct!(stmts, data, target::Symbol, t::ExprColumn, summands)
    f = getf(t)
    f === mo1 || error("sbimpl: unsupported direct-summand term `$f`")
    inner_name, raw = _sb_inner_data(:mo1, only(getargs(t)))
    n_levels, idx = _sb_level_index(raw)
    n_levels >= 2 || error("sbimpl: `mo1($inner_name)` needs >= 2 levels (got $n_levels)")
    idx_name = Symbol(inner_name, :_idx)
    col_name = Symbol(:mo1_, inner_name)
    data[idx_name] = idx
    push!(stmts, :($col_name ~ _sb_mo(; x=$idx_name)))
    push!(summands, col_name)
end

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
        (id_sym, lhs, desc)
    elseif length(args) == 3
        lhs, id_sym_raw, raw_group = args
        id_sym = _as_symbol(id_sym_raw)
        isnothing(id_sym) && error("sbimpl: `(e | ID | g)` middle must be a Symbol, got $(typeof(id_sym_raw))")
        desc, gr_id_sym = _sb_normalize_group(raw_group)
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
                           cv_groups=Set{Symbol}())
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
        _sb_emit_ranef_block!(stmts, data, target, desc, gterms, summands; cv_groups)
    end
    for k in id_keys_seen
        gterms = id_terms_by_bucket[k]
        isempty(gterms) && error("sbimpl: ranef `(… | $(k[1]) | $(k[2]))` has no terms after dropping `0`")
        # The `|ID|` bucket path shares one pre-emitted `ranef_correlated_draws`
        # block (prepass 2, before cv_groups is known) -- not yet cv-contagious.
        # Error rather than silently emit an in-sample RE for an opted-in group.
        k[2] in cv_groups && error(
            "sbimpl: cv-contagious sizing requested for group `$(k[2])`, but it ",
            "appears in a `(… |$(k[1])| $(k[2]))` cross-formula ID bucket, whose ",
            "shared draws block is emitted in a prepass and is not yet cv-aware. ",
            "Use a plain `(… | $(k[2]))` ranef for the cv group, or extend the ",
            "bucket emitter -- not yet supported.")
        info = get(id_lookup, (brmi_key, k), nothing)
        info === nothing && error("sbimpl: internal — no pre-emitted bucket for (target=$brmi_key, id=$(k[1]), group=$(k[2]))")
        _sb_emit_id_ranef_block!(stmts, data, target, info, gterms, summands)
    end
end

# Emit a single ranef block for one normalized group descriptor.
# `cv_groups`: groups whose RE should be sized cv-contagiously (opt-in, for CV
# model artifacts) -- emits the `_cv` submodel variants so `maybecv(:<g>_idx)`
# flips the RE to a generated-quantities population re-draw. Empty by default,
# in which case the emitted Stan is byte-identical to before.
function _sb_emit_ranef_block!(stmts, data, target::Symbol, group::NamedColumn, gterms, summands;
                                cv_groups=Set{Symbol}())
    g_backing = _as_data_column(parent(group))
    isnothing(g_backing) && error("sbimpl: group `$(name(group))` must be a raw data column")
    g = name(group)
    is_cv = g in cv_groups
    n_levels, g_idx = _sb_level_index(parent(g_backing))
    idx_name = Symbol(g, :_idx)
    n_name   = Symbol(:n_, g)
    data[idx_name] = g_idx
    data[n_name]   = n_levels
    r_name = Symbol(:r_, target, :_, g)
    if length(gterms) == 1 && gterms[1] === 1
        # (1 | g) fast/equivalent path -- stays on ranef_intercept so the
        # emitted Stan matches existing sb.3 smoke tests bit for bit. The cv
        # variant sizes `xi` from `maximum(group_idx)` and takes no `n_groups`.
        if is_cv
            push!(stmts, :($r_name ~ ranef_intercept_cv(; group_idx=$idx_name)))
        else
            push!(stmts, :($r_name ~ ranef_intercept(; group_idx=$idx_name, n_groups=$n_name)))
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
        if is_cv
            push!(stmts, :($r_name ~ ranef_correlated_cv(;
                Z=$Z_name, group_idx=$idx_name, n_terms=$k_name)))
        else
            push!(stmts, :($r_name ~ ranef_correlated(;
                Z=$Z_name, group_idx=$idx_name,
                n_groups=$n_name, n_terms=$k_name)))
        end
    end
    push!(summands, r_name)
end

function _sb_emit_ranef_block!(stmts, data, target::Symbol, group::Tuple{NamedColumn,NamedColumn}, gterms, summands;
                                cv_groups=Set{Symbol}())
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

# ---- group-block prepass (Prepass 2.5) ---------------------------------------
#
# Scan brmi.operations for `mu ~ f(...)` where f has a _sb_term_group_block
# declaration and mu is a MissingColumn (parameter, not data). Collect them;
# a subsequent emit step allocates one ranef_correlated_draws block per
# (f, group-column) pair and builds a lookup for the emit phase.

function _sb_collect_group_block_terms(brmi::BRMI)
    result = Any[]
    for (key, op_nc) in pairs(brmi.operations)
        op = _as_expr_column(parent(op_nc)); isnothing(op) && continue
        getf(op) === (~) || continue
        lhs, rhs = getargs(op, 2)
        lhs_nc = _as_named_column(lhs); isnothing(lhs_nc) && continue
        isnothing(_as_missing_column(parent(lhs_nc))) && continue
        rhs_e = _as_expr_column(rhs); isnothing(rhs_e) && continue
        f = getf(rhs_e)
        decl = _sb_term_group_block(f)
        isnothing(decl) && continue
        push!(result, (; key, f, rhs_e, decl))
    end
    result
end

# Allocate one ranef_correlated_draws block per (f, group-column) pair and
# return a lookup Dict{(f, group_name) => (; block_name, idx_name, n_per_group)}.
# Idempotent: duplicate (f, group) pairs within the same BRMI are skipped.
function _sb_emit_group_blocks!(stmts, data, gb_terms)
    lookup = Dict{Tuple{Any,Symbol}, NamedTuple}()
    for (; f, rhs_e, decl) in gb_terms
        pos = get(decl, :group_arg_pos, 1)
        args = getargs(rhs_e)
        pos <= length(args) || error(
            "sbimpl: group-block term `$(nameof(f))` has group_arg_pos=$pos ",
            "but only $(length(args)) args")
        group_col = _as_named_column(args[pos])
        isnothing(group_col) && error(
            "sbimpl: group-block term `$(nameof(f))` arg $pos must be a ",
            "NamedColumn group, got $(typeof(args[pos]))")
        gname = name(group_col)
        lk = (f, gname)
        haskey(lookup, lk) && continue
        suffix = Symbol(nameof(f), :_, gname)
        block_name = Symbol(:b_, suffix)
        n_terms_name = Symbol(:n_terms_, suffix)
        data[n_terms_name] = decl.n_per_group
        idx_name, n_name = _sb_ensure_group_data!(data, group_col)
        push!(stmts, :($block_name ~ ranef_correlated_draws(;
            group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name)))
        lookup[lk] = (; block_name, idx_name, n_per_group=decl.n_per_group)
    end
    lookup
end

# Look up block_info for a group-block term call at emit time, or nothing.
# Calls _sb_term_group_block(f) to retrieve group_arg_pos, then matches the
# concrete group-column name against the prepass-built lookup.
function _sb_find_group_block(f, rhs_e, group_block_lookup)
    decl = _sb_term_group_block(f)
    isnothing(decl) && return nothing
    pos = get(decl, :group_arg_pos, 1)
    args = getargs(rhs_e)
    pos > length(args) && return nothing
    group_col = _as_named_column(args[pos])
    isnothing(group_col) && return nothing
    get(group_block_lookup, (f, name(group_col)), nothing)
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
function _sb_emit_id_buckets!(stmts, data, buckets)
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
        idx_name = _sb_emit_id_bucket_sampling!(stmts, data, bucket_name, n_terms_name, desc)
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
# Plain group -> `ranef_correlated_draws`; `gr(g, by=b)` group -> stratified
# `ranef_correlated_by_draws`. Returns the idx_name callers use to slice the
# draw matrix per sub-formula.
function _sb_emit_id_bucket_sampling!(stmts, data, bucket_name, n_terms_name, g::NamedColumn)
    idx_name, n_name = _sb_ensure_group_data!(data, g)
    push!(stmts, :($bucket_name ~ ranef_correlated_draws(;
        group_idx=$idx_name, n_groups=$n_name, n_terms=$n_terms_name)))
    idx_name
end
function _sb_emit_id_bucket_sampling!(stmts, data, bucket_name, n_terms_name,
                                       g::Tuple{NamedColumn,NamedColumn})
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
_sb_pop_cols!(cols, t, data, stmts, pop_terms=(); obs_n=nothing) =
    push!(cols, _sb_predictor_col(t, data, stmts, pop_terms; obs_n))
_sb_pop_cols!(cols, t::ExprColumn, data, stmts, pop_terms=(); obs_n=nothing) =
    _sb_pop_cols_expr!(cols, getf(t), t, data, stmts, pop_terms; obs_n)
_sb_pop_cols_expr!(cols, ::Any, t, data, stmts, pop_terms=(); obs_n=nothing) =
    push!(cols, _sb_predictor_col(t, data, stmts, pop_terms; obs_n))
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
    l = _sb_interaction_operand(args[1])
    r = _sb_interaction_operand(args[2])
    _sb_interaction_expand!(cols, data, l, r)
end

_sb_interaction_operand(t::NamedColumn) = begin
    d_raw = parent(t)
    d = _as_data_column(d_raw)
    isnothing(d) && error(
        "sbimpl: interaction operand `$(name(t))` must be a raw data column, got $(typeof(d_raw))"
    )
    v = parent(d)
    _sb_interaction_operand_kind(t, v, _sb_cat_levels(t))
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
_sb_interaction_operand(t) = error(
    "sbimpl: interaction operand must be a raw-data NamedColumn, got $(typeof(t)); ",
    "interactions with `mo` / `me` / `s` / `ar` are not supported yet"
)

# cont x cont
_sb_interaction_expand!(cols, data, l::NamedTuple{<:Any,<:Tuple}, r::NamedTuple{<:Any,<:Tuple}) = begin
    if l.kind === :cont && r.kind === :cont
        col_name = Symbol(:int_, l.name, :_x_, r.name)
        data[col_name] = l.vec .* r.vec
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
_sb_predictor_col(t::Int, data, _stmts, pop_terms=(); obs_n::Union{Symbol,Nothing}=nothing) = begin
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
_sb_predictor_col(t::ExprColumn, data, stmts, _pop_terms=(); kwargs...) = _sb_predictor_term!(stmts, data, getf(t), t)
_sb_predictor_col(t, _data, _stmts, _pop_terms=(); kwargs...) = error("sbimpl: unsupported predictor term $(typeof(t)): $t")

# Monotonic-effect predictor: emit `mo_<c> ~ _sb_mo(; x=<c>_idx)` and return
# `mo_<c>` as the column. Scope: single NamedColumn inner arg backed by raw
# data; `mo1`, `mm`, `s` etc. fall through the default error.
_sb_predictor_term!(stmts, data, ::typeof(mo), t) = begin
    inner_name, raw = _sb_inner_data(:mo, only(getargs(t)))
    n_levels, idx = _sb_level_index(raw)
    n_levels >= 2 || error("sbimpl: `mo($inner_name)` needs >= 2 levels (got $n_levels)")
    idx_name = Symbol(inner_name, :_idx)
    col_name = Symbol(:mo_, inner_name)
    data[idx_name] = idx
    push!(stmts, :($col_name ~ _sb_mo(; x=$idx_name)))
    col_name
end
# Measurement-error predictor `me(x_obs, sd_x)`: emit a submodel that allocates
# a length-N latent `me_<x>` with prior std_normal and an observation
# likelihood `x_obs ~ normal(me_<x>, sd_x)`. Returns `me_<x>` as the predictor
# column so popefs supplies a free beta. `sd_x` must be a positive constant;
# per-row error sizes would require a vector kwarg and a tweaked submodel.
_sb_predictor_term!(stmts, data, ::typeof(me), t) = begin
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
# Cubic-spline predictor `s(x)`. Precomputes the basis matrix from the raw
# data column and stashes it under `X_basis_<x>`; the submodel `_sb_s` owns
# the coefficient vector + prior. Returns the per-row smooth contribution as
# a single column (so popefs multiplies by an overall beta). Only the default
# basis is supported right now -- `bs`, `t2`, and kwargs like `k=` or
# `knots=` will need extra dispatches.
_sb_predictor_term!(stmts, data, ::typeof(s), t) = begin
    args = getargs(t)
    length(args) == 1 || error("sbimpl: `s(x)` expects 1 positional arg, got $(length(args))")
    xname, raw = _sb_inner_data(:s, only(args))
    v = _sb_real_vec(:s, xname, raw)
    basis, _ = _sb_spline_basis_ncs(v; n_interior=2)
    X_name = Symbol(:X_basis_, xname)
    data[X_name] = basis
    col_name = Symbol(:s_, xname)
    push!(stmts, :($col_name ~ _sb_s(; X_basis=$X_name)))
    col_name
end
# `gp(x; k=K, c=C)` HSGP predictor. Precomputes the Riutort-Mayol eigen-basis
# from the raw data (matrix PHI of size N x K, vector lambda of length K) and
# stashes them in the data dict. The submodel `_sb_hsgp` owns log_rho,
# log_sigma, beta_raw and returns a length-N smooth contribution. popefs
# multiplies by an overall beta -- harmless extra slope but conventional.
# Defaults match vimpl's: K=20 basis fns, c=1.5 boundary factor.
_sb_predictor_term!(stmts, data, ::typeof(gp), t) = begin
    args = getargs(t); kw = getkwargs(t)
    length(args) == 1 || error("sbimpl: `gp(x; k=K, c=C)` expects 1 positional arg, got $(length(args))")
    xname, raw_col = _sb_inner_data(:gp, only(args))
    raw = _sb_real_vec(:gp, xname, raw_col)
    K = get(kw, :k, 20)
    c = get(kw, :c, 1.5)
    PHI, lambda = _hsgp_basis(raw, K, c)
    PHI_name    = Symbol(:PHI_, xname)
    lambda_name = Symbol(:lambda_, xname)
    data[PHI_name]    = PHI
    data[lambda_name] = lambda
    col_name = Symbol(:gp_, xname)
    push!(stmts, :($col_name ~ _sb_hsgp(; PHI=$PHI_name, lambda=$lambda_name)))
    col_name
end

# `_hsgp_basis` is defined in vimpl.jl; reuse rather than redefine here
# to avoid method-overwriting precompile errors.

# `ar(time; p=1)` AR(1) residual submodel. Routes to `_sb_ar1`, which owns the
# phi / epsilon parameters and returns the per-row u[t] as a single length-N
# column. popefs multiplies by an overall beta -- harmless, but a direct-
# summand variant would skip it.
_sb_predictor_term!(stmts, data, ::typeof(ar), t) = begin
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
for (fn, transform) in (
        (:zscale,      :_sb_zscale),
        (:standardize, :_sb_zscale),
        (:center,      :_sb_center),
    )
    @eval function _sb_predictor_term!(stmts, data, ::typeof($fn), t)
        inner = only(getargs(t))
        v = collect(Float64, _sb_materialize_vec(inner))
        v_t = $transform(v)
        cn = _sb_wrapper_col_name(Symbol($(QuoteNode(fn))), inner)
        data[cn] = v_t
        cn
    end
end

# Fallback: a "plain" expression like `log(exposure)` or `a^2` reaching this
# point is treated as an implicit `protect(...)` -- materialize the whole
# subtree to a Stan data vector and let popefs supply the beta. Errors out
# if any leaf isn't a raw data column (e.g. references a sampled parameter
# directly), preserving the old "unsupported" diagnostic.
function _sb_predictor_term!(stmts, data, f::Function, t)
    try
        v = collect(Float64, _sb_materialize_vec(t))
        cn = _sb_wrapper_col_name(Symbol(f), t)
        data[cn] = v
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

_sb_zscale(v::AbstractVector{<:Real}) = let mu = sum(v) / length(v)
    sd = sqrt(sum((x - mu)^2 for x in v) / length(v))
    sd > 0 || error("sbimpl: zscale: zero variance")
    (v .- mu) ./ sd
end
_sb_center(v::AbstractVector{<:Real}) = v .- (sum(v) / length(v))

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
_sb_collect_rhs_refs!(target_obs, x::NamedColumn, obs_name) =
    (get!(target_obs, name(x), obs_name); nothing)
_sb_collect_rhs_refs!(target_obs, x::ExprColumn, obs_name) =
    foreach(a -> _sb_collect_rhs_refs!(target_obs, a, obs_name), getargs(x))
_sb_any_data_symbol(data) = begin
    isempty(data) && error("sbimpl: can't emit `rep_vector(1., n)` — no data column seen yet. Make sure an observed `~` comes before the intercept-only predictor, or add a concrete covariate.")
    # Prefer a flat length-N vector (numeric / integer) so `num_elements(...)` in
    # Stan resolves to an int. Skip ragged `Vector{<:AbstractVector}` layouts
    # (bruno-ext's `dose_times`) which StanBlocks serializes as a
    # `tuple(vector, array[] int)` that Stan's `num_elements` rejects.
    for (k, v) in data
        hit = _flat_vec_key(k, v)
        isnothing(hit) || return hit
    end
    first(keys(data))
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
    _sb_lik_family!(stmts, target, f, getargs(rhs), data)
end
_sb_likelihood!(stmts, target, rhs, _) =
    error("sbimpl: likelihood RHS for `$target` must be an ExprColumn (got $(typeof(rhs)))")

# One dispatch per likelihood family. Each method states the Stan name and
# implicitly the arity (by destructuring `args`). Caveat on NegativeBinomial:
# Distributions.jl parameterizes by (r, p); Stan's neg_binomial is (alpha, beta)
# — the emitted model is NOT posterior-identical to the Julia side. Convert
# upstream if that matters.
_sb_lik_stan!(stmts, target, name::Symbol, args, data) =
    push!(stmts, Expr(:call, :~, target,
        Expr(:call, name, (_sb_scalar_expr(a, data) for a in args)...)))

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
_sb_stan_dist_name(::Type{<:Weibull})             = :weibull
_sb_stan_dist_name(::Type{<:InverseGamma})        = :inv_gamma
_sb_stan_dist_name(::Type{<:Bernoulli})           = :bernoulli
_sb_stan_dist_name(::Type{<:BernoulliLogit})      = :bernoulli_logit
_sb_stan_dist_name(::Type{<:Binomial})            = :binomial
_sb_stan_dist_name(::Type{<:BinomialLogit})       = :binomial_logit
_sb_stan_dist_name(::Type{<:Poisson})             = :poisson
_sb_stan_dist_name(::Type{<:ZeroInflatedPoisson}) = :zero_inflated_poisson
_sb_stan_dist_name(::Type{<:NegativeBinomial})    = :neg_binomial
_sb_stan_dist_name(::Type) = nothing

# Per-family arg shape adjustment between Julia call args and Stan call
# args. Default: pass through unchanged. Override when Julia's
# constructor signature differs from Stan's distribution signature.
_sb_stan_dist_args(::Type, args) = args
# `TDist(nu)` -> Stan `student_t(nu, 0, 1)`. Pads location/scale defaults.
_sb_stan_dist_args(::Type{<:TDist}, args::Tuple{Any}) = (args[1], 0, 1)

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
        _sb_lik_stan!(stmts, target, stan_name, _sb_stan_dist_args(D, args), data)
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
    TruncatedNormal

BRM formula marker for the bordet censored-normal observation model.
`log_obs ~ TruncatedNormal(log_y, sigma, lloq, uloq)` — `sigma` is a
biomarker-level parameter declared inside `bordet_hierarchical_parametric`;
`lloq`/`uloq` are biomarker-level data columns.
Emits: `log_obs ~ truncated_normal(log_y, sigma[biomarker_idxs], lloq[biomarker_idxs], uloq[biomarker_idxs])`.
Censoring math lives in StanBlocks:bordet's lpxf triad (not Stan T[,] truncation).
"""
function TruncatedNormal end

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

# bordet_hierarchical_parametric submodel RHS.
# All kwargs are data/hyperparameter columns; collected here because the generic
# prepass only walks positional args (ExprColumn kwargs fix is a targeted local
# collect, plus the global fix above for future terms).
function _sb_submodel_rhs!(stmts, data, target,
                           ::typeof(bordet_hierarchical_parametric), rhs)
    kw = getkwargs(rhs)
    for v in values(kw)
        _sb_collect_data!(data, v)
    end
    # Sizes in transformed_data (deterministic from int[] index arrays)
    push!(stmts, :(n_biomarkers = max(biomarker_idxs)))
    push!(stmts, :(n_persons    = max(person_idxs)))
    push!(stmts, :(n_series     = n_biomarkers * n_persons))
    # G3: sigma is biomarker-level — bypasses _sb_emit_prior! which emits scalar only
    push!(stmts, :(sigma ~ exponential(sigma_rate; n=n_biomarkers, lower=0.)))
    # Derived transdata: obs-level index + input transforms
    push!(stmts, :(series_idxs = linear_idxs(biomarker_idxs, person_idxs)))
    push!(stmts, :(log_time    = log(broadcasted_max(time, 0.001))))
    push!(stmts, :(log_dose    = log(broadcasted_max(dose, 0.1))))
    push!(stmts, :(affectable  = broadcasted_gt(time .* dose, 0.0)))
    # 6 per-(biomarker×person) parameter priors.
    # TODO(bordet-floor): swap std_normal stubs for adaptive hierarchical prior
    # (pending supervisor clarification on adaptive_vector_hierarchical builtin /
    # group-block floor pattern for 2-level biomarker×person hierarchy).
    push!(stmts, :(baseline       ~ std_normal(; n=n_series)))
    push!(stmts, :(time_loc       ~ std_normal(; n=n_series)))
    push!(stmts, :(time_log_slope ~ std_normal(; n=n_series)))
    push!(stmts, :(time_mag       ~ std_normal(; n=n_series)))
    push!(stmts, :(dose_loc       ~ std_normal(; n=n_series)))
    push!(stmts, :(dose_log_slope ~ std_normal(; n=n_series)))
    # log_y via StanBlocks:bordet kernel builtins (decomposed into intermediates)
    push!(stmts, :(time_response = bordet_time_response(
        log_time,
        time_loc[series_idxs],
        time_log_slope[series_idxs],
        time_mag[series_idxs])))
    push!(stmts, :(dose_response = exp(bordet_dose_response(
        log_dose,
        dose_loc[series_idxs],
        dose_log_slope[series_idxs]))))
    push!(stmts, :($target = baseline[series_idxs]
        + affectable .* time_response .* dose_response))
    :emitted
end
