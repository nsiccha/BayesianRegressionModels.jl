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
function me end
function s end
function ar end

# Local marker type for cumulative-link ordinal likelihoods. Mirrors the
# `Normal` / `BernoulliLogit` pattern (formula surface uses the type as a
# likelihood family). Not a real `Distributions.Distribution` -- Distributions.jl
# does not ship `OrderedLogistic`, and the @brm `_x` parser never actually
# calls the constructor, so a bare marker struct is enough. If Distributions
# later adds its own `OrderedLogistic`, drop this in favour of that.
struct OrderedLogistic end

const popefs = StanBlocks.@slic begin
    n_covariates = dims(X)[2]
    beta_pop ~ std_normal(; n=n_covariates)
    return X * beta_pop
end

const cdirichlet = StanBlocks.@slic begin
    increments ~ dirichlet(alpha)
    return cumulative_sum(increments)
end

const c0dirichlet = StanBlocks.@slic begin
    increments ~ dirichlet(alpha)
    return cumulative_sum(increments) - increments[1]
end

const c01dirichlet = StanBlocks.@slic begin
    increments ~ dirichlet(alpha)
    return append_row(0., cumulative_sum(increments))
end

# Monotonic effect contrast (Buerkner & Charpentier 2018). Returns the per-obs
# contrast vector; the walker hcat's it as one column of X_pop so popefs
# supplies the free beta (matches vimpl's free-beta `mo` variant, not `mo1`).
# Named `_sb_mo` to avoid clashing with vimpl's marker function `mo`.
const _sb_mo = StanBlocks.@slic begin
    n_levels = maximum(x)
    simplex_incr ~ dirichlet(rep_vector(1., n_levels - 1))
    return cumulative_sum(append_row(0., simplex_incr))[x]
end

# (1 | g) random intercept. Mirrors vimpl's scalar grouped_normal + chol(n=1)
# collapse: Part{chol}(1x1) -> log_scale ~ N(0,1), L[1,1] = exp(log_scale);
# Part{grouped_normal}(n_groups, 1) -> xi ~ N(0,1), values = L[1,1] * xi;
# per-obs contribution is values[group_idx]. No LKJ needed at n=1.
const ranef_intercept = StanBlocks.@slic begin
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
const ranef_correlated = StanBlocks.@slic begin
    L      ~ lkj_corr_cholesky(1.; n=n_terms)
    tau    ~ std_normal(; n=n_terms, lower=0.)
    z_flat ~ std_normal(; n=n_terms * n_groups)
    z = reshape(z_flat, n_terms, n_groups)
    b = (diag_pre_multiply(tau, L) * z)'   # n_groups x n_terms
    return rows_dot_product(Z, b[group_idx, :])
end

# `(expr | gr(g, by=b))` stratified random effects: independent LKJ-Cholesky +
# tau per level of `b`, so each stratum has its own full covariance structure.
# `stratum_idx[g]` maps each group-level to its stratum (walker pre-computes it
# and errors if any group straddles strata).
#   L   :: array[n_strata] cholesky_factor_corr[n_terms]
#   tau :: array[n_strata] vector<lower=0>[n_terms]
#   z   :: array[n_groups] vector[n_terms]
# Per-group contribution: b[g, :] = (diag_pre_multiply(tau[s], L[s]) * z[g])'
# where s = stratum_idx[g]. Uses SB's `lkj_corr_cholesky_lpdfs` array-broadcast
# (one-liner added to StanBlocks.jl builtins) so L's sampling statement
# vectorizes across strata.
const ranef_correlated_by = StanBlocks.@slic begin
    L   ~ lkj_corr_cholesky(1.; n=n_terms, m=n_strata)
    tau ~ std_normal(; n=n_terms, m=n_strata, lower=0., type=vector)
    z   ~ std_normal(; n=n_terms, m=n_groups, type=vector)
    b = rep_matrix(0., n_groups, n_terms)
    for g in 1:n_groups
        b[g, :] = (diag_pre_multiply(tau[stratum_idx[g]], L[stratum_idx[g]]) * z[g])'
    end
    return rows_dot_product(Z, b[group_idx, :])
end

# Treatment-coded categorical predictor. Allocates K-1 free betas; reference
# level 1 contributes 0. Mirrors vimpl's `AbstractVector{<:Integer}` dispatch.
# `x` is the per-row 1-based level index, `n_levels = K`.
const _sb_cat = StanBlocks.@slic begin
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

const _sb_ar1 = StanBlocks.@slic begin
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
const _sb_s = StanBlocks.@slic begin
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
const _sb_me = StanBlocks.@slic begin
    x_true ~ std_normal(; n=num_elements(x_obs))
    x_obs ~ normal(x_true, sd_x)
    return x_true
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

struct SBBRMI{P<:BRMI, M, D<:AbstractDict}
    parent::P
    model::M
    data::D
end

SBBRMI(brmi::BRMI; mod::Module=@__MODULE__) = begin
    stmts = Any[]
    data = Dict{Symbol,Any}()
    # Prepass: stash every data-backed NamedColumn so later intercept-only
    # predictors have a length probe to hang `rep_vector(1., num_elements(...))`
    # off, regardless of iteration order.
    for (_, op) in pairs(brmi.operations)
        _sb_collect_data!(data, op)
    end
    for (key, op) in pairs(brmi.operations)
        op isa NamedColumn || error("sbimpl: top-level op `$key` is not a NamedColumn")
        _sb_emit!(stmts, data, key, parent(op))
    end
    body = Expr(:block, stmts...)
    model = StanBlocks.SlicModel(body, data, mod)
    SBBRMI(brmi, model, data)
end

_sb_collect_data!(data, x) = nothing
_sb_collect_data!(data, x::NamedColumn) = begin
    d = parent(x)
    d isa DataColumn && (data[name(x)] = parent(d))
    _sb_collect_data!(data, d)
end
_sb_collect_data!(data, x::ExprColumn) = foreach(a -> _sb_collect_data!(data, a), getargs(x))

stan_code(sb::SBBRMI) = StanBlocks.stan_code(sb.model)

Base.show(io::IO, sb::SBBRMI) = begin
    print(io, "SBBRMI with data keys = ", sort(collect(keys(sb.data))), "\n")
    print(io, "emitted @slic body:\n")
    print(io, sb.model.model)
end


# ---- top-level op dispatch ---------------------------------------------------

_sb_emit!(stmts, data, key, op::ExprColumn) = _sb_emit_expr!(stmts, data, key, getf(op), op)
# Raw data / missing columns appear as top-level ops when the formula mentions
# them as bare references (e.g. `c2` in `loc ~ 1 + c2`). Nothing to emit — the
# prepass already stashed data columns in `data`.
_sb_emit!(_, _, _, ::DataColumn) = nothing
_sb_emit!(_, _, _, ::MissingColumn) = nothing
_sb_emit!(_, _, key, op) = error("sbimpl: top-level op for `$key` not an ExprColumn (got $(typeof(op)))")

_sb_emit_expr!(stmts, data, key, ::typeof(~), op) = begin
    lhs, rhs = getargs(op, 2)
    _sb_sampling!(stmts, data, key, lhs, rhs)
end
_sb_emit_expr!(stmts, data, key, ::typeof(assign), op) = begin
    _, rhs = getargs(op, 2)
    target_expr = _sb_scalar_expr(rhs, data)
    push!(stmts, :($key = $target_expr))
end
_sb_emit_expr!(_, _, key, f, _) = error("sbimpl: unsupported top-level op `$f` for `$key`")


# ---- sampling: likelihood vs linear-predictor split --------------------------

# Extension hook. Overridden (e.g. in bruno-ext.jl) to route `target ~ f(...)`
# where `f` is a known submodel family (`logistic_dr`, `gamma_time`, etc.)
# straight to `target ~ <slic>(; kwargs)`, bypassing the pop-linear-predictor
# wrap (which would wrongly multiply the submodel output by a beta). Return
# anything non-`nothing` to claim the binding; return `nothing` to fall
# through. Default: no hook registered.
_sb_submodel_rhs!(stmts, data, target, f, rhs) = nothing

# LHS backed by real data => this is a likelihood. Record the observed values
# under the formula name in `data` and emit `key ~ dist(args...)`.
_sb_sampling!(stmts, data, key, lhs::NamedColumn, rhs) = begin
    backing = parent(lhs)
    if backing isa DataColumn
        data[key] = parent(backing)
        push!(stmts, _sb_likelihood(key, rhs, data))
    elseif backing isa MissingColumn
        if rhs isa ExprColumn &&
           _sb_submodel_rhs!(stmts, data, key, getf(rhs), rhs) !== nothing
            return
        end
        _sb_linear_predictor!(stmts, data, key, rhs)
    else
        error("sbimpl: unsupported LHS backing for `$key` ($(typeof(backing)))")
    end
end

# Link-transformed LHS: `log(err) ~ 1 + d`, `logit(p) ~ 1 + x`, etc.
# Sample the linear predictor on the linked scale, then invert to recover the
# response. Mirrors vimpl's `inverse(getf(lhs))` path — any link whose Julia
# `inverse` is a function with a Stan-known name (log/exp/logit/logistic/
# sqrt/square, ...) works; unknown links error at transpile time.
_sb_sampling!(stmts, data, key, lhs::ExprColumn, rhs) = begin
    f = getf(lhs)
    inner = only(getargs(lhs))
    inner isa NamedColumn || error("sbimpl: expected NamedColumn inside link, got $(typeof(inner))")
    inv_f = InverseFunctions.inverse(f)
    inner_name = name(inner)
    pre_name = Symbol(nameof(f), :_, inner_name)
    _sb_linear_predictor!(stmts, data, pre_name, rhs)
    push!(stmts, :($inner_name = $(Symbol(nameof(inv_f)))($pre_name)))
end


# ---- linear predictor: emit `X_<name> = hcat(...); <name> ~ popefs(; X=X_<name>)` --

function _sb_linear_predictor!(stmts, data, target::Symbol, rhs)
    terms = _sb_terms(rhs)
    pop_terms    = Any[]
    ran_terms    = Any[]  # `(expr | group)` -> collected per-group below
    direct_terms = Any[]  # e.g. `mo1(c)` -> direct summand, no popefs beta
    for t in terms
        if t isa ExprColumn && getf(t) === (|)
            push!(ran_terms, t)
        elseif t isa ExprColumn && getf(t) === mo1
            push!(direct_terms, t)
        elseif _sb_is_categorical(t)
            push!(direct_terms, t)
        else
            push!(pop_terms, t)
        end
    end
    isempty(pop_terms) && isempty(ran_terms) && isempty(direct_terms) &&
        error("sbimpl: empty RHS for `$target` — no predictor terms")

    summands = Symbol[]

    if !isempty(pop_terms)
        col_exprs = Any[_sb_predictor_col(t, data, stmts) for t in pop_terms]
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

    _sb_emit_ranefs!(stmts, data, target, ran_terms, summands)

    if length(summands) == 1
        push!(stmts, :($target = $(only(summands))))
    else
        push!(stmts, :($target = $(Expr(:call, :+, summands...))))
    end
end

# Classify a predictor term as "direct" (allocates its own parameters, no
# popefs multiplication). Matches vimpl: integer-backed NamedColumns are
# treated as treatment-coded categoricals; floats go through popefs.
_sb_is_categorical(t::NamedColumn) = begin
    d = parent(t)
    d isa DataColumn || return false
    v = parent(d)
    v isa AbstractVector{<:Integer} || v isa CA.CategoricalVector
end
_sb_is_categorical(_) = false

# Free-summand terms (no popefs beta): `mo1(c)`, categorical NamedColumns.
# Categoricals emit `cat_<c> ~ _sb_cat(; x=<c>_idx, n_levels=<c>_n_levels)`.
# `mo1(c)` reuses the `_sb_mo` submodel already used for free-beta `mo(c)`.
function _sb_emit_direct!(stmts, data, target::Symbol, t, summands)
    if t isa NamedColumn
        _sb_emit_cat!(stmts, data, t, summands)
        return
    end
    f = getf(t)
    f === mo1 || error("sbimpl: unsupported direct-summand term `$f`")
    inner = only(getargs(t))
    inner isa NamedColumn || error("sbimpl: `mo1(…)` expects a NamedColumn, got $(typeof(inner))")
    backing = parent(inner)
    backing isa DataColumn || error("sbimpl: `mo1($(name(inner)))` expects a raw data column, got $(typeof(backing))")
    n_levels, idx = _sb_level_index(parent(backing))
    n_levels >= 2 || error("sbimpl: `mo1($(name(inner)))` needs >= 2 levels (got $n_levels)")
    idx_name = Symbol(name(inner), :_idx)
    col_name = Symbol(:mo1_, name(inner))
    data[idx_name] = idx
    push!(stmts, :($col_name ~ _sb_mo(; x=$idx_name)))
    push!(summands, col_name)
end

# Categorical population-level predictor. Allocates K-1 betas via `_sb_cat`
# and pushes the per-row contribution column into `summands`.
function _sb_emit_cat!(stmts, data, t::NamedColumn, summands)
    backing = parent(t)
    n_levels, idx = _sb_level_index(parent(backing))
    n_levels >= 2 || error("sbimpl: categorical `$(name(t))` needs >= 2 levels (got $n_levels)")
    idx_name = Symbol(name(t), :_idx)
    n_name   = Symbol(name(t), :_n_levels)
    col_name = Symbol(:cat_, name(t))
    data[idx_name] = idx
    data[n_name]   = n_levels
    push!(stmts, :($col_name ~ _sb_cat(; x=$idx_name, n_levels=$n_name)))
    push!(summands, col_name)
end

# Expand a ranef LHS term into one-or-more design-matrix column references.
# Continuous/intercept terms produce a single column; categorical NamedColumns
# expand to K-1 treatment-coded dummy columns (level 1 is reference), matching
# the design-matrix that brms / lme4 build for `(1 + c | g)`.
function _sb_ranef_cols!(cols, data, stmts, t)
    if _sb_is_categorical(t)
        backing = parent(t)
        n_levels, idx = _sb_level_index(parent(backing))
        n_levels >= 2 || error("sbimpl: categorical ranef term `$(name(t))` needs >= 2 levels (got $n_levels)")
        for lvl in 2:n_levels
            col_name = Symbol(name(t), :_dummy_, lvl)
            data[col_name] = Float64[l == lvl ? 1.0 : 0.0 for l in idx]
            push!(cols, col_name)
        end
    else
        push!(cols, _sb_predictor_col(t, data, stmts))
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
# `ranef_correlated_by` block. Mirrors vimpl's `_normalize_group`.
_sb_normalize_group(g::NamedColumn) = g
_sb_normalize_group(g::ExprColumn) = begin
    getf(g) === gr || error("sbimpl: expected NamedColumn or `gr(...)` on RHS of `|`, got `$(getf(g))`")
    args = getargs(g); kw = getkwargs(g)
    length(args) == 1 || error("sbimpl: `gr(...)` expects exactly one positional group, got $(length(args))")
    group = args[1]
    group isa NamedColumn || error("sbimpl: `gr(...)` expects a NamedColumn group, got $(typeof(group))")
    by = get(kw, :by, nothing)
    by === nothing && return group
    by isa NamedColumn || error("sbimpl: `gr(...; by=...)` expects a NamedColumn for `by`, got $(typeof(by))")
    (group, by)
end
_sb_normalize_group(g) = error("sbimpl: expected NamedColumn or `gr(...)` on RHS of `|`, got $(typeof(g))")

# Walker-side key used to coalesce ran_terms. Plain group -> `Symbol`; stratified
# `gr(g, by=b)` -> `(Symbol, Symbol)` so the two don't accidentally merge.
_sb_group_key(g::NamedColumn) = name(g)
_sb_group_key(g::Tuple{NamedColumn,NamedColumn}) = (name(g[1]), name(g[2]))

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

function _sb_emit_ranefs!(stmts, data, target::Symbol, ran_terms, summands)
    isempty(ran_terms) && return
    # Group `(expr | g)` / `(expr | gr(g, by=b))` terms by normalized group key,
    # preserving first-seen order. Bare-NamedColumn and gr-by groups key
    # differently so they never coalesce.
    keys_seen = Any[]
    by_group = Dict{Any, Vector{Any}}()
    descs = Dict{Any, Any}()  # key -> normalized group descriptor
    for rt in ran_terms
        lhs, raw_group = getargs(rt, 2)
        g = _sb_normalize_group(raw_group)
        k = _sb_group_key(g)
        haskey(by_group, k) || (push!(keys_seen, k); by_group[k] = Any[]; descs[k] = g)
        # Flatten `lhs` on `+` (same walker as pop terms; reuses `0` drop).
        append!(by_group[k], _sb_terms(lhs))
    end
    for k in keys_seen
        gterms = by_group[k]
        desc = descs[k]
        isempty(gterms) && error("sbimpl: ranef `(… | $k)` has no terms after dropping `0`")
        _sb_emit_ranef_block!(stmts, data, target, desc, gterms, summands)
    end
end

# Emit a single ranef block for one normalized group descriptor.
function _sb_emit_ranef_block!(stmts, data, target::Symbol, group::NamedColumn, gterms, summands)
    g_backing = parent(group)
    g_backing isa DataColumn || error("sbimpl: group `$(name(group))` must be a raw data column")
    g = name(group)
    n_levels, g_idx = _sb_level_index(parent(g_backing))
    idx_name = Symbol(g, :_idx)
    n_name   = Symbol(:n_, g)
    data[idx_name] = g_idx
    data[n_name]   = n_levels
    r_name = Symbol(:r_, target, :_, g)
    if length(gterms) == 1 && gterms[1] === 1
        # (1 | g) fast/equivalent path -- stays on ranef_intercept so the
        # emitted Stan matches existing sb.3 smoke tests bit for bit.
        push!(stmts, :($r_name ~ ranef_intercept(; group_idx=$idx_name, n_groups=$n_name)))
    else
        col_exprs = Any[]
        for t in gterms
            _sb_ranef_cols!(col_exprs, data, stmts, t)
        end
        Z_name = Symbol(:Z_, target, :_, g)
        k_name = Symbol(:n_terms_, target, :_, g)
        data[k_name] = length(col_exprs)
        push!(stmts, :($Z_name = $(Expr(:call, :hcat, col_exprs...))))
        push!(stmts, :($r_name ~ ranef_correlated(;
            Z=$Z_name, group_idx=$idx_name,
            n_groups=$n_name, n_terms=$k_name)))
    end
    push!(summands, r_name)
end

function _sb_emit_ranef_block!(stmts, data, target::Symbol, group::Tuple{NamedColumn,NamedColumn}, gterms, summands)
    gcol, bcol = group
    g_backing = parent(gcol)
    b_backing = parent(bcol)
    g_backing isa DataColumn || error("sbimpl: group `$(name(gcol))` must be a raw data column")
    b_backing isa DataColumn || error("sbimpl: `by=$(name(bcol))` must be a raw data column")
    g, b = name(gcol), name(bcol)
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
        _sb_ranef_cols!(col_exprs, data, stmts, t)
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
_sb_collect_terms_expr!(acc, _, x) = push!(acc, x)

# Predictor column emitter. `stmts` is threaded in so terms that need their own
# `~` statement (e.g. `mo(c)`) can push before returning their column symbol.
# Integer `1` -> intercept, NamedColumn -> reference by name, ExprColumn(mo, c)
# -> submodel-sampled contrast column.
_sb_predictor_col(t::Int, data, _stmts) = begin
    t == 1 || error("sbimpl: integer term must be `1` for intercept, got `$t`")
    # Use any data column to get n
    probe = _sb_any_data_symbol(data)
    :(rep_vector(1., num_elements($probe)))
end
_sb_predictor_col(t::NamedColumn, data, _stmts) = begin
    d = parent(t)
    # If the named column was already bound earlier in the walker (e.g.
    # `ftime ~ gamma_time(...)` emitted a `ftime ~ _sb_gamma_time(...)` stmt),
    # its parent is the sampling ExprColumn rather than a raw data column --
    # just reference the Stan variable by name.
    d isa ExprColumn && return name(t)
    d isa DataColumn || error("sbimpl: expected data-backed NamedColumn for `$(name(t))`, got $(typeof(d))")
    v = parent(d)
    v isa AbstractVector{<:Real} ||
        error("sbimpl: non-numeric predictor `$(name(t))` not supported yet (wrap in `categorical(…)` once we add it)")
    data[name(t)] = collect(Float64, v)
    name(t)
end
_sb_predictor_col(t::ExprColumn, data, stmts) = _sb_predictor_term!(stmts, data, getf(t), t)
_sb_predictor_col(t, _, _) = error("sbimpl: unsupported predictor term $(typeof(t)): $t")

# Monotonic-effect predictor: emit `mo_<c> ~ _sb_mo(; x=<c>_idx)` and return
# `mo_<c>` as the column. Scope: single NamedColumn inner arg backed by raw
# data; `mo1`, `mm`, `s` etc. fall through the default error.
_sb_predictor_term!(stmts, data, ::typeof(mo), t) = begin
    inner = only(getargs(t))
    inner isa NamedColumn || error("sbimpl: `mo(…)` expects a NamedColumn, got $(typeof(inner))")
    backing = parent(inner)
    backing isa DataColumn || error("sbimpl: `mo($(name(inner)))` expects a raw data column, got $(typeof(backing))")
    n_levels, idx = _sb_level_index(parent(backing))
    n_levels >= 2 || error("sbimpl: `mo($(name(inner)))` needs >= 2 levels (got $n_levels)")
    idx_name = Symbol(name(inner), :_idx)
    col_name = Symbol(:mo_, name(inner))
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
    inner isa NamedColumn || error("sbimpl: `me(...)` expects a NamedColumn first arg, got $(typeof(inner))")
    backing = parent(inner)
    backing isa DataColumn || error("sbimpl: `me($(name(inner)), ...)` expects a raw data column, got $(typeof(backing))")
    v = parent(backing)
    v isa AbstractVector{<:Real} || error("sbimpl: `me($(name(inner)), ...)` expects a numeric data column, got $(typeof(v))")
    sd_arg isa Real || error("sbimpl: `me(x, sd)` expects a numeric constant `sd`, got $(typeof(sd_arg))")
    sd_arg > 0 || error("sbimpl: `me(x, sd)` expects sd > 0 (got $sd_arg)")
    xname = name(inner)
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
    inner = only(args)
    inner isa NamedColumn || error("sbimpl: `s(...)` expects a NamedColumn, got $(typeof(inner))")
    backing = parent(inner)
    backing isa DataColumn || error("sbimpl: `s($(name(inner)))` expects a raw data column, got $(typeof(backing))")
    v = parent(backing)
    v isa AbstractVector{<:Real} || error("sbimpl: `s($(name(inner)))` expects numeric data, got $(typeof(v))")
    basis, _ = _sb_spline_basis_ncs(v; n_interior=2)
    xname = name(inner)
    X_name = Symbol(:X_basis_, xname)
    data[X_name] = basis
    col_name = Symbol(:s_, xname)
    push!(stmts, :($col_name ~ _sb_s(; X_basis=$X_name)))
    col_name
end
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
    time = only(args)
    time isa NamedColumn || error("sbimpl: `ar(time)` expects a NamedColumn, got $(typeof(time))")
    backing = parent(time)
    backing isa DataColumn || error("sbimpl: `ar($(name(time)))` expects a raw data column, got $(typeof(backing))")
    xname = name(time)
    # Ensure the time column lands in `data`. The prepass already handles this
    # for named data columns, but be defensive -- the submodel uses it as a
    # length probe via `num_elements(time)`.
    data[xname] = collect(Float64, parent(backing))
    col_name = Symbol(:ar_, xname)
    push!(stmts, :($col_name ~ _sb_ar1(; time=$xname)))
    col_name
end
_sb_predictor_term!(_, _, f, _) =
    error("sbimpl: unsupported predictor-term function `$f` (supported: `mo`, `mo1`, `me`, `s`, `ar`)")

_sb_n_obs_probe(terms) = begin
    for t in terms
        t isa NamedColumn && parent(t) isa DataColumn && return name(t)
    end
    nothing
end
_sb_any_data_symbol(data) = begin
    isempty(data) && error("sbimpl: can't emit `rep_vector(1., n)` — no data column seen yet. Make sure an observed `~` comes before the intercept-only predictor, or add a concrete covariate.")
    # Prefer a flat length-N vector (numeric / integer) so `num_elements(...)` in
    # Stan resolves to an int. Skip ragged `Vector{<:AbstractVector}` layouts
    # (bruno-ext's `dose_times`) which StanBlocks serializes as a
    # `tuple(vector, array[] int)` that Stan's `num_elements` rejects.
    for (k, v) in data
        v isa AbstractVector && !(eltype(v) <: AbstractVector) && return k
    end
    first(keys(data))
end


# ---- likelihood emitters: `y ~ Normal(loc, sigma)` etc. ----------------------

function _sb_likelihood(target::Symbol, rhs::ExprColumn, data)
    f = getf(rhs)
    _sb_lik_family(target, f, getargs(rhs), data)
end
_sb_likelihood(target, rhs, _) =
    error("sbimpl: likelihood RHS for `$target` must be an ExprColumn (got $(typeof(rhs)))")

# One dispatch per likelihood family. Each method states the Stan name and
# implicitly the arity (by destructuring `args`). Caveat on NegativeBinomial:
# Distributions.jl parameterizes by (r, p); Stan's neg_binomial is (alpha, beta)
# — the emitted model is NOT posterior-identical to the Julia side. Convert
# upstream if that matters.
_sb_lik_stan(target, name::Symbol, args, data) =
    Expr(:call, :~, target,
        Expr(:call, name, (_sb_scalar_expr(a, data) for a in args)...))

# `y ~ OrderedLogistic(eta)`: cumulative-link ordinal likelihood. Blocked on
# StanBlocks support for Stan's `ordered_logistic_lpmf`. Two concrete gaps:
#   1. `ordered_logistic_lpmf` is not in StanBlocks' @builtin_module list, so
#      `y ~ ordered_logistic(eta, c)` errors at symbol resolution.
#   2. `type=ordered` (for declaring an ordered-vector parameter inside a
#      `std_normal(; n, type=ordered)` call) is not surfaced by @slic --
#      symbol resolution for `ordered` fails in the caller's module lookup.
# Workaround attempts that failed: @deffun wrapping + manual `lpxf_expr`
# registration (the inner call to `ordered_logistic_lpmf` still hits (1));
# @defsig-declared external signature (generated code references
# StanBlocks-internal `stan_size`, not reachable from outside the module).
# Leaving the marker struct and MWE in place so the card is ready to light
# up once StanBlocks lands the builtin.
_sb_lik_family(target, ::Type{<:OrderedLogistic},  args, data) =
    error("sbimpl: `OrderedLogistic` likelihood is blocked on StanBlocks adding `ordered_logistic_lpmf` to its builtin distributions; see comment in sbimpl.jl")
_sb_lik_family(target, ::Type{<:Normal},           args::Tuple{Any,Any}, data) = _sb_lik_stan(target, :normal,       args, data)
_sb_lik_family(target, ::Type{<:Bernoulli},        args::Tuple{Any},     data) = _sb_lik_stan(target, :bernoulli,    args, data)
_sb_lik_family(target, ::Type{<:BernoulliLogit},   args::Tuple{Any},     data) = _sb_lik_stan(target, :bernoulli_logit, args, data)
_sb_lik_family(target, ::Type{<:Binomial},         args::Tuple{Any,Any}, data) = _sb_lik_stan(target, :binomial,     args, data)
_sb_lik_family(target, ::Type{<:Poisson},          args::Tuple{Any},     data) = _sb_lik_stan(target, :poisson,      args, data)
_sb_lik_family(target, ::Type{<:Gamma},            args::Tuple{Any,Any}, data) = _sb_lik_stan(target, :gamma,        args, data)
_sb_lik_family(target, ::Type{<:NegativeBinomial}, args::Tuple{Any,Any}, data) = _sb_lik_stan(target, :neg_binomial, args, data)
_sb_lik_family(target, ::Type{<:Beta},             args::Tuple{Any,Any}, data) = _sb_lik_stan(target, :beta,         args, data)

_sb_lik_family(target, fam, args, _) =
    error("sbimpl: likelihood family `$fam` (arity $(length(args))) not supported yet")


# ---- scalar-expression reducer (unwraps NamedColumn references etc.) --------

_sb_scalar_expr(x::Symbol, _) = x
_sb_scalar_expr(x::Real, _) = x
_sb_scalar_expr(x::NamedColumn, data) = begin
    d = parent(x)
    if d isa DataColumn
        data[name(x)] = parent(d)
    end
    name(x)
end
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
