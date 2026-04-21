using StanBlocks


# ==============================================================================
# SlicModel helpers (ported verbatim from /home/niko/github/nsiccha/bruno/src/qt.jl,
# `popefs`/`ranefs`/`popranefs`/`cdirichlet` family, lines 303-344). Kept here
# as module-local bindings so the walker can emit calls to them by name without
# depending on bruno. Duplication is intentional for now.
# ==============================================================================

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

# Treatment-coded categorical predictor. Allocates K-1 free betas; reference
# level 1 contributes 0. Mirrors vimpl's `AbstractVector{<:Integer}` dispatch.
# `x` is the per-row 1-based level index, `n_levels = K`.
const _sb_cat = StanBlocks.@slic begin
    beta ~ std_normal(; n=n_levels - 1)
    return append_row(0., beta)[x]
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

SBBRMI(brmi::BRMI) = begin
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
    model = StanBlocks.SlicModel(body, data, @__MODULE__)
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

# Link-transformed LHS: `log(err) ~ 1 + d` etc.
_sb_sampling!(stmts, data, key, lhs::ExprColumn, rhs) = begin
    f = getf(lhs)
    f === log || error("sbimpl: only `log(x)` links supported for now (got `$f`)")
    inner = getargs(lhs, 1)[1]
    inner isa NamedColumn || error("sbimpl: expected NamedColumn inside link, got $(typeof(inner))")
    inner_name = name(inner)
    log_name = Symbol(:log_, inner_name)
    _sb_linear_predictor!(stmts, data, log_name, rhs)
    push!(stmts, :($inner_name = exp($log_name)))
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
function _sb_emit_ranefs!(stmts, data, target::Symbol, ran_terms, summands)
    isempty(ran_terms) && return
    # Group `(expr | g)` terms by the group symbol, preserving first-seen order.
    groups = Symbol[]
    by_group = Dict{Symbol, Vector{Any}}()
    for rt in ran_terms
        lhs, group = getargs(rt, 2)
        group isa NamedColumn || error("sbimpl: expected NamedColumn on RHS of `|`, got $(typeof(group))")
        g_backing = parent(group)
        g_backing isa DataColumn || error("sbimpl: group `$(name(group))` must be a raw data column")
        g = name(group)
        haskey(by_group, g) || (push!(groups, g); by_group[g] = Any[])
        # Flatten `lhs` on `+` (same walker as pop terms; reuses `0` drop).
        append!(by_group[g], _sb_terms(lhs))
    end
    for g in groups
        gterms = by_group[g]
        isempty(gterms) && error("sbimpl: ranef `(… | $g)` has no terms after dropping `0`")
        # Resolve group index/count from the first occurrence (same backing data).
        example = first(rt for rt in ran_terms if name(getargs(rt, 2)[2]) === g)
        group_col = getargs(example, 2)[2]
        n_levels, g_idx = _sb_level_index(parent(parent(group_col)))
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
_sb_predictor_term!(_, _, f, _) =
    error("sbimpl: unsupported predictor-term function `$f` (supported: `mo`, `mo1`)")

_sb_n_obs_probe(terms) = begin
    for t in terms
        t isa NamedColumn && parent(t) isa DataColumn && return name(t)
    end
    nothing
end
_sb_any_data_symbol(data) = begin
    isempty(data) && error("sbimpl: can't emit `rep_vector(1., n)` — no data column seen yet. Make sure an observed `~` comes before the intercept-only predictor, or add a concrete covariate.")
    # Prefer any vector-valued entry
    for (k, v) in data
        v isa AbstractVector && return k
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
_sb_scalar_expr(x::ExprColumn, data) = Expr(:call, getf(x), (_sb_scalar_expr(a, data) for a in getargs(x))...)
_sb_scalar_expr(x, _) = error("sbimpl: cannot lift to Stan expression: $(typeof(x)): $x")
