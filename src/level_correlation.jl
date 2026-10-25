# ---- correlation across the levels of a grouping factor (`cor=`) ------------
#
# `gr(g; cor=Omega)` and `hsgp(x; by=g, cor=Omega)` correlate the LEVELS of the
# grouping factor `g` with each other, where every other grouping keeps its
# levels independent. Each backend draws the term's level-by-coefficient matrix
# of standardized (or level-independent) values as it always does and multiplies
# it on the left by the lower Cholesky factor `L` of one `n_levels x n_levels`
# correlation matrix, so every coefficient's (or basis function's) level values
# have correlation `L * L'`: the intrinsic coregionalisation model. Terms that
# name the same `Omega` share it.
#
# `Omega` is a model-scope declaration `Omega ~ LKJCholesky(K, eta)` (a sampled
# correlation) or a data matrix (a known correlation, factored once in Julia).
#
# `hsgp(x; by=g, cor=multigroup(Omega, tau))` replaces the constant correlation
# by the multigroup coherence: basis frequency `omega_b` correlates the levels
# by `C_b = exp(-(rho omega_b)^2 / 2 * D2)` with the Gram distance
# `D2 = 2 tau (1 - Omega)`, so slow components are shared more than fast ones.
# It is the HSGP form of the kernel `(1 + D2)^(-d/2) k_SE(dx; rho sqrt(1 + D2))`
# (`brm_hsgp_multigroup_weights`). A random-effect block has no frequencies and
# refuses it.

"""
    multigroup(Omega, tau)

The frequency-dependent level correlation of a grouped HSGP,
`hsgp(x...; by = g, cor = multigroup(Omega, tau))`. Each basis frequency `ω`
correlates the levels of `g` by `exp(-(ρ ω)² / 2 · D2)` with the Gram distance
`D2 = 2 tau (1 - Omega)`: levels share their slow components more than their
fast ones, and the lag-0 correlation of two levels is `(1 + D2)^(-1/2)` (one
axis). `Omega` is an `LKJCholesky` declaration or a known correlation matrix,
as for `cor = Omega`; `tau` is a declared positive scalar parameter or a
positive constant. Formula-only: it is read by `@brm` and never called.
"""
function multigroup end

"""
    _BRMLevelCorrelation

The resolved `cor=` of a grouping: the declaration or data field it names
(`name`), whether that is a sampled `LKJCholesky` factor (`kind = :parameter`)
or a known correlation matrix (`kind = :data`, with its lower Cholesky
`factor`), and its dimension `K`, which must equal the grouping's level count.
"""
struct _BRMLevelCorrelation
    name::Symbol
    kind::Symbol
    K::Int
    factor::Union{Nothing,Matrix{Float64}}
end

_brm_level_correlation(::Nothing; kwargs...) = nothing
function _brm_level_correlation(value; what::AbstractString,
                                prefix::AbstractString="BRM backend lowering")
    value isa NamedColumn || error(
        "$prefix: `$what` must name an `LKJCholesky` declaration or a data " *
        "matrix, got `$(repr(value))`")
    source = parent(value)
    if source isa DataColumn
        C = _brm_correlation_matrix(parent(source); what)
        return _BRMLevelCorrelation(name(value), :data, size(C, 1),
                                    _brm_correlation_factor(C; what))
    end
    op = _as_expr_column(source)
    if !isnothing(op) && getf(op) === (~)
        rhs = _as_expr_column(last(getargs(op, 2)))
        if !isnothing(rhs) && getf(rhs) === LKJCholesky
            spec = _brm_lkj_cholesky_spec(name(value), rhs; prefix)
            return _BRMLevelCorrelation(name(value), :parameter, spec.K, nothing)
        end
    end
    error("$prefix: `$what` names `$(name(value))`, which is neither a " *
          "`$(name(value)) ~ LKJCholesky(K, eta)` declaration nor a data " *
          "matrix. Declare the correlation before the formula that reads it.")
end

"""
    _BRMMultigroupCorrelation

The resolved `cor = multigroup(Omega, tau)` of a grouped HSGP: the level
correlation `base` (as for `cor = Omega`) and the Gram scale `tau`, either the
name of a declared scalar parameter or a positive constant.
"""
struct _BRMMultigroupCorrelation
    base::_BRMLevelCorrelation
    tau::Union{Symbol,Float64}
end

function _brm_level_correlation(value::ExprColumn{typeof(multigroup)}; what::AbstractString,
                                prefix::AbstractString="BRM backend lowering")
    args = getargs(value)
    (length(args) == 2 && isempty(getkwargs(value))) || error(
        "$prefix: `$what` takes `multigroup(Omega, tau)`: a level correlation " *
        "and a positive Gram scale, no keywords")
    _BRMMultigroupCorrelation(_brm_level_correlation(first(args); what, prefix),
                              _brm_multigroup_tau(last(args); what, prefix))
end

# The Gram scale of `multigroup(Omega, tau)`: a declared scalar parameter (its
# prior, written by the author, gives it positive support) or a positive
# constant, literal or data.
function _brm_multigroup_tau(value::Real; what, prefix)
    isfinite(value) && value > 0 || error(
        "$prefix: `$what` needs a positive finite Gram scale `tau`, got $value")
    Float64(value)
end
function _brm_multigroup_tau(value::NamedColumn; what, prefix)
    source = parent(value)
    if source isa DataColumn
        data = parent(source)
        data isa Real || error(
            "$prefix: `$what` reads `tau = $(name(value))` from data, which " *
            "must be one positive number, got a $(typeof(data))")
        return _brm_multigroup_tau(data; what, prefix)
    end
    op = _as_expr_column(source)
    (!isnothing(op) && getf(op) === (~)) || error(
        "$prefix: `$what` names `tau = $(name(value))`, which is neither a " *
        "declared scalar parameter (`$(name(value)) ~ LogNormal(0, 1)`) nor a " *
        "positive constant")
    name(value)
end
_brm_multigroup_tau(value; what, prefix) = error(
    "$prefix: `$what` needs a declared scalar parameter or a positive " *
    "constant as `tau`, got `$(repr(value))`")

# A level correlation fits a grouping of `n_levels` levels exactly.
_brm_check_level_correlation(lc::_BRMMultigroupCorrelation, n_levels::Integer; kwargs...) =
    (_brm_check_level_correlation(lc.base, n_levels; kwargs...); lc)
function _brm_check_level_correlation(lc::_BRMLevelCorrelation, n_levels::Integer;
                                      what, group, prefix="BRM backend lowering")
    lc.K == n_levels || error(
        "$prefix: `$what` is $(lc.K) x $(lc.K), but grouping `$group` has " *
        "$n_levels levels; the correlation needs one row and column per level")
    lc
end

# The `cor=` a random-effect declaration's `gr(...)` carries, unresolved.
_brm_declaration_level_cor(d) = _brm_group_level_cor(d.raw_group)
_brm_group_level_cor(raw::ExprColumn) =
    getf(raw) === gr ? get(getkwargs(raw), :cor, nothing) : nothing
_brm_group_level_cor(_raw) = nothing

_brm_level_cor_name(::Nothing) = nothing
_brm_level_cor_name(value::NamedColumn) = name(value)
_brm_level_cor_name(value::ExprColumn{typeof(multigroup)}) =
    Symbol(_brm_level_cor_label(value))
_brm_level_cor_label(value::NamedColumn) = name(value)
_brm_level_cor_label(value::ExprColumn{typeof(multigroup)}) =
    "multigroup(" * join(map(_brm_level_cor_label, getargs(value)), ", ") * ")"
_brm_level_cor_label(value) = repr(value)

# A random-effect block correlates whole levels with one correlation; the
# multigroup coherence varies with frequency, which only an HSGP has.
_brm_refuse_block_multigroup(::_BRMLevelCorrelation, _what) = nothing
_brm_refuse_block_multigroup(lc::_BRMMultigroupCorrelation, what) = error(
    "BRM backend lowering: `$what` is refused: `multigroup(...)` makes the " *
    "level correlation depend on the basis frequency, which only a grouped " *
    "`hsgp(...; by=..., cor=multigroup(...))` has. Give the block " *
    "`cor = $(lc.base.name)` to share the same `Omega`")

# The one `cor=` shared by declarations that lower as one block. Every term of
# a block must name the same correlation (or none): the block's levels are
# correlated as a whole.
function _brm_block_level_cor(declarations, label; prefix="BRM backend lowering")
    values = [_brm_declaration_level_cor(d) for d in declarations]
    names = unique(_brm_level_cor_name.(values))
    length(names) == 1 || error(
        "$prefix: $label lowers as one random-effect block, but its terms " *
        "name different level correlations ($(join(repr.(names), ", "))); " *
        "give every term of the block the same `gr(...; cor=...)`")
    first(values)
end
