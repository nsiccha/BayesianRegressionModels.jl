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

# A level correlation fits a grouping of `n_levels` levels exactly.
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
_brm_level_cor_label(value::NamedColumn) = name(value)
_brm_level_cor_label(value) = repr(value)

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
