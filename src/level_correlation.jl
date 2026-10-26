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
#
# `multigroup((x, y), lambda; k)` gives the levels known coordinates instead:
# `D2 = |s_i - s_j|^2 / lambda^2`, so frequency `omega_b` correlates the levels
# by `exp(-d^2 / (2 r_b^2))` with spatial range `r_b = lambda / (rho omega_b)`.
# Each frequency's correlation is approximated with `k` functions per axis,
# by a Hilbert-space sine basis (`method = :hsgp`, the default) or inducing
# points on a grid (`method = :inducing`, meant for approximate-inference
# loops), with no `n_levels x n_levels` factor, so thousands of levels stay
# cheap (`brm_hsgp_multigroup_coordinate_weights`).

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

    multigroup((x, y), lambda; k = 20, method = :hsgp, c = 1.5)
    multigroup((x, y); k = 20, method = :hsgp, c = 1.5)

The same coherence over known level coordinates: `D2 = |s_i - s_j|² / lambda²`,
so frequency `ω` correlates two levels at distance `d` by `exp(-d² / (2 r²))`
with spatial range `r = lambda / (ρ ω)`. One coordinate field per axis (one to
three axes), each holding one value per level of `g` in the grouping's level
order, as a known `cor = C` has one row per level. `lambda` is a declared
positive scalar parameter or a positive constant, in the coordinates' unit;
left out, the term declares its own (see below).
Each frequency's correlation is approximated with `k` functions per axis,
rescaled to a unit diagonal, so the cost grows linearly in the number of
levels: `method = :hsgp` (the default) uses a sine basis over a box of
half-width `√(L₀² + (3.2 r)²)` with `L₀ = c · half-extent` (`c = 1.5` by
default); `method = :inducing`, meant for approximate-inference loops, places
inducing points on a grid of spacing `√(h₀² + (r / 1.5)²)` with
`h₀ = extent / (k - 1)` per axis.

The approximation resolves ranges down to `r₀`: the HSGP length-scale floor of
the box `L₀`, or 1.25 base spacings `h₀`. A frequency whose range is below
`r₀` has `ρ ω > lambda / r₀`, so its spectral amplitude stays below 1/100 of
the zero-frequency amplitude (the HSGP truncation tolerance) whenever
`lambda ≥ 2√(ln 100) r₀ ≈ 4.3 r₀`; a larger `k` lowers `r₀`. With `lambda`
left out, the term's own spatial scale has the prior `LogNormal(log(E), 1)`,
`E` the coordinates' largest extent, bounded below at `2√(ln 100) r₀`. A
declared `lambda`'s prior is its author's: give it `lower = ...` at that bound,
or raise `k`. `brm_multigroup_coherence` reports `r₀` and which frequencies
are resolved.
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

"""
    _BRMMultigroupCoordinates

The resolved `cor = multigroup((x, y), lambda; k, method, c)` of a grouped
HSGP: the coordinate fields' names (`axes`), their values as levels × axes
(`coordinates`), the spatial scale `lambda` (a declared scalar parameter's
name, a positive constant, or `nothing` for the term's own), the functions per
axis `k`, the approximation
`method` (`:inducing` or `:hsgp`) and the HSGP box factor `c`. With
`inducing = (ux, uy)`, `positions` holds one entry per axis, `(; name, values)`
with the data values or `nothing` for a declared vector parameter, and
`n_points` their common length; otherwise `positions` is empty and `n_points`
the grid's `k^D`.
"""
struct _BRMMultigroupCoordinates
    axes::Vector{Symbol}
    coordinates::Matrix{Float64}
    lambda::Union{Symbol,Float64,Nothing}
    k::Int
    method::Symbol
    c::Float64
    positions::Vector{NamedTuple{(:name, :values),Tuple{Symbol,Union{Nothing,Vector{Float64}}}}}
    n_points::Int
end

const _BRM_MULTIGROUP_COORDINATE_K = 20
const _BRM_MULTIGROUP_HSGP_C = 1.5
# The approximation's code in the emitted programs (Stan has no symbols).
_brm_multigroup_method_code(method::Symbol) = _brm_multigroup_method_code(Val(method))
_brm_multigroup_method_code(::Val{:inducing}) = 1
_brm_multigroup_method_code(::Val{:hsgp}) = 2

function _brm_level_correlation(value::ExprColumn{typeof(multigroup)}; what::AbstractString,
                                prefix::AbstractString="BRM backend lowering")
    args = getargs(value)
    _brm_multigroup_coordinates_form(value) &&
        return _brm_multigroup_coordinates(value; what, prefix)
    (length(args) == 2 && isempty(getkwargs(value))) || error(
        "$prefix: `$what` takes `multigroup(Omega, tau)`: a level correlation " *
        "and a positive Gram scale, no keywords; or " *
        "`multigroup((x, y), lambda; k = $(_BRM_MULTIGROUP_COORDINATE_K))`: " *
        "the levels' coordinates and a positive spatial scale")
    _BRMMultigroupCorrelation(_brm_level_correlation(first(args); what, prefix),
                              _brm_multigroup_scale(last(args), :tau; what, prefix))
end

# `multigroup(coordinates, lambda; k)` names its coordinates as a tuple of
# fields, which tells it apart from `multigroup(Omega, tau)` without data.
_brm_multigroup_coordinates_form(value::ExprColumn{typeof(multigroup)}) =
    !isempty(getargs(value)) && first(getargs(value)) isa Tuple
_brm_multigroup_coordinates_form(_value) = false

# The number of standardized spatial weights per frequency of
# `multigroup(coordinates, lambda; k)`, read from the formula alone: `k^D` grid
# points or basis functions, or the length of the `inducing` positions.
function _brm_multigroup_grid_size(value)
    haskey(getkwargs(value), :inducing) &&
        return _brm_multigroup_position_count(getkwargs(value)[:inducing],
            length(first(getargs(value))); what="multigroup(coordinates, lambda; inducing)",
            prefix="BRM backend lowering")
    k = get(getkwargs(value), :k, _BRM_MULTIGROUP_COORDINATE_K)
    k isa Integer && k >= 2 || error(
        "BRM backend lowering: `multigroup(coordinates, lambda; k)` needs an " *
        "integer `k >= 2` functions per axis, got `$(repr(k))`")
    Int(k)^length(first(getargs(value)))
end

function _brm_multigroup_coordinates(value; what, prefix)
    args, kw = getargs(value), getkwargs(value)
    length(args) in (1, 2) || error(
        "$prefix: `$what` takes `multigroup((x, y), lambda; k = " *
        "$(_BRM_MULTIGROUP_COORDINATE_K))`: the levels' coordinates and, " *
        "optionally, a positive spatial scale")
    extra = setdiff(keys(kw), (:k, :method, :c, :inducing))
    isempty(extra) || error(
        "$prefix: `$what` takes the keywords `k` (functions per axis), " *
        "`method` (`:inducing` or `:hsgp`), for `:hsgp` `c`, and for " *
        "`:inducing` the positions `inducing`; not " *
        join(("`$key`" for key in extra), ", "))
    method = get(kw, :method, :hsgp)
    method in (:inducing, :hsgp) || error(
        "$prefix: `$what` takes `method = :inducing` or `method = :hsgp`, " *
        "got `$(repr(method))`")
    method === :inducing && haskey(kw, :c) && error(
        "$prefix: `$what` takes the box factor `c` only with `method = :hsgp`; " *
        "the inducing grid spans the coordinates themselves")
    c = get(kw, :c, _BRM_MULTIGROUP_HSGP_C)
    c isa Real && isfinite(c) && c > 1 || error(
        "$prefix: `$what` needs a box factor `c > 1`, got `$(repr(c))`")
    fields = first(args)
    1 <= length(fields) <= 3 || error(
        "$prefix: `$what` needs one to three coordinate axes, got $(length(fields))")
    columns = map(fields) do field
        field isa NamedColumn && parent(field) isa DataColumn || error(
            "$prefix: `$what` reads each coordinate from a data field, one " *
            "value per level; got `$(repr(field))`")
        raw = parent(parent(field))
        raw isa AbstractVector{<:Real} && all(isfinite, raw) || error(
            "$prefix: `$what` coordinate `$(name(field))` must be a vector of " *
            "finite numbers, one per level")
        Float64.(raw)
    end
    allequal(length.(columns)) || error(
        "$prefix: `$what` coordinate fields have different lengths " *
        "($(join(length.(columns), ", "))); each holds one value per level")
    k = get(kw, :k, _BRM_MULTIGROUP_COORDINATE_K)
    k isa Integer && k >= 2 || error(
        "$prefix: `$what` needs an integer `k >= 2` functions per axis, " *
        "got `$(repr(k))`")
    positions = haskey(kw, :inducing) ?
        _brm_multigroup_positions(kw, length(fields); what, prefix) : []
    _BRMMultigroupCoordinates(collect(Symbol, name.(fields)), reduce(hcat, columns),
                              length(args) == 1 ? nothing :
                                  _brm_multigroup_scale(last(args), :lambda; what, prefix),
                              Int(k), method, Float64(c), positions,
                              isempty(positions) ? Int(k)^length(fields) :
                                  _brm_multigroup_position_count(kw[:inducing], length(fields); what, prefix))
end

# `lambda / r₀` at which a frequency the approximation cannot resolve has at
# most 1/100 of the zero-frequency spectral amplitude: `exp(-ratio² / 4) = 1/100`.
const _BRM_MULTIGROUP_RESOLUTION_RATIO = 2sqrt(log(100))
# The finest range the approximation resolves (`brm_multigroup_coordinate_grid`,
# `brm_multigroup_inducing_resolution`).
_brm_multigroup_resolution(lc::_BRMMultigroupCoordinates) = isempty(lc.positions) ?
    last(brm_multigroup_coordinate_grid(lc.coordinates, lc.k,
                                        _brm_multigroup_method_code(lc.method), lc.c)) :
    brm_multigroup_inducing_resolution(lc.coordinates, lc.n_points)
# The term's own spatial scale, `multigroup((x, y); k)` (decision `100gkpq`):
# `LogNormal(log(E), 1)` with `E` the coordinates' largest extent, bounded
# below where every frequency the basis cannot resolve stays negligible.
function _brm_multigroup_default_lambda(lc::_BRMMultigroupCoordinates)
    extent = maximum(maximum(c) - minimum(c) for c in eachcol(lc.coordinates))
    extent > 0 || error(
        "BRM backend lowering: `multigroup((x, y); k)` sets its spatial scale " *
        "from the coordinates' extent, but every level has the same coordinates; " *
        "give `lambda` explicitly")
    (; location=log(extent),
       lower=_BRM_MULTIGROUP_RESOLUTION_RATIO * _brm_multigroup_resolution(lc))
end

# `inducing = (ux, uy)`: one field per coordinate axis, each a data vector or
# a declared vector parameter (`ux[1:M] .~ ...`), all of one length `M`: the
# positions an approximate-inference loop optimizes and a later fit fixes.
function _brm_multigroup_positions(kw, n_axes; what, prefix)
    get(kw, :method, :hsgp) === :inducing || error(
        "$prefix: `$what` takes `inducing` positions only with `method = :inducing`")
    haskey(kw, :k) && error(
        "$prefix: `$what` takes either `k` (a grid of `k` points per axis) or " *
        "`inducing` positions, not both; the positions' length is their number")
    fields = kw[:inducing]
    _brm_multigroup_position_count(fields, n_axes; what, prefix)
    map(collect(fields)) do field
        source = parent(field)
        source isa DataColumn || return (; name=name(field), values=nothing)
        raw = parent(source)
        raw isa AbstractVector{<:Real} && all(isfinite, raw) || error(
            "$prefix: `$what` inducing positions `$(name(field))` must be a " *
            "vector of finite numbers")
        (; name=name(field), values=Vector{Float64}(raw))
    end
end
# The common length `M >= 2` of the positions' fields, one per coordinate axis.
function _brm_multigroup_position_count(fields, n_axes; what, prefix)
    fields isa Tuple && length(fields) == n_axes || error(
        "$prefix: `$what` needs `inducing = (ux, ...)` with one field per " *
        "coordinate axis ($n_axes), got `$(repr(fields))`")
    lengths = map(fields) do field
        field isa NamedColumn || error(
            "$prefix: `$what` reads each inducing position from a data field or " *
            "a declared vector parameter, got `$(repr(field))`")
        source = parent(field)
        source isa DataColumn && return length(parent(source))
        n = _brm_multigroup_declared_length(field)
        isnothing(n) && error(
            "$prefix: `$what` names inducing positions `$(name(field))`, which " *
            "is neither a data vector nor a declared vector parameter " *
            "(`$(name(field))[1:M] .~ Normal.(...)`)")
        n
    end
    allequal(lengths) || error(
        "$prefix: `$what` positions have different lengths " *
        "($(join(lengths, ", "))); give every axis the same points")
    M = first(lengths)
    M >= 2 || error("$prefix: `$what` needs at least two inducing positions, got $M")
    M
end
# The length `n` of a declared `name[1:n] .~ law`, or `nothing`.
function _brm_multigroup_declared_length(field::NamedColumn)
    op = _as_expr_column(parent(field))
    (!isnothing(op) && getf(op) === (~) &&
     last(getargs(op)) isa ExprColumn{typeof(_brm_vector_prior)}) || return nothing
    n = _brm_data_value(first(getargs(last(getargs(op)))))
    n isa Real && isinteger(n) ? Int(n) : nothing
end

# A positive scale of `multigroup` (`tau`, `lambda`): a declared scalar
# parameter (its prior, written by the author, gives it positive support) or a
# positive constant, literal or data.
function _brm_multigroup_scale(value::Real, label; what, prefix)
    isfinite(value) && value > 0 || error(
        "$prefix: `$what` needs a positive finite $(_brm_multigroup_scale_noun(label)) " *
        "`$label`, got $value")
    Float64(value)
end
function _brm_multigroup_scale(value::NamedColumn, label; what, prefix)
    source = parent(value)
    if source isa DataColumn
        data = parent(source)
        data isa Real || error(
            "$prefix: `$what` reads `$label = $(name(value))` from data, which " *
            "must be one positive number, got a $(typeof(data))")
        return _brm_multigroup_scale(data, label; what, prefix)
    end
    op = _as_expr_column(source)
    (!isnothing(op) && getf(op) === (~)) || error(
        "$prefix: `$what` names `$label = $(name(value))`, which is neither a " *
        "declared scalar parameter (`$(name(value)) ~ LogNormal(0, 1)`) nor a " *
        "positive constant")
    name(value)
end
_brm_multigroup_scale(value, label; what, prefix) = error(
    "$prefix: `$what` needs a declared scalar parameter or a positive " *
    "constant as `$label`, got `$(repr(value))`")
_brm_multigroup_scale_noun(label) = label === :tau ? "Gram scale" : "spatial scale"

# A level correlation fits a grouping of `n_levels` levels exactly.
_brm_check_level_correlation(lc::_BRMMultigroupCorrelation, n_levels::Integer; kwargs...) =
    (_brm_check_level_correlation(lc.base, n_levels; kwargs...); lc)
function _brm_check_level_correlation(lc::_BRMMultigroupCoordinates, n_levels::Integer;
                                      what, group, prefix="BRM backend lowering")
    size(lc.coordinates, 1) == n_levels || error(
        "$prefix: `$what` has $(size(lc.coordinates, 1)) coordinates per axis, " *
        "but grouping `$group` has $n_levels levels; give one coordinate per " *
        "level, in the grouping's level order")
    lc
end
function _brm_check_level_correlation(lc::_BRMLevelCorrelation, n_levels::Integer;
                                      what, group, prefix="BRM backend lowering")
    lc.K == n_levels || error(
        "$prefix: `$what` is $(lc.K) x $(lc.K), but grouping `$group` has " *
        "$n_levels levels; the correlation needs one row and column per level")
    lc
end

# Hyper-predicted scales of a grouped HSGP with `cor=` (`log(sd(hsgp(x))) ~
# 1 + (1 | g)`, `log(length_scale(hsgp(x))) ~ ...`): each level's spectral
# scales multiply its correlated weights, so basis function `j` draws the level
# weights from `MVN(0, diag(s_j) C_j diag(s_j))` with `s_j[g] = sqrt(S_g(omega_j))`
# (`C_j = Omega`, or a multigroup coherence). Both multigroup coherences, over
# a level correlation or over known coordinates, are functions of the one
# length scale every level shares, so a per-level length scale leaves them
# undefined; a per-level amplitude does not.
_brm_check_hyper_level_correlation(::Nothing, _hyper_plans; kwargs...) = nothing
_brm_check_hyper_level_correlation(::_BRMLevelCorrelation, _hyper_plans; kwargs...) = nothing
function _brm_check_hyper_level_correlation(
        ::Union{_BRMMultigroupCorrelation,_BRMMultigroupCoordinates}, hyper_plans;
        what, prefix="BRM backend lowering")
    any(p -> p.hyper === :length_scale, hyper_plans) && error(
        "$prefix: `$what` with a hyper-predicted length scale is refused: the " *
        "multigroup coherence is a function of one length scale shared by " *
        "every level, which a per-level length scale removes. Predict the " *
        "amplitude only (`log(sd(hsgp(...))) ~ 1 + (1 | g)`); per-level length " *
        "scales are available under a constant level correlation `cor = Omega`")
    nothing
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
function _brm_level_cor_label(value::ExprColumn{typeof(multigroup)})
    kw = getkwargs(value)
    "multigroup(" * join(map(_brm_level_cor_label, getargs(value)), ", ") *
        (isempty(kw) ? "" : "; " * join(("$key = $(_brm_level_cor_label(v))" for (key, v) in pairs(kw)), ", ")) *
        ")"
end
_brm_level_cor_label(value::Tuple) =
    "(" * join(map(_brm_level_cor_label, value), ", ") * ")"
_brm_level_cor_label(value) = repr(value)

# A random-effect block correlates whole levels with one correlation; the
# multigroup coherence varies with frequency, which only an HSGP has.
_brm_refuse_block_multigroup(::_BRMLevelCorrelation, _what) = nothing
_brm_refuse_block_multigroup(lc::_BRMMultigroupCorrelation, what) = error(
    "BRM backend lowering: `$what` is refused: `multigroup(...)` makes the " *
    "level correlation depend on the basis frequency, which only a grouped " *
    "`hsgp(...; by=..., cor=multigroup(...))` has. Give the block " *
    "`cor = $(lc.base.name)` to share the same `Omega`")
_brm_refuse_block_multigroup(::_BRMMultigroupCoordinates, what) = error(
    "BRM backend lowering: `$what` is refused: `multigroup(...)` makes the " *
    "level correlation depend on the basis frequency, which only a grouped " *
    "`hsgp(...; by=..., cor=multigroup(...))` has")

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
