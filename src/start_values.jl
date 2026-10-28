# Address-keyed starting values -> one complete unconstrained point of a backend.
#
# The caller names model quantities by the addresses priors use (a linear
# predictor's coefficients, an `|ID|` block's scales and correlation, a named
# parameter) instead of by emitted coordinate names. `_brm_start_targets`
# parses that grammar once against the BRMI. Each backend then writes the
# addressed CONSTRAINED values into its constrained view of a base point and
# inverts its own transforms: RK through its semantic coordinate records and the
# layout's `unconstrain`, Stan through the descriptor's resolvers and BridgeStan's
# `param_unconstrain`. Coordinates no address reaches keep the base point's
# constrained value, so standardized random-effect draws stay at the base.

"""
    brm_start(rk::RKBRMI, inits::NamedTuple; base) -> Vector{Float64}
    brm_start(sb::SBBRMI, stan_model, inits::NamedTuple; base) -> Vector{Float64}

Build a complete unconstrained starting point from initial values keyed by
the model's prior addresses. The RK method returns a packed point of `rk`'s
layout, the value `rk_logdensity_problem` and [`brm_nlme_start`](@ref) take.
The Stan method returns a point of the compiled `stan_model` (a
`BridgeStan.StanModel` compiled from `sb`, e.g. `stan_instantiate(sb; path).model`),
the value a sampler takes as its initial position.

`inits` keys are resolved against the model:

- `lp = (; Intercept = 1.2, x = 0.3)` sets population coefficients of linear
  predictor `lp`, on its link scale (`effect(lp, Intercept)`,
  `effect(lp, x)`). Use the public predictor name: `CL` for
  `log(CL) ~ 1 + x`. A categorical predictor column takes level-keyed pairs,
  `lp = (; arm = ["b" => 0.4, "c" => -0.1])`, one per non-reference level.
  The first categorical term of an intercept-free predictor (`0 + arm`,
  `0 + factor(g)`) is coded as cell means and has no reference level: every
  level is a coefficient, `lp = (; arm = ["a" => 1.1, "b" => 0.4, ...])`.
- `ID = Σ` sets the scales and correlation of the shared `(… | ID | g)` block
  from the covariance matrix `Σ` of its natural effects. Rows and columns
  follow `ranefcoefnames(brmi, ID)`. `ID = (; sd, cor)` sets either part:
  `sd` is one positive value per margin (or one for all) and `cor` a
  correlation matrix in the same order.
- `ID = (; sd, cor, effects)` also places the block's groups:
  `effects = ["a" => 1.2, "b" => -0.4, …]` gives each listed level its
  effect, one number for a one-margin block and one vector in
  `ranefcoefnames` order otherwise. An effect is what the level adds to each
  margin's linear predictor, on its link scale: the `effects` that
  [`brm_ranef_draws`](@ref) reads back. The block's draws are solved for under
  its scales and correlation as given (or at `base`), and a Student-t block's
  mixing weights at `base`, so each listed group starts at its own value
  instead of the population value. Unlisted levels keep their coordinates at
  `base`: their standardized draws, or their effects in a centred block.
- A stratified `(… | ID | gr(g, by = b))` block has its own scales and
  correlation in each stratum (level of `b`). `sd`, `cor` and `ID = Σ` set
  every stratum; `stratum => value` pairs set the listed ones,
  `sd = ["small" => 0.4, "large" => [0.6, 0.2]]` (one value for all margins,
  or one per margin), `cor = ["small" => C]`, `ID = ["small" => Σ]`. Unlisted
  strata keep their values at `base`. Each listed level's `effects` hold
  under its own stratum's scales and correlation.
- `name = value` sets a scalar parameter (`sigma ~ Exponential(1)`) or a vector
  parameter (cutpoints, a `Dirichlet` simplex) on its constrained scale.
- `name = values` sets a latent declared in a `@plate` cell (`log_T ~
  Normal(loc, s)`) on its constrained scale: one flat vector, cell after cell,
  or one vector per cell when its length varies by cell (one number per cell
  for a scalar latent). Both backends take both forms and give the same
  physical point.

Every other coordinate keeps the constrained value it has at `base`, a full
unconstrained point of the same backend (default all zeros: coefficients 0,
scales 1, identity correlations, standardized random effects 0). A random
`base` gives Stan-style random starting values for the unaddressed rest. The
natural random effects are zero whenever their standardized draws are zero,
whatever scale and correlation are set.

On an `SBBRMI` with exact total-coefficient blocks, a coefficient the block
absorbed has no sampled coordinate: its value places every group's total at
its population value `A β` — with group-level predictors in the block, the
group's own `A (W[g,:] .* β)` — plus the group's effect given through the
`|ID|` the block absorbed (`(; effects)`, the deviation of the total), with
the block's unaddressed absorbed coefficients and effects at 0. Such an
`|ID|` keeps its address: `ID = Σ` or `(; sd)` sets the block's total scale,
the SD of each group's total around `A β`, which is the margin's
between-group SD. An `|ID|` that a sum-to-zero block (`s2z_groups`) claimed
keeps its address the same way: `(; sd)` sets the block's scale `tau`, and
`(; effects)`, which must list every level, sets the levels' deviations from
their mean `μ`. A coefficient a sum-to-zero block absorbed, or effects given
for it, set the block's sampled population coefficient `theta = β + B μ`,
with `μ` the mean of the given effects (zero without) and an unaddressed
coefficient at 0.

An `|ID|` block whose margins are partly integrated out of the model (an
omitted response) samples only its active margins; their scales, correlation
and effects are the corresponding parts of the given values.

Unknown addresses, ambiguous names, wrong shapes and values outside a
parameter's support are errors. The result is checked: constraining it
reproduces every addressed value. No input is mutated.
"""
function brm_start end

# One parsed address per target, independent of the backend.
#   (; kind=:population, predictor, coefficient, value)
#   (; kind=:categorical, predictor, coefficient, levels, values)
#   (; kind=:ranef, id, margins, sd, cor, effects)
#   (; kind=:parameter, name, value)
function _brm_start_targets(brmi::BRMI, inits::NamedTuple)
    targets = NamedTuple[]
    for (key, value) in pairs(inits)
        margins = ranefcoefnames(brmi, key)
        if !isnothing(margins)
            isnothing(popcoefnames(brmi, key)) || error(
                "brm_start: `$key` names both an `|ID|` random-effect block and a " *
                "linear predictor; the address is ambiguous")
            push!(targets, _brm_start_ranef_target(key,
                Tuple{Symbol,Symbol}[(m.predictor, m.coefficient) for m in margins],
                value))
        elseif value isa NamedTuple
            for (coefficient, v) in pairs(value)
                push!(targets, _brm_start_population_target(key, coefficient, v))
            end
        else
            push!(targets, (; kind=:parameter, name=key,
                value=_brm_start_parameter_value(key, value)))
        end
    end
    targets
end

_brm_start_real(what, x::Real) = isfinite(x) ? Float64(x) :
    error("brm_start: $what must be finite, got $x")
_brm_start_real(what, x) = error("brm_start: $what must be a real number, got $(repr(x))")

_brm_start_population_target(predictor, coefficient, value::Real) =
    (; kind=:population, predictor, coefficient,
       value=_brm_start_real("effect($predictor, $coefficient)", value))
_brm_start_population_target(predictor, coefficient, value::AbstractDict) =
    _brm_start_population_target(predictor, coefficient, collect(pairs(value)))
function _brm_start_population_target(predictor, coefficient,
        value::AbstractVector{<:Pair})
    levels = _brm_transport_level.(first.(value))
    allunique(levels) || error(
        "brm_start: categorical `effect($predictor, $coefficient)` repeats a level")
    (; kind=:categorical, predictor, coefficient, levels,
       values=[_brm_start_real("effect($predictor, $coefficient) at level " *
           repr(level), v) for (level, v) in zip(levels, last.(value))])
end
_brm_start_population_target(predictor, coefficient, value) = error(
    "brm_start: `effect($predictor, $coefficient)` takes a real value, or " *
    "level => value pairs for a categorical column; got $(repr(value))")

_brm_start_parameter_value(name, value::Real) = _brm_start_real("`$name`", value)
_brm_start_parameter_value(name, value::AbstractVector{<:Real}) =
    [_brm_start_real("`$name`", x) for x in value]
# One vector per `@plate` cell, for a cell latent whose length varies by cell.
_brm_start_parameter_value(name, value::AbstractVector{<:AbstractVector{<:Real}}) =
    [[_brm_start_real("`$name` in cell $g", x) for x in v] for (g, v) in enumerate(value)]
_brm_start_parameter_value(name, value) = error(
    "brm_start: `$name` is neither a linear predictor (give its coefficients " *
    "as a NamedTuple) nor an `|ID|` block; as a named parameter it takes a " *
    "real value or a vector, or one vector per cell for a `@plate` cell " *
    "latent, got $(repr(value))")

# A parameter's values in its output's element order. Per-cell vectors are a
# ragged cell latent's segments, cell after cell; anything else has no cells.
_brm_start_flat_values(name, output, value::Real) = [value]
_brm_start_flat_values(name, output, value::AbstractVector{<:Real}) = value
function _brm_start_flat_values(name, output, value::AbstractVector{<:AbstractVector})
    ends = output.segments
    isnothing(ends) && error(
        "brm_start: `$name` takes one vector per cell only as a `@plate` cell " *
        "latent whose length varies by cell; give its values as one vector")
    lengths = diff([0; ends])
    length(value) == length(lengths) || throw(DimensionMismatch(
        "brm_start: cell latent `$name` has $(length(lengths)) cells, got " *
        "$(length(value)) vectors"))
    for (g, (v, n)) in enumerate(zip(value, lengths))
        length(v) == n || throw(DimensionMismatch(
            "brm_start: cell latent `$name` has $n values in cell $g, got $(length(v))"))
    end
    reduce(vcat, value; init=Float64[])
end

# `sd`/`cor` set every stratum of a stratified `gr(g, by = b)` block;
# `stratum_sd`/`stratum_cor` (`(; levels, values)`, from `stratum => value`
# pairs) set the listed strata only.
_brm_start_ranef_target(id, margins; sd=nothing, cor=nothing, effects=nothing,
        stratum_sd=nothing, stratum_cor=nothing) =
    (; kind=:ranef, id, margins, sd, cor=length(margins) == 1 ? nothing : cor, effects,
       stratum_sd, stratum_cor=length(margins) == 1 ? nothing : stratum_cor)
function _brm_start_ranef_target(id, margins, Σ::AbstractMatrix)
    sd, cor = _brm_start_covariance_parts(id, margins, Σ)
    _brm_start_ranef_target(id, margins; sd, cor)
end
# One covariance per listed stratum of a stratified `gr(g, by = b)` block.
function _brm_start_ranef_target(id, margins, value::Union{AbstractDict,
        AbstractVector{<:Pair}})
    _, strata = _brm_start_block_part(id, "covariance", value,
        Σ -> _brm_start_covariance_parts(id, margins, Σ))
    _brm_start_ranef_target(id, margins;
        stratum_sd=(; strata.levels, values=first.(strata.values)),
        stratum_cor=(; strata.levels, values=last.(strata.values)))
end
function _brm_start_ranef_target(id, margins, value::NamedTuple)
    extra = setdiff(keys(value), (:sd, :cor, :effects))
    isempty(extra) || error(
        "brm_start: `|$id|` takes a covariance matrix or `(; sd, cor, effects)`; " *
        "unknown field(s) $(Tuple(extra))")
    K = length(margins)
    sd, stratum_sd = _brm_start_block_part(id, "sd", get(value, :sd, nothing),
        x -> _brm_start_block_sd(id, K, x))
    cor, stratum_cor = _brm_start_block_part(id, "cor", get(value, :cor, nothing),
        x -> _brm_start_block_cor(id, margins, x))
    effects = haskey(value, :effects) ?
        _brm_start_block_effects(id, margins, value.effects) : nothing
    _brm_start_ranef_target(id, margins; sd, cor, effects, stratum_sd, stratum_cor)
end
_brm_start_ranef_target(id, margins, value) = error(
    "brm_start: `|$id|` takes a covariance matrix or `(; sd, cor, effects)`, got " *
    "$(repr(value))")

# The scales and correlation of a covariance in `ranefcoefnames` order.
function _brm_start_covariance_parts(id, margins, Σ::AbstractMatrix)
    K = length(margins)
    size(Σ) == (K, K) || throw(DimensionMismatch(
        "brm_start: covariance of `|$id|` has size $(size(Σ)); its $K margins are " *
        "$(Tuple(margins)) (`ranefcoefnames` order)"))
    all(isfinite, Σ) && issymmetric(Σ) || error(
        "brm_start: covariance of `|$id|` must be finite and symmetric")
    isposdef(Symmetric(Matrix{Float64}(Σ))) || error(
        "brm_start: covariance of `|$id|` is not positive definite")
    sd = sqrt.(Float64.(diag(Σ)))
    cor = Matrix{Float64}(Σ) ./ (sd * sd')
    cor[diagind(cor)] .= 1
    sd, cor
end
_brm_start_covariance_parts(id, margins, Σ) = error(
    "brm_start: the covariance of a stratum of `|$id|` must be a matrix, got $(repr(Σ))")

# A block part given for every stratum, `(parse(value), nothing)`, or per
# stratum of a stratified `gr(g, by = b)` block as `stratum => value` pairs,
# `(nothing, (; levels, values))` with each value parsed.
_brm_start_block_part(id, part, ::Nothing, parse) = (nothing, nothing)
_brm_start_block_part(id, part, value, parse) = (parse(value), nothing)
_brm_start_block_part(id, part, value::AbstractDict, parse) =
    _brm_start_block_part(id, part, collect(pairs(value)), parse)
function _brm_start_block_part(id, part, value::AbstractVector{<:Pair}, parse)
    isempty(value) && error("brm_start: `|$id|` $part lists no stratum")
    levels = _brm_transport_level.(first.(value))
    allunique(levels) || error("brm_start: `|$id|` $part repeats a stratum")
    (nothing, (; levels, values=[parse(v) for v in last.(value)]))
end

# `stratum => value` parts address the strata of a `gr(g, by = b)` block only.
function _brm_start_check_unstratified(t)
    isnothing(t.stratum_sd) && isnothing(t.stratum_cor) && return nothing
    error("brm_start: `|$(t.id)|` is not a stratified `gr(g, by = b)` block, so " *
        "its scales and correlation take no `stratum => value` pairs; give `sd` " *
        "per margin (or one value for all) and `cor` as one matrix")
end

# `(s, value)` for each stratum `s` (a position in `strata`, the block's stratum
# labels in index order) that `all` (every stratum) or `by` (the listed ones)
# sets.
function _brm_start_stratum_values(t, part, all, by, strata)
    isnothing(all) || return [(s, all) for s in eachindex(strata)]
    isnothing(by) && return Tuple{Int,Any}[]
    normalized = _brm_transport_level.(strata)
    map(zip(by.levels, by.values)) do (level, value)
        s = findall(l -> isequal(l, level), normalized)
        isempty(s) && error("brm_start: `|$(t.id)|` $part names stratum " *
            "$(repr(level)), which is not a stratum of this block; its strata are " *
            "$(repr(normalized))")
        (only(s), value)
    end
end

# Per-level natural effects of an `|ID|` block, keyed by level label: one
# number per level for a one-margin block, one vector in `ranefcoefnames` order
# otherwise. Parsed into the labels and a levels × margins matrix.
_brm_start_block_effects(id, margins, value::AbstractDict) =
    _brm_start_block_effects(id, margins, collect(pairs(value)))
function _brm_start_block_effects(id, margins, value::AbstractVector{<:Pair})
    isempty(value) && return nothing
    K = length(margins)
    levels = _brm_transport_level.(first.(value))
    allunique(levels) || error("brm_start: `|$id|.effects` repeats a level")
    values = Matrix{Float64}(undef, length(value), K)
    for (i, (level, v)) in enumerate(zip(levels, last.(value)))
        values[i, :] = _brm_start_level_effect(id, margins, level, v)
    end
    (; levels, values)
end
_brm_start_block_effects(id, margins, value) = error(
    "brm_start: `|$id|.effects` takes level => value pairs, got $(repr(value))")

function _brm_start_level_effect(id, margins, level, v::Real)
    length(margins) == 1 || throw(DimensionMismatch(
        "brm_start: `|$id|` has $(length(margins)) margins $(Tuple(margins)); give " *
        "level $(repr(level)) one effect per margin as a vector"))
    [_brm_start_real("the effect of `|$id|` at level $(repr(level))", v)]
end
function _brm_start_level_effect(id, margins, level, v::AbstractVector{<:Real})
    length(v) == length(margins) || throw(DimensionMismatch(
        "brm_start: `|$id|` has $(length(margins)) margins $(Tuple(margins)) " *
        "(`ranefcoefnames` order); level $(repr(level)) has $(length(v)) effects"))
    [_brm_start_real("the effect of `|$id|` at level $(repr(level))", x) for x in v]
end
_brm_start_level_effect(id, margins, level, v) = error(
    "brm_start: the effect of `|$id|` at level $(repr(level)) must be a real " *
    "number or one per margin, got $(repr(v))")

# The positions in `levels` (a block's level labels) of the given labels.
function _brm_start_level_positions(id, given, levels)
    normalized = _brm_transport_level.(levels)
    map(given) do level
        i = findall(l -> isequal(l, level), normalized)
        isempty(i) && error("brm_start: `|$id|` has no level $(repr(level)); " *
            "its levels are $(repr(normalized))")
        length(i) == 1 || error("brm_start: internal: level $(repr(level)) " *
            "occurs $(length(i)) times among the levels of `|$id|`")
        only(i)
    end
end

_brm_start_block_sd(id, K, sd::Real) = _brm_start_block_sd(id, K, fill(sd, K))
function _brm_start_block_sd(id, K, sd::AbstractVector{<:Real})
    length(sd) == K || throw(DimensionMismatch(
        "brm_start: `|$id|` has $K margins, got $(length(sd)) scales"))
    all(x -> isfinite(x) && x > 0, sd) || error(
        "brm_start: scales of `|$id|` must be finite and positive, got $(sd)")
    Vector{Float64}(sd)
end
_brm_start_block_sd(id, K, sd) = error(
    "brm_start: `|$id|.sd` must be a positive number or vector, got $(repr(sd))")

function _brm_start_block_cor(id, margins, C::AbstractMatrix{<:Real})
    K = length(margins)
    size(C) == (K, K) || throw(DimensionMismatch(
        "brm_start: correlation of `|$id|` has size $(size(C)); its $K margins are " *
        "$(Tuple(margins))"))
    all(isfinite, C) && issymmetric(C) && all(isone, diag(C)) &&
        isposdef(Symmetric(Matrix{Float64}(C))) || error(
        "brm_start: correlation of `|$id|` must be symmetric, positive definite " *
        "and have a unit diagonal")
    Matrix{Float64}(C)
end
_brm_start_block_cor(id, margins, C) = error(
    "brm_start: `|$id|.cor` must be a correlation matrix, got $(repr(C))")

# The block's margins `ms` as positions in the target's `ranefcoefnames` order.
function _brm_start_margin_positions(t, ms)
    map(collect(ms)) do m
        j = findall(==(Tuple(m)), t.margins)
        length(j) == 1 || error(
            "brm_start: internal: margin $m of `|$(t.id)|` occurs $(length(j)) " *
            "times among $(Tuple(t.margins))")
        only(j)
    end
end

_brm_start_lower_factor(C) = Matrix{Float64}(cholesky(Symmetric(C)).L)

function _brm_start_base(base, n, what)
    length(base) == n || throw(DimensionMismatch(
        "brm_start: base has length $(length(base)), the $what has $n coordinates"))
    all(isfinite, base) || error("brm_start: base must be finite")
    Vector{Float64}(base)
end

function _brm_start_check(what, got, expected)
    isapprox(got, expected; atol=1e-9, rtol=1e-9) || error(
        "brm_start: $what is $(got) after unconstraining, expected $(expected)")
end

# ---------------------------------------------------------------------------
# ReactiveKernels: semantic coordinate records + the layout's inverse.

# Implemented by `BayesianRegressionModelsReactiveKernelsExt`: the packed
# unconstrained point of named constrained values.
function _rk_unconstrained_point end

function brm_start(rk::RKBRMI, inits::NamedTuple;
        base=zeros(Float64, length(_rk_layout_coordinate_names(rk))))
    u0 = _brm_start_base(base, length(_rk_layout_coordinate_names(rk)), "RK layout")
    records = [_brm_rk_start_record(rk, r) for r in _rk_coordinate_records(rk.plan)]
    values = _rk_constrained_values(rk, u0)
    writes = Tuple{Symbol,Tuple,Any}[]
    effects = NamedTuple[]
    for target in _brm_start_targets(parent(rk), inits)
        _brm_rk_start_writes!(writes, Val(target.kind), target, records, values)
        target.kind === :ranef && !isnothing(target.effects) &&
            push!(effects, (; target, record=_brm_rk_ranef_record(target, records)))
    end
    _brm_rk_start_point(rk, values, writes, effects)
end

# Write `(declaration, index, value)` constrained values into `values` (the
# constrained view of the base point), then the standardized draws that give
# each block's addressed effects under its scales, correlation and mixing
# weights as written, invert the layout, and check that the point reproduces
# every written value and effect.
function _brm_rk_start_point(rk::RKBRMI, values, writes, effects=NamedTuple[])
    _brm_rk_check_distinct(writes)
    for (declaration, index, value) in writes
        values = _brm_rk_set_declaration_value(values, declaration, index, value)
    end
    z_writes = Tuple{Symbol,Tuple,Any}[]
    for (; target, record) in effects
        _brm_rk_effect_writes!(z_writes, rk, values, target, record)
    end
    for (declaration, index, value) in z_writes
        values = _brm_rk_set_declaration_value(values, declaration, index, value)
    end
    writes = [writes; z_writes]
    _brm_rk_check_distinct(writes)
    u = Vector{Float64}(_rk_unconstrained_point(rk, values))
    all(isfinite, u) || error("brm_start: the RK point is not finite")
    check = _rk_constrained_values(rk, u)
    for (declaration, index, value) in writes
        got = _brm_rk_declaration_value(check, declaration)
        _brm_start_check("RK value `$declaration`$(collect(index))",
            isempty(index) ? got : got[index...], value)
    end
    for (; target, record) in effects
        (; rows, P) = _brm_rk_effect_rows(rk, target, record)
        E = _brm_rk_block_effects(check, record, length(P))
        _brm_start_check("effects of `|$(target.id)|`", E[rows, :],
            target.effects.values[:, P])
    end
    u
end

function _brm_rk_check_distinct(writes)
    seen = Set{Tuple{Symbol,Tuple}}()
    for (declaration, index, _) in writes
        (declaration, index) in seen && error(
            "brm_start: two addresses set RK value `$declaration`$(collect(index))")
        push!(seen, (declaration, index))
    end
end

function _brm_rk_start_writes!(writes, ::Val{:population}, t, records, values)
    found = Tuple{Symbol,Tuple}[]
    for r in records
        if r.kind === :population && r.predictor === t.predictor &&
                r.coefficient === t.coefficient
            push!(found, (r.declaration, get(r, :index, ())))
        elseif r.kind === :population_block && r.predictor === t.predictor
            for i in findall(==(t.coefficient), collect(r.labels))
                push!(found, (r.declaration, (i,)))
            end
        end
    end
    length(found) == 1 || _brm_start_unknown_population(t, records, length(found))
    push!(writes, (first(only(found)), last(only(found)), t.value))
end

function _brm_rk_start_writes!(writes, ::Val{:categorical}, t, records, values)
    blocks = [r for r in records if r.kind === :population_block &&
        r.predictor === t.predictor && r.coefficient === t.coefficient]
    length(blocks) == 1 || _brm_start_unknown_population(t, records, length(blocks))
    r = only(blocks)
    levels = _brm_transport_level.(collect(r.level_values))
    for (level, value) in zip(t.levels, t.values)
        i = findall(l -> isequal(l, level), levels)
        length(i) == 1 || error(
            "brm_start: `effect($(t.predictor), $(t.coefficient))` has no " *
            "coefficient for level $(repr(level)); its coefficient levels are " *
            "$(repr(levels))")
        push!(writes, (r.declaration, (only(i),), value))
    end
end

function _brm_start_unknown_population(t, records, n)
    n > 1 && error("brm_start: internal: `effect($(t.predictor), " *
        "$(t.coefficient))` matches $n RK coordinate records")
    available = [(r.predictor, r.coefficient) for r in records
        if r.kind === :population && r.predictor === t.predictor]
    append!(available, [(r.predictor, label) for r in records
        if r.kind === :population_block && r.predictor === t.predictor
        for label in (t.kind === :categorical ? (r.coefficient,) : r.labels)])
    error("brm_start: `effect($(t.predictor), $(t.coefficient))` is not a sampled " *
        "population coefficient of this model. Addressable coefficients of " *
        "`$(t.predictor)`: " *
        (isempty(available) ? "(none)" : join(("$c" for (_, c) in available), ", ")))
end

function _brm_rk_start_writes!(writes, ::Val{:parameter}, t, records, values)
    # A `@plate` cell latent, addressed by its authored name.
    latent = [r for r in records if r.kind === :cell_latent && r.name === t.name]
    if !isempty(latent)
        length(latent) == 1 || error("brm_start: two `@plate` cells declare a " *
            "latent `$(t.name)`; it has no single address")
        r = only(latent)
        current = _brm_rk_declaration_value(values, r.declaration)
        push!(writes, (r.declaration, (), _brm_rk_cell_latent_value(t.name, current, t.value)))
        return
    end
    found = [r for r in records if r.kind in (:scalar, :vector) &&
        r.declaration === t.name]
    if isempty(found)
        available = [r.kind === :cell_latent ? r.name : r.declaration
            for r in records if r.kind in (:scalar, :vector, :cell_latent)]
        error("brm_start: `$(t.name)` is not a linear predictor, an `|ID|` block or " *
            "a named parameter of this model. Named parameters: " *
            (isempty(available) ? "(none)" : join(string.(available), ", ")))
    end
    value = _brm_rk_start_value(t.name, t.value)
    current = _brm_rk_declaration_value(values, t.name)
    size(current) == size(value) || throw(DimensionMismatch(
        "brm_start: `$(t.name)` has size $(size(current)), got $(size(value))"))
    push!(writes, (t.name, (), value))
end

_brm_rk_start_value(name, value) = value
_brm_rk_start_value(name, value::AbstractVector{<:AbstractVector}) = error(
    "brm_start: `$name` takes one vector per cell only as a `@plate` cell " *
    "latent; give its values as one vector")

# A cell latent's start in RK's constrained form: one number per cell for a
# scalar latent, one vector per cell for an array latent, given per cell or as
# one flat vector cell after cell (SBBRMI's two forms).
function _brm_rk_cell_latent_value(name, current::AbstractVector{<:Real}, value)
    value isa AbstractVector{<:Real} && length(value) == length(current) ||
        throw(DimensionMismatch("brm_start: cell latent `$name` holds one value " *
            "in each of its $(length(current)) cells; give one number per cell"))
    Vector{Float64}(value)
end
function _brm_rk_cell_latent_value(name, current::AbstractVector{<:AbstractVector}, value)
    lengths = length.(current)
    if value isa AbstractVector{<:AbstractVector}
        length(value) == length(lengths) || throw(DimensionMismatch(
            "brm_start: cell latent `$name` has $(length(lengths)) cells, got " *
            "$(length(value)) vectors"))
        for (g, (v, n)) in enumerate(zip(value, lengths))
            length(v) == n || throw(DimensionMismatch(
                "brm_start: cell latent `$name` has $n values in cell $g, got $(length(v))"))
        end
        return [Vector{Float64}(v) for v in value]
    end
    value isa AbstractVector{<:Real} && length(value) == sum(lengths; init=0) ||
        throw(DimensionMismatch("brm_start: cell latent `$name` has " *
            "$(sum(lengths; init=0)) values in cells of lengths $lengths; give " *
            "one vector per cell or one flat vector cell after cell"))
    ends = cumsum(lengths)
    [Vector{Float64}(value[(e - n + 1):e]) for (n, e) in zip(lengths, ends)]
end
_brm_rk_cell_latent_value(name, current, value) = error(
    "brm_start: cell latent `$name` holds $(typeof(current)) values; only " *
    "scalar and vector cell latents take a start")

# A coordinate record as the writes read it. A stratified block's
# (`gr(g, by = b)`) also carries its stratum labels in `levels(by)` order and
# the stratum of each level of its grouping.
_brm_rk_start_record(rk, r) = _brm_rk_start_record(rk, Val(r.kind), r)
_brm_rk_start_record(rk, ::Val, r) = r
function _brm_rk_start_record(rk, ::Val{:stratified_ranef}, r)
    columns = rk.plan.columns
    strata = _rk_grouping_levels(columns[r.by])
    level_strata = _brm_level_strata(columns[r.group], columns[r.by],
        _rk_grouping_levels(columns[r.group]), strata, r; caller="brm_start", fail=error)
    merge(r, (; strata, level_strata))
end

function _brm_rk_ranef_record(t, records)
    blocks = [r for r in records if r.kind in (:ranef, :stratified_ranef) &&
        isequal(r.id, t.id)]
    length(blocks) == 1 || error(
        "brm_start: `|$(t.id)|` has $(length(blocks)) RK random-effect records; " *
        "expected one block")
    only(blocks)
end

function _brm_rk_start_writes!(writes, ::Val{:ranef}, t, records, values)
    r = _brm_rk_ranef_record(t, records)
    _brm_rk_ranef_writes!(writes, Val(r.kind), t, r, values)
end

# Each stratum `s` of a stratified block has its own scales, row `s` of `tau`
# (strata × margins), or entry `s` of each margin's `tau_k` when the margins'
# laws differ, and its own correlation factor `L[:, :, s]` (margins × margins ×
# strata; `_rk_ast_stratified_group_component!`). `sd` and `cor` set every
# stratum, `stratum => value` pairs the listed ones.
function _brm_rk_ranef_writes!(writes, ::Val{:stratified_ranef}, t, r, values)
    P = _brm_start_margin_positions(t, r.margins)
    for (s, sd) in _brm_start_stratum_values(t, "sd", t.sd, t.stratum_sd, r.strata),
            (k, j) in enumerate(P)
        push!(writes, isnothing(r.scales) ? (r.scale, (s, k), sd[j]) :
            (r.scales[k], (s,), sd[j]))
    end
    isnothing(r.L) && return writes
    for (s, C) in _brm_start_stratum_values(t, "cor", t.cor, t.stratum_cor, r.strata)
        push!(writes, (r.L, (:, :, s), _brm_start_lower_factor(C[P, P])))
    end
    writes
end

function _brm_rk_ranef_writes!(writes, ::Val{:ranef}, t, r, values)
    _brm_start_check_unstratified(t)
    P = _brm_start_margin_positions(t, r.margins)
    known = get(r, :known_scales, nothing)
    if !isnothing(t.sd)
        # A known-value block names only its sampled scales; an address agrees
        # with each known one and writes nothing for it.
        sd = map(enumerate(P)) do (k, j)
            isnothing(known) || isnothing(known[k]) ? t.sd[j] :
                _brm_start_known_scale(t, j, known[k])
        end
        _brm_rk_scale_writes!(writes, r, sd, values)
    end
    known_cor = get(r, :known_cor, nothing)
    if !isnothing(t.cor) && !isnothing(known_cor)
        _brm_start_known_cor(t, P, known_cor)
    elseif !isnothing(t.cor) && !isnothing(r.L)
        push!(writes, (r.L, (), _brm_start_lower_factor(t.cor[P, P])))
    end
end

# The rows of an RK block's standardized draws `z` (its levels) that the
# target's given levels are, and the block's margins as positions `P` in the
# target's `ranefcoefnames` order.
function _brm_rk_effect_rows(rk, t, r)
    levels = _rk_ranef_record_levels(rk, r)
    (; rows=_brm_start_level_positions(t.id, t.effects.levels, levels),
       P=_brm_start_margin_positions(t, r.margins))
end

# The standardized draws of an RK block whose given levels have the given
# effects, the other levels' draws kept: one write of the whole `z`.
function _brm_rk_effect_writes!(writes, rk, values, t, r)
    (; rows, P) = _brm_rk_effect_rows(rk, t, r)
    K = length(P)
    D = t.effects.values[:, P]
    z = copy(_brm_rk_declaration_value(values, r.z))
    Z = z isa AbstractMatrix ? z : reshape(z, :, 1)
    size(Z, 2) == K || error(
        "brm_start: internal: RK draws `$(r.z)` have $(size(Z, 2)) margins, `|$(t.id)|` $K")
    level_A = _brm_rk_level_factor(values, r, K)
    level = _brm_rk_level_correlation(values, r)
    # A level-correlated block's effects are `level * B`, where row `g` of `B`
    # is `level_A(g) * Z[g, :]`; the unaddressed levels keep their draws.
    B = if isnothing(level)
        D
    else
        rest = setdiff(axes(Z, 1), rows)
        Brest = zeros(Float64, length(rest), K)
        for (i, g) in enumerate(rest)
            Brest[i, :] = level_A(g) * Z[g, :]
        end
        level[rows, rows] \ (D - level[rows, rest] * Brest)
    end
    for (i, g) in enumerate(rows)
        Z[g, :] = level_A(g) \ B[i, :]
    end
    push!(writes, (r.z, (), z))
end

# An RK block's effects at constrained `values`, levels × margins, as the
# emitted group component and gather compute them (`_rk_ast_group_component!`
# in src/rk_components.jl, the level-correlated gather in src/rk_values.jl):
# row `g` is `level_A(g) * z[g, :]` (`_brm_rk_level_factor`), multiplied on
# the left by the levels' Cholesky factor in a level-correlated block.
function _brm_rk_block_effects(values, r, K)
    z = _brm_rk_declaration_value(values, r.z)
    Z = z isa AbstractMatrix ? z : reshape(z, :, 1)
    level_A = _brm_rk_level_factor(values, r, K)
    B = Matrix{Float64}(undef, size(Z)...)
    for g in axes(Z, 1)
        B[g, :] = level_A(g) * Z[g, :]
    end
    level = _brm_rk_level_correlation(values, r)
    isnothing(level) ? B : level * B
end

# The factor `level_A(g)` that takes level `g`'s standardized draws to its
# effects at constrained `values`: `sqrt(w[g]) diag(tau) L` (`w` only for a
# Student-t block), or, in a stratified block, its stratum's
# `diag(tau[s, :]) L[:, :, s]`.
_brm_rk_level_factor(values, r, K) = _brm_rk_level_factor(values, Val(r.kind), r, K)
function _brm_rk_level_factor(values, ::Val, r, K)
    (; A, w) = _brm_rk_effect_factor(values, r, K)
    g -> isnothing(w) ? A : sqrt(w[g]) * A
end
function _brm_rk_level_factor(values, ::Val{:stratified_ranef}, r, K)
    factors = [_brm_nlme_natural_factor(values, r, K, s) for s in eachindex(r.strata)]
    g -> factors[r.level_strata[g]]
end

# `diag(tau) L` of a plain RK block at constrained `values`, and its per-level
# mixing weights (`nothing` for a Gaussian block). A known scale or correlation
# (`sd(...) = value`, `cor(:, ID) = R`) is the constant the block was emitted
# with.
function _brm_rk_effect_factor(values, r, K)
    tau = if isnothing(r.scales)
        v = _brm_rk_declaration_value(values, r.scale)
        v isa Number ? [v] : collect(v)
    else
        known = get(r, :known_scales, nothing)
        [isnothing(name) ? known[k] : _brm_rk_declaration_value(values, name)
            for (k, name) in enumerate(r.scales)]
    end
    length(tau) == K || error(
        "brm_start: internal: RK scale `$(r.scale)` has $(length(tau)) values for $K margins")
    known_cor = get(r, :known_cor, nothing)
    L = !isnothing(r.L) ? Matrix{Float64}(_brm_rk_declaration_value(values, r.L)) :
        isnothing(known_cor) ? Matrix{Float64}(I, K, K) :
        _brm_start_lower_factor(known_cor)
    mixing = get(r, :mixing, nothing)
    (; A=Diagonal(Vector{Float64}(tau)) * L,
       w=isnothing(mixing) ? nothing : _brm_rk_declaration_value(values, mixing))
end

# The lower Cholesky factor of an RK block's level correlation (`gr(g;
# cor=...)`): the sampled `LKJCholesky` declaration, or the known factor the
# record carries; `nothing` for a block whose levels are independent.
function _brm_rk_level_correlation(values, r)
    name = get(r, :level_correlation, nothing)
    isnothing(name) && return nothing
    factor = get(r, :level_factor, nothing)
    Matrix{Float64}(isnothing(factor) ? _brm_rk_declaration_value(values, name) : factor)
end

# A random-effect record's scales, `sd[k]` for its margin `k`; a `nothing`
# entry leaves that scale at the base point.
function _brm_rk_scale_writes!(writes, r, sd, values)
    if isnothing(r.scales)
        current = _brm_rk_declaration_value(values, r.scale)
        if current isa Number
            length(sd) == 1 || error(
                "brm_start: internal: scalar scale `$(r.scale)` for $(length(sd)) margins")
            isnothing(only(sd)) || push!(writes, (r.scale, (), only(sd)))
        else
            for (k, x) in enumerate(sd)
                isnothing(x) || push!(writes, (r.scale, (k,), x))
            end
        end
    else
        for (k, x) in enumerate(sd)
            isnothing(x) || push!(writes, (r.scales[k], (), x))
        end
    end
    writes
end

# Replace one addressed element (or the whole value) of a declaration in the
# nested constrained values, copying the leaf rather than mutating it.
function _brm_rk_set_declaration_value(values, declaration::Symbol, index::Tuple, value)
    path = hasproperty(values, declaration) ? (declaration,) :
        Tuple(Symbol.(split(String(declaration), '.')))
    _brm_rk_set_path(values, path, index, value)
end
function _brm_rk_set_path(values::NamedTuple, path::Tuple, index, value)
    head = first(path)
    hasproperty(values, head) || error(
        "brm_start: internal: RK constrained values carry no `$head`")
    leaf = getproperty(values, head)
    new = length(path) == 1 ? _brm_rk_set_leaf(leaf, index, value) :
        _brm_rk_set_path(leaf, Base.tail(path), index, value)
    merge(values, NamedTuple{(head,)}((new,)))
end
_brm_rk_set_leaf(leaf, ::Tuple{}, value) = value
function _brm_rk_set_leaf(leaf::AbstractArray, index::Tuple{Any,Vararg}, value)
    out = copy(leaf)
    out[index...] = value
    out
end

# ---------------------------------------------------------------------------
# StanBlocks: descriptor resolvers + BridgeStan's inverse transform.

function brm_start(sb::SBBRMI, stan_model, inits::NamedTuple;
        base=zeros(Float64, BridgeStan.param_unc_num(stan_model)))
    u0 = _brm_start_base(base, BridgeStan.param_unc_num(stan_model), "compiled Stan model")
    names = String.(BridgeStan.param_names(stan_model))
    d = brm_descriptor(sb)
    totals = total_effect_blocks(d.plan)
    writes = Tuple{Int,Float64}[]
    # What is written per block once every target is read: the coefficients
    # exact-total (`absorbed`) and S2Z (`s2z`) blocks absorbed, by predictor;
    # the effects of `|ID|`s those blocks claimed, by (block binding, column);
    # and the effects of conventional `|ID|` blocks.
    deferred = (; absorbed=Dict{Symbol,Vector{Pair{Symbol,Float64}}}(),
        s2z=Dict{Symbol,Vector{Pair{Symbol,Float64}}}(),
        claimed=Dict{Tuple{Symbol,Int},NamedTuple}(), blocks=NamedTuple[])
    total_latents = _brm_total_latent_kinds(parent(sb))
    # Values of a `T::shares(total)` cell latent, checked against the model's
    # once the point is built: they must sum to each cell's total.
    shares_values = Pair{Symbol,Vector{Float64}}[]
    for target in _brm_start_targets(parent(sb), inits)
        # A cell latent with a given total starts at its shares' coordinates.
        if target.kind === :parameter && haskey(total_latents, target.name)
            push!(shares_values, target.name =>
                _brm_start_flat_values(target.name, brm_output(d, target.name), target.value))
            target = _brm_total_start_target(target, total_latents[target.name], d)
        end
        _brm_sb_start_writes!(writes, deferred, Val(target.kind), target, d,
            names, totals)
    end
    _brm_sb_start_totals!(writes, deferred, sb, d, names, totals)
    θ = BridgeStan.param_constrain(stan_model, u0)
    _brm_sb_start_apply!(θ, writes, names)
    # An S2Z block's coordinates depend on its scale as written.
    s2z_writes = _brm_sb_start_s2z!(deferred, d, names, θ)
    writes = [writes; s2z_writes]
    _brm_sb_start_apply!(θ, writes, names)
    u = BridgeStan.param_unconstrain(stan_model, θ)
    all(isfinite, u) || error("brm_start: the Stan point is not finite")
    u = _brm_sb_start_block_effects(u, stan_model, d, deferred.blocks)
    check = BridgeStan.param_constrain(stan_model, u)
    for (i, value) in writes
        _brm_start_check("Stan value `$(names[i])`", check[i], value)
    end
    isempty(shares_values) || _brm_sb_start_check_shares(stan_model, d, u, shares_values)
    u
end

# Write `(index, value)` pairs into the constrained values `θ`, refusing two
# writes of one value.
function _brm_sb_start_apply!(θ, writes, names)
    seen = Set{Int}()
    for (i, _) in writes
        i in seen && error("brm_start: two addresses set Stan value `$(names[i])`")
        push!(seen, i)
    end
    for (i, value) in writes
        θ[i] = value
    end
    θ
end

# The start's values of a `T::shares(total)` latent are the model's only when
# they sum to each cell's total; otherwise refused, never rescaled.
function _brm_sb_start_check_shares(stan_model, d, u, shares_values)
    θ = BridgeStan.param_constrain(stan_model, u; include_tp=true)
    names = String.(BridgeStan.param_names(stan_model; include_tp=true))
    for (name, values) in shares_values
        model_values = θ[_brm_element_coordinates(brm_output(d, name), names)]
        isapprox(model_values, values; rtol=1e-8) || error(
            "brm_start: cell latent `$name::shares(total)`'s values must sum to " *
            "each cell's total; the model's values for this start are " *
            "$(model_values), the given ones $(values)")
    end
end

_brm_sb_absorbing_block(totals, predictor, coefficient) =
    findfirst(b -> b.predictor === predictor && coefficient in b.population_columns,
        totals)

function _brm_sb_start_writes!(writes, deferred, ::Val{:population}, t, d, names, totals)
    k = _brm_sb_absorbing_block(totals, t.predictor, t.coefficient)
    if !isnothing(k)
        push!(get!(deferred.absorbed, totals[k].predictor, Pair{Symbol,Float64}[]),
            t.coefficient => t.value)
        return
    end
    # A sum-to-zero block samples `theta = β + B μ`, written with the block's
    # effects (`_brm_sb_start_s2z!`).
    if !isnothing(_brm_sb_absorbing_block(s2z_effect_blocks(d.plan), t.predictor,
            t.coefficient))
        push!(get!(deferred.s2z, t.predictor, Pair{Symbol,Float64}[]),
            t.coefficient => t.value)
        return
    end
    q = brm_population_effect_coordinates(d, t.predictor, names;
        coefficient=t.coefficient)
    hasproperty(q, :contrasts) && error(
        "brm_start: `effect($(t.predictor), $(t.coefficient))` is a categorical " *
        "column; give one value per non-reference level as level => value pairs")
    q.recovered && error(
        "brm_start: internal: `effect($(t.predictor), $(t.coefficient))` resolved " *
        "to a recovered carrier outside every exact-total block")
    push!(writes, (only(q.coordinates), t.value))
end

function _brm_sb_start_writes!(writes, deferred, ::Val{:categorical}, t, d, names, totals)
    q = brm_population_effect_coordinates(d, t.predictor, names;
        coefficient=t.coefficient)
    hasproperty(q, :contrasts) || error(
        "brm_start: `effect($(t.predictor), $(t.coefficient))` is a numeric " *
        "coefficient; give it one real value")
    # `coordinates` pairs with `nonreference_levels`: one per non-reference
    # level under treatment coding, one per level for the cell means of an
    # intercept-free predictor's first categorical term.
    levels = _brm_transport_level.(q.nonreference_levels)
    for (level, value) in zip(t.levels, t.values)
        i = findall(l -> isequal(l, level), levels)
        length(i) == 1 || error(
            "brm_start: `effect($(t.predictor), $(t.coefficient))` has no " *
            "coefficient for level $(repr(level)); its coefficient levels are " *
            "$(repr(levels))")
        push!(writes, (q.coordinates[only(i)], value))
    end
end

function _brm_sb_start_writes!(writes, deferred, ::Val{:parameter}, t, d, names, totals)
    isempty(brm_outputs(d; logical=t.name)) && error(
        "brm_start: `$(t.name)` is not a linear predictor, an `|ID|` block or a " *
        "named parameter of this model")
    output = brm_output(d, t.name)
    order = _brm_element_coordinates(output, names)
    isempty(order) && error(
        "brm_start: `$(t.name)` has no sampled coordinate in the compiled Stan model")
    values = _brm_start_flat_values(t.name, output, t.value)
    length(order) == length(values) || throw(DimensionMismatch(
        "brm_start: `$(t.name)` has $(length(order)) Stan values, got $(length(values))"))
    for (i, value) in zip(order, values)
        push!(writes, (i, value))
    end
end

function _brm_sb_start_writes!(writes, deferred, ::Val{:ranef}, t, d, names, totals)
    blocks = [b for b in ranef_blocks(d) if isequal(b.id, t.id)]
    if isempty(blocks)
        scales = _brm_claimed_id_scales(d, t.id)
        isempty(scales) && error(
            "brm_start: `|$(t.id)|` has no sampled random-effect block in this " *
            "Stan model")
        return _brm_sb_start_total_scales!(writes, deferred, t, scales, names)
    end
    (; block, margins) = _brm_sb_ranef_block(d,
        (; group=only(unique(b.group for b in blocks)), id=t.id, margins=());
        caller="brm_start")
    block.generated && error(
        "brm_start: `|$(t.id)|` block `$(block.binding)` is drawn in generated " *
        "quantities and has no sampled coordinates")
    P = _brm_start_margin_positions(t, margins)
    # The effects are the block's draws under its scales and correlation as
    # written, solved for once the point holds them.
    isnothing(t.effects) || push!(deferred.blocks, (; t, block, P))
    isnothing(block.by) || return _brm_sb_start_stratified_writes!(writes, t, d, block,
        P, names)
    _brm_start_check_unstratified(t)
    pos = Dict(n => i for (i, n) in enumerate(names))
    lookup(name) = get(pos, name) do
        error("brm_start: `|$(t.id)|` expects Stan value `$name`, which the compiled " *
            "model does not declare")
    end
    K = length(P)
    spec = _RANEF_FAMILIES[block.family]
    if get(spec, :known, false)
        return _brm_sb_start_known_writes!(writes, t, d, block, P, lookup)
    end
    if !isnothing(t.sd)
        tau = get(spec, :tau, nothing)
        if isnothing(tau)
            K == 1 || error("brm_start: `|$(t.id)|` block `$(block.binding)` " *
                "(`$(block.family)`) has no per-margin scale")
            push!(writes, (lookup(string(block.binding, "_log_scale")), log(t.sd[only(P)])))
        else
            carrier = string(block.binding, "_", tau)
            for (k, j) in enumerate(P)
                name = string(carrier, ".", k)
                push!(writes, (K == 1 && !haskey(pos, name) ? lookup(carrier) :
                    lookup(name), t.sd[j]))
            end
        end
    end
    if !isnothing(t.cor) && K > 1
        L = _brm_start_lower_factor(t.cor[P, P])
        carrier = _sb_ranef_correlation_carrier(block)
        for i in 1:K, j in 1:K
            push!(writes, (lookup(string(carrier, ".", i, ".", j)), L[i, j]))
        end
    end
end

# A stratified `gr(g, by = b)` block has no block-wide scale or correlation:
# each stratum has its own scales and its own correlation factor. They are
# sampled as the stratum plate cells' own parameters, the block's `:scale` and
# `:correlation_cholesky` internals (`<binding>_tau_s_tau`, margins × strata,
# and `<binding>_L_s_L`; `_RANEF_FAMILIES`), which the `tau_s`/`L_s` that
# `brm_ranef_draws` reads collect as transformed parameters. `sd` and `cor` set
# every stratum, `stratum => value` pairs the listed ones; the others keep their
# base values.
function _brm_sb_start_stratified_writes!(writes, t, d, block, P, names)
    spec = _RANEF_FAMILIES[block.family]
    (; strata) = _brm_ranef_strata(d, block, "brm_start")
    S, K = length(strata), length(P)
    function coordinates(component, n)
        carrier = Symbol(block.binding, :_,
            only(k for (k, v) in pairs(spec.internals) if v === component))
        c = _brm_element_coordinates(
            _brm_ranef_carrier_output(d, block, carrier, "brm_start"), names)
        length(c) == n || error("brm_start: internal: `$carrier` of `|$(t.id)|` " *
            "has $(length(c)) Stan values, expected $n")
        c
    end
    values(part, all, by) = _brm_start_stratum_values(t, part, all, by, strata)
    tau = reshape(coordinates(:scale, K * S), K, S)
    for (s, sd) in values("sd", t.sd, t.stratum_sd), (k, j) in enumerate(P)
        push!(writes, (tau[k, s], sd[j]))
    end
    K == 1 && return nothing
    L = reshape(coordinates(:correlation_cholesky, S * K * K), S, K, K)
    for (s, C) in values("cor", t.cor, t.stratum_cor)
        F = _brm_start_lower_factor(C[P, P])
        for i in 1:K, j in 1:K
            push!(writes, (L[s, i, j], F[i, j]))
        end
    end
end

# A known value (`sd(...) = value`, `cor(:, ID) = R`) is not a coordinate on
# either backend. An initial value for it must restate it, is checked, and writes
# nothing; the sampled scales of the same block are written as usual.
function _brm_start_known_scale(t, j, known)
    isapprox(t.sd[j], known; rtol=1e-8) || error(
        "brm_start: the SD of margin $(t.margins[j]) of `|$(t.id)|` is held at " *
        "the known value $known (`sd(...) = value`); the initial value " *
        "$(t.sd[j]) contradicts it")
    nothing
end

function _brm_start_known_cor(t, P, known)
    isapprox(t.cor[P, P], known; rtol=1e-8, atol=1e-10) || error(
        "brm_start: the correlation of `|$(t.id)|` is held at the known matrix " *
        "$(known) (`cor(:, $(t.id)) = R`); the initial correlation " *
        "$(t.cor[P, P]) contradicts it")
    nothing
end

# A known-value SB block (`_sb_emit_known_id_bucket!`): its sampled scales are
# `_sb_known_scale_carrier`, and a known correlation has no `L`.
function _brm_sb_start_known_writes!(writes, t, d, block, P, lookup)
    record = d.plan.bindings[block.binding]
    known = record.known_scales
    free = findall(isnothing, collect(known))
    carrier = _sb_known_scale_carrier(block.binding, known)
    if !isnothing(t.sd)
        for (k, j) in enumerate(P)
            if isnothing(known[k])
                push!(writes, (lookup(string(carrier, ".", only(findall(==(k), free)))),
                    t.sd[j]))
            else
                _brm_start_known_scale(t, j, known[k])
            end
        end
    end
    (isnothing(t.cor) || length(P) == 1) && return nothing
    if isnothing(record.known_cor)
        L = _brm_start_lower_factor(t.cor[P, P])
        carrier = _sb_ranef_correlation_carrier(block)
        for i in eachindex(P), j in eachindex(P)
            push!(writes, (lookup(string(carrier, ".", i, ".", j)), L[i, j]))
        end
    else
        _brm_start_known_cor(t, P, record.known_cor)
    end
    nothing
end

# An exact-total block samples each group's total `A (W[g,:] .* β) + deviation`,
# the deviation being the group's effect. A block whose absorbed coefficients
# or claimed `|ID|` effects are given places every total at its group's
# population value plus its given effect; an unaddressed coefficient or effect
# counts as 0.
function _brm_sb_start_totals!(writes, deferred, sb, d, names, totals)
    for block in totals
        given = get(deferred.absorbed, block.predictor, Pair{Symbol,Float64}[])
        effects = [get(deferred.claimed, (block.binding, k), nothing)
            for k in eachindex(block.columns)]
        isempty(given) && all(isnothing, effects) && continue
        β = zeros(Float64, length(block.population_columns))
        for (coefficient, value) in given
            β[only(findall(==(coefficient), collect(block.population_columns)))] = value
        end
        mu = _total_location(block, β)
        levels = d.plan.preproc[block.group_index].const_.levels
        length(levels) == size(mu, 1) || error(
            "brm_start: internal: exact-total block of `$(block.predictor)` has " *
            "$(length(levels)) level labels for $(size(mu, 1)) groups")
        for (k, e) in enumerate(effects)
            isnothing(e) && continue
            mu[_brm_start_level_positions(e.id, e.levels, levels), k] .+= e.values
        end
        coordinates = _total_coordinates(sb, block, names).totals
        size(coordinates) == size(mu) || error(
            "brm_start: internal: exact-total block of `$(block.predictor)` has " *
            "$(size(coordinates)) totals for $(size(mu)) population values")
        for g in axes(coordinates, 1), k in axes(coordinates, 2)
            push!(writes, (coordinates[g, k], mu[g, k]))
        end
    end
end

# An `|ID|` an exact-total or S2Z block claimed: each margin's scale is the
# block's scale `tau` (for totals, the SD of each group's total around `A β`),
# and its effects are written with the block (`_brm_sb_start_totals!`,
# `_brm_sb_start_s2z!`). The claimed margins are independent, so a correlation
# other than the identity has no coordinate.
function _brm_sb_start_total_scales!(writes, deferred, t, scales, names)
    _brm_start_check_unstratified(t)
    P = _brm_start_margin_positions(t, [s.margin for s in scales])
    length(P) == length(t.margins) || error(
        "brm_start: internal: $(length(P)) of the $(length(t.margins)) margins " *
        "of `|$(t.id)|` are in exact total-coefficient or sum-to-zero blocks")
    isnothing(t.cor) || isapprox(t.cor, I; atol=1e-12) || error(
        "brm_start: the margins of `|$(t.id)|` are independent in this model's " *
        "exact total-coefficient or sum-to-zero blocks; its correlation must be the identity")
    isnothing(t.sd) && isnothing(t.effects) && return
    for (s, j) in zip(scales, P)
        s.output.kind === :generated_quantity && error(
            "brm_start: `|$(t.id)|` block `$(s.block.binding)` is drawn " *
            "in generated quantities and has no sampled coordinates")
        isnothing(t.sd) ||
            push!(writes, (_brm_claimed_scale_coordinate(s, names, "brm_start"), t.sd[j]))
        isnothing(t.effects) || (deferred.claimed[(s.block.binding, s.column)] =
            (; id=t.id, levels=t.effects.levels, values=t.effects.values[:, j]))
    end
end

# A sum-to-zero block (`s2z_groups`) samples, per column `k`, the deviations
# `r[:, k]` of its levels' effects from their mean `μ[k]` (through its contrast
# or group coordinates) and `theta = β + B μ`, the block means `μ` being
# integrated out. A column's given effects `e`, one per level, place
# `r[:, k] = e .- μ[k]` with `μ[k] = mean(e)` under the column's scale at `θ`;
# a column without effects keeps its coordinates and has `μ[k] = 0`. Each
# `theta[p]` whose coefficient is given, or which a given column's mean reaches,
# is written; an unaddressed coefficient counts as 0. Returns the writes.
function _brm_sb_start_s2z!(deferred, d, names, θ)
    writes = Tuple{Int,Float64}[]
    for block in s2z_effect_blocks(d.plan)
        given = get(deferred.s2z, block.predictor, Pair{Symbol,Float64}[])
        effects = [get(deferred.claimed, (block.binding, k), nothing)
            for k in eachindex(block.columns)]
        isempty(given) && all(isnothing, effects) && continue
        coordinates = _s2z_coordinates(d.plan, block, names)
        levels = d.plan.preproc[block.group_index].const_.levels
        J = length(levels)
        μ = zeros(Float64, length(block.columns))
        written = falses(length(block.population_columns))
        for (k, e) in enumerate(effects)
            isnothing(e) && continue
            rows = _brm_start_level_positions(e.id, e.levels, levels)
            length(rows) == J || error(
                "brm_start: `|$(e.id)|` is a sum-to-zero block (`s2z_groups`), " *
                "which samples its levels' effects as deviations from their mean; " *
                "give the effects of all $J levels, not $(length(rows))")
            values = zeros(Float64, J)
            values[rows] = e.values
            μ[k] = sum(values) / J
            x = _s2z_block_coordinates(block, values .- μ[k],
                θ[coordinates.scales[k]], k, θ[coordinates.contrasts[:, k]])
            append!(writes, zip(coordinates.contrasts[:, k], x))
            written .|= block.B[:, k] .!= 0
        end
        β = zeros(Float64, length(block.population_columns))
        for (coefficient, value) in given
            p = only(findall(==(coefficient), collect(block.population_columns)))
            β[p] = value
            written[p] = true
        end
        theta = β + block.B * μ
        for p in findall(written)
            push!(writes, (coordinates.theta[p], theta[p]))
        end
    end
    writes
end

# The effects of conventional `|ID|` blocks, written into the unconstrained
# point `u` that already holds every other start value. A block's effect
# carrier, the one `brm_ranef_draws` reads, is linear in the block's own
# coordinates (its standardized draws, or the effects themselves when centred)
# while its scales, correlation and mixing weights stay as they are at `u`. So
# the carrier's response to each coordinate is measured at `u`, and the
# coordinates that give the effects are solved for, whatever the block's
# emission. Levels are independent cells, measured together; a level-correlated
# block (`gr(g; cor=...)`) couples its levels, so its given levels form one
# cell. Levels without given effects keep their coordinates.
function _brm_sb_start_block_effects(u, stan_model, d, pending)
    isempty(pending) && return u
    unc = String.(BridgeStan.param_unc_names(stan_model))
    tp = String.(BridgeStan.param_names(stan_model; include_tp=true))
    u = copy(u)
    for (; t, block, P) in pending
        rows = _brm_start_level_positions(t.id, t.effects.levels, block.levels)
        D = t.effects.values[:, P]
        position = ranef_coordinates(block, unc)
        cells = _ranef_level_correlated(block.family) ? [collect(eachindex(rows))] :
            [[i] for i in eachindex(rows)]
        # A cell's coordinates and effects, margin fastest, then level.
        coordinates(c) = vec(position[:, rows[c]])
        effects(E, c) = vec(permutedims(E[rows[c], :]))
        n = maximum(c -> length(coordinates(c)), cells)
        probes = [copy(u) for _ in 1:n]
        for c in cells, (j, x) in enumerate(coordinates(c))
            probes[j][x] += 1
        end
        E = _brm_sb_start_effect_draws(stan_model, d, block, [[u]; probes], tp)
        for c in cells
            x = coordinates(c)
            e0 = effects(E[1, :, :], c)
            M = Matrix{Float64}(undef, length(e0), length(x))
            for j in eachindex(x)
                M[:, j] = effects(E[1 + j, :, :], c) .- e0
            end
            F = lu(M; check=false)
            issuccess(F) || error(
                "brm_start: the effects of `|$(t.id)|` do not determine the " *
                "coordinates of its block `$(block.binding)` at this point")
            u[x] .+= F \ (vec(permutedims(D[c, :])) .- e0)
        end
        all(isfinite, u) || error("brm_start: the Stan point is not finite")
        E = _brm_sb_start_effect_draws(stan_model, d, block, [u], tp)
        _brm_start_check("effects of `|$(t.id)|`", E[1, rows, :], D)
    end
    u
end

# Draws × levels × margins effects of a conventional `block` at the
# unconstrained points `us`, read through the carriers `brm_ranef_draws` reads.
function _brm_sb_start_effect_draws(stan_model, d, block, us, names)
    draws = permutedims(reduce(hcat, [BridgeStan.param_constrain(stan_model, x;
        include_tp=true) for x in us]))
    isnothing(block.by) ? _brm_ranef_block_draws(d, block, draws, names).effects :
        _brm_ranef_stratified_draws(d, block, draws, names).effects
end
