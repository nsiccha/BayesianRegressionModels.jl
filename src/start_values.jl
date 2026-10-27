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
group's own `A (W[g,:] .* β)` — (zero deviations), with the block's unaddressed
absorbed coefficients at 0. An `|ID|` such a block absorbed keeps its address:
`ID = Σ` or `(; sd)` sets the block's total scale, the SD of each group's
total around `A β`, which is the margin's between-group SD. An `|ID|` that a
sum-to-zero block (`s2z_groups`) claimed keeps its address the same way:
`(; sd)` sets the block's scale `tau`. A coefficient a sum-to-zero block
absorbed sets the block's sampled population coefficient `theta = β + B μ`
to the value, taking the block means `μ` as zero.

An `|ID|` block whose margins are partly integrated out of the model (an
omitted response) samples only its active margins; their scales and
correlation are the corresponding sub-block of the given covariance.

Unknown addresses, ambiguous names, wrong shapes and values outside a
parameter's support are errors. The result is checked: constraining it
reproduces every addressed value. No input is mutated.
"""
function brm_start end

# One parsed address per target, independent of the backend.
#   (; kind=:population, predictor, coefficient, value)
#   (; kind=:categorical, predictor, coefficient, levels, values)
#   (; kind=:ranef, id, margins, sd, cor)
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

function _brm_start_ranef_target(id, margins, Σ::AbstractMatrix)
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
    (; kind=:ranef, id, margins, sd, cor=K == 1 ? nothing : cor)
end
function _brm_start_ranef_target(id, margins, value::NamedTuple)
    extra = setdiff(keys(value), (:sd, :cor))
    isempty(extra) || error(
        "brm_start: `|$id|` takes a covariance matrix or `(; sd, cor)`; unknown " *
        "field(s) $(Tuple(extra))")
    K = length(margins)
    sd = haskey(value, :sd) ? _brm_start_block_sd(id, K, value.sd) : nothing
    cor = haskey(value, :cor) ? _brm_start_block_cor(id, margins, value.cor) : nothing
    (; kind=:ranef, id, margins, sd, cor=K == 1 ? nothing : cor)
end
_brm_start_ranef_target(id, margins, value) = error(
    "brm_start: `|$id|` takes a covariance matrix or `(; sd, cor)`, got $(repr(value))")

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
    records = _rk_coordinate_records(rk.plan)
    values = _rk_constrained_values(rk, u0)
    writes = Tuple{Symbol,Tuple,Any}[]
    for target in _brm_start_targets(parent(rk), inits)
        _brm_rk_start_writes!(writes, Val(target.kind), target, records, values)
    end
    _brm_rk_start_point(rk, values, writes)
end

# Write `(declaration, index, value)` constrained values into `values` (the
# constrained view of the base point), invert the layout, and check that the
# point reproduces every written value.
function _brm_rk_start_point(rk::RKBRMI, values, writes)
    seen = Set{Tuple{Symbol,Tuple}}()
    for (declaration, index, _) in writes
        (declaration, index) in seen && error(
            "brm_start: two addresses set RK value `$declaration`$(collect(index))")
        push!(seen, (declaration, index))
    end
    for (declaration, index, value) in writes
        values = _brm_rk_set_declaration_value(values, declaration, index, value)
    end
    u = Vector{Float64}(_rk_unconstrained_point(rk, values))
    all(isfinite, u) || error("brm_start: the RK point is not finite")
    check = _rk_constrained_values(rk, u)
    for (declaration, index, value) in writes
        got = _brm_rk_declaration_value(check, declaration)
        _brm_start_check("RK value `$declaration`$(collect(index))",
            isempty(index) ? got : got[index...], value)
    end
    u
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

function _brm_rk_start_writes!(writes, ::Val{:ranef}, t, records, values)
    blocks = [r for r in records if r.kind === :ranef && isequal(r.id, t.id)]
    length(blocks) == 1 || error(
        "brm_start: `|$(t.id)|` has $(length(blocks)) RK random-effect records; " *
        "expected one plain block")
    r = only(blocks)
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
    absorbed = Dict{Symbol,Vector{Pair{Symbol,Float64}}}()
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
        _brm_sb_start_writes!(writes, absorbed, Val(target.kind), target, d,
            names, totals)
    end
    _brm_sb_start_totals!(writes, absorbed, sb, names, totals)
    seen = Set{Int}()
    for (i, _) in writes
        i in seen && error("brm_start: two addresses set Stan value `$(names[i])`")
        push!(seen, i)
    end
    θ = BridgeStan.param_constrain(stan_model, u0)
    for (i, value) in writes
        θ[i] = value
    end
    u = BridgeStan.param_unconstrain(stan_model, θ)
    all(isfinite, u) || error("brm_start: the Stan point is not finite")
    check = BridgeStan.param_constrain(stan_model, u)
    for (i, value) in writes
        _brm_start_check("Stan value `$(names[i])`", check[i], value)
    end
    isempty(shares_values) || _brm_sb_start_check_shares(stan_model, d, u, shares_values)
    u
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

function _brm_sb_start_writes!(writes, absorbed, ::Val{:population}, t, d, names, totals)
    k = _brm_sb_absorbing_block(totals, t.predictor, t.coefficient)
    if !isnothing(k)
        push!(get!(absorbed, totals[k].predictor, Pair{Symbol,Float64}[]),
            t.coefficient => t.value)
        return
    end
    # A sum-to-zero block samples `theta = β + B μ`; with its block means at
    # zero, `theta` is the coefficient's value.
    s2z = s2z_effect_blocks(d.plan)
    k = _brm_sb_absorbing_block(s2z, t.predictor, t.coefficient)
    if !isnothing(k)
        block = s2z[k]
        outputs = [o for o in d.outputs if o.name === block.population]
        length(outputs) == 1 || error(
            "brm_start: sum-to-zero coefficients `$(block.population)` own " *
            "$(length(outputs)) descriptor outputs; expected exactly one")
        coordinates = _brm_element_coordinates(only(outputs), names)
        length(coordinates) == length(block.population_columns) || error(
            "brm_start: sum-to-zero coefficients `$(block.population)` have " *
            "$(length(block.population_columns)) columns but " *
            "$(length(coordinates)) Stan values")
        push!(writes, (coordinates[findfirst(==(t.coefficient),
            block.population_columns)], t.value))
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

function _brm_sb_start_writes!(writes, absorbed, ::Val{:categorical}, t, d, names, totals)
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

function _brm_sb_start_writes!(writes, absorbed, ::Val{:parameter}, t, d, names, totals)
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

function _brm_sb_start_writes!(writes, absorbed, ::Val{:ranef}, t, d, names, totals)
    blocks = [b for b in ranef_blocks(d) if isequal(b.id, t.id)]
    if isempty(blocks)
        scales = _brm_claimed_id_scales(d, t.id)
        isempty(scales) && error(
            "brm_start: `|$(t.id)|` has no sampled random-effect block in this " *
            "Stan model")
        return _brm_sb_start_total_scales!(writes, t, scales, names)
    end
    (; block, margins) = _brm_sb_ranef_block(d,
        (; group=only(unique(b.group for b in blocks)), id=t.id, margins=());
        caller="brm_start")
    block.generated && error(
        "brm_start: `|$(t.id)|` block `$(block.binding)` is drawn in generated " *
        "quantities and has no sampled coordinates")
    P = _brm_start_margin_positions(t, margins)
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

# An exact-total block samples each group's total `A (W[g,:] .* β) + deviation`;
# place every total at the population value the absorbed coefficients give.
function _brm_sb_start_totals!(writes, absorbed, sb, names, totals)
    for block in totals
        given = get(absorbed, block.predictor, nothing)
        isnothing(given) && continue
        β = zeros(Float64, length(block.population_columns))
        for (coefficient, value) in given
            β[only(findall(==(coefficient), collect(block.population_columns)))] = value
        end
        mu = _total_location(block, β)
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
# block's scale `tau` (for totals, the SD of each group's total around `A β`).
# The claimed margins are independent, so a correlation other than the
# identity has no coordinate.
function _brm_sb_start_total_scales!(writes, t, scales, names)
    P = _brm_start_margin_positions(t, [s.margin for s in scales])
    length(P) == length(t.margins) || error(
        "brm_start: internal: $(length(P)) of the $(length(t.margins)) margins " *
        "of `|$(t.id)|` are in exact total-coefficient or sum-to-zero blocks")
    isnothing(t.cor) || isapprox(t.cor, I; atol=1e-12) || error(
        "brm_start: the margins of `|$(t.id)|` are independent in this model's " *
        "exact total-coefficient or sum-to-zero blocks; its correlation must be the identity")
    isnothing(t.sd) && return
    for (s, j) in zip(scales, P)
        s.output.kind === :generated_quantity && error(
            "brm_start: `|$(t.id)|` block `$(s.block.binding)` is drawn " *
            "in generated quantities and has no sampled coordinates")
        push!(writes, (_brm_claimed_scale_coordinate(s, names, "brm_start"), t.sd[j]))
    end
end
