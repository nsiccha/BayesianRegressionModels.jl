# src/rk_adaptive_centering.jl — adaptive partial centring of RKBRMI random
# effects under WarmupHMC (snag rkbrmi-has-no-ce-3158a399).
#
# RKBRMI emits every random effect non-centred: a block's standardized draws
# `<draws>.z.<g>.<k>`, its scales `<draws>.tau.<k>` (or `<draws>.tau_<k>`) and
# its correlation factor `<draws>.L.<m>`. Those unconstrained values are the
# ones the SBBRMI emission samples (the coordinate transport pairs them by
# `:identity`, `:exp` and `:cholesky`, a permutation): a scale coordinate is
# its log, and a factor coordinate Stan's `cholesky_factor_corr` free value.
# So the ordinary random-effect map of `adaptive_centering_problem` applies to
# an RK log density unchanged once its blocks name RK positions. This file
# builds those blocks and their hierarchical locations from RK's own coordinate
# records and layout, with no SBBRMI emission or Stan compile; the WarmupHMC
# extension wraps `rk_logdensity_problem(rk; ad_backend)` with them.

# A record's draws root: `b_g` of `b_g.z`.
function _rk_adaptive_binding(record)
    z = String(record.z)
    Symbol(endswith(z, ".z") ? z[1:end-2] : z)
end

# The RK random-effect records adaptive centring reads, in emission order:
# ordinary and stratified blocks, and the adaptive-only records of
# multi-membership (`:mm_ranef`) and R2D2-scaled (`:r2d2_ranef`) blocks.
_rk_adaptive_records(rk::RKBRMI) =
    [r for r in _rk_coordinate_records(rk.plan; adaptive=true)
     if r.kind in (:ranef, :stratified_ranef, :mm_ranef, :r2d2_ranef)]

# A record's grouping levels, the rows of its draws `z`: a multi-membership
# block's are the joint levels of its member columns (`levels(vcat(...))`).
_rk_adaptive_multi(record) =
    record.kind === :mm_ranef || get(record, :form, :plain) === :mm
_rk_adaptive_levels(rk::RKBRMI, record) = _rk_adaptive_multi(record) ?
    _rk_grouping_levels(vcat((rk.plan.columns[c] for c in record.group)...)) :
    _rk_ranef_record_levels(rk, record)

# A block's levels and the layout positions of its standardized draws
# `<z>.<g>.<k>` (`K x G`), or `nothing` for a block with no sampler coordinate
# (no observed likelihood reaches it). The layout must hold exactly `G` rows.
function _rk_adaptive_effects(rk::RKBRMI, record, pos)
    K = length(record.margins)
    levels = _rk_adaptive_levels(rk, record)
    G = length(levels)
    z = String(record.z)
    z_names = [string(z, ".", g, ".", k) for k in 1:K, g in 1:G]
    all(n -> !haskey(pos, n), z_names) && return nothing
    haskey(pos, string(z, ".", G + 1, ".1")) && error(
        "BRM adaptive centering on RKBRMI: block `$(_rk_adaptive_binding(record))` ",
        "has more rows of draws in the RK layout than its $G grouping levels.")
    levels, reshape(_rk_adaptive_positions(pos, vec(z_names), record,
        "standardized-draw"), K, G)
end

_rk_adaptive_refuse(record, why) = error(
    "BRM adaptive centering on RKBRMI: random-effect block over ",
    "`$(record.group)`", isnothing(record.id) ? "" : " (`|$(record.id)|`)",
    " with margins $(record.margins) ", why, " Sample it with ",
    "`SBBRMI(brmi)` and `adaptive_centering_problem(sb, …)`, or without ",
    "adaptive centering; refusing rather than leaving it silently fixed.")

# The layout position of each name, or an error naming the missing ones.
function _rk_adaptive_positions(pos, names, record, role)
    out = [get(pos, n, 0) for n in names]
    missing = names[out .== 0]
    isempty(missing) || error(
        "BRM adaptive centering on RKBRMI: block `$(_rk_adaptive_binding(record))` ",
        "expects $role coordinates the RK layout does not have: ",
        join(missing, ", "), ". `unc_names` must be the RKBRMI's layout ",
        "coordinate names (`rk_logdensity_problem`'s position order).")
    out
end

"""
    adaptive_centering_blocks(rk::RKBRMI, unc_names=<the RK layout names>)

The random-effect blocks of an [`RKBRMI`](@ref) for adaptive partial centring,
with indices that are positions of `rk_logdensity_problem(rk; ad_backend)`'s
unconstrained vector. Every block is compiled non-centred (`target_c = 0`):
`effects[k, g]` is the standardized draw of term `k`, level `g`, and
`cholesky_free` its correlation factor's free coordinates. `ranef.binding` is
the block's RK draws declaration (`b_g` of `b_g.z.1.1`), `ranef.levels` its
grouping's levels in RK's row order of `z`.

A Gaussian block whose scales are each sampled as `exp` of a coordinate (a zero
lower bound, unbounded above) is an [`AdaptiveCenteringBlock`](@ref):
`log_scales` are those coordinates. Every other covered block is an
[`R2D2AdaptiveCenteringBlock`](@ref), as on SBBRMI: term `k`'s log scale is a
function of the unconstrained vector that reads, never transforms, its inputs:

- a bounded scale (`Uniform(a, b)`, `truncated(...; lower, upper)`) is the log
  of its RK constrained value, `lo + exp(u)` or `lo + (hi - lo) * logistic(u)`;
- a known scale (`sd(lp, g[, coefficient]) = value`) is the constant
  `log(value)`, and a known correlation (`cor(:, g) = R`) the constant factor
  `cholesky_known`, with no Cholesky coordinates;
- a Student-t block (`ranef(:, g) ~ TDist(nu)`) scales level `g`'s factor by
  `sqrt(w[g])`, its mixing weight, read through `mixing[g]` (the coordinate
  `<draws>.w.<g>`, `log w[g]`).

A stratified `gr(g, by=b)` block is one block per stratum of `b`, in RK's
stratum order: each holds the stratum's groups and reads the stratum's own
scales and factor, as [`adaptive_centering_blocks`](@ref) does for an SBBRMI.
Covered groupings: one grouping column or a shared `|ID|` bucket's joint
columns, scalar or correlated, `||` blocks included.

A block whose levels are correlated (`gr(g; cor=...)`) raises, as on SBBRMI.
Multi-membership and R2D2-scaled blocks, HSGP weights, `car`, `cdar` and
`gmrf` fields, cell latents and completions keep the emitted frame: they are
not cells of this wrapper, and their coordinates pass through unchanged.
"""
function adaptive_centering_blocks(rk::RKBRMI,
        unc_names=_rk_layout_coordinate_names(rk))
    names = String.(unc_names)
    pos = Dict{String,Int}(n => i for (i, n) in enumerate(names))
    transforms = _rk_layout_coordinate_transforms(rk)
    length(transforms) == length(names) || error(
        "BRM adaptive centering on RKBRMI: `unc_names` has $(length(names)) ",
        "entries for a layout of $(length(transforms)) coordinates.")
    bounds = _rk_adaptive_scale_bounds(rk, transforms)
    out = AbstractAdaptiveCenteringBlock[]
    for record in _rk_adaptive_records(rk)
        haskey(record, :level_correlation) && _rk_adaptive_refuse(record,
            "correlates the levels of its grouping (`gr(g; cor=...)`); per-level " *
            "centring of such a block is not implemented on either backend.")
        if record.kind === :stratified_ranef
            append!(out, _rk_adaptive_stratified_blocks(rk, record, pos, bounds))
            continue
        end
        found = _rk_adaptive_effects(rk, record, pos)
        isnothing(found) && continue
        levels, effects = found
        K, G = size(effects)
        if record.kind === :r2d2_ranef
            push!(out, _rk_adaptive_r2d2_block(rk, record, levels, effects, pos,
                transforms, bounds, names))
            continue
        end
        # A known-value block names its sampled scales only (`nothing` at a
        # known margin) and records the known values (`known_scales`, and a
        # known correlation's matrix `known_cor`, which samples no factor).
        known = get(record, :known_scales, ntuple(_ -> nothing, K))
        scale_names = isnothing(record.scales) ?
            [string(record.scale, ".", k) for k in 1:K] :
            [isnothing(s) ? nothing : String(s) for s in record.scales]
        findall(!isnothing, known) == findall(isnothing, scale_names) ||
            error("BRM adaptive centering on RKBRMI: block ",
                "`$(_rk_adaptive_binding(record))` records known scales ",
                "$(known) that do not complement its sampled scales ",
                "$(record.scales).")
        scales = Union{Nothing,Int}[isnothing(n) ? nothing :
            only(_rk_adaptive_positions(pos, [n], record, "scale")) for n in scale_names]
        cholesky_free = isnothing(record.L) ? Int[] :
            _rk_adaptive_positions(pos,
                [string(record.L, ".", m) for m in 1:(K * (K - 1)) ÷ 2],
                record, "Cholesky")
        known_cor = get(record, :known_cor, nothing)
        ranef = RanefBlock(_rk_adaptive_binding(record),
            K == 1 ? :rk_ranef_scalar : :rk_ranef_correlated, record.group,
            record.id, nothing, levels, K, G, record.z, true, false)
        # A Student-t block also draws one mixing weight per level,
        # `w ~ InverseGamma(nu / 2, nu / 2)`, sampled as `log w`.
        mixing = haskey(record, :mixing) ?
            _rk_adaptive_exp_positions(pos, transforms,
                [string(record.mixing, ".", g) for g in 1:G], record, "mixing-weight") :
            Int[]
        push!(out, _rk_adaptive_block(ranef, effects, cholesky_free, scales, known,
            known_cor, mixing, bounds, record, names))
    end
    claimed = Int[]
    for block in out
        append!(claimed, vec(block.effects), block.cholesky_free)
        block isa AdaptiveCenteringBlock && append!(claimed, block.log_scales)
    end
    allunique(claimed) || error(
        "BRM adaptive centering on RKBRMI: random-effect blocks claim overlapping ",
        "layout coordinates; refusing an ambiguous transform.")
    # The map's triangular Jacobian assumes no block's scale (or weight) reads a
    # coordinate the map rewrites.
    effects = Set{Int}(Iterators.flatten(vec(b.effects) for b in out))
    for block in out
        block isa R2D2AdaptiveCenteringBlock || continue
        isdisjoint(_adaptive_read_coordinates(block), effects) || error(
            "BRM adaptive centering on RKBRMI: block `$(block.ranef.binding)` ",
            "derives its scale from a random-effect coordinate this transform ",
            "rewrites; refusing an inexact transform.")
    end
    out
end

# An R2D2-scaled block (`sd(...) ~ r2d2(...)`, `:r2d2_ranef`) in derived-scale
# form, as `_adaptive_r2d2_block` reads the SBBRMI emission. Margin `k`'s scale
# is `reference * sqrt(phi[entry] * R2 / (1 - R2))` (`brm_r2d2m2_scale`, unit
# variance), so its log is linear in log-atoms of RK coordinates:
# `log(reference)` (a constant, a data scalar, or a sampled positive scale's
# coordinate), `(log R2 - log(1 - R2)) / 2` (`R2` sampled on `(0, 1)`) and
# `log(phi[entry]) / 2` (RK's stick-breaking simplex). A margin a partial ICC
# statement leaves unaddressed samples its own scale (`free`).
function _rk_adaptive_r2d2_block(rk::RKBRMI, record, levels, effects, pos, transforms,
                                 bounds, names)
    K, G = size(effects)
    binding = _rk_adaptive_binding(record)
    coordinate(name, role) = only(_rk_adaptive_positions(pos, [String(name)], record, role))
    function scale_atom!(linear, bounded, i)
        b = bounds[i]
        isnothing(b) && _rk_adaptive_refuse(record, "derives its scale from " *
            "`$(names[i])`, which RK samples under the transform " *
            "`$(repr(transforms[i]))`, not as a positive scale.")
        b == (0.0, Inf) ? push!(linear, (1.0, i)) : push!(bounded, (1.0, i, b...))
    end
    scales = map(record.r2d2) do spec
        linear = Tuple{Float64,Int}[]
        unit = Tuple{Float64,Float64,Int}[]
        simplex = _AdaptiveSimplexLog[]
        bounded = Tuple{Float64,Int,Float64,Float64}[]
        constant = 0.0
        if haskey(spec, :free)
            scale_atom!(linear, bounded, coordinate(spec.free, "scale"))
        else
            reference = spec.reference
            if reference isa Real
                reference > 0 || _rk_adaptive_refuse(record,
                    "has the reference scale $(reference), which is not positive.")
                constant += log(Float64(reference))
            elseif reference isa Symbol && haskey(pos, String(reference))
                scale_atom!(linear, bounded, pos[String(reference)])
            elseif reference isa Symbol &&
                    get(rk.plan.columns, reference, nothing) isa Real
                constant += log(Float64(rk.plan.columns[reference]))
            else
                _rk_adaptive_refuse(record, "has the reference scale " *
                    "`$(reference)`, which is not a positive constant, a data " *
                    "scalar or a sampled positive scale.")
            end
            r2 = coordinate(spec.r2, "R2")
            bounds[r2] == (0.0, 1.0) || _rk_adaptive_refuse(record, "samples its " *
                "`R2` under the RK transform `$(repr(transforms[r2]))`, not on (0, 1).")
            push!(unit, (0.5, -0.5, r2))
            if spec.n > 1
                phi = [coordinate(Symbol(spec.phi, ".", i), "share") for i in 1:spec.n - 1]
                all(i -> transforms[i] === :simplex, phi) || _rk_adaptive_refuse(record,
                    "samples its shares `$(spec.phi)` under a transform other than " *
                    "RK's simplex.")
                push!(simplex, _AdaptiveSimplexLog(0.5, phi, spec.entry, true))
            end
        end
        _AdaptiveLogScale(constant, linear, unit, simplex, bounded)
    end
    cholesky_free = isnothing(record.L) ? Int[] :
        _rk_adaptive_positions(pos,
            [string(record.L, ".", m) for m in 1:(K * (K - 1)) ÷ 2], record, "Cholesky")
    ranef = RanefBlock(binding, K == 1 ? :ranef_intercept_r2d2 : :ranef_correlated_r2d2,
        record.group, record.id, nothing, levels, K, G, record.z, true, false)
    R2D2AdaptiveCenteringBlock(ranef, 0.0, effects, cholesky_free, collect(scales))
end

# A stratified `gr(g, by=b)` block as one block per stratum of `b`, in RK's
# stratum order (`levels(b)`), as `_adaptive_stratified_blocks` does on SBBRMI.
# Stratum `s` samples its own scales (`tau.<s>.<k>`, or `tau_<k>.<s>` when the
# margins' laws differ) and factor (`L.<m>.<s>`), and only its groups' draws
# `z.<g>.<k>` read them, so it is exactly an ordinary block over those groups.
# A group must lie in one stratum on every row (`_brm_level_strata`).
function _rk_adaptive_stratified_blocks(rk::RKBRMI, record, pos, bounds)
    found = _rk_adaptive_effects(rk, record, pos)
    isnothing(found) && return AbstractAdaptiveCenteringBlock[]
    levels, effects = found
    K = size(effects, 1)
    columns = rk.plan.columns
    strata = _rk_grouping_levels(columns[record.by])
    # Each level's stratum (0 for a level without rows, which stays unmapped).
    stratum = _brm_level_strata(columns[record.group], columns[record.by], levels,
        strata, record; caller="BRM adaptive centering on RKBRMI", fail=error,
        unseen=0)
    out = AbstractAdaptiveCenteringBlock[]
    for s in eachindex(strata)
        members = findall(==(s), stratum)
        isempty(members) && continue
        scale_names = isnothing(record.scales) ?
            [string(record.scale, ".", s, ".", k) for k in 1:K] :
            [string(record.scales[k], ".", s) for k in 1:K]
        scales = Union{Nothing,Int}[
            _rk_adaptive_positions(pos, scale_names, record, "scale")...]
        cholesky_free = isnothing(record.L) ? Int[] :
            _rk_adaptive_positions(pos,
                [string(record.L, ".", m, ".", s) for m in 1:(K * (K - 1)) ÷ 2],
                record, "Cholesky")
        ranef = RanefBlock(_rk_adaptive_binding(record),
            K == 1 ? :rk_ranef_scalar : :rk_ranef_correlated, record.group,
            record.id, record.by, levels[members], K, length(members), record.z,
            true, false)
        push!(out, _rk_adaptive_block(ranef, effects[:, members], cholesky_free,
            scales, ntuple(_ -> nothing, K), nothing, Int[], bounds, record,
            Dict(i => n for (n, i) in zip(scale_names, scales))))
    end
    out
end

# Each layout coordinate's constrained bounds `(lower, upper)` as the adaptive
# scale readers take them (`_adaptive_log_constrain`: `exp` is `(0, Inf)`,
# `lo + exp(u)` is `(lo, Inf)`, `lo + (hi - lo) * logistic(u)` is `(lo, hi)`),
# or `nothing` for a transform that is not a positive scale's (an identity, an
# upper bound alone, a simplex or a factor).
function _rk_adaptive_scale_bounds(rk::RKBRMI, transforms)
    lo, hi = _rk_layout_coordinate_bounds(rk)
    map(eachindex(transforms)) do i
        t = transforms[i]
        t === :exp ? (0.0, Inf) :
        t === :logistic ? (0.0, 1.0) :
        t === :floored ? (isfinite(lo[i]) ? (lo[i], Inf) : nothing) :
        t === :interval ? (isfinite(lo[i]) && isfinite(hi[i]) ? (lo[i], hi[i]) : nothing) :
        t === :bounded && isfinite(lo[i]) ? (lo[i], isfinite(hi[i]) ? hi[i] : Inf) :
        nothing
    end
end

# The positions of coordinates sampled as the log of a positive value.
function _rk_adaptive_exp_positions(pos, transforms, names, record, role)
    out = _rk_adaptive_positions(pos, names, record, role)
    for (i, n) in zip(out, names)
        transforms[i] === :exp || _rk_adaptive_refuse(record, "samples its " *
            "$role `$n` under the RK transform `$(repr(transforms[i]))`, not as " *
            "the log of a positive value.")
    end
    out
end

# One block of the RKBRMI wrapper. A Gaussian block whose sampled scales are
# each `exp` of its coordinate is an ordinary `AdaptiveCenteringBlock`. Every
# other block is a derived-scale `R2D2AdaptiveCenteringBlock`, as on SBBRMI: a
# known SD is the constant `log(value)`, a sampled SD the log of its
# coordinate's constrained value (`exp`, or a bounded transform's), a known
# correlation the constant factor `cholesky(known_cor).L`, and a Student-t
# block's level `g` reads its weight `w[g]` through `mixing[g]`.
function _rk_adaptive_block(ranef, effects, cholesky_free, scales, known, known_cor,
                            mixing, bounds, record, names)
    K = ranef.n_terms
    bound = map(scales) do i
        isnothing(i) && return nothing
        b = bounds[i]
        isnothing(b) && _rk_adaptive_refuse(record, "samples scale " *
            "`$(names[i])` under an RK transform that is not a positive scale's.")
        b
    end
    plain = all(isnothing, known) && isnothing(known_cor) && isempty(mixing) &&
        all(b -> b == (0.0, Inf), bound)
    plain && return AdaptiveCenteringBlock(ranef, 0.0, effects, cholesky_free,
                                           Int[something(i) for i in scales])
    log_scales = map(1:K) do k
        linear = Tuple{Float64,Int}[]
        bounded = Tuple{Float64,Int,Float64,Float64}[]
        constant = 0.0
        if isnothing(scales[k])
            constant = log(Float64(known[k]))
        elseif bound[k] == (0.0, Inf)
            push!(linear, (1.0, scales[k]))
        else
            push!(bounded, (1.0, scales[k], bound[k]...))
        end
        _AdaptiveLogScale(constant, linear, Tuple{Float64,Float64,Int}[],
                          _AdaptiveSimplexLog[], bounded)
    end
    R2D2AdaptiveCenteringBlock(ranef, 0.0, effects, cholesky_free, log_scales;
        cholesky_known=isnothing(known_cor) ? zeros(0, 0) :
            Matrix{Float64}(cholesky(Symmetric(Matrix{Float64}(known_cor))).L),
        mixing)
end

# Margin labels `(; predictor, coefficient)` by block binding, in RK margin
# order: the labels `adaptive_centering_controls` and the locations read.
function _rk_adaptive_margins(rk::RKBRMI)
    Dict{Symbol,Tuple}(_rk_adaptive_binding(r) =>
        Tuple((; predictor=p, coefficient=c) for (p, c) in r.margins)
        for r in _rk_adaptive_records(rk))
end

# Each block's row codes, `1:n_groups` into its levels, when it groups by one
# column whose every row names one of its levels; the rows `_adaptive_containers`
# and the nesting rule compare. A joint `|ID|` block has none.
function _rk_adaptive_group_rows(rk::RKBRMI, blocks)
    rows = Dict{Symbol,AbstractVector}()
    joint = Set{Symbol}(_rk_adaptive_binding(r) for r in _rk_adaptive_records(rk)
                        if haskey(r, :joint_columns))
    for block in blocks
        group = block.ranef.group
        group isa Symbol && haskey(rk.plan.columns, group) || continue
        block.ranef.binding in joint && continue
        lookup = Dict{Any,Int}(l => i for (i, l) in enumerate(block.ranef.levels))
        codes = Int[get(lookup, _brm_transport_level(v), 0)
                    for v in rk.plan.columns[group]]
        !isempty(codes) && extrema(codes) == (1, block.ranef.n_groups) &&
            (rows[block.ranef.binding] = codes)
    end
    rows
end

# The sampled, unbounded population coefficient coordinate of each
# `(predictor, coefficient)`: RK's `:population` records whose coordinate the
# layout keeps under the identity transform (a bounded coefficient's coordinate
# is not its value).
function _rk_adaptive_population(rk::RKBRMI, pos)
    transforms = _rk_layout_coordinate_transforms(rk)
    out = Dict{Tuple{Symbol,Symbol},Int}()
    for r in _rk_coordinate_records(rk.plan)
        r.kind === :population || continue
        index = get(r, :index, ())
        name = isempty(index) ? String(r.declaration) :
            string(r.declaration, ".", join(index, "."))
        i = get(pos, name, 0)
        i == 0 || transforms[i] !== :identity ||
            (out[(r.predictor, r.coefficient)] = i)
    end
    out
end

# The categorical population terms of `predictor` whose level is constant
# within each level of a grouping with row codes `group_rows` (`G` levels), as
# SBBRMI's `_adaptive_factor_levels` reads them: per term, the layout position
# of each grouping level's coefficient, 0 for a level without one (a
# treatment-coded reference). Read from RK's `:population_block` records, whose
# coefficient `<declaration>.<i>` belongs to level `level_values[i]` of the
# term's column; a coefficient the layout does not sample unbounded is skipped.
function _rk_adaptive_factor_levels(rk::RKBRMI, predictor, group_rows, G, pos,
                                    transforms)
    out = Vector{Int}[]
    for r in _rk_coordinate_records(rk.plan)
        r.kind === :population_block && r.predictor === predictor || continue
        level = _rk_adaptive_factor_level(rk, r, group_rows, G, pos, transforms)
        isnothing(level) || push!(out, level)
    end
    out
end

# One `:population_block` record's coefficient position per grouping level, or
# `nothing` when its column does not align with the grouping's rows, a level
# holds rows of two of its levels, or a coefficient is not sampled unbounded.
function _rk_adaptive_factor_level(rk::RKBRMI, r, group_rows, G, pos, transforms)
    column = get(rk.plan.columns, get(r, :column, nothing), nothing)
    column isa AbstractVector && length(column) == length(group_rows) || return nothing
    coefficient = Dict{Any,Int}()
    for (i, level) in enumerate(r.level_values)
        j = get(pos, string(r.declaration, ".", i), 0)
        j != 0 && transforms[j] === :identity || return nothing
        coefficient[_brm_transport_level(level)] = j
    end
    level = fill(-1, G)
    for (g, v) in zip(group_rows, column)
        j = get(coefficient, _brm_transport_level(v), 0)
        level[g] == -1 && (level[g] = j)
        level[g] == j || return nothing
    end
    any(==(-1), level) ? nothing : level
end

# Hierarchical locations of an RKBRMI's blocks: the rule of the SBBRMI method
# (`_adaptive_ranef_locations(model, blocks, unc_names)`), read from RK records.
# A cell of term `k` is centred on its predictor's sampled, unbounded population
# coefficient for that term, plus, for an intercept cell, the coefficient of
# each categorical population term's level holding its level, plus the effect,
# for the same term, of every coarser scalar grouping of the same predictor
# holding its level; a grouping nested in a coarser one that cannot be read as a
# container stays centred on 0.
function _adaptive_ranef_locations(rk::RKBRMI, blocks, unc_names)
    out = IdDict{Any,_AdaptiveRanefLocations}()
    isempty(blocks) && return out
    names = String.(unc_names)
    pos = Dict{String,Int}(n => i for (i, n) in enumerate(names))
    transforms = _rk_layout_coordinate_transforms(rk)
    population = _rk_adaptive_population(rk, pos)
    margins = _rk_adaptive_margins(rk)
    rows = _rk_adaptive_group_rows(rk, blocks)
    nested = _rk_adaptive_nested_bindings(rows, margins)
    for block in blocks
        K = block.ranef.n_terms
        G = block.ranef.n_groups
        binding = block.ranef.binding
        labels = get(margins, binding, ())
        locs = zeros(Int, K)
        factors = nothing
        containers = nothing
        # As on SBBRMI, a multi-membership block (its group a tuple of member
        # columns) stays centred on 0: its rows average several levels.
        if block.ranef.group isa Symbol && length(labels) == K
            for k in 1:K
                predictor, coefficient = labels[k].predictor, labels[k].coefficient
                outer = _adaptive_containers(block, predictor, coefficient, blocks,
                                             margins, rows)
                isnothing(outer) && (binding, predictor) in nested && continue
                locs[k] = get(population, (predictor, coefficient), 0)
                isnothing(outer) && continue
                levels = coefficient === :Intercept ?
                    _rk_adaptive_factor_levels(rk, predictor, rows[binding], G, pos,
                                               transforms) : Vector{Int}[]
                isempty(outer) && isempty(levels) && continue
                if isnothing(containers)
                    factors = [Int[] for _ in 1:K, _ in 1:G]
                    containers = [Tuple{Any,Int,Int}[] for _ in 1:K, _ in 1:G]
                end
                for g in 1:G
                    for level in levels
                        level[g] == 0 || push!(factors[k, g], level[g])
                    end
                    for (container, term, parent) in outer
                        push!(containers[k, g], (container, term, parent[g]))
                    end
                end
            end
        end
        out[block] = _AdaptiveRanefLocations(locs, factors, containers)
    end
    out
end

# `(binding, predictor)` pairs whose grouping nests in a coarser grouping of
# another block on the same predictor (`_adaptive_nested_bindings`, over RK row
# codes).
function _rk_adaptive_nested_bindings(rows, margins)
    by_predictor = Dict{Symbol,Vector{Symbol}}()
    for (binding, labels) in margins, label in labels
        haskey(rows, binding) || continue
        list = get!(by_predictor, label.predictor, Symbol[])
        binding in list || push!(list, binding)
    end
    nested = Set{Tuple{Symbol,Symbol}}()
    for (predictor, bindings) in by_predictor, a in bindings, b in bindings
        a === b && continue
        _adaptive_nested_in(rows[a], rows[b]) && push!(nested, (a, predictor))
    end
    nested
end

# ---- Areal `car` fields and `cdar` walks -----------------------------------
#
# RK emits both as SBBRMI does, so their cells are SBBRMI's block types over RK
# positions and the extension's car and cdar maps apply unchanged. A BYM2 field
# is `sigma .* (sqrt.(rho .* inv_scale) .* phi_full .+ sqrt.(1 .- rho .* is_struct)
# .* theta)` (`_rk_ast_car_block!`), the SBBRMI emission with `inv_scale = 1 / s_c`,
# and an ICAR field `sigma .* phi_full`; a `cdar` walk reads its standard-normal
# innovations `eta[p, w]`, stored column-major.

# The plan term of a `car` or `cdar` component declaration.
function _rk_adaptive_term(rk::RKBRMI, kind, declaration)
    for predictor in rk.plan.predictors, term in predictor.terms
        term.kind === kind && term.options.value === declaration && return term
    end
    error("BRM adaptive centering on RKBRMI: internal: no `$kind` term declares ",
          "`$declaration`.")
end

# One scalar coordinate of a component and its constrained bounds, which must
# be a positive scale's (`lower >= 0`) or, for `unit`, inside `[0, 1]`.
function _rk_adaptive_hyper(pos, bounds, names, declaration, ref, what; unit=false)
    name = string(declaration, ".", ref)
    i = get(pos, name, 0)
    i == 0 && error("BRM adaptive centering on RKBRMI: `$declaration` has no ",
                    "$what coordinate `$name` in the RK layout.")
    b = bounds[i]
    ok = !isnothing(b) && (!unit || (0 <= b[1] && b[2] <= 1))
    ok || error(
        "BRM adaptive centering on RKBRMI: `$declaration`'s $what `$name` is not ",
        "sampled on ", unit ? "a subinterval of [0, 1]" : "a positive scale",
        " by the RK layout.")
    i, b...
end

function _rk_adaptive_car_blocks(rk::RKBRMI, unc_names=_rk_layout_coordinate_names(rk))
    names = String.(unc_names)
    pos = Dict{String,Int}(n => i for (i, n) in enumerate(names))
    bounds = _rk_adaptive_scale_bounds(rk, _rk_layout_coordinate_transforms(rk))
    population = _rk_adaptive_population(rk, pos)
    out = _CARAdaptiveCenteringBlock[]
    for r in _rk_coordinate_records(rk.plan)
        r.kind === :car || continue
        r.separation === nothing || error(
            "BRM adaptive centering on RKBRMI: car field `$(r.declaration)` is ",
            "separated into independent parts (`car_units`); its cells are not ",
            "built on RKBRMI yet. Sample it without adaptive centering, or ",
            "without `car_units`.")
        o = _rk_adaptive_term(rk, :car, r.declaration).options
        bym2 = r.type === :bym2
        S = o.n_struct
        struct_pos = rk.plan.columns[o.struct_pos]
        struct_idx = Int[p <= S ? p : 0 for p in struct_pos]
        phi_names = [string(r.declaration, ".", r.locals.phi, ".", j) for j in 1:S]
        # A field no observed likelihood reaches has no sampler coordinate.
        all(n -> !haskey(pos, n), phi_names) && continue
        structured = [get(pos, n, 0) for n in phi_names]
        any(iszero, structured) && error(
            "BRM adaptive centering on RKBRMI: car field `$(r.declaration)` lacks ",
            "structured coordinates in the RK layout.")
        area_of = zeros(Int, S)
        for (k, j) in enumerate(struct_idx)
            j > 0 && (area_of[j] = k)
        end
        unstructured = bym2 ?
            [pos[string(r.declaration, ".", r.locals.theta, ".", k)] for k in eachindex(struct_pos)] :
            Int[]
        log_inv_scale = bym2 ? log.(Float64.(rk.plan.columns[o.inv_scale][area_of])) :
            zeros(S)
        sigma = _rk_adaptive_hyper(pos, bounds, names, r.declaration, r.locals.sigma,
                                   "marginal SD")
        rho = bym2 ? _rk_adaptive_hyper(pos, bounds, names, r.declaration, r.locals.rho,
                                        "mixing share"; unit=true) : (0, 0.0, 1.0)
        push!(out, _CARAdaptiveCenteringBlock(r.predictor, r.declaration, bym2,
            structured, unstructured, struct_idx, area_of, log_inv_scale,
            sigma..., rho...,
            bym2 ? get(population, (r.predictor, :Intercept), 0) : 0,
            collect(Any, o.nodes)))
    end
    out
end

function _rk_adaptive_cdar_blocks(rk::RKBRMI, unc_names=_rk_layout_coordinate_names(rk))
    names = String.(unc_names)
    pos = Dict{String,Int}(n => i for (i, n) in enumerate(names))
    bounds = _rk_adaptive_scale_bounds(rk, _rk_layout_coordinate_transforms(rk))
    out = _CDARAdaptiveCenteringBlock[]
    for r in _rk_coordinate_records(rk.plan)
        r.kind === :cdar || continue
        P, W = length(r.levels), length(r.steps)
        eta_names = [string(r.declaration, ".", r.locals.eta, ".", p, ".", w)
                     for p in 1:P, w in 1:W]
        all(n -> !haskey(pos, n), eta_names) && continue
        effects = [get(pos, n, 0) for n in vec(eta_names)]
        any(iszero, effects) && error(
            "BRM adaptive centering on RKBRMI: cdar walk `$(r.declaration)` lacks ",
            "innovation coordinates in the RK layout.")
        sigma = _rk_adaptive_hyper(pos, bounds, names, r.declaration, r.locals.sigma,
                                   "marginal SD")
        rho = _rk_adaptive_hyper(pos, bounds, names, r.declaration, r.locals.rho,
                                 "persistence"; unit=true)
        L = r.cor_factor
        cdiag = L === nothing ? ones(P) : [sum(abs2, view(L, p, :)) for p in 1:P]
        push!(out, _CDARAdaptiveCenteringBlock(r.predictor, r.declaration, P, W, effects,
            sigma..., rho..., cdiag))
    end
    out
end
