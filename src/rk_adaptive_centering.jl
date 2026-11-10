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

# The RK random-effect records adaptive centring reads, in emission order.
_rk_adaptive_records(rk::RKBRMI) =
    [r for r in _rk_coordinate_records(rk.plan)
     if r.kind === :ranef || r.kind === :stratified_ranef]

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

The ordinary random-effect blocks of an [`RKBRMI`](@ref) as
[`AdaptiveCenteringBlock`](@ref)s whose indices are positions of
`rk_logdensity_problem(rk; ad_backend)`'s unconstrained vector. Every block is
compiled non-centred (`target_c = 0`): `effects[k, g]` is the standardized draw
of term `k`, level `g`, `log_scales` the block's scale coordinates (each the log
of its scale) and `cholesky_free` its correlation factor's free coordinates.
`ranef.binding` is the block's RK draws declaration (`b_g` of `b_g.z.1.1`),
`ranef.levels` its grouping's levels in RK's row order of `z`.

Covered: Gaussian blocks with sampled scales over one grouping column or a
shared `|ID|` bucket's joint columns, scalar or correlated, `||` blocks
included. A Student-t block, a block holding a scale or its correlation at a
known value, a bounded scale (`Uniform`, `truncated`), a stratified
`gr(g, by=b)` block and a block whose levels are correlated (`gr(g; cor=...)`)
raise. Multi-membership and R2D2-scaled blocks, HSGP weights, `car`, `cdar` and
`gmrf` fields, cell latents and completions keep the emitted frame: they are
not cells of this wrapper, and their coordinates pass through unchanged.
"""
function adaptive_centering_blocks(rk::RKBRMI,
        unc_names=_rk_layout_coordinate_names(rk))
    names = String.(unc_names)
    pos = Dict{String,Int}(n => i for (i, n) in enumerate(names))
    transforms = _rk_layout_coordinate_transforms(rk)
    floors = _rk_layout_coordinate_floors(rk)
    length(transforms) == length(names) || error(
        "BRM adaptive centering on RKBRMI: `unc_names` has $(length(names)) ",
        "entries for a layout of $(length(transforms)) coordinates.")
    out = AbstractAdaptiveCenteringBlock[]
    for record in _rk_adaptive_records(rk)
        record.kind === :stratified_ranef && _rk_adaptive_refuse(record,
            "is stratified (`gr(g, by=$(record.by))`); its per-stratum frames " *
            "are not built on RKBRMI yet.")
        haskey(record, :level_correlation) && _rk_adaptive_refuse(record,
            "correlates the levels of its grouping (`gr(g; cor=...)`); per-level " *
            "centring of such a block is not implemented on either backend.")
        haskey(record, :mixing) && _rk_adaptive_refuse(record,
            "is Student-t (`ranef(:, ID) ~ TDist(nu)`); its mixing weights are " *
            "not read by the RKBRMI wrapper yet.")
        haskey(record, :known_scales) && _rk_adaptive_refuse(record,
            "holds a scale or its correlation at a known value; known-value " *
            "blocks are not built on RKBRMI yet.")
        K = length(record.margins)
        levels = _rk_ranef_record_levels(rk, record)
        G = length(levels)
        z = String(record.z)
        z_names = [string(z, ".", g, ".", k) for k in 1:K, g in 1:G]
        # A block no observed likelihood reaches has no sampler coordinate.
        all(n -> !haskey(pos, n), z_names) && continue
        effects = reshape(_rk_adaptive_positions(pos, vec(z_names), record,
            "standardized-draw"), K, G)
        scale_names = isnothing(record.scales) ?
            [string(record.scale, ".", k) for k in 1:K] : String.(collect(record.scales))
        log_scales = _rk_adaptive_positions(pos, scale_names, record, "scale")
        for i in log_scales
            transforms[i] === :exp ||
                (transforms[i] === :floored && iszero(floors[i])) ||
                _rk_adaptive_refuse(record, "samples scale `$(names[i])` " *
                    "under the RK transform `$(repr(transforms[i]))`; only a " *
                    "scale sampled as `exp` of its coordinate (a zero lower " *
                    "bound, unbounded above) is read by the RKBRMI wrapper yet.")
        end
        cholesky_free = isnothing(record.L) ? Int[] :
            _rk_adaptive_positions(pos,
                [string(record.L, ".", m) for m in 1:(K * (K - 1)) ÷ 2],
                record, "Cholesky")
        ranef = RanefBlock(_rk_adaptive_binding(record),
            K == 1 ? :rk_ranef_scalar : :rk_ranef_correlated, record.group,
            record.id, nothing, levels, K, G, record.z, true, false)
        push!(out, AdaptiveCenteringBlock(ranef, 0.0, effects, cholesky_free,
                                          log_scales))
    end
    claimed = Int[]
    for block in out
        append!(claimed, vec(block.effects), block.cholesky_free, block.log_scales)
    end
    allunique(claimed) || error(
        "BRM adaptive centering on RKBRMI: random-effect blocks claim overlapping ",
        "layout coordinates; refusing an ambiguous transform.")
    out
end

# Margin labels `(; predictor, coefficient)` by block binding, in RK margin
# order: the labels `adaptive_centering_controls` and the locations read.
function _rk_adaptive_margins(rk::RKBRMI)
    Dict{Symbol,Tuple}(_rk_adaptive_binding(r) =>
        Tuple((; predictor=p, coefficient=c) for (p, c) in r.margins)
        for r in _rk_adaptive_records(rk) if r.kind === :ranef)
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

# Hierarchical locations of an RKBRMI's blocks: the rule of the SBBRMI method
# (`_adaptive_ranef_locations(model, blocks, unc_names)`), read from RK records.
# A cell of term `k` is centred on its predictor's sampled, unbounded population
# coefficient for that term, plus the effect, for the same term, of every coarser
# scalar grouping of the same predictor holding its level; a grouping nested in
# a coarser one that cannot be read as a container stays centred on 0. Levels of
# categorical population terms are not added to an intercept cell's location on
# RKBRMI yet (its location is then the intercept alone, or 0 without one).
function _adaptive_ranef_locations(rk::RKBRMI, blocks, unc_names)
    out = IdDict{Any,_AdaptiveRanefLocations}()
    isempty(blocks) && return out
    names = String.(unc_names)
    pos = Dict{String,Int}(n => i for (i, n) in enumerate(names))
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
        containers = nothing
        if length(labels) == K
            for k in 1:K
                predictor, coefficient = labels[k].predictor, labels[k].coefficient
                outer = _adaptive_containers(block, predictor, coefficient, blocks,
                                             margins, rows)
                isnothing(outer) && (binding, predictor) in nested && continue
                locs[k] = get(population, (predictor, coefficient), 0)
                (isnothing(outer) || isempty(outer)) && continue
                isnothing(containers) &&
                    (containers = [Tuple{Any,Int,Int}[] for _ in 1:K, _ in 1:G])
                for g in 1:G, (container, term, parent) in outer
                    push!(containers[k, g], (container, term, parent[g]))
                end
            end
        end
        out[block] = _AdaptiveRanefLocations(locs,
            isnothing(containers) ? nothing : [Int[] for _ in 1:K, _ in 1:G],
            containers)
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
