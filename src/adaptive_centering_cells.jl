# src/adaptive_centering_cells.jl — the public description of adaptive
# centring's random-effect cells (snag public-rkbrmi-ad-4eeb6b28, decision
# `BayesianRegressionModels:snag.public-rkbrmi-ad-4eeb6b28/decisions/2026-11-15T08-00-53-597-0e4u3mv`):
# what each cell is centred on and the order the WarmupHMC wrapper maps cells
# in, as plain data, so a caller outside BRM can rebuild that wrapper's exact
# triangular map (ReactiveKernelMethods emits it for JAX). The wrapper itself
# (ext/BayesianRegressionModelsWarmupHMCExt.jl) builds its pairs from the same
# blocks, locations and order, which `test/adaptive_centering_cells.jl` checks.

# Containing blocks first (snag brm-zone-8-adapt-1b407672): a cell centred on a
# containing cell's effect reads that cell's model value, which WarmupHMC hands
# an accessor only for EARLIER pairs, so the containing pairs must come earlier
# (and the learned-order inverse rebuilds them first). A block sits after every
# block it is centred on: it has strictly more containing blocks than any of
# them (containment is transitive). The sort is stable, so a model without
# containing cells keeps its pair order. The WarmupHMC extension's
# `_containers_first!` is this rule.
function _adaptive_containers_first!(blocks, locations)
    count_containers(block) = begin
        l = get(locations, block, nothing)
        l isa _AdaptiveRanefLocations && !isnothing(l.containers) ?
            length(unique(objectid(first(c)) for cs in l.containers for c in cs)) : 0
    end
    sort!(blocks; by=count_containers, alg=Base.Sort.MergeSort)
end

# The random-effect blocks of `model` in the wrapper's pair order, and their
# locations: ordinary blocks first, then derived-scale blocks
# (`R2D2AdaptiveCenteringBlock`), each family containers-first. Every wrapper
# that maps random-effect cells (the per-pair one, the joint one, the learned
# order) uses this order.
function _adaptive_cell_blocks(model, names)
    blocks = adaptive_centering_blocks(model, names)
    locations = _adaptive_ranef_locations(model, blocks, names)
    ordinary = AdaptiveCenteringBlock[b for b in blocks if b isa AdaptiveCenteringBlock]
    derived = R2D2AdaptiveCenteringBlock[
        b for b in blocks if b isa R2D2AdaptiveCenteringBlock]
    _adaptive_containers_first!(ordinary, locations)
    _adaptive_containers_first!(derived, locations)
    vcat(AbstractAdaptiveCenteringBlock[], ordinary, derived), locations
end

# Margin labels `(; predictor, coefficient)` by block binding.
_adaptive_cell_margins(rk::RKBRMI) = _rk_adaptive_margins(rk)
_adaptive_cell_margins(model) = _adaptive_block_margins(
    model isa BRMDescriptor ? model : brm_descriptor(model))

# A level label as plain data: a categorical value unwrapped, a joint `|ID|`
# block's tuple label element by element.
_adaptive_cell_level(x) = x isa Tuple ? map(_adaptive_cell_level, x) :
    _brm_transport_level(x)

const _AdaptiveCellLinear = NamedTuple{(:weight, :coordinate),Tuple{Float64,Int}}
const _AdaptiveCellUnit = NamedTuple{(:log_weight, :log1m_weight, :coordinate),
                                     Tuple{Float64,Float64,Int}}
const _AdaptiveCellSimplex = NamedTuple{(:weight, :coordinates, :entry, :transform),
                                        Tuple{Float64,Vector{Int},Int,Symbol}}
const _AdaptiveCellBounded = NamedTuple{(:weight, :coordinate, :lower, :upper),
                                        Tuple{Float64,Int,Float64,Float64}}

# Term `k`'s log scale as atoms, the inspectable `_AdaptiveLogScale` form.
_adaptive_cell_log_scale(s::_AdaptiveLogScale) = (;
    constant = s.constant,
    linear = _AdaptiveCellLinear[(; weight=w, coordinate=i) for (w, i) in s.linear],
    unit = _AdaptiveCellUnit[(; log_weight=a, log1m_weight=b, coordinate=i)
                             for (a, b, i) in s.unit],
    simplex = _AdaptiveCellSimplex[(; weight=a.weight, coordinates=copy(a.coordinates),
                                     entry=a.entry,
                                     transform=a.stick ? :stick_breaking : :stan)
                                   for a in s.simplex],
    bounded = _AdaptiveCellBounded[(; weight=w, coordinate=i, lower, upper)
                                   for (w, i, lower, upper) in s.bounded])

# An ordinary block's term scale is its coordinate itself: the one atom `1 * x[i]`
# (the joint wrapper reads it so, `_adaptive_derived_block`).
_adaptive_cell_log_scales(b::AdaptiveCenteringBlock) =
    [_adaptive_cell_log_scale(_AdaptiveLogScale(0.0, [(1.0, i)],
        Tuple{Float64,Float64,Int}[], _AdaptiveSimplexLog[])) for i in b.log_scales]
_adaptive_cell_log_scales(b::R2D2AdaptiveCenteringBlock) =
    [_adaptive_cell_log_scale(s) for s in b.scales]

_adaptive_cell_cpc(::AdaptiveCenteringBlock) = Float64[]
_adaptive_cell_cpc(b::R2D2AdaptiveCenteringBlock) = copy(b.cholesky_cpc)
_adaptive_cell_mixing(::AdaptiveCenteringBlock) = Int[]
_adaptive_cell_mixing(b::R2D2AdaptiveCenteringBlock) = copy(b.mixing)

# The cell families an SBBRMI wrapper adapts beside random effects, which these
# rows do not describe yet (a follow-up of decision 0e4u3mv adds them in the same
# schema): each has its own scale and location form.
function _adaptive_cells_other_families(model, names)
    out = String[]
    isempty(_adaptive_hsgp_level_centering_blocks(model, names)) ||
        push!(out, "grouped HSGP level cells")
    isempty(_total_blocks_by_placement(model).sampled) ||
        push!(out, "exact total coefficients")
    isempty(s2z_effect_blocks(model)) || push!(out, "S2Z contrasts")
    out
end

# A model's `car` fields and `cdar` walks, as the wrapper builds them: an
# SBBRMI's always, an RKBRMI's under `fields = true` (its default passes them
# through).
_adaptive_cell_car_blocks(rk::RKBRMI, names, fields) =
    fields ? _rk_adaptive_car_blocks(rk, names) : _CARAdaptiveCenteringBlock[]
_adaptive_cell_car_blocks(model, names, _) = _adaptive_car_centering_blocks(model, names)
_adaptive_cell_cdar_blocks(rk::RKBRMI, names, fields) =
    fields ? _rk_adaptive_cdar_blocks(rk, names) : _CDARAdaptiveCenteringBlock[]
_adaptive_cell_cdar_blocks(model, names, _) = _adaptive_cdar_centering_blocks(model, names)

# A model's HSGP basis-weight blocks, as the wrapper builds them: an SBBRMI's
# always, an RKBRMI's unless `hsgp = false`.
_adaptive_cell_hsgp_blocks(rk::RKBRMI, names, hsgp) =
    hsgp ? _rk_adaptive_hsgp_blocks(rk, names) : _HSGPAdaptiveCenteringBlock[]
_adaptive_cell_hsgp_blocks(model, names, _) = _adaptive_hsgp_centering_blocks(model, names)

# A hyperparameter coordinate and its declared bounds (`_adaptive_constrain`).
_adaptive_cell_hyper(coordinate, lower, upper) = (; coordinate, lower, upper)

function _adaptive_centering_cells(model, names::Vector{String}; fields=false, hsgp=true)
    ordered, locations = _adaptive_cell_blocks(model, names)
    margins = _adaptive_cell_margins(model)
    lookup = IdDict{Any,Matrix{Int}}()
    p = 0
    for b in ordered
        lookup[b] = collect(reshape((p + 1):(p + b.ranef.n_terms * b.ranef.n_groups),
                                    b.ranef.n_terms, b.ranef.n_groups))
        p += b.ranef.n_terms * b.ranef.n_groups
    end
    effects = BitSet(Iterators.flatten(vec(b.effects) for b in ordered))
    blocks = NamedTuple[]
    cells = NamedTuple[]
    for (bi, b) in enumerate(ordered)
        r = b.ranef
        K, G = r.n_terms, r.n_groups
        labels = get(margins, r.binding, ())
        margin = [length(labels) == K ? labels[k] : nothing for k in 1:K]
        push!(blocks, (; block=bi, family=:ranef, binding=r.binding, group=r.group,
            id=r.id, by=r.by,
            levels=[_adaptive_cell_level(l) for l in r.levels], n_terms=K, n_levels=G,
            margins=margin, target=b.target_c, log_scales=_adaptive_cell_log_scales(b),
            correlation=(; free=copy(b.cholesky_free), cpc=_adaptive_cell_cpc(b)),
            mixing=_adaptive_cell_mixing(b)))
        locs = get(locations, b, nothing)
        for g in 1:G, k in 1:K
            q = lookup[b][k, g]
            coefficient = isnothing(locs) ? 0 : locs.coefficient[k]
            factors = isnothing(locs) || isnothing(locs.factors) ? Int[] :
                copy(locs.factors[k, g])
            containers = Int[]
            if !isnothing(locs) && !isnothing(locs.containers)
                for (container, term, level) in locs.containers[k, g]
                    cl = get(lookup, container, nothing)
                    isnothing(cl) && error(
                        "BRM adaptive centering cells: block `$(r.binding)` is centred " *
                        "on the effects of block `$(container.ranef.binding)`, which " *
                        "the wrapper does not map; refusing an inexact description.")
                    a = cl[term, level]
                    a < q || error(
                        "BRM adaptive centering cells: block `$(r.binding)` is centred " *
                        "on the effects of block `$(container.ranef.binding)`, which " *
                        "is mapped after it; containing blocks must come first.")
                    push!(containers, a)
                end
            end
            for j in Iterators.flatten((coefficient, factors))
                j == 0 || !(j in effects) || error(
                    "BRM adaptive centering cells: a random-effect cell is centred " *
                    "on coordinate $j, which another cell rewrites; refusing an " *
                    "inexact description.")
            end
            push!(cells, (; cell=q, family=:ranef, block=bi, binding=r.binding,
                group=r.group, level=_adaptive_cell_level(r.levels[g]), level_index=g,
                term=k, margin=margin[k], effect=b.effects[k, g],
                name=names[b.effects[k, g]],
                conditioning=[lookup[b][l, g] for l in 1:k-1],
                location=(; coefficient, factors, containers)))
        end
    end
    # HSGP basis-weight cells next (`_joint_centering_state`): term by term, a
    # grouped term level by level, each block basis by basis
    # (`BRMHSGPAdaptiveCenteringState`). A cell's compiled frame is its own:
    # an SBBRMI term may fix a different exponent per basis.
    term_levels = Dict{Tuple{Symbol,Symbol},Int}()
    for b in _adaptive_cell_hsgp_blocks(model, names, hsgp)
        bi = length(blocks) + 1
        key = (b.logical, b.term)
        level_index = term_levels[key] = get(term_levels, key, 0) + 1
        spectrum = isempty(b.harmonics) ? :exp_quad : :periodic
        sigma = _adaptive_cell_hyper(b.sd, b.sd_lower, b.sd_upper)
        length_scales = [_adaptive_cell_hyper(i, lo, hi) for (i, lo, hi) in
                         zip(b.length_scales, b.length_scale_lower, b.length_scale_upper)]
        push!(blocks, (; block=bi, family=:hsgp, binding=b.logical, term=b.term,
            level_index, n_basis=length(b.effects), spectrum, target=copy(b.target_c)))
        for k in eachindex(b.effects)
            push!(cells, (; cell=length(cells) + 1, family=:hsgp, block=bi,
                binding=b.logical, margin=(; predictor=b.logical, coefficient=b.term),
                level=nothing, level_index, term=k, effect=b.effects[k],
                name=names[b.effects[k]], target=b.target_c[k],
                scale=(; spectrum, sigma, length_scales,
                         omega2=spectrum === :exp_quad ? b.omega2[k, :] : Float64[],
                         harmonic=spectrum === :periodic ? b.harmonics[k] : 0.0),
                location=(;)))
        end
    end
    # The joint wrapper's later families, in its pair order: `car` cells, then
    # `cdar` cells (`_joint_centering_state`). Both are compiled non-centred.
    for b in _adaptive_cell_car_blocks(model, names, fields)
        bi = length(blocks) + 1
        push!(blocks, (; block=bi, family=:car, binding=b.logical, term=b.term,
            type=b.bym2 ? :bym2 : :icar, areas=[_adaptive_cell_level(a) for a in b.areas],
            target=0.0))
        for k in 1:_adaptive_car_n_cells(b)
            c = _adaptive_car_cell(b, k)
            area = _adaptive_car_area(b, k)
            push!(cells, (; cell=length(cells) + 1, family=:car, block=bi,
                binding=b.logical, margin=(; predictor=b.logical, coefficient=b.term),
                level=_adaptive_cell_level(b.areas[area]),
                level_index=area, term=b.bym2 ? 2 : 1, effect=c.index,
                name=names[c.index],
                scale=(; sigma=_adaptive_cell_hyper(c.sigma, c.sigma_lower, c.sigma_upper),
                         rho=_adaptive_cell_hyper(c.rho, c.rho_lower, c.rho_upper),
                         share=c.share),
                location=(; intercept=c.intercept, phi=c.phi,
                            log_inv_scale=c.log_inv_scale)))
        end
    end
    for b in _adaptive_cell_cdar_blocks(model, names, fields)
        bi = length(blocks) + 1
        push!(blocks, (; block=bi, family=:cdar, binding=b.logical, term=b.term,
            n_groups=b.n_groups, n_steps=b.n_steps, target=0.0))
        for pair in eachindex(b.effects)
            group = mod(pair - 1, b.n_groups) + 1
            step = div(pair - 1, b.n_groups) + 1
            push!(cells, (; cell=length(cells) + 1, family=:cdar, block=bi,
                binding=b.logical, margin=(; predictor=b.logical, coefficient=b.term),
                level=nothing, level_index=group, term=pair,
                effect=b.effects[pair], name=names[b.effects[pair]],
                scale=(; sigma=_adaptive_cell_hyper(b.sigma, b.sigma_lower, b.sigma_upper),
                         rho=_adaptive_cell_hyper(b.rho, b.rho_lower, b.rho_upper),
                         cdiag=b.cdiag[group], step),
                location=(;)))
        end
    end
    (; blocks=identity.(blocks), cells=identity.(cells))
end

"""
    adaptive_centering_cells(rk::RKBRMI, unc_names=<the RK layout names>; hsgp=true, hsgp_levels=false, fields=false) -> (; blocks, cells)
    adaptive_centering_cells(model, unc_names) -> (; blocks, cells)

The cells [`adaptive_centering_problem`](@ref) adapts for `model` (an
[`RKBRMI`](@ref), `SBBRMI` or `GenerativePlan`): its random-effect cells, its
HSGP basis-weight cells (an RKBRMI's unless `hsgp = false`), and its areal
`car` field cells and `cdar` walk cells (an RKBRMI's under `fields = true`),
each set as its wrapper adapts it, described as plain data:
numbers, symbols, strings, `nothing`, vectors and named tuples. Indices are
positions of the unconstrained vector `unc_names` names: for an `RKBRMI`, by
default, `rk_logdensity_problem(rk)`'s layout; for an `SBBRMI`, BridgeStan's
`param_unc_names`. Together the rows define the wrapper's map exactly (below),
so a caller outside BRM can rebuild it without BRM internals. Reading them
needs no WarmupHMC and builds no wrapper.

`cells[p]`, in the wrapper's cell order (the order of `centeredness`,
[`adaptive_centering_controls`](@ref) and checkpoints: random-effect cells of
ordinary blocks, then of derived-scale blocks, each containing blocks first,
then level, then term; then HSGP cells; then `car` cells; then `cdar` cells). A
random-effect cell (`family = :ranef`) has:

- `cell` (`== p`), `family` (`:ranef`), `block` (its row in `blocks`),
  `binding`, `group`, `level` (the level's label) and `level_index`, `term` (its
  margin's declared position), `margin` (`(; predictor, coefficient)`, or
  `nothing` when the block has no margin labels);
- `effect`, the coordinate of the cell's compiled value, and `name`, its name;
- `conditioning`: the cells of the same level's earlier terms, in term order;
- `location`: `coefficient`, the coordinate of the population coefficient the
  cell is centred on (`0` for none); `factors`, the coordinates of the
  `factor(f)` levels holding its level; `containers`, the cells of the coarser
  groupings holding its level, coarsest first, every one earlier than `p`.

An HSGP cell (`family = :hsgp`, one per basis weight: term by term, a grouped
`by = g` term level by level, then basis by basis) has `cell`, `family`,
`block`, `binding` (the predictor), `margin` (`(; predictor, coefficient)`, the
coefficient being the term), `level` (`nothing`) and `level_index` (the level's
position in the grouping's level order, 1 for an ungrouped term), `term` (the
basis function), `effect`, `name`, `target` (the weight's compiled frame: 0 on
an RKBRMI; an SBBRMI term may compile each basis at its own `centeredness`),
`scale = (; spectrum, sigma, length_scales, omega2, harmonic)` and
`location = (;)`. `spectrum` is `:exp_quad` or `:periodic`; `sigma` and each of
`length_scales` are `(; coordinate, lower, upper)`, `coordinate == 0` marking a
known value (`lower == upper`); `omega2` holds the basis function's squared
frequency per axis (`:exp_quad`), and `harmonic` its harmonic (`:periodic`).

A `car` cell (`family = :car`, one per area; an `:icar` field's cells in its
emitted `phi` order, a `:bym2` field's in sorted area order) has `cell`,
`family`, `block`, `binding` (the predictor), `margin` (`(; predictor,
coefficient)`, the coefficient being the term), `level` (the area's label) and
`level_index`, `term` (1 for an `:icar` area's `phi`, 2 for a `:bym2` area's
`theta`), `effect`, `name`, `scale = (; sigma, rho, share)` (`sigma` and `rho`
each `(; coordinate, lower, upper)`; `rho.coordinate == 0` for `:icar`) and
`location = (; intercept, phi, log_inv_scale)`.

A `cdar` cell (`family = :cdar`, one per group × step innovation, group
fastest) has `cell`, `family`, `block`, `binding`, `margin`, `level` (`nothing`),
`level_index` (the group), `term` (the innovation's position), `effect`,
`name`, `scale = (; sigma, rho, cdiag, step)` and `location = (;)`.

`blocks[b]` of a random-effect block: `block`, `family` (`:ranef`), `binding`,
`group`, `id`, `by`, `levels`, `n_terms`,
`n_levels`, `margins` (per term), `target` (the compiled frame's control: 0 for
a block emitted non-centred, every RKBRMI block; 1 for an SBBRMI block in
`centered_groups`), `log_scales[k]` (term `k`'s log scale as atoms `(; constant,
linear, unit, simplex, bounded)`; an ordinary block's is the single linear atom
`1.0 * x[i]` of its scale coordinate), `correlation` (`(; free, cpc)`: the
coordinates of Stan's `cholesky_factor_corr` free values, or the constant
canonical partial correlations of a known correlation; both empty for one term)
and `mixing` (per level, the coordinate of a Student-t level's `log w`, or empty).
An HSGP block (one per term, per level of a grouped term): `block`, `family`
(`:hsgp`), `binding` (the predictor), `term`, `level_index`, `n_basis`,
`spectrum` and `target` (per basis function).
A `car` block: `block`, `family` (`:car`), `binding` (the predictor), `term`,
`type` (`:icar` or `:bym2`), `areas`, `target` (0). A `cdar` block: `block`,
`family` (`:cdar`), `binding`, `term`, `n_groups`, `n_steps`, `target` (0).

# The map

Read `x` as the sampler's unconstrained vector and map cells in order; a
coordinate no cell names passes through unchanged. For cell `p` (block `b`,
term `k`, level `g`, control `c`, target `t`):

- `log τ_i = constant + Σ weight * x[coordinate]` (linear atoms)
  `+ Σ (log_weight * log σ(x[i]) + log1m_weight * log(1 - σ(x[i])))` (unit;
  `σ` the logistic) `+ Σ weight * log(simplex_constrain(x[coordinates])[entry])`
  (simplex; `transform` names the simplex transform: `:stan`, Stan's isometric
  log-ratio and softmax, for an SBBRMI; `:stick_breaking`, ReactiveKernelsPPL's,
  for an RKBRMI's R2D2 shares, whose `N` coordinates `u` give break fractions
  `z_i = σ(u_i + log(N + 1 - i))`, entry `j ≤ N` the value
  `z_j * Π_{i<j} (1 - z_i)` and entry `N + 1` the remaining stick)
  `+ Σ weight * log(lower + (upper - lower) * σ(x[i]))`
  (bounded; `log(exp(x[i]) + lower)` when `upper` is infinite);
- `L` is Stan's `cholesky_factor_corr` of the canonical partial correlations
  `tanh(x[free[j]])` (or `cpc[j]`), row `i` holding `j = (i-1)(i-2)/2 + 1, …`;
  `C = r_g * diag(τ) * L` with `r_g = exp(x[mixing[g]] / 2)` (1 without
  mixing), and `s = C[k, k]`;
- the earlier terms' innovations `ζ_l = (y_l - t * loc_l) / s_l^t` over
  `conditioning` (`y_l` their compiled values; `ζ_l = y_l` when `t = 0`), and
  `m = Σ_l C[k, l] * ζ_l`;
- `loc = m + x[coefficient] + Σ x[factors] + Σ_a e_a`, a containing cell `a`'s
  effect `e_a = s_a * y_a` in a block with `target = 0` and `y_a` otherwise;
- the compiled value `y = t * loc + (x[effect] - c * loc) * s^(t - c)`, and the
  log-Jacobian adds `(t - c) * log s`.

An HSGP cell is centred on 0 at its spectral scale `s`: with target `t`, the
compiled value `y = x[effect] * s^(t - c)`, and the log-Jacobian adds
`(t - c) * log s`. Write `σ` and `ρ_j` for the constrained `sigma` and
`length_scales[j]` (below; a known value is its `lower`), and
`ℓ = log σ + Σ_a (log(ρ_a * sqrt(2π)) / 2 - ρ_a^2 * omega2[a] / 4)` over the axes
`a` of `omega2` for `:exp_quad` (`ρ_a = ρ_1` for every axis when
`length_scales` holds one), or
`ℓ = log σ + (log 2 + log(besselix(harmonic, 1 / ρ_1^2))) / 2` for `:periodic`
(`besselix(ν, z) = I_ν(z) * exp(-z)`, the scaled modified Bessel function).
Then `log s = ℓ` while `ℓ > -110`, and `-150 + log(1 + exp(ℓ + 150))` below,
a smooth floor that keeps `s^(t - c)` a normal Float64 at every control.

A `car` or `cdar` cell is a scalar cell with target 0: with its location `loc`
and scale `s`, `y = (x[effect] - c * loc) * s^(-c)` and the log-Jacobian adds
`-c * log s`. Write `σ = constrain(x[sigma.coordinate])` and
`ρ = constrain(x[rho.coordinate])`, where `constrain(v) = lower + exp(v)` for an
infinite `upper` and `lower + (upper - lower) * σ(v)` otherwise:

- `car`: `log s = log σ + share * log(1 - ρ) / 2`, and `loc` is
  `x[intercept]` (0 when `intercept == 0`) plus, when `phi != 0`,
  `σ * sqrt(ρ * exp(log_inv_scale)) * x[phi]`;
- `cdar`: `loc = 0` and `log s = log σ + log(cdiag) / 2 + log(v) / 2` with the
  walk's spread `v = (1 - ρ^(2 step)) / (1 - ρ^2)`, or `v = step` where
  `1 - ρ^2 <= 1e-12`.

These cells read only coordinates no cell rewrites.

Where `s` leaves Float64's normal range BRM sums `log s` as `log τ_k`, the logs
of the stick factors of row `k` and `log r_g`, which is finite wherever the
coordinates are. `ranef_order = :learned` permutes, at restarting windows, the
order of a correlated block's terms within a level; these rows describe the
declared order, the default.

A model whose wrapper also adapts grouped HSGP level cells (`cor =`, or
per-level scales; an SBBRMI's always, an RKBRMI's under `hsgp_levels = true`), or
an `SBBRMI`'s exact-total or S2Z cells, raises, naming them: their cells have their
own scale and location forms and are not described here yet. Under an RKBRMI's
default `hsgp_levels = false` such a term keeps its emitted frame, so it has no
cells. Blocks the wrapper does not build raise as in
[`adaptive_centering_blocks`](@ref).
"""
function adaptive_centering_cells end

function adaptive_centering_cells(rk::RKBRMI, unc_names=_rk_layout_coordinate_names(rk);
                                  hsgp=true, hsgp_levels=false, fields=false)
    names = String.(unc_names)
    _adaptive_cells_refuse(hsgp_levels && !isempty(_rk_adaptive_hsgp_level_blocks(rk, names)) ?
        ["grouped HSGP level cells"] : String[])
    _adaptive_centering_cells(rk, names; fields, hsgp)
end

function adaptive_centering_cells(model, unc_names)
    names = String.(unc_names)
    _adaptive_cells_refuse(_adaptive_cells_other_families(model, names))
    _adaptive_centering_cells(model, names)
end

_adaptive_cells_refuse(other) = isempty(other) || error(
    "adaptive_centering_cells: this model's adaptive centring also adapts ",
    join(other, ", "), ", whose cells are not described by these rows yet; ",
    "random-effect, HSGP basis-weight, car and cdar cells are.")
