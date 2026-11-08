# src/nlme_svi_slices.jl — a factor-level start for minibatch SVI from fits of
# the same declaration on slices of the data (one sweep of expectation
# propagation on partitioned data, Vehtari et al. 2020, in the variational
# form of partitioned VI, Ashman et al. 2022).
#
# Slice s was fitted to p₀(g, z_s) L_s(g, z_s). Its Gaussian factor divided by
# the prior is a Gaussian "site" on the coordinates it shares with the other
# slices, and the full model starts from the prior times the product of the
# sites, coordinate by coordinate in unconstrained space:
#
#     P = P₀ + Σ_s (P_s − P₀),        P m = P₀ m₀ + Σ_s (P_s m_s − P₀ m₀).
#
# The prior is counted once. A unit's factor comes from the one slice that
# holds its rows. Random-effect draws are standardized, so every draw's factor
# (a unit's, and a shared level's sites) is first carried from its slice's
# scales, correlation and mu-referenced coefficients to the combined ones, by
# the map that keeps the effects its data informed. The result is a start,
# never a prior: the full fit's target is the full model, and its optimization
# removes what the product approximates.

# ── Semantic coordinate identity ─────────────────────────────────────────────

# Packed coordinate j's identity across bindings of one declaration. A
# random-effect draw of the units, a nested or a shared grouping is its block,
# margin and level label (the view's subject margins, and the same key for a
# shared grouping's draws, labelled as its layout orders them, which a frozen
# replay takes from its training); any other coordinate is its packed name,
# which the frozen preprocessing below keeps fixed.
function _brm_svi_coordinate_keys(m::BRMNLMEModel)
    view = m.view
    result = Any[(; coordinate=name) for name in view.coordinates]
    for (block, margins) in zip(view.subject_coordinates, view.subject_margins),
            (j, margin) in zip(block, margins)
        result[j] = margin
    end
    position = Dict(name => j for (j, name) in enumerate(view.coordinates))
    for record in _brm_nlme_records(m.backend.plan)
        record.kind in (:ranef, :stratified_ranef) && record.group in view.shared_groups ||
            continue
        levels = _rk_ranef_record_levels(m.backend, record)
        block = _brm_nlme_block_identity(record)
        for (k, (predictor, coefficient)) in enumerate(record.margins),
                (g, level) in enumerate(levels)
            j = get(position, Symbol(record.z, ".", g, ".", k), 0)
            j == 0 || (result[j] = (; block, predictor, coefficient, level))
        end
        mixing = get(record, :mixing, nothing)
        mixing === nothing && continue
        for (g, level) in enumerate(levels)
            j = get(position, Symbol(mixing, ".", g), 0)
            j == 0 || (result[j] = (; block, predictor=:none, coefficient=:mixing, level))
        end
    end
    result
end

# The groupings whose level tables may differ between a slice and the full
# model: the units', the nested ones and every shared grouping whose levels
# reach the packed layout only through its random-effect draws, which the keys
# above name by label. A shared grouping that a population factor or any other
# record reads by position must keep one level table, so it is left out.
function _brm_svi_label_keyed_groups(m::BRMNLMEModel)
    groups = Symbol[m.view.group, keys(m.view.nested_groups)...]
    records = _brm_nlme_records(m.backend.plan)
    regression = _brm_nlme_regression(m.backend.plan)
    for g in m.view.shared_groups
        positional = any(records) do record
            record.kind in (:ranef, :stratified_ranef) && return false
            any(v -> v === g || (v isa Union{Tuple,AbstractVector} && any(x -> x === g, v)),
                values(record))
        end || any(regression.predictors) do predictor
            any(t -> t.kind === :factor && g in t.columns, predictor.terms)
        end
        positional || push!(groups, g)
    end
    Tuple(groups)
end

# The packed coordinates this binding's data reaches: every coordinate an
# observation cell reads, and every coordinate a prior of a reached coordinate
# reads (its hyperparameters, transitively). A slice's factor says nothing
# about the others beyond the prior, so they contribute no site.
function _brm_svi_informed(m::BRMNLMEModel, factors=_rk_svi_factors(m.backend, m.unit))
    n = length(m.unit)
    reached = falses(n)
    queue = Int[]
    reach!(j) = reached[j] || (reached[j] = true; push!(queue, j); nothing)
    foreach(footprint -> foreach(reach!, footprint), m.cell_gradient.footprints)
    owning = [Int[] for _ in 1:n]
    for (e, owned) in enumerate(factors.owned), j in owned
        push!(owning[j], e)
    end
    visited = falses(length(factors.owned))
    head = 1
    while head <= length(queue)
        j = queue[head]
        head += 1
        for e in owning[j]
            visited[e] && continue
            visited[e] = true
            foreach(reach!, factors.support[e])
        end
    end
    reached
end

# What a data-only state records about its global factor beyond the factor
# itself: the semantic key of each global and whether this binding's data
# reached it. A streaming carry keeps a global reached once reached.
function _brm_svi_global_record(m, q, globals, factors; previous=nothing)
    informed = _brm_svi_informed(m, factors)[globals]
    previous === nothing || !hasproperty(previous, :informed) ||
        (informed .|= previous.informed)
    merge(_brm_svi_global_factor(q), (; keys=_brm_svi_coordinate_keys(m)[globals],
        informed=Vector{Bool}(informed)))
end

# ── Sources: what a slice state, or a start, says about each coordinate ─────

# A source's global factor is N(mean, diag(scale²) + ΒΒᵀ); `var` is its
# diagonal, the marginal variances.
struct _BRMSVISource
    keys::Vector{Any}
    mean::Vector{Float64}
    var::Vector{Float64}
    scale::Vector{Float64}
    Β::Matrix{Float64}
    informed::Vector{Bool}
    units::Dict{Any,NamedTuple}
end

function _brm_svi_source(caller, state::BRMSVIState, which)
    # A learned-centering fit's unit factors are over its learned coordinates
    # (src/nlme_svi_centering.jl), whose map to the packed draws moves with the
    # globals, so they are no Gaussian site of those draws.
    hasproperty(state.layout, :centering) && throw(ArgumentError(
        "$caller: $which is a learned-centering fit (centering=LearnedCentering()), whose " *
        "unit factors are over its learned coordinates, not the packed draws a " *
        "combination multiplies; fit the slices without centering"))
    q = state.q
    hasproperty(q, :keys) && hasproperty(q, :informed) &&
        all(u -> hasproperty(u, :semantic), values(state.units)) || throw(ArgumentError(
        "$caller: $which records neither which coordinates its data reached nor the " *
        "semantic identity of its unit coordinates (a BRM without brm_svi_combine wrote " *
        "it); take brm_svi_checkpoint of the fit again"))
    var = exp.(2 .* q.ρ) .+ vec(sum(abs2, q.Β; dims=2))
    units = Dict{Any,NamedTuple}(label => (; keys=collect(Any, u.semantic), M=u.M,
        Λ=_brm_svi_saved_precision(u)) for (label, u) in state.units)
    _BRMSVISource(collect(Any, q.keys), copy(q.μ), var, exp.(q.ρ), Matrix{Float64}(q.Β),
        collect(q.informed), units)
end

# A saved unit factor's precision as a dense matrix: a dense unit's own, or the
# inverse of a diagonal-plus-low-rank unit's covariance diag(exp(2ρ)) + ΒΒᵀ.
_brm_svi_saved_precision(u) = get(u, :family, :dense) === :diagonal_lowrank ?
    Matrix(inv(Symmetric(Diagonal(exp.(2 .* u.ρ)) + u.Β * u.Β'))) : Matrix{Float64}(u.Λ)

# Where each key has a factor: `globals[key]` lists (source, position) of the
# informed global coordinates, `units[key]` lists (source, label, position).
function _brm_svi_source_index(sources)
    globals = Dict{Any,Vector{Tuple{Int,Int}}}()
    units = Dict{Any,Vector{Tuple{Int,Any,Int}}}()
    for (s, source) in enumerate(sources)
        for (k, key) in enumerate(source.keys)
            source.informed[k] && push!(get!(() -> Tuple{Int,Int}[], globals, key), (s, k))
        end
        for (label, unit) in source.units, (k, key) in enumerate(unit.keys)
            push!(get!(() -> Tuple{Int,Any,Int}[], units, key), (s, label, k))
        end
    end
    covariance = Dict{Tuple{Int,Any},Matrix{Float64}}()
    (; sources, globals, units, covariance)
end

_brm_svi_unit_covariance(index, s, label) = get!(index.covariance, (s, label)) do
    inv(cholesky(Symmetric(index.sources[s].units[label].Λ)))
end

# A key holding a level label names its coordinate in any binding. A packed
# name does so only inside the unit that owns it: an entry of a vector indexed
# by the units sits at another position, under another name, in another
# binding, so the same name in another unit can be another unit's entry.
_brm_svi_label_key(key) = hasproperty(key, :level)

# Every marginal Gaussian the sources hold for global `key`: (source, mean,
# variance), from their globals and, for a label key, from inside their units.
function _brm_svi_marginals(index, key)
    sites = Tuple{Int,Float64,Float64}[]
    for (s, k) in get(index.globals, key, ())
        source = index.sources[s]
        push!(sites, (s, source.mean[k], source.var[k]))
    end
    _brm_svi_label_key(key) || return sites
    for (s, label, k) in get(index.units, key, ())
        unit = index.sources[s].units[label]
        push!(sites, (s, unit.M[k], _brm_svi_unit_covariance(index, s, label)[k, k]))
    end
    sites
end

# A unit's factor over `keys`, from the sources. Keys held by one source unit
# keep that unit's joint marginal; a label key the sources hold as a global
# gets its marginal; a packed-name key matches only the same unit's factors;
# the others start at `mean` with precision `precision`, `sourced` lists the
# positions the sources set and `origin[p]` the source that set position p (0
# for none). A key with factors in two sources is refused: a unit's factor
# comes from the one slice that holds its rows.
function _brm_svi_assemble_unit(caller, index, label, ukeys, mean, precision)
    d = length(ukeys)
    M, Λ = copy(mean), zeros(d, d)
    groups = Dict{Any,Vector{Tuple{Int,Int}}}()
    cold = Int[]
    for (p, key) in enumerate(ukeys)
        unit_hits = get(index.units, key, Tuple{Int,Any,Int}[])
        global_hits = get(index.globals, key, Tuple{Int,Int}[])
        if !_brm_svi_label_key(key)
            unit_hits = filter(h -> isequal(h[2], label), unit_hits)
            global_hits = Tuple{Int,Int}[]
        end
        hits = length(unit_hits) + length(global_hits)
        if hits == 0
            push!(cold, p)
            continue
        end
        hits == 1 || throw(ArgumentError("$caller: unit $(repr(label))'s coordinate " *
            "$(repr(key)) has factors in slices " *
            join(sort!(unique!(Int[[h[1] for h in unit_hits]; [h[1] for h in global_hits]])), ", ") *
            "; a unit's factor comes from the one slice that holds its rows, so slices " *
            "that share a unit (overlapping time windows, or connected units split " *
            "across slices) are not combined"))
        source = isempty(unit_hits) ? (:global, only(global_hits)...) :
            (:unit, first(only(unit_hits)), only(unit_hits)[2])
        position = isempty(unit_hits) ? last(only(global_hits)) : last(only(unit_hits))
        push!(get!(() -> Tuple{Int,Int}[], groups, source), (p, position))
    end
    for p in cold
        Λ[p, p] = precision
    end
    origin = zeros(Int, d)
    for (source, pairs) in groups
        P, K = first.(pairs), last.(pairs)
        origin[P] .= source[2]
        if first(source) === :global
            s = source[2]
            for (p, k) in pairs
                M[p] = index.sources[s].mean[k]
                Λ[p, p] = 1 / index.sources[s].var[k]
            end
        else
            s, from = source[2], source[3]
            unit = index.sources[s].units[from]
            M[P] .= unit.M[K]
            Λ[P, P] .= length(K) == length(unit.M) ? unit.Λ[K, K] :
                inv(cholesky(Symmetric(_brm_svi_unit_covariance(index, s, from)[K, K])))
        end
    end
    M, Matrix(Symmetric(Λ)), setdiff(1:d, cold), origin
end

# ── The prior's Gaussian projection ──────────────────────────────────────────

# The smallest precision the projection iterates with (an sd of 10⁴), so that a
# flat or very wide coordinate's draws stay finite where other priors read it.
const _BRM_SVI_PRIOR_FLOOR = 1e-8

# The diagonal Gaussian closest to the global prior term P(g) in the
# variational sense (the factor a fit on no data would reach), by the same
# natural-gradient rule the units use: precision from Stein's lemma on an
# antithetic pair, mean by a Newton step, each damped and trust-bounded, and
# Polyak-averaged over the second half. A coordinate whose prior gradient is
# zero at every draw has a flat prior: precision 0.
function _brm_svi_prior_projection(t; rng, pairs=16, iterations=150, step=0.5)
    D = length(t.globals)
    m = _brm_svi_start(t.m, nothing, nothing)[t.globals]
    p = ones(D)
    flat = trues(D)
    mean_sum, precision_sum, averaged = zeros(D), zeros(D), 0
    G, H = zeros(D), zeros(D)
    for iteration in 1:iterations
        fill!(G, 0.0)
        fill!(H, 0.0)
        root = sqrt.(p)
        for _ in 1:pairs
            ε = randn(rng, D)
            δ = ε ./ root
            _, up = _brm_svi_global_term(t, m .+ δ)
            _, down = _brm_svi_global_term(t, m .- δ)
            all(isfinite, up) && all(isfinite, down) || throw(ArgumentError(
                "brm_svi_combine: the prior gradient is not finite at a draw of its " *
                "Gaussian projection, at " * join(string.(t.m.view.coordinates[t.globals[
                findall(j -> !(isfinite(up[j]) && isfinite(down[j])), 1:D)]]), ", ")))
            @. flat &= (up == 0) & (down == 0)
            @. G += (up + down) / 2
            @. H -= (up - down) / 2 * ε * root
        end
        G ./= pairs
        H ./= pairs
        # A step changes a precision at most 2.5-fold and a mean by at most one
        # sd of the current factor, as the units' trust region does.
        @. p = max((1 - step) * p + step * clamp(H, p / 2.5, 2.5p), _BRM_SVI_PRIOR_FLOOR)
        @. m += clamp(step * G / p, -1 / sqrt(p), 1 / sqrt(p))
        if 2iteration > iterations
            averaged += 1
            mean_sum .+= m
            precision_sum .+= p
        end
    end
    precision = precision_sum ./ averaged
    precision[flat] .= 0.0
    (; mean=mean_sum ./ averaged, precision, flat, pairs, iterations)
end

# ── Carrying draws to the combined hyperparameters ──────────────────────────

# A random-effect draw is standardized: level g of a block has the natural
# effect A z, with A = √w_g diag(τ) L (its stratum's factor in a stratified
# block), and a margin's total adds the population coefficients that block
# absorbs. A slice's data informs those effects and totals, not z itself.
# Under the combined τ, L and coefficients a copied z would mean another
# effect: a slice that fitted a small τ holds large z, which a larger combined
# τ multiplies (under a log link, into overflow). So every draw's factor is
# carried from its slice's hyperparameters to the combined ones by the affine
# map that keeps its effects and totals:
#
#     z_s = C z_c + d,    C = A_s⁻¹ A_c,    d = A_s⁻¹ δ,
#
# where δ is the change of the absorbed coefficients times their design.

# The block levels whose draws the combination carries. Each instance is one
# level of a random-effect block of the units (or of a nested grouping) or of
# a shared grouping: its record, level index `g`, the packed positions of its
# draws in margin order, and the unit whose subject block holds it (0 for a
# shared grouping's level). A level-correlated block's levels are not
# separable and a family with a fixed covariance has nothing to carry, so
# neither has instances.
function _brm_svi_draw_instances(m::BRMNLMEModel)
    view, rk = m.view, m.backend
    carried(r) = r.kind in (:ranef, :stratified_ranef) &&
        get(r, :level_correlation, nothing) === nothing
    records = Dict{Symbol,Any}()
    record_for(r) = get!(() -> _brm_rk_start_record(rk, r), records, r.z)
    levels = Dict{Symbol,Dict{Any,Int}}()
    level_index(r) = get!(levels, r.z) do
        Dict{Any,Int}(l => g for (g, l) in enumerate(_rk_ranef_record_levels(rk, r)))
    end
    family_of = Dict(f.block => f for f in view.families)
    instances = NamedTuple[]
    for (i, (block, margins)) in enumerate(zip(view.subject_coordinates, view.subject_margins))
        slots, order = Dict{Any,Vector{Int}}(), Any[]
        for (p, margin) in enumerate(margins)
            margin.coefficient === :mixing && continue
            key = (margin.block, margin.level)
            haskey(slots, key) || push!(order, key)
            push!(get!(() -> Int[], slots, key), p)
        end
        for key in order
            f = get(family_of, first(key), nothing)
            (f === nothing || f.fixed !== nothing || !carried(f.record)) && continue
            P = slots[key]
            g = get(level_index(f.record), last(key), 0)
            length(P) == length(f.record.margins) && g != 0 || continue
            push!(instances, (; record=record_for(f.record), g, positions=block[P], unit=i))
        end
    end
    position = Dict(name => j for (j, name) in enumerate(view.coordinates))
    for r in _brm_nlme_records(rk.plan)
        r.kind in (:ranef, :stratified_ranef) && r.group in view.shared_groups &&
            carried(r) || continue
        for g in eachindex(_rk_ranef_record_levels(rk, r))
            positions = [get(position, Symbol(r.z, ".", g, ".", k), 0)
                for k in eachindex(r.margins)]
            any(iszero, positions) ||
                push!(instances, (; record=record_for(r), g, positions, unit=0))
        end
    end
    instances
end

# `A` of an instance at constrained `values`: its level's effects are `A z`.
_brm_svi_draw_factor(values, instance) = Matrix{Float64}(
    _brm_rk_level_factor(values, instance.record, length(instance.positions))(instance.g))

# The packed coordinates of the instances' scales and correlation factors. Their
# names mean the same in every binding, so a slice that holds one inside a
# unit (the one unit of a family in that slice) still names its value.
function _brm_svi_hyperparameters(m::BRMNLMEModel, instances)
    names = m.view.coordinates
    empty = _rk_layout_empty_declarations(m.backend)
    positions = Set{Int}()
    for record in unique(inst.record for inst in instances)
        scales = isnothing(record.scales) ? (record.scale,) : record.scales
        for declaration in (scales..., record.L)
            declaration === nothing && continue
            union!(positions, _brm_nlme_declaration_coordinates(names, declaration, empty))
        end
    end
    positions
end

# A block margin's identity across strata: a stratified block has one family
# per stratum, and one population coefficient adds into all of them.
_brm_svi_margin_key(block, predictor, coefficient) =
    (block.group, block.id, block.predictor, predictor, coefficient)

# The population coefficients each random-effect margin absorbs,
# `margin key => [(packed position, design per unit)]`. A coefficient adds into
# the units' block margin the NLME view mu-references it into (the view's own
# references for a flat view, found the same way over its block margins when
# its units hold different families). Otherwise it adds into the first margin
# of a shared grouping's block whose design column is the coefficient's own
# (`(odl_offset, …) ~ 1 + (1 | ID | device)`: the intercept), with design 1.
# Each coefficient adds into one block, so a total never moves twice. With
# nested groupings a total adds several levels' draws, so none are kept there.
# Only a coefficient whose packed coordinate is its value (identity
# transform) is kept, so that a coordinate's change is the coefficient's change.
function _brm_svi_location_references(m::BRMNLMEModel)
    view, rk = m.view, m.backend
    references = Dict{Any,Vector{Tuple{Int,Vector{Float64}}}}()
    isempty(view.nested_groups) || return references
    names = view.coordinates
    position = Dict(name => j for (j, name) in enumerate(names))
    empty = _rk_layout_empty_declarations(rk)
    population = Dict{Tuple{Symbol,Symbol},Int}()
    for r in _brm_nlme_records(rk.plan)
        r.kind === :population || continue
        index = get(r, :index, ())
        hits = isempty(index) ? _brm_nlme_declaration_coordinates(names, r.declaration, empty) :
            [get(position, Symbol(r.declaration, ".", join(index, ".")), 0)]
        length(hits) == 1 && only(hits) != 0 && view.role[only(hits)] === :population &&
            (population[(r.predictor, r.coefficient)] = only(hits))
    end
    rows, nrows = view.rows, length(rk.plan.columns[view.group])
    units = view.ragged ? _brm_nlme_mu_references(rk, view.block_margins, population, rows, nrows;
            known=Set(f.block for f in view.families if _brm_nlme_standardized_block(f.record))) :
        view.mu_references
    absorbed = Set(ref.population for ref in units)
    shared = [(; block=_brm_nlme_block_identity(r), predictor, coefficient)
        for r in _brm_nlme_records(rk.plan)
        if r.kind in (:ranef, :stratified_ranef) && r.group in view.shared_groups
        for (predictor, coefficient) in r.margins]
    rest = Dict(k => j for (k, j) in population if j ∉ absorbed)
    crossed = filter(_brm_nlme_mu_references(rk, shared, rest, rows, nrows)) do ref
        shared[ref.subject].coefficient === ref.coefficient    # its own column, design 1
    end
    transforms = _rk_layout_coordinate_transforms(rk)
    for (margins, refs) in ((view.block_margins, units), (shared, crossed)), ref in refs
        transforms[ref.population] === :identity || continue
        margin = margins[ref.subject]
        push!(get!(() -> Tuple{Int,Vector{Float64}}[], references,
                _brm_svi_margin_key(margin.block, margin.predictor, margin.coefficient)),
            (ref.population, ref.design))
    end
    references
end

# δ of an instance: the change from `before` to `after` of the coefficients its
# margins absorb, times their design for unit `i` (1 for a shared level).
function _brm_svi_location_change(instance, references, i, after, before)
    block = _brm_nlme_block_identity(instance.record)
    δ = zeros(length(instance.positions))
    for (k, (predictor, coefficient)) in enumerate(instance.record.margins),
            (j, design) in get(references, _brm_svi_margin_key(block, predictor, coefficient), ())
        δ[k] += (i == 0 ? 1.0 : design[i]) * (after[j] - before[j])
    end
    δ
end

# Unit i's density at the globals `g` and its draws `z`, the draws' prior included; -Inf
# where it has no finite value. Carrying keeps a unit's totals under the combined
# coefficients, which is right while a draw can express that: under a log link, slices
# whose scales differ 40-fold read -1,367 carried and -189,452 copied. It is wrong where
# a draw cannot: ODL pins its device blocks' scales at 1e-6, so a location change of 0.001
# would need a draw of 1,000, and the units' median |z| went from 0.24 to 10 and the
# start's density to -Inf, which copying them left at -2e5 and an ELBO above a cold fit's
# by 7 orders of magnitude. No threshold on the draws tells these apart, and the
# density does, per unit, at two evaluations each. Carrying stands unless it costs more
# than d/2 for its d draws, which is what a draw from the unit's own factor reads below
# the factor's mode.
function _brm_svi_unit_density(t, i, g, z)
    value = try
        _brm_svi_unit_term!(zeros(length(g)), zeros(length(z)), t, i, g, z)
    catch e
        e isa DomainError || rethrow()
        -Inf
    end
    isfinite(value) ? value : -Inf
end

# A unit's factor (M, Λ) carried by x_s = T x_c + d. The factor is the prior of
# the carried draws, N(0, I), times the site of its data; the site moves with
# the map and the prior is applied once more:
#
#     Λ' = Tᵀ (Λ − Π) T + Π,    Λ' M' = Tᵀ (Λ M − (Λ − Π) d),
#
# Π the identity on the carried draws. With T = I and d = 0 this is the copy;
# for a unit whose data pins its effects it keeps their totals; for a unit with
# little data the prior pulls its effects to the combined population, as in a
# fresh fit. A slice's factor can be wider than the prior in some direction (a
# negative site), which a large map can turn into a precision that is not
# positive definite: then the site's negative part is dropped, as a global's
# negative site is (`:dropped`). Should that still fail, the factor is carried
# as a change of variables, prior and all (`:mapped`).
function _brm_svi_carry_unit(M, Λ, T, d, carried)
    Π = Diagonal(Float64.(carried))
    S = Λ - Π
    h = Λ * M
    function site(S, h)
        Λc = Symmetric(T' * S * T + Π)
        F = cholesky(Λc; check=false)
        issuccess(F) ? (F \ (T' * (h - S * d)), Matrix(Λc)) : nothing
    end
    result = site(S, h)
    result === nothing || return (result..., :site)
    E = eigen(Symmetric(S))
    V = E.vectors[:, E.values .> 0]
    result = site(V * Diagonal(E.values[E.values .> 0]) * V', V * (V' * h))
    result === nothing || return (result..., :dropped)
    (T \ (M - d), Matrix(Symmetric(T' * Λ * T)), :mapped)
end

# The product of the sites of one shared level's K draws, each slice's sites
# carried to the combined hyperparameters first: in z_c, slice s's diagonal
# site (λ_s, h_s) is Cᵀ diag(λ_s) C, Cᵀ (h_s − λ_s d) with C = A_s⁻¹ A_c and
# d = A_s⁻¹ δ_s. `sites[k]` lists draw k's (slice, mean, variance);
# `carry(s)` is slice s's `(A_s, δ_s)`. Returns the means, the product's
# precision, the dropped sites per draw and the spread (as for any global, in
# the carried coordinates), or `nothing` when the product is not positive
# definite.
function _brm_svi_carry_shared(sites, p0, m0, A_c, carry)
    K = length(sites)
    slices = sort!(unique!([s for draw in sites for (s, _, _) in draw]))
    clamped = zeros(Int, K)
    carried = map(slices) do s
        λ, h = zeros(K), zeros(K)
        for k in 1:K, (r, ms, vs) in sites[k]
            r == s || continue
            ps = 1 / vs
            ps - p0[k] < 0 && (clamped[k] += 1; continue)
            λ[k] += ps - p0[k]
            h[k] += ps * ms - p0[k] * m0[k]
        end
        A_s, δ = carry(s)
        C, d = A_s \ A_c, A_s \ δ
        (; s, P=C' * Diagonal(λ) * C, η=C' * (h - λ .* d))
    end
    P0, η0 = Diagonal(p0), p0 .* m0
    precision = Matrix(Symmetric(P0 + sum(c.P for c in carried)))
    F = cholesky(Symmetric(precision); check=false)
    issuccess(F) || return nothing
    μ = F \ (η0 + sum(c.η for c in carried))
    spread, spread_slice = fill(NaN, K), zeros(Int, K)
    for k in 1:K
        length(sites[k]) >= 2 || continue
        for c in carried
            any(site -> first(site) == c.s, sites[k]) || continue
            Fs = cholesky(Symmetric(P0 + c.P); check=false)
            issuccess(Fs) || continue
            distance = abs((Fs \ (η0 + c.η))[k] - μ[k]) / sqrt(inv(Fs)[k, k])
            (isnan(spread[k]) || distance > spread[k]) &&
                (spread[k] = distance; spread_slice[k] = c.s)
        end
    end
    (; μ, precision, clamped, spread, spread_slice)
end

# ── Sites over the globals ───────────────────────────────────────────────────

# A slice's factor over its globals is N(m_s, Σ_s), Σ_s = diag(s²) + ΒΒᵀ. Its site
# over coordinates R is what the slice's data added to the prior's diagonal
# projection (m₀, P₀), counted once in the product:
#
#     S_s = Σ_s[R, R]⁻¹ − P₀[R],    h_s = Σ_s[R, R]⁻¹ m_s[R] − P₀[R] m₀[R].
#
# Three things separate that from the plain product of these sites, which reads
# noise and contradiction as information. On ODL's 16 Länder it put 629 of 1,090 means
# over 3 prior sds out, field coefficients no Land informs up to 25 sds out and a length
# 51, and the fit from there died at its first draw. Coordinate by coordinate it stayed
# near the priors but held a rate's scale at 1.67 ± 0.009 where the Länder range from
# −4.06 to 2.28, and that fit died at iteration 13 (snag svi-combine-join-6aabb60b).
#
# Which coordinates. Data can only narrow a prior, Σ_s ≼ P₀⁻¹, and a coordinate at
# τ = Σ_s[k, k] P₀[k, k] = 1 has no row in S_s at all. An uninformed coordinate's τ
# scatters around 1 under SVI (ODL: 0.85 to 1.16 from the 1st to the 99th percentile,
# 46% above 1), so there a rank row is noise, and a coupling read from it pulls with
# the full precision of whatever it couples to. A coordinate joins a slice's joint
# site if the slice narrows it beyond that scatter, or if the rank part carries it
# (a ridge's narrow-prior side: a slice pinning a + b, a's prior 10 times b's, leaves
# b at τ = 0.99 with all its variance along the ridge). Elsewhere it has a marginal
# site where τ < 1, and none where τ ≥ 1, as in the coordinate-by-coordinate product.
# Taking the sub-block of Σ_s marginalizes the other coordinates out of the slice.
#
# Which directions. A direction wider than the prior has no site. "Wider" is the
# generalized eigenproblem of (P₀, Σ_s⁻¹), so with Σ_s = L Lᵀ and Lᵀ P₀ L = Y K Yᵀ the
# site is S⁺ = W Wᵀ, W = L⁻ᵀ Y₊ (I − K₊)^{1/2}, over the eigenvalues below 1, with
# h⁺ about m₀ = W d, d = (I − K₊)^{-1/2} Y₊ᵀ L⁻¹ (m_s − m₀). That is invariant under
# a rescaling of the coordinates, needs no case for a flat prior, and does not let a
# prior sd of 1e-6 decide the directions, as an eigenbasis of S itself does.
#
# How far to trust them. Slices can contradict each other by thousands of their own
# sds (ODL: 4,953 on one correlation coordinate). The product then resolves that
# through whatever small cross-covariance it finds: a slice's correlation of 0.2
# between a coordinate j and one it pins, l, and a gap of 70 slice sds on l, moves j by
# 14 of its sds. And it sits where the narrowest slices put it, as narrow as they are.
# So the slices are read as random effects, as in a random-effects meta-analysis:
# each one's own x_s ~ N(x, T), T diagonal, which adds T to a site's variance,
# S' = W (I + Wᵀ T W)⁻¹ Wᵀ, h' = W (I + Wᵀ T W)⁻¹ d (about m₀), the plain site at T = 0.
# The coordinate then sits among the slices, and no narrower than a cold fit starts,
# for the fit to settle.
#
# T_k is the Paule-Mandel estimate over the marginal sites of the slices that pin k by
# themselves, where Cochran's Q says they contradict, more than SVI's narrow sds explain.
# Those slices are what keeps ridges: a slice's marginal along its own ridge says little about the coordinate,
# so two slices pinning a + b and a − b can read as contradicting in their marginals
# (truth (2, 5) under N(0, 3²) gives `a` means 3.5 and −1.5, each of sd 2.1) though
# their ridges cross exactly. So a slice pins k if it narrows k beyond the scatter and
# the rank part does not carry k, or if its marginal is so narrow that no ridge of its
# own moves k far (τ < `_BRM_SVI_PINNED`). This cannot see a contradiction along a
# ridge, which a diagonal T could not express anyway, and which no longer reaches
# coordinates outside the sites; a mean that leaves what supports it falls back.
#
# A global that a slice holds inside one of its units enters through that unit
# factor's marginal, as a site of its own.

# How far an uninformed coordinate's τ scatters around 1 under SVI, and how much of a
# ridge's variance the rank part holds.
const _BRM_SVI_SITE_SCATTER = 0.2
# A marginal this far below the prior's variance, in τ (a tenth of its sd), pins a
# coordinate whatever carries it.
const _BRM_SVI_PINNED = 0.01
# Cochran's Q beyond this quantile of its χ² is a contradiction, not sampling noise, once
# Q allows for SVI's sds running narrower than the posterior's, down to this fraction of
# NUTS's: three slices of one truth then read 3 of their sds apart.
const _BRM_SVI_CONTRADICTION_LEVEL = 0.999
const _BRM_SVI_SD_NARROWNESS = 0.25
# A joint mean is supported within this many sds of a counted site or of the prior.
const _BRM_SVI_SUPPORT_SDS = 3
# A contradicted coordinate starts no wider than a cold fit's default `global_scale`:
# its between-slice width is what the slices say about x, not how a first draw should
# scatter, which on a scale under a log link multiplies through the exponential.
const _BRM_SVI_CONTRADICTED_SD = 0.1

_brm_svi_in_joint_site(τ, rank_share) = τ < 1 - _BRM_SVI_SITE_SCATTER ||
    (τ < 1 + _BRM_SVI_SITE_SCATTER && rank_share > 1 - _BRM_SVI_SITE_SCATTER)
_brm_svi_pins(τ, rank_share) = τ < 1 - _BRM_SVI_SITE_SCATTER &&
    (rank_share <= 1 - _BRM_SVI_SITE_SCATTER || τ < _BRM_SVI_PINNED)

# A coordinate's marginal site, as its mean and variance: what the slice's marginal
# (mean, var) added to the prior's (m0, 1/p0). `nothing` where the slice is not narrower.
function _brm_svi_marginal_site(mean, var, p0, m0)
    τ = var * p0
    τ < 1 || return nothing
    (m0 + (mean - m0) / (1 - τ), var / (1 - τ))
end

# The between-slice variance T that brings Q(T) = Σ w (c − θ)², w = 1/(v + T), θ the
# w-weighted mean, to K − 1 (Paule and Mandel 1982); 0 where Q(0) is no contradiction,
# allowing for sds that narrow.
function _brm_svi_paule_mandel(sites)
    K = length(sites)
    K < 2 && return 0.0
    c, v = first.(sites), last.(sites)
    Q(T) = let w = 1 ./ (v .+ T), θ = sum(w .* c) / sum(w)
        sum(w .* (c .- θ) .^ 2)
    end
    Q(0.0) * _BRM_SVI_SD_NARROWNESS^2 <= quantile(Chisq(K - 1), _BRM_SVI_CONTRADICTION_LEVEL) && return 0.0
    lo, hi = 0.0, (maximum(c) - minimum(c))^2
    while Q(hi) > K - 1
        hi *= 2
    end
    for _ in 1:80
        mid = (lo + hi) / 2
        Q(mid) > K - 1 ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end

# The between-slice variance of global `key`, from the slices that pin it by themselves.
function _brm_svi_heterogeneity(index, key, p0, m0)
    sites = Tuple{Float64,Float64}[]
    for (s, k) in get(index.globals, key, ())
        source = index.sources[s]
        _brm_svi_pins(source.var[k] * p0, sum(abs2, view(source.Β, k, :)) / source.var[k]) &&
            push!(sites, _brm_svi_marginal_site(source.mean[k], source.var[k], p0, m0))
    end
    if _brm_svi_label_key(key)
        for (s, label, k) in get(index.units, key, ())
            var = _brm_svi_unit_covariance(index, s, label)[k, k]
            _brm_svi_pins(var * p0, 0.0) && push!(sites,
                _brm_svi_marginal_site(index.sources[s].units[label].M[k], var, p0, m0))
        end
    end
    _brm_svi_paule_mandel(sites)
end

# A slice's joint site over the rows `R`, as (W, d) above; `clamped` counts each dropped
# direction at the coordinate it weighs most, in that coordinate's marginal sds.
function _brm_svi_site_factor(R, mean, scale, Β, p0, m0, clamped)
    Σ = Symmetric(Diagonal(scale .^ 2) + Β * Β')
    L = cholesky(Σ).L
    E = eigen(Symmetric(L' * (p0 .* Matrix(L))))
    keep = E.values .< 1
    sd = sqrt.(diag(Σ))
    for i in findall(!, keep)
        clamped[R[argmax(abs.(L * view(E.vectors, :, i)) ./ sd)]] += 1
    end
    V = L' \ E.vectors[:, keep]
    root = sqrt.(1 .- E.values[keep])
    (; R, W=V .* root', d=(V' * (mean .- m0)) ./ root)
end

# A site's precision and natural parameter at between-slice variance `T`.
function _brm_svi_site_widened(site, T, m0)
    F = cholesky(Symmetric(I + site.W' * (T[site.R] .* site.W)))
    G = site.W / F.U
    S = G * G'
    (; S, h=site.W * (F \ site.d) .+ S * m0[site.R])
end

# The rows of `A` whose product mean lies outside everything that supports it: the
# interval from the lowest to the highest of the counted sites' and the prior's ranges.
# A flat prior supports anything. A slice wider than the prior has no site, so a
# runaway slice widens nothing.
function _brm_svi_joint_outlying(index, gkeys, ks, A, μ, p0, m0)
    filter(A) do r
        p0[r] > 0 || return false
        lo, hi = m0[r] - _BRM_SVI_SUPPORT_SDS / sqrt(p0[r]), m0[r] + _BRM_SVI_SUPPORT_SDS / sqrt(p0[r])
        for (_, ms, vs) in _brm_svi_marginals(index, gkeys[ks[r]])
            vs * p0[r] < 1 || continue
            lo, hi = min(lo, ms - _BRM_SVI_SUPPORT_SDS * sqrt(vs)), max(hi, ms + _BRM_SVI_SUPPORT_SDS * sqrt(vs))
        end
        !(lo <= μ[r] <= hi)
    end
end

# The product of the slices' sites over the globals `ks` (positions into `gkeys`) at
# between-slice variances `T`, as described above: dense over the block `A` of rows in
# joint sites, diagonal elsewhere. `status` is `:joint`, or why the
# coordinate-by-coordinate product stands instead: `:limit` (more than `limit`
# coordinates in joint sites), `:indefinite` (not positive definite), `:outlying` (a
# mean no site and not the prior supports).
function _brm_svi_joint_product(index, gkeys, ks, p0, m0, T, limit)
    n = length(ks)
    row = Dict(gkeys[k] => r for (r, k) in enumerate(ks))
    clamped = zeros(Int, n)
    marginal = [Tuple{Float64,Float64}[] for _ in 1:n]
    function count_marginal!(r, ms, vs)
        site = _brm_svi_marginal_site(ms, vs, p0[r], m0[r])
        site === nothing ? (clamped[r] += 1) : push!(marginal[r], site)
    end
    members = NamedTuple[]
    for source in index.sources
        R, J = Int[], Int[]
        for (j, key) in enumerate(source.keys)
            source.informed[j] || continue
            r = get(row, key, 0)
            r == 0 || (push!(R, r); push!(J, j))
        end
        isempty(R) && continue
        Β = source.Β[J, :]
        inside = iszero(Β) ? falses(length(R)) : _brm_svi_in_joint_site.(source.var[J] .* p0[R],
            vec(sum(abs2, Β; dims=2)) ./ source.var[J])
        for (q, (r, j)) in enumerate(zip(R, J))
            inside[q] || count_marginal!(r, source.mean[j], source.var[j])
        end
        any(inside) && push!(members, (; source, R=R[inside], J=J[inside]))
    end
    for (r, k) in enumerate(ks)
        _brm_svi_label_key(gkeys[k]) || continue
        for (s, label, j) in get(index.units, gkeys[k], ())
            count_marginal!(r, index.sources[s].units[label].M[j],
                _brm_svi_unit_covariance(index, s, label)[j, j])
        end
    end
    A = sort!(unique!(reduce(vcat, (m.R for m in members); init=Int[])))
    (limit == 0 || length(A) > limit) && return (; status=:limit)
    position = zeros(Int, n)
    position[A] .= eachindex(A)
    diagonal, η = copy(p0), p0 .* m0
    for (r, sites_r) in enumerate(marginal), (c, v) in sites_r
        diagonal[r] += 1 / (v + T[r])
        η[r] += c / (v + T[r])
    end
    P = Matrix(Diagonal(diagonal[A]))
    ηA = η[A]
    for m in members
        site = _brm_svi_site_factor(m.R, m.source.mean[m.J], m.source.scale[m.J], m.source.Β[m.J, :],
            p0[m.R], m0[m.R], clamped)
        w = _brm_svi_site_widened(site, T, m0)
        b = position[m.R]
        P[b, b] .+= w.S
        ηA[b] .+= w.h
    end
    F = cholesky(Symmetric(P); check=false)
    issuccess(F) || return (; status=:indefinite)
    μ = η ./ diagonal
    μ[A] .= F \ ηA
    diagonal[A] .= diag(P)
    outlying = _brm_svi_joint_outlying(index, gkeys, ks, A, μ, p0, m0)
    isempty(outlying) || return (; status=:outlying, outlying)
    (; status=:joint, μ, diagonal, block=A, precision=Matrix(Symmetric(P)), clamped)
end

# ── The combined start ───────────────────────────────────────────────────────

"""
    BRMSVIStart

A factor-level start for [`brm_svi_fit`](@ref), from [`brm_svi_combine`](@ref):
the means and mean-field standard deviations of the global coordinates it
informs, keyed by semantic coordinate identity, with the precision of each
block of them that is correlated (`globals.correlated`, `positions` into
`globals.keys` and a dense `precision`), the unit factors (mean and precision)
keyed by unit label and semantic identity, the identity of the model it was
combined for, and `receipt`, which says where every number came from.
It holds no data and no compiled model, so it serializes like a
[`BRMSVIState`](@ref) (with the same package versions). Pass it as
`brm_svi_fit(x; start)`: it sets where the fit's factors start and is not a
prior.
"""
struct BRMSVIStart
    format::Int
    source_id::String
    preprocessing::BRMFittedState
    group::Symbol
    nested::Tuple{Vararg{Symbol}}
    shared::Tuple{Vararg{Symbol}}
    globals::NamedTuple
    units::Dict{Any,NamedTuple}
    receipt::NamedTuple
end

function Base.show(io::IO, s::BRMSVIStart)
    print(io, "BRMSVIStart(", length(s.globals.keys), " globals and ", length(s.units),
        " units from ", s.receipt.slices, " slices)")
end

_brm_svi_slice_state(caller, x::BRMSVIState, which) = x
_brm_svi_slice_state(caller, x::NamedTuple, which) =
    hasproperty(x, :diagnostics) && hasproperty(x.diagnostics, :q) ? brm_svi_checkpoint(x) :
    throw(ArgumentError("$caller: $which is a named tuple that is not a brm_svi_fit result; " *
        "a snapshot records no semantic unit keys, so pass the fit result or its " *
        "brm_svi_checkpoint"))
_brm_svi_slice_state(caller, x::BRMSVILaplaceState, which) = throw(ArgumentError(
    "$caller: $which is a GaussianBlockLaplace state, whose correlated common factor is " *
    "not combined; pass ordinary SVI fits"))
_brm_svi_slice_state(caller, x, which) = throw(ArgumentError(
    "$caller: $which is a $(typeof(x)); expected a brm_svi_fit result or a BRMSVIState"))

# Whether `start=` asks for a factor-level start rather than a packed point.
_brm_svi_factor_start(x) = x isa Union{BRMSVIStart,BRMSVIState,BRMSVILaplaceState} ||
    x isa NamedTuple && hasproperty(x, :diagnostics) ||
    x isa AbstractVector && !(eltype(x) <: Real) && !isempty(x) && all(_brm_svi_factor_start, x)

# A slice (or a start) belongs to this model: the same declaration, program and
# callables, the same groupings, and the same preprocessing except the level
# tables of label-keyed groupings.
function _brm_svi_check_slice(caller, m, which, source_id, preprocessing, group, nested, shared)
    source_id == _brm_svi_source_id(m.backend) || throw(ArgumentError(
        "$caller: $which comes from another model (declaration, emitted program or " *
        "callable bindings differ)"))
    group === m.view.group && nested == Tuple(keys(m.view.nested_groups)) &&
        Set(shared) == Set(m.view.shared_groups) || throw(ArgumentError(
        "$caller: $which has another subject, nested or shared grouping"))
    _brm_svi_check_preprocessing(m.backend, preprocessing, _brm_svi_label_keyed_groups(m);
        caller="$caller: $which", hint="bind the slice with the full model's frozen " *
            "preprocessing (RKBRMI(builder(slice_data), brm_fitted_state(full); " *
            "resample_groups))")
end

# What `exclude=` may be: nothing to leave out, a name (Symbol or String), a
# collection of names, or a function of a coordinate's name. Checked before the
# full model is prepared, so a wrong argument costs no preparation.
function _brm_svi_check_exclude(caller, exclude)
    exclude isa Union{Function,Symbol,AbstractString} && return nothing
    exclude isa Union{AbstractVector,Tuple,AbstractSet} &&
        all(e -> e isa Union{Symbol,AbstractString}, exclude) && return nothing
    throw(ArgumentError("$caller: exclude takes coordinate names (a Symbol or String, " *
        "or a vector of them) or a function of a coordinate's name; got $(repr(exclude))"))
end

# Which of the global coordinates `names` (the full model's, aligned with
# `receipt.globals.coordinates`) `exclude` leaves out of the product. A name
# selects the coordinate of that name and every coordinate under it, its name
# followed by `.`: `natural_field` selects `natural_field.amplitude` and
# `natural_field.z.3`, `natural_field.z` only the latter. A function selects the
# names for which it returns `true`. A selection of nothing is refused, since a
# misspelt name would otherwise leave the start as it was without a word.
function _brm_svi_excluded(caller, names, exclude)
    hit = falses(length(names))
    strings = String.(names)
    if exclude isa Function
        for (k, name) in enumerate(names)
            chosen = exclude(name)
            chosen isa Bool || throw(ArgumentError("$caller: exclude's function returned " *
                "$(repr(chosen)) for $(repr(name)); it must return true or false"))
            hit[k] = chosen
        end
        any(hit) || throw(ArgumentError("$caller: exclude's function selects no global " *
            "coordinate of the full model (receipt.globals.coordinates lists them)"))
        return hit
    end
    for selector in (exclude isa Union{Symbol,AbstractString} ? (exclude,) : exclude)
        s = String(selector)
        found = [n == s || startswith(n, s * ".") for n in strings]
        any(found) || throw(ArgumentError("$caller: exclude names $(repr(selector)), which " *
            "is no global coordinate of the full model and no declaration of one " *
            "(receipt.globals.coordinates lists them)"))
        hit .|= found
    end
    hit
end

# The product of the sites for every global coordinate of `t`, and the units'
# factors, from the slices' states. A coordinate `exclude` selects takes no site:
# it starts as one no slice informs.
function _brm_svi_combine(t, states; caller="brm_svi_combine", rng=Random.Xoshiro(1),
        prior_pairs=16, prior_iterations=150, joint_limit=4096, carry_all=false, exclude=())
    isempty(states) && throw(ArgumentError("$caller: no slices"))
    _brm_svi_check_exclude(caller, exclude)
    m = t.m
    m.view.unit_partition === :subjects || throw(ArgumentError(
        "$caller: the full model's units are connected components " *
        "(`unit_partition=:$(m.view.unit_partition)`), whose members and labels change " *
        "between bindings; combine slices of a model with `unit_partition=:subjects`"))
    sources = map(enumerate(states)) do (s, state)
        which = "slice $s"
        state.unit_partition === :subjects || throw(ArgumentError(
            "$caller: $which was fitted with `unit_partition=:$(state.unit_partition)`; " *
            "combine slices fitted with `unit_partition=:subjects`"))
        _brm_svi_check_slice(caller, m, which, state.source_id, state.preprocessing,
            state.group, state.nested, state.shared)
        _brm_svi_source(caller, state, which)
    end
    index = _brm_svi_source_index(sources)
    coordinate_keys = _brm_svi_coordinate_keys(m)
    gkeys = coordinate_keys[t.globals]
    allunique(gkeys) || error("brm_svi_combine: internal: two global coordinates share " *
        "the semantic key $(repr(gkeys[findfirst(k -> count(isequal(k), gkeys) > 1, gkeys)]))")
    names = m.view.coordinates[t.globals]
    excluded = _brm_svi_excluded(caller, names, exclude)
    prior = _brm_svi_prior_projection(t; rng, pairs=prior_pairs, iterations=prior_iterations)
    D = length(t.globals)
    informed, clamped = zeros(Int, D), zeros(Int, D)
    spread, spread_slice = fill(NaN, D), zeros(Int, D)
    μ, σ = fill(NaN, D), fill(NaN, D)
    # The draws of shared levels combine after the other globals, carried to
    # the hyperparameters those give.
    instances = _brm_svi_draw_instances(m)
    global_index = Dict(j => k for (k, j) in enumerate(t.globals))
    shared = [inst for inst in instances if inst.unit == 0 &&
        all(j -> haskey(global_index, j), inst.positions)]
    later = falses(D)
    for inst in shared, j in inst.positions
        later[global_index[j]] = true
    end
    marginals = Dict{Int,Vector{Tuple{Int,Float64,Float64}}}()
    heterogeneity = zeros(D)
    for (k, key) in enumerate(gkeys)
        later[k] && continue
        sites = _brm_svi_marginals(index, key)
        isempty(sites) && continue
        # Left out: counted as informed, it keeps no site and takes the default.
        excluded[k] && (informed[k] = length(sites); continue)
        marginals[k] = sites
        p0, m0 = prior.precision[k], prior.mean[k]
        heterogeneity[k] = _brm_svi_heterogeneity(index, key, p0, m0)
        P, η = p0, p0 * m0
        for (_, ms, vs) in sites
            site = _brm_svi_marginal_site(ms, vs, p0, m0)
            site === nothing && (clamped[k] += 1; continue)
            P += 1 / (last(site) + heterogeneity[k])
            η += first(site) / (last(site) + heterogeneity[k])
        end
        informed[k] = length(sites)
        μ[k] = η / P
        σ[k] = 1 / sqrt(P)
    end
    # Where a slice's global factor has a rank part, its sites are joint: the
    # product above, coordinate by coordinate, is then replaced by the product
    # of the joint sites, up to `joint_limit` coordinates in them. `correlated`
    # lists the blocks of the start's precision that are not diagonal.
    correlated = NamedTuple[]
    joined = sort!(collect(keys(marginals)))
    outlying = falses(D)
    joint = if !any(source -> !iszero(source.Β), sources) || isempty(joined)
        :diagonal
    else
        result = _brm_svi_joint_product(index, gkeys, joined, prior.precision[joined],
            prior.mean[joined], heterogeneity[joined], joint_limit)
        if result.status === :joint
            μ[joined] .= result.μ
            σ[joined] .= 1 ./ sqrt.(result.diagonal)
            clamped[joined] .= result.clamped
            isdiag(result.precision) || push!(correlated,
                (; globals=joined[result.block], precision=result.precision))
        elseif result.status === :outlying
            outlying[joined[result.outlying]] .= true
        end
        result.status
    end
    for k in findall(>(0), heterogeneity)
        σ[k] = min(σ[k], _BRM_SVI_CONTRADICTED_SD)
    end
    for (k, sites) in marginals
        length(sites) >= 2 || continue
        distances = [abs(ms - μ[k]) / sqrt(vs) for (_, ms, vs) in sites]
        spread[k], at = findmax(distances)
        spread_slice[k] = first(sites[at])
    end
    median = _brm_svi_start(m, nothing, nothing)
    assembled = Dict{Int,Any}()
    for (i, label) in enumerate(m.view.levels)
        assembled[i] = _brm_svi_assemble_unit(caller, index, label,
            coordinate_keys[t.units[i]], median[t.units[i]], 1.0)
    end
    # The combined point and each slice's: the full model's packed point with
    # the combined global means, or slice s's own, and the units' means.
    combined_point = copy(median)
    for k in 1:D
        isnan(μ[k]) || (combined_point[t.globals[k]] = μ[k])
    end
    for (i, (M, _, sourced, _)) in assembled
        combined_point[t.units[i][sourced]] .= M[sourced]
    end
    key_index = Dict(key => k for (k, key) in enumerate(gkeys))
    references = _brm_svi_location_references(m)
    # A slice's own scales, correlation factors and mu-referenced coefficients,
    # from its globals or, where it holds one inside a unit, from that unit.
    readable = union(_brm_svi_hyperparameters(m, instances),
        (j for refs in values(references) for (j, _) in refs))
    slice_values = Dict{Int,Any}()
    values_of(s) = get!(slice_values, s) do
        u = copy(combined_point)
        for (j, key) in enumerate(sources[s].keys)
            k = get(key_index, key, 0)
            k == 0 || (u[t.globals[k]] = sources[s].mean[j])
        end
        for unit in values(sources[s].units), (j, key) in enumerate(unit.keys)
            k = get(key_index, key, 0)
            k != 0 && t.globals[k] in readable && (u[t.globals[k]] = unit.M[j])
        end
        (; point=u, values=_rk_constrained_values(m.backend, u))
    end
    combined = _rk_constrained_values(m.backend, combined_point)
    carried_globals = falses(D)
    for inst in shared
        ks = [global_index[j] for j in inst.positions]
        sites = [_brm_svi_marginals(index, gkeys[k]) for k in ks]
        for (r, k) in enumerate(ks)
            excluded[k] || continue
            informed[k] = length(sites[r])
            sites[r] = similar(sites[r], 0)
        end
        any(!isempty, sites) || continue
        result = _brm_svi_carry_shared(sites, prior.precision[ks], prior.mean[ks],
            _brm_svi_draw_factor(combined, inst),
            s -> let slice = values_of(s)
                (_brm_svi_draw_factor(slice.values, inst), _brm_svi_location_change(inst,
                    references, 0, combined_point, slice.point))
            end)
        for (r, k) in enumerate(ks)
            isempty(sites[r]) && continue
            informed[k] = length(sites[r])
            if result === nothing
                # Not positive definite once carried: the plain product.
                p0, m0 = prior.precision[k], prior.mean[k]
                P, η = p0, p0 * m0
                for (_, ms, vs) in sites[r]
                    1 / vs - p0 < 0 && (clamped[k] += 1; continue)
                    P += 1 / vs - p0
                    η += ms / vs - p0 * m0
                end
                μ[k], σ[k] = η / P, 1 / sqrt(P)
            else
                μ[k], σ[k] = result.μ[r], 1 / sqrt(result.precision[r, r])
                clamped[k] = result.clamped[r]
                spread[k], spread_slice[k] = result.spread[r], result.spread_slice[r]
                carried_globals[k] = true
            end
        end
        if result !== nothing && !isdiag(result.precision)
            informing = findall(!isempty, sites)
            push!(correlated, (; globals=ks[informing],
                precision=result.precision[informing, informing]))
        end
    end
    # Each unit's draws carried from its slice's hyperparameters.
    # A unit holds its own levels and any shared level only it reads.
    unit_of = Dict(j => i for (i, block) in enumerate(t.units) for j in block)
    by_unit = Dict{Int,Vector{NamedTuple}}()
    for inst in instances
        owner = inst.unit != 0 ? inst.unit : let us = unique(get(unit_of, j, 0) for j in inst.positions)
            length(us) == 1 ? only(us) : 0
        end
        owner == 0 || push!(get!(() -> NamedTuple[], by_unit, owner), inst)
    end
    units = Dict{Any,NamedTuple}()
    copied, partial, cold = 0, Any[], Any[]
    carried_units, dropped, mapped, uncarried = 0, Any[], Any[], Any[]
    # The combined globals, for reading what a unit's density says of its candidates.
    combined_globals = median[t.globals]
    let known = findall(!isnan, μ)
        combined_globals[known] .= μ[known]
    end
    _brm_svi_field_joint!(t, let u = copy(median)
        u[t.globals] .= combined_globals
        u
    end)
    for (i, label) in enumerate(m.view.levels)
        ukeys = coordinate_keys[t.units[i]]
        M, Λ, sourced, origin = assembled[i]
        if isempty(sourced)
            push!(cold, label)
            continue
        end
        length(sourced) == length(ukeys) ? (copied += 1) : push!(partial, label)
        n = length(ukeys)
        T, d, carry = Matrix{Float64}(I, n, n), zeros(n), falses(n)
        here = Dict(j => p for (p, j) in enumerate(t.units[i]))
        for inst in get(by_unit, i, NamedTuple[])
            P = [get(here, j, 0) for j in inst.positions]
            any(iszero, P) && continue
            s = origin[first(P)]
            s != 0 && all(==(s), origin[P]) || continue
            slice = values_of(s)
            A_s = _brm_svi_draw_factor(slice.values, inst)
            A_c = _brm_svi_draw_factor(combined, inst)
            δ = _brm_svi_location_change(inst, references, i, combined_point, slice.point)
            T[P, P] .= A_s \ A_c
            d[P] .= A_s \ δ
            carry[P] .= true
        end
        if any(carry)
            Mc, Λc, how = _brm_svi_carry_unit(M, Λ, T, d, carry)
            value = carry_all ? 0.0 : _brm_svi_unit_density(t, i, combined_globals, Mc)
            if carry_all || (isfinite(value) &&
                    value >= _brm_svi_unit_density(t, i, combined_globals, M) - count(carry) / 2)
                M, Λ = Mc, Λc
                carried_units += 1
                how === :dropped && push!(dropped, label)
                how === :mapped && push!(mapped, label)
            else
                push!(uncarried, label)
            end
        end
        units[label] = (; keys=ukeys[sourced], M=M[sourced], Λ=Λ[sourced, sourced])
    end
    kept = findall(!isnan, μ)
    position = Dict(k => p for (p, k) in enumerate(kept))
    receipt = (; slices=length(states),
        globals=(; coordinates=names, informed, clamped, spread, spread_slice,
            carried=collect(carried_globals), joint, heterogeneity, outlying=collect(outlying),
            excluded=collect(excluded),
            prior_mean=prior.mean, prior_sd=[p > 0 ? 1 / sqrt(p) : Inf for p in prior.precision]),
        units=(; copied, partial, cold, carried=carried_units, dropped, mapped, uncarried),
        prior=(; pairs=prior.pairs, iterations=prior.iterations, flat=names[prior.flat]),
        slice_rank=[size(source.Β, 2) for source in sources])
    BRMSVIStart(2, _brm_svi_source_id(m.backend), deepcopy(brm_fitted_state(m.backend)),
        m.view.group, Tuple(keys(m.view.nested_groups)), m.view.shared_groups,
        (; keys=gkeys[kept], μ=μ[kept], σ=σ[kept],
            correlated=[(; positions=[position[k] for k in block.globals], block.precision)
                for block in correlated]),
        units, receipt)
end

# The start's global factor projected onto a diagonal-plus-rank-`rank` one, for
# the globals at `rows` of the fit (`rows[p]` for the start's global p, 0 where
# the fit has no such global). The diagonal is the conditional variance
# 1/P_kk of the start's precision P: the mean-field factor closest to it (in
# KL(q‖p)), and the start's σ. The rank part follows the directions where P is
# widest relative to that diagonal: with D = diag(1/P_kk) and the eigenpairs
# (ω, u) of D^{1/2} P D^{1/2}, each ω < 1 is a direction wider than the
# diagonal says, and Β takes the `rank` smallest, D^{1/2} u √(1/ω − 1), which
# for a fixed diagonal minimizes KL(q‖p) column by column. Blocks are
# independent, so a column lives in one block.
function _brm_svi_start_rank(start, rows, rank)
    candidates = Tuple{Float64,Vector{Int},Vector{Float64}}[]
    for block in get(start.globals, :correlated, ())
        found = findall(p -> rows[p] != 0, block.positions)
        length(found) >= 2 || continue
        P = block.precision[found, found]
        d = 1 ./ sqrt.(diag(P))
        E = eigen(Symmetric(d .* P .* d'))
        for i in eachindex(E.values)
            ω = E.values[i]
            0 < ω < 1 - sqrt(eps()) || continue
            push!(candidates, (ω, rows[block.positions[found]],
                d .* E.vectors[:, i] .* sqrt(1 / ω - 1)))
        end
    end
    sort!(candidates; by=first)
    candidates[1:min(rank, length(candidates))]
end

# The start as a source: its globals informed, with their mean-field scales,
# its units as they are.
_brm_svi_source(start::BRMSVIStart) = _BRMSVISource(collect(Any, start.globals.keys),
    copy(start.globals.μ), start.globals.σ .^ 2, copy(start.globals.σ),
    zeros(length(start.globals.keys), 0), fill(true, length(start.globals.keys)),
    start.units)

# q for `t` from a factor-level start: each global's mean and scale from the
# start's factor for its key, each unit's factor from the start's factors for
# its keys; everything the start does not hold keeps the ordinary default (the
# prior-median point, `global_scale`, `unit_scale`). A `LowRankGlobal` fit's
# rank part takes the start's widest correlated directions
# (`_brm_svi_start_rank`); without correlated blocks it starts at zero, as in
# a cold fit.
function _brm_svi_factor_initialize(t, start, family; inits, global_scale, unit_scale,
        unit_dense_limit=64, unit_rank=8, caller="brm_svi_fit")
    inits === nothing || throw(ArgumentError(
        "$caller: give either a factor-level `start` or `inits`, not both"))
    combined = start isa BRMSVIStart ? start : _brm_svi_combine(t, [_brm_svi_slice_state(
        caller, x, "start[$s]") for (s, x) in enumerate(start isa AbstractVector ? start : [start])];
        caller)
    m = t.m
    _brm_svi_check_slice(caller, m, "the start", combined.source_id, combined.preprocessing,
        combined.group, combined.nested, combined.shared)
    point = _brm_svi_start(m, nothing, nothing)
    q = _brm_svi_params(t, point[t.globals], family; global_scale, unit_scale,
        unit_dense_limit, unit_rank)
    index = _brm_svi_source_index([_brm_svi_source(combined)])
    coordinate_keys = _brm_svi_coordinate_keys(m)
    started = 0
    rows = zeros(Int, length(combined.globals.keys))
    for (k, key) in enumerate(coordinate_keys[t.globals])
        sites = _brm_svi_marginals(index, key)
        isempty(sites) && continue
        _, μk, vk = only(sites)
        q.μ[k] = μk
        q.ρ[k] = log(vk) / 2
        for (_, p) in get(index.globals, key, ())
            rows[p] = k
        end
        started += 1
    end
    for (c, (_, R, b)) in enumerate(_brm_svi_start_rank(combined, rows, size(q.Β, 2)))
        q.Β[R, c] .= b
    end
    units_started = 0
    for (i, block) in enumerate(t.units)
        M, Λ, sourced, _ = _brm_svi_assemble_unit(caller, index, m.view.levels[i],
            coordinate_keys[block], point[block], 1 / unit_scale^2)
        q.M[i] .= M
        if q.Λ[i] isa Matrix
            q.Λ[i] = Λ
            q.U[i] = Matrix(cholesky(Symmetric(Λ)).L)
        else
            # A unit above `unit_dense_limit`: its diagonal-plus-low-rank factor
            # takes the start's marginal sds, its rank part starting at zero.
            rank = size(q.Λ[i].factor.Β, 2)
            f = _BRMSVILowRankUnit(log.(sqrt.(diag(inv(Symmetric(Λ))))), zeros(length(M), rank))
            q.Λ[i], q.U[i] = _BRMSVIUnitPrecision(f), _BRMSVIUnitRoot(f)
        end
        units_started += !isempty(sourced)
    end
    receipt = merge(combined.receipt, (; applied=(; globals=started, units=units_started)))
    merge(t, (; start_receipt=receipt)), q
end

const _BRM_SVI_COMBINE_PHASES = (lower="Lower to ReactiveKernels", prepare="Prepare NLME model",
    target="Minibatch target", combine="Combine slice factors")

"""
    brm_svi_combine(x, slices; exclude=(), rng=Random.Xoshiro(1), prior_pairs=16,
        prior_iterations=150, joint_limit=4096, ad_backend=nothing,
        shared_point=nothing, subject_group=nothing, nested_groups=NamedTuple(),
        shared_groups=(), unit_partition=nothing, global_coordinates=nothing,
        progress=nothing) -> BRMSVIStart

Combine [`brm_svi_fit`](@ref) fits of one declaration on slices of the data
into a factor-level start for the fit of `x`, the same declaration on all of
it: one sweep of expectation propagation on partitioned data (Vehtari et al.
2020, JMLR 21(17)) in the variational form of partitioned VI (Ashman et al.
2022, arXiv:2202.12275). Pass the result as `brm_svi_fit(x; start)`.

- `x` — the full model, as `brm_svi_fit` takes it, with the same preparation
  keywords. Pass a prepared [`BRMNLMEModel`](@ref) to prepare its minibatch
  target once for this call and the fit.
- `slices` — the slices' fit results or their [`brm_svi_checkpoint`](@ref)
  states (a streaming carry's state included). Each slice is the same
  declaration with the same groupings, bound to a subset of the rows, and each
  unit's rows lie in one slice (units nested in the slices, such as stations
  in regions). Its preprocessing must equal `x`'s except the level tables of
  the unit grouping, nested groupings, and shared groupings whose levels reach
  the coordinates only through their random-effect draws: a fitted transform
  (`standardize`, `constant`) fitted on a slice's rows is refused, so bind such
  a slice with `RKBRMI(builder(slice_data), brm_fitted_state(full);
  resample_groups=(unit_grouping,))`. A state from an older BRM is refused; take
  `brm_svi_checkpoint` of the fit again.

Coordinates match across bindings by semantic identity: a random-effect draw
of the units, a nested or a shared grouping by its block, margin and level
label; any other coordinate by its packed name, so a global model-scope
vector must mean the same entry in every slice (one frame for a field's basis,
for example). A unit's packed-name coordinate matches only that unit's factor
in a slice, since an entry of a vector indexed by the units sits at another
position in another binding; where the names differ it keeps the default.

- **Global coordinates** combine as a product of Gaussian sites in
  unconstrained coordinates, `P = P₀ + Σ_s (P_s − P₀)` and `P m = P₀ m₀ +
  Σ_s (P_s m_s − P₀ m₀)`, with the prior counted once. `(m₀, P₀)` is the
  diagonal Gaussian projection of the global prior (the factor a fit on no
  data reaches, from `prior_iterations` natural-gradient steps on
  `prior_pairs` antithetic draws of `rng`); a flat prior has `P₀ = 0`. A slice
  contributes a site only where its data reaches the coordinate (an
  observation reads it, or a prior of a reached coordinate does); a slice
  holding a global inside one of its units contributes that unit factor's
  marginal. `P_s` is the inverse of the slice factor's covariance over the
  coordinates it reaches, so a `LowRankGlobal` slice's rank part enters as
  correlation: a ridge in a slice (a direction its data leaves to the prior,
  such as a level two coefficients share) stays wide in its site, and the
  product lands where the slices' ridges cross, which a coordinate-by-coordinate
  product cannot find. `MeanFieldGlobal` slices hold no correlation, so their
  sites are diagonal; fit the slices with `LowRankGlobal(r)`, `r` at least the
  number of such directions, where they disagree. A joint site holds the
  coordinates a slice narrows (a marginal variance below 0.8 of the prior
  projection's) or carries along a ridge (below 1.2, with at least 0.8 of it in the
  rank part); a rank row on a coordinate the slice leaves at its prior is noise,
  so there the coordinate has a marginal site where it is narrower than the prior
  and none otherwise. A site's directions wider than the prior's projection are
  dropped and counted. The product is dense in the coordinates in joint sites; above
  `joint_limit` of them (or at 0) the globals combine coordinate by coordinate
  through their marginal variances, and they do the same if a joint mean leaves
  the range that supports it: from the lowest to the highest of 3 sds around each
  counted site's mean and the prior's.
  Slices can contradict each other by thousands of their own sds, on a coordinate
  each pins by itself. A plain product would hold that coordinate as narrow as the
  narrowest slice, where few of them put it, and through small correlations move
  coordinates no slice informs. So the slices are read as random effects, as in a
  random-effects meta-analysis, in either product: where Cochran's Q says the
  slices that pin a coordinate contradict (at 0.999, allowing for sds a quarter of
  the posterior's, as SVI's can be), their between-slice variance
  (a Paule-Mandel estimate, `receipt.globals.heterogeneity`) is added to the sites'
  variance, and the coordinate sits among the slices, starting no narrower than the
  product says and no wider than a cold fit's default 0.1. A slice pins a
  coordinate if it narrows it and the rank part does not carry it, or if its
  marginal variance is below 0.01 of the prior projection's; a ridge's marginals
  say little about their coordinates, so ridges that cross keep it at 0. The
  start keeps the product's precision (`globals.correlated`) where it is not
  diagonal; its `σ` is the mean-field part, `1/√P_kk`. A `LowRankGlobal(r)`
  fit from it starts its rank part on the `r` directions in which that
  precision is widest relative to `σ`, and `MeanFieldGlobal` fits start at
  `σ`.
- **Coordinates left out** (`exclude`). A global coordinate whose meaning
  depends on the slice's domain is no estimate of the full model's: a spatial
  field's amplitude fitted on one region measures the field's variation inside
  that region, so slices that partition space all put it low. `spread` and the
  between-slice variance compare the slices with each other, so neither can see
  a bias they share, and the product starts it among them. `exclude` names
  such coordinates, and each takes no site and starts as a coordinate no slice
  informs: at the full model's default start (the prior-median point, with the
  fit's `global_scale`), exactly as if every slice had left it uninformed, a
  joint site included (its marginal over the other coordinates). Draws are
  carried to that default where it is one of their hyperparameters. A name
  (a Symbol or String) selects that coordinate and every coordinate under it:
  `exclude=[:natural_field]` leaves out `natural_field.amplitude` and every
  `natural_field.z.*`; a function of a name selects where it returns `true`. A
  selection of nothing, or a name of no global coordinate, is refused.
  `receipt.globals.excluded` marks them, and `informed` still counts their
  slices' sites.
- **Draws are carried to the combined hyperparameters.** A random-effect draw
  `z` is standardized: its level's effect is `A z`, `A = √w diag(τ) L`, and a
  margin's total adds the population coefficients its block absorbs: those
  the NLME view mu-references into the units' block (`mu ~ 1 + x + (1 + x |
  unit)`: the intercept and `x`), and otherwise a coefficient whose own
  column a shared grouping's block repeats (`(gain, offset) ~ 1 + (1 | ID |
  device)`: each intercept), so that every total moves once. A slice's data
  informs those effects and totals, not `z`, and a copied `z` under the
  combined `τ` means another effect (a slice that fitted a small `τ` holds
  large `z`). So each slice's factor over a level's draws is first carried by
  `z_s = C z_c + d`, `C = A_s⁻¹ A_c`, `d = A_s⁻¹ δ` (`A_s` at the slice's own
  scales, correlation and coefficients, `A_c` at the combined ones, `δ` the
  change of the absorbed coefficients times their design), which keeps the
  effects and totals. A shared level that is a global coordinate has its
  sites (each draw's marginal) carried before they multiply, per level, after
  the other globals; the level's product precision joins
  `globals.correlated` when the carry correlates its draws. One that only a
  single unit reads is that unit's coordinate and is carried with it.
- **Units** take their slice's factor (mean and precision, or the marginal of
  the coordinates the slice holds), carried as a site: the factor is the
  draws' `N(0, I)` prior times the site of the unit's data, and the site moves
  with the map while the prior is applied once, `Λ' = Cᵀ(Λ − I)C + I`,
  `Λ'M' = Cᵀ(ΛM − (Λ − I)d)`. A unit whose data pins its effects keeps its
  totals; one with little data moves to the combined population. Where the
  map would make a negative site (a slice factor wider than the prior)
  indefinite, the negative part is dropped; failing that, the factor is
  carried as a plain change of variables. A unit coordinate with factors in
  two slices is refused: overlapping slices are not combined. Draws of a
  level-correlated block, of a family with a fixed covariance (completions,
  grouped HSGP weights) and other unit coordinates are copied as they are. A
  slice unit's diagonal-plus-low-rank factor enters through its precision; in
  the fit, a unit above `unit_dense_limit` takes the start's mean and marginal
  sds, its rank part starting at zero. Units that are connected components
  (`unit_partition=:connected`), whose members change between bindings, are
  refused.

The combination is a start: the fit's target is the full model. SVI's global
factors are narrower than the posterior, so the product of many slices' sites
starts the globals narrower still; the fit's optimizer widens them. SVI is a
local optimizer, so the start can decide which optimum the fit reaches: a
coordinate the slices bias the same way (a domain-dependent scale) can hold the
fit in an optimum where other coordinates (the units' effects) carry what it
should, even at a higher ELBO than a cold fit's. Compare the fit with a cold
one on held-out data, not only by its ELBO.

`receipt` reports, per global coordinate of `x` (`receipt.globals`, aligned
with `coordinates`): `informed`, how many slices had a site; `clamped`, how
many of those were dropped; `spread`, the largest distance of a slice's mean
from the combined mean in that slice's standard deviations (`NaN` below two
sites; for a carried draw, both in the combined hyperparameters), with
`spread_slice` naming that slice, so a large value marks slices that
disagree on that coordinate's marginal (along a ridge they may still agree
jointly); `carried`, whether its sites were carried (a shared level's draw);
`joint`, how the globals combined (`:joint`, `:diagonal` when no slice has a
rank part, `:limit` above `joint_limit`; `:indefinite` when the joint product
was not positive definite and `:outlying` when one of its means left the range
above, where the coordinate-by-coordinate one was used); `heterogeneity`, the
between-slice variance added to that coordinate's sites; `outlying`, which
coordinates' joint means left that range; `excluded`, which coordinates `exclude`
left out; and the prior projection's `prior_mean` and `prior_sd` (`Inf` where
flat).
`clamped` counts dropped site directions, each at the coordinate it weighs
most. `receipt.slice_rank` lists each slice's global rank. `receipt.units` counts the units `copied`
whole and `carried` (some draws carried), and lists the `partial` and `cold`
(in no slice) labels, the `dropped` ones (a negative site part dropped), the
`mapped` ones (carried as a change of variables) and the `uncarried` ones (whose own
density at the combined globals is clearly better, by over half a nat per draw, with
their slice's draws as they were); `receipt.prior` records
the projection's draws and its `flat` coordinates.
"""
function brm_svi_combine(x, slices; exclude=(), rng=Random.Xoshiro(1), prior_pairs=16,
        prior_iterations=150, joint_limit=4096, progress=nothing, ad_backend=nothing,
        shared_point=nothing, subject_group=nothing, nested_groups=NamedTuple(),
        shared_groups=(), unit_partition=nothing, global_coordinates=nothing)
    caller = "brm_svi_combine"
    slices isa Union{AbstractVector,Tuple} || throw(ArgumentError(
        "$caller: pass the slices' fits or states as a vector"))
    _brm_svi_check_exclude(caller, exclude)
    states = [_brm_svi_slice_state(caller, s, "slice $i") for (i, s) in enumerate(slices)]
    prior_pairs isa Integer && prior_pairs >= 1 && prior_iterations isa Integer &&
        prior_iterations >= 2 || throw(ArgumentError(
        "$caller: prior_pairs must be positive and prior_iterations at least 2"))
    joint_limit isa Integer && joint_limit >= 0 || throw(ArgumentError(
        "$caller: joint_limit must be a nonnegative integer"))
    _brm_with_progress(progress; description="BRM SVI combine") do node
        _brm_with_phases(node, _BRM_SVI_COMBINE_PHASES) do phases
            lowered = _brm_svi_lowered(x, phases.lower; subject_group, nested_groups,
                shared_groups, unit_partition)
            m = _brm_svi_model(lowered, phases.prepare; ad_backend, shared_point,
                subject_group, nested_groups, shared_groups, unit_partition)
            prepared, _ = _brm_run_phase(phases.target) do phase
                _brm_with_cell_progress(() -> _brm_svi_checkout!(m; global_coordinates,
                    previous=nothing), phase)
            end
            start = _brm_run_phase(phases.combine) do _
                _brm_svi_combine(prepared.target, states; caller, rng, prior_pairs,
                    prior_iterations, joint_limit, exclude)
            end
            # As after a fit: a call that throws keeps no target.
            _brm_svi_checkin!(m, prepared)
            start
        end
    end
end
