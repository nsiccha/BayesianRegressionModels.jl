# src/nlme_svi_centering.jl — learned partial centering for `brm_svi_fit`.
#
# A cell is one packed coordinate y whose physical value is b = μ + exp(ℓ) z for
# a location μ, a log scale ℓ and a standardized z, packed as y = c0 μ +
# exp(c0 ℓ) z: a random effect's standardized draw (`c0 = 0`, y = z) or an
# authored Gaussian vector element `x ~ Normal(μ, exp(ℓ))` (`c0 = 1`, y = b). A
# centering control c ∈ (0, 1) gives the cell the variational coordinate of
# Gorinova, Moore and Hoffman (2020)'s VIP,
#
#     u = c μ + exp(c ℓ) z = c μ + exp((c - c0) ℓ) (y - c0 μ),
#
# non-centred (u = z) at c = 0 and centred (u = b) at c = 1. The map u → y is
# ReparametrizableDistributions' partial-centering primitive (`reparam`, from
# `PartiallyCentered(c)` to `PartiallyCentered(c0)`), with log-Jacobian
# (c0 - c) ℓ.
#
# Random effects: the RK model reads a block instance through z, with physical
# effects b = μ0 + A z for the family's lower-triangular natural factor
# A = diag(scale) L and the unit's population location μ0 (its mu-referenced
# population coefficients, zero without one). In the block's margin order, cell
# m given the earlier cells has location μ_m = μ0_m + Σ_{k<m} A[m, k] z_k and log
# scale ℓ_m = log A[m, m]; the map is applied cell by cell in that order, so it
# stays triangular. A level of a block over a `shared_groups` grouping whose
# draws one unit owns is centred in that unit the same way, located at the
# population coefficients its margins absorb (zero without one). Authored
# Gaussian vectors read their law's mean and scale per element,
# `<x>_prior_mean` and `<x>_prior_scale`, which RK names for them.
#
# SVI fits q over u and the controls jointly by the ELBO of
#
#     F_c(g, u) = F(g, y(u; g, c)) + Σ (c0 - c) ℓ(g),
#
# where F is the existing minibatch target. Its gradients come from F's own
# gradients by one reverse sweep through the cells (`_brm_svi_centering_backward!`),
# which returns the adjoints of every location and scale. Those go back to the
# packed coordinates through one RK pullback of the scale, correlation, mean and
# scale values per iteration, so every parameter-dependent transform stays in the
# RK graph and no dense Jacobian is formed.

using ReparametrizableDistributions: PartiallyCentered, reparam

"""
    LearnedCentering(; start=0.5, step=0.05)

The default on a fresh [`brm_svi_fit`](@ref): learned partial centering with one control
`c ∈ (0, 1)` per eligible cell, optimized jointly with the variational factors
by the ELBO (the VIP of Gorinova, Moore and Hoffman, 2020). `c = 0` is the
non-centred coordinate and `c = 1` the centred one: a random effect located at
its population predictor and, in a correlated block, at its conditional mean
given the block's earlier margins; an authored Gaussian vector element at its
law's mean. Controls start at `start` (grouped HSGP weights at `min(start, 0.1)`);
`step` is the Adam step size of
the controls' logits, taken whenever their unit is in the batch.
Pass `centering=nothing` for the plain factorization. Streaming and optimizer
continuation retain the saved policy.
"""
struct LearnedCentering
    start::Float64
    step::Float64
    function LearnedCentering(; start::Real=0.5, step::Real=0.05)
        0 < start < 1 || throw(ArgumentError(
            "LearnedCentering: start must lie strictly between 0 and 1, got $start"))
        isfinite(step) && step > 0 || throw(ArgumentError(
            "LearnedCentering: step must be finite and positive, got $step"))
        new(Float64(start), Float64(step))
    end
end

# Omission selects learned coordinates on a fresh fit. A streaming update keeps
# its saved policy, including old states without a centering field. Explicit
# nothing remains the original independent factorization on a fresh fit.
function _brm_svi_resolve_centering(previous, centering)
    if previous === nothing
        ismissing(centering) && return LearnedCentering()
        centering === nothing || centering isa LearnedCentering || throw(ArgumentError(
            "brm_svi_fit: centering must be LearnedCentering() or nothing, got $(repr(centering))"))
        return centering
    end
    ismissing(centering) || centering === nothing || throw(ArgumentError(
        "brm_svi_fit: a streaming update carries the previous fit's centering; omit centering"))
    hasproperty(previous.layout, :centering) ?
        LearnedCentering(; previous.layout.centering...) : nothing
end

# The logistic of a control's unconstrained value, and its inverse.
_brm_svi_control(x::Real) = x >= 0 ? inv(1 + exp(-x)) : (e = exp(x); e / (1 + e))
_brm_svi_control_logit(c::Real) = log(c) - log1p(-c)

# One cell: y = T(u) and log |∂y/∂u|, through the shared primitive.
@inline _brm_svi_cell(c0, c, u, loc, ℓ) =
    reparam(PartiallyCentered(c0), PartiallyCentered(c), u, loc, ℓ)

# The adjoints of one cell's (u, c, location, log scale) given ȳ, the total
# adjoint of its value y, including the log-Jacobian's own derivatives. With
# E = exp((c0 - c) ℓ) and r = y - c0 μ = (u - c μ) E:
#   ∂y/∂u = E, ∂y/∂c = -μ E - ℓ r, ∂y/∂μ = c0 - c E, ∂y/∂ℓ = (c0 - c) r,
#   ∂(c0 - c) ℓ / ∂c = -ℓ, ∂(c0 - c) ℓ / ∂ℓ = c0 - c.
@inline function _brm_svi_cell_adjoint(c0, c, y, loc, ℓ, ȳ)
    E = exp((c0 - c) * ℓ)
    r = y - c0 * loc
    (ȳ * E, ȳ * (-loc * E - ℓ * r) - ℓ, ȳ * (c0 - c * E), (c0 - c) * (ȳ * r + 1))
end

# A random-effect block instance's cells, in the family's margin order:
# `positions` index the unit's coordinate block, `cells` the unit's control
# vector, `family` the plan's natural factors, and margin m's population location
# reads `location[m]`: pairs (k, w) of the k-th global (a fixed packed value
# `base[-k]` when k < 0) times the design weight w.
struct _BRMSVICenteringBlock
    family::Int
    positions::Vector{Int}
    cells::UnitRange{Int}
    location::Vector{Vector{Tuple{Int,Float64}}}
end

# Authored Gaussian vector elements a unit owns: position p reads element
# `elements[p]` (0 for a law argument shared by every element) of the frame's
# `mean` and `scale` ports.
struct _BRMSVICenteringVector
    mean::Int
    scale::Int
    positions::Vector{Int}
    elements::Vector{Int}
    cells::UnitRange{Int}
end

const _BRMSVICenteringInstance = Union{_BRMSVICenteringBlock,_BRMSVICenteringVector}

_brm_svi_element(x::Number, j) = x
_brm_svi_element(x, j) = x[j]

function _brm_svi_location(inst::_BRMSVICenteringBlock, m, g, base)
    loc = 0.0
    for (k, w) in inst.location[m]
        loc += w * (k > 0 ? g[k] : base[-k])
    end
    loc
end

# y = T(u) for one instance, in place over the unit's block (positions outside
# every instance keep y = u). Records each cell's location and log scale for the
# reverse sweep and returns the log-Jacobian.
function _brm_svi_centering_forward!(y, μ, ℓ, u, inst::_BRMSVICenteringBlock, c, frame, g, base)
    A = frame.A[inst.family]
    logjac = 0.0
    pos = inst.positions
    for m in eachindex(pos)
        loc = _brm_svi_location(inst, m, g, base)
        for k in 1:(m - 1)
            loc += A[m, k] * y[pos[k]]
        end
        cell = inst.cells[m]
        s = log(A[m, m])
        lj, ym = _brm_svi_cell(0.0, c[cell], u[pos[m]], loc, s)
        y[pos[m]], μ[cell], ℓ[cell] = ym, loc, s
        logjac += lj
    end
    logjac
end

function _brm_svi_centering_forward!(y, μ, ℓ, u, inst::_BRMSVICenteringVector, c, frame, g, base)
    mean, scale = frame.values[inst.mean], frame.values[inst.scale]
    logjac = 0.0
    for (n, (p, j)) in enumerate(zip(inst.positions, inst.elements))
        cell = inst.cells[n]
        loc, s = _brm_svi_element(mean, j), log(_brm_svi_element(scale, j))
        lj, y[p] = _brm_svi_cell(1.0, c[cell], u[p], loc, s)
        μ[cell], ℓ[cell] = loc, s
        logjac += lj
    end
    logjac
end

# u = T⁻¹(y): the learned coordinates of packed coordinates y, in place over
# `u` (which holds y elsewhere). Records each cell's location and log scale and
# returns log |∂u/∂y|.
function _brm_svi_centering_inverse!(u, μ, ℓ, y, inst::_BRMSVICenteringBlock, c, frame, g, base)
    A = frame.A[inst.family]
    pos = inst.positions
    logjac = 0.0
    for m in eachindex(pos)
        loc = _brm_svi_location(inst, m, g, base)
        for k in 1:(m - 1)
            loc += A[m, k] * y[pos[k]]
        end
        cell = inst.cells[m]
        s = log(A[m, m])
        lj, u[pos[m]] = _brm_svi_cell(c[cell], 0.0, y[pos[m]], loc, s)
        μ[cell], ℓ[cell] = loc, s
        logjac += lj
    end
    logjac
end

function _brm_svi_centering_inverse!(u, μ, ℓ, y, inst::_BRMSVICenteringVector, c, frame, g, base)
    mean, scale = frame.values[inst.mean], frame.values[inst.scale]
    logjac = 0.0
    for (n, (p, j)) in enumerate(zip(inst.positions, inst.elements))
        cell = inst.cells[n]
        loc, s = _brm_svi_element(mean, j), log(_brm_svi_element(scale, j))
        lj, u[p] = _brm_svi_cell(c[cell], 1.0, y[p], loc, s)
        μ[cell], ℓ[cell] = loc, s
        logjac += lj
    end
    logjac
end

# The packed frame's centering of an instance's cells.
_brm_svi_packed_centering(::_BRMSVICenteringBlock) = 0.0
_brm_svi_packed_centering(::_BRMSVICenteringVector) = 1.0

# The adjoints a unit's reverse sweep leaves for the batch: one per family's
# natural factor, sparse (port, element, value) terms of the vector laws' mean
# and scale ports (element 0 for a scalar port), and (global, value) terms of
# the locations' population coefficients. The batch adds them to the global
# gradient, so a unit never writes a global outside its own touched set.
struct _BRMSVICenteringAdjoint
    A::Vector{Matrix{Float64}}
    ports::Vector{Tuple{Int,Int,Float64}}
    globals::Vector{Tuple{Int,Float64}}
end
_BRMSVICenteringAdjoint(frame) = _BRMSVICenteringAdjoint([zeros(size(A)) for A in frame.A],
    Tuple{Int,Int,Float64}[], Tuple{Int,Float64}[])

# The reverse sweep of `_brm_svi_centering_forward!` plus its log-Jacobian. On
# entry `gy` holds ∂F/∂y over the unit's block; on return it holds ∂F_c/∂u. Adds
# ∂F_c/∂c into `gc` and the scale and location adjoints into `adjoint`.
function _brm_svi_centering_backward!(gy, gc, adjoint, y, μ, ℓ,
        inst::_BRMSVICenteringInstance, c, frame)
    c0 = _brm_svi_packed_centering(inst)
    for n in reverse(eachindex(inst.positions))
        p, cell = inst.positions[n], inst.cells[n]
        ū, c̄, μ̄, ℓ̄ = _brm_svi_cell_adjoint(c0, c[cell], y[p], μ[cell], ℓ[cell], gy[p])
        gy[p] = ū
        gc[cell] += c̄
        _brm_svi_location_adjoint!(gy, adjoint, inst, n, μ̄, ℓ̄, y, ℓ, frame)
    end
    nothing
end

# A unit's y = T(u) over all its instances; returns the log-Jacobian and the
# cells' locations and log scales.
function _brm_svi_unit_centering!(y, u, instances, ncells, c, frame, g, base)
    copyto!(y, u)
    μ, ℓ = zeros(ncells), zeros(ncells)
    logjac = 0.0
    for inst in instances
        logjac += _brm_svi_centering_forward!(y, μ, ℓ, u, inst, c, frame, g, base)
    end
    logjac, μ, ℓ
end

function _brm_svi_unit_centering_backward!(gy, gc, adjoint, y, μ, ℓ, instances, c, frame)
    for inst in Iterators.reverse(instances)
        _brm_svi_centering_backward!(gy, gc, adjoint, y, μ, ℓ, inst, c, frame)
    end
    nothing
end

# ── Which cells a target centers ─────────────────────────────────────────────

# Everything a centered target reads besides the controls: the graph values its
# cells' scales and locations come from (`ports`), with one reader and one
# pullback of them all; the natural-factor families; every unit's instances and
# cell labels; the packed point fixed values come from; and what stays
# non-centred, with the reason.
struct _BRMSVICenteringPlan{R,P}
    start::Float64
    step::Float64
    ports::Tuple{Vararg{Symbol}}
    families::Vector{NamedTuple}
    instances::Vector{Vector{_BRMSVICenteringInstance}}
    ncells::Vector{Int}
    cells::Vector{Vector{NamedTuple}}
    uncentered::Vector{NamedTuple}
    # Authored vector elements of carried units absent from this window, by
    # packed position: (mean port, scale port, element).
    absent::Dict{Int,NTuple{3,Int}}
    # The same for a shared grouping's levels (`shared_groups`), whose draws a
    # carried unit holds by packed name: position => (instance, margin), each
    # instance over its level's margins 1:K, located through the frame's point.
    absent_blocks::Dict{Int,Tuple{Int,Int}}
    absent_instances::Vector{_BRMSVICenteringBlock}
    read::R
    pullback::P
    base::Vector{Float64}
end

_brm_svi_centering(t) = hasproperty(t, :centering) ? t.centering : nothing

# The element index of a vector declaration's packed coordinate `x.<j>`.
function _brm_svi_vector_element(name::Symbol, declaration::Symbol)
    s, prefix = String(name), string(declaration, ".")
    startswith(s, prefix) || error("brm_svi_fit: internal: `$name` is not an element of `$declaration`")
    parse(Int, s[length(prefix)+1:end])
end

# Why a family of standardized draws with a fixed identity covariance stays
# uncentred: its draws' scale is computed inside a kernel, not read as a value.
function _brm_svi_fixed_family_reason(block)
    hasproperty(block, :hsgp) && return "its levels are correlated by `cor=`, so a " *
        "level's weights are not scaled by their spectral scales alone"
    hasproperty(block, :cdar) && return "a walk innovation's scale is computed inside " *
        "its kernel and is not read as a graph value"
    hasproperty(block, :car) && return "a car field draw's scale is not read as a graph value"
    get(block, :completion, false) === true && return "a completion's law scale is not " *
        "read as a graph value"
    "its covariance is fixed"
end

# A grouped HSGP's family as the centering reads it: a diagonal natural factor
# whose scales are the term's spectral scales, the value `<declaration>.<spectral>`
# of the program, taking `level`'s row of it when a hyper-predictor scales `sd`
# or `length_scale` by level (levels × modes).
function _brm_svi_spectral_family(record, level)
    stratum = any(!isnothing, values(record.locals.hypers)) ?
        findfirst(isequal(level), record.levels) : nothing
    stratum === nothing && any(!isnothing, values(record.locals.hypers)) && error(
        "brm_svi_fit: internal: $(repr(level)) is not a level of HSGP `$(record.declaration)`")
    (; scale=Symbol(record.declaration, ".", record.spectral), scales=nothing, L=nothing,
       record.group), stratum
end

# A spectral scale as its cell reads it: floored smoothly in log scale, exactly as
# WarmupHMC's adaptive HSGP centering floors it (`_adaptive_hsgp_floor`), so a
# high mode whose scale underflows to zero keeps a normal Float64 frame for every
# control. The map, its inverse and its log-Jacobian all read the floored scale,
# so the floor moves coordinates, never the target. `_brm_svi_spectral_slope` is
# its derivative in the scale, 1 above the floor's exact range and at zero.
_brm_svi_spectral_floor(s) = exp(_adaptive_hsgp_floor(log(s)))
function _brm_svi_spectral_slope(s)
    s > 0 && log(s) <= _ADAPTIVE_HSGP_LOG_SCALE_EXACT || return 1.0
    _brm_svi_spectral_floor(s) / s / (1 + exp(_ADAPTIVE_HSGP_LOG_SCALE_FLOOR - log(s)))
end

# The cells of target `t`: each instance of an estimated random-effect family
# whose draws one unit owns, located at its mu-referenced population coefficients
# (`BRMNLMEView.mu_references`); each level of a block over a `shared_groups`
# grouping whose draws one unit owns, located at the coefficients its margins
# absorb (`_brm_svi_location_references`); and each authored Gaussian vector
# element a unit owns whose law reads no unit coordinate.
function _brm_svi_centering_plan(t, centering::LearnedCentering)
    m = t.m
    view = m.view
    owner = t.ownership.owner
    local_index = [Dict(j => p for (p, j) in enumerate(block)) for block in t.units]
    global_index = Dict(j => k for (k, j) in enumerate(t.globals))
    ports = Symbol[]
    function port!(name)
        k = findfirst(==(name), ports)
        k === nothing || return k
        push!(ports, name)
        length(ports)
    end
    families = NamedTuple[]
    # Keyed by the view's family for a unit's own block, by (family, level row) for
    # a grouped HSGP's, whose by-level scales are one family per level, and by
    # (draws, stratum) for a shared grouping's block, which the view gives no family.
    family_slot = Dict{Any,Int}()
    function family!(key, record, K, stratum; spectral=false)
        get!(family_slot, key) do
            foreach(port!, _brm_nlme_scale_declarations(record))
            isnothing(record.L) || port!(record.L)
            push!(families, (; record, K, stratum, spectral))
            length(families)
        end
    end
    N = length(t.units)
    instances = [_BRMSVICenteringInstance[] for _ in 1:N]
    ncells = zeros(Int, N)
    cells = [NamedTuple[] for _ in 1:N]
    uncentered = NamedTuple[]
    references = Dict{Int,Vector{NamedTuple}}()
    for r in view.mu_references
        push!(get!(references, r.subject, NamedTuple[]), r)
    end
    for v in eachindex(view.subject_coordinates)
        offset = 0
        for f in view.subject_families[v]
            K = view.family_sizes[f]
            r = (offset + 1):(offset + K)
            offset += K
            family = view.families[f]
            packed = view.subject_coordinates[v][r]
            labels = view.subject_margins[v][r]
            i = owner[first(packed)]
            margins = [k for (k, b) in enumerate(view.block_margins)
                if isequal(b.block, family.block) && b.coefficient !== :mixing]
            location = [Tuple{Int,Float64}[] for _ in 1:K]
            # A grouped HSGP whose levels are the units: each level's standardized
            # weights, scaled by the term's spectral scales (`<id>.spectral_scale`,
            # a row per level when a hyper-predictor scales them by level).
            hsgp = family.fixed !== nothing && hasproperty(family.block, :hsgp) &&
                !family.record.level_correlation
            reason = family.fixed !== nothing && !hsgp ?
                    _brm_svi_fixed_family_reason(family.block) :
                get(family.record, :mixing, nothing) !== nothing ?
                    "a Student-t block's scale reads its unit's mixing weights" :
                i == 0 || any(j -> owner[j] != i, packed) ? "its draws are not one unit's" :
                    nothing
            if reason === nothing
                length(margins) == K || error("brm_svi_fit: internal: family `$(family.block)` " *
                    "has $(length(margins)) margins for $K draws")
                for k in 1:K, ref in get(references, margins[k], NamedTuple[])
                    p, w = ref.population, Float64(ref.design[v])
                    if haskey(global_index, p)
                        push!(location[k], (global_index[p], w))
                    elseif owner[p] == 0
                        push!(location[k], (-p, w))
                    else
                        reason = "its population location is a unit coordinate"
                    end
                end
            end
            if reason !== nothing
                push!(uncentered, (; kind=hasproperty(family.block, :hsgp) ? :hsgp : :ranef,
                    block=family.block, level=first(labels).level,
                    coordinates=view.coordinates[packed], reason))
                continue
            end
            slot = if hsgp
                record, stratum = _brm_svi_spectral_family(family.record, first(labels).level)
                family!((f, stratum), record, K, stratum; spectral=true)
            else
                family!(f, family.record, K, family.stratum)
            end
            cellrange = (ncells[i] + 1):(ncells[i] + K)
            ncells[i] += K
            push!(instances[i], _BRMSVICenteringBlock(slot,
                [local_index[i][j] for j in packed], cellrange, location))
            for k in 1:K
                push!(cells[i], (; kind=hsgp ? :hsgp : :ranef, labels[k].block,
                    labels[k].predictor, labels[k].coefficient, labels[k].level,
                    coordinate=view.coordinates[packed[k]], located=!isempty(location[k])))
            end
        end
    end
    absent_set = Set(hasproperty(t.ownership, :absent) ? t.ownership.absent : Int[])
    # Blocks over a grouping crossed with the units (`shared_groups`): the view
    # reads every draw as a shared latent, but inference ownership gives a level
    # that one unit reads to that unit, so its draws are centred there as a
    # unit's own block's are. A level a carried absent unit holds keeps an
    # instance over its margins, so its draws map through that unit's controls.
    absent_blocks = Dict{Int,Tuple{Int,Int}}()
    absent_instances = _BRMSVICenteringBlock[]
    shared_records = [r for r in _brm_nlme_records(m.backend.plan)
        if r.kind in (:ranef, :stratified_ranef) && r.group in view.shared_groups]
    references = isempty(shared_records) ? nothing : _brm_svi_location_references(m)
    position = Dict(name => j for (j, name) in enumerate(view.coordinates))
    columns = m.backend.plan.columns
    for record in shared_records
        K = length(record.margins)
        levels = _rk_ranef_record_levels(m.backend, record)
        record_block = _brm_nlme_block_identity(record)
        stratified = record.kind === :stratified_ranef
        strata = stratified ? Any[_rk_grouping_levels(columns[record.by])...] : nothing
        stratum_of = stratified ? _brm_level_strata(columns[record.group], columns[record.by],
            levels, strata, record; caller="brm_svi_fit", unseen=0) : nothing
        # The view checks none of these for a shared grouping's block.
        block_reason = get(record, :mixing, nothing) !== nothing ?
                "a Student-t block's scale reads its levels' mixing weights" :
            haskey(record, :level_correlation) ?
                "its levels are correlated with each other (`gr(...; cor=...)`)" :
            haskey(record, :known_scales) || haskey(record, :known_cor) ?
                "its scale or correlation is a known value, not read as a graph value" :
                nothing
        for (g, level) in enumerate(levels)
            packed = [get(position, Symbol(record.z, ".", g, ".", k), 0) for k in 1:K]
            any(iszero, packed) && continue    # not a packed draw of this binding
            s = stratified ? stratum_of[g] : nothing
            block = stratified && s != 0 ? (; record_block..., stratum=strata[s]) : record_block
            i = owner[first(packed)]
            absent = all(in(absent_set), packed)
            reason = block_reason !== nothing ? block_reason :
                s == 0 ? "its level has no row, so its stratum is unknown" :
                absent ? nothing :
                i == 0 || any(j -> owner[j] != i, packed) ? "its draws are not one unit's" :
                    nothing
            location = [Tuple{Int,Float64}[] for _ in 1:K]
            for (k, (predictor, coefficient)) in enumerate(record.margins), (p, _) in
                    get(references, _brm_svi_margin_key(record_block, predictor, coefficient), ())
                reason === nothing || break
                # A shared level's design is 1 (`_brm_svi_location_change`). An
                # absent level reads its location from the frame's whole point.
                if absent || owner[p] == 0
                    push!(location[k], (absent || !haskey(global_index, p) ? -p :
                        global_index[p], 1.0))
                else
                    reason = "its population location is a unit coordinate"
                end
            end
            if reason !== nothing
                push!(uncentered, (; kind=:ranef, block, level,
                    coordinates=view.coordinates[packed], reason))
                continue
            end
            slot = family!((record.z, s), record, K, s)
            if absent
                push!(absent_instances, _BRMSVICenteringBlock(slot, collect(1:K), 1:K, location))
                for k in 1:K
                    absent_blocks[packed[k]] = (length(absent_instances), k)
                end
                continue
            end
            cellrange = (ncells[i] + 1):(ncells[i] + K)
            ncells[i] += K
            push!(instances[i], _BRMSVICenteringBlock(slot,
                [local_index[i][j] for j in packed], cellrange, location))
            for (k, (predictor, coefficient)) in enumerate(record.margins)
                push!(cells[i], (; kind=:ranef, block, predictor, coefficient, level,
                    coordinate=view.coordinates[packed[k]], located=!isempty(location[k])))
            end
        end
    end
    absent = Dict{Int,NTuple{3,Int}}()
    for record in _brm_nlme_records(m.backend.plan)
        record.kind === :vector && get(record, :gaussian, nothing) !== nothing || continue
        owned = Dict{Int,Vector{Tuple{Int,Int}}}()
        missing_elements = Tuple{Int,Int}[]
        for (p, name) in enumerate(view.coordinates)
            _brm_nlme_owned(name, record.declaration) || continue
            j = _brm_svi_vector_element(name, record.declaration)
            p in absent_set && push!(missing_elements, (p, j))
            owner[p] > 0 && push!(get!(owned, owner[p], Tuple{Int,Int}[]), (p, j))
        end
        isempty(owned) && isempty(missing_elements) && continue
        law = (record.gaussian.mean, record.gaussian.scale)
        for (p, j) in missing_elements
            absent[p] = (port!(law[1]), port!(law[2]), j)
        end
        isempty(owned) && continue
        reads = _rk_spl_value_reads(m.backend, law, m.unit)
        if any(p -> owner[p] > 0, reads)
            for i in sort!(collect(keys(owned)))
                list = owned[i]
                push!(uncentered, (; kind=:vector, record.declaration, level=nothing,
                    coordinates=view.coordinates[first.(list)],
                    reason="its law's mean or scale reads a unit coordinate"))
            end
            continue
        end
        mean, scale = port!(law[1]), port!(law[2])
        for i in sort!(collect(keys(owned)))
            list = owned[i]
            cellrange = (ncells[i] + 1):(ncells[i] + length(list))
            ncells[i] += length(list)
            push!(instances[i], _BRMSVICenteringVector(mean, scale,
                [local_index[i][p] for (p, _) in list], last.(list), cellrange))
            for (p, j) in list
                push!(cells[i], (; kind=:vector, record.declaration, element=j,
                    coordinate=view.coordinates[p]))
            end
        end
    end
    # Shared field weights are global coordinates. A gmrf/car/cdar field's scales
    # are computed inside its kernel; an HSGP's are a graph value, but its shared
    # weights are globals, and a grouped HSGP whose levels are the units is a
    # family above.
    globals = Set(t.globals)
    for record in _brm_nlme_records(m.backend.plan)
        record.kind in (:hsgp, :gmrf_footprint, :car, :cdar) || continue
        if record.kind === :hsgp
            weights = Symbol(record.declaration, ".",
                record.grouped ? record.locals.weights : :beta_raw)
            shared = [p for (p, name) in enumerate(view.coordinates)
                if _brm_nlme_owned(name, weights) && p in globals]
            isempty(shared) || push!(uncentered, (; kind=:hsgp, record.declaration,
                level=nothing, coordinates=view.coordinates[shared],
                reason="an HSGP term's weights are shared by the units (global " *
                    "coordinates); learned centering covers a grouped HSGP whose " *
                    "levels are the units"))
            continue
        end
        owned = [p for (p, name) in enumerate(view.coordinates)
            if _brm_nlme_owned(name, record.declaration)]
        shared = [p for p in owned if p in globals && view.role[p] === :shared_latent]
        isempty(shared) || push!(uncentered, (; kind=record.kind,
            record.declaration, level=nothing, coordinates=view.coordinates[shared],
            reason="a shared field's coordinates are global, and its weights' scales " *
                "are computed inside its kernel, not read as graph values"))
        local_states = [p for p in owned if owner[p] > 0]
        isempty(local_states) || push!(uncentered, (; kind=record.kind,
            record.declaration, level=nothing, coordinates=view.coordinates[local_states],
            reason="a field's unit-owned states have conditional scales computed inside " *
                "its kernel, not read as graph values"))
    end
    # `@plate` cell latents and the sampled locals of a component called in a cell: the
    # plan reads no location or scale for them, so they keep their packed coordinates
    # (linear coupling still conditions a unit's on the globals, src/nlme_svi_coupling.jl).
    for record in _brm_nlme_records(m.backend.plan)
        record.kind in (:cell_latent, :cell_component) || continue
        owned = [p for (p, name) in enumerate(view.coordinates)
            if _brm_nlme_owned(name, record.declaration)]
        what = record.kind === :cell_latent ? "an authored `@plate` cell latent" :
            "a sampled local of a component called in a `@plate` cell"
        local_entries = [p for p in owned if owner[p] > 0]
        isempty(local_entries) || push!(uncentered, (; kind=record.kind,
            record.declaration, record.name, level=nothing,
            coordinates=view.coordinates[local_entries],
            reason="$what: learned centering reads no location or scale for it"))
        shared = [p for p in owned if p in globals]
        isempty(shared) || push!(uncentered, (; kind=record.kind,
            record.declaration, record.name, level=nothing,
            coordinates=view.coordinates[shared],
            reason="$what whose coordinates are global, not a unit's"))
    end
    names = Tuple(ports)
    read = isempty(names) ? nothing : _rk_value_reader(m.backend, names)
    pullback = isempty(names) ? nothing : _rk_value_pullback(m.backend, names, copy(m.unit))
    _BRMSVICenteringPlan(centering.start, centering.step, names, families, instances, ncells, cells,
        uncentered, absent, absent_blocks, absent_instances, read, pullback, copy(m.unit))
end

# The values every cell of an iteration reads, at the globals `g`: the ports'
# graph values and each family's natural factor, scales and correlation factor.
function _brm_svi_centering_frame(plan::_BRMSVICenteringPlan, t, g)
    point = copy(plan.base)
    point[t.globals] .= g
    plan.read === nothing && return (; point, nt=NamedTuple(), values=(), A=Matrix{Float64}[],
        scales=Vector{Float64}[], L=Matrix{Float64}[], raw=Vector{Float64}[])
    nt = plan.read(point)
    # An HSGP family's scales are its spectral scales, floored; `raw` keeps them
    # unfloored for the floor's slope in the gradient.
    raw = [f.spectral ? _brm_nlme_stratum_scales(Float64.(_brm_rk_declaration_value(nt,
        f.record.scale)), f.stratum) : Float64[] for f in plan.families]
    scales = [f.spectral ? _brm_svi_spectral_floor.(r) :
        _brm_nlme_joint_scale(nt, f.record, f.K, f.stratum)
        for (f, r) in zip(plan.families, raw)]
    A = [f.spectral ? Matrix{Float64}(Diagonal(s)) :
        Matrix{Float64}(_brm_nlme_natural_factor(nt, f.record, f.K, f.stratum))
        for (f, s) in zip(plan.families, scales)]
    L = [Diagonal(1 ./ s) * a for (s, a) in zip(scales, A)]
    (; point, nt, values=Tuple(nt), A, scales, L, raw)
end

# The gradient over the globals of the batch's summed scale and location
# adjoints (each unit's weighted as its term is): A = diag(s) L puts
# s̄[m] = Σ_k Ā[m, k] L[m, k] and L̄ = diag(s) Ā on the family's ports, and one RK
# pullback takes every port's cotangent to the packed coordinates.
function _brm_svi_centering_gradient(plan::_BRMSVICenteringPlan, t, frame, Ā, terms)
    plan.pullback === nothing && return zeros(length(t.globals))
    cot = Any[x isa Number ? 0.0 : zeros(Float64, size(x)) for x in frame.values]
    slot(name) = findfirst(==(name), plan.ports)
    for (ϕ, f) in enumerate(plan.families)
        record, stratum = f.record, f.stratum
        s̄ = vec(sum(Ā[ϕ] .* frame.L[ϕ]; dims=2))
        # Through the floor of an HSGP family's spectral scales.
        f.spectral && (s̄ .*= _brm_svi_spectral_slope.(frame.raw[ϕ]))
        if isnothing(record.scales)
            p = slot(record.scale)
            if stratum === nothing
                cot[p] isa Number ? (cot[p] += only(s̄)) : (cot[p] .+= s̄)
            else
                view(cot[p], stratum, :) .+= s̄
            end
        else
            for (k, name) in enumerate(record.scales)
                name === nothing && continue    # a known margin's scale is no port
                p = slot(name)
                stratum === nothing ? (cot[p] += s̄[k]) : (cot[p][stratum] += s̄[k])
            end
        end
        if !isnothing(record.L)
            L̄ = Diagonal(frame.scales[ϕ]) * Ā[ϕ]
            p = slot(record.L)
            stratum === nothing ? (cot[p] .+= L̄) : (view(cot[p], :, :, stratum) .+= L̄)
        end
    end
    for (p, j, v) in terms
        cot[p] isa Number ? (cot[p] += v) : (cot[p][j] += v)
    end
    plan.pullback(frame.point, Tuple(cot))[t.globals]
end

# Each unit's cell positions in its coordinate block, in cell order.
function _brm_svi_cell_positions(plan::_BRMSVICenteringPlan, i)
    positions = zeros(Int, plan.ncells[i])
    for inst in plan.instances[i], (n, p) in enumerate(inst.positions)
        positions[inst.cells[n]] = p
    end
    positions
end

# The highest control an HSGP weight's cell starts from. Its log scale moves
# steeply with the length scale on the high modes, by ρ ω² / 2 per unit of ρ, so
# with c well above 0 the factor over u makes z = u s^-c swing with every draw of
# ρ. The ELBO answers by collapsing q(ρ), and once ρ is narrow nothing pushes c
# down: from 0.5 a toy's length scale ended 1.7 NUTS sds off with an sd of 0.02
# NUTS's, and the ELBO 5-6 nats below the plain fit's. From a low start the modes
# the data inform climb to about 0.8 and the others stay near 0. Measured on
# `hsgp_fit`'s toy in test/nlme_svi_centering.jl: 0.05 and 0.1 gave the same
# ELBO, about 2 nats above plain, and the collapse begins by 0.2.
const _BRM_SVI_SPECTRAL_START = 0.1

_brm_svi_cell_start(plan::_BRMSVICenteringPlan, cell) =
    cell.kind === :hsgp ? min(plan.start, _BRM_SVI_SPECTRAL_START) : plan.start

# The controls of a target's cells: their logits, one Adam per unit and each
# cell's position in its unit's block. A carried unit's cells start from its
# previous logits, one per coordinate (`NaN` where none was learned).
function _brm_svi_centering_controls(plan::_BRMSVICenteringPlan; carried=nothing)
    positions = [_brm_svi_cell_positions(plan, i) for i in eachindex(plan.ncells)]
    x = map(eachindex(positions)) do i
        old = carried === nothing ? nothing : carried[i]
        [old === nothing || isnan(old[p]) ?
            _brm_svi_control_logit(_brm_svi_cell_start(plan, plan.cells[i][k])) : old[p]
            for (k, p) in enumerate(positions[i])]
    end
    (; plan.start, plan.step, x, opt=[_BRMSVIAdam(n, plan.step) for n in plan.ncells],
       positions)
end

# Unit i's logits as one value per coordinate of its block, `NaN` off the cells.
function _brm_svi_coordinate_controls(q, i, d)
    x = fill(NaN, d)
    x[q.centering.positions[i]] .= q.centering.x[i]
    x
end

# A resumed target's plan, which its checkpoint's controls must fit cell for cell.
function _brm_svi_centered_target(t, q)
    plan = _brm_svi_centering_plan(t, LearnedCentering(; q.centering.start,
        q.centering.step))
    length.(q.centering.x) == plan.ncells || throw(ArgumentError(
        "brm_svi_resume: the checkpoint's centering controls do not fit this target's cells"))
    merge(t, (; centering=plan))
end

_brm_svi_unit_controls(q, i) = _brm_svi_control.(q.centering.x[i])

# Unit i's learned coordinates u → its packed coordinates y at frame `frame`.
function _brm_svi_centered_coordinates(plan::_BRMSVICenteringPlan, frame, q, i, g, u)
    y = similar(u)
    _brm_svi_unit_centering!(y, u, plan.instances[i], plan.ncells[i],
        _brm_svi_unit_controls(q, i), frame, g, plan.base)
    y
end

# Packed coordinates y → learned coordinates u for unit i.
function _brm_svi_learned_coordinates(plan::_BRMSVICenteringPlan, frame, q, i, g, y)
    u = copy(y)
    c = _brm_svi_unit_controls(q, i)
    μ, ℓ = zeros(plan.ncells[i]), zeros(plan.ncells[i])
    for inst in plan.instances[i]
        _brm_svi_centering_inverse!(u, μ, ℓ, y, inst, c, frame, g, plan.base)
    end
    u
end

# Unit i's centered term at learned coordinates u: F's unit term at y = T(u)
# plus the log-Jacobian. Adds ∂/∂u into `gu` (zero on entry) and the unit term's
# global gradient into `gg`, as `_brm_svi_unit_local!` does, and returns the
# value, ∂/∂c and the scale and location adjoints. A carried streaming unit's
# prior is its transported factor, in `_brm_svi_unit_local!`'s order.
function _brm_svi_centered_unit_term!(gg, gu, centering::_BRMSVICenteringPlan, frame, t, q,
        i, g, u, plan=t.plan, work=first(t.work))
    c = _brm_svi_unit_controls(q, i)
    y = similar(u)
    logjac, μ, ℓ = _brm_svi_unit_centering!(y, u, centering.instances[i],
        centering.ncells[i], c, frame, g, centering.base)
    gc = zeros(centering.ncells[i])
    adjoint = _BRMSVICenteringAdjoint(frame)
    prior = t.prior
    value = if prior isa _BRMSVIStreamingPrior && prior.carried[i]
        _brm_svi_unit_partition!(gg, gu, t, i, g, y, plan) +
            _brm_svi_transported_prior!(gu, adjoint, centering, frame, prior.units, i, g, y,
                prior.controls[i]) +
            _brm_svi_field_term!(gg, gu, t, i, y, work)
    else
        _brm_svi_unit_local!(gg, gu, t, i, g, y, plan, work)
    end
    _brm_svi_unit_centering_backward!(gu, gc, adjoint, y, μ, ℓ, centering.instances[i], c,
        frame)
    value + logjac, gc, adjoint
end

# Each cell's learned control, one row per cell in unit order.
function _brm_svi_centering_report(plan::_BRMSVICenteringPlan, q, levels)
    rows = NamedTuple[]
    for i in eachindex(plan.cells), (k, cell) in enumerate(plan.cells[i])
        push!(rows, (; unit=levels[i], cell..., c=_brm_svi_control(q.centering.x[i][k])))
    end
    (; cells=rows, uncentered=plan.uncentered, start=plan.start)
end

# The batch's scale and location adjoints, each unit's antithetic pair weighted
# as its global gradient is (`weights[k] / 2` per draw), summed in batch order.
function _brm_svi_centering_batch_gradient(plan::_BRMSVICenteringPlan, t, frame, centered,
        weights)
    Ā = [zeros(size(A)) for A in frame.A]
    terms = Tuple{Int,Int,Float64}[]
    locations = zeros(length(t.globals))
    for (k, unit) in enumerate(centered)
        # A unit evaluated on an orthogonal design (`unit_curvature = :orthogonal`)
        # brings 2d draws, each weighted 1/d of a pair's draw.
        w = weights[k] / 2 * (hasproperty(unit, :weight) ? unit.weight : 1.0)
        for a in unit.adjoints
            for (ϕ, x) in enumerate(a.A)
                Ā[ϕ] .+= w .* x
            end
            for (p, j, v) in a.ports
                push!(terms, (p, j, w * v))
            end
            for (p, v) in a.globals
                locations[p] += w * v
            end
        end
    end
    _brm_svi_centering_gradient(plan, t, frame, Ā, terms) .+ locations
end

# A sparse global's site stands for its reader's whole term (src/nlme_svi_sites.jl),
# which under centering reads the global through its cells' locations and scales
# as well: the sd of a `gr(g, by=b)` stratum few units belong to, say. At c = 1 that
# chain term all but cancels the term's direct gradient there, so the site must
# see both. Adds each batch unit's chain term, summed over its antithetic pair as
# `gg` is, at its sparse entries. `chain` is the batch's weighted sum
# (`_brm_svi_centering_batch_gradient`), and a cell's locations and scales are
# read by its own unit's term, so a sparse global with one reader in the batch
# takes the sum whole; one with several takes each reader's own pullback.
_brm_svi_centering_site_terms!(terms, ::Nothing, plan, t, frame, centered, batch, weights,
    chain) = terms
function _brm_svi_centering_site_terms!(terms, sites::_BRMSVISites, plan, t, frame,
        centered, batch, weights, chain)
    readers = zeros(Int, length(sites.globals))
    for i in batch, entry in sites.entries[i]
        readers[sites.slot[_brm_svi_unit_touched(t, i)[entry]]] += 1
    end
    for (k, i) in enumerate(batch)
        entries = sites.entries[i]
        isempty(entries) && continue
        touched = _brm_svi_unit_touched(t, i)
        own = nothing
        for entry in entries
            p = touched[entry]
            terms[k].gg[entry] += if readers[sites.slot[p]] == 1
                2chain[p] / weights[k]
            else
                own === nothing && (own = _brm_svi_centering_batch_gradient(plan, t, frame,
                    (centered[k],), (2.0,)))
                own[p]
            end
        end
    end
    terms
end

# One Adam ascent step of each batch unit's control logits along its own term's
# gradient (∂/∂x = ∂/∂c · c (1 - c)), as its factor steps on its own term.
function _brm_svi_control_steps!(q, batch, centered)
    for (k, i) in enumerate(batch)
        x = q.centering.x[i]
        isempty(x) && continue
        c = _brm_svi_control.(x)
        _brm_svi_step!(q.centering.opt[i], x, centered[k].gc .* c .* (1 .- c))
    end
    nothing
end

# Learned centering on a target and its fresh factors: the plan, the controls at
# their start, and each unit's factor moved into its learned coordinates at the
# starting globals.
# At fixed globals the learned map is affine, u = C y + b. Keep one small
# triangular block per correlated effect and diagonal maps for vector cells;
# never construct a Jacobian over all units or globals. The natural factor and
# location also anchor a saved random effect's physical value during slice carry.
function _brm_svi_centering_affine(inst::_BRMSVICenteringBlock, c, frame, g, base)
    A = frame.A[inst.family]
    controls = c[inst.cells]
    location = [_brm_svi_location(inst, k, g, base) for k in eachindex(inst.positions)]
    diagonal = exp.(controls .* log.(diag(A)))
    C = if isdiag(A)
        Diagonal(diagonal)
    else
        result = zeros(size(A))
        for k in axes(A, 1)
            result[k, k] = diagonal[k]
            for j in 1:(k - 1)
                result[k, j] = controls[k] * A[k, j]
            end
        end
        LowerTriangular(result)
    end
    (; positions=copy(inst.positions), C, b=controls .* location,
       natural=(; factor=isdiag(A) ? Diagonal(diag(A)) : copy(A), location))
end

function _brm_svi_centering_affine(inst::_BRMSVICenteringVector, c, frame, g, base)
    mean, scale = frame.values[inst.mean], frame.values[inst.scale]
    controls = c[inst.cells]
    location = [_brm_svi_element(mean, j) for j in inst.elements]
    diagonal = [exp((controls[k] - 1) * log(_brm_svi_element(scale, j)))
        for (k, j) in enumerate(inst.elements)]
    (; positions=copy(inst.positions), C=Diagonal(diagonal),
       b=(controls .- diagonal) .* location, natural=nothing)
end

function _brm_svi_centering_maps(t, q)
    plan = _brm_svi_centering(t)
    plan === nothing && return nothing
    frame = _brm_svi_centering_frame(plan, t, q.μ)
    [[_brm_svi_centering_affine(inst, _brm_svi_unit_controls(q, i), frame, q.μ, plan.base)
        for inst in instances] for (i, instances) in enumerate(plan.instances)]
end

# Transport an existing unit Gaussian into its new learned frame. Dense units
# retain all covariance, including the triangular map of a correlated block.
# A structured unit retains the transformed rank part and the marginal variance
# of the transformed diagonal noise; off-diagonal noise is projected away.
function _brm_svi_unit_affine!(q, i, maps)
    Λ = q.Λ[i]
    for map in maps
        P, C = map.positions, map.C
        q.M[i][P] .= C * q.M[i][P] + map.b
        if Λ isa _BRMSVIUnitPrecision
            f = Λ.factor
            variance = exp.(2 .* f.ρ[P])
            f.ρ[P] .= log.(C isa Diagonal ? abs2.(diag(C)) .* variance : abs2.(C) * variance) ./ 2
            f.Β[P, :] .= C * f.Β[P, :]
        else
            Λ[:, P] .= Λ[:, P] / C
            Λ[P, :] .= C' \ Λ[P, :]
        end
    end
    if !(Λ isa _BRMSVIUnitPrecision) && !isempty(maps)
        Λ .= (Λ + Λ') / 2
        q.U[i] .= cholesky(Symmetric(Λ)).L
    end
    q
end

function _brm_svi_with_centering(t, q, centering::LearnedCentering)
    plan = _brm_svi_centering_plan(t, centering)
    prior = hasproperty(t, :prior) && t.prior isa _BRMSVIStreamingPrior ? t.prior : nothing
    carried = prior === nothing ? nothing : prior.controls
    q = merge(q, (; centering=_brm_svi_centering_controls(plan; carried)))
    t = merge(t, (; centering=plan))
    maps = _brm_svi_centering_maps(t, q)
    for i in eachindex(t.units)
        plan.ncells[i] == 0 && continue
        # A carried unit's factor is already in its previous controls' frame.
        prior !== nothing && prior.carried[i] && continue
        _brm_svi_unit_affine!(q, i, maps[i])
    end
    t, q
end

# ── Streaming: a carried unit's prior in its previous controls' frame ────────

# A carried unit's factor is a Gaussian over its previous learned coordinates
# u_old = T_old⁻¹(y), with controls `x_old` (one logit per coordinate, `NaN`
# where none was learned, which is that coordinate's packed frame). As a density
# over the packed y it is q_old(T_old⁻¹(y)) |∂T_old⁻¹/∂y|, whatever the current
# controls; this returns its value, adds its gradient into `gy` and its scale and
# location adjoints into `adjoint`. Only the last window's frame is
# carried: each window's factors are its next window's whole prior.
function _brm_svi_transported_prior!(gy, adjoint, plan::_BRMSVICenteringPlan, frame,
        units, i, g, y, x_old)
    n = plan.ncells[i]
    c = zeros(n)
    for inst in plan.instances[i], (k, p) in enumerate(inst.positions)
        c[inst.cells[k]] = isnan(x_old[p]) ? _brm_svi_packed_centering(inst) :
            _brm_svi_control(x_old[p])
    end
    u, μ, ℓ = copy(y), zeros(n), zeros(n)
    logjac = 0.0
    for inst in plan.instances[i]
        logjac += _brm_svi_centering_inverse!(u, μ, ℓ, y, inst, c, frame, g, plan.base)
    end
    δ = u - units.M[i]
    ū = -(units.Λ[i] * δ)
    value = dot(δ, ū) / 2 + _brm_svi_unit_logdet_root(units.U[i]) - length(u) * log(2π) / 2 +
        logjac
    # u = y off the cells; each cell's u reads its own y and, in a block, the
    # earlier cells' y through its location.
    gy .+= ū
    for inst in plan.instances[i], (k, p) in enumerate(inst.positions)
        cell = inst.cells[k]
        ȳ, _, μ̄, ℓ̄ = _brm_svi_cell_adjoint(c[cell], _brm_svi_packed_centering(inst), u[p],
            μ[cell], ℓ[cell], ū[p])
        gy[p] += ȳ - ū[p]
        _brm_svi_location_adjoint!(gy, adjoint, inst, k, μ̄, ℓ̄, y, ℓ, frame)
    end
    value
end

# A cell's location and log-scale adjoints μ̄, ℓ̄ onto what they read.
function _brm_svi_location_adjoint!(gy, adjoint, inst::_BRMSVICenteringBlock, m, μ̄, ℓ̄,
        y, ℓ, frame)
    A, Ā = frame.A[inst.family], adjoint.A[inst.family]
    for (k, w) in inst.location[m]
        k > 0 && push!(adjoint.globals, (k, w * μ̄))
    end
    for k in 1:(m - 1)
        Ā[m, k] += μ̄ * y[inst.positions[k]]
        gy[inst.positions[k]] += μ̄ * A[m, k]
    end
    Ā[m, m] += ℓ̄ / A[m, m]
    nothing
end
function _brm_svi_location_adjoint!(gy, adjoint, inst::_BRMSVICenteringVector, n, μ̄, ℓ̄,
        y, ℓ, frame)
    j = inst.elements[n]
    push!(adjoint.ports, (inst.mean, j, μ̄), (inst.scale, j, ℓ̄ / exp(ℓ[inst.cells[n]])))
    nothing
end

# An absent carried unit's draw `z` in its learned frame, mapped to the packed
# coordinates at `destination` (the draw's frame): its centered vector elements,
# and its centred levels of a shared grouping's block, each in margin order.
function _brm_svi_absent_draw!(z, plan::_BRMSVICenteringPlan, frame, unit, source, destination)
    hasproperty(unit, :centering) || return z
    levels = Dict{Int,Vector{Int}}()
    for (k, j) in zip(source, destination)
        x = unit.centering[k]
        isnan(x) && continue
        if haskey(plan.absent_blocks, j)
            b, n = plan.absent_blocks[j]
            get!(() -> zeros(Int, length(plan.absent_instances[b].positions)), levels, b)[n] = k
            continue
        end
        haskey(plan.absent, j) || throw(ArgumentError("brm_svi_fit: the absent " *
            "carried unit's coordinate `$(unit.keys[k].coordinate)` was centred, but " *
            "this window cannot centre it, so its carried factor cannot be mapped to " *
            "the packed draws"))
        mean, scale, element = plan.absent[j]
        _, z[k] = _brm_svi_cell(1.0, _brm_svi_control(x), z[k],
            _brm_svi_element(frame.values[mean], element),
            log(_brm_svi_element(frame.values[scale], element)))
    end
    for (b, ks) in levels
        all(>(0), ks) || error("brm_svi_fit: internal: an absent unit's centred level " *
            "carries controls for only some of its margins")
        u = z[ks]
        y = similar(u)
        μ, ℓ = zero(u), zero(u)
        _brm_svi_centering_forward!(y, μ, ℓ, u, plan.absent_instances[b],
            _brm_svi_control.(unit.centering[ks]), frame, nothing, frame.point)
        z[ks] .= y
    end
    z
end

# ── Handing learned controls to WarmupHMC ────────────────────────────────────

_brm_svi_control_key(group, predictor, coefficient, level) =
    (Symbol(group), Symbol(predictor), Symbol(coefficient), string(_brm_transport_level(level)))

"""
    brm_svi_centeredness(fit, controls) -> Vector{Float64}

A learned-centering SVI fit's controls as the `centeredness` vector of
[`adaptive_centering_problem`](@ref), in the order of `controls`, the rows
[`adaptive_centering_controls`](@ref) reports for that wrapper: each random-effect
row whose grouping, margin and level a fit cell shares takes that cell's `c`; every
other row keeps its current `c`. WarmupHMC then adapts from there. The wrapper
locates a cell at its matched population coefficient (and containing terms), where
SVI locates it at all its mu-referenced coefficients, so the controls are a start,
not the same parametrization.
"""
function brm_svi_centeredness(fit, controls)
    report = hasproperty(fit, :diagnostics) && hasproperty(fit.diagnostics, :centering) ?
        fit.diagnostics.centering : nothing
    report === nothing && throw(ArgumentError(
        "brm_svi_centeredness: expected a brm_svi_fit result with centering=LearnedCentering()"))
    learned = Dict(_brm_svi_control_key(cell.block.group, cell.predictor, cell.coefficient,
        cell.level) => cell.c for cell in report.cells if cell.kind === :ranef)
    map(controls) do row
        row.family === :ranef || return Float64(row.c)
        get(learned, _brm_svi_control_key(row.group, row.margin.predictor,
            row.margin.coefficient, row.level), Float64(row.c))
    end
end
