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
# stays triangular. Authored Gaussian vectors read their law's mean and scale per
# element, `<x>_prior_mean` and `<x>_prior_scale`, which RK names for them.
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

Learned partial centering for [`brm_svi_fit`](@ref): one centering control
`c ∈ (0, 1)` per eligible cell, optimized jointly with the variational factors
by the ELBO (the VIP of Gorinova, Moore and Hoffman, 2020). `c = 0` is the
non-centred coordinate and `c = 1` the centred one: a random effect located at
its population predictor and, in a correlated block, at its conditional mean
given the block's earlier margins; an authored Gaussian vector element at its
law's mean. Every control starts at `start`; `step` is the Adam step size of
the controls' logits, taken whenever their unit is in the batch.
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
    hasproperty(block, :hsgp) && return "an HSGP weight's spectral scale is computed " *
        "inside its basis kernel and is not read as a graph value"
    hasproperty(block, :cdar) && return "a walk innovation's scale is computed inside " *
        "its kernel and is not read as a graph value"
    hasproperty(block, :car) && return "a car field draw's scale is not read as a graph value"
    get(block, :completion, false) === true && return "a completion's law scale is not " *
        "read as a graph value"
    "its covariance is fixed"
end

# The cells of target `t`: each instance of an estimated random-effect family
# whose draws one unit owns, located at its mu-referenced population coefficients
# (`BRMNLMEView.mu_references`), and each authored Gaussian vector element a unit
# owns whose law reads no unit coordinate.
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
    family_slot = Dict{Int,Int}()
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
            reason = family.fixed !== nothing ? _brm_svi_fixed_family_reason(family.block) :
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
                push!(uncentered, (; kind=:ranef, block=family.block,
                    level=first(labels).level, coordinates=view.coordinates[packed], reason))
                continue
            end
            slot = get!(family_slot, f) do
                record = family.record
                foreach(port!, isnothing(record.scales) ? (record.scale,) : record.scales)
                isnothing(record.L) || port!(record.L)
                push!(families, (; family=f, record, K, family.stratum))
                length(families)
            end
            cellrange = (ncells[i] + 1):(ncells[i] + K)
            ncells[i] += K
            push!(instances[i], _BRMSVICenteringBlock(slot,
                [local_index[i][j] for j in packed], cellrange, location))
            for k in 1:K
                push!(cells[i], (; kind=:ranef, labels[k].block, labels[k].predictor,
                    labels[k].coefficient, labels[k].level,
                    coordinate=view.coordinates[packed[k]], located=!isempty(location[k])))
            end
        end
    end
    absent = Dict{Int,NTuple{3,Int}}()
    absent_set = Set(hasproperty(t.ownership, :absent) ? t.ownership.absent : Int[])
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
    # Shared field weights are global coordinates, read through scales that are
    # computed inside their kernels.
    globals = Set(t.globals)
    for record in _brm_nlme_records(m.backend.plan)
        record.kind in (:hsgp, :gmrf_footprint, :car, :cdar) || continue
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
    names = Tuple(ports)
    read = isempty(names) ? nothing : _rk_value_reader(m.backend, names)
    pullback = isempty(names) ? nothing : _rk_value_pullback(m.backend, names, copy(m.unit))
    _BRMSVICenteringPlan(centering.start, centering.step, names, families, instances, ncells, cells,
        uncentered, absent, read, pullback, copy(m.unit))
end

# The values every cell of an iteration reads, at the globals `g`: the ports'
# graph values and each family's natural factor, scales and correlation factor.
function _brm_svi_centering_frame(plan::_BRMSVICenteringPlan, t, g)
    point = copy(plan.base)
    point[t.globals] .= g
    plan.read === nothing && return (; point, nt=NamedTuple(), values=(), A=Matrix{Float64}[],
        scales=Vector{Float64}[], L=Matrix{Float64}[])
    nt = plan.read(point)
    scales = [_brm_nlme_joint_scale(nt, f.record, f.K, f.stratum) for f in plan.families]
    A = [Matrix{Float64}(_brm_nlme_natural_factor(nt, f.record, f.K, f.stratum))
        for f in plan.families]
    L = [Diagonal(1 ./ s) * a for (s, a) in zip(scales, A)]
    (; point, nt, values=Tuple(nt), A, scales, L)
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
        if isnothing(record.scales)
            p = slot(record.scale)
            if stratum === nothing
                cot[p] isa Number ? (cot[p] += only(s̄)) : (cot[p] .+= s̄)
            else
                view(cot[p], stratum, :) .+= s̄
            end
        else
            for (k, name) in enumerate(record.scales)
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

# The controls of a target's cells: their logits, one Adam per unit and each
# cell's position in its unit's block. A carried unit's cells start from its
# previous logits, one per coordinate (`NaN` where none was learned).
function _brm_svi_centering_controls(plan::_BRMSVICenteringPlan; carried=nothing)
    x0 = _brm_svi_control_logit(plan.start)
    positions = [_brm_svi_cell_positions(plan, i) for i in eachindex(plan.ncells)]
    x = map(eachindex(positions)) do i
        old = carried === nothing ? nothing : carried[i]
        [old === nothing || isnan(old[p]) ? x0 : old[p] for p in positions[i]]
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
        w = weights[k] / 2
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
# as well: a window's first state located at its boundary, say. At c = 1 that
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
# starting globals (mean through T⁻¹, scales by each cell's exp((c - c0) ℓ)).
function _brm_svi_with_centering(t, q, centering::LearnedCentering)
    plan = _brm_svi_centering_plan(t, centering)
    prior = hasproperty(t, :prior) && t.prior isa _BRMSVIStreamingPrior ? t.prior : nothing
    carried = prior === nothing ? nothing : prior.controls
    q = merge(q, (; centering=_brm_svi_centering_controls(plan; carried)))
    frame = _brm_svi_centering_frame(plan, t, q.μ)
    for i in eachindex(t.units)
        plan.ncells[i] == 0 && continue
        # A carried unit's factor is already in its previous controls' frame.
        prior !== nothing && prior.carried[i] && continue
        c = _brm_svi_unit_controls(q, i)
        y = copy(q.M[i])
        _, _, ℓ = _brm_svi_unit_centering!(similar(y), y, plan.instances[i], plan.ncells[i],
            c, frame, q.μ, plan.base)
        q.M[i] .= _brm_svi_learned_coordinates(plan, frame, q, i, q.μ, y)
        scale = ones(length(y))
        for inst in plan.instances[i], (n, p) in enumerate(inst.positions)
            cell = inst.cells[n]
            c0 = inst isa _BRMSVICenteringBlock ? 0.0 : 1.0
            scale[p] = exp((c[cell] - c0) * ℓ[cell])
        end
        _brm_svi_unit_rescale!(q, i, scale)
    end
    merge(t, (; centering=plan)), q
end

# Unit i's factor with its coordinates scaled by `s`: Σ ↦ D Σ D for D = diag(s).
function _brm_svi_unit_rescale!(q, i, s)
    Λ, U = q.Λ[i], q.U[i]
    if Λ isa _BRMSVIUnitPrecision
        f = Λ.factor
        f.ρ .+= log.(s)
        f.Β .*= s
    else
        Λ .= Diagonal(1 ./ s) * Λ * Diagonal(1 ./ s)
        U .= Diagonal(1 ./ s) * U
    end
    q
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
# coordinates at `destination` (the draw's frame): its centered vector elements.
function _brm_svi_absent_draw!(z, plan::_BRMSVICenteringPlan, frame, unit, source, destination)
    hasproperty(unit, :centering) || return z
    for (k, j) in zip(source, destination)
        x = unit.centering[k]
        isnan(x) && continue
        mean, scale, element = plan.absent[j]
        _, z[k] = _brm_svi_cell(1.0, _brm_svi_control(x), z[k],
            _brm_svi_element(frame.values[mean], element),
            log(_brm_svi_element(frame.values[scale], element)))
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
