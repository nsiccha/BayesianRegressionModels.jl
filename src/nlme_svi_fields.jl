# A read-pair `gmrf` field's states as units' own coordinates (`field_states=:units`).
#
# A state read by one unit joins that unit's Gaussian factor, next to its effects, so a
# unit's field slice is the unit's own: its cost and storage follow its reads, not the
# field. A state read by several units stays global. The approximation keeps units
# independent; the prior couples them only through its conditionals' means.
#
# The field's joint prior is a sum of one conditional per state (`gmrf_vecchia_local.jl`),
# assigned to the state's owner, so the objective stays exactly
#
#     log p(u) = P(g) + F_0(g, ·) + Σ_i [U_i + ℓ_i + F_i],
#
# and a batch of units scaled up is unbiased for it. A batch unit also takes the gradient
# of conditionals elsewhere that condition on its states: that is the whole model's
# gradient on its factor, and it is unscaled like the rest of a unit's local gradient.
# Parents in other units are read from one draw of their factors per iteration, taken in a
# fixed order before any task, so a run is the same at any number of tasks.

# The read-pair fields of `m`, before ownership decides who holds each state.
function _brm_svi_field_inventory(m)
    names = m.view.coordinates
    fields = NamedTuple[]
    for record in _brm_nlme_records(m.backend.plan)
        record.kind === :gmrf_multiresolution && _brm_nlme_error(
            "brm_svi_fit: `$(record.predictor)` is a `multiresolution=` field, whose states are " *
            "shared innovations rather than reads, so `field_states=:units` has no unit to " *
            "give them to; use `neighbors=m`")
        record.kind === :gmrf_footprint || continue
        support = only(m.backend.plan.columns[record.fitted])
        base = Symbol(record.declaration, ".", record.locals.field)
        positions = findall(name -> _brm_nlme_owned(name, base), names)
        length(positions) == support.geometry.ncoordinates || _brm_nlme_error(
            "brm_svi_fit: `$(record.predictor)` has $(length(positions)) packed coordinates " *
            "and $(support.geometry.ncoordinates) field coordinates")
        push!(fields, (; record, footprint=support.geometry, positions))
    end
    isempty(fields) && throw(ArgumentError(
        "brm_svi_fit: field_states=:units needs a `gmrf(...; neighbors=m)` field, and this model has none"))
    Tuple(fields)
end

# Without the fields' joint prior events, which their conditionals replace, ownership
# follows footprints: a state read by one unit is that unit's. A read's channels stay
# together, as they share a whitening solve.
_brm_svi_strip_fields(factors, ::Tuple{}) = factors
function _brm_svi_strip_fields(factors, fields)
    states = Set{Int}()
    foreach(field -> union!(states, field.positions), fields)
    keep = [k for k in eachindex(factors.events)
        if isempty(factors.owned[k]) || !all(in(states), factors.owned[k])]
    channels = Vector{Int}[]
    for field in fields
        C = field.footprint.nchannels
        C > 1 && append!(channels, [field.positions[(C * (s - 1) + 1):(C * s)]
            for s in 1:length(field.footprint.position)])
    end
    merge(factors, (; events=factors.events[keep],
        owned=vcat(factors.owned[keep], channels), support=vcat(factors.support[keep], channels)))
end

function _brm_svi_field_length(m, record, slot, globals)
    prior = getproperty(record.locals.priors, slot)
    prior isa Real && return (; known=Float64(prior), place=0, transform=:known, floor=0.0)
    name = Symbol(record.declaration, ".", getproperty(record.locals, slot))
    entry = _brm_svi_layout_entry(m, name)
    entry.transform in (:exp, :floored) && entry.size == 1 || _brm_nlme_error(
        "brm_svi_fit: `$name` has the transform `$(entry.transform)`; field_states=:units " *
        "reads a length through `:exp` or `:floored`, which positive priors give")
    place = _brm_sorted_slot(globals, entry.offset)
    place == 0 && _brm_nlme_error("brm_svi_fit: `$name` is not a global coordinate; " *
        "field_states=:units reads a field's lengths from the globals")
    (; known=NaN, place, transform=entry.transform,
       floor=entry.transform === :floored ? Float64(entry.lo) : 0.0)
end

function _brm_svi_layout_entry(m, name)
    for entry in m.backend.model.layout.entries
        entry.name === name && return entry
    end
    _brm_nlme_error("brm_svi_fit: the layout has no entry `$name`")
end

# The field's channel factor: identity, known, or sampled through an LKJ transform.
function _brm_svi_field_factor(m, record, footprint, globals)
    correlation = record.correlation
    C = footprint.nchannels
    correlation === nothing && return (; known=Matrix{Float64}(I, C, C), places=Int[])
    correlation.factor === nothing || return (; known=Matrix{Float64}(correlation.factor), places=Int[])
    entry = _brm_svi_layout_entry(m, correlation.name)
    entry.kind === :cholesky_corr || _brm_nlme_error(
        "brm_svi_fit: `$(correlation.name)` is not a Cholesky correlation factor")
    places = [_brm_sorted_slot(globals, j) for j in entry.offset:(entry.offset + entry.size - 1)]
    all(>(0), places) || _brm_nlme_error(
        "brm_svi_fit: `$(correlation.name)` is not global; field_states=:units reads a " *
        "field's channel factor from the globals")
    (; known=nothing, places)
end

# Implemented by the ReactiveKernels Enzyme extension: an LKJ factor and its Jacobian.
function _rk_svi_lkj_value_jacobian end

_brm_svi_field_length_value(h, g) = h.transform === :known ? (h.known, 0.0) :
    (x = exp(g[h.place]); (h.floor + x, x))

# Everything of one field that a step reads from the global draw.
function _brm_svi_field_law(field, g)
    spatial, dspatial = _brm_svi_field_length_value(field.spatial, g)
    temporal, dtemporal = _brm_svi_field_length_value(field.temporal, g)
    C = field.f.nchannels
    factor = field.factor
    L, jacobian = factor.known === nothing ?
        _rk_svi_lkj_value_jacobian(g[factor.places], C) : (factor.known, nothing)
    (; spatial, dspatial, temporal, dtemporal, L, jacobian)
end

# `(positions in the field, place in a block or the globals)` of what one owner reads.
# Kept per unit, so stored once: a plan's coordinates are its states' channels, read
# through `plan.states`, and `blanket[i + 1]`, the other units owner `i` reads, is
# joined over the fields as it is built.
function _brm_svi_unit_fields(m, ownership, inventory, globals, units)
    N = length(units)
    slot = zeros(Int, length(m.unit))
    for block in units
        slot[block] .= eachindex(block)
    end
    blanket = [Int32[] for _ in 0:N]
    fields = map(inventory) do field
        f = field.footprint
        C = f.nchannels
        owner_of = [ownership.owner[field.positions[C * (s - 1) + 1]] for s in 1:length(f.position)]
        plans = _brm_vecchia_local_plans(f, owner_of, N)
        unit_slots = map(enumerate(plans)) do (i, plan)
            [(c + C * (s - 1), slot[field.positions[c + C * (s - 1)]]) for s in plan.owned for c in 1:C
                if i > 1]
        end
        global_slots = map(plans) do plan
            pairs = Tuple{Int,Int}[]
            for s in plan.states, c in 1:C
                k = c + C * (s - 1)
                p = _brm_sorted_slot(globals, field.positions[k])
                p == 0 || push!(pairs, (k, p))
            end
            pairs
        end
        for (i, plan) in enumerate(plans), s in plan.states
            o = owner_of[s]
            o != 0 && o != i - 1 && push!(blanket[i], o)
        end
        spatial = _brm_svi_field_length(m, field.record, :spatial_length, globals)
        temporal = _brm_svi_field_length(m, field.record, :temporal_length, globals)
        factor = _brm_svi_field_factor(m, field.record, f, globals)
        global_states = [(field.positions[k], _brm_sorted_slot(globals, field.positions[k]))
            for k in 1:f.ncoordinates if _brm_sorted_slot(globals, field.positions[k]) != 0]
        (; f, field.positions, plans, unit_slots, global_slots, spatial,
           temporal, factor, global_states, law=Ref{Any}(nothing))
    end
    foreach(list -> sort!(unique!(list)), blanket)
    (; fields=Tuple(fields), blanket, others=copy(m.unit))
end

# What a target's unit touches beyond its likelihood: global states and hyperparameters.
function _brm_svi_field_touched!(touched, state)
    state === nothing && return touched
    for (i, list) in enumerate(touched), field in state.fields
        append!(list, (p for (_, p) in field.global_slots[i + 1]))
        field.spatial.place > 0 && push!(list, field.spatial.place)
        field.temporal.place > 0 && push!(list, field.temporal.place)
        append!(list, field.factor.places)
    end
    foreach(list -> sort!(unique!(list)), touched)
    touched
end

_brm_svi_field_work(state) = map(field -> (; local_work=_BRMVecchiaLocalWork(field.f),
    x=zeros(field.f.ncoordinates)), state.fields)

_brm_svi_field_laws!(state, g) = foreach(field -> field.law[] = _brm_svi_field_law(field, g), state.fields)

# Sets the fields' reading of the other units to a joint point `u` of the packed
# coordinates, as an evaluation of the whole density at `u` reads them.
function _brm_svi_field_joint!(t, u)
    state = t.fields
    state === nothing && return nothing
    copyto!(state.others, u)
    _brm_svi_field_laws!(state, u[t.globals])
    nothing
end

# A step's reading of the world outside its batch: the laws at `g`, the global states, and
# one draw of every unit a batch's conditionals read. Serial, before any task.
function _brm_svi_field_prepare!(rng, t::NamedTuple, q, g, batch)
    state = get(t, :fields, nothing)
    state === nothing && return nothing
    _brm_svi_field_laws!(state, g)
    for field in state.fields, (position, p) in field.global_states
        state.others[position] = g[p]
    end
    readers = reduce(vcat, (state.blanket[i + 1] for i in batch); init=copy(state.blanket[1]))
    for u in sort!(unique!(readers))
        u = Int(u)
        state.others[t.units[u]] .= _brm_svi_draw_unit(q, u, _brm_svi_unit_noise(rng, q.U[u]))
    end
    nothing
end

# Owner `i`'s conditionals of one field. Adds to `gg` their share in the globals and to
# `gz` the whole density's gradient on the owner's states; returns their value.
function _brm_svi_field_piece!(gg, gz, field, i, z, others, work)
    plan = field.plans[i + 1]
    isempty(plan.states) && return 0.0
    x, local_work = work.x, work.local_work
    C = field.f.nchannels
    @inbounds for s in plan.states, c in 1:C
        k = c + C * (s - 1)
        x[k] = others[field.positions[k]]
    end
    unit_slots = field.unit_slots[i + 1]
    @inbounds for (k, slot) in unit_slots
        x[k] = z[slot]
    end
    law = field.law[]
    part = _brm_vecchia_local!(local_work, plan, field.f, x, law.spatial, law.temporal, law.L)
    @inbounds for (k, p) in field.global_slots[i + 1]
        gg[p] += local_work.gown[k]
    end
    @inbounds for (k, slot) in unit_slots
        gz[slot] += local_work.gall[k]
    end
    field.spatial.place > 0 && (gg[field.spatial.place] += part.spatial_length * law.dspatial)
    field.temporal.place > 0 && (gg[field.temporal.place] += part.temporal_length * law.dtemporal)
    if law.jacobian !== nothing
        pulled = law.jacobian' * vec(part.L)
        for (a, p) in enumerate(field.factor.places)
            gg[p] += pulled[a]
        end
    end
    _brm_vecchia_local_clear!(local_work, plan, field.f)
    part.value
end

function _brm_svi_field_term!(gg, gz, t, i, z, work)
    state = t.fields
    state === nothing && return 0.0
    value = 0.0
    for (b, field) in enumerate(state.fields)
        value += _brm_svi_field_piece!(gg, gz, field, i, z, state.others, work.fields[b])
    end
    value
end
