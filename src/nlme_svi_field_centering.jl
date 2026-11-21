struct _BRMSVIFieldRow
    J::Matrix{Float64}  # conditional coefficients and log scale, with hyperparameter derivatives
    c::Float64
    D::Float64          # own-coordinate multiplier
    H::Float64          # conditional-location multiplier
end

# The RK extension prepares a selected conditional factor, including
# its derivative in just the field hyperparameters, and the bound parent layout.
function _rk_svi_conditional_reader end

# A station absent from a new window keeps its previous Gaussian. Its field
# coordinates can still enter another station's one-pass map. Draw that
# Gaussian once per visit, conditional on the current globals, and retain the
# conditional mean's slope for the gradient in those globals.
function _brm_svi_absent_fields(t, state)
    sources = NamedTuple[]
    by_position = Dict{Int,NamedTuple}()
    absent = get(t.ownership, :absent, Int[])
    isempty(absent) && return (; sources, by_position)
    positions = Set(j for f in state.fields for j in f.positions)
    names = Dict(t.m.view.coordinates[j] => j for j in absent if j in positions)
    for label in sort!(collect(keys(t.previous.units)); by=repr)
        saved = t.previous.units[label]
        entries = Tuple{Int,Int}[]
        coupled = get(saved, :coupling, nothing)
        for (slot, key) in enumerate(saved.keys)
            j = get(names, get(key, :coordinate, nothing), 0)
            j == 0 && continue
            push!(entries, (j, slot))
            logit = hasproperty(saved, :centering) ? saved.centering[slot] : NaN
            c = isnan(logit) ? 1.0 : _brm_svi_control(logit)
            row = coupled === nothing ? nothing : findfirst(==(slot), coupled.rows)
            slope = row === nothing ? nothing :
                (; coupled.columns, weights=view(coupled.W, row, :))
            by_position[j] = (; source=length(sources)+1, c, slope)
        end
        isempty(entries) || push!(sources, (; saved, entries))
    end
    (; sources, by_position)
end

# An owner's likelihood reads its own rows and packed global rows. Its prior
# additionally reads each owned row's direct parents. Mapping those outputs
# into learned coordinates adds precisely one further layer of inputs.
function _brm_svi_sweep_field(field, t, cell, authored; absent=Dict{Int,NamedTuple}())
    places = sort!(unique!(filter(>(0), [field.spatial.place; field.temporal.place; field.factor.places])))
    if field.conditional_reader[] === nothing
        field.conditional_reader[] = _rk_svi_conditional_reader(t.m.backend, field.record,
            t.m.unit, t.globals[places], t.m.population_prior.ad_backend)
    end
    reader = field.conditional_reader[]
    layout, parents = reader.layout, reader.layout.inputs
    owner = t.ownership.owner[field.positions]
    retained = Dict(k => absent[j] for (k,j) in enumerate(field.positions) if haskey(absent,j))
    slot = zeros(Int, length(owner))
    owned = [Int[] for _ in 0:length(t.units)]
    for (o, pairs) in enumerate(field.unit_slots), (k, p) in pairs
        slot[k] = p
        push!(owned[o], k)
    end
    for (k, j) in enumerate(field.positions)
        if owner[k] == 0
            slot[k] = _brm_sorted_slot(t.globals, j)
            push!(owned[1], k)
        end
    end
    outputs = map(owned) do mine
        result = copy(mine)
        authored && foreach(k -> append!(result, parents[k]), mine)
        sort!(unique!(result))
    end
    inputs = map(outputs) do rows
        result = copy(rows)
        for k in rows
            (owner[k] > 0 || haskey(retained,k)) && append!(result, parents[k])
        end
        sort!(unique!(result))
    end
    (; field, layout, parents, owner, slot, owned, outputs, inputs, places, cell, retained, reader)
end

function _brm_svi_field_sweep(t, centering)
    state = get(t, :field_geometry, get(t, :fields, nothing))
    state === nothing && return t
    index = Dict(row.coordinate => k for rows in centering.cells for (k, row) in enumerate(rows)
        if row.kind === :gmrf)
    absent = _brm_svi_absent_fields(t, state)
    # Packed states need no map: neither this window's nor an absent owner's.
    isempty(index) && all(source -> source.c == 1.0, values(absent.by_position)) && return t
    authored = t.fields !== nothing
    fields = map(state.fields) do field
        cell = [get(index, t.m.view.coordinates[j], 0) for j in field.positions]
        _brm_svi_sweep_field(field, t, cell, authored; absent=absent.by_position)
    end
    # Weight each external owner's term by the number of inputs from the
    # selected unit and direct edges from it to the owner's states. For each
    # term, readers lists the factors that supply its joint draw.
    N = length(t.units)
    counts = [Dict{Int,Int}() for _ in 1:N]
    readers = [[o] for o in 0:N]
    touched = deepcopy(t.touched)
    for f in fields, o in 0:N
        !authored && o == 0 && continue
        for k in f.inputs[o+1]
            u = f.owner[k]
            if u > 0
                u == o || (counts[u][o] = get(counts[u], o, 0) + 1)
                push!(readers[o+1], u)
            elseif haskey(f.retained, k)
                source = f.retained[k]
                push!(readers[o+1], -source.source)
                o > 0 && source.slope !== nothing && append!(touched[o], source.slope.columns)
            elseif o > 0 && f.slot[k] > 0
                push!(touched[o], f.slot[k])
            end
        end
        for k in f.owned[o+1], j in f.parents[k]
            u = f.owner[j]
            u > 0 && u != o && (counts[u][o] = get(counts[u],o,0) + 1)
        end
        o > 0 && !isempty(f.outputs[o+1]) && append!(touched[o], f.places)
    end
    foreach(xs -> foreach(x -> sort!(unique!(x)), xs), (readers, touched))
    neighbors = map(counts) do count
        owners = sort!(collect(keys(count)))
        mass = [count[o] for o in owners]
        sampler = isempty(owners) ? nothing : _brm_svi_alias_sampler(mass ./ sum(mass))
        (; owners, sampler)
    end
    visits = [@NamedTuple{o::Int,weight::Float64}[] for _ in 1:N]
    empty_draw = map(_ -> Dict{Int,_BRMSVIFieldRow}(), fields)
    sweep = (; fields, neighbors, visits, readers, authored, absent, neighbor_calls=zeros(Int,N), others=copy(t.m.unit), draw=Ref(empty_draw))
    merge(t, (; field_sweep=sweep, touched))
end

# An external owner's term contributes only to the selected unit's local
# derivatives. The same auxiliary owners are used on both sides of every
# antithetic pair, in the orthogonal design and in its curvature probes.
# Two draws retained the gain in within-station covariance on both fitted
# seeds, and bound each selected unit to three likelihood owners.
function _brm_svi_neighbor_terms(rng, neighbors, i)
    terms = [(; o=i, weight=1.0)]
    if length(neighbors.owners) <= 2
        append!(terms, ((; o, weight=1.0) for o in neighbors.owners))
    else
        batch = _brm_svi_batch(rng, neighbors.sampler, 2)
        for (k,j) in enumerate(batch.indices)
            push!(terms,(; o=neighbors.owners[j], weight=batch.weights[k]))
        end
    end
    sort!(terms; by=term -> term.o)
end

# Deterministic joint-point evaluations include every term. During a fit,
# prepare! installs the sample before any worker evaluates its unit.
function _brm_svi_sweep_visits(s,i)
    visits = s.visits[i]
    isempty(visits) || return visits
    [(; o,weight=1.0) for o in sort!([i; s.neighbors[i].owners])]
end

# Only patterns touched by these terms are solved. Shared rows use c=1,
# giving y=u even when their conditional mean depends on unit-owned states.
# Only a `focus` owner's term pulls back to the hyperparameters: through the
# whole row of each of its outputs, and through the scale alone of a state it
# reads as a parent. Every other row is read for its values.
function _brm_svi_sweep_draw(f, q, g, owners; derivatives=true, focus=owners)
    needed = sort!(unique!(reduce(vcat, (f.inputs[o+1] for o in owners); init=Int[])))
    key(k) = (f.layout.pattern[k], f.layout.channel[k])
    whole, scaled = Set{Tuple{Int,Int}}(), Set{Tuple{Int,Int}}()
    if derivatives
        for o in focus
            union!(whole, (key(k) for k in f.outputs[o+1]))
            union!(scaled, (key(k) for k in f.inputs[o+1]))
        end
    end
    hyper = g[f.places]
    matrices = Dict{Tuple{Int,Int},Matrix{Float64}}()
    Dict(k => begin
        J = get!(matrices, key(k)) do
            f.reader.coefficients(hyper, k, key(k) in scaled; scale_only=!(key(k) in whole))
        end
        c = if f.cell[k] > 0
            _brm_svi_control(q.centering.x[f.owner[k]][f.cell[k]])
        elseif haskey(f.retained,k)
            f.retained[k].c
        else
            1.0
        end
        D = exp((1-c) * J[end,1])
        _BRMSVIFieldRow(J, c, D, 1-c*D)
    end for k in needed)
end

function _brm_svi_sweep_value(f, rows, k, raw)
    row = rows[k]
    u = raw(k)
    row.H == 0 && return row.D * u
    loc = 0.0
    for (a, j) in enumerate(f.parents[k])
        loc += row.J[a,1] * rows[j].D * raw(j)
    end
    row.D * u + row.H * loc
end

# The scalar output's contribution to an owner's control and factor gradients.
# share=true also pulls this term back to the global states and hyperparameters.
function _brm_svi_sweep_reverse!(gg, gu, gc, f, rows, k, raw, focus, share, ybar, jac)
    row = rows[k]
    J, c, D, H = row.J, row.c, row.D, row.H
    function add!(j, du, dc)
        o, p = f.owner[j], f.slot[j]
        if o == 0
            if share && p > 0
                gg[p] += du
            elseif share && haskey(f.retained,j)
                slope = f.retained[j].slope
                if slope !== nothing
                    for (a, col) in enumerate(slope.columns)
                        gg[col] += du * slope.weights[a]
                    end
                end
            end
        elseif o == focus
            gu[p] += du
            gc[f.cell[j]] += dc
        end
    end
    add!(k, D * ybar, 0.0)
    f.cell[k] == 0 && !haskey(f.retained,k) && return
    loc = 0.0
    for (a, j) in enumerate(f.parents[k])
        parent = rows[j]
        x = parent.D * raw(j)
        w = J[a,1]
        loc += w * x
        xb = ybar * H * w
        add!(j, xb * parent.D, -xb * x * parent.J[end,1])
        if share
            for (h, p) in enumerate(f.places)
                gg[p] += ybar * H * x * J[a,h+1] +
                    xb * x * (1-parent.c) * parent.J[end,h+1]
            end
        end
    end
    residual = D * (raw(k) - c * loc)
    logbar = (1-c) * (ybar * residual + jac)
    add!(k, 0.0, -ybar * loc * D - J[end,1] * (ybar * residual + jac))
    if share
        for (h, p) in enumerate(f.places)
            gg[p] += logbar * J[end,h+1]
        end
    end
end

function _brm_svi_sweep_prepare!(rng, t, q, g, batch, noise, η)
    s = t.field_sweep
    s.neighbor_calls[batch] .= 0
    owners = s.authored ? [0] : Int[]
    for i in batch
        s.visits[i] = _brm_svi_neighbor_terms(rng,s.neighbors[i],i)
        append!(owners,(v.o for v in s.visits[i]))
    end
    sort!(unique!(owners))
    readers = sort!(unique!(reduce(vcat, (s.readers[o+1] for o in owners); init=Int[])))
    coupling = _brm_svi_coupling(q)
    for i in readers
        i == 0 && continue
        if i < 0
            source = s.absent.sources[-i]
            z = _brm_svi_saved_unit_draw(rng, source.saved, g)
            for (position, slot) in source.entries
                s.others[position] = z[slot]
            end
            continue
        end
        ζ = coupling === nothing ? nothing : _brm_svi_coupling_noise(coupling, i, noise, η)
        s.others[t.units[i]] .= _brm_svi_coupled_mean(coupling, q, i, ζ) .+
            _brm_svi_unit_offset(q, i, _brm_svi_unit_noise(rng, q.U[i]))
    end
    focus = s.authored ? [0; batch] : batch
    s.draw[] = map(f -> _brm_svi_sweep_draw(f, q, g, owners; focus), s.fields)
    nothing
end

# Evaluate one owner's conditional factors at the mapped states, and differentiate
# them into the packed states and the sampled hyperparameters.
function _brm_svi_sweep_prior!(gy, gg, f, rows, mine, y, share)
    value = 0.0
    for k in mine
        J = rows[k].J
        μ = sum((J[a,1] * y[j] for (a,j) in enumerate(f.parents[k])); init=0.0)
        ℓ = J[end,1]
        residual = y[k] - μ
        μbar = residual * exp(-2ℓ)
        ℓbar = residual * μbar - 1
        value -= (log(2π) + 2ℓ + residual * μbar) / 2
        gy[k] -= μbar
        for (a,j) in enumerate(f.parents[k])
            gy[j] += μbar * J[a,1]
        end
        if share
            for (h,p) in enumerate(f.places)
                dμ = sum((J[a,h+1] * y[j] for (a,j) in enumerate(f.parents[k])); init=0.0)
                gg[p] += μbar * dμ + ℓbar * J[end,h+1]
            end
        end
    end
    value
end

# A curvature probe shifts one global while holding the Gaussian noises fixed.
# An absent parent's saved conditional then shifts by its regression slope.
function _brm_svi_absent_probe(value, retained, k, probe)
    probe === nothing && return value
    source = get(retained, k, nothing)
    (source === nothing || source.slope === nothing) && return value
    slot = _brm_sorted_slot(source.slope.columns, probe.position)
    slot == 0 ? value : value + source.slope.weights[slot] * probe.shift
end

# A selected owner's term contributes value and global gradients once. The
# other terms whose field rows read its learned coordinates contribute just
# their derivative in that owner's factor and controls, without batch weights.
function _brm_svi_sweep_term!(gg, gu, centering, frame, t, q, focus, g, u, plan;
        draw=t.field_sweep.draw[], probe=nothing, neighbor_value=nothing)
    s = t.field_sweep
    gc = focus == 0 ? Float64[] : zeros(centering.ncells[focus])
    adjoint = focus == 0 ? nothing :
        _BRMSVICenteringAdjoint(frame, centering.instances[focus], length(gc))
    value = 0.0
    focus == 0 && !s.authored && return value, gc, adjoint
    visits = focus == 0 ? ((; o=0,weight=1.0),) : _brm_svi_sweep_visits(s,focus)
    for (; o,weight) in visits
        share = o == focus
        o > 0 && !share && (s.neighbor_calls[focus] += 1)
        # The worker's gg may already have contributions; a neighboring
        # partition must never write its globals into that buffer.
        globals = share ? gg : nothing
        z = o == 0 ? Float64[] : share ? u : s.others[t.units[o]]
        c = o == 0 ? Float64[] : _brm_svi_unit_controls(q, o)
        before = similar(z)
        ordinary = if o > 0
            _brm_svi_unit_centering!(before, z, centering.instances[o],
                centering.ncells[o], c, frame, g, centering.base)
        end
        packed = copy(before)
        sections = map(s.fields, draw) do f, rows
            raw(k) = if f.owner[k] == 0
                f.slot[k] == 0 ? _brm_svi_absent_probe(s.others[f.field.positions[k]],
                    f.retained, k, probe) : g[f.slot[k]]
            elseif f.owner[k] == focus
                u[f.slot[k]]
            else
                s.others[f.field.positions[k]]
            end
            y = Dict(k => _brm_svi_sweep_value(f, rows, k, raw) for k in f.outputs[o+1])
            if o > 0
                for k in f.owned[o+1]
                    packed[f.slot[k]] = y[k]
                end
            end
            (; raw, y)
        end
        grad = zeros(length(z))
        if o > 0
            v = _brm_svi_unit_partition!(globals, grad, t, o, g, packed, plan)
            if s.authored || !t.prior.carried[o]
                v += _brm_svi_unit_prior!(globals, grad, t.prior, t, o, g, packed)
            elseif share
                # The previous field map is retained at every draw of g. Its
                # inverse and Jacobian cancel the current field map exactly.
                v += _brm_svi_transported_prior!(gu, adjoint, centering, frame,
                    t.prior.units, o, g, before, t.prior.controls[o];
                    gg, carried=t.prior.coupling[o])
            end
            if share
                value += v + first(ordinary)
            elseif neighbor_value !== nothing
                neighbor_value[] += weight * v
            end
        end
        for (f, rows, section) in zip(s.fields, draw, sections)
            mine = f.owned[o+1]
            gy = Dict(k => 0.0 for k in f.outputs[o+1])
            if o > 0
                for k in mine
                    gy[k] += grad[f.slot[k]]
                    grad[f.slot[k]] = 0.0
                end
            end
            if s.authored
                v = _brm_svi_sweep_prior!(gy, gg, f, rows, mine, section.y, share)
                if share
                    value += v
                elseif neighbor_value !== nothing
                    neighbor_value[] += weight * v
                end
            end
            for k in f.outputs[o+1]
                jac = Int(s.authored && f.owner[k] == o && o > 0)
                share && jac == 1 && (value += (1-rows[k].c)*rows[k].J[end,1])
                _brm_svi_sweep_reverse!(gg, gu, gc, f, rows, k, section.raw,
                    focus, share, weight * gy[k], weight * jac)
            end
        end
        if o > 0 && share
            gu .+= grad
            _brm_svi_unit_centering_backward!(gu, gc, adjoint, before, ordinary[2],
                ordinary[3], centering.instances[o], c, frame)
        end
    end
    if !s.authored && focus > 0
        for f in s.fields, k in f.owned[focus+1]
            gc[f.cell[k]] = 0.0
        end
    end
    value, gc, adjoint
end

function _brm_svi_sweep_draws!(point, t, q, g)
    s = t.field_sweep
    for f in s.fields
        rows = _brm_svi_sweep_draw(f, q, g, 0:length(t.units); derivatives=false)
        raw = point[f.field.positions]
        for k in eachindex(raw)
            f.owner[k] > 0 || haskey(f.retained,k) || continue
            point[f.field.positions[k]] = _brm_svi_sweep_value(f, rows, k, j -> raw[j])
        end
    end
end

# A supplied packed starting point (including a factor-level start's means)
# is located at q.μ. Recover the scaled coordinates in triangular order so
# the one-pass map returns that same point, including across unit boundaries.
function _brm_svi_field_start!(t, q)
    for f in t.field_sweep.fields
        rows = _brm_svi_sweep_draw(f, q, q.μ, 0:length(t.units); derivatives=false)
        scaled = zeros(length(f.owner))
        for k in eachindex(scaled)
            o, p = f.owner[k], f.slot[k]
            if o == 0
                scaled[k] = p > 0 ? q.μ[p] : t.m.unit[f.field.positions[k]]
            else
                row = rows[k]
                location = sum((row.J[a,1] * scaled[j] for (a,j) in enumerate(f.parents[k])); init=0.0)
                scaled[k] = q.M[o][p] - row.H * location
                q.M[o][p] = scaled[k] / row.D
            end
        end
    end
end

# At the mean of the globals, the one-pass field map is affine. Store its
# sparse rows by field coordinate name. Slice initialization can then project
# a unit's marginal including uncertainty coming from other units.
function _brm_svi_field_projection(t, q)
    hasproperty(t, :field_sweep) || return nothing
    projection = Dict{Symbol,NamedTuple}()
    names = t.m.view.coordinates
    for f in t.field_sweep.fields
        rows = _brm_svi_sweep_draw(f,q,q.μ,0:length(t.units); derivatives=false)
        for k in eachindex(f.owner)
            f.owner[k] == 0 && !haskey(f.retained,k) && continue
            name = names[f.field.positions[k]]
            inputs, weights = [name], [rows[k].D]
            offset = 0.0
            for (a,j) in enumerate(f.parents[k])
                w = rows[k].H * rows[k].J[a,1] * rows[j].D
                if f.owner[j] == 0 && !haskey(f.retained,j)
                    p = f.slot[j]
                    offset += w * (p == 0 ? t.m.unit[f.field.positions[j]] : q.μ[p])
                else
                    push!(inputs,names[f.field.positions[j]])
                    push!(weights,w)
                end
            end
            projection[name] = (; inputs,weights,offset)
        end
    end
    projection
end

function _brm_svi_field_source_units(units, state, projection)
    index = Dict(key.coordinate => (label,slot) for (label,saved) in state.units
        for (slot,key) in enumerate(saved.keys) if hasproperty(key,:coordinate))
    result = copy(units)
    roots = Dict{Any,LowerTriangular{Float64,Matrix{Float64}}}()
    for (label,src) in units
        saved = state.units[label]
        selected = [(i,projection[key.coordinate]) for (i,key) in enumerate(saved.keys)
            if hasproperty(key,:coordinate) && haskey(projection,key.coordinate)]
        isempty(selected) && continue
        d = length(src.M)
        M = copy(src.M)
        operators = Dict{Any,Matrix{Float64}}(label => Matrix{Float64}(I,d,d))
        for (i,row) in selected
            operators[label][i,:] .= 0
            M[i] = row.offset
            for (key,w) in zip(row.inputs,row.weights)
                owner,j = index[key]
                A = get!(operators,owner) do
                    zeros(d,length(units[owner].M))
                end
                A[i,j] += w
                μ = units[owner].M[j]
                # An absent owner's conditional can still be based at the
                # globals of an earlier window.
                coupled = get(state.units[owner], :coupling, nothing)
                if coupled !== nothing
                    r = findfirst(==(j),coupled.rows)
                    if r !== nothing
                        μ += dot(view(coupled.W,r,:), state.q.μ[coupled.columns] .- coupled.center)
                    end
                end
                M[i] += w * μ
            end
        end
        covariance = zeros(d,d)
        for (owner,A) in operators
            L = get!(roots, owner) do
                cholesky(Symmetric(units[owner].Λ)).L
            end
            R = L \ A'
            covariance .+= R' * R
        end
        result[label] = merge(src,(; M,Λ=Matrix(inv(cholesky(Symmetric(covariance))))))
    end
    result
end

# A factor-level start provides a Gaussian in packed coordinates. At fixed
# surrounding units the within-unit map has Jacobian A. Its inverse transports
# the supplied covariance; the cross-unit mean was solved jointly above.
function _brm_svi_field_start_factors!(t, q, sourced)
    projection = _brm_svi_field_projection(t,q)
    for (i,block) in enumerate(t.units)
        names = t.m.view.coordinates[block]
        P = [p for p in sourced[i] if haskey(projection, names[p])]
        isempty(P) && continue
        slot = Dict(names[p]=>j for (j,p) in enumerate(P))
        A = zeros(length(P),length(P))
        for (j,p) in enumerate(P)
            row = projection[names[p]]
            for (name,w) in zip(row.inputs,row.weights)
                k = get(slot,name,0)
                k == 0 || (A[j,k] += w)
            end
        end
        C = inv(A)
        _brm_svi_unit_affine!(q,i,[(;positions=P,C)]; transport_mean=false)
    end
end

# Reproduce a failed evaluation with every affected unit mapped at the same
# globals and neighboring draws. This path runs only when reporting an error.
function _brm_svi_diagnostic_point(t,q,centering,frame,i,g,z)
    if !hasproperty(t,:field_sweep)
        packed = _brm_svi_packed_draw(centering,frame,q,i,g)
        return _brm_svi_term_point(t,g,t.units[i],packed(z))
    end
    point = copy(t.field_sweep.others)
    point[t.globals] .= g
    i == 0 || (point[t.units[i]] .= z)
    for (o,block) in enumerate(t.units)
        point[block] .= _brm_svi_centered_coordinates(centering,frame,q,o,g,point[block])
    end
    _brm_svi_sweep_draws!(point,t,q,g)
    point
end

# The diagnostic coordinate swap is in packed space. Invert its proposed
# point before evaluating the same learned target. No worker or sampler
# shares the temporary neighbor state.
function _brm_svi_sweep_finite(t,q,i,x,plan)
    s, g = t.field_sweep,x[t.globals]
    frame = _brm_svi_centering_frame(t.centering,t,g)
    raw = copy(x)
    for (o,block) in enumerate(t.units)
        raw[block] .= _brm_svi_learned_coordinates(t.centering,frame,q,o,g,x[block])
    end
    draw = map(s.fields) do f
        rows = _brm_svi_sweep_draw(f,q,g,0:length(t.units))
        scaled = copy(x[f.field.positions])
        for k in eachindex(scaled)
            r = rows[k]
            r.H == 0 && continue
            loc = sum((r.J[a,1]*scaled[j] for (a,j) in enumerate(f.parents[k])); init=0.0)
            scaled[k] -= r.H*loc
        end
        for k in eachindex(scaled)
            raw[f.field.positions[k]] = scaled[k]/rows[k].D
        end
        rows
    end
    local_sweep = merge(s,(; others=raw,neighbor_calls=zeros(Int,length(t.units))))
    target = merge(t,(;field_sweep=local_sweep))
    v,gg = i == 0 ? _brm_svi_global_term(t.prior,t,g) : (0.0,zeros(length(g)))
    u = i == 0 ? Float64[] : raw[t.units[i]]
    gu = zeros(length(u))
    y,gc,a = _brm_svi_sweep_term!(gg,gu,t.centering,frame,target,q,i,g,u,plan; draw)
    if i > 0
        gg .+= _brm_svi_centering_batch_gradient(t.centering,t,frame,[(;gc,adjoints=(a,))],[2.0])
    end
    isfinite(v+y) && all(isfinite,gg) && all(isfinite,gu) && all(isfinite,gc)
end
