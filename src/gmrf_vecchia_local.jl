# One unit's share of a read-pair prior. The joint log density is a sum of one Gaussian
# conditional per state, so assigning each conditional to its state's owner splits it
# exactly: Σ_i F_i, with F_0 for states no single unit owns. A unit needs two things:
#
#   - F_i and its derivative in the hyperparameters and in global states, an unbiased
#     piece of the whole under minibatching;
#   - the whole gradient on its own states, which also takes the conditionals of states
#     owned elsewhere that condition on one of its own.
#
# Both cost the conditionals touching the unit, never the field. Conditionals with the
# same geometry solve once per evaluation.

"""
The conditionals one owner evaluates, by pattern: `own` are its states, `other` the
states elsewhere conditioning on one of its own, `states` everything either reads.
"""
struct _BRMVecchiaLocalPlan
    owned::Vector{Int32}
    patterns::Vector{Int32}
    own_ptr::Vector{Int}
    own::Vector{Int32}
    other_ptr::Vector{Int}
    other::Vector{Int32}
    states::Vector{Int32}
end

# `owner[s]` in 0:units for every state: 0 for a state no single unit owns. Entry `i + 1`
# of the result is owner i's plan. Storage and time follow the conditioning entries.
function _brm_vecchia_local_plans(f::_BRMGMRFFootprint, owner::AbstractVector{<:Integer}, units::Integer)
    n = length(f.position)
    length(owner) == n || throw(DimensionMismatch(
        "gmrf local plans need one owner per state: $(length(owner)) for $n"))
    pattern = Vector{Int32}(undef, n)
    for p in 1:(length(f.pattern_ptr) - 1), slot in f.pattern_ptr[p]:(f.pattern_ptr[p + 1] - 1)
        pattern[f.pattern_states[slot]] = p
    end
    owned = [Int32[] for _ in 0:units]
    other = [Int32[] for _ in 0:units]
    reads = [Int32[] for _ in 0:units]
    for s in 1:n
        o = owner[s]
        0 <= o <= units || throw(ArgumentError("gmrf state $s has owner $o outside 0:$units"))
        push!(owned[o + 1], s)
        push!(reads[o + 1], s)
        for q in f.neighbor_ptr[s]:(f.neighbor_ptr[s + 1] - 1)
            parent = f.neighbor[q]
            push!(reads[o + 1], parent)
            u = owner[parent]
            if u != o && u != 0
                push!(other[u + 1], s)
            end
        end
    end
    map(0:units) do i
        mine = owned[i + 1]
        theirs = sort!(unique!(other[i + 1]))
        states = reads[i + 1]
        for s in theirs
            push!(states, s)
            for q in f.neighbor_ptr[s]:(f.neighbor_ptr[s + 1] - 1)
                push!(states, f.neighbor[q])
            end
        end
        sort!(unique!(states))
        patterns = sort!(unique!(vcat(pattern[mine], pattern[theirs])))
        own_ptr, own_list = _brm_vecchia_by_pattern(patterns, mine, pattern)
        other_ptr, other_list = _brm_vecchia_by_pattern(patterns, theirs, pattern)
        _BRMVecchiaLocalPlan(mine, patterns, own_ptr, own_list, other_ptr, other_list, states)
    end
end

function _brm_vecchia_by_pattern(patterns, states, pattern)
    ptr = zeros(Int, length(patterns) + 1)
    ptr[1] = 1
    place(p) = searchsortedfirst(patterns, p)
    for s in states
        ptr[place(pattern[s]) + 1] += 1
    end
    cumsum!(ptr, ptr)
    next = copy(ptr)
    out = Vector{Int32}(undef, length(states))
    for s in states
        k = place(pattern[s])
        out[next[k]] = s
        next[k] += 1
    end
    ptr, out
end

# Scratch for one caller at a time. Field-sized, like a sampler's point, and only the
# entries a plan touches are written.
struct _BRMVecchiaLocalWork
    solve::_BRMVecchiaWork
    z::Matrix{Float64}
    own::Matrix{Float64}
    all::Matrix{Float64}
    gown::Vector{Float64}
    gall::Vector{Float64}
    gx::Vector{Float64}
end
function _BRMVecchiaLocalWork(f::_BRMGMRFFootprint)
    C, n = f.nchannels, length(f.position)
    _BRMVecchiaLocalWork(_BRMVecchiaWork(f), zeros(C, n), zeros(C, n), zeros(C, n),
        zeros(f.ncoordinates), zeros(f.ncoordinates), zeros(C))
end

"""
    _brm_vecchia_local!(work, plan, footprint, field, spatial_length, temporal_length, L)

`F_i`, the sum of the plan's owned conditionals at `field`, which need hold values only
at `plan.states`. Returns its value and its derivatives in both lengths and in `L`.
`work.gown` holds its gradient on the field, and `work.gall` the whole density's on the
owner's own coordinates, which also takes the plan's `other` conditionals. Read them at
`plan.states`, then call `_brm_vecchia_local_clear!`.
"""
function _brm_vecchia_local!(work::_BRMVecchiaLocalWork, plan::_BRMVecchiaLocalPlan,
        f::_BRMGMRFFootprint, field, spatial_length, temporal_length, L)
    _brm_gmrf_positive(spatial_length, "spatial length scale")
    _brm_gmrf_positive(temporal_length, "temporal length scale")
    C = f.nchannels
    z = work.z
    @inbounds for s in plan.states, c in 1:C
        value = field[c + C * (s - 1)]
        for k in 1:(c - 1)
            value -= z[k, s] * L[c, k]
        end
        z[c, s] = value / L[c, c]
    end
    solve = work.solve
    kappa = sqrt(2.0) / spatial_length
    value, dkappa, dtemporal = 0.0, 0.0, 0.0
    for (k, p) in enumerate(plan.patterns)
        own = view(plan.own, plan.own_ptr[k]:(plan.own_ptr[k + 1] - 1))
        other = view(plan.other, plan.other_ptr[k]:(plan.other_ptr[k + 1] - 1))
        solved = _brm_vecchia_solve!(solve, f, p, kappa, temporal_length, !isempty(own))
        m, F = solved.m, solved.variance
        if !isempty(own)
            q = _brm_vecchia_states_gradient!(work.own, z, f, own, C, m, F, solve.b, solve.v)
            count = length(own) * C
            value -= (count * log(F) + q / F) / 2
            lengths = _brm_vecchia_length_gradient(solve, f, p, m, F, q, count)
            dkappa += lengths.kappa
            dtemporal += lengths.temporal
        end
        isempty(other) ||
            _brm_vecchia_states_gradient!(work.all, z, f, other, C, m, F, solve.b, solve.v)
    end
    logL = 0.0
    @inbounds for c in 1:C
        logL += log(L[c, c])
    end
    n = length(plan.owned)
    value -= n * logL + n * C * log(2pi) / 2
    # Back from whitened coordinates, as `_brm_vecchia_gradient` does: x = L z.
    L_gradient = zeros(C, C)
    gx = work.gx
    @inbounds for s in plan.states
        for c in C:-1:1
            g = work.own[c, s]
            for k in (c + 1):C
                g -= gx[k] * L[k, c]
            end
            gx[c] = g / L[c, c]
            work.gown[c + C * (s - 1)] = gx[c]
        end
        for c in 1:C, k in 1:c
            L_gradient[c, k] -= gx[c] * z[k, s]
        end
        for c in C:-1:1
            g = work.own[c, s] + work.all[c, s]
            for k in (c + 1):C
                g -= gx[k] * L[k, c]
            end
            gx[c] = g / L[c, c]
            work.gall[c + C * (s - 1)] = gx[c]
        end
    end
    @inbounds for c in 1:C
        L_gradient[c, c] -= n / L[c, c]
    end
    (; value, spatial_length=dkappa * (-kappa / spatial_length), temporal_length=dtemporal, L=L_gradient)
end

function _brm_vecchia_local_clear!(work::_BRMVecchiaLocalWork, plan::_BRMVecchiaLocalPlan, f::_BRMGMRFFootprint)
    C = f.nchannels
    @inbounds for s in plan.states, c in 1:C
        work.own[c, s] = 0.0
        work.all[c, s] = 0.0
        work.gown[c + C * (s - 1)] = 0.0
        work.gall[c + C * (s - 1)] = 0.0
    end
    work
end
