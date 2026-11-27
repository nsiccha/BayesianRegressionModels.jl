# In the private integration frame each independent cdar chain has precision
# x₁² + Σₜ (xₜ - ρ xₜ₋₁)² / (1 - ρ²). Its observation reads have width one
# and its prior is tridiagonal, irrespective of the number of fitted steps.
# RK scores the proper state density at zero, including its ρ normalizer.
mutable struct _BRMSPLCDAR
    declaration::Symbol
    z::Matrix{Int}
    port::Int
    rho::Float64
    diagonal::Matrix{Int}
    lower::Matrix{Int}
    upper::Matrix{Int}
end

# Prepared queries belong to one target/chain. A copied target prepares its
# own reader and pullback; the private backend and its bound data are read-only.
struct _BRMLaplaceFieldFrame{B}
    backend::Any
    blocks::B
    ports::Tuple{Vararg{Symbol}}
    read::Base.RefValue{Any}
    pullback::Base.RefValue{Any}
    cdar::Vector{_BRMSPLCDAR}
end

function _BRMLaplaceFieldFrame(backend, blocks, integrated)
    ports = Tuple(unique(b.port for b in blocks))
    zof = Dict(i => z for (z, i) in enumerate(integrated))
    fields = _BRMSPLCDAR[]
    for b in blocks
        b.kind === :cdar || continue
        z = reshape([zof[i] for i in b.ids], b.shape)
        g, t = size(z)
        push!(fields, _BRMSPLCDAR(b.declaration, z, findfirst(==(b.port), ports), NaN,
            zeros(Int, g, t), zeros(Int, g, t - 1), zeros(Int, g, t - 1)))
    end
    _BRMLaplaceFieldFrame(backend, blocks, ports, Ref{Any}(nothing), Ref{Any}(nothing), fields)
end

function Base.deepcopy_internal(frame::_BRMLaplaceFieldFrame, stack::IdDict)
    haskey(stack, frame) && return stack[frame]
    stack[frame] = _BRMLaplaceFieldFrame(get(stack, frame.backend, frame.backend),
        Base.deepcopy_internal(frame.blocks, stack), frame.ports, Ref{Any}(nothing),
        Ref{Any}(nothing), Base.deepcopy_internal(frame.cdar, stack))
end

function _brm_laplace_field_values(frame::_BRMLaplaceFieldFrame, u)
    frame.read[] === nothing && (frame.read[] = _rk_value_reader(frame.backend, frame.ports))
    frame.read[](u)
end

function _brm_spl_cdar_parameters!(p, u)
    frame = p.walk_frame
    frame === nothing && return nothing
    isempty(frame.cdar) && return nothing
    values = _brm_laplace_field_values(frame, u)
    for f in frame.cdar
        f.rho = values[f.port]
    end
    nothing
end

_brm_spl_cdar_pairs(p) = _brm_spl_cdar_pairs(p.walk_frame)
_brm_spl_cdar_pairs(::Nothing) = Tuple{Int,Int}[]
_brm_spl_cdar_pairs(frame::_BRMLaplaceFieldFrame) =
    [(f.z[g, t - 1], f.z[g, t]) for f in frame.cdar
        for t in 2:size(f.z, 2) for g in axes(f.z, 1)]

function _brm_spl_cdar_positions!(p)
    frame = p.walk_frame
    frame === nothing && return nothing
    for f in frame.cdar
        for i in eachindex(f.z)
            f.diagonal[i] = _brm_spl_nzindex(p.pattern, f.z[i], f.z[i])
        end
        for t in 2:size(f.z, 2), g in axes(f.z, 1)
            a, b = f.z[g, t - 1], f.z[g, t]
            f.lower[g, t - 1] = _brm_spl_nzindex(p.pattern, b, a)
            f.upper[g, t - 1] = _brm_spl_nzindex(p.pattern, a, b)
        end
    end
    nothing
end

function _brm_spl_cdar_restrict!(p, zmap)
    frame = p.walk_frame
    frame === nothing && return nothing
    # Releasing a field rebuilds the working backend in its public frame;
    # this path only remaps fields when other integrated terms are removed.
    for f in frame.cdar
        f.z .= zmap[f.z]
        all(>(0), f.z) || error("a released cdar field needs a new integration frame")
    end
    nothing
end

_brm_spl_cdar_quad(p, z) = _brm_spl_cdar_quad(p.walk_frame, z)
_brm_spl_cdar_quad(::Nothing, z) = 0.0
function _brm_spl_cdar_quad(frame::_BRMLaplaceFieldFrame, z)
    q = 0.0
    for f in frame.cdar
        r = f.rho
        v = 1 - r^2
        for g in axes(f.z, 1)
            q += z[f.z[g, 1]]^2
        end
        for t in 2:size(f.z, 2), g in axes(f.z, 1)
            q += (z[f.z[g, t]] - r * z[f.z[g, t - 1]])^2 / v
        end
    end
    q
end

_brm_spl_cdar_prior!(fc, gradient, p, z, fb, correction) =
    _brm_spl_cdar_prior!(fc, gradient, p, p.walk_frame, z, fb, correction)
_brm_spl_cdar_prior!(fc, gradient, p, ::Nothing, z, fb, correction) = nothing
function _brm_spl_cdar_prior!(fc, gradient, p, frame::_BRMLaplaceFieldFrame, z, fb, correction)
    for f in frame.cdar
        r, v = f.rho, 1 - f.rho^2
        for g in axes(f.z, 1)
            a = f.z[g, 1]
            value = -z[a]^2 / 2
            _brm_spl_add_value!(fc, correction, p.component[a], value)
            fb === nothing || (fb[p.sblock[a]] += value)
            gradient === nothing || (gradient[a] -= z[a])
        end
        for t in 2:size(f.z, 2), g in axes(f.z, 1)
            a, b = f.z[g, t - 1], f.z[g, t]
            residual = z[b] - r * z[a]
            value = -residual^2 / (2v)
            _brm_spl_add_value!(fc, correction, p.component[a], value)
            if fb !== nothing
                fb[p.sblock[a]] += value
                p.sblock[a] == p.sblock[b] || (fb[p.sblock[b]] += value)
            end
            if gradient !== nothing
                gradient[a] += r * residual / v
                gradient[b] -= residual / v
            end
        end
    end
    nothing
end

# Visit Q's entries, including both triangles, and their derivatives in ρ.
# Diagonal contributions accumulate across adjacent transitions.
function _brm_cdar_entries(visit::F, field::_BRMSPLCDAR) where {F}
    r = field.rho
    v = 1 - r^2
    d = 2r / v^2
    off = -(1 + r^2) / v^2
    for g in axes(field.z, 1)
        a = field.z[g, 1]
        visit(a, a, field.diagonal[g, 1], 1.0, 0.0)
    end
    for t in 2:size(field.z, 2), g in axes(field.z, 1)
        a, b = field.z[g, t - 1], field.z[g, t]
        visit(a, a, field.diagonal[g, t - 1], r^2 / v, d)
        visit(b, b, field.diagonal[g, t], 1 / v, d)
        visit(a, b, field.upper[g, t - 1], -r / v, off)
        visit(b, a, field.lower[g, t - 1], -r / v, off)
    end
    nothing
end

_brm_spl_cdar_precision!(nz, p) = _brm_spl_cdar_precision!(nz, p.walk_frame)
_brm_spl_cdar_precision!(nz, ::Nothing) = nothing
function _brm_spl_cdar_precision!(nz, frame::_BRMLaplaceFieldFrame)
    for f in frame.cdar
        _brm_cdar_entries(f) do a, b, k, value, derivative
            nz[k] += value
        end
    end
    nothing
end

function _brm_spl_field_sparse(p, n)
    Q = _brm_spl_gmrf_sparse(p, n)
    frame = p.walk_frame
    frame === nothing && return Q
    isempty(frame.cdar) && return Q
    rows, cols, vals = Int[], Int[], Float64[]
    for f in frame.cdar
        _brm_cdar_entries(f) do a, b, k, value, derivative
            push!(rows, a); push!(cols, b); push!(vals, value)
        end
    end
    A = sparse(rows, cols, vals, n, n)
    Q === nothing ? A : Q + A
end

# Contract the moving precision with the covariance, mode and implicit mean
# shift. The zero-state prior's RK gradient already contains the normalizer.
_brm_spl_cdar_gradient!(gradient, p, state, S, v, active) =
    _brm_spl_cdar_gradient!(gradient, p, p.walk_frame, state, S, v, active)
_brm_spl_cdar_gradient!(gradient, p, ::Nothing, state, S, v, active) = gradient
function _brm_spl_cdar_gradient!(gradient, p, frame::_BRMLaplaceFieldFrame, state, S, v, active)
    isempty(frame.cdar) && return gradient
    components = active === nothing ? nothing : Set(p.term_component[t]
        for t in eachindex(active) if active[t])
    seeds = zeros(length(frame.ports))
    z = state.z
    for f in frame.cdar
        _brm_cdar_entries(f) do a, b, k, value, derivative
            components === nothing || p.component[a] in components || return
            seeds[f.port] -= (S[k] + z[a] * z[b] + v[a] * z[b] + z[a] * v[b]) * derivative / 2
        end
    end
    frame.pullback[] === nothing &&
        (frame.pullback[] = _rk_value_pullback(frame.backend, frame.ports, state.uA))
    gradient .+= frame.pullback[](state.uA, Tuple(seeds))[p.population]
    gradient
end
