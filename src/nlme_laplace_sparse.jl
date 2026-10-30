# src/nlme_laplace_sparse.jl — one sparse Laplace over chosen random-effect
# groupings.
#
# `BRMStructuredLaplaceProblem` integrates each unit's block on its own, d
# effects and a d × d Hessian per unit. Here the integrated effects are every
# level of the named groupings' random-effect blocks, wherever they sit:
# nested in the units of an NLME view or not, crossed, with any number of
# levels and rows per level. The rows are the structured path's
# (`_brm_lap_row`, `_brm_slap_terms`): term t's link-scale parameters are
#
#     ψ_t = A_t(φ) + Σ_k Z_t,k b_k,ℓ,    b_k,ℓ = M_k,ℓ(φ) z_k,ℓ,
#
# one level ℓ = ℓ_k(t) per integrated block k the row reads, Z_t,k the row's
# design on that block (constant data) and M_k,ℓ the level's draws map
# (diag(scales) L; per level, so a stratified block's strata differ). With
# C = ∂ψ/∂z the negative Hessian in z is the sparse
#
#     h = I + Cᵀ W C,    W = −ℓ″ (one 1 × 1 or 2 × 2 block per term),
#
# factored by CHOLMOD at every Newton step and at the mode. The value and
# gradient are the structured path's (its header), with each unit's h_i⁻¹
# replaced by the selected inverse of h on h's own pattern
# (SelectedInversion.jl). That holds every entry τ and Γ read, because a term
# couples exactly the effects it reads.
#
# The mode search steps per connected component of h's graph: units that
# share no term (the units of an NLME view, nested groupings inside them) each
# take their own step length, stop and backtracking, as in the structured
# path; a crossed grouping joins the units it crosses into one component.

# An integrated random-effect block: its draws' indices in the integrated
# vector z (levels × margins), and the `omega` entries (scales, correlation
# factor) its draws map M reads.
struct _BRMSlapBlock
    declaration::Symbol
    group::Symbol
    K::Int
    z::Matrix{Int}
    omega::Vector{Int}
end

"""
    BRMSparseLaplaceProblem

The Laplace of a regression-shaped RK model with every level of chosen
random-effect groupings integrated by one sparse Laplace; see
[`brm_laplace_problem`](@ref)'s `integrate`. Same interface as
[`BRMStructuredLaplaceProblem`](@ref): LogDensityProblems order 1 over `names`
(packed positions `population`), [`brm_laplace_draws`](@ref), and the counters
`nonfinite` and `unconverged`. `integrate` names the integrated groupings,
`integrated` their draws' packed positions, `components` the groups of
integrated draws that share likelihood terms.
"""
mutable struct BRMSparseLaplaceProblem{K}
    backend::RKBRMI
    population::Vector{Int}
    names::Vector{Symbol}
    integrate::Vector{Symbol}
    integrated::Vector{Int}
    blocks::Vector{_BRMSlapBlock}
    kernels::K
    # Likelihood terms and their parameter entries, as in the structured path.
    family::Vector{Int8}
    y::Vector{Float64}
    trials::Vector{Float64}
    constant::Vector{Float64}
    entry::Vector{Int}
    moving::Vector{Bool}
    port::Vector{Int}
    index::Vector{Int}
    link::Vector{Symbol}
    fixed::Vector{Float64}
    # The blocks each entry reads: reads `rptr[e]:rptr[e+1]-1`, read r of block
    # `rblock[r]` at level `rlevel[r]` with design Z_e,k = `zval[zoff[r] .+ (1:K)]`.
    rptr::Vector{Int}
    rblock::Vector{Int}
    rlevel::Vector{Int}
    zoff::Vector{Int}
    zval::Vector{Float64}
    # Each term's effects (`tz[tptr[t]:tptr[t+1]-1]`, ascending), the local
    # position of read r's first effect (`rloc[r]`, its K effects consecutive
    # and in margin order) and h's nonzeros at the term's m × m effect pairs
    # (`tnz[nptr[t] + (j-1) m + (i-1)]`).
    tptr::Vector{Int}
    tz::Vector{Int}
    rloc::Vector{Int}
    nptr::Vector{Int}
    tnz::Vector{Int}
    pattern::SparseMatrixCSC{Float64,Int}
    diagonal::Vector{Int}
    component::Vector{Int}
    components::Vector{Vector{Int}}
    term_component::Vector{Int}
    omega::Vector{Int}
    affine::Bool
    φ0::Vector{Float64}
    A0::Vector{Float64}
    jptr::Vector{Int}
    jcol::Vector{Int}
    jval::Vector{Float64}
    u::Vector{Float64}
    factor::Base.RefValue{Any}
    tol::Float64
    maxiter::Int
    nonfinite::Int
    unconverged::Int
end

Base.show(io::IO, p::BRMSparseLaplaceProblem) = print(io,
    "BRMSparseLaplaceProblem(", length(p.population), " population coordinates, ",
    length(p.integrated), " integrated random effects of ", join(p.integrate, ", "),
    " in ", length(p.components), " components, ", length(p.family),
    " likelihood terms)")

function _brm_slap_zero_integrated!(u, p::BRMSparseLaplaceProblem)
    u[p.integrated] .= 0.0
    u
end

# The model's random-effect blocks over the named groupings.
function _brm_spl_records(backend::RKBRMI, integrate)
    records = [r for r in _rk_coordinate_records(backend.plan)
        if r.kind in (:ranef, :stratified_ranef)]
    available = unique(Symbol[r.group for r in records])
    for g in integrate
        g in available || _brm_nlme_error("brm_laplace_problem: integrate names " *
            "`$g`, which groups no random-effect block of the model; its random-effect " *
            "groupings are $(isempty(available) ? "none" : join(repr.(available), ", "))")
    end
    chosen = [r for r in records if r.group in integrate]
    for r in chosen
        r.kind === :ranef || _brm_nlme_error("brm_laplace_problem: the random-effect " *
            "block over `$(r.group)` is stratified, which the sparse Laplace does not " *
            "integrate yet")
        get(r, :mixing, nothing) === nothing || _brm_nlme_error("brm_laplace_problem: " *
            "the random-effect block over `$(r.group)` is Student-t, so its draws are " *
            "not Gaussian and cannot be integrated by a Laplace")
    end
    chosen
end

function _brm_sparse_laplace_problem(backend::RKBRMI, integrate; ad_backend, tol,
        maxiter)
    integrate = Symbol[g for g in (integrate isa Symbol ? (integrate,) : integrate)]
    isempty(integrate) && throw(ArgumentError(
        "brm_laplace_problem: integrate names no grouping"))
    refusal = _brm_slap_response_refusal(backend)
    refusal === nothing || _brm_nlme_error("brm_laplace_problem: integrate = " *
        "$(repr(Tuple(integrate))) needs every observation to be a regression row " *
        "of a closed-form family, but $refusal")
    responses = _brm_slap_responses(backend.plan)
    records = _brm_spl_records(backend, integrate)
    names = _rk_layout_coordinate_names(backend)
    position = Dict{Symbol,Int}(n => i for (i, n) in enumerate(names))
    owned(declaration) = [i for (i, n) in enumerate(names) if _brm_nlme_owned(n, declaration)]
    # The integrated draws, block by block, levels × margins.
    integrated = Int[]
    omega = Int[]
    blocks = _BRMSlapBlock[]
    for r in records
        K = length(r.margins)
        nlevels = length(_rk_ranef_record_levels(backend, r))
        z = Matrix{Int}(undef, nlevels, K)
        for l in 1:nlevels, k in 1:K
            push!(integrated, get(position, Symbol(r.z, ".", l, ".", k)) do
                _brm_nlme_error("brm_laplace_problem: internal: draw `$(r.z).$l.$k` " *
                    "is not packed")
            end)
            z[l, k] = length(integrated)
        end
        own = Int[]
        for declaration in (isnothing(r.scales) ? (r.scale,) : r.scales)
            append!(own, owned(declaration))
        end
        isnothing(r.L) || append!(own, owned(r.L))
        js = Int[]
        for i in own
            j = findfirst(==(i), omega)
            j === nothing && (push!(omega, i); j = length(omega))
            push!(js, j)
        end
        push!(blocks, _BRMSlapBlock(_brm_nlme_ranef_declaration(r), r.group, K, z, js))
    end
    allunique(integrated) || _brm_nlme_error("brm_laplace_problem: internal: a draw " *
        "belongs to two integrated blocks")
    population = setdiff(1:length(names), integrated)
    # Random-effect draws left to the target make A bilinear in the population.
    sampled = any(r -> r.group ∉ integrate, (r for r in _rk_coordinate_records(backend.plan)
        if r.kind in (:ranef, :stratified_ranef)))
    families, ports = _brm_slap_ports(backend, responses)
    u0 = zeros(Float64, length(names))
    effects = Tuple(b.declaration for b in blocks)
    kernels = _rk_slap_kernels(backend, Tuple(ports), effects, omega, u0, ad_backend)
    nrows = length(rk_bound_data(backend)[first(responses).response])
    t = _brm_slap_terms(backend, responses, families, ports, _rk_slap_values(kernels, u0),
        (1:nrows,))
    nψ = length(t.port)
    p = BRMSparseLaplaceProblem(backend, population, names[population], integrate,
        integrated, blocks, kernels, t.family, t.y, t.trials, t.constant, t.entry,
        fill(true, length(t.family)), t.port, t.index, t.link, t.fixed,
        Int[], Int[], Int[], Int[], Float64[], Int[], Int[], Int[], Int[], Int[],
        spzeros(0, 0), Int[], Int[], Vector{Int}[], Int[], omega, false,
        u0[population], zeros(nψ), Int[], Int[], Float64[], copy(u0), Ref{Any}(nothing),
        Float64(tol), Int(maxiter), 0, 0)
    _brm_spl_design!(p, u0)
    sampled || _brm_slap_population_design!(p, u0)
    _brm_slap_verify!(p, u0, t.response, t.row)
    _brm_spl_verify_prior(p, u0)
    _brm_spl_pattern!(p)
    p
end

# Every integrated level's draws map M (b = M z) at a packed point, block by
# block as K × K × levels, and per `omega` coordinate its derivative (zero
# for the blocks that coordinate does not move). Probe k sets every level's
# draw k to one in every block with at least k margins.
function _brm_spl_effect_maps(p::BRMSparseLaplaceProblem, u)
    usyn = _brm_slap_zero_integrated!(copy(u), p)
    M = [zeros(b.K, b.K, size(b.z, 1)) for b in p.blocks]
    dM = [[zeros(b.K, b.K, size(b.z, 1)) for b in p.blocks] for _ in p.omega]
    for k in 1:maximum(b -> b.K, p.blocks)
        probe = copy(usyn)
        for b in p.blocks
            b.K >= k && (probe[p.integrated[view(b.z, :, k)]] .= 1.0)
        end
        value, tangents = _rk_slap_effects(p.kernels, probe)
        for (j, b) in enumerate(p.blocks)
            b.K >= k || continue
            _brm_spl_store_column!(M[j], value[j], k, b)
            for o in b.omega
                _brm_spl_store_column!(dM[o][j], tangents[o][j], k, b)
            end
        end
    end
    M, dM
end

function _brm_spl_store_column!(dest, draws, k, b::_BRMSlapBlock)
    size(draws, 1) == size(b.z, 1) || _brm_nlme_error("brm_laplace_problem: internal: " *
        "block `$(b.declaration)` reads $(size(draws, 1)) levels, the layout $(size(b.z, 1))")
    for l in axes(b.z, 1), a in 1:b.K
        dest[a, k, l] = draws isa AbstractVector ? draws[l] : draws[l, a]
    end
    dest
end

# The entries' reads: which level of which integrated block each entry reads,
# and its design there, from unit steps of the draws at u0. Shifting draw k of
# every level of a block by one moves each entry by its C there; shifting
# level ℓ's by ℓ moves it by ℓ times that, which names the level. Then
# Z = C M⁻¹ per read, and the structure is checked at another point.
function _brm_spl_design!(p::BRMSparseLaplaceProblem, u0)
    nψ = length(p.port)
    base = _brm_slap_zero_integrated!(copy(u0), p)
    ψ0 = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, base))
    M0, _ = _brm_spl_effect_maps(p, base)
    ψ1 = zeros(nψ); ψ2 = zeros(nψ)
    # Per entry and block: the level read (0 none) and C there (K numbers).
    level = [zeros(Int, nψ) for _ in p.blocks]
    C = [zeros(b.K, nψ) for b in p.blocks]
    for (j, b) in enumerate(p.blocks), k in 1:b.K
        unit = copy(base); scaled = copy(base)
        for l in axes(b.z, 1)
            unit[p.integrated[b.z[l, k]]] = 1.0
            scaled[p.integrated[b.z[l, k]]] = l
        end
        _brm_slap_psi!(ψ1, p, _rk_slap_values(p.kernels, unit))
        _brm_slap_psi!(ψ2, p, _rk_slap_values(p.kernels, scaled))
        for e in 1:nψ
            d1 = ψ1[e] - ψ0[e]
            abs(d1) <= 1e-12 * max(1.0, abs(ψ0[e])) && continue
            ratio = (ψ2[e] - ψ0[e]) / d1
            l = round(Int, ratio)
            (1 <= l <= size(b.z, 1) && abs(ratio - l) <= 1e-6 * max(1.0, abs(ratio))) ||
                _brm_nlme_error("brm_laplace_problem: likelihood parameter $e reads " *
                    "several levels of the random-effect block over `$(b.group)`, " *
                    "which the sparse Laplace does not integrate")
            level[j][e] == 0 || level[j][e] == l || _brm_nlme_error(
                "brm_laplace_problem: likelihood parameter $e reads two levels of the " *
                "random-effect block over `$(b.group)`")
            level[j][e] = l
            C[j][k, e] = d1
        end
    end
    p.rptr = [1]; p.rblock = Int[]; p.rlevel = Int[]; p.zoff = Int[]; p.zval = Float64[]
    for e in 1:nψ
        for (j, b) in enumerate(p.blocks)
            l = level[j][e]
            l == 0 && continue
            m = view(M0[j], :, :, l)
            cond(m) < 1e10 || _brm_nlme_error("brm_laplace_problem: the draws map of " *
                "the random-effect block over `$(b.group)` is singular at the zero " *
                "unconstrained point")
            push!(p.rblock, j); push!(p.rlevel, l); push!(p.zoff, length(p.zval))
            append!(p.zval, m' \ view(C[j], :, e))
        end
        push!(p.rptr, length(p.rblock) + 1)
    end
    for t in eachindex(p.family)
        e = p.entry[t]
        p.moving[t] = _brm_lap_width(p.family[t]) == 2 && p.rptr[e + 2] > p.rptr[e + 1]
    end
    # The structure at another population point and other draws.
    rng = Random.Xoshiro(20261030)
    u1 = copy(u0)
    u1[p.population] .+= 0.1 .* randn(rng, length(p.population))
    z1 = randn(rng, length(p.integrated))
    u1[p.integrated] .= z1
    uA = _brm_slap_zero_integrated!(copy(u1), p)
    A1 = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, uA))
    exact = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, u1))
    M1, _ = _brm_spl_effect_maps(p, uA)
    for e in 1:nψ
        predicted = A1[e] + _brm_spl_cdot(p, M1, e, z1)
        isapprox(predicted, exact[e]; rtol=1e-8, atol=1e-8) || _brm_nlme_error(
            "brm_laplace_problem: likelihood parameter $e is $(exact[e]) in the model " *
            "and $predicted as affine in the integrated random effects, so the sparse " *
            "Laplace does not apply")
    end
    p
end

# C_e z for an entry, over its reads: Σ_r Z_rᵀ M_r z_r.
function _brm_spl_cdot(p::BRMSparseLaplaceProblem, M, e, z)
    s = 0.0
    @inbounds for r in p.rptr[e]:(p.rptr[e + 1] - 1)
        b = p.blocks[p.rblock[r]]
        l = p.rlevel[r]
        Mr = M[p.rblock[r]]
        for a in 1:b.K, c in 1:b.K
            s += p.zval[p.zoff[r] + a] * Mr[a, c, l] * z[b.z[l, c]]
        end
    end
    s
end

# The integrated draws' prior must be N(0, I) and depend on nothing else:
# the value adds −|ẑ|²/2 and the population prior is RK's at ẑ = 0.
function _brm_spl_verify_prior(p::BRMSparseLaplaceProblem, u0)
    rng = Random.Xoshiro(20261031)
    for _ in 1:2
        u = copy(u0)
        u[p.population] .+= 0.1 .* randn(rng, length(p.population))
        z = randn(rng, length(p.integrated))
        uA = _brm_slap_zero_integrated!(copy(u), p)
        u[p.integrated] .= z
        Δ = _rk_slap_prior(p.kernels, u) - _rk_slap_prior(p.kernels, uA)
        isapprox(Δ, -sum(abs2, z) / 2; rtol=1e-9, atol=1e-9) || _brm_nlme_error(
            "brm_laplace_problem: the integrated random effects' prior is not N(0, I) " *
            "in their packed coordinates (adaptive centring or a constrained block), " *
            "so the sparse Laplace does not apply")
    end
    nothing
end

# h's pattern (the identity and every pair of effects a term reads), each
# term's place in it, and the components: effects connected through terms.
function _brm_spl_pattern!(p::BRMSparseLaplaceProblem)
    n = length(p.integrated)
    p.tptr = [1]; p.tz = Int[]; p.rloc = zeros(Int, length(p.rblock))
    for t in eachindex(p.family)
        e = p.entry[t]
        es = e:(e + _brm_lap_width(p.family[t]) - 1)
        zs = Int[]
        for e2 in es, r in p.rptr[e2]:(p.rptr[e2 + 1] - 1)
            b = p.blocks[p.rblock[r]]
            append!(zs, view(b.z, p.rlevel[r], :))
        end
        sort!(unique!(zs))
        for e2 in es, r in p.rptr[e2]:(p.rptr[e2 + 1] - 1)
            b = p.blocks[p.rblock[r]]
            first_z = b.z[p.rlevel[r], 1]
            p.rloc[r] = searchsortedfirst(zs, first_z)
            all(zs[p.rloc[r] + c - 1] == b.z[p.rlevel[r], c] for c in 1:b.K) ||
                error("brm_laplace_problem: internal: a read's effects are not consecutive")
        end
        append!(p.tz, zs)
        push!(p.tptr, length(p.tz) + 1)
    end
    I = collect(1:n); J = collect(1:n)
    for t in eachindex(p.family), i in p.tptr[t]:(p.tptr[t + 1] - 1),
            j in p.tptr[t]:(p.tptr[t + 1] - 1)
        push!(I, p.tz[i]); push!(J, p.tz[j])
    end
    pattern = sparse(I, J, 1.0, n, n)
    nonzeros(pattern) .= 0.0
    index(i, j) = (r = nzrange(pattern, j); r[searchsortedfirst(view(rowvals(pattern), r), i)])
    p.diagonal = [index(i, i) for i in 1:n]
    p.nptr = [1]; p.tnz = Int[]
    for t in eachindex(p.family)
        zs = view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        for j in zs, i in zs
            push!(p.tnz, index(i, j))
        end
        push!(p.nptr, length(p.tnz) + 1)
    end
    p.pattern = pattern
    # Components by union-find over the terms' effects.
    parent = collect(1:n)
    root(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    for t in eachindex(p.family)
        zs = view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        for i in Iterators.drop(zs, 1)
            a, b = root(first(zs)), root(i)
            a == b || (parent[a] = b)
        end
    end
    roots = [root(i) for i in 1:n]
    labels = Dict{Int,Int}()
    p.component = [get!(labels, r, length(labels) + 1) for r in roots]
    p.components = [Int[] for _ in 1:length(labels)]
    for i in 1:n
        push!(p.components[p.component[i]], i)
    end
    p.term_component = [p.tptr[t + 1] > p.tptr[t] ? p.component[p.tz[p.tptr[t]]] : 0
        for t in eachindex(p.family)]
    p
end

# One term's C rows on its local effects (c1 for its first parameter, c2 for
# its second), at the levels' draws maps M.
function _brm_spl_crows!(c1, c2, p::BRMSparseLaplaceProblem, M, t)
    fill!(c1, 0.0); fill!(c2, 0.0)
    e = p.entry[t]
    for (e2, c) in ((e, c1), (e + 1, c2))
        e2 == e || _brm_lap_width(p.family[t]) == 2 || break
        @inbounds for r in p.rptr[e2]:(p.rptr[e2 + 1] - 1)
            b = p.blocks[p.rblock[r]]
            l = p.rlevel[r]
            Mr = M[p.rblock[r]]
            for col in 1:b.K, a in 1:b.K
                c[p.rloc[r] + col - 1] += p.zval[p.zoff[r] + a] * Mr[a, col, l]
            end
        end
    end
    nothing
end

# The terms' rows at effects z: each term's local C rows and parameter
# values, its density summed per component into `fc` (terms that read no
# integrated effect into the returned total), and with `derivatives` the
# z-gradient `g` and h's nonzeros `hnz`. `third` selects the gradient's rows
# (full values and third derivatives, see `_brm_lap_row`); `rows` and `crows`,
# when given, keep each term's row and C rows.
function _brm_spl_assemble!(fc, g, hnz, p::BRMSparseLaplaceProblem, A, M, z;
        derivatives::Bool, third::Bool=false, rows=nothing, crows=nothing)
    for (c, members) in enumerate(p.components)
        fc[c] = -sum(i -> z[i]^2, members) / 2
    end
    if derivatives
        g .= .-z
        fill!(nonzeros(hnz), 0.0)
        nonzeros(hnz)[p.diagonal] .= 1.0
    end
    rest = 0.0
    mmax = maximum(t -> p.tptr[t + 1] - p.tptr[t], eachindex(p.family); init=0)
    c1 = zeros(mmax); c2 = zeros(mmax)
    nz = derivatives ? nonzeros(hnz) : Float64[]
    @inbounds for t in eachindex(p.family)
        comp = p.term_component[t]
        # A term no integrated effect moves matters only to the full value.
        comp == 0 && !third && continue
        e = p.entry[t]
        m = p.tptr[t + 1] - p.tptr[t]
        zs = view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        two = _brm_lap_width(p.family[t]) == 2
        moving = p.moving[t]
        _brm_spl_crows!(c1, c2, p, M, t)
        a = A[e]; bb = two ? A[e + 1] : 0.0
        for i in 1:m
            a += c1[i] * z[zs[i]]
            moving && (bb += c2[i] * z[zs[i]])
        end
        r = _brm_lap_row(p.family[t], p.y[t], p.trials[t], a, bb, third, moving)
        rows === nothing || (rows[t] = r)
        crows === nothing || (crows[t] = (c1[1:m], c2[1:m]))
        if comp == 0
            rest += r.v
            continue
        end
        fc[comp] += r.v
        derivatives || continue
        for i in 1:m
            g[zs[i]] += c1[i] * r.g1 + (moving ? c2[i] * r.g2 : 0.0)
        end
        base = p.nptr[t] - 1
        for j in 1:m, i in 1:m
            v = r.h11 * c1[i] * c1[j]
            moving && (v += r.h12 * (c1[i] * c2[j] + c2[i] * c1[j]) + r.h22 * c2[i] * c2[j])
            nz[p.tnz[base + (j - 1) * m + i]] -= v
        end
    end
    rest
end

# The factor of h + shift I, reusing the symbolic analysis.
function _brm_spl_factor!(p::BRMSparseLaplaceProblem, h; shift=0.0)
    H = Hermitian(h, :L)
    if p.factor[] === nothing
        F = cholesky(H; shift, check=false)
        p.factor[] = F
        return F
    end
    cholesky!(p.factor[], H; shift, check=false)
end

# The Newton direction h⁻¹ g, or where h is not positive definite away from
# the mode, (h + μ I)⁻¹ g with the smallest μ = 10^k (k ≥ −4) that is.
function _brm_spl_direction(p::BRMSparseLaplaceProblem, h, g)
    F = _brm_spl_factor!(p, h)
    issuccess(F) && return F \ g
    μ = 1e-4
    for _ in 1:12
        F = _brm_spl_factor!(p, h; shift=μ)
        issuccess(F) && return F \ g
        μ *= 10
    end
    nothing
end

# The population point's A (link-scale parameters at z = 0), draws maps and,
# when A is not affine, the RK values behind it.
function _brm_spl_population(p::BRMSparseLaplaceProblem, φ::AbstractVector)
    length(φ) == length(p.population) || throw(DimensionMismatch(
        "the population point has length $(length(φ)), the problem has " *
        "$(length(p.population)) population coordinates"))
    uA = copy(p.u)
    uA[p.population] .= φ
    _brm_slap_zero_integrated!(uA, p)
    values = p.affine ? nothing : _rk_slap_values(p.kernels, uA)
    A = p.affine ? _brm_slap_affine_A(p, φ) : _brm_slap_psi!(zeros(length(p.port)), p, values)
    M, dM = _brm_spl_effect_maps(p, uA)
    (; uA, values, A, M, dM)
end

# Every component's mode at population point φ, from the previous modes: one
# factor per Newton step, each component its own step length (backtracking on
# its own terms), stop (Newton decrement at most `tol`, or at its rounding
# floor) and failure. Then h at the modes and its factor.
function _brm_spl_mode!(p::BRMSparseLaplaceProblem, φ::AbstractVector)
    pop = _brm_spl_population(p, φ)
    if !(all(isfinite, pop.A) && all(m -> all(isfinite, m), pop.M))
        p.nonfinite += 1
        return (; finite=false, value=-Inf, iterations=0)
    end
    n = length(p.integrated)
    nc = length(p.components)
    z = p.u[p.integrated]
    fc = zeros(nc); g = zeros(n); h = copy(p.pattern)
    _brm_spl_assemble!(fc, g, h, p, pop.A, pop.M, z; derivatives=true)
    if !(all(isfinite, fc) && all(isfinite, g))
        z .= 0.0
        _brm_spl_assemble!(fc, g, h, p, pop.A, pop.M, z; derivatives=true)
        all(isfinite, fc) && all(isfinite, g) || (p.nonfinite += 1;
            return (; finite=false, value=-Inf, iterations=0))
    end
    converged = falses(nc)
    previous = fill(Inf, nc)
    λ = zeros(nc)
    ft = zeros(nc)
    iterations = 0
    while !all(converged)
        if iterations == p.maxiter
            p.unconverged += 1
            @warn("brm_laplace_problem: $(count(!, converged)) of $nc components' " *
                "modes did not converge in $(p.maxiter) Newton iterations at population " *
                "point $(repr(collect(φ))); its log density is -Inf", maxlog=3)
            return (; finite=false, value=-Inf, iterations)
        end
        iterations += 1
        all(isfinite, nonzeros(h)) || (p.nonfinite += 1;
            return (; finite=false, value=-Inf, iterations))
        δ = _brm_spl_direction(p, h, g)
        δ === nothing && (p.nonfinite += 1; return (; finite=false, value=-Inf, iterations))
        fill!(λ, 0.0)
        for i in 1:n
            λ[p.component[i]] += g[i] * δ[i]
        end
        for c in 1:nc
            converged[c] && continue
            if λ[c] <= p.tol || (λ[c] <= 1e-8 && λ[c] >= previous[c] / 4)
                converged[c] = true
            else
                previous[c] = λ[c]
            end
        end
        all(converged) && break
        # Each moving component halves its own step until it gains.
        t = [converged[c] ? 0.0 : 1.0 for c in 1:nc]
        accepted = copy(converged)
        trial = copy(z)
        for _ in 1:60
            for i in 1:n
                c = p.component[i]
                accepted[c] && continue
                trial[i] = z[i] + t[c] * δ[i]
            end
            _brm_spl_assemble!(ft, g, h, p, pop.A, pop.M, trial; derivatives=false)
            for c in 1:nc
                accepted[c] && continue
                if isfinite(ft[c]) && (λ[c] <= 1e-8 || ft[c] >= fc[c] + 1e-4 * t[c] * λ[c])
                    accepted[c] = true
                    for i in p.components[c]
                        z[i] = trial[i]
                    end
                else
                    t[c] /= 2
                end
            end
            all(accepted) && break
        end
        all(accepted) || (p.nonfinite += 1; return (; finite=false, value=-Inf, iterations))
        _brm_spl_assemble!(fc, g, h, p, pop.A, pop.M, z; derivatives=true)
    end
    F = _brm_spl_factor!(p, h)
    issuccess(F) || _brm_nlme_error("brm_laplace_problem: the integrated effects' " *
        "density has a stationary point that is not a maximum at this population point")
    p.u[p.integrated] .= z
    u = copy(pop.uA)
    u[p.integrated] .= z
    (; finite=true, pop..., z, u, h, F, iterations)
end

function _brm_spl_value(p::BRMSparseLaplaceProblem, state)
    state.finite || return -Inf
    fc = zeros(length(p.components))
    rest = _brm_spl_assemble!(fc, nothing, nothing, p, state.A, state.M, state.z;
        derivatives=false, third=true)
    sum(p.constant) + sum(fc) + rest + length(p.integrated) / 2 * log(2π) -
        logdet(state.F) / 2 + _rk_slap_prior(p.kernels, state.uA)
end

# Value and gradient over the population coordinates: the structured path's
# (its header) with S = h⁻¹ on h's pattern for each unit's h_i⁻¹. Per term,
# with local C rows c1, c2: m = cᵀ S c, τ from m and ℓ‴, s = ½ Σ Cᵀ τ, v =
# h⁻¹ s, w = ℓ′ + ½ τ + ℓ″ C v; per integrated level, Γ = Σ Z (w ẑᵀ + ℓ′ vᵀ +
# (ℓ″ C S)ᵀ) on its draws, contracted with ∂M.
function _brm_spl_value_and_gradient(p::BRMSparseLaplaceProblem, state)
    n = length(p.integrated)
    nterms = length(p.family)
    fc = zeros(length(p.components))
    rows = Vector{_BRMLapRow}(undef, nterms)
    crows = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, nterms)
    rest = _brm_spl_assemble!(fc, nothing, nothing, p, state.A, state.M, state.z;
        derivatives=false, third=true, rows, crows)
    value = sum(p.constant) + sum(fc) + rest + n / 2 * log(2π) - logdet(state.F) / 2
    Z = selinv(state.F; depermute=true).Z
    S = nonzeros(selinv_extract(Z, p.pattern))
    s = zeros(n)
    taus = zeros(2, nterms)
    @inbounds for t in 1:nterms
        p.term_component[t] == 0 && continue
        r = rows[t]
        c1, c2 = crows[t]
        m = length(c1)
        base = p.nptr[t] - 1
        moving = p.moving[t]
        m11 = m12 = m22 = 0.0
        for j in 1:m, i in 1:m
            Sij = S[p.tnz[base + (j - 1) * m + i]]
            m11 += c1[i] * Sij * c1[j]
            moving && (m12 += c1[i] * Sij * c2[j]; m22 += c2[i] * Sij * c2[j])
        end
        two = _brm_lap_width(p.family[t]) == 2
        τ1 = m11 * r.t111 + 2m12 * r.t112 + m22 * r.t122
        τ2 = two ? m11 * r.t112 + 2m12 * r.t122 + m22 * r.t222 : 0.0
        taus[1, t] = τ1; taus[2, t] = τ2
        zs = view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        for i in 1:m
            s[zs[i]] += (c1[i] * τ1 + (moving ? c2[i] * τ2 : 0.0)) / 2
        end
    end
    v = state.F \ s
    w = zeros(length(p.port))
    Γ = [zeros(b.K, b.K, size(b.z, 1)) for b in p.blocks]
    LCS = Float64[]
    @inbounds for t in 1:nterms
        r = rows[t]
        e = p.entry[t]
        two = _brm_lap_width(p.family[t]) == 2
        if p.term_component[t] == 0
            w[e] = r.g1
            two && (w[e + 1] = r.g2)
            continue
        end
        c1, c2 = crows[t]
        m = length(c1)
        zs = view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        moving = p.moving[t]
        cv1 = cv2 = 0.0
        for i in 1:m
            cv1 += c1[i] * v[zs[i]]
            moving && (cv2 += c2[i] * v[zs[i]])
        end
        w1 = r.g1 + taus[1, t] / 2 + r.h11 * cv1 + r.h12 * cv2
        w[e] = w1
        w2 = 0.0
        if two
            w2 = r.g2 + taus[2, t] / 2 + r.h12 * cv1 + r.h22 * cv2
            w[e + 1] = w2
        end
        # (ℓ″ C S) per parameter, on the term's local effects.
        base = p.nptr[t] - 1
        resize!(LCS, 2m); fill!(LCS, 0.0)
        for j in 1:m, i in 1:m
            Sij = S[p.tnz[base + (j - 1) * m + i]]
            LCS[j] += (r.h11 * c1[i] + (moving ? r.h12 * c2[i] : 0.0)) * Sij
            moving && (LCS[m + j] += (r.h12 * c1[i] + r.h22 * c2[i]) * Sij)
        end
        for (k, (e2, wk, gk)) in enumerate(((e, w1, r.g1), (e + 1, w2, r.g2)))
            k == 2 && !moving && break
            for rr in p.rptr[e2]:(p.rptr[e2 + 1] - 1)
                b = p.blocks[p.rblock[rr]]
                l = p.rlevel[rr]
                G = Γ[p.rblock[rr]]
                loc = p.rloc[rr]
                for c in 1:b.K
                    zc = zs[loc + c - 1]
                    rhs = wk * state.z[zc] + gk * v[zc] + LCS[(k - 1) * m + loc + c - 1]
                    for a in 1:b.K
                        G[a, c, l] += p.zval[p.zoff[rr] + a] * rhs
                    end
                end
            end
        end
    end
    gradient = p.affine ? _brm_slap_affine_pullback(p, w) :
        _rk_slap_pullback(p.kernels, state.uA,
            _brm_slap_cotangents(p, state.values, w))[p.population]
    for (j, o) in enumerate(p.omega)
        k = findfirst(==(o), p.population)
        for (bj, b) in enumerate(p.blocks)
            j in b.omega || continue
            gradient[k] += dot(Γ[bj], state.dM[j][bj])
        end
    end
    prior, prior_gradient = _rk_slap_prior_gradient(p.kernels, state.uA)
    value + prior, gradient .+ prior_gradient[p.population]
end

LogDensityProblems.capabilities(::Type{<:BRMSparseLaplaceProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::BRMSparseLaplaceProblem) = length(p.population)
LogDensityProblems.logdensity(p::BRMSparseLaplaceProblem, φ::AbstractVector) =
    _brm_spl_value(p, _brm_spl_mode!(p, φ))
function LogDensityProblems.logdensity_and_gradient(p::BRMSparseLaplaceProblem,
        φ::AbstractVector)
    state = _brm_spl_mode!(p, φ)
    state.finite || return -Inf, fill(NaN, length(p.population))
    _brm_spl_value_and_gradient(p, state)
end

# Draws of the integrated effects through the factor at the modes: z = ẑ +
# (L′P)⁻¹ ε. With `conditional = K`, each component's draws are importance
# resampled from K such proposals by its own terms' exact density; ε's
# entries at a component's permuted positions drive only that component.
function brm_laplace_draws(p::BRMSparseLaplaceProblem, P::AbstractMatrix;
        rng=Random.default_rng(), conditional::Integer=0)
    size(P, 1) == length(p.population) || throw(DimensionMismatch(
        "population draws have $(size(P, 1)) rows, the problem has " *
        "$(length(p.population)) population coordinates"))
    conditional >= 0 || throw(ArgumentError(
        "brm_laplace_draws: conditional = $conditional proposals; pass 0 for " *
        "Laplace draws or a positive count for importance resampling"))
    n = length(p.integrated)
    nc = length(p.components)
    draws = Matrix{Float64}(undef, length(p.u), size(P, 2))
    newton_steps = Vector{Int}(undef, size(P, 2))
    importance_ess = Matrix{Float64}(undef, conditional > 0 ? nc : 0, size(P, 2))
    for k in axes(P, 2)
        state = _brm_spl_mode!(p, view(P, :, k))
        state.finite || _brm_nlme_error(
            "brm_laplace_draws: population draw $k has a nonfinite density")
        newton_steps[k] = state.iterations
        draws[:, k] .= state.u
        UP = state.F.UP
        if conditional == 0
            draws[p.integrated, k] .+= UP \ randn(rng, n)
            continue
        end
        perm_component = p.component[state.F.p]
        fc = zeros(nc)
        logw = zeros(nc, conditional)
        proposals = Matrix{Float64}(undef, n, conditional)
        for j in 1:conditional
            ε = randn(rng, n)
            proposals[:, j] .= state.z .+ UP \ ε
            _brm_spl_assemble!(fc, nothing, nothing, p, state.A, state.M,
                view(proposals, :, j); derivatives=false)
            logw[:, j] .= fc
            for q in 1:n
                logw[perm_component[q], j] += ε[q]^2 / 2
            end
        end
        for c in 1:nc
            lw = view(logw, c, :)
            wts = exp.(lw .- maximum(lw))
            importance_ess[c, k] = sum(wts)^2 / sum(abs2, wts)
            pick = min(searchsortedfirst(cumsum(wts), rand(rng) * sum(wts)), conditional)
            for i in p.components[c]
                draws[p.integrated[i], k] = proposals[i, pick]
            end
        end
    end
    diagnostics = conditional > 0 ? (; newton_steps, importance_ess) : (; newton_steps)
    (; draws, diagnostics)
end
