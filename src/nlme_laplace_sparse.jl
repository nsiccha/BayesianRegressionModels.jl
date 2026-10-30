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
#     ψ_t = A_t(φ) + Σ_k C_t,k(φ) z_k,ℓ,
#
# one level ℓ = ℓ_k(t) per integrated block k the row reads, its draws z_k,ℓ
# entering through C_t,k(φ) (a random-effect row's design times the block's
# diag(scales) L). C is probed from the model's own parameters at every
# population point (`_brm_spl_cmaps`), with its derivatives along the blocks'
# scale and correlation coordinates, so whatever maps a block's draws to the
# rows serves. The negative Hessian in z is the sparse
#
#     h = I + Cᵀ W C,    W = −ℓ″ (one 1 × 1 or 2 × 2 block per term),
#
# factored by CHOLMOD at every Newton step and at the mode. The value and
# gradient are the structured path's (its header), with each unit's h_i⁻¹
# replaced by the selected inverse of h on h's own pattern
# (SelectedInversion.jl). That holds every entry the gradient reads, because a term
# couples exactly the effects it reads.
#
# The mode search steps per connected component of h's graph: units that
# share no term (the units of an NLME view, nested groupings inside them) each
# take their own step length, stop and backtracking, as in the structured
# path; a crossed grouping joins the units it crosses into one component.

# Implemented by `BayesianRegressionModelsReactiveKernelsEnzymeExt`: the
# likelihood parameters at a packed point with their forward derivatives along
# the `omega` coordinates.
function _rk_slap_values_and_tangents end

# An integrated block: its draws' indices in the integrated vector z (levels ×
# margins), the `omega` entries (scales, correlations, persistence) that move
# how its draws enter the rows, and its prior precision: the identity
# (standardized draws) or, for an areal field's draws (one margin), the graph
# Laplacian of its `edges` (level pairs) plus λ 11ᵀ over each soft-sum
# component's levels.
struct _BRMSlapBlock
    declaration::Symbol
    group::Symbol
    K::Int
    z::Matrix{Int}
    omega::Vector{Int}
    edges::Matrix{Int}
    sums::Vector{Vector{Int}}
    λ::Vector{Float64}
end
_brm_spl_identity(b::_BRMSlapBlock) = isempty(b.sums) && size(b.edges, 2) == 0

"""
    BRMSparseLaplaceProblem

The Laplace of a regression-shaped RK model with every level of chosen
groupings' random-effect blocks, `cdar` walks and `car` fields integrated by
one sparse Laplace; see
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
    # `rblock[r]` at level `rlevel[r]` by entry `rentry[r]`. At a population
    # point its C over the level's K draws is `C[coff[r] .+ (1:K)]`
    # (`_brm_spl_cmaps`); `block_reads[j]` lists block j's reads.
    rptr::Vector{Int}
    rblock::Vector{Int}
    rlevel::Vector{Int}
    rentry::Vector{Int}
    coff::Vector{Int}
    block_reads::Vector{Vector{Int}}
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
    # The prior precision's sparse part on h's pattern (`prior_nz`, `prior_val`)
    # and its soft sums' rank-one part U Λ Uᵀ, U's columns the effects of
    # `wcols[c]`, Λ = diag(`wλ`), which stays out of the factor (Woodbury).
    prior_nz::Vector{Int}
    prior_val::Vector{Float64}
    wcols::Vector{Vector{Int}}
    wλ::Vector{Float64}
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
    length(p.integrated), " integrated effects of ", join(p.integrate, ", "),
    " in ", length(p.components), " components, ", length(p.family),
    " likelihood terms)")

# A copy (one per chain) gets its own factor: CHOLMOD's is a C object that a
# plain deepcopy would share with the original, whose finalizer frees it.
function Base.deepcopy_internal(p::BRMSparseLaplaceProblem, stackdict::IdDict)
    haskey(stackdict, p) && return stackdict[p]
    fields = map(fieldnames(typeof(p))) do f
        f === :factor ? Ref{Any}(nothing) : Base.deepcopy_internal(getfield(p, f), stackdict)
    end
    stackdict[p] = typeof(p)(fields...)
end

function _brm_slap_zero_integrated!(u, p::BRMSparseLaplaceProblem)
    u[p.integrated] .= 0.0
    u
end

# The grouping column a latent block is integrated by: a random-effect block's
# and a `cdar` walk's group, a `car` field's area column.
_brm_spl_record_group(r) = r.kind === :car ? r.area : r.group
const _BRM_SPL_LATENT = (:ranef, :stratified_ranef, :car, :cdar)

# The model's latent blocks over the named grouping columns.
function _brm_spl_records(backend::RKBRMI, integrate)
    records = [r for r in _rk_coordinate_records(backend.plan) if r.kind in _BRM_SPL_LATENT]
    available = unique(Symbol[_brm_spl_record_group(r) for r in records])
    for g in integrate
        g in available || _brm_nlme_error("brm_laplace_problem: integrate names " *
            "`$g`, which groups no random-effect block, `cdar` walk or `car` field of " *
            "the model; its groupings are " *
            "$(isempty(available) ? "none" : join(repr.(available), ", "))")
    end
    chosen = [r for r in records if _brm_spl_record_group(r) in integrate]
    for r in chosen
        r.kind === :stratified_ranef && _brm_nlme_error("brm_laplace_problem: the " *
            "random-effect block over `$(r.group)` is stratified, which the sparse " *
            "Laplace does not integrate yet")
        r.kind === :ranef && get(r, :mixing, nothing) !== nothing &&
            _brm_nlme_error("brm_laplace_problem: the random-effect block over " *
                "`$(r.group)` is Student-t, so its draws are not Gaussian and cannot be " *
                "integrated by a Laplace")
        r.kind === :car && r.separation !== nothing && _brm_nlme_error(
            "brm_laplace_problem: the `car($(r.area))` field is lowered with " *
            "`car_units`, whose draws the sparse Laplace does not integrate; build the " *
            "RKBRMI without `car_units` to integrate the field whole")
    end
    chosen
end

# A default-lowered `car` field's graph over its structured areas (its `phi`
# draws): the edges, and per connected component its areas and the precision
# λ = 1/(0.001 n)² of its soft sum-to-zero, from the plan's `car` term.
function _brm_spl_car_graph(backend::RKBRMI, r)
    terms = [t for predictor in _brm_slap_predictors(backend.plan) for t in predictor.terms
        if t.kind === :car && t.options.value === r.declaration]
    length(terms) == 1 || _brm_nlme_error("brm_laplace_problem: internal: " *
        "$(length(terms)) `car` terms define `$(r.declaration)`")
    options = only(terms).options
    data = rk_bound_data(backend)
    edges = permutedims(hcat(Int.(data[options.node1]), Int.(data[options.node2])))
    sums = [collect(start:(start + size - 1)) for (start, size) in options.components]
    (; S=sum(length, sums; init=0), edges, sums, λ=[1 / (0.001 * length(c))^2 for c in sums])
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
    function draws!(labels::AbstractMatrix{Symbol})
        z = Matrix{Int}(undef, size(labels))
        for l in axes(labels, 1), k in axes(labels, 2)
            push!(integrated, get(position, labels[l, k]) do
                _brm_nlme_error("brm_laplace_problem: internal: draw `$(labels[l, k])` " *
                    "is not packed")
            end)
            z[l, k] = length(integrated)
        end
        z
    end
    function block!(declaration, group, z, hyper; edges=zeros(Int, 2, 0),
            sums=Vector{Int}[], λ=Float64[])
        js = Int[]
        for d in hyper, i in owned(d)
            j = findfirst(==(i), omega)
            j === nothing && (push!(omega, i); j = length(omega))
            push!(js, j)
        end
        push!(blocks, _BRMSlapBlock(declaration, group, size(z, 2), z, unique(js), edges,
            sums, λ))
    end
    for r in records
        if r.kind === :ranef
            nlevels = length(_rk_ranef_record_levels(backend, r))
            z = draws!([Symbol(r.z, ".", l, ".", k) for l in 1:nlevels, k in 1:length(r.margins)])
            block!(_brm_nlme_ranef_declaration(r), r.group, z,
                ((isnothing(r.scales) ? (r.scale,) : r.scales)...,
                    (isnothing(r.L) ? () : (r.L,))...))
        elseif r.kind === :cdar
            # standardized innovations, groups × steps; scale and persistence
            eta = Symbol(r.declaration, ".", r.locals.eta)
            z = draws!([Symbol(eta, ".", g, ".", w) for g in 1:length(r.levels),
                w in 1:length(r.steps)])
            block!(r.declaration, r.group, z, (Symbol(r.declaration, ".", r.locals.sigma),
                Symbol(r.declaration, ".", r.locals.rho)))
        else
            # an areal field's structured values under its ICAR prior and, for
            # BYM2, each area's iid part
            graph = _brm_spl_car_graph(backend, r)
            hyper = (Symbol(r.declaration, ".", r.locals.sigma),
                (r.type === :bym2 ? (Symbol(r.declaration, ".", r.locals.rho),) : ())...)
            phi = Symbol(r.declaration, ".", r.locals.phi)
            block!(r.declaration, r.area, draws!(reshape([Symbol(phi, ".", l)
                for l in 1:graph.S], :, 1)), hyper; graph.edges, graph.sums, graph.λ)
            if r.type === :bym2
                theta = Symbol(r.declaration, ".", r.locals.theta)
                ntheta = count(n -> _brm_nlme_owned(n, theta), names)
                block!(theta, r.area, draws!(reshape([Symbol(theta, ".", l)
                    for l in 1:ntheta], :, 1)), hyper)
            end
        end
    end
    allunique(integrated) || _brm_nlme_error("brm_laplace_problem: internal: a draw " *
        "belongs to two integrated blocks")
    population = setdiff(1:length(names), integrated)
    # Latent draws left to the target make A bilinear in the population.
    sampled = any(r -> r.kind in _BRM_SPL_LATENT && _brm_spl_record_group(r) ∉ integrate,
        _rk_coordinate_records(backend.plan))
    families, ports = _brm_slap_ports(backend, responses)
    u0 = zeros(Float64, length(names))
    kernels = _rk_slap_kernels(backend, Tuple(ports), (), omega, u0, ad_backend)
    nrows = length(rk_bound_data(backend)[first(responses).response])
    t = _brm_slap_terms(backend, responses, families, ports, _rk_slap_values(kernels, u0),
        (1:nrows,))
    nψ = length(t.port)
    p = BRMSparseLaplaceProblem(backend, population, names[population], integrate,
        integrated, blocks, kernels, t.family, t.y, t.trials, t.constant, t.entry,
        fill(true, length(t.family)), t.port, t.index, t.link, t.fixed,
        Int[], Int[], Int[], Int[], Int[], Vector{Int}[], Int[], Int[], Int[], Int[], Int[],
        spzeros(0, 0), Int[], Int[], Vector{Int}[], Int[], Int[], Float64[], Vector{Int}[],
        Float64[], omega, false,
        u0[population], zeros(nψ), Int[], Int[], Float64[], copy(u0), Ref{Any}(nothing),
        Float64(tol), Int(maxiter), 0, 0)
    _brm_spl_design!(p, u0)
    sampled || _brm_slap_population_design!(p, u0)
    _brm_slap_verify!(p, u0, t.response, t.row)
    _brm_spl_verify_prior(p, u0)
    _brm_spl_pattern!(p)
    p
end

# Link-scale tangents of the entries, from their parameters' tangents.
function _brm_spl_dpsi(p::BRMSparseLaplaceProblem, values, tangents)
    dψ = zeros(length(p.port))
    for e in eachindex(dψ)
        k = p.port[e]
        k == 0 && continue
        θ, t = p.index[e] == 0 ? (values[k], tangents[k]) :
            (values[k][p.index[e]], tangents[k][p.index[e]])
        dψ[e] = _brm_lap_dlink(p.link[e], θ) * t
    end
    dψ
end

# At a packed point with the integrated draws at zero: the entries' values A,
# the parameter values behind them (`nothing` when A is affine and comes from
# its Jacobian), and every read's C (its entry's change per unit of each of
# the level's draws) with its derivative along each `omega` coordinate of its
# block. One parameter evaluation and pushforward along the block's own
# coordinates per block margin, with that margin's draw at one in every level:
# rows are affine in the draws and read one level of a block, and A does not
# move with the blocks' coordinates (both checked at construction), so C is
# that evaluation's difference from A and ∂C its tangent. This serves every
# block kind alike (random effects, areal fields, walks), whatever maps its
# draws to the rows.
function _brm_spl_cmaps(p::BRMSparseLaplaceProblem, uA)
    nψ = length(p.port)
    values = p.affine ? nothing : _rk_slap_values(p.kernels, uA)
    A = p.affine ? _brm_slap_affine_A(p, view(uA, p.population)) :
        _brm_slap_psi!(zeros(nψ), p, values)
    nc = isempty(p.coff) ? 0 : p.coff[end] + p.blocks[p.rblock[end]].K
    C = zeros(nc)
    dC = [zeros(nc) for _ in p.omega]
    ψ = zeros(nψ)
    for (j, b) in enumerate(p.blocks), k in 1:b.K
        isempty(p.block_reads[j]) && continue
        probe = copy(uA)
        probe[p.integrated[view(b.z, :, k)]] .= 1.0
        pv, pt = _rk_slap_values_and_tangents(p.kernels, probe, b.omega)
        _brm_slap_psi!(ψ, p, pv)
        dψ = [_brm_spl_dpsi(p, pv, t) for t in pt]
        for r in p.block_reads[j]
            e = p.rentry[r]
            C[p.coff[r] + k] = ψ[e] - A[e]
            for (i, o) in enumerate(b.omega)
                dC[o][p.coff[r] + k] = dψ[i][e]
            end
        end
    end
    C, dC, A, values
end

# The entries' reads: which level of which integrated block each entry reads,
# from unit steps of the draws at u0. Shifting draw k of every level of a
# block by one moves each entry by its C there; shifting level ℓ's by ℓ moves
# it by ℓ times that, which names the level. The structure is then checked at
# another point, and C must move with the blocks' own coordinates alone.
function _brm_spl_design!(p::BRMSparseLaplaceProblem, u0)
    nψ = length(p.port)
    base = _brm_slap_zero_integrated!(copy(u0), p)
    ψ0 = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, base))
    ψ1 = zeros(nψ); ψ2 = zeros(nψ)
    level = [zeros(Int, nψ) for _ in p.blocks]
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
                    "several levels of the integrated block `$(b.declaration)`, " *
                    "which the sparse Laplace does not integrate")
            level[j][e] == 0 || level[j][e] == l || _brm_nlme_error(
                "brm_laplace_problem: likelihood parameter $e reads two levels of the " *
                "integrated block `$(b.declaration)`")
            level[j][e] = l
        end
    end
    p.rptr = [1]; p.rblock = Int[]; p.rlevel = Int[]; p.rentry = Int[]; p.coff = Int[]
    p.block_reads = [Int[] for _ in p.blocks]
    nc = 0
    for e in 1:nψ
        for (j, b) in enumerate(p.blocks)
            l = level[j][e]
            l == 0 && continue
            push!(p.rblock, j); push!(p.rlevel, l); push!(p.rentry, e); push!(p.coff, nc)
            push!(p.block_reads[j], length(p.rblock))
            nc += b.K
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
    exact = _brm_slap_psi!(zeros(nψ), p, _rk_slap_values(p.kernels, u1))
    C1, _, A1, _ = _brm_spl_cmaps(p, uA)
    for e in 1:nψ
        predicted = A1[e] + _brm_spl_cdot(p, C1, e, z1)
        isapprox(predicted, exact[e]; rtol=1e-8, atol=1e-8) || _brm_nlme_error(
            "brm_laplace_problem: likelihood parameter $e is $(exact[e]) in the model " *
            "and $predicted as affine in the integrated random effects, so the sparse " *
            "Laplace does not apply")
    end
    # A must not move with the blocks' coordinates (they act through C).
    values, tangents = _rk_slap_values_and_tangents(p.kernels, uA, collect(eachindex(p.omega)))
    for t in tangents
        dA = _brm_spl_dpsi(p, values, t)
        all(d -> abs(d) <= 1e-9, dA) || _brm_nlme_error("brm_laplace_problem: the " *
            "likelihood parameters move with an integrated block's scale or correlation " *
            "where its draws are zero, so the sparse Laplace does not apply")
    end
    # The gradient differentiates C along the blocks' own coordinates only.
    others = setdiff(p.population, p.omega)
    if !isempty(others)
        u2 = copy(uA)
        u2[others] .+= 0.1 .* randn(rng, length(others))
        C2, _, _, _ = _brm_spl_cmaps(p, u2)
        isapprox(C2, C1; rtol=1e-9, atol=1e-9) || _brm_nlme_error(
            "brm_laplace_problem: the integrated effects enter the likelihood scaled by " *
            "population coordinates other than their own scales and correlations, " *
            "which the sparse Laplace does not differentiate")
    end
    p
end

# C_e z for an entry, over its reads.
function _brm_spl_cdot(p::BRMSparseLaplaceProblem, C, e, z)
    s = 0.0
    @inbounds for r in p.rptr[e]:(p.rptr[e + 1] - 1)
        b = p.blocks[p.rblock[r]]
        for k in 1:b.K
            s += C[p.coff[r] + k] * z[b.z[p.rlevel[r], k]]
        end
    end
    s
end

# zᵀ P z for the integrated draws' prior precision P.
function _brm_spl_prior_quad(p::BRMSparseLaplaceProblem, z)
    q = 0.0
    for b in p.blocks
        if _brm_spl_identity(b)
            q += sum(i -> z[i]^2, b.z; init=0.0)
            continue
        end
        for e in axes(b.edges, 2)
            q += (z[b.z[b.edges[1, e], 1]] - z[b.z[b.edges[2, e], 1]])^2
        end
        for (c, levels) in enumerate(b.sums)
            q += b.λ[c] * sum(l -> z[b.z[l, 1]], levels)^2
        end
    end
    q
end

# The integrated draws' prior must be the blocks' Gaussian, −½ zᵀPz, and
# depend on nothing else: the value adds that and RK's prior at z = 0.
function _brm_spl_verify_prior(p::BRMSparseLaplaceProblem, u0)
    rng = Random.Xoshiro(20261031)
    for _ in 1:2
        u = copy(u0)
        u[p.population] .+= 0.1 .* randn(rng, length(p.population))
        z = randn(rng, length(p.integrated))
        uA = _brm_slap_zero_integrated!(copy(u), p)
        u[p.integrated] .= z
        Δ = _rk_slap_prior(p.kernels, u) - _rk_slap_prior(p.kernels, uA)
        isapprox(Δ, -_brm_spl_prior_quad(p, z) / 2; rtol=1e-9, atol=1e-9) ||
            _brm_nlme_error("brm_laplace_problem: the integrated effects' prior is not " *
                "the Gaussian of their blocks in their packed coordinates (adaptive " *
                "centring or a constrained block), so the sparse Laplace does not apply")
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
    edges = [(b.z[b.edges[1, e], 1], b.z[b.edges[2, e], 1]) for b in p.blocks
        for e in axes(b.edges, 2)]
    for (i, j) in edges
        push!(I, i, j); push!(J, j, i)
    end
    pattern = sparse(I, J, 1.0, n, n)
    nonzeros(pattern) .= 0.0
    index(i, j) = (r = nzrange(pattern, j); r[searchsortedfirst(view(rowvals(pattern), r), i)])
    p.diagonal = [index(i, i) for i in 1:n]
    # The prior precision's sparse part: the identity, or a field's Laplacian.
    prior = Dict{Int,Float64}()
    for b in p.blocks
        _brm_spl_identity(b) || continue
        for i in b.z
            prior[index(i, i)] = get(prior, index(i, i), 0.0) + 1.0
        end
    end
    for (i, j) in edges, (a, c, v) in ((i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0))
        prior[index(a, c)] = get(prior, index(a, c), 0.0) + v
    end
    p.prior_nz = collect(keys(prior))
    p.prior_val = [prior[k] for k in p.prior_nz]
    p.wcols = [[b.z[l, 1] for l in levels] for b in p.blocks for levels in b.sums]
    p.wλ = [λ for b in p.blocks for λ in b.λ]
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
    join!(i, j) = (a = root(i); b = root(j); a == b || (parent[a] = b); nothing)
    for t in eachindex(p.family)
        zs = view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        for i in Iterators.drop(zs, 1)
            join!(first(zs), i)
        end
    end
    for (i, j) in edges
        join!(i, j)
    end
    for cols in p.wcols, i in Iterators.drop(cols, 1)
        join!(first(cols), i)
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
    read = falses(n)
    read[p.tz] .= true
    for cols in p.wcols
        any(i -> read[i], cols) || _brm_nlme_error("brm_laplace_problem: a component of " *
            "$(length(cols)) areas of a `car` field is read by no row, so the field's " *
            "level there is its prior's alone and h is singular; integrate a model whose " *
            "rows read every component")
    end
    p
end

# One term's C rows on its local effects (c1 for its first parameter, c2 for
# its second).
function _brm_spl_crows!(c1, c2, p::BRMSparseLaplaceProblem, C, t)
    fill!(c1, 0.0); fill!(c2, 0.0)
    e = p.entry[t]
    for (e2, c) in ((e, c1), (e + 1, c2))
        e2 == e || _brm_lap_width(p.family[t]) == 2 || break
        @inbounds for r in p.rptr[e2]:(p.rptr[e2 + 1] - 1)
            b = p.blocks[p.rblock[r]]
            for k in 1:b.K
                c[p.rloc[r] + k - 1] += C[p.coff[r] + k]
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
function _brm_spl_assemble!(fc, g, hnz, p::BRMSparseLaplaceProblem, A, C, z;
        derivatives::Bool, third::Bool=false, rows=nothing, crows=nothing)
    _brm_spl_prior!(fc, derivatives ? g : nothing, derivatives ? hnz : nothing, p, z)
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
        _brm_spl_crows!(c1, c2, p, C, t)
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

# The integrated draws' prior −½ zᵀPz into `fc` per component, its gradient
# −Pz into `g` and P's sparse part into h's nonzeros (the soft sums' rank-one
# part stays out of h, see `_brm_spl_woodbury`).
function _brm_spl_prior!(fc, g, hnz, p::BRMSparseLaplaceProblem, z)
    fill!(fc, 0.0)
    g === nothing || fill!(g, 0.0)
    for b in p.blocks
        if _brm_spl_identity(b)
            for i in b.z
                fc[p.component[i]] -= z[i]^2 / 2
                g === nothing || (g[i] -= z[i])
            end
            continue
        end
        for e in axes(b.edges, 2)
            i, j = b.z[b.edges[1, e], 1], b.z[b.edges[2, e], 1]
            d = z[i] - z[j]
            fc[p.component[i]] -= d^2 / 2
            if g !== nothing
                g[i] -= d
                g[j] += d
            end
        end
        for (c, levels) in enumerate(b.sums)
            total = sum(l -> z[b.z[l, 1]], levels)
            fc[p.component[b.z[first(levels), 1]]] -= b.λ[c] * total^2 / 2
            if g !== nothing
                for l in levels
                    g[b.z[l, 1]] -= b.λ[c] * total
                end
            end
        end
    end
    if hnz !== nothing
        nz = nonzeros(hnz)
        fill!(nz, 0.0)
        for (k, v) in zip(p.prior_nz, p.prior_val)
            nz[k] += v
        end
    end
    nothing
end

# With the factor F of A = h − U Λ Uᵀ (h without the soft sums' rank-one
# part): X = A⁻¹U and the Cholesky factor of G = Λ⁻¹ + UᵀX, or `nothing` when
# no block has soft sums. Then h⁻¹ = A⁻¹ − X G⁻¹ Xᵀ and log det h = log det A +
# log det G + Σ log λ, with k extra solves for k sums.
function _brm_spl_woodbury(p::BRMSparseLaplaceProblem, F)
    isempty(p.wcols) && return nothing
    U = zeros(length(p.integrated), length(p.wcols))
    for (c, cols) in enumerate(p.wcols)
        U[cols, c] .= 1.0
    end
    X = F \ U
    (; U, X, G=cholesky(Symmetric(Diagonal(1 ./ p.wλ) + U' * X)))
end

_brm_spl_solve(F, W, b) = (x = F \ b; W === nothing ? x : x .- W.X * (W.G \ (W.U' * x)))
_brm_spl_logdet(p::BRMSparseLaplaceProblem, F, W) =
    logdet(F) + (W === nothing ? 0.0 : logdet(W.G) + sum(log, p.wλ))

# A draw from N(0, h⁻¹): x ~ N(0, A⁻¹) through A's factor, then for the soft
# sums x − X G⁻¹ (Uᵀx + η), η ~ N(0, Λ⁻¹), whose covariance is
# A⁻¹ − X G⁻¹ Xᵀ = h⁻¹.
function _brm_spl_draw(p::BRMSparseLaplaceProblem, state, rng)
    x = state.F.UP \ randn(rng, length(p.integrated))
    W = state.W
    W === nothing && return x
    x .- W.X * (W.G \ (W.U' * x .+ randn(rng, length(p.wλ)) ./ sqrt.(p.wλ)))
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
    issuccess(F) && return _brm_spl_solve(F, _brm_spl_woodbury(p, F), g)
    μ = 1e-4
    for _ in 1:12
        F = _brm_spl_factor!(p, h; shift=μ)
        issuccess(F) && return _brm_spl_solve(F, _brm_spl_woodbury(p, F), g)
        μ *= 10
    end
    nothing
end

# The population point's A (link-scale parameters at z = 0), the parameter
# values behind it, and every read's C with its derivatives.
function _brm_spl_population(p::BRMSparseLaplaceProblem, φ::AbstractVector)
    length(φ) == length(p.population) || throw(DimensionMismatch(
        "the population point has length $(length(φ)), the problem has " *
        "$(length(p.population)) population coordinates"))
    uA = copy(p.u)
    uA[p.population] .= φ
    _brm_slap_zero_integrated!(uA, p)
    C, dC, A, values = _brm_spl_cmaps(p, uA)
    (; uA, values, A, C, dC)
end

# Every component's mode at population point φ, from the previous modes: one
# factor per Newton step, each component its own step length (backtracking on
# its own terms), stop (Newton decrement at most `tol`, or at its rounding
# floor) and failure. Then h at the modes and its factor.
function _brm_spl_mode!(p::BRMSparseLaplaceProblem, φ::AbstractVector)
    pop = _brm_spl_population(p, φ)
    if !(all(isfinite, pop.A) && all(isfinite, pop.C))
        p.nonfinite += 1
        return (; finite=false, value=-Inf, iterations=0)
    end
    n = length(p.integrated)
    nc = length(p.components)
    z = p.u[p.integrated]
    fc = zeros(nc); g = zeros(n); h = copy(p.pattern)
    _brm_spl_assemble!(fc, g, h, p, pop.A, pop.C, z; derivatives=true)
    if !(all(isfinite, fc) && all(isfinite, g))
        z .= 0.0
        _brm_spl_assemble!(fc, g, h, p, pop.A, pop.C, z; derivatives=true)
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
            _brm_spl_assemble!(ft, g, h, p, pop.A, pop.C, trial; derivatives=false)
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
        _brm_spl_assemble!(fc, g, h, p, pop.A, pop.C, z; derivatives=true)
    end
    F = _brm_spl_factor!(p, h)
    issuccess(F) || _brm_nlme_error("brm_laplace_problem: the integrated effects' " *
        "density has a stationary point that is not a maximum at this population point")
    p.u[p.integrated] .= z
    u = copy(pop.uA)
    u[p.integrated] .= z
    (; finite=true, pop..., z, u, h, F, W=_brm_spl_woodbury(p, F), iterations)
end

function _brm_spl_value(p::BRMSparseLaplaceProblem, state)
    state.finite || return -Inf
    fc = zeros(length(p.components))
    rest = _brm_spl_assemble!(fc, nothing, nothing, p, state.A, state.C, state.z;
        derivatives=false, third=true)
    sum(p.constant) + sum(fc) + rest + length(p.integrated) / 2 * log(2π) -
        _brm_spl_logdet(p, state.F, state.W) / 2 + _rk_slap_prior(p.kernels, state.uA)
end

# Value and gradient over the population coordinates: the structured path's
# (its header) with S = h⁻¹ on h's pattern for each unit's h_i⁻¹. Per term,
# with local C rows c1, c2: m = cᵀ S c, τ from m and ℓ‴, s = ½ Σ Cᵀ τ, v =
# h⁻¹ s, w = ℓ′ + ½ τ + ℓ″ C v; and per read and draw, (w ẑ + ℓ′ v + ℓ″ C S)
# times C's derivative along each scale or correlation coordinate of the block.
function _brm_spl_value_and_gradient(p::BRMSparseLaplaceProblem, state)
    n = length(p.integrated)
    nterms = length(p.family)
    fc = zeros(length(p.components))
    rows = Vector{_BRMLapRow}(undef, nterms)
    crows = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, nterms)
    rest = _brm_spl_assemble!(fc, nothing, nothing, p, state.A, state.C, state.z;
        derivatives=false, third=true, rows, crows)
    value = sum(p.constant) + sum(fc) + rest + n / 2 * log(2π) -
        _brm_spl_logdet(p, state.F, state.W) / 2
    Z = selinv(state.F; depermute=true).Z
    S = nonzeros(selinv_extract(Z, p.pattern))
    if state.W !== nothing
        # h⁻¹ = A⁻¹ − X G⁻¹ Xᵀ on the pattern.
        Y = (state.W.G \ state.W.X')'
        prow = rowvals(p.pattern)
        for j in axes(p.pattern, 2), k in nzrange(p.pattern, j)
            S[k] -= dot(view(Y, prow[k], :), view(state.W.X, j, :))
        end
    end
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
    v = _brm_spl_solve(state.F, state.W, s)
    w = zeros(length(p.port))
    gω = zeros(length(p.omega))
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
                loc = p.rloc[rr]
                for c in 1:b.K
                    zc = zs[loc + c - 1]
                    rhs = wk * state.z[zc] + gk * v[zc] + LCS[(k - 1) * m + loc + c - 1]
                    for o in b.omega
                        gω[o] += rhs * state.dC[o][p.coff[rr] + c]
                    end
                end
            end
        end
    end
    gradient = p.affine ? _brm_slap_affine_pullback(p, w) :
        _rk_slap_pullback(p.kernels, state.uA,
            _brm_slap_cotangents(p, state.values, w))[p.population]
    for (j, o) in enumerate(p.omega)
        gradient[findfirst(==(o), p.population)] += gω[j]
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
        if conditional == 0
            draws[p.integrated, k] .+= _brm_spl_draw(p, state, rng)
            continue
        end
        fc = zeros(nc)
        logw = zeros(nc, conditional)
        proposals = Matrix{Float64}(undef, n, conditional)
        for j in 1:conditional
            d = _brm_spl_draw(p, state, rng)
            proposals[:, j] .= state.z .+ d
            _brm_spl_assemble!(fc, nothing, nothing, p, state.A, state.C,
                view(proposals, :, j); derivatives=false)
            logw[:, j] .= fc
            # less the proposal's log density, −½ dᵀ h d per component (h is
            # block-diagonal over the components), up to its constant
            hd = state.h * d
            for i in 1:n
                logw[p.component[i], j] += d[i] * hd[i] / 2
            end
            for (c, cols) in enumerate(p.wcols)
                logw[p.component[first(cols)], j] += p.wλ[c] * sum(view(d, cols))^2 / 2
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
