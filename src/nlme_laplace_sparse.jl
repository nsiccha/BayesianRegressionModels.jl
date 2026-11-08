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
# factored exactly on bounded small components, or applied through iterative
# solves on large ones (nlme_laplace_matrixfree.jl). The exact value and
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
    # `:gmrf` for a native field, whose structured prior `p.gmrf` holds
    prior::Symbol
end
_BRMSlapBlock(declaration, group, K, z, omega, edges, sums, λ) =
    _BRMSlapBlock(declaration, group, K, z, omega, edges, sums, λ, :standard)
_brm_spl_identity(b::_BRMSlapBlock) = b.prior === :standard && isempty(b.sums) &&
    size(b.edges, 2) == 0

# The kernels run once per population point. Coordinate loops read concrete
# arrays, including when a declared field has millions of elements.
struct _BRMSPLGaussianPrior
    blocks::Vector{NamedTuple{(:z, :meanport, :scaleport),Tuple{Vector{Int},Int,Int}}}
    ports::Tuple{Vararg{Symbol}}
    read::Any
    pullback::Any
    mean::Vector{Float64}
    precision::Vector{Float64}
    values::Base.RefValue{Any}
end

"""
    BRMSparseLaplaceProblem

The Laplace of a regression-shaped RK model with every level of chosen
groupings' random-effect blocks, `cdar` walks and `car` fields integrated by
one sparse Laplace; see
[`brm_laplace_problem`](@ref)'s `integrate`. Same interface as
[`BRMStructuredLaplaceProblem`](@ref): LogDensityProblems order 1 over `names`
(packed positions `population`), [`brm_laplace_draws`](@ref), and the counters
`nonfinite`, `unconverged` and `folds`, plus `nonfinite_gradient` (gradients that
are not finite although the log density is; a warning names the first nonfinite
term) and `search` (the last evaluation's Newton steps, shifted steps per stage,
largest shift, refused full steps, final decrement and stop reason per component).
`integrate` names the integrated groupings,
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
    # Independent declared Gaussian priors, in their authored coordinates.
    gaussian::Union{Nothing,_BRMSPLGaussianPrior}
    # Native GMRF fields' structured priors (`nlme_laplace_gmrf.jl`).
    gmrf::Any
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
    # each row term's observation and row (`_brm_spl_term_keys`)
    response::Vector{Symbol}
    row::Vector{Int}
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
    # and in margin order). Pair locations are looked up in the sparse column;
    # no cell retains a width-squared map into the shared pattern.
    tptr::Vector{Int}
    tz::Vector{Int}
    rloc::Vector{Int}
    pattern_stats::NamedTuple
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
    # Step blocks inside the components (`_brm_spl_step_blocks!`): each
    # effect's block (`sblock`), each block's effects and component, the blocks
    # each term touches (`tsb[tsbptr[t]:tsbptr[t+1]-1]`) and each soft sum's.
    sblock::Vector{Int}
    sblocks::Vector{Vector{Int}}
    sblock_component::Vector{Int}
    tsbptr::Vector{Int}
    tsb::Vector{Int}
    wsblocks::Vector{Vector{Int}}
    # Factors by observation (`_brm_spl_cells!`): per observation its
    # cell query, compact AD and cells, each with its selection, its integrated
    # draws (`lz`, ascending) at selection positions `lpos`, its population
    # coordinates (`pidx`) at `ppos`, and its term (after the rows').
    cells::Vector{Any}
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
    # gradients that came out nonfinite at a finite log density
    # (`_brm_spl_nonfinite_gradient!`)
    nonfinite_gradient::Int
    # how often a component's full step failed and its blocks stepped on their
    # own, and how often a block Gauss–Seidel sweep followed
    blockwise::Int
    sweeps::Int
    # evaluations in which a component's followed mode needed a shifted step
    # after the continuation's first stage (`_brm_spl_mode!`), and the last
    # evaluation's mode-search record
    folds::Int
    search::Any
    # the `BRMLaplacePlan` it was built from (src/nlme_laplace_plan.jl), or
    # `nothing` for groupings named directly
    plan::Any
    algebra::Any
end

Base.show(io::IO, p::BRMSparseLaplaceProblem) = print(io,
    "BRMSparseLaplaceProblem(", length(p.population), " population coordinates, ",
    length(p.integrated), " integrated effects of ", join(p.integrate, ", "),
    " in ", length(p.components), " components, ", length(p.family),
    " likelihood terms; ", p.algebra === nothing ? "exact algebra" :
        "$(count(p.algebra.large)) matrix-free components, $(p.algebra.options.probes) fixed probes",
    ")")

# A copy (one per chain) gets its own factor: CHOLMOD's is a C object that a
# plain deepcopy would share with the original, whose finalizer frees it.
function Base.deepcopy_internal(p::BRMSparseLaplaceProblem, stackdict::IdDict)
    haskey(stackdict, p) && return stackdict[p]
    fields = map(fieldnames(typeof(p))) do f
        # A seeded backend (plan materialization) is shared by the RKBRMI
        # method (src/rk_backend.jl), on this field and through the kernels.
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
_brm_spl_record_group(r) = r.kind === :car ? r.area :
    r.kind in (:vector, :gmrf, :cell_gmrf) ? r.declaration : r.group
_brm_spl_record_declaration(r) =
    r.kind in (:ranef, :stratified_ranef) ? _brm_nlme_ranef_declaration(r) : r.declaration
const _BRM_SPL_LATENT = (:ranef, :stratified_ranef, :car, :cdar, :vector, :gmrf, :cell_gmrf)

# The model's latent blocks over the named grouping columns.
function _brm_spl_records(backend::RKBRMI, integrate; declarations=nothing)
    records = [r for r in _rk_coordinate_records(backend.plan) if r.kind in _BRM_SPL_LATENT]
    available = unique(Symbol[_brm_spl_record_group(r) for r in records])
    for g in integrate
        g in available || _brm_nlme_error("brm_laplace_problem: integrate names " *
            "`$g`, which groups no random-effect block, `cdar` walk or `car` field " *
            "and names no declared vector of the model; its groupings and declarations are " *
            "$(isempty(available) ? "none" : join(repr.(available), ", "))")
    end
    chosen = [r for r in records if _brm_spl_record_group(r) in integrate &&
        (declarations === nothing || _brm_spl_record_declaration(r) in declarations)]
    for r in chosen
        r.kind === :vector && get(r, :gaussian, nothing) === nothing &&
            _brm_nlme_error("brm_laplace_problem: `$(r.declaration)` does not have an " *
                "unconstrained independent Normal prior")
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

# Packed latent layout only; safe before any derivative preparation.
function _brm_spl_layout(backend, integrate; declarations=nothing)
    records = _brm_spl_records(backend, integrate; declarations)
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
            sums=Vector{Int}[], λ=Float64[], prior=:standard)
        js = Int[]
        for d in hyper
            # Known scales/correlations have no sampled coordinate declaration.
            d === nothing && continue
            for i in owned(d)
                j = findfirst(==(i), omega)
                j === nothing && (push!(omega, i); j = length(omega))
                push!(js, j)
            end
        end
        push!(blocks, _BRMSlapBlock(declaration, group, size(z, 2), z, unique(js), edges,
            sums, λ, prior))
    end
    for r in records
        if r.kind === :ranef
            nlevels = length(_rk_ranef_record_levels(backend, r))
            z = draws!([Symbol(r.z, ".", l, ".", k) for l in 1:nlevels, k in 1:length(r.margins)])
            block!(_brm_nlme_ranef_declaration(r), r.group, z,
                ((isnothing(r.scales) ? (r.scale,) : r.scales)...,
                    (isnothing(r.L) ? () : (r.L,))...))
        elseif r.kind === :vector
            labels = names[owned(r.declaration)]
            block!(r.declaration, r.declaration, draws!(reshape(labels, :, 1)), ())
        elseif r.kind === :gmrf
            # The field's state draws in state order; its SD scales the reads.
            field = Symbol(r.declaration, ".", r.locals.field)
            n = count(name -> _brm_nlme_owned(name, field), names)
            sigma = Symbol(r.declaration, ".", r.locals.sigma)
            block!(r.declaration, r.declaration,
                draws!(reshape([Symbol(field, ".", j) for j in 1:n], :, 1)),
                (haskey(position, sigma) ? sigma : nothing,); prior=:gmrf)
        elseif r.kind === :cell_gmrf
            # One field per cell, each with its own SD.
            field = Symbol(r.declaration, ".", r.locals.field)
            sigma = Symbol(r.declaration, ".", r.locals.sigma)
            ncells = length(rk_bound_data(backend)[r.labels])
            for cell in 1:ncells
                prefix = Symbol(field, ".", cell)
                n = count(name -> startswith(string(name), string(prefix, ".")), names)
                s = Symbol(sigma, ".", cell)
                block!(r.declaration, r.declaration,
                    draws!(reshape([Symbol(prefix, ".", j) for j in 1:n], :, 1)),
                    (haskey(position, s) ? s : nothing,); prior=:gmrf)
            end
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
    (; records, names, integrated, population, blocks, omega,
        partitioned=_rk_spl_partitioned(backend, integrated))
end

function _brm_sparse_laplace_problem(backend::RKBRMI, integrate; ad_backend, tol,
        maxiter, declarations=nothing, max_cell_width=128, screened=false,
        solver=:auto, matrixfree=MatrixFreeLaplace(), progress=nothing)
    max_cell_width = _brm_spl_width_limit(max_cell_width)
    integrate = Symbol[g for g in (integrate isa Symbol ? (integrate,) : integrate)]
    isempty(integrate) && throw(ArgumentError(
        "brm_laplace_problem: integrate names no grouping"))
    refusal = _brm_slap_response_refusal(backend; cells=true, values=true)
    refusal === nothing || _brm_nlme_error("brm_laplace_problem: integrate = " *
        "$(repr(Tuple(integrate))) needs every observation to be a regression row " *
        "of a closed-form family or an observation factor, but $refusal")
    responses = _brm_slap_responses(backend.plan)
    (; records, names, integrated, population, blocks, omega, partitioned) =
        _brm_spl_layout(backend, integrate; declarations)
    if !screened
        candidates = [(; declaration=_brm_spl_record_declaration(r),
            grouping=_brm_spl_record_group(r)) for r in records]
        _brm_spl_screen(backend, candidates, Set(t.declaration for t in candidates),
            max_cell_width)
    end
    # Latent draws left to the target make A bilinear in the population.
    # The plan may integrate only some declarations over the same grouping.
    selected = Set(_brm_spl_record_declaration(r) for r in records)
    sampled = any(r -> r.kind in _BRM_SPL_LATENT && _brm_spl_record_declaration(r) ∉ selected,
        _rk_coordinate_records(backend.plan))
    families, ports = _brm_slap_ports(backend, responses)
    u0 = zeros(Float64, length(names))
    kernels = _rk_slap_kernels(backend, Tuple(ports), (), omega, u0, ad_backend)
    nrows = isempty(responses) ? 0 :
        length(rk_bound_data(backend)[first(responses).response])
    t = _brm_slap_terms(backend, responses, families, ports, _rk_slap_values(kernels, u0),
        (1:nrows,))
    nψ = length(t.port)
    p = BRMSparseLaplaceProblem(backend, population, names[population], integrate,
        integrated, blocks, kernels, nothing, nothing, t.family, t.y, t.trials, t.constant, t.entry,
        fill(true, length(t.family)), t.port, t.index, t.link, t.fixed, t.response, t.row,
        Int[], Int[], Int[], Int[], Int[], Vector{Int}[], Int[], Int[], Int[], (;),
        spzeros(0, 0), Int[], Int[], Vector{Int}[], Int[], Int[], Float64[], Vector{Int}[],
        Float64[], Int[], Vector{Int}[], Int[], Int[], Int[], Vector{Int}[], Any[], omega,
        false,
        u0[population], zeros(nψ), Int[], Int[], Float64[], copy(u0), Ref{Any}(nothing),
        Float64(tol), Int(maxiter), 0, 0, 0, 0, 0, 0, nothing, nothing, nothing)
    p.gaussian = _brm_spl_gaussian_prior(p, records, u0)
    _brm_spl_gaussian_parameters!(p, u0)
    p.gmrf = _brm_spl_gmrf_prior(p, records, u0)
    _brm_spl_gmrf_parameters!(p, u0)
    _brm_spl_design!(p, u0)
    sampled || _brm_slap_population_design!(p, u0)
    cells = Set(_brm_slap_other_observations(backend.plan))
    _brm_slap_verify!(p, u0, t.response, t.row; cells)
    _brm_spl_verify_prior(p, u0)
    _brm_spl_cells!(p, u0, sort!(collect(cells)), ad_backend; partitioned, progress)
    _brm_spl_pattern!(p)
    _brm_spl_step_blocks!(p)
    _brm_spl_algebra!(p, solver, matrixfree)
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
        # A native field's entries read several of its levels: probe them
        # one level color at a time (`_brm_gmrf_level_colors`).
        colors = _brm_spl_gmrf_colors(p, j)
        for color in (colors === nothing ? (0,) : 1:maximum(colors))
            probe = copy(uA)
            for l in axes(b.z, 1)
                color == 0 || colors[l] == color || continue
                probe[p.integrated[b.z[l, k]]] = 1.0
            end
            pv, pt = _rk_slap_values_and_tangents(p.kernels, probe, b.omega)
            _brm_slap_psi!(ψ, p, pv)
            dψ = [_brm_spl_dpsi(p, pv, t) for t in pt]
            for r in p.block_reads[j]
                color == 0 || colors[p.rlevel[r]] == color || continue
                e = p.rentry[r]
                C[p.coff[r] + k] = ψ[e] - A[e]
                for (i, o) in enumerate(b.omega)
                    dC[o][p.coff[r] + k] = dψ[i][e]
                end
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
    # A native field's entries read several levels (a row's vertices), one
    # per level color: `fields[j][e]` lists them.
    fields = Dict{Int,Vector{Vector{Int}}}()
    for (j, b) in enumerate(p.blocks), k in 1:b.K
        colors = _brm_spl_gmrf_colors(p, j)
        colors === nothing && continue
        reads = [Int[] for _ in 1:nψ]
        for color in 1:maximum(colors)
            unit = copy(base); scaled = copy(base)
            for l in axes(b.z, 1)
                colors[l] == color || continue
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
                        "two field levels that a row's vertex reads cannot share; the " *
                        "sparse Laplace integrates a field read through its rows' vertices")
                push!(reads[e], l)
            end
        end
        fields[j] = reads
    end
    for (j, b) in enumerate(p.blocks), k in 1:b.K
        haskey(fields, j) && continue
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
            for l in (haskey(fields, j) ? sort(fields[j][e]) : (level[j][e],))
                l == 0 && continue
                push!(p.rblock, j); push!(p.rlevel, l); push!(p.rentry, e); push!(p.coff, nc)
                push!(p.block_reads[j], length(p.rblock))
                nc += b.K
            end
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

# zᵀ P z - 2 muᵀ P z for the integrated draws' prior difference from zero.
function _brm_spl_prior_quad(p::BRMSparseLaplaceProblem, z)
    q = 0.0
    for b in p.blocks
        if _brm_spl_identity(b)
            for i in b.z
                mu, precision = _brm_spl_gaussian_parameters(p, i)
                q += precision * z[i] * (z[i] - 2mu)
            end
            continue
        end
        for e in axes(b.edges, 2)
            q += (z[b.z[b.edges[1, e], 1]] - z[b.z[b.edges[2, e], 1]])^2
        end
        for (c, levels) in enumerate(b.sums)
            q += b.λ[c] * sum(l -> z[b.z[l, 1]], levels)^2
        end
    end
    q + _brm_spl_gmrf_quad(p, z)
end

# The integrated draws' prior must match their conditional Gaussian blocks:
# the value adds their difference from zero and RK's complete prior at z=0.
function _brm_spl_verify_prior(p::BRMSparseLaplaceProblem, u0)
    rng = Random.Xoshiro(20261031)
    for _ in 1:2
        u = copy(u0)
        u[p.population] .+= 0.1 .* randn(rng, length(p.population))
        z = randn(rng, length(p.integrated))
        uA = _brm_slap_zero_integrated!(copy(u), p)
        _brm_spl_gaussian_parameters!(p, uA)
        _brm_spl_gmrf_parameters!(p, uA)
        u[p.integrated] .= z
        Δ = _rk_slap_prior(p.kernels, u) - _rk_slap_prior(p.kernels, uA)
        isapprox(Δ, -_brm_spl_prior_quad(p, z) / 2; rtol=1e-9, atol=1e-9) ||
            _brm_nlme_error("brm_laplace_problem: the integrated effects' prior is not " *
                "the Gaussian of their blocks in their packed coordinates (adaptive " *
                "centring or a constrained block), so the sparse Laplace does not apply")
    end
    _brm_spl_gaussian_parameters!(p, u0)
    _brm_spl_gmrf_parameters!(p, u0)
    nothing
end

# Implemented by `BayesianRegressionModelsReactiveKernelsEnzymeExt`: one cell's
# compact AD (`_rk_spl_cell_ad`), its gradient and Hessian block over its
# selection inside a Newton pass (`_rk_spl_cell_newton`, `_rk_spl_cell_block`),
# and additive derivative contractions inside a shared pass (`_rk_spl_cell_pass`,
# `_rk_spl_cell_accumulate`). The correction accumulates third-order products,
# then a gradient and one HVP along the solved implicit direction.
# The widest Enzyme batch one cell's Hessian block compiles; wider cells take
# several batches (`_rk_spl_cell_ad`). On a 168-coordinate nonlinear cell the
# first compile took 92, 107, 174 and 377 s at widths 2, 4, 8 and 16, with a
# warm cost per direction of about 0.13 ms from width 4 on.
const _BRM_SPL_CELL_HVP_WIDTH = 4
function _rk_spl_cell_ad end
function _rk_spl_cell_whole end
function _rk_spl_cell_newton end
function _rk_spl_cell_block end
function _rk_spl_cell_accumulate end
function _rk_spl_cell_footprint end
_rk_spl_cell_footprint(q::_BRMNLMECellQuery; kwargs...) =
    _rk_spl_cell_footprint(_brm_nlme_cell_query(q); kwargs...)
function _rk_spl_cell_reads end
function _rk_spl_cell_batch end
function _rk_spl_partitioned end
function _rk_spl_cell_pass end
function _rk_spl_cell_value end

# Observation factors, including model-scope value observations: each cell's
# selection (its structural packed-coordinate footprint; a cell with fewer
# reads than its observation's widest is padded with coordinates it does not
# read, which give zero rows), its integrated draws and population
# coordinates, and one compact AD per observation. Each cell's value must be
# its pointwise density.
# A cell's HVP directions in the group's shared buffers: direction `k` is the
# unit at the cell's `k`-th integrated position; the buffers are zero between
# cells.
function _brm_spl_cell_directions!(dirs, lpos, v)
    for (k, pos) in enumerate(lpos)
        dirs[k][pos] = v
    end
    dirs
end

function _brm_spl_cells!(p::BRMSparseLaplaceProblem, u0, names, ad_backend; partitioned,
        progress=nothing)
    isempty(names) && return p
    n = length(p.integrated)
    zof = zeros(Int, length(u0)); zof[p.integrated] .= 1:n
    pof = zeros(Int, length(u0)); pof[p.population] .= eachindex(p.population)
    pointwise = _rk_slap_pointwise(p.kernels, u0)
    term = length(p.family)
    for name in names
        expected = pointwise[name]
        isempty(expected) && continue
        term = _brm_with_progress(progress;
                description="Observation `$name`, $(length(expected)) cells") do node
            _brm_spl_observation_cells!(p, u0, name, expected, term, zof, pof, ad_backend, node;
                partitioned)
        end
    end
    p
end

# One observation's factors. Returns the last term number it took.
function _brm_spl_observation_cells!(p, u0, name, expected, term, zof, pof, ad_backend, progress;
        partitioned)
    # Only the compact route evaluates this query, and only with nothing computed whole.
    query = _rk_nlme_cell_query(p.backend, (name,); partitioned, u0, locality=:ignore)
    factors = _rk_spl_factor_queries(p.backend, name; partitioned, u0)
    prepared = _brm_spl_prepare_factors(p.backend, name, u0, length(expected),
        p.integrated, p.population; query=factors, whole=_rk_spl_cell_whole(query))
    if prepared !== nothing
        prepared = _brm_spl_factor_operators(prepared, u0, ad_backend)
        cells = _rk_spl_cell_batch(factors.factors, u0) do batch
            map(prepared.cells) do c
                value = _brm_spl_factor_values(prepared, u0, c; context=batch).value
                isapprox(value, sum(expected[c.i]); rtol=1e-9, atol=1e-9) ||
                    _brm_nlme_error("sum-factor density does not match cell $(c.i) of `$name`")
                term += 1
                merge(c, (; term))
            end
        end
        prepared = merge(prepared, (; cells))
        @debug "sparse Laplace sum-factor group" name cells=length(cells) maxrss_gib=Sys.maxrss() / 2^30
        push!(p.cells, _BRMSumCellGroup(name, query, cells, prepared))
        return term
    end
    # An ordinary cell keeps its original graph: the factored outer can
    # conservatively widen captured arrays even when it has no sums.
    footprint = _rk_spl_cell_footprint(query; shared=false)
    localdata = _rk_spl_cell_batch(query, u0) do batch
        map(eachindex(expected)) do i
            reads = _rk_spl_cell_reads(footprint, u0, i; context=batch)
            value = _rk_nlme_cell_value(query, batch, i)
            (; reads, value)
        end
    end
    # (cells that read no coordinate still get a one-coordinate selection,
    # so they are terms like the others; a pad is always partitioned, and
    # their population derivatives come from the accumulated prefix pullback)
    m = max(1, maximum(d -> length(d.reads), localdata; init=0))
    cells = map(eachindex(expected)) do i
        r = localdata[i].reads
        value = localdata[i].value
        isapprox(value, sum(expected[i]); rtol=1e-9, atol=1e-9) || _brm_nlme_error(
            "brm_laplace_problem: internal: cell $i of `$name` has value $value, " *
            "its pointwise density $(sum(expected[i]))")
        pad = Int[]
        for j in p.integrated
            length(pad) == m - length(r) && break
            insorted(j, r) || push!(pad, j)
        end
        sel = sort!(vcat(r, pad))
        z = [(zof[x], k) for (k, x) in enumerate(sel) if zof[x] > 0 && insorted(x, r)]
        sort!(z)
        ps = [(pof[x], k) for (k, x) in enumerate(sel) if pof[x] > 0 && insorted(x, r)]
        term += 1
        (; i, sel, lz=first.(z), lpos=last.(z), pidx=first.(ps), ppos=last.(ps), term)
    end
    # each cell's HVP directions: the units at its integrated draws (K of
    # them, zero past a cell's own), whose results are the Hessian's
    # columns there. The group holds one zero buffer per direction and the
    # evaluation sets a cell's units in place (`_brm_spl_cell_directions!`):
    # a buffer per cell is `K × m` floats each, tens of gigabytes over a
    # Gemeinde-level observation's cells.
    K = max(1, maximum(c -> length(c.lz), cells))
    dirs = ntuple(_ -> zeros(m), K)
    ad = _rk_spl_cell_ad(p.backend, (name,), copy(u0), first(cells).sel, ad_backend, K;
        partitioned)
    @debug "sparse Laplace cell group" name cells=length(cells) m K maxrss_gib=Sys.maxrss() / 2^30
    _brm_spl_compile_cell_block(progress, ad, dirs, u0, first(cells); m, K)
    push!(p.cells, (; name, query, ad, m, dirs, cells))
    term
end

# A cell's gradient and Hessian block, inside a Newton pass.
function _brm_spl_cell_hessian(group, u, c, batch)
    _brm_spl_cell_directions!(group.dirs, c.lpos, 1.0)
    try
        _rk_spl_cell_block(group.ad, group.dirs, u, c.sel, c.i; context=batch, used=length(c.lz))
    finally
        _brm_spl_cell_directions!(group.dirs, c.lpos, 0.0)
    end
end

# Enzyme compiles a cell's Hessian block on its first use. With a progress
# node, that happens here, under a node naming the block's size and timing it,
# rather than unnamed inside the first assembly. Its directions are all zero.
_brm_spl_compile_cell_block(::Nothing, ad, dirs, u, cell; m, K) = nothing
function _brm_spl_compile_cell_block(progress, ad, dirs, u, cell; m, K)
    _brm_with_progress(progress;
            description="Compiling a Hessian block: $m coordinates, $K integrated") do _
        _rk_spl_cell_newton(ad, u) do batch
            _rk_spl_cell_block(ad, dirs, u, cell.sel, cell.i; context=batch, used=1)
        end
    end
    nothing
end

_brm_spl_nterms(p::BRMSparseLaplaceProblem) = length(p.family) +
    sum(g -> length(g.cells), p.cells; init=0)

# Importance diagnostics perturb only local coordinates. Keep one active batch
# per observation across the diagnostic, while compact active values supply
# each perturbed cell point. Distinct observation families own distinct caches.
function _brm_spl_cell_batches(f, p, u)
    contexts = Dict{Symbol,Any}()
    function enter(k)
        k > length(p.cells) && return f(contexts)
        group = p.cells[k]
        ad = group isa _BRMSumCellGroup ? first(group.cells).outer : group.ad
        result, _ = _rk_spl_cell_pass(ad, u) do batch
            contexts[group.name] = batch
            enter(k + 1)
        end
        result
    end
    enter(1)
end

# Footprints are sorted unique effect indices. Deduplicate them before visiting
# pairs, then union each sparse column directly. Construction storage is the
# distinct input footprints plus the retained pattern, never duplicate triplets.
function _brm_spl_pattern(footprints, n, edges)
    distinct = Set{Vector{Int}}()
    input_footprints = input_indices = 0
    for zs in footprints
        input_footprints += 1
        input_indices += length(zs)
        push!(distinct, collect(zs))
    end
    columns = [Set{Int}((j,)) for j in 1:n]
    pair_visits = 0
    for zs in distinct
        pair_visits += length(zs)^2
        for j in zs
            union!(columns[j], zs)
        end
    end
    for (i, j) in edges
        push!(columns[j], i)
        push!(columns[i], j)
    end
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    for j in 1:n
        colptr[j + 1] = colptr[j] + length(columns[j])
    end
    rowval = Vector{Int}(undef, last(colptr) - 1)
    for j in 1:n
        copyto!(rowval, colptr[j], sort!(collect(columns[j])), 1, length(columns[j]))
    end
    pattern = SparseMatrixCSC(n, n, colptr, rowval, zeros(length(rowval)))
    stats = (; input_footprints, input_indices, distinct_footprints=length(distinct),
        distinct_indices=sum(length, distinct; init=0), pair_visits,
        retained_nnz=length(nonzeros(pattern)))
    (; pattern, stats)
end

@inline function _brm_spl_nzindex(pattern, i, j)
    span = nzrange(pattern, j)
    first(span) - 1 + searchsortedfirst(view(rowvals(pattern), span), i)
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
    for group in p.cells, c in group.cells
        append!(p.tz, c.lz)
        push!(p.tptr, length(p.tz) + 1)
    end
    nterms = _brm_spl_nterms(p)
    sumterms = Set(c.term for group in p.cells if group isa _BRMSumCellGroup for c in group.cells)
    footprints = Iterators.flatten(((view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        for t in 1:nterms if t ∉ sumterms),
        (term.lz for group in p.cells if group isa _BRMSumCellGroup
            for c in group.cells for reduction in c.terms for term in reduction)))
    edges = [(b.z[b.edges[1, e], 1], b.z[b.edges[2, e], 1]) for b in p.blocks
        for e in axes(b.edges, 2)]
    # A native field's prior couples its draws through Q's pattern.
    couplings = p.gmrf === nothing ? edges : vcat(edges, _brm_gmrf_pattern_pairs(p.gmrf))
    built = _brm_spl_pattern(footprints, n, couplings)
    pattern = built.pattern
    p.pattern_stats = built.stats
    index(i, j) = _brm_spl_nzindex(pattern, i, j)
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
    p.gmrf === nothing || _brm_gmrf_pattern_positions!(p.gmrf, pattern)
    p.wcols = [[b.z[l, 1] for l in levels] for b in p.blocks for levels in b.sums]
    p.wλ = [λ for b in p.blocks for λ in b.λ]
    p.pattern = pattern
    # Components by union-find over the terms' effects.
    parent = collect(1:n)
    root(i) = (while parent[i] != i; parent[i] = parent[parent[i]]; i = parent[i]; end; i)
    join!(i, j) = (a = root(i); b = root(j); a == b || (parent[a] = b); nothing)
    for t in 1:nterms
        zs = view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        for i in Iterators.drop(zs, 1)
            join!(first(zs), i)
        end
    end
    for (i, j) in couplings
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
        for t in 1:nterms]
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
# when given, keep each term's row and C rows (`crows = (c1, c2)`, flat
# vectors indexed like `p.tz`), and `tv` each term's density
# with its constant (rows, then cells; with `third`).
#
# `weight` multiplies the likelihood (every term, cell and sum) and leaves the
# integrated effects' prior as it is: the mode search's continuation density
# prior + weight × likelihood (`_brm_spl_mode!`). At weight 0 the likelihood is
# not evaluated at all, so a reader that overflows cannot turn the prior's
# value into 0 × Inf.
function _brm_spl_assemble!(fc, g, hnz, p::BRMSparseLaplaceProblem, A, C, z;
        derivatives::Bool, third::Bool=false, rows=nothing, crows=nothing, fb=nothing,
        uA=nothing, tv=nothing, weight::Float64=1.0)
    correction = p.algebra === nothing ? nothing : zeros(length(fc))
    _brm_spl_prior!(fc, derivatives ? g : nothing, derivatives ? hnz : nothing,
        p, z, fb; correction)
    rest = 0.0
    mmax = maximum(t -> p.tptr[t + 1] - p.tptr[t], eachindex(p.family); init=0)
    c1 = zeros(mmax); c2 = zeros(mmax)
    nz = derivatives ? nonzeros(hnz) : Float64[]
    @inbounds for t in 1:(weight == 0 ? 0 : length(p.family))
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
        tv === nothing || (tv[t] = r.v + p.constant[t])
        if crows !== nothing
            for i in 1:m
                crows[1][p.tptr[t] + i - 1] = c1[i]
                crows[2][p.tptr[t] + i - 1] = c2[i]
            end
        end
        if comp == 0
            rest += r.v
            continue
        end
        _brm_spl_add_value!(fc, correction, comp, weight * r.v)
        if fb !== nothing
            for k in p.tsbptr[t]:(p.tsbptr[t + 1] - 1)
                fb[p.tsb[k]] += weight * r.v
            end
        end
        derivatives || continue
        for i in 1:m
            g[zs[i]] += weight * (c1[i] * r.g1 + (moving ? c2[i] * r.g2 : 0.0))
        end
        for j in 1:m, i in 1:m
            v = r.h11 * c1[i] * c1[j]
            moving && (v += r.h12 * (c1[i] * c2[j] + c2[i] * c1[j]) + r.h22 * c2[i] * c2[j])
            nz[_brm_spl_nzindex(p.pattern, zs[i], zs[j])] -= weight * v
        end
    end
    isempty(p.cells) && return rest
    # The `@plate` cells: each its value at the packed point, and with
    # `derivatives` its gradient and Hessian block over its selection.
    u = copy(uA)
    u[p.integrated] .= z
    for group in (weight == 0 ? empty(p.cells) : p.cells)
        group isa _BRMSumCellGroup && continue
        # A move to reverse over forward restarts the Newton pass (`_rk_spl_cell_newton`).
        # The sweep resumes at the cell that raised, which has changed nothing yet: its
        # block comes before anything it adds.
        next = 1
        _rk_spl_cell_batch(group.query, u) do values
            sweep = batch -> while next <= length(group.cells)
                c = group.cells[next]
                comp = p.term_component[c.term]
                if comp == 0 && !third
                    next += 1
                    continue
                end
                value = _rk_nlme_cell_value(group.query, values, c.i)
                tv === nothing || (tv[c.term] = value)
                if comp == 0
                    rest += value
                    next += 1
                    continue
                end
                block = derivatives ? _brm_spl_cell_hessian(group, u, c, batch) : nothing
                _brm_spl_add_value!(fc, correction, comp, weight * value)
                if fb !== nothing
                    for k in p.tsbptr[c.term]:(p.tsbptr[c.term + 1] - 1)
                        fb[p.tsb[k]] += weight * value
                    end
                end
                if block !== nothing
                    gsel, columns = block
                    m = length(c.lz)
                    for a in 1:m
                        g[c.lz[a]] += weight * gsel[c.lpos[a]]
                        for b in 1:m
                            nz[_brm_spl_nzindex(p.pattern, c.lz[a], c.lz[b])] -=
                                weight * columns[b][c.lpos[a]]
                        end
                    end
                end
                next += 1
            end
            derivatives ? _rk_spl_cell_newton(sweep, group.ad, u) : sweep(nothing)
        end
    end
    rest + _brm_spl_assemble_sums!(fc, g, hnz, p, u; derivatives, third, fb, tv, correction,
        weight)
end

# Step blocks: inside a component, the levels of the coarsest integrated
# grouping its rows read (a Kreis with its Gemeinden and their field values),
# so each takes its own step length (`per-unit-steps`). A term's key is the
# level it reads of its coarsest block (fewest levels); an effect joins the
# key most of its terms have; an effect no term reads joins a neighbour's
# along the field's edges, or stays alone. A component whose effects share one
# key is one block, as a unit of a nested model is.
function _brm_spl_step_blocks!(p::BRMSparseLaplaceProblem)
    n = length(p.integrated)
    nterms = _brm_spl_nterms(p)
    key = Vector{Tuple{Int,Int}}(undef, nterms)
    # each effect's block and level, for a cell's key
    where_ = Vector{Tuple{Int,Int}}(undef, n)
    for (j, b) in enumerate(p.blocks), l in axes(b.z, 1), k in axes(b.z, 2)
        where_[b.z[l, k]] = (j, l)
    end
    for group in p.cells, c in group.cells
        best = (0, 0); size_best = typemax(Int)
        for i in c.lz
            j, l = where_[i]
            m = size(p.blocks[j].z, 1)
            if m < size_best || (m == size_best && (j, l) < best)
                best = (j, l); size_best = m
            end
        end
        key[c.term] = best
    end
    for t in eachindex(p.family)
        best = (0, 0); size_best = typemax(Int)
        e = p.entry[t]
        for e2 in e:(e + _brm_lap_width(p.family[t]) - 1), r in p.rptr[e2]:(p.rptr[e2 + 1] - 1)
            m = size(p.blocks[p.rblock[r]].z, 1)
            if m < size_best || (m == size_best && (p.rblock[r], p.rlevel[r]) < best)
                best = (p.rblock[r], p.rlevel[r]); size_best = m
            end
        end
        key[t] = best
    end
    counts = [Dict{Tuple{Int,Int},Int}() for _ in 1:n]
    for t in 1:nterms, i in view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        counts[i][key[t]] = get(counts[i], key[t], 0) + 1
    end
    effect_key = Vector{Union{Nothing,Tuple{Int,Int}}}(nothing, n)
    for i in 1:n
        isempty(counts[i]) && continue
        effect_key[i] = first(sort!(collect(counts[i]); by=kv -> (-kv[2], kv[1])))[1]
    end
    # A field's prior couples all its draws: it steps as one block.
    for (j, b) in enumerate(p.blocks)
        b.prior === :gmrf || continue
        for i in b.z
            effect_key[i] = (j, 0)
        end
    end
    neighbours = [Int[] for _ in 1:n]
    for b in p.blocks, e in axes(b.edges, 2)
        i, j = b.z[b.edges[1, e], 1], b.z[b.edges[2, e], 1]
        push!(neighbours[i], j); push!(neighbours[j], i)
    end
    queue = [i for i in 1:n if effect_key[i] !== nothing]
    next = 1
    while next <= length(queue)
        i = queue[next]
        next += 1
        for j in neighbours[i]
            effect_key[j] === nothing || continue
            effect_key[j] = effect_key[i]
            push!(queue, j)
        end
    end
    ids = Dict{Tuple{Int,Any},Int}()
    p.sblock = [get!(ids, (p.component[i], something(effect_key[i], (-i, 0))),
        length(ids) + 1) for i in 1:n]
    p.sblocks = [Int[] for _ in 1:length(ids)]
    p.sblock_component = zeros(Int, length(ids))
    for i in 1:n
        push!(p.sblocks[p.sblock[i]], i)
        p.sblock_component[p.sblock[i]] = p.component[i]
    end
    p.tsbptr = [1]; p.tsb = Int[]
    for t in 1:nterms
        append!(p.tsb, unique(p.sblock[i] for i in view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))))
        push!(p.tsbptr, length(p.tsb) + 1)
    end
    p.wsblocks = [unique(p.sblock[i] for i in cols) for cols in p.wcols]
    p
end

# The integrated draws' prior −½ zᵀPz into `fc` per component, its gradient
# −Pz into `g` and P's sparse part into h's nonzeros (the soft sums' rank-one
# part stays out of h, see `_brm_spl_woodbury`).
# A large component's density can dwarf the final Newton gain. Compensating
# its accumulation preserves that gain for the operator line search without
# relaxing its acceptance or convergence tolerances.
@inline function _brm_spl_add_value!(fc, ::Nothing, component, value)
    fc[component] += value
    nothing
end
@inline function _brm_spl_add_value!(fc, correction::AbstractVector, component, value)
    increment = value - correction[component]
    total = fc[component] + increment
    correction[component] = (total - fc[component]) - increment
    fc[component] = total
    nothing
end

function _brm_spl_prior!(fc, g, hnz, p::BRMSparseLaplaceProblem, z, fb=nothing;
        correction=p.algebra === nothing ? nothing : zeros(length(fc)))
    fill!(fc, 0.0)
    correction === nothing || fill!(correction, 0.0)
    g === nothing || fill!(g, 0.0)
    fb === nothing || fill!(fb, 0.0)
    w = 0
    for b in p.blocks
        if _brm_spl_identity(b)
            for i in b.z
                mu, precision = _brm_spl_gaussian_parameters(p, i)
                delta = precision * z[i] * (mu - z[i] / 2)
                _brm_spl_add_value!(fc, correction, p.component[i], delta)
                fb === nothing || (fb[p.sblock[i]] += delta)
                g === nothing || (g[i] += precision * (mu - z[i]))
            end
            continue
        end
        for e in axes(b.edges, 2)
            i, j = b.z[b.edges[1, e], 1], b.z[b.edges[2, e], 1]
            d = z[i] - z[j]
            _brm_spl_add_value!(fc, correction, p.component[i], -d^2 / 2)
            if fb !== nothing
                fb[p.sblock[i]] -= d^2 / 2
                p.sblock[j] == p.sblock[i] || (fb[p.sblock[j]] -= d^2 / 2)
            end
            if g !== nothing
                g[i] -= d
                g[j] += d
            end
        end
        for (c, levels) in enumerate(b.sums)
            w += 1
            total = sum(l -> z[b.z[l, 1]], levels)
            _brm_spl_add_value!(fc, correction, p.component[b.z[first(levels), 1]],
                -b.λ[c] * total^2 / 2)
            if fb !== nothing
                for k in p.wsblocks[w]
                    fb[k] -= b.λ[c] * total^2 / 2
                end
            end
            if g !== nothing
                for l in levels
                    g[b.z[l, 1]] -= b.λ[c] * total
                end
            end
        end
    end
    _brm_spl_gmrf_prior!(fc, g, p, z, fb, correction)
    if hnz !== nothing
        nz = nonzeros(hnz)
        fill!(nz, 0.0)
        for (k, v) in zip(p.prior_nz, p.prior_val)
            nz[k] += v
        end
        _brm_spl_gmrf_precision!(nz, p)
        if p.gaussian !== nothing
            for b in p.gaussian.blocks, i in b.z
                nz[p.diagonal[i]] += p.gaussian.precision[i] - 1
            end
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

# Each component's share of log det h, from the factor's diagonal in its
# permuted order (h is block diagonal over the components). CHOLMOD's
# diagonal is L's for an LLᵀ factor and D's for an LDLᵀ one.
function _brm_spl_component_logdets(p::BRMSparseLaplaceProblem, F)
    d = diag(F)
    out = zeros(length(p.components))
    for (j, i) in enumerate(F.p)
        out[p.component[i]] += log(abs(d[j]))
    end
    total = sum(out)
    abs(total) < 1e-12 || isapprox(2total, logdet(F); rtol=1e-10) ? 2 .* out : out
end

# The blocks of h⁻¹ on sets of integrated draws (a level's margins), from the
# selected inverse where every pair of a set is on h's pattern (terms that
# read a level read all its margins), else by solving for the set's columns.
function _brm_spl_inverse_blocks(p::BRMSparseLaplaceProblem, state, sets)
    state.F isa _BRMOperatorFactor && return _brm_operator_inverse_blocks(state.F, sets)
    state.F isa _BRMAugmentedFactor && return _brm_spl_sum_inverse_blocks(state.F, sets)
    state.W === nothing || _brm_nlme_error("brm_laplace_problem: internal: " *
        "covariance blocks of a problem with soft sums are not built")
    on_pattern(i, j) = any(k -> rowvals(p.pattern)[k] == i, nzrange(p.pattern, j))
    selected = [all(on_pattern(i, j) for i in set, j in set) for set in sets]
    pairs = [(i, j) for (set, ok) in zip(sets, selected) if ok for i in set for j in set]
    Zs = isempty(pairs) ? nothing : selinv_extract(selinv(state.F; depermute=true).Z,
        sparse(first.(pairs), last.(pairs), 1.0, length(p.integrated),
            length(p.integrated)))
    map(zip(sets, selected)) do (set, ok)
        ok && return [Zs[i, j] for i in set, j in set]
        E = zeros(length(p.integrated), length(set))
        for (c, j) in enumerate(set)
            E[j, c] = 1.0
        end
        X = state.F \ E
        Matrix(Symmetric(X[set, :]))
    end
end

# Each term's owner unit, as `owner(response, row)` names it for a row term
# and `owner(observation, cell)` for a cell (rows first, then cells, in term
# order).
_brm_spl_term_owners(owner, p::BRMSparseLaplaceProblem) = vcat(
    [owner(p.response[t], p.row[t]) for t in eachindex(p.family)],
    [owner(group.name, c.i) for group in p.cells for c in group.cells])

# Each unit's share of the Laplace value at a mode `state`, with no population
# prior: its terms' densities and, per component its terms read, the
# integrated draws' prior less half that component's log determinant. Summed
# over the units it is the problem's value less its population prior and
# n/2 log 2π. `owners[t]` is term t's unit; a component's terms share one.
function _brm_spl_unit_values(p::BRMSparseLaplaceProblem, state, owners, nunits)
    state.W === nothing || _brm_nlme_error("brm_laplace_problem: internal: unit " *
        "values of a problem with soft sums are not built")
    nterms = _brm_spl_nterms(p)
    length(owners) == nterms || throw(DimensionMismatch(
        "$(length(owners)) owners for $nterms terms"))
    tv = zeros(nterms)
    fc = zeros(length(p.components))
    _brm_spl_assemble!(fc, nothing, nothing, p, state.A, state.C, state.z;
        derivatives=false, third=true, uA=state.uA, tv)
    _brm_spl_prior!(fc, nothing, nothing, p, state.z)
    logdets = _brm_spl_component_logdets(p, state.F)
    values = zeros(nunits)
    cowner = zeros(Int, length(p.components))
    for t in 1:nterms
        values[owners[t]] += tv[t]
        c = p.term_component[t]
        c == 0 && continue
        cowner[c] in (0, owners[t]) || _brm_nlme_error("brm_laplace_problem: " *
            "internal: a component of integrated effects reads two units' terms")
        cowner[c] = owners[t]
    end
    # (a component no term reads sits at z = 0 with h = I there: zero)
    for c in eachindex(p.components)
        cowner[c] == 0 || (values[cowner[c]] += fc[c] - logdets[c] / 2)
    end
    values
end

# A draw from N(0, h⁻¹): x ~ N(0, A⁻¹) through A's factor, then for the soft
# sums x − X G⁻¹ (Uᵀx + η), η ~ N(0, Λ⁻¹), whose covariance is
# A⁻¹ − X G⁻¹ Xᵀ = h⁻¹.
function _brm_spl_draw(p::BRMSparseLaplaceProblem, state, rng)
    state.F isa _BRMOperatorFactor && return _brm_operator_draw(state.F, rng)
    state.F isa _BRMAugmentedFactor && return _brm_spl_covariance_root(
        b -> state.F \ b, randn(rng, length(p.integrated))).draw
    x = state.F.UP \ randn(rng, length(p.integrated))
    W = state.W
    W === nothing && return x
    x .- W.X * (W.G \ (W.U' * x .+ randn(rng, length(p.wλ)) ./ sqrt.(p.wλ)))
end

# A shift for every effect: one number for all, or one per component.
_brm_spl_effect_shift(p::BRMSparseLaplaceProblem, shift::Real) = shift
_brm_spl_effect_shift(p::BRMSparseLaplaceProblem, shift::AbstractVector) =
    shift[p.component]

# The factor of h + shift I, reusing the symbolic analysis. `shift` is one
# number, or one per component (`p.components`; h is block diagonal over them).
function _brm_spl_factor!(p::BRMSparseLaplaceProblem, h; shift=0.0)
    p.algebra === nothing || return _brm_operator_factor(p, h; shift)
    if shift isa AbstractVector
        hs = copy(h)
        nonzeros(hs)[p.diagonal] .+= _brm_spl_effect_shift(p, shift)
        H = Hermitian(hs, :L); shift = 0.0
    else
        H = Hermitian(h, :L)
    end
    if p.factor[] === nothing
        F = cholesky(H; shift, check=false)
        p.factor[] = F
        return F
    end
    cholesky!(p.factor[], H; shift, check=false)
end

# The shifts the Newton direction tries where a component's h is not positive
# definite: 10^k for k = −4, …, 7.
const _BRM_SPL_SHIFTS = ntuple(k -> 10.0^(k - 5), 12)

# Whether component `members`' block of h + μ I is positive definite (its
# soft sums' rank-one part, which `_brm_spl_woodbury` keeps out of the
# factor, only adds a positive semidefinite term).
_brm_spl_block_definite(p::BRMSparseLaplaceProblem, h::SparseMatrixCSC, members, μ) =
    issuccess(cholesky(Hermitian(h[members, members], :L); shift=μ, check=false))

# The Newton direction and the shift each component took: h⁻¹ g, or for each
# component whose h is not positive definite away from its mode,
# (h + μ_c I)⁻¹ g with the smallest μ_c in `_BRM_SPL_SHIFTS` that is. Every
# component's shift is its own: one component's curvature never damps
# another's step. `nothing` when a component's h stays indefinite.
function _brm_spl_shifted_direction(p::BRMSparseLaplaceProblem, h, g)
    p.algebra === nothing || return _brm_operator_direction(p, h, g)
    nc = length(p.components)
    F = _brm_spl_factor!(p, h)
    issuccess(F) && return (_brm_spl_solve(F, _brm_spl_woodbury(p, F), g), zeros(nc))
    shift = zeros(nc)
    for (c, members) in enumerate(p.components)
        _brm_spl_block_definite(p, h, members, 0.0) && continue
        k = findfirst(μ -> _brm_spl_block_definite(p, h, members, μ), _BRM_SPL_SHIFTS)
        k === nothing && return nothing
        shift[c] = _BRM_SPL_SHIFTS[k]
    end
    F = _brm_spl_factor!(p, h; shift)
    issuccess(F) || return nothing
    (_brm_spl_solve(F, _brm_spl_woodbury(p, F), g), shift)
end

# The direction alone (`_brm_spl_shifted_direction`).
function _brm_spl_direction(p::BRMSparseLaplaceProblem, h, g)
    r = _brm_spl_shifted_direction(p, h, g)
    r === nothing ? nothing : first(r)
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
    _brm_spl_gaussian_parameters!(p, uA)
    _brm_spl_gmrf_parameters!(p, uA)
    C, dC, A, values = _brm_spl_cmaps(p, uA)
    (; uA, values, A, C, dC)
end

# The continuation's likelihood weights (`_brm_spl_mode!`): one stage when
# the integrated effects' density is log-concave (its mode is unique), else
# `_BRM_LAP_STAGES` equal stages from the prior. Intermediate stages stop at a
# decrement of `_BRM_LAP_STAGE_TOL` (modes to about 1e-6), the last at `tol`.
const _BRM_LAP_STAGES = 10
const _BRM_LAP_STAGE_TOL = 1e-12
_brm_lap_stages(concave::Bool) = concave ? (1.0,) :
    ntuple(k -> k / _BRM_LAP_STAGES, _BRM_LAP_STAGES)

# Row terms of the closed-form families are log-concave in their first
# parameter, which the effects enter linearly, and the prior is Gaussian. A
# moving second parameter (an effect on a scale or shape) or an observation
# factor (`@plate` cell, sum) leaves concavity unknown.
_brm_spl_concave(p::BRMSparseLaplaceProblem) = isempty(p.cells) && !any(p.moving)

# The integrated effects' prior mode at the population point. The prior is
# Gaussian, so one Newton step from zero reaches it: zero, except where a
# declared vector's prior mean is not.
function _brm_spl_prior_mode(p::BRMSparseLaplaceProblem, pop)
    n = length(p.integrated)
    z = zeros(n); g = zeros(n); h = _brm_spl_precision(p)
    _brm_spl_assemble!(zeros(length(p.components)), g, h, p, pop.A, pop.C, z;
        derivatives=true, uA=pop.uA, weight=0.0)
    all(iszero, g) && return z
    r = _brm_spl_shifted_direction(p, h, g)
    r === nothing ? z : first(r)
end

# Which components are at their rounding floor (`_brm_lap_floor_candidate` in
# nlme_laplace.jl), by one assembly at every component's probe point and one
# solve with the step's h.
function _brm_spl_at_floor(p::BRMSparseLaplaceProblem, pop, z, δ, g, h, λ; weight::Float64)
    ξ = _BRM_LAP_FLOOR_PROBE
    nc = length(p.components)
    r = zeros(length(z))
    _brm_spl_assemble!(zeros(nc), r, _brm_spl_precision(p), p, pop.A, pop.C, z .+ ξ .* δ;
        derivatives=true, uA=pop.uA, weight)
    r .-= (1 - ξ) .* g
    all(isfinite, r) || return falses(nc)
    direction = _brm_spl_shifted_direction(p, h, r)
    direction === nothing && return falses(nc)
    ρ = zeros(nc)
    for i in eachindex(r)
        ρ[p.component[i]] += r[i] * first(direction)[i]
    end
    _brm_lap_at_floor.(ρ, λ)
end

# Newton's method on the continuation density prior + s × likelihood, from z
# (in place), for every component at once: one factor per step, with each
# component's own shift where its h is not positive definite
# (`_brm_spl_shifted_direction`), its own step length and backtracking
# (`_brm_spl_step!`), and its own stop: decrement at most `tol`, or at its
# rounding floor on an unshifted step (`_brm_spl_at_floor`; a shifted decrement
# shrinks slowly without being at the mode). Each component's steps and shifted
# steps go into `search`, stage `stage`. With `start`, a likelihood that is not
# finite at z moves z halfway toward zero effects until it is (deterministically).
function _brm_spl_stage!(p::BRMSparseLaplaceProblem, pop, z, s, tol, search, stage;
        start::Bool=false)
    n = length(z); nc = length(p.components)
    fc = zeros(nc); g = zeros(n); h = _brm_spl_precision(p)
    _brm_spl_assemble!(fc, g, h, p, pop.A, pop.C, z; derivatives=true, uA=pop.uA, weight=s)
    halvings = 0
    while start && any(!iszero, z) && !(all(isfinite, fc) && all(isfinite, g)) &&
            halvings <= 60
        halvings += 1
        halvings > 60 ? fill!(z, 0.0) : (z ./= 2)
        _brm_spl_assemble!(fc, g, h, p, pop.A, pop.C, z; derivatives=true, uA=pop.uA,
            weight=s)
    end
    all(isfinite, fc) && all(isfinite, g) || return (; status=:nonfinite, iterations=0, h)
    converged = falses(nc)
    previous = fill(Inf, nc)
    λ = zeros(nc)
    iterations = 0
    while true
        _brm_spl_finite(h) || return (; status=:nonfinite, iterations, h)
        r = _brm_spl_shifted_direction(p, h, g)
        r === nothing && return (; status=:nonfinite, iterations, h)
        δ, shift = r
        all(isfinite, δ) || return (; status=:nonfinite, iterations, h)
        fill!(λ, 0.0)
        for i in 1:n
            λ[p.component[i]] += g[i] * δ[i]
        end
        candidate = [!converged[c] && shift[c] == 0 &&
            _brm_lap_floor_candidate(λ[c], previous[c], tol) for c in 1:nc]
        floor = any(candidate) ? _brm_spl_at_floor(p, pop, z, δ, g, h, λ; weight=s) : candidate
        for c in 1:nc
            converged[c] && continue
            if λ[c] <= tol || (candidate[c] && floor[c])
                converged[c] = true
                search.decrement[c] = λ[c]
                search.stop[c] = λ[c] <= tol ? :tol : :floor
            else
                previous[c] = shift[c] == 0 ? λ[c] : Inf
                search.steps[c] += 1
                if shift[c] > 0
                    search.stage_shifts[c, stage] += 1
                    search.max_shift[c] = max(search.max_shift[c], shift[c])
                end
            end
        end
        all(converged) && return (; status=:converged, iterations, h)
        iterations == p.maxiter &&
            return (; status=:unconverged, iterations, h, converged, λ)
        iterations += 1
        _brm_spl_step!(p, pop, z, δ, g, h, fc, converged; weight=s, search) ||
            return (; status=:nonfinite, iterations, h)
    end
end

# One evaluation's mode-search record, per component: Newton steps, shifted
# steps per stage, the largest shift, steps whose full Newton step was refused,
# and the last stage's final decrement and stop (`:tol`, or `:floor` for the
# rounding floor).
_brm_spl_search(nc, stages) = (; stages=collect(stages), steps=zeros(Int, nc),
    max_shift=zeros(nc), stage_shifts=zeros(Int, nc, length(stages)),
    backtracks=zeros(Int, nc), decrement=fill(NaN, nc), stop=fill(:none, nc))

# Every component's mode at population point φ, by continuation from the
# prior: start at the prior's mode (`_brm_spl_prior_mode`) and follow each
# component's mode as the likelihood's weight rises to one over the stages of
# `_brm_lap_stages` (`_brm_spl_stage!`). The mode reached is the one
# connected to the prior's mode, for every call alike: it does not depend on
# earlier evaluations, and as φ varies it changes only where the followed mode
# ceases to exist. Then h at the modes and its factor. `p.search` records the
# evaluation's steps per component; `p.folds` counts evaluations in which a
# component's followed mode needed a shifted step after the first stage.
function _brm_spl_mode!(p::BRMSparseLaplaceProblem, φ::AbstractVector)
    p.algebra === nothing || empty!(p.algebra.diagnostics)
    pop = _brm_spl_population(p, φ)
    if !(all(isfinite, pop.A) && all(isfinite, pop.C))
        p.nonfinite += 1
        return (; finite=false, value=-Inf, iterations=0)
    end
    nc = length(p.components)
    stages = _brm_lap_stages(_brm_spl_concave(p))
    search = _brm_spl_search(nc, stages)
    p.search = search
    z = _brm_spl_prior_mode(p, pop)
    iterations = 0
    h = nothing
    for (k, s) in enumerate(stages)
        r = _brm_spl_stage!(p, pop, z, s, s == 1 ? p.tol : max(p.tol, _BRM_LAP_STAGE_TOL),
            search, k; start=k == 1)
        iterations += r.iterations
        if r.status === :unconverged
            p.unconverged += 1
            @warn("brm_laplace_problem: $(count(!, r.converged)) of $nc components' " *
                "modes did not converge in $(p.maxiter) Newton iterations of " *
                "continuation stage $k of $(length(stages)) at population point " *
                "$(repr(collect(φ))); its log density is -Inf. Decrements: " *
                "$(r.λ[.!r.converged])", maxlog=3)
            return (; finite=false, value=-Inf, iterations)
        elseif r.status === :nonfinite
            p.nonfinite += 1
            return (; finite=false, value=-Inf, iterations)
        end
        h = r.h
    end
    any(>(0), view(search.stage_shifts, :, 2:length(stages))) && (p.folds += 1)
    F = _brm_spl_factor!(p, h)
    issuccess(F) || _brm_nlme_error("brm_laplace_problem: the integrated effects' " *
        "density has a stationary point that is not a maximum at this population point")
    p.u[p.integrated] .= z
    u = copy(pop.uA)
    u[p.integrated] .= z
    (; finite=true, pop..., z, u, h, F, W=_brm_spl_woodbury(p, F), iterations)
end

# One step along the Newton direction δ for the moving components, in place
# in z. A component first tries the full step (Armijo on its density, or its
# decrement at the rounding floor). Where that fails, each of its step blocks
# halves its own step length until its own terms gain (Armijo on the terms and
# prior pieces that touch it, with its share of the decrement); a component
# whose density still did not rise then takes a block Gauss–Seidel sweep: each
# block in turn its own Newton step on its own terms with the others fixed,
# which can only raise it. Every accepted trial also has finite first and
# second derivatives. Updates fc, g and h to the accepted z; the usual full
# step therefore needs just one derivative assembly. False when a component
# finds no finite gain.
function _brm_spl_step!(p::BRMSparseLaplaceProblem, pop, z, δ, g, h, fc, converged;
        weight::Float64=1.0, search=nothing)
    p.algebra === nothing || return _brm_operator_step!(p, pop, z, δ, g, h, fc, converged;
        weight, search)
    n = length(z)
    nb = length(p.sblocks)
    λc = zeros(length(fc))
    for i in 1:n
        λc[p.component[i]] += g[i] * δ[i]
    end
    ft = similar(fc); gt = similar(g); ht = _brm_spl_precision(p)
    trial = copy(z)
    for i in 1:n
        converged[p.component[i]] || (trial[i] = z[i] + δ[i])
    end
    _brm_spl_assemble!(ft, gt, ht, p, pop.A, pop.C, trial; derivatives=true, uA=pop.uA,
        weight)
    valid = _brm_spl_finite_blocks(p, gt, ht)
    converged = copy(converged)
    for c in eachindex(fc)
        converged[c] && continue
        if isfinite(ft[c]) && all(valid[p.sblock[i]] for i in p.components[c]) &&
                (λc[c] <= 1e-8 || ft[c] >= fc[c] + 1e-4 * λc[c])
            converged[c] = true
            for i in p.components[c]
                z[i] = trial[i]
            end
        elseif search !== nothing
            search.backtracks[c] += 1
        end
    end
    if all(converged)
        copyto!(fc, ft); copyto!(g, gt); _brm_spl_copy_precision!(h, ht)
        return true
    end
    p.blockwise += 1
    moving(k) = !converged[p.sblock_component[k]]
    λb = zeros(nb)
    for i in 1:n
        λb[p.sblock[i]] += g[i] * δ[i]
    end
    fb = zeros(nb); fbt = zeros(nb)
    _brm_spl_assemble!(similar(fc), nothing, nothing, p, pop.A, pop.C, z; derivatives=false,
        fb, uA=pop.uA, weight)
    t = [moving(k) ? 1.0 : 0.0 for k in 1:nb]
    accepted = [!moving(k) for k in 1:nb]
    trial = copy(z)
    for _ in 1:60
        for i in 1:n
            k = p.sblock[i]
            accepted[k] || (trial[i] = z[i] + t[k] * δ[i])
        end
        _brm_spl_assemble!(ft, gt, ht, p, pop.A, pop.C, trial; derivatives=true,
            fb=fbt, uA=pop.uA, weight)
        valid = _brm_spl_finite_blocks(p, gt, ht)
        for k in 1:nb
            moving(k) || continue
            if valid[k] && isfinite(fbt[k]) &&
                    (λb[k] <= 1e-8 || fbt[k] >= fb[k] + 1e-4 * t[k] * λb[k])
                accepted[k] = true
            else
                accepted[k] = false
                t[k] /= 2
            end
        end
        all(accepted) && break
    end
    for c in eachindex(fc)
        converged[c] && continue
        if all(accepted[k] for k in 1:nb if p.sblock_component[k] == c) && isfinite(ft[c]) &&
                ft[c] >= fc[c] - 1e-12 * abs(fc[c])
            for i in p.components[c]
                z[i] = trial[i]
            end
        else
            p.sweeps += 1
            _brm_spl_sweep!(p, pop, z, c; weight) || return false
        end
    end
    _brm_spl_assemble!(fc, g, h, p, pop.A, pop.C, z; derivatives=true, uA=pop.uA, weight)
    true
end

# A block Gauss–Seidel sweep over component c's step blocks: each block's own
# Newton step on its own terms (h's block, shifted where it is not positive
# definite) with the other blocks fixed, halved until the component's density
# rises. False when a block finds no finite gain.
function _brm_spl_sweep!(p::BRMSparseLaplaceProblem, pop, z, c; weight::Float64=1.0)
    nc = length(p.components)
    fc = zeros(nc); ft = zeros(nc); g = zeros(length(z)); h = _brm_spl_precision(p)
    gt = similar(g); ht = _brm_spl_precision(p)
    _brm_spl_assemble!(fc, g, h, p, pop.A, pop.C, z; derivatives=true, uA=pop.uA, weight)
    for (k, members) in enumerate(p.sblocks)
        p.sblock_component[k] == c || continue
        gk = g[members]
        δ = _brm_spl_block_direction(p, h, members, gk)
        (δ === nothing || !all(isfinite, δ)) && return false
        λ = dot(gk, δ)
        λ <= 1e-8 && continue
        trial = copy(z)
        step = 1.0
        gained = false
        for _ in 1:60
            trial[members] .= z[members] .+ step .* δ
            _brm_spl_assemble!(ft, gt, ht, p, pop.A, pop.C, trial;
                derivatives=true, uA=pop.uA, weight)
            valid = _brm_spl_finite_blocks(p, gt, ht)
            if isfinite(ft[c]) && all(valid[p.sblock[i]] for i in p.components[c]) &&
                    ft[c] >= fc[c] + 1e-4 * step * λ
                gained = true
                break
            end
            step /= 2
        end
        gained || return false
        z[members] .= trial[members]
        copyto!(fc, ft); copyto!(g, gt); _brm_spl_copy_precision!(h, ht)
    end
    true
end

function _brm_spl_value(p::BRMSparseLaplaceProblem, state)
    state.finite || return -Inf
    fc = zeros(length(p.components))
    rest = _brm_spl_assemble!(fc, nothing, nothing, p, state.A, state.C, state.z;
        derivatives=false, third=true, uA=state.uA)
    sum(p.constant) + sum(fc) + rest + length(p.integrated) / 2 * log(2π) -
        _brm_spl_logdet(p, state.F, state.W) / 2 + _rk_slap_prior(p.kernels, state.uA)
end

# Value and gradient over the population coordinates: the structured path's
# (its header) with S = h⁻¹ on h's pattern for each unit's h_i⁻¹. Per term,
# with local C rows c1, c2: m = cᵀ S c, τ from m and ℓ‴, s = ½ Σ Cᵀ τ, v =
# h⁻¹ s, w = ℓ′ + ½ τ + ℓ″ C v; and per read and draw, (w ẑ + ℓ′ v + ℓ″ C S)
# times C's derivative along each scale or correlation coordinate of the block.
#
# With `active` (one flag per term, rows then cells) the gradient is that of
# the active terms' units only (`_brm_spl_unit_values`): an inactive term adds
# nothing to s, w or the cells' parts, so v = h⁻¹s is zero on the components
# only inactive terms read. The value stays the whole problem's. `prior =
# false` leaves the population prior out of both.
#
# The mode search accepts only finite first and second derivatives, but the
# gradient also needs the likelihood's third derivatives and the population
# pullbacks, which can still leave Float64's range where those do not. `bad`
# names the first such term in evaluation order; a nonfinite gradient at a
# finite value is then counted and reported (`_brm_spl_nonfinite_gradient!`)
# and returned unchanged.
function _brm_spl_value_and_gradient(p::BRMSparseLaplaceProblem, state;
        active=nothing, prior::Bool=true)
    bad = nothing
    n = length(p.integrated)
    nterms = length(p.family)
    fc = zeros(length(p.components))
    rows = Vector{_BRMLapRow}(undef, nterms)
    crows = (zeros(length(p.tz)), zeros(length(p.tz)))
    rest = _brm_spl_assemble!(fc, nothing, nothing, p, state.A, state.C, state.z;
        derivatives=false, third=true, rows, crows, uA=state.uA)
    value = sum(p.constant) + sum(fc) + rest + n / 2 * log(2π) -
        _brm_spl_logdet(p, state.F, state.W) / 2
    S = _brm_spl_selected_precision(p, state.F)
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
    on(t) = active === nothing || active[t]
    @inbounds for t in 1:nterms
        (p.term_component[t] == 0 || !on(t)) && continue
        r = rows[t]
        span = p.tptr[t]:(p.tptr[t + 1] - 1)
        c1 = view(crows[1], span); c2 = view(crows[2], span)
        m = length(span)
        moving = p.moving[t]
        m11 = m12 = m22 = 0.0
        zs = view(p.tz, span)
        for j in 1:m, i in 1:m
            Sij = S[_brm_spl_nzindex(p.pattern, zs[i], zs[j])]
            m11 += c1[i] * Sij * c1[j]
            moving && (m12 += c1[i] * Sij * c2[j]; m22 += c2[i] * Sij * c2[j])
        end
        two = _brm_lap_width(p.family[t]) == 2
        τ1 = m11 * r.t111 + 2m12 * r.t112 + m22 * r.t122
        τ2 = two ? m11 * r.t112 + 2m12 * r.t122 + m22 * r.t222 : 0.0
        bad === nothing && !(isfinite(τ1) && isfinite(τ2)) &&
            (bad = "the third derivative of row $(p.row[t]) of `$(p.response[t])`")
        taus[1, t] = τ1; taus[2, t] = τ2
        zs = view(p.tz, p.tptr[t]:(p.tptr[t + 1] - 1))
        for i in 1:m
            s[zs[i]] += (c1[i] * τ1 + (moving ? c2[i] * τ2 : 0.0)) / 2
        end
    end
    # The cells: contract D³ℓ with S=d(logdet approximation)/dH on each
    # bounded selection. Its latent part goes into s and its population part
    # goes straight into the gradient. S is the inverse for exact algebra,
    # or the signed sensitivity of the fixed-probe determinant.
    gcell = zeros(length(p.population))
    u = copy(state.uA)
    u[p.integrated] .= state.z
    for group in p.cells
        group isa _BRMSumCellGroup && continue
        _, correction = _rk_spl_cell_pass(group.ad, u) do batch
            for c in group.cells
                on(c.term) || continue
                m = length(c.lz)
                m == 0 && continue
                Sb = [S[_brm_spl_nzindex(p.pattern, a, b)] for a in c.lz, b in c.lz]
                Lb, signs = _brm_spl_sensitivity_root(state.F, Sb)
                for k in 1:m
                    direction = zeros(group.m)
                    direction[c.lpos] .= view(Lb, :, k)
                    weight = signs === nothing ? 0.5 : 0.5signs[k]
                    _, tv = _rk_spl_cell_accumulate(batch, group.ad, 2, direction, weight,
                        u, c.sel, c.i)
                    bad === nothing && !all(isfinite, tv) &&
                        (bad = "the third derivative of cell $(c.i) of `$(group.name)`")
                end
            end
        end
        bad === nothing && !all(isfinite, correction) &&
            (bad = "the third derivatives of the shared values of `$(group.name)`")
        s .+= correction[p.integrated]
        gcell .+= correction[p.population]
    end
    _brm_spl_sum_gradient!(s, gcell, p, state, u, on)
    bad === nothing && !(all(isfinite, s) && all(isfinite, gcell)) &&
        (bad = "the third derivatives of the sum factors of " * _brm_spl_sum_names(p))
    v = _brm_spl_solve(state.F, state.W, s)
    bad === nothing && !all(isfinite, v) &&
        (bad = "the solve with the conditional precision at the modes")
    _brm_spl_sum_implicit!(gcell, p, state, v, on)
    bad === nothing && !all(isfinite, gcell) &&
        (bad = "the curvature-vector products of the sum factors of " * _brm_spl_sum_names(p))
    # H_c[population, z] v_z directly, plus the cell's population gradient.
    # This replaces m HVPs and m identical gradients in the contractions above
    # with one of each, and needs no retained population-by-m matrix per cell.
    for group in p.cells
        group isa _BRMSumCellGroup && continue
        _, correction = _rk_spl_cell_pass(group.ad, u) do batch
            for c in group.cells
                on(c.term) || continue
                direction = zeros(group.m)
                direction[c.lpos] .= v[c.lz]
                _, gsel = _rk_spl_cell_accumulate(batch, group.ad, 0, nothing, 1.0,
                    u, c.sel, c.i)
                gfinite = all(isfinite, gsel)
                _, hv = _rk_spl_cell_accumulate(batch, group.ad, 1, direction, 1.0,
                    u, c.sel, c.i)
                bad === nothing && !(gfinite && all(isfinite, hv)) &&
                    (bad = "the gradient or curvature-vector product of cell $(c.i) of " *
                        "`$(group.name)` at the modes")
            end
        end
        gcell .+= correction[p.population]
        bad === nothing && !all(isfinite, correction) &&
            (bad = "the gradient or curvature-vector products of the shared values of " *
                "`$(group.name)` at the modes")
    end
    w = zeros(length(p.port))
    gω = zeros(length(p.omega))
    LCS = Float64[]
    @inbounds for t in 1:nterms
        on(t) || continue
        r = rows[t]
        e = p.entry[t]
        two = _brm_lap_width(p.family[t]) == 2
        if p.term_component[t] == 0
            w[e] = r.g1
            two && (w[e + 1] = r.g2)
            bad === nothing && !(isfinite(r.g1) && isfinite(r.g2)) &&
                (bad = "the first derivatives of row $(p.row[t]) of `$(p.response[t])`")
            continue
        end
        span = p.tptr[t]:(p.tptr[t + 1] - 1)
        c1 = view(crows[1], span); c2 = view(crows[2], span)
        m = length(span)
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
        bad === nothing && !(isfinite(w1) && isfinite(w2)) &&
            (bad = "the parameter derivatives of row $(p.row[t]) of `$(p.response[t])`")
        # (ℓ″ C S) per parameter, on the term's local effects.
        resize!(LCS, 2m); fill!(LCS, 0.0)
        for j in 1:m, i in 1:m
            Sij = S[_brm_spl_nzindex(p.pattern, zs[i], zs[j])]
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
    bad === nothing && !all(isfinite, gω) &&
        (bad = "the row terms' derivatives along the random-effect scales and correlations")
    gradient = p.affine ? _brm_slap_affine_pullback(p, w) :
        _rk_slap_pullback(p.kernels, state.uA,
            _brm_slap_cotangents(p, state.values, w))[p.population]
    bad === nothing && !all(isfinite, gradient) &&
        (bad = "the pullback of the row terms' parameters to the population coordinates")
    for (j, o) in enumerate(p.omega)
        gradient[findfirst(==(o), p.population)] += gω[j]
    end
    _brm_spl_gaussian_gradient!(gradient, p, state, S, v, active)
    bad === nothing && !all(isfinite, gradient) &&
        (bad = "the declared Gaussian vectors' prior terms")
    _brm_spl_gmrf_gradient!(gradient, p, state, S, v, active)
    bad === nothing && !all(isfinite, gradient) &&
        (bad = "the gmrf fields' prior terms")
    if !prior
        gradient .+= gcell
        _brm_spl_nonfinite_gradient!(p, state, value, gradient, bad)
        return value, gradient
    end
    lp, prior_gradient = _rk_slap_prior_gradient(p.kernels, state.uA)
    bad === nothing && !all(isfinite, prior_gradient) &&
        (bad = "the population prior's gradient")
    gradient .+= gcell .+ prior_gradient[p.population]
    _brm_spl_nonfinite_gradient!(p, state, value + lp, gradient, bad)
    value + lp, gradient
end

_brm_spl_sum_names(p::BRMSparseLaplaceProblem) =
    join(("`$(g.name)`" for g in p.cells if g isa _BRMSumCellGroup), ", ")

# A finite log density whose gradient is not. Every component's mode converged
# with finite first and second derivatives, so the cause is a derivative only
# the gradient needs: typically a likelihood's third derivative that automatic
# differentiation cannot represent in Float64 although its value and curvature
# are finite. The gradient is returned as it is (a sampler then fails or
# rejects the point); this counts it and names its first nonfinite term.
function _brm_spl_nonfinite_gradient!(p::BRMSparseLaplaceProblem, state, value,
        gradient, bad)
    (isfinite(value) && !all(isfinite, gradient)) || return nothing
    p.nonfinite_gradient += 1
    coordinates = findall(!isfinite, gradient)
    @warn("brm_laplace_problem: the gradient is not finite in $(length(coordinates)) of " *
        "$(length(gradient)) population coordinates (the first is " *
        "`$(p.names[first(coordinates)])`) although the log density $value is finite and " *
        "every component's mode converged, at population point " *
        "$(repr(state.uA[p.population])). The first nonfinite term is " *
        "$(something(bad, "none on its own (their sum is not finite)")). " *
        "Counted in `nonfinite_gradient`.", maxlog=3)
    nothing
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
                view(proposals, :, j); derivatives=false, uA=state.uA)
            logw[:, j] .= fc
            # less the proposal's log density, −½ dᵀ h d per component (h is
            # block-diagonal over the components), up to its constant
            hd = state.h * d
            for i in 1:n
                logw[p.component[i], j] += d[i] * hd[i] / 2
            end
            for (c, cols) in enumerate(p.wcols)
                state.h isa _BRMSumPrecision && continue
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
