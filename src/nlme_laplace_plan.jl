# src/nlme_laplace_plan.jl — which latent terms the sparse Laplace integrates.
#
# `brm_laplace_plan` sorts a model's latent terms (random-effect blocks, `cdar`
# walks, `car` fields and declared Gaussian vectors) by how its rows read them:
#
# - integrated: every row reading it is Normal, so the Laplace is exact there;
# - gated: read by other closed-form families (counts, binomial), model-value
#   observations or `@plate` cells; integrated only where a per-block diagnostic
#   admits it;
# - never: the term moves a shape or dispersion, is Student-t, stratified, or
#   a `car_units` field, which the sparse Laplace does not integrate;
# - sampled: named in `sample`, or gated and not admitted.
#
# The diagnostic is measured after the factorization, block by block: a block
# is one level of the term (one area of a field), its proposal the Gaussian of
# h restricted to it given the rest at the mode, and its importance weights the
# exact density of the terms and prior pieces touching it against that
# Gaussian. `importance_ess` is reported as its minimum, median and the number
# of blocks below the floor; a gated term is admitted when no block is.

# A private, read-only template. Targets copy it before evaluating or remapping
# any workspace; copying a plan need not duplicate this retained preparation.
struct _BRMLaplacePreparation
    problem::BRMSparseLaplaceProblem
    ad_backend::Any
end
Base.deepcopy_internal(p::_BRMLaplacePreparation, ::IdDict) = p

"""
    BRMLaplacePlan

Which latent terms of a model [`brm_laplace_problem`](@ref) integrates, from
[`brm_laplace_plan`](@ref). `plan.terms` holds one entry per latent term
(random-effect block, `cdar` walk, `car` field or declared vector): its `declaration`, `grouping`,
`kind`, `class` (`:integrated`, `:gated`, `:sampled` or `:never`), `reason`,
whether it is `admitted` (integrated by the problem), and for a gated term its
`importance_ess` (`(; min, median, below, blocks, draws)`) or `nothing`.
`plan.ess_floor` is the floor, a fraction of `draws`. Pass a plan as
`brm_laplace_problem(rk; integrate = plan)` to integrate exactly its admitted
terms; [`brm_laplace_plan`](@ref)'s `integrate` and `sample` override the
classes. A generated plan retains its preparation: applying it to the same
`rk` with the same `ad_backend` copies the prepared workspaces without preparing
the observations again, including when only some candidate terms are admitted.
Each returned problem has independent mutable state. Keep the backend's inputs
unchanged while using the plan; another backend or AD configuration is prepared
afresh. This is an in-process reuse contract: diagnostic exports and
reconstructed plans do not retain a portable cache of compiled factors.
"""
struct BRMLaplacePlan
    terms::Vector{NamedTuple}
    ess_floor::Float64
    draws::Int
    max_cell_width::Int
    width_events::Dict{Symbol,NamedTuple}
    integrated_dimension::Int
    sampled_dimension::Int
    prepared::Union{Nothing,_BRMLaplacePreparation}
end

# Keep the original diagnostic-only constructor. Packed dimensions need a
# backend and are unknown for a manually reconstructed plan.
BRMLaplacePlan(terms, ess_floor, draws) = BRMLaplacePlan(terms, ess_floor, draws,
    128, Dict{Symbol,NamedTuple}(), -1, -1, nothing)
BRMLaplacePlan(terms, ess_floor, draws, prepared::Union{Nothing,_BRMLaplacePreparation}) =
    BRMLaplacePlan(terms, ess_floor, draws, 128, Dict{Symbol,NamedTuple}(), -1, -1, prepared)

_brm_plan_admitted(plan::BRMLaplacePlan) = [t for t in plan.terms if t.admitted]

function Base.show(io::IO, ::MIME"text/plain", plan::BRMLaplacePlan)
    n = count(t -> t.admitted, plan.terms)
    println(io, "BRMLaplacePlan: ", n, " of ", length(plan.terms),
        " latent terms integrated (importance ESS floor ", plan.ess_floor, " of ",
        plan.draws, " draws per block)")
    dimension(n) = n < 0 ? "unknown" : n
    println(io, "  dimensions: ", dimension(plan.integrated_dimension), " integrated, ",
        dimension(plan.sampled_dimension), " sampled; max_cell_width=", plan.max_cell_width)
    for t in plan.terms
        ess = t.importance_ess === nothing ? "" : string("; importance ESS min ",
            round(t.importance_ess.min; digits=1), ", median ",
            round(t.importance_ess.median; digits=1), ", ", t.importance_ess.below, " of ",
            t.importance_ess.blocks, " blocks below the floor")
        println(io, "  ", t.admitted ? "✓" : "·", " ", t.declaration, " (", t.kind,
            " over `", t.grouping, "`): ", t.class, ", ", t.reason, ess)
    end
end
Base.show(io::IO, plan::BRMLaplacePlan) = print(io, "BRMLaplacePlan(",
    count(t -> t.admitted, plan.terms), " of ", length(plan.terms), " terms integrated)")

"""
    brm_laplace_plan(rk::RKBRMI; ad_backend, integrate = :auto, sample = (),
        ess_floor = 0.5, draws = 200, at = nothing, rng = Random.Xoshiro(20261031),
        max_cell_width = 128)
        -> BRMLaplacePlan

Sort `rk`'s latent terms for the sparse Laplace ([`BRMLaplacePlan`](@ref)):

Declared independent Gaussian vectors, such as `field[1:n] .~ Normal.(mean,
scale)`, participate in automatic planning and can be named in `integrate`
or `sample` by their declaration. Each element is a diagnostic block; width
screening keeps or samples the whole vector. Fixed or outer-dependent means
and scales retain the authored prior and its derivatives. A term supplying
another Gaussian law's parameters stays sampled. Constrained, non-Gaussian
and correlated vector priors are never admitted. Derived Normal observations
remain gated because their dependence on the vector can be nonlinear.

- **integrated:** every row reading the term is Normal, so its Laplace is exact;
- **gated:** read by counts, binomial rows, model-value observations or `@plate`
  cells. The term is
  integrated where its per-block importance diagnostic admits it. That
  diagnostic proposes each block (one level of the term, one area of a field)
  from the Gaussian of the Laplace's precision restricted to it, given the rest
  at the mode, and weighs it by the exact density of the terms and prior pieces
  that touch it. A term is admitted when no block's importance ESS falls below
  `ess_floor × draws`. The diagnostic is measured at the population point `at`
  (default: the population coordinates' prior medians; a whole packed point is
  taken too);
- **never:** the term moves a shape or dispersion, or is a Student-t,
  stratified or `car_units` block, which the sparse Laplace does not integrate;
- **sampled:** named in `sample`, or gated and not admitted.

Before preparing derivatives, `max_cell_width` (a finite positive integer,
default 128) bounds the integrated footprint of each ordinary observation cell,
or each summand/direct outer block of a recognized sum. Cells are visited in
observation-name/cell order. An overwide cell samples the whole unforced term
contributing the most coordinates, breaking ties by declaration order, and
updates every cell that reads that term. Correlated margins stay together.
Forced terms are never silently sampled: an unresolved conflict raises with
the observation, cell, width and bound. All likelihoods and priors remain.
`plan.width_events` reports those choices; `integrated_dimension` and
`sampled_dimension` report the final packed dimensions. A plan with no admitted
terms supplies the full model's density and gradient through the same workflow.

`integrate = (names…)` and `sample = (names…)` override the classes by
declaration or grouping. A term on a shape or dispersion can be named in
`integrate`; a Student-t, stratified or `car_units` term cannot (the sparse
Laplace does not integrate it). `ess_floor = 0.5`
and `draws = 200` are defaults, not measured thresholds; the plan reports each
gated term's importance ESS so a reader can judge.
"""
function brm_laplace_plan(backend::RKBRMI; ad_backend, integrate=:auto, sample=(),
        ess_floor::Real=0.5, draws::Integer=200, at=nothing,
        rng=Random.Xoshiro(20261031), tol=1e-24, maxiter=50, max_cell_width=128)
    _brm_laplace_plan(backend; ad_backend, integrate, sample, ess_floor=Float64(ess_floor),
        draws=Int(draws), at, rng, tol, maxiter, max_cell_width).plan
end

# The plan and the sparse problem its diagnostic was measured on (over every
# candidate term). The public plan retains it too, including when the
# diagnostic admits only a subset of those terms.
function _brm_laplace_plan(backend::RKBRMI; ad_backend, integrate, sample, ess_floor,
        draws, at, rng, tol, maxiter, max_cell_width=128)
    max_cell_width = _brm_spl_width_limit(max_cell_width)
    forced = integrate === :auto ? Symbol[] : Symbol[_brm_plan_names(integrate)...]
    sampled = Symbol[_brm_plan_names(sample)...]
    named(t, names) = t.declaration in names || t.grouping in names
    records = [r for r in _rk_coordinate_records(backend.plan) if r.kind in _BRM_SPL_LATENT]
    isempty(records) && _brm_nlme_error("brm_laplace_plan: the model has no random-effect " *
        "block, `cdar` walk, `car` field or declared vector to integrate")
    terms = map(records) do r
        (; declaration=_brm_spl_record_declaration(r), grouping=_brm_spl_record_group(r),
            kind=r.kind === :stratified_ranef ? :ranef : r.kind,
            never=_brm_plan_never(r))
    end
    outer = _brm_spl_gaussian_outer(backend, [t for t in terms if t.never === nothing])
    terms = [any(d -> _brm_nlme_owned(d, t.declaration), outer) ?
        merge(t, (; never="its draws supply a declared Gaussian prior's mean or scale and must stay sampled")) : t
        for t in terms]
    for name in vcat(forced, sampled)
        any(t -> named(t, (name,)), terms) || _brm_nlme_error("brm_laplace_plan: `$name` " *
            "names no latent term; the terms are " *
            join(("`$(t.declaration)` over `$(t.grouping)`" for t in terms), ", "))
    end
    for t in terms
        named(t, forced) && named(t, sampled) && throw(ArgumentError(
            "brm_laplace_plan: `$(t.declaration)` is both forced and sampled"))
        t.never !== nothing && named(t, forced) && _brm_nlme_error("brm_laplace_plan: " *
            "`$(t.declaration)` cannot be integrated: $(t.never)")
    end
    candidates = [t for t in terms if t.never === nothing && !named(t, sampled)]
    screen = isempty(candidates) ? (; active=Bool[], events=Dict{Symbol,NamedTuple}(),
        integrated_dimension=0, sampled_dimension=length(_rk_layout_coordinate_names(backend))) :
        _brm_spl_screen(backend, candidates,
            Set(t.declaration for t in candidates if named(t, forced)), max_cell_width)
    candidates = candidates[screen.active]
    width_reason(t) = let e = screen.events[t.declaration]
        "observation `$(e.observation)`, cell $(e.cell), $(e.part) has integrated width $(e.width) above max_cell_width=$(e.bound); sampled whole term (contribution $(e.contribution))"
    end
    if isempty(candidates)
        entries = [_brm_plan_entry(t, t.never === nothing ? :sampled : :never,
            haskey(screen.events, t.declaration) ? width_reason(t) :
                something(t.never, "named in `sample`"), false, nothing) for t in terms]
        return (; plan=BRMLaplacePlan(entries, ess_floor, draws, max_cell_width,
            screen.events, 0, screen.sampled_dimension, nothing), problem=nothing)
    end
    p = _brm_sparse_laplace_problem(backend, unique(t.grouping for t in candidates);
        ad_backend, tol, maxiter, declarations=Set(t.declaration for t in candidates),
        max_cell_width, screened=true)
    reads = _brm_plan_reads(p)
    gated = Dict{Symbol,Any}()
    for t in candidates
        r = reads[t.declaration]
        r.moving && (gated[t.declaration] = :never; continue)
        (r.cells || r.values || any(!=(_BRM_LAP_NORMAL), r.families)) &&
            (gated[t.declaration] = r)
    end
    # the diagnostic, at the population point `at`, for the gated terms the
    # caller did not force
    measure = [t for t in candidates if get(gated, t.declaration, nothing) isa NamedTuple &&
        !named(t, forced)]
    ess = Dict{Symbol,Any}()
    if !isempty(measure)
        # `at` over this problem's population coordinates, or a whole packed point
        φ = at === nothing ? _brm_laplace_prior_median(p) :
            length(at) == length(p.u) ? Float64.(at[p.population]) : collect(Float64, at)
        state = _brm_spl_mode!(p, φ)
        state.finite || _brm_nlme_error("brm_laplace_plan: the Laplace is not finite at " *
            "the diagnostic's population point")
        for t in measure
            ess[t.declaration] = _brm_plan_ess(p, state, t.declaration, draws, ess_floor,
                rng)
        end
    end
    entries = map(terms) do t
        t.never !== nothing && return _brm_plan_entry(t, :never, t.never, false, nothing)
        named(t, sampled) && return _brm_plan_entry(t, :sampled, "named in `sample`",
            false, nothing)
        haskey(screen.events, t.declaration) && return _brm_plan_entry(t, :sampled,
            width_reason(t), false, nothing)
        g = get(gated, t.declaration, nothing)
        if g === :never
            reason = "it moves a shape or dispersion parameter"
            named(t, forced) && return _brm_plan_entry(t, :integrated,
                string(reason, "; named in `integrate`"), true, nothing)
            return _brm_plan_entry(t, :never, reason, false, nothing)
        end
        read = g === nothing ? "every row reading it is Normal" :
            string("read by ", join(_brm_plan_family_names(g), ", "))
        named(t, forced) && return _brm_plan_entry(t, g === nothing ? :integrated : :gated,
            string(read, "; named in `integrate`"), true, nothing)
        g === nothing && return _brm_plan_entry(t, :integrated,
            string(read, ", so its Laplace is exact"), true, nothing)
        e = ess[t.declaration]
        admitted = e.below == 0
        _brm_plan_entry(t, admitted ? :gated : :sampled, string(read, admitted ?
            "; every block's importance ESS is above the floor" :
            "; blocks below the importance ESS floor"), admitted, e)
    end
    # Retain preparation, not the diagnostic's numerical factorization and
    # selected inverse. Each target owns and fills that cache independently.
    p.factor[] = nothing
    admitted = Set(t.declaration for t in entries if t.admitted)
    ni = sum(length(b.z) for b in p.blocks if any(d -> _brm_nlme_owned(b.declaration, d), admitted); init=0)
    plan = BRMLaplacePlan(entries, ess_floor, draws, max_cell_width, screen.events,
        ni, length(p.u) - ni, _BRMLaplacePreparation(p, ad_backend))
    (; plan, problem=p)
end

_brm_plan_names(x::Symbol) = (x,)
_brm_plan_names(x) = Tuple(Symbol(n) for n in x)

_brm_plan_entry(t, class, reason, admitted, importance_ess) = (; t.declaration,
    t.grouping, t.kind, class, reason, admitted, importance_ess)

# Why the sparse Laplace never integrates a record, or `nothing`.
function _brm_plan_never(r)
    r.kind === :vector && get(r, :gaussian, nothing) === nothing &&
        return "the declared vector does not have an unconstrained independent Normal prior"
    r.kind === :stratified_ranef && return "a stratified block is not integrated yet"
    r.kind === :ranef && get(r, :mixing, nothing) !== nothing &&
        return "a Student-t block's draws are not Gaussian"
    r.kind === :car && r.separation !== nothing &&
        return "a `car_units` field's draws are not integrated"
    nothing
end

const _BRM_PLAN_FAMILY_NAMES = Dict(_BRM_LAP_POISSON => "Poisson",
    _BRM_LAP_BINOMIAL => "binomial", _BRM_LAP_NB2 => "NegativeBinomial2",
    _BRM_LAP_NORMAL => "Normal")
_brm_plan_family_names(r) = vcat(sort!([_BRM_PLAN_FAMILY_NAMES[f] for f in r.families]),
    r.cells ? ["`@plate` cells"] : String[],
    r.values ? ["model-value observations"] : String[])

# Per block declaration of `p`: the families of the row terms that read it,
# whether a `@plate` cell or model-value observation reads it, and whether it
# moves a term's second parameter (a shape, dispersion or log scale). A BYM2
# field's iid part counts toward its field.
function _brm_plan_reads(p::BRMSparseLaplaceProblem)
    second = falses(length(p.port))
    for t in eachindex(p.family)
        p.moving[t] && _brm_lap_width(p.family[t]) == 2 && (second[p.entry[t] + 1] = true)
    end
    term_of = zeros(Int, length(p.port))
    for t in eachindex(p.family)
        term_of[p.entry[t]] = t
        _brm_lap_width(p.family[t]) == 2 && (term_of[p.entry[t] + 1] = t)
    end
    roots = unique(blk.declaration for blk in p.blocks if !any(o -> o !== blk.declaration &&
        _brm_nlme_owned(blk.declaration, o), (x.declaration for x in p.blocks)))
    root(b) = only(r for r in roots if _brm_nlme_owned(b.declaration, r))
    out = Dict(r => (; families=Set{Int8}(), cells=false, values=false, moving=false)
        for r in roots)
    zblock = zeros(Int, length(p.integrated))
    for (j, b) in enumerate(p.blocks), i in b.z
        zblock[i] = j
    end
    for (j, b) in enumerate(p.blocks)
        d = root(b)
        fams = out[d].families; moving = out[d].moving
        for rr in p.block_reads[j]
            e = p.rentry[rr]
            t = term_of[e]
            t == 0 && continue
            push!(fams, p.family[t])
            second[e] && (moving = true)
        end
        out[d] = (; families=fams, out[d].cells, out[d].values, moving)
    end
    plates = _brm_nlme_plate_observations(p.backend.plan)
    for group in p.cells, c in group.cells, i in c.lz
        d = root(p.blocks[zblock[i]])
        out[d] = merge(out[d], group.name in plates ? (; cells=true) : (; values=true))
    end
    out
end

# One gated term's importance ESS per block at a mode `state`: a block is one
# level of the term (its margins), proposed from N(ẑ, h_II⁻¹) with the rest at
# the mode and weighed by the exact density of the terms and prior pieces that
# touch it.
function _brm_plan_ess(p::BRMSparseLaplaceProblem, state, declaration, draws, ess_floor,
        rng)
    local_terms = _brm_plan_local_terms(p)
    u = copy(state.uA)
    u[p.integrated] .= state.z
    sets = [b.z[l, :] for b in p.blocks if _brm_nlme_owned(b.declaration, declaration)
        for l in axes(b.z, 1)]
    cache = _brm_plan_local_cache(p, state)
    z = copy(state.z)
    values = map(sets) do I
        H = _brm_spl_precision_block(state.h, I)
        if !(state.h isa _BRMSumPrecision)
            for w in unique(w for i in I for w in cache.soft_by_effect[i])
                inI = [i in cache.soft_members[w] for i in I]
                H .+= p.wλ[w] .* (inI * inI')
            end
        end
        F = cholesky(Symmetric(H); check=false)
        issuccess(F) || return 0.0
        terms = unique(t for i in I for t in local_terms[i])
        f0 = _brm_plan_local_value(p, state, terms, I, z, u, cache)
        logw = map(1:draws) do _
            d = F.U \ randn(rng, length(I))
            z[I] .= state.z[I] .+ d
            lw = _brm_plan_local_value(p, state, terms, I, z, u, cache) - f0 + dot(d, H * d) / 2
            z[I] .= state.z[I]
            lw
        end
        all(isfinite, logw) || return 0.0
        w = exp.(logw .- maximum(logw))
        sum(w)^2 / sum(abs2, w)
    end
    (; min=minimum(values), median=median(values),
        below=count(<(ess_floor * draws), values), blocks=length(values), draws)
end

# Per integrated draw, the terms (rows, then cells) that read it.
function _brm_plan_local_terms(p::BRMSparseLaplaceProblem)
    out = [Int[] for _ in p.integrated]
    for t in 1:_brm_spl_nterms(p), k in p.tptr[t]:(p.tptr[t + 1] - 1)
        push!(out[p.tz[k]], t)
    end
    out
end

# The density of `terms` and of the prior pieces touching the draws `I` at
# integrated draws `z` (the population part of `state` held).
function _brm_plan_local_value(p::BRMSparseLaplaceProblem, state, terms, I, z, u,
        cache=_brm_plan_local_cache(p, state))
    total = 0.0
    nrows = length(p.family)
    mmax = cache.mmax
    c1 = zeros(mmax); c2 = zeros(mmax)
    for t in terms
        if t <= nrows
            e = p.entry[t]
            m = p.tptr[t + 1] - p.tptr[t]
            two = _brm_lap_width(p.family[t]) == 2
            _brm_spl_crows!(c1, c2, p, state.C, t)
            a = state.A[e]; bb = two ? state.A[e + 1] : 0.0
            for k in 1:m
                zk = z[p.tz[p.tptr[t] + k - 1]]
                a += c1[k] * zk
                p.moving[t] && (bb += c2[k] * zk)
            end
            total += _brm_lap_row(p.family[t], p.y[t], p.trials[t], a, bb, false,
                p.moving[t]).v
        else
            group, c = cache.cells[t]
            u[p.integrated[I]] .= z[I]
            total += _brm_plan_cell_value(group, c, I, u, cache.sums)
            u[p.integrated[I]] .= state.z[I]
        end
    end
    for i in I
        if cache.identity[i]
            mu, precision = _brm_spl_gaussian_parameters(p, i)
            total += precision * z[i] * (mu - z[i] / 2)
        end
    end
    for e in unique(e for i in I for e in cache.edges_by_effect[i])
        i, j = cache.edges[e]
        total -= (z[i] - z[j])^2 / 2
    end
    for c in unique(c for i in I for c in cache.soft_by_effect[i])
        δ = sum(z[i] - state.z[i] for i in I if i in cache.soft_members[c])
        total -= p.wλ[c] * (cache.soft_values[c] + δ)^2 / 2
    end
    total
end

# Materialize independent mutable state from the diagnostic's preparation.
# A plan applied to another backend or AD configuration still builds that
# backend's target, as does a manually constructed three-argument plan.
function _brm_sparse_laplace_problem_from_plan(backend::RKBRMI, r; ad_backend, tol,
        maxiter)
    admitted = _brm_plan_admitted(r.plan)
    isempty(admitted) && return _brm_sampled_laplace_problem(backend, r.plan; ad_backend, tol, maxiter)
    declarations = Set(t.declaration for t in admitted)
    prepared = r.plan.prepared
    if prepared !== nothing && prepared.problem.backend === backend &&
            isequal(prepared.ad_backend, ad_backend) &&
            issubset(declarations, Set(_brm_plan_problem_terms(prepared.problem)))
        # Backend inputs are read-only. Do not clone the model and its bound
        # data along with each target's derivative and solver workspaces.
        p = Base.deepcopy_internal(prepared.problem, IdDict{Any,Any}(backend => backend))
        Set(_brm_plan_problem_terms(p)) == declarations ||
            _brm_spl_restrict!(p, declarations)
        p.tol = Float64(tol)
        p.maxiter = Int(maxiter)
        fill!(p.u, 0.0)
        p.nonfinite = p.unconverged = p.blockwise = p.sweeps = 0
    else
        p = _brm_sparse_laplace_problem(backend, unique(t.grouping for t in admitted);
            ad_backend, tol, maxiter, declarations, max_cell_width=r.plan.max_cell_width)
    end
    p.plan = r.plan
    p
end

# Reclassify a prepared selection's actual reads. Padding in ordinary compact
# cells remains padding; newly sampled effects become population derivatives.
function _brm_spl_repartition(c, zof, pof, maps)
    positions = sort!(vcat(c.lpos, c.ppos))
    z = sort!([(zof[c.sel[k]], k) for k in positions if zof[c.sel[k]] > 0])
    ps = [(pof[c.sel[k]], k) for k in positions if pof[c.sel[k]] > 0]
    share(v) = get!(maps, v, v)
    merge(c, (; lz=first.(z), lpos=share(last.(z)),
        pidx=share(first.(ps)), ppos=share(last.(ps))))
end

function _brm_spl_restrict_cells(group, zof, pof, maps)
    # Newer preparation shares zero direction buffers across an observation's
    # cells. Keep that representation and its prepared AD arity: evaluation
    # fills the directions from each remapped cell's lpos, then clears them.
    if hasproperty(group, :dirs)
        foreach(d -> fill!(d, 0.0), group.dirs)
        cells = map(c -> _brm_spl_repartition(c, zof, pof, maps), group.cells)
        return merge(group, (; cells))
    end
    cells = map(group.cells) do c
        c = _brm_spl_repartition(c, zof, pof, maps)
        # Keep the prepared operator's direction arity; unused directions are
        # zero. The leading directions still give the selected Hessian columns.
        dirs = ntuple(length(c.dirs)) do k
            v = zeros(length(c.sel))
            k <= length(c.lpos) && (v[c.lpos[k]] = 1.0)
            v
        end
        merge(c, (; dirs))
    end
    merge(group, (; cells))
end

function _brm_spl_restrict_cells(group::_BRMSumCellGroup, zof, pof, maps)
    share(v) = get!(maps, v, v)
    cells = map(group.cells) do c
        direct = _brm_spl_repartition(c.direct, zof, pof, maps)
        terms = [map(t -> _brm_spl_repartition(t, zof, pof, maps), reduction)
            for reduction in c.terms]
        lz = sort!(unique!(vcat(direct.lz,
            [z for reduction in terms for t in reduction for z in t.lz])))
        localof = Dict(z => k for (k, z) in enumerate(lz))
        direct = merge(direct, (; loc=share([localof[z] for z in direct.lz])))
        terms = [map(t -> merge(t, (; loc=share([localof[z] for z in t.lz]))), reduction)
            for reduction in terms]
        r = length(terms)
        active = vcat(collect(1:r), r .+ direct.lpos)
        merge(c, (; direct, terms, lz, active))
    end
    _BRMSumCellGroup(group.name, group.query, cells, merge(group.prepared, (; cells)))
end

# Restrict a private copy of a prepared problem, without querying footprints or
# preparing AD again. Keep omega's original indexing, which the copied row
# kernels use; the retained blocks reference their original entries in it.
function _brm_spl_restrict!(p::BRMSparseLaplaceProblem, declarations)
    keep = findall(b -> any(d -> _brm_nlme_owned(b.declaration, d), declarations), p.blocks)
    oldz = sort!([z for j in keep for z in p.blocks[j].z])
    zmap = zeros(Int, length(p.integrated)); zmap[oldz] .= eachindex(oldz)
    blockmap = zeros(Int, length(p.blocks)); blockmap[keep] .= eachindex(keep)
    p.integrated = p.integrated[oldz]
    if p.gaussian !== nothing
        g = p.gaussian
        blocks = [merge(b, (; z=zmap[b.z])) for b in g.blocks if all(i -> zmap[i] > 0, b.z)]
        p.gaussian = _BRMSPLGaussianPrior(blocks, g.ports, g.read, g.pullback,
            g.mean[oldz], g.precision[oldz], g.values)
    end
    p.population = setdiff(eachindex(p.u), p.integrated)
    p.names = _rk_layout_coordinate_names(p.backend)[p.population]
    p.blocks = [_BRMSlapBlock(b.declaration, b.group, b.K, zmap[b.z], b.omega,
        b.edges, b.sums, b.λ) for b in p.blocks[keep]]
    p.integrate = unique(b.group for b in p.blocks)
    old = (; p.rptr, p.rblock, p.rlevel, p.rentry)
    p.rptr = [1]; p.rblock = Int[]; p.rlevel = Int[]; p.rentry = Int[]; p.coff = Int[]
    p.block_reads = [Int[] for _ in p.blocks]
    nc = 0
    for e in eachindex(p.port)
        for r in old.rptr[e]:(old.rptr[e + 1] - 1)
            j = blockmap[old.rblock[r]]
            j == 0 && continue
            push!(p.rblock, j); push!(p.rlevel, old.rlevel[r])
            push!(p.rentry, e); push!(p.coff, nc)
            push!(p.block_reads[j], length(p.rblock))
            nc += p.blocks[j].K
        end
        push!(p.rptr, length(p.rblock) + 1)
    end
    for t in eachindex(p.family)
        e = p.entry[t]
        p.moving[t] = _brm_lap_width(p.family[t]) == 2 && p.rptr[e + 2] > p.rptr[e + 1]
    end
    # Sampled draws times their scales need the existing nonlinear row path.
    p.affine = false
    p.φ0 = zeros(length(p.population))
    p.jptr = Int[]; p.jcol = Int[]; p.jval = Float64[]
    zof = zeros(Int, length(p.u)); zof[p.integrated] .= eachindex(p.integrated)
    pof = zeros(Int, length(p.u)); pof[p.population] .= eachindex(p.population)
    # Preserve read-only position-map sharing when reclassifying coordinates;
    # these maps belong to this target, never to the retained template.
    maps = Dict{Vector{Int},Vector{Int}}()
    p.cells = [_brm_spl_restrict_cells(g, zof, pof, maps) for g in p.cells]
    _brm_spl_pattern!(p)
    _brm_spl_step_blocks!(p)
    p
end

# The latent terms a sparse problem integrates (a BYM2 field's iid part under
# its field).
_brm_plan_problem_terms(p::BRMSparseLaplaceProblem) = unique(b.declaration
    for b in p.blocks if !any(o -> o.declaration !== b.declaration &&
        _brm_nlme_owned(b.declaration, o.declaration), p.blocks))
