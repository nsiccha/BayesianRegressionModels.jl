# src/nlme_laplace_plan.jl — which latent terms the sparse Laplace integrates.
#
# `brm_laplace_plan` sorts a model's latent terms (random-effect blocks, `cdar`
# walks, `car` fields: the sparse Laplace's records) by how its rows read them:
#
# - integrated: every row reading it is Normal, so the Laplace is exact there;
# - gated: read by other closed-form families (counts, binomial) or by `@plate`
#   cells; integrated only where a per-block diagnostic admits it;
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

"""
    BRMLaplacePlan

Which latent terms of a model [`brm_laplace_problem`](@ref) integrates, from
[`brm_laplace_plan`](@ref). `plan.terms` holds one entry per latent term
(random-effect block, `cdar` walk, `car` field): its `declaration`, `grouping`,
`kind`, `class` (`:integrated`, `:gated`, `:sampled` or `:never`), `reason`,
whether it is `admitted` (integrated by the problem), and for a gated term its
`importance_ess` (`(; min, median, below, blocks, draws)`) or `nothing`.
`plan.ess_floor` is the floor, a fraction of `draws`. Pass a plan as
`brm_laplace_problem(rk; integrate = plan)` to integrate exactly its admitted
terms; [`brm_laplace_plan`](@ref)'s `integrate` and `sample` override the
classes.
"""
struct BRMLaplacePlan
    terms::Vector{NamedTuple}
    ess_floor::Float64
    draws::Int
end

_brm_plan_admitted(plan::BRMLaplacePlan) = [t for t in plan.terms if t.admitted]

function Base.show(io::IO, ::MIME"text/plain", plan::BRMLaplacePlan)
    n = count(t -> t.admitted, plan.terms)
    println(io, "BRMLaplacePlan: ", n, " of ", length(plan.terms),
        " latent terms integrated (importance ESS floor ", plan.ess_floor, " of ",
        plan.draws, " draws per block)")
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
        ess_floor = 0.5, draws = 200, at = nothing, rng = Random.Xoshiro(20261031))
        -> BRMLaplacePlan

Sort `rk`'s latent terms for the sparse Laplace ([`BRMLaplacePlan`](@ref)):

- **integrated:** every row reading the term is Normal, so its Laplace is exact;
- **gated:** read by counts, binomial rows or `@plate` cells. The term is
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

`integrate = (names…)` and `sample = (names…)` override the classes by
declaration or grouping. A term on a shape or dispersion can be named in
`integrate`; a Student-t, stratified or `car_units` term cannot (the sparse
Laplace does not integrate it). `ess_floor = 0.5`
and `draws = 200` are defaults, not measured thresholds; the plan reports each
gated term's importance ESS so a reader can judge.
"""
function brm_laplace_plan(backend::RKBRMI; ad_backend, integrate=:auto, sample=(),
        ess_floor::Real=0.5, draws::Integer=200, at=nothing,
        rng=Random.Xoshiro(20261031), tol=1e-24, maxiter=50)
    _brm_laplace_plan(backend; ad_backend, integrate, sample, ess_floor=Float64(ess_floor),
        draws=Int(draws), at, rng, tol, maxiter).plan
end

# The plan and the sparse problem its diagnostic was measured on (over every
# candidate term), which `brm_laplace_problem` reuses when the plan admits all
# of them.
function _brm_laplace_plan(backend::RKBRMI; ad_backend, integrate, sample, ess_floor,
        draws, at, rng, tol, maxiter)
    forced = integrate === :auto ? Symbol[] : Symbol[_brm_plan_names(integrate)...]
    sampled = Symbol[_brm_plan_names(sample)...]
    named(t, names) = t.declaration in names || t.grouping in names
    records = [r for r in _rk_coordinate_records(backend.plan) if r.kind in _BRM_SPL_LATENT]
    isempty(records) && _brm_nlme_error("brm_laplace_plan: the model has no random-effect " *
        "block, `cdar` walk or `car` field to integrate")
    terms = map(records) do r
        (; declaration=_brm_spl_record_declaration(r), grouping=_brm_spl_record_group(r),
            kind=r.kind === :stratified_ranef ? :ranef : r.kind,
            never=_brm_plan_never(r))
    end
    for name in vcat(forced, sampled)
        any(t -> named(t, (name,)), terms) || _brm_nlme_error("brm_laplace_plan: `$name` " *
            "names no latent term; the terms are " *
            join(("`$(t.declaration)` over `$(t.grouping)`" for t in terms), ", "))
    end
    for t in terms
        t.never !== nothing && named(t, forced) && _brm_nlme_error("brm_laplace_plan: " *
            "`$(t.declaration)` cannot be integrated: $(t.never)")
    end
    candidates = [t for t in terms if t.never === nothing && !named(t, sampled)]
    isempty(candidates) && return (; plan=BRMLaplacePlan([_brm_plan_entry(t,
        t.never === nothing ? :sampled : :never,
        something(t.never, "named in `sample`"), false, nothing) for t in terms],
        ess_floor, draws), problem=nothing)
    p = _brm_sparse_laplace_problem(backend, unique(t.grouping for t in candidates);
        ad_backend, tol, maxiter, declarations=Set(t.declaration for t in candidates))
    reads = _brm_plan_reads(p)
    gated = Dict{Symbol,Any}()
    for t in candidates
        r = reads[t.declaration]
        r.moving && (gated[t.declaration] = :never; continue)
        (r.cells || any(!=(_BRM_LAP_NORMAL), r.families)) && (gated[t.declaration] = r)
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
    plan = BRMLaplacePlan(entries, ess_floor, draws)
    (; plan, problem=p)
end

_brm_plan_names(x::Symbol) = (x,)
_brm_plan_names(x) = Tuple(Symbol(n) for n in x)

_brm_plan_entry(t, class, reason, admitted, importance_ess) = (; t.declaration,
    t.grouping, t.kind, class, reason, admitted, importance_ess)

# Why the sparse Laplace never integrates a record, or `nothing`.
function _brm_plan_never(r)
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
    r.cells ? ["`@plate` cells"] : String[])

# Per block declaration of `p`: the families of the row terms that read it,
# whether a `@plate` cell reads it, and whether it moves a term's second
# parameter (a shape, dispersion or log scale). A BYM2 field's iid part counts
# toward its field.
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
    out = Dict(r => (; families=Set{Int8}(), cells=false, moving=false) for r in roots)
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
        out[d] = (; families=fams, out[d].cells, moving)
    end
    for group in p.cells, c in group.cells, i in c.lz
        d = root(p.blocks[zblock[i]])
        out[d] = (; out[d].families, cells=true, out[d].moving)
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
    values = map(sets) do I
        H = Matrix(Symmetric(state.h[I, I]))
        for (w, cols) in enumerate(p.wcols)
            inI = [i in cols for i in I]
            any(inI) && (H .+= p.wλ[w] .* (inI * inI'))
        end
        F = cholesky(Symmetric(H); check=false)
        issuccess(F) || return 0.0
        terms = unique(t for i in I for t in local_terms[i])
        z = copy(state.z)
        f0 = _brm_plan_local_value(p, state, terms, I, z, u)
        logw = map(1:draws) do _
            d = F.U \ randn(rng, length(I))
            z[I] .= state.z[I] .+ d
            lw = _brm_plan_local_value(p, state, terms, I, z, u) - f0 + dot(d, H * d) / 2
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
function _brm_plan_local_value(p::BRMSparseLaplaceProblem, state, terms, I, z, u)
    total = 0.0
    nrows = length(p.family)
    mmax = maximum(t -> p.tptr[t + 1] - p.tptr[t], eachindex(p.family); init=0)
    c1 = zeros(mmax); c2 = zeros(mmax)
    cells = nothing
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
            cells === nothing && (cells = Dict(c.term => (group, c)
                for group in p.cells for c in group.cells))
            group, c = cells[t]
            u[p.integrated[I]] .= z[I]
            total += _rk_nlme_cell_value(group.query, u, c.i)
            u[p.integrated[I]] .= state.z[I]
        end
    end
    for b in p.blocks
        if _brm_spl_identity(b)
            for i in I
                any(==(i), b.z) && (total -= z[i]^2 / 2)
            end
            continue
        end
        mine = Set(I)
        for e in axes(b.edges, 2)
            i, j = b.z[b.edges[1, e], 1], b.z[b.edges[2, e], 1]
            (i in mine || j in mine) && (total -= (z[i] - z[j])^2 / 2)
        end
        for (c, levels) in enumerate(b.sums)
            cols = [b.z[l, 1] for l in levels]
            any(in(mine), cols) && (total -= b.λ[c] * sum(z[cols])^2 / 2)
        end
    end
    total
end

# The sparse problem of a plan's admitted terms: the diagnostic's own problem
# when the plan admits every term it was built over.
function _brm_sparse_laplace_problem_from_plan(backend::RKBRMI, r; ad_backend, tol,
        maxiter)
    admitted = _brm_plan_admitted(r.plan)
    isempty(admitted) && _brm_nlme_error("brm_laplace_problem: the plan integrates no " *
        "latent term:\n" * sprint(show, MIME"text/plain"(), r.plan))
    declarations = Set(t.declaration for t in admitted)
    p = r.problem
    if p === nothing || Set(_brm_plan_problem_terms(p)) != declarations
        p = _brm_sparse_laplace_problem(backend, unique(t.grouping for t in admitted);
            ad_backend, tol, maxiter, declarations)
    end
    p.plan = r.plan
    p
end

# The latent terms a sparse problem integrates (a BYM2 field's iid part under
# its field).
_brm_plan_problem_terms(p::BRMSparseLaplaceProblem) = unique(b.declaration
    for b in p.blocks if !any(o -> o.declaration !== b.declaration &&
        _brm_nlme_owned(b.declaration, o.declaration), p.blocks))
