# src/nlme_laplace_plan.jl — which latent terms the sparse Laplace integrates.
#
# `brm_laplace_plan` sorts a model's latent terms (random-effect blocks, `cdar`
# walks, `car` fields and declared Gaussian vectors) by how its rows read them:
#
# - integrated: every row reading it is Normal, so the Laplace is exact there;
# - gated: affine in the first parameter of other closed-form families
#   (counts, binomial); integrated only where a per-block diagnostic admits it;
# - never: the term moves a shape or dispersion, is Student-t, stratified, or
#   a `car_units` field, which the sparse Laplace does not integrate;
# - sampled: named in `sample`, read by uncertified model-value observations
#   or `@plate` cells, or gated and not admitted. Explicit `integrate` opts in
#   to supported terms without a structural concavity guarantee.
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
    algebra_policy::NamedTuple
end

# Preserve the diagnostic-only constructors, including the original full
# constructor. New plans carry their numerical policy into reuse and replans.
BRMLaplacePlan(terms, ess_floor, draws, max_cell_width, width_events,
        integrated_dimension, sampled_dimension, prepared) =
    BRMLaplacePlan(terms, ess_floor, draws, max_cell_width, width_events,
        integrated_dimension, sampled_dimension, prepared,
        (; solver=:auto, matrixfree=MatrixFreeLaplace()))

function _brm_plan_algebra_policy(plan, solver, matrixfree)
    policy = (; solver=something(solver, plan.algebra_policy.solver),
        matrixfree=something(matrixfree, plan.algebra_policy.matrixfree))
    isequal(policy, plan.algebra_policy) && return plan
    BRMLaplacePlan(plan.terms, plan.ess_floor, plan.draws, plan.max_cell_width,
        plan.width_events, plan.integrated_dimension, plan.sampled_dimension,
        plan.prepared, policy)
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
    (; solver, matrixfree) = plan.algebra_policy
    println(io, "  conditional algebra: ", solver, "; exact_limit=", matrixfree.exact_limit,
        ", fixed probes=", matrixfree.probes, ", solve_maxiter=", matrixfree.solve_maxiter)
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
        max_cell_width = 128, solver=:auto, matrixfree=MatrixFreeLaplace())
        -> BRMLaplacePlan

Sort `rk`'s latent terms for the sparse Laplace ([`BRMLaplacePlan`](@ref)):

Declared independent Gaussian vectors, such as `field[1:n] .~ Normal.(mean,
scale)`, participate in automatic planning and can be named in `integrate`
or `sample` by their declaration. Each element is a diagnostic block; width
screening keeps or samples the whole vector. Fixed or outer-dependent means
and scales retain the authored prior and its derivatives. A term supplying
another Gaussian law's parameters stays sampled. Constrained, non-Gaussian
and correlated vector priors are never admitted. Derived Normal observations
stay sampled automatically because their dependence on the vector can be nonlinear;
name the vector in `integrate` to opt in.

- **integrated:** every row reading the term is recognized as affine Normal,
  with scale fixed conditional on the sampled coordinates. Its Laplace is exact
  with exact determinant algebra;
- **gated:** recognized affine log-Poisson, logit-binomial/Bernoulli, or
  log-NegativeBinomial2 rows with shape fixed conditional on the sampled
  coordinates. These have log-concave conditionals. The term is integrated
  where its per-block importance diagnostic admits it. That
  diagnostic proposes each block (one level of the term, one area of a field)
  from the Gaussian of the Laplace's precision restricted to it, given the rest
  at the mode, and weighs it by the exact density of the terms and prior pieces
  that touch it. A term is admitted when no block's importance ESS falls below
  `ess_floor × draws`. The diagnostic is measured at the population point `at`
  (default: the population coordinates' prior medians; a whole packed point is
  taken too);
- **never:** the term moves a shape or dispersion, or is a Student-t,
  stratified or `car_units` block, which the sparse Laplace does not integrate;
- **sampled:** named in `sample`, gated and not admitted, or read by arbitrary
  `@plate` cells/model-value observations without a structural log-concavity
  guarantee. This conservative rule also declines linear cells whose algebra
  has not been recognized. The reason identifies the relevant observations.

Automatic selection requires the recognized structure at every outer point;
a good local ESS or a positive mode Hessian cannot establish it. Log-concavity
does not guarantee an accurate Gaussian approximation. The local diagnostic
checks only the chosen point and one block at a time, so passing it does not
certify joint tails or accuracy elsewhere. Scale and shape may still depend on
sampled outer parameters. The selected partition stays fixed in the target.

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
declaration or grouping. Naming a supported term in `integrate` explicitly
opts in even without a concavity guarantee and bypasses the ESS gate, including
cells, model-value observations and terms on shape or dispersion. The caller
must assess that approximation; the automatic guarantee does not extend to
the joint conditional containing such forced terms. A Student-t, stratified
or `car_units` term cannot be forced (the sparse
Laplace does not integrate it). `ess_floor = 0.5`
and `draws = 200` are defaults, not measured thresholds; the plan reports each
gated term's importance ESS so a reader can judge.

`solver` and `matrixfree` select the conditional algebra as in
[`brm_laplace_problem`](@ref). The plan retains that policy when applied to a
target or replanned by [`brm_ladder_fit`](@ref); explicit target keywords override it.

`progress` reports the plan through Treebars: a Treebars node, under which the
plan builds its own, or a backend symbol such as `:state`. Each observation gets
a node, and each ordinary cell group a node naming its Hessian block's size while
Enzyme compiles it, so the node's time shows where preparation goes.
"""
function brm_laplace_plan(backend::RKBRMI; ad_backend, integrate=:auto, sample=(),
        ess_floor::Real=0.5, draws::Integer=200, at=nothing,
        rng=Random.Xoshiro(20261031), tol=1e-24, maxiter=_BRM_LAP_MAXITER, max_cell_width=128,
        solver=:auto, matrixfree=MatrixFreeLaplace(), progress=nothing)
    _brm_laplace_plan(backend; ad_backend, integrate, sample, ess_floor=Float64(ess_floor),
        draws=Int(draws), at, rng, tol, maxiter, max_cell_width, solver, matrixfree,
        progress).plan
end

# The plan and the sparse problem its diagnostic was measured on (over every
# candidate term). The public plan retains it too, including when the
# diagnostic admits only a subset of those terms.
_brm_laplace_plan(backend::RKBRMI; progress=nothing, kwargs...) =
    _brm_with_progress(progress; description="Laplace plan") do node
        _brm_laplace_plan_steps(backend; progress=node, kwargs...)
    end

function _brm_laplace_plan_steps(backend::RKBRMI; ad_backend, integrate, sample, ess_floor,
        draws, at, rng, tol, maxiter, max_cell_width=128,
        solver=:auto, matrixfree=MatrixFreeLaplace(), progress=nothing)
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
        uncertified=Dict{Symbol,Set{Symbol}}(),
        integrated_dimension=0, sampled_dimension=length(_rk_layout_coordinate_names(backend))) :
        _brm_with_progress(progress; description="Cell width screening") do _
            _brm_spl_screen(backend, candidates,
                Set(t.declaration for t in candidates if named(t, forced)), max_cell_width)
        end
    candidates = candidates[screen.active]
    width_reason(t) = let e = screen.events[t.declaration]
        "observation `$(e.observation)`, cell $(e.cell), $(e.part) has integrated width $(e.width) above max_cell_width=$(e.bound); sampled whole term (contribution $(e.contribution))"
    end
    unknown = Dict(t.declaration => "conditional log-concavity is not structurally certified: " *
        "read by `@plate` cells or model-value observations (" *
        join(sort!(string.(collect(screen.uncertified[t.declaration]))), ", ") *
        "); name it in `integrate` to opt in" for t in candidates
        if haskey(screen.uncertified, t.declaration))
    candidates = [t for t in candidates if !haskey(unknown, t.declaration) || named(t, forced)]
    if isempty(candidates)
        entries = [_brm_plan_entry(t, t.never === nothing ? :sampled : :never,
            haskey(screen.events, t.declaration) ? width_reason(t) :
                get(unknown, t.declaration, something(t.never, "named in `sample`")),
            false, nothing) for t in terms]
        return (; plan=BRMLaplacePlan(entries, ess_floor, draws, max_cell_width,
            screen.events, 0, length(_rk_layout_coordinate_names(backend)), nothing,
            (; solver, matrixfree)), problem=nothing)
    end
    p = _brm_sparse_laplace_problem(backend, unique(t.grouping for t in candidates);
        ad_backend, tol, maxiter, declarations=Set(t.declaration for t in candidates),
        max_cell_width, screened=true, solver, matrixfree, progress)
    reads = _brm_plan_reads(p)
    gated = Dict{Symbol,Any}()
    for t in candidates
        r = reads[t.declaration]
        # Prepared-factor reads must obey the same boundary if a future
        # factor representation reports a read absent from width screening.
        (r.cells || r.values) && get!(unknown, t.declaration,
            "conditional log-concavity is not structurally certified: read by " *
            join(_brm_plan_family_names(r), ", ") * "; name it in `integrate` to opt in")
        r.moving && (gated[t.declaration] = :never; continue)
        (r.cells || r.values || any(!=(_BRM_LAP_NORMAL), r.families)) &&
            (gated[t.declaration] = r)
    end
    # A rejected dispersion term must be sampled while diagnosing retained
    # effects. Otherwise its mode can distort their supposedly concave
    # conditional, even though the returned plan never integrates it.
    retained = Set(t.declaration for t in candidates if named(t, forced) ||
        (!haskey(unknown, t.declaration) && get(gated, t.declaration, nothing) !== :never))
    if !isempty(retained) && length(retained) < length(candidates)
        _brm_spl_restrict!(p, retained)
        _brm_spl_algebra!(p, solver, matrixfree)
    end
    # the diagnostic, at the population point `at`, for the gated terms the
    # caller did not force
    measure = [t for t in candidates if get(gated, t.declaration, nothing) isa NamedTuple &&
        t.declaration in retained && !named(t, forced)]
    ess = Dict{Symbol,Any}()
    isempty(measure) || _brm_with_progress(progress;
            description="Importance diagnostic of $(length(measure)) terms") do _
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
    forced_unknown = any(t -> named(t, forced) && (haskey(unknown, t.declaration) ||
        get(gated, t.declaration, nothing) === :never), candidates)
    entries = map(terms) do t
        t.never !== nothing && return _brm_plan_entry(t, :never, t.never, false, nothing)
        named(t, sampled) && return _brm_plan_entry(t, :sampled, "named in `sample`",
            false, nothing)
        haskey(screen.events, t.declaration) && return _brm_plan_entry(t, :sampled,
            width_reason(t), false, nothing)
        if haskey(unknown, t.declaration)
            return _brm_plan_entry(t, named(t, forced) ? :gated : :sampled,
                unknown[t.declaration] * (named(t, forced) ? "; named in `integrate`" : ""),
                named(t, forced), nothing)
        end
        g = get(gated, t.declaration, nothing)
        if g === :never
            reason = "it moves a shape or dispersion parameter"
            named(t, forced) && return _brm_plan_entry(t, :integrated,
                string(reason, "; named in `integrate`"), true, nothing)
            return _brm_plan_entry(t, :never, reason, false, nothing)
        end
        read = g === nothing ? "every row reading it is affine Normal in this term" :
            string(forced_unknown ? "recognized log-concave first-parameter dependence; read by " :
                "recognized log-concave conditional; read by ", join(_brm_plan_family_names(g), ", "))
        forced_unknown && (read *= "; joint conditional includes explicitly selected uncertified terms")
        named(t, forced) && return _brm_plan_entry(t, g === nothing ? :integrated : :gated,
            string(read, "; named in `integrate`"), true, nothing)
        g === nothing && return _brm_plan_entry(t, :integrated,
            string(read, forced_unknown ? "" : ", so its Laplace is exact with exact algebra"), true, nothing)
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
        ni, length(p.u) - ni, _BRMLaplacePreparation(p, ad_backend), (; solver, matrixfree))
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

# A value plan routes even ordinary predictor observations through factors.
# Recognize only direct predictor reads, with the required link and second
# parameters drawn solely from data or sampled scalars. Do not infer affine
# dependence from numerical probes of arbitrary assignments or cell bodies.
function _brm_plan_certified_observations(backend)
    plan = backend.plan
    out = Dict{Symbol,Int8}()
    plan isa _RKValuePlan || return out
    predictor_names = Set(p.name for p in plan.regression.predictors)
    scalars = Set(p.name for p in plan.regression.parameters)
    static = setdiff(Set(keys(rk_bound_data(backend))), union(predictor_names,
        Set(a.name for a in plan.assignments),
        Set(v.name for v in plan.regression.vector_parameters)))
    # A formula predictor may itself contain nonlinear model offsets or
    # model-valued design columns. Only data-backed columns establish the
    # usual affine dependence; the predictor name alone is insufficient.
    predictors = Dict(p.name => p.link for p in plan.regression.predictors
        if all(t -> !_rk_model_offset(t) && all(c -> c in static, t.columns), p.terms))
    fixed(x) = x isa Number || (x isa _BRMPreparedRef &&
        (x.name in static || x.name in scalars))
    for o in plan.observations
        o isa _BRMPreparedObservation || continue
        o.modifier === nothing && o.weight === nothing && o.missing_response === nothing || continue
        d = o.distribution
        d isa _BRMPreparedExpr && isempty(d.kwargs) || continue
        family, link, firstarg, fixedargs = if d.callable === Normal && length(d.args) == 2
            (_BRM_LAP_NORMAL, :identity, d.args[1], (d.args[2],))
        elseif d.callable === Poisson && length(d.args) == 1
            (_BRM_LAP_POISSON, :log, d.args[1], ())
        elseif d.callable === Bernoulli && length(d.args) == 1
            (_BRM_LAP_BINOMIAL, :logit, d.args[1], ())
        elseif d.callable === Binomial && length(d.args) == 2
            (_BRM_LAP_BINOMIAL, :logit, d.args[2], (d.args[1],))
        elseif d.callable === NegativeBinomial2 && length(d.args) == 2
            (_BRM_LAP_NB2, :log, d.args[1], (d.args[2],))
        else
            continue
        end
        firstarg isa _BRMPreparedRef && get(predictors, firstarg.name, nothing) === link || continue
        all(fixed, fixedargs) || continue
        out[o.name] = family
    end
    out
end

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
    certified = _brm_plan_certified_observations(p.backend)
    for group in p.cells, c in group.cells, i in c.lz
        d = root(p.blocks[zblock[i]])
        if haskey(certified, group.name)
            push!(out[d].families, certified[group.name])
        else
            out[d] = merge(out[d], group.name in plates ? (; cells=true) : (; values=true))
        end
    end
    out
end

# One gated term's importance ESS per block at a mode `state`: a block is one
# level of the term (its margins), proposed from N(ẑ, h_II⁻¹) with the rest at
# the mode and weighed by the exact density of the terms and prior pieces that
# touch it.
function _brm_plan_ess(p::BRMSparseLaplaceProblem, state, declaration, draws, ess_floor,
        rng)
    _brm_spl_cell_batches(p, state.u) do contexts
        _brm_plan_ess(p, state, declaration, draws, ess_floor, rng, contexts)
    end
end

function _brm_plan_ess(p::BRMSparseLaplaceProblem, state, declaration, draws, ess_floor,
        rng, contexts)
    local_terms = _brm_plan_local_terms(p)
    u = copy(state.uA)
    u[p.integrated] .= state.z
    sets = [b.z[l, :] for b in p.blocks if _brm_nlme_owned(b.declaration, declaration)
        for l in axes(b.z, 1)]
    cache = _brm_plan_local_cache(p, state; contexts)
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
            total += _brm_plan_cell_value(group, c, I, u, cache.sums;
                context=get(cache.contexts, group.name, u))
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
    # A native GMRF field's −½ zᵀQz, as far as it depends on z[I]: a pair
    # inside I once each way, a pair leaving I twice.
    Q = cache.gmrf
    if Q !== nothing
        rows, vals = rowvals(Q), nonzeros(Q)
        for i in I, k in nzrange(Q, i)
            j = rows[k]
            total -= (j in I ? 0.5 : 1.0) * z[i] * vals[k] * z[j]
        end
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
        maxiter, solver=nothing, matrixfree=nothing, progress=nothing)
    plan = _brm_plan_algebra_policy(r.plan, solver, matrixfree)
    (; solver, matrixfree) = plan.algebra_policy
    admitted = _brm_plan_admitted(plan)
    isempty(admitted) && return _brm_sampled_laplace_problem(backend, plan; ad_backend, tol, maxiter)
    declarations = Set(t.declaration for t in admitted)
    prepared = r.plan.prepared
    if prepared !== nothing && prepared.problem.backend === backend &&
            isequal(prepared.ad_backend, ad_backend) &&
            issubset(declarations, Set(_brm_plan_problem_terms(prepared.problem)))
        # Backend inputs are read-only. Do not clone the model and its bound
        # data along with each target's derivative and solver workspaces.
        p = Base.deepcopy_internal(prepared.problem, IdDict{Any,Any}(backend => backend))
        restricted = Set(_brm_plan_problem_terms(p)) != declarations
        restricted && _brm_spl_restrict!(p, declarations)
        p.tol = Float64(tol)
        p.maxiter = Int(maxiter)
        fill!(p.u, 0.0)
        p.nonfinite = p.unconverged = p.nonfinite_gradient = p.blockwise = p.sweeps = p.folds = 0
        # A full copy already owns the same fixed probes and reference. Only
        # a changed selection or numerical policy needs new preparation.
        retain = !restricted && p.algebra !== nothing && p.algebra.solver === solver &&
            isequal(p.algebra.options, matrixfree)
        retain || _brm_spl_algebra!(p, solver, matrixfree)
    else
        p = _brm_sparse_laplace_problem(backend, unique(t.grouping for t in admitted);
            ad_backend, tol, maxiter, declarations, max_cell_width=r.plan.max_cell_width,
            solver, matrixfree, progress)
    end
    p.plan = plan
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

function _brm_spl_restrict_cells(group, zof, pof, maps, layout)
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

function _brm_spl_restrict_cells(group::_BRMSumCellGroup, zof, pof, maps, layout)
    share(v) = get!(maps, v, v)
    repartition(c) = _brm_spl_factor_indices(
        _brm_spl_repartition(c, zof, pof, maps), maps, layout)
    cells = map(group.cells) do c
        direct = repartition(c.direct)
        terms = [map(repartition, reduction)
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
    p.gmrf = _brm_spl_gmrf_restrict(p.gmrf, blockmap, zmap)
    p.population = setdiff(eachindex(p.u), p.integrated)
    p.names = _rk_layout_coordinate_names(p.backend)[p.population]
    p.blocks = [_BRMSlapBlock(b.declaration, b.group, b.K, zmap[b.z], b.omega,
        b.edges, b.sums, b.λ, b.prior) for b in p.blocks[keep]]
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
    layout = (; integrated=copy(p.integrated), population=copy(p.population))
    p.cells = [_brm_spl_restrict_cells(g, zof, pof, maps, layout) for g in p.cells]
    _brm_spl_pattern!(p)
    _brm_spl_step_blocks!(p)
    p
end

# The latent terms a sparse problem integrates (a BYM2 field's iid part under
# its field).
_brm_plan_problem_terms(p::BRMSparseLaplaceProblem) = unique(b.declaration
    for b in p.blocks if !any(o -> o.declaration !== b.declaration &&
        _brm_nlme_owned(b.declaration, o.declaration), p.blocks))
