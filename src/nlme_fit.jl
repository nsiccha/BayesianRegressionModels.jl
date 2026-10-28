# NLME estimates read in the model's own names and natural units, and the
# one-call fit through NLMEEstimation's estimators (implemented by its
# extension). The NLME partition and the RK coordinate records remain the only
# sources of names and layout.

# The key of a random-effect block: its `|ID|`, or, without one, its predictor
# and grouping column, since every predictor owns its own unnamed block.
_brm_nlme_block_key(record) = record.id === nothing ?
    Symbol(first(first(record.margins)), "|", record.group) : record.id

# A record's natural value in the constrained values: one coefficient element,
# or a whole declaration.
function _brm_nlme_record_value(values, record)
    value = _brm_rk_declaration_value(values, record.declaration)
    index = get(record, :index, ())
    isempty(index) ? value : value[index...]
end

# Each record kind adds its entries to `population` (predictor => coefficient
# pairs), `scalars`, or `other` (the constrained values of declarations without
# a BRM address). Random-effect blocks are read from Ω separately.
_brm_nlme_named!(::Val{:ranef}, acc, record, values) = acc
_brm_nlme_named!(::Val{:stratified_ranef}, acc, record, values) = acc
function _brm_nlme_named!(::Val{:population}, acc, record, values)
    push!(get!(acc.population, record.predictor, Pair{Symbol,Any}[]),
        record.coefficient => _brm_nlme_record_value(values, record))
    acc
end
function _brm_nlme_named!(::Val{:population_block}, acc, record, values)
    # Level => value pairs, as `brm_start` takes a categorical column.
    contrasts = _brm_rk_declaration_value(values, record.declaration)
    push!(get!(acc.population, record.predictor, Pair{Symbol,Any}[]),
        record.coefficient => [level => x
            for (level, x) in zip(record.level_values, contrasts)])
    acc
end
function _brm_nlme_named!(::Val{:scalar}, acc, record, values)
    push!(acc.scalars, record.declaration => _brm_nlme_record_value(values, record))
    acc
end
function _brm_nlme_named!(::Val, acc, record, values)
    for root in sort!(collect(_rk_record_roots(record)))
        hasproperty(values, root) && !any(p -> first(p) === root, acc.other) &&
            push!(acc.other, root => getproperty(values, root))
    end
    acc
end

_brm_nlme_namedtuple(pairs) = NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))

function _brm_nlme_check_point(m::BRMNLMEModel, θ, σ, Ω)
    length(θ) == length(m.theta) || throw(DimensionMismatch(
        "brm_nlme_estimates: θ has length $(length(θ)), expected $(length(m.theta))"))
    length(σ) == length(m.sigma) || throw(DimensionMismatch(
        "brm_nlme_estimates: σ has length $(length(σ)), expected $(length(m.sigma))"))
    all(isfinite, θ) && all(isfinite, σ) || _brm_nlme_error(
        "brm_nlme_estimates: θ and σ must be finite")
    if _brm_nlme_ragged(m.view)
        Ω isa AbstractVector{<:AbstractMatrix} && length(Ω) == length(m.view.family_sizes) &&
            all(size(Ω[f]) == (K, K) for (f, K) in enumerate(m.view.family_sizes)) ||
            throw(DimensionMismatch("brm_nlme_estimates: a nested model's Ω is one " *
                "covariance matrix per family, of sizes $(m.view.family_sizes)"))
    else
        nη = length(m.view.block_margins)
        Ω isa AbstractMatrix && size(Ω) == (nη, nη) || throw(DimensionMismatch(
            "brm_nlme_estimates: Ω must be the $nη × $nη natural-effect covariance"))
    end
    nothing
end

function _brm_nlme_covariance(Ω::AbstractMatrix)
    covariance = Matrix{Float64}(Ω)
    sd = sqrt.(diag(covariance))
    cor = covariance ./ (sd * sd')
    cor[diagind(cor)] .= 1 # exactly, as `brm_start` requires of a correlation
    (; covariance, sd, cor)
end

_brm_nlme_margins(record) =
    [(; predictor, coefficient) for (predictor, coefficient) in record.margins]

# A block under its key: `(; group, margins, covariance, sd, cor)`. A
# stratified block is one entry, `(; group, by, margins, strata)`, whose
# `strata` pairs each `by` level with that stratum's `(; covariance, sd, cor)`.
function _brm_nlme_add_block!(blocks, record, ::Nothing, Ω, backend)
    push!(blocks, _brm_nlme_block_key(record) => (; group=record.group,
        margins=_brm_nlme_margins(record), _brm_nlme_covariance(Ω)...))
end
function _brm_nlme_add_block!(blocks, record, stratum::Int, Ω, backend)
    key = _brm_nlme_block_key(record)
    at = findfirst(p -> first(p) === key, blocks)
    if at === nothing
        push!(blocks, key => (; group=record.group, by=record.by,
            margins=_brm_nlme_margins(record), strata=Pair{Any,Any}[]))
        at = length(blocks)
    end
    level = _rk_grouping_levels(backend.plan.columns[record.by])[stratum]
    push!(last(blocks[at]).strata, level => _brm_nlme_covariance(Ω))
    blocks
end

# A block over a grouping in `shared_groups` has no Ω: its scales, correlation
# and level effects are θ coordinates, read from the constrained values.
# `(; group, crossed=true, margins, covariance, sd, cor, effects)`, a stratified
# one `(; group, by, crossed=true, margins, strata, effects)`; `effects` pairs
# each level with its effect on the margins, as the model adds it to the rows
# (`diag(sd) L z`, times √w for a Student-t block, the levels mixed by a
# `gr(g; cor=...)` factor).
function _brm_nlme_crossed_block(record, values, backend)
    K = length(record.margins)
    levels = _rk_ranef_record_levels(backend, record)
    z = reshape(_brm_rk_declaration_value(values, record.z), length(levels), K)
    margins = _brm_nlme_margins(record)
    if record.kind === :stratified_ranef
        strata = Any[_rk_grouping_levels(backend.plan.columns[record.by])...]
        stratum = _brm_level_strata(backend.plan.columns[record.group],
            backend.plan.columns[record.by], levels, strata, record;
            caller="brm_nlme_estimates")
        factors = [_brm_nlme_natural_factor(values, record, K, s) for s in eachindex(strata)]
        effects = reduce(vcat, (z[g:g, :] * factors[stratum[g]]' for g in eachindex(levels));
            init=zeros(0, K))
        return (; group=record.group, by=record.by, crossed=true, margins,
            strata=Pair{Any,Any}[level => _brm_nlme_covariance(A * A')
                for (level, A) in zip(strata, factors)],
            effects=_brm_nlme_level_effects(levels, effects))
    end
    A = _brm_nlme_crossed_factor(values, record, K)
    effects = z * A'
    mixing = get(record, :mixing, nothing)
    mixing === nothing || (effects = sqrt.(_brm_rk_declaration_value(values, mixing)) .* effects)
    if haskey(record, :level_correlation)
        # A known correlation's factor is data; a declared one is a value.
        factor = record.level_factor !== nothing ? record.level_factor :
            _brm_rk_declaration_value(values, record.level_correlation)
        effects = factor * effects
    end
    (; group=record.group, crossed=true, margins, _brm_nlme_covariance(A * A')...,
        effects=_brm_nlme_level_effects(levels, effects))
end

_brm_nlme_level_effects(levels, effects) =
    Pair{Any,Vector{Float64}}[level => effects[g, :] for (g, level) in enumerate(levels)]

# `diag(scale) L` of a plain block, its known scales and correlation included
# (a known margin's scale is its number; a known correlation's factor is the
# Cholesky factor of the matrix).
function _brm_nlme_crossed_factor(values, record, K)
    known = get(record, :known_scales, nothing)
    known === nothing && record.L !== nothing && return _brm_nlme_natural_factor(
        values, record, K, nothing)
    scale = known === nothing ? _brm_nlme_joint_scale(values, record, K) :
        Float64[k === nothing ? _brm_rk_declaration_value(values, record.scales[j]) : k
            for (j, k) in enumerate(known)]
    cor = get(record, :known_cor, nothing)
    L = record.L !== nothing ? Matrix{Float64}(_brm_rk_declaration_value(values, record.L)) :
        cor !== nothing ? Matrix{Float64}(cholesky(Symmetric(cor)).L) :
        Matrix{Float64}(I, K, K)
    Diagonal(scale) * L
end

"""
    brm_nlme_estimates(m::BRMNLMEModel, θ, σ, Ω[, ebes])
    brm_nlme_estimates(m::BRMNLMEModel, result[, ebes])

Read an NLME point of `m` in the model's own names and natural units. `θ` and
`σ` are `m`'s unconstrained vectors (`m.theta`, `m.sigma` order) and `Ω` the
natural-effect covariance: one block-diagonal matrix for a flat model, one
matrix per family for a nested one. With NLMEEstimation loaded, `result` may
be a FOCE(I) or SAEM result; a FOCE(I) result carries its own EBEs. `ebes` is
an `empirical_bayes` / `posthoc` result for `m`.

Returns `(; point, eta)`:

- `point` — one named tuple keyed by the model's identifiers:
  - a predictor's population coefficients under the predictor's name, one
    entry per coefficient address (`point.log_CL.weight` for
    `effect(log_CL, weight)`), on the scale the formula gives them. A
    categorical column is one entry of `level => value` pairs, one per
    non-reference level, as [`brm_start`](@ref) takes it;
  - a sampled scalar under its name, constrained (`point.sigma`, not its log);
  - a random-effect block under its `|ID|` (`point.pk`), or
    `var"<predictor>|<group>"` for a block without one, as
    `(; group, margins, covariance, sd, cor)` on the η scale; a stratified
    `gr(g, by=b)` block as `(; group, by, margins, strata)`, whose `strata`
    pairs each level of `b` with its stratum's `(; covariance, sd, cor)`. A
    block over a grouping in `shared_groups` (its coordinates are θ) adds
    `crossed = true` and `effects`, one `level => effects` pair per level of
    the grouping: the block's effect on its margins, as the model adds it to
    the rows;
  - `other` — the constrained values of estimated or fixed declarations that
    have no BRM address (shared HSGP and imputation latents, simplexes,
    thresholds), under their program names; empty when there are none.
- `eta` — `nothing` without `ebes`; otherwise `(; subjects, names, values)`:
  the subjects' grouping levels, the margin names of
  `NLMEEstimation.parameter_names(m).eta`, and the natural EBEs in `m`'s
  native shape (one column, or one compact vector, per subject).

Predictor names, scalar names and block keys share one namespace; a model in
which two coincide is refused. No input is mutated.
"""
function brm_nlme_estimates(m::BRMNLMEModel, θ::AbstractVector, σ::AbstractVector, Ω,
        ebes=nothing)
    _brm_nlme_check_point(m, θ, σ, Ω)
    view, backend = m.view, m.backend
    u = copy(m.unit)
    u[m.theta] = θ
    u[m.sigma] = σ
    values = _rk_constrained_values(backend, u)
    acc = (; population=OrderedDict{Symbol,Vector{Pair{Symbol,Any}}}(),
        scalars=Pair{Symbol,Any}[], other=Pair{Symbol,Any}[])
    for record in _brm_nlme_records(backend.plan)
        _brm_nlme_named!(Val(record.kind), acc, record, values)
    end
    blocks = Pair{Symbol,Any}[]
    for (; family, record, stratum, fixed, indices) in _brm_nlme_families(backend, view)
        # A fixed family (a completion's unit draws) is no random-effect block.
        fixed === nothing || continue
        _brm_nlme_add_block!(blocks, record, stratum,
            _brm_nlme_ragged(view) ? Ω[family] : Ω[indices, indices], backend)
    end
    for record in _brm_nlme_records(backend.plan)
        record.kind in (:ranef, :stratified_ranef) && record.group in view.shared_groups ||
            continue
        push!(blocks, _brm_nlme_block_key(record) =>
            _brm_nlme_crossed_block(record, values, backend))
    end
    # Predictors and their coefficients in formula order; records arrive in
    # emission order.
    specs = _brm_nlme_regression(backend.plan).predictors
    order = Dict(p.name => i for (i, p) in enumerate(specs))
    terms = Dict(p.name => Dict(t.addressee => j for (j, t) in enumerate(p.terms))
        for p in specs)
    population = sort!(collect(acc.population); by=p -> get(order, first(p), typemax(Int)))
    for (predictor, coefficients) in population
        position = get(terms, predictor, Dict{Symbol,Int}())
        sort!(coefficients; by=c -> get(position, first(c), typemax(Int)))
    end
    entries = vcat(Pair{Symbol,Any}[p => _brm_nlme_namedtuple(c) for (p, c) in population],
        acc.scalars, blocks, Pair{Symbol,Any}[:other => _brm_nlme_namedtuple(acc.other)])
    keys = first.(entries)
    repeated = unique(k for k in keys if count(==(k), keys) > 1)
    isempty(repeated) || _brm_nlme_error(
        "brm_nlme_estimates: $(join(string.("`", repeated, "`"), ", ")) names more than " *
        "one of a predictor, a scalar, a random-effect block and `other`")
    eta = ebes === nothing ? nothing : _brm_nlme_named_eta(m, ebes)
    (; point=_brm_nlme_namedtuple(entries), eta)
end

function _brm_nlme_named_eta(m::BRMNLMEModel, ebes)
    names = _brm_nlme_parameter_names(m).eta
    (; subjects=m.view.levels[ebes.subjects],
       names=_brm_nlme_ragged(m.view) ? names[ebes.subjects] : names,
       values=copy(ebes.modes))
end

"""
    brm_nlme_fit(x, estimator; start=nothing, inits=nothing, ebes=true,
        ad_backend=nothing, shared_point=nothing, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), progress=nothing, kwargs...)

Fit an NLME model through the RK backend with one of NLMEEstimation's own
estimators and read the result in the model's names and natural units.
Requires NLMEEstimation and Enzyme to be loaded.

- `x` — a `@brm` model ([`BRMI`](@ref)), its [`RKBRMI`](@ref), or a prepared
  [`BRMNLMEModel`](@ref). For a BRMI or RKBRMI the model is prepared with
  `ad_backend` (default reverse-mode `AutoEnzyme`) and the `shared_point`,
  `subject_group`, `nested_groups` and `shared_groups` of
  [`brm_nlme_model`](@ref); passing those to a prepared model is refused.
- `estimator` — `NLMEEstimation.saem`, `NLMEEstimation.focei`, or
  `NLMEEstimation.empirical_bayes` (equivalently `posthoc`, which estimates
  the subject effects at the start's population values). `kwargs` pass
  through to it (`rng`, `options`, `kernel`, `subjects`, `method`, …); FOCE
  without interaction is `focei` with `FOCEOptions(interaction=false)`.
- `start` — `nothing` for the prior-median start of
  [`brm_nlme_start`](@ref)`(m)`, a full RK packed point `u` for
  `brm_nlme_start(m, u)`, a start returned by `brm_nlme_start`, or
  NLMEEstimation's own start: a NamedTuple `(; theta, sigma, omega)` with
  optional `eta` (one column, or compact vector, per subject) and the
  `subjects` that `eta` belongs to, as grouping levels the way
  `brm_nlme_start` reports them (all of the model's without it), or an
  NLMEEstimation `saem` or `focei` result, which starts from its population
  point and warm-starts from its subjects' final SAEM states or FOCE(I)
  EBEs. Ω is derived from `omega`. The start's subject effects are matched to
  the estimator's `subjects` by subject; a caller's `η0` replaces them, and a
  subject the start has none for needs one.
- `inits` — initial values keyed by prior address, as [`brm_start`](@ref)
  takes them; every coordinate they do not address keeps its prior-median
  start. Give either `start` or `inits`.
- `ebes` — after `saem`, run `empirical_bayes` at the fitted population point
  (warm-started from SAEM's final subject states) so the result reports EBEs.
  SAEM's own `eta` are chain states, not EBEs. `true` uses `empirical_bayes`'s
  defaults, `false` skips the pass, and a NamedTuple of `empirical_bayes`
  keywords tunes it (`ebes=(; options=EBEOptions(maxiter=20),
  require_convergence=false)`); `subjects` and `η0` come from the SAEM result
  and are refused there. `focei` always returns its EBEs and `empirical_bayes`
  is the pass itself, so for them `ebes` is ignored and keywords are refused.
- `progress` — report the whole fit through Treebars: a Treebars node, under
  which the call builds a `BRM NLME fit` node, or a backend symbol such as
  `:state` that starts a new root. Its phases are `Lower to ReactiveKernels`,
  `Prepare NLME model` (the steps of [`brm_nlme_model`](@ref)), `Starting
  point`, `Estimate (<estimator>)` holding the estimator's own node, and after
  `saem` `Posthoc EBEs` holding `empirical_bayes`'s node; a stage `x` has
  already passed shows as reused. An interrupt requested on that node or an
  ancestor (`Treebars.request_interrupt!`) raises `Treebars.ProgressInterrupted`
  when the next preparation phase starts; once the estimator runs it honours
  the interrupt as NLMEEstimation documents (an interrupted `saem` returns its
  last completed iteration with `interrupted = true`), and the EBE pass is then
  skipped, returning `ebes = nothing`. `nothing` (the default) reports nothing.

Returns `(; estimates, result, ebes, start, model)`: `estimates` as
[`brm_nlme_estimates`](@ref) returns them, NLMEEstimation's raw `result`, the
EBE result used (or `nothing`), the start, and the prepared model.
"""
function brm_nlme_fit end
