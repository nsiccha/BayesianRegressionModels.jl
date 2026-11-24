# src/fixed_hyper.jl — MCMC over every latent of an RK model with its
# hyperparameters held at a point, such as an SVI fit's estimate.
#
# The user's direction `inference-not-model`: "I would rather use approximate
# inference (saem, svi, fixed hyperparameters into mcmc) than compromise on the
# statistical model. brm should be working on making this automatic." Built on
# design decision `0b9jgl5` (snag `mixed-membership-1a1930a6`).
#
# `brm_nlme_conditional_problem` frees only the units' draws and holds the
# shared latents (a shared HSGP's weights, a `car` field) with the population;
# `brm_two_stage_fit` holds the whole population. Here the split is between the
# hyperparameters, read from the RK coordinate records, and everything else.

# The weights of an ungrouped `hsgp(...)`: the RK component's `beta_raw`
# (src/rk_components.jl). A grouped one names its weights in `record.locals`.
const _BRM_HSGP_UNGROUPED_WEIGHTS = :beta_raw

"""
    brm_hyperparameters(rk::RKBRMI) -> Vector{Symbol}

The packed coordinates of `rk` that BRM reads as hyperparameters, by name, in
packed order. They are read from the RK emitter's coordinate records, never
from names alone:

- a random-effect block's scales, correlation factor and any other coordinate
  of the block besides its standardized draws (`<block>.z`);
- a structured term's own parameters, its draws excepted: an `hsgp`'s
  amplitude, length scales and hyper-predictor coordinates (not its basis
  weights); a `cdar`'s scale and AR coefficient (not its innovations); an
  `rw`'s scale (not its steps); a `car`'s scale and mixing share (not its
  field); a `gmrf`'s scale and lengths (not its states);
- a declared level correlation, `Omega ~ LKJCholesky(K, eta)`;
- a declared scalar whose law is constrained (positive or bounded support,
  such as `sigma ~ Exponential(1)` or `tau ~ LogNormal(0, 1)`), which BRM reads
  as a scale, dispersion, shape or concentration.

Everything else is a latent: population coefficients, declared vectors and
matrices (a matrix of simplex rows included), cell latents, completions,
component locals and a declared scalar on the whole real line
(`b ~ Normal(0, 1)`), which BRM reads as a coefficient. A real-line scalar that
only sets a term parameter (`sd(mu, hsgp(x)) = exp(s)`) is therefore not
listed; name it in `fixed=` of [`brm_fixed_hyper_problem`](@ref).
"""
function brm_hyperparameters(rk::RKBRMI)
    names = Vector{Symbol}(_rk_layout_coordinate_names(rk))
    transforms = _rk_layout_coordinate_transforms(rk)
    length(transforms) == length(names) || error(
        "brm_hyperparameters: internal: $(length(transforms)) layout transforms " *
        "for $(length(names)) coordinates")
    hyper = falses(length(names))
    for record in _brm_nlme_records(rk.plan)
        _brm_mark_hyperparameters!(hyper, names, transforms, record)
    end
    names[hyper]
end

# Mark the coordinates `declaration` owns, without those owned by `latents`.
function _brm_mark_owned_except!(hyper, names, declaration::Symbol, latents)
    for (i, name) in enumerate(names)
        _brm_nlme_owned(name, declaration) || continue
        any(latent -> _brm_nlme_owned(name, latent), latents) && continue
        hyper[i] = true
    end
    hyper
end

_brm_mark_hyperparameters!(hyper, names, transforms, record) =
    _brm_mark_hyperparameters!(hyper, names, transforms, Val(record.kind), record)

# Every other record kind holds latents only.
_brm_mark_hyperparameters!(hyper, names, transforms, ::Val, record) = hyper

# A random-effect block `<block>`: `<block>.z` are its standardized draws; its
# scales `<block>.tau…`, factor `<block>.L` and mixing weights are not.
function _brm_mark_hyperparameters!(hyper, names, transforms,
        ::Union{Val{:ranef},Val{:stratified_ranef}}, record)
    z = String(record.z)
    dot = findlast('.', z)
    dot === nothing && error(
        "brm_hyperparameters: internal: random-effect draws `$z` have no block prefix")
    _brm_mark_owned_except!(hyper, names, Symbol(z[1:prevind(z, dot)]), (record.z,))
end

function _brm_mark_hyperparameters!(hyper, names, transforms, ::Val{:hsgp}, record)
    weights = record.grouped ? record.locals.weights : _BRM_HSGP_UNGROUPED_WEIGHTS
    _brm_mark_owned_except!(hyper, names, record.declaration,
        (Symbol(record.declaration, ".", weights),))
end

# A structured term whose draws are named in its locals.
function _brm_mark_hyperparameters!(hyper, names, transforms,
        kind::Union{Val{:cdar},Val{:rw},Val{:car},Val{:gmrf},Val{:gmrf_footprint},
                    Val{:gmrf_multiresolution}}, record)
    locals = _brm_term_draw_locals(kind, record)
    _brm_mark_owned_except!(hyper, names, record.declaration,
        Tuple(Symbol(record.declaration, ".", l) for l in locals))
end

_brm_term_draw_locals(::Val{:cdar}, record) = (record.locals.eta,)
function _brm_term_draw_locals(::Val{:rw}, record)
    record.sum_to_zero || return (record.locals.z,)
    (record.order == 2 ? record.locals.basis : record.locals.phi,)
end
_brm_term_draw_locals(::Val{:car}, record) =
    Tuple(getproperty(record.locals, l) for l in (:phi, :theta, :z)
        if hasproperty(record.locals, l) && getproperty(record.locals, l) !== nothing)
_brm_term_draw_locals(::Union{Val{:gmrf},Val{:gmrf_footprint},Val{:gmrf_multiresolution}},
        record) = (record.locals.field,)

_brm_mark_hyperparameters!(hyper, names, transforms, ::Val{:lkj_cholesky}, record) =
    _brm_mark_owned_except!(hyper, names, record.declaration, ())

function _brm_mark_hyperparameters!(hyper, names, transforms, ::Val{:scalar}, record)
    for (i, name) in enumerate(names)
        name === record.declaration && transforms[i] !== :identity && (hyper[i] = true)
    end
    hyper
end

"""
    BRMFixedHyperProblem
    brm_fixed_hyper_problem(rk::RKBRMI, point; ad_backend,
        fixed = brm_hyperparameters(rk))

The full RK target of `rk` with the coordinates `fixed` held at `point` and
every other coordinate free: the returned problem implements the
LogDensityProblems interface over the free coordinates. Its log density at a
free point `x` is the target's at `point` with the free coordinates replaced by
`x`, the conditional posterior of the latents given the hyperparameters up to a
constant; its gradient is the target's restricted to the free coordinates.

- `point`: a packed RK unconstrained point of `rk` (`rk_logdensity_problem`'s
  order), or a [`brm_svi_fit`](@ref) result of the same model, read as the
  mean of its packed draws.
- `fixed`: coordinate names or packed indices. A name that is no coordinate
  but owns some (`:Omega`, a block's `Symbol("b_s.tau")`) stands for all of
  them. The default is [`brm_hyperparameters`](@ref)`(rk)`; edit that vector
  to fix more or fewer. An empty `fixed` frees every coordinate: the full
  model, started from `point`.
- `ad_backend`: as for [`rk_logdensity_problem`](@ref), for example
  reverse-mode `AutoEnzyme`.

Fields: `backend`, `target` (the full problem), `point` (the complete point),
`free` and `fixed` (packed indices), `names` (the free coordinates' names),
`fixed_names`, and `position` (the initial free point, `point[free]`).
[`brm_fixed_hyper_point`](@ref) completes free points; `deepcopy` gives a copy
its own AD workspace and shares the backend.
"""
struct BRMFixedHyperProblem{B,P}
    backend::B
    target::P
    point::Vector{Float64}
    free::Vector{Int}
    fixed::Vector{Int}
    names::Vector{Symbol}
    fixed_names::Vector{Symbol}
    position::Vector{Float64}
end

function Base.show(io::IO, p::BRMFixedHyperProblem)
    print(io, "BRMFixedHyperProblem(", length(p.free), " free, ",
        length(p.fixed), " fixed coordinates)")
end

function Base.deepcopy_internal(p::BRMFixedHyperProblem, stack::IdDict)
    haskey(stack, p) && return stack[p]
    # The backend is read-only provenance; the target owns the AD workspace.
    copied = BRMFixedHyperProblem(p.backend, Base.deepcopy_internal(p.target, stack),
        copy(p.point), copy(p.free), copy(p.fixed), copy(p.names),
        copy(p.fixed_names), copy(p.position))
    stack[p] = copied
    copied
end

# An SVI fit: a NamedTuple with packed `draws` and their `coordinates`.
function _brm_fixed_hyper_start(caller, names, fit::NamedTuple)
    (hasproperty(fit, :draws) && hasproperty(fit, :coordinates) &&
        hasproperty(fit, :diagnostics)) || throw(ArgumentError(
        "$caller: a NamedTuple start must be a brm_svi_fit result " *
        "(with `draws`, `coordinates` and `diagnostics`)"))
    Symbol.(fit.coordinates) == names || throw(ArgumentError(
        "$caller: the SVI fit's coordinates are not this model's packed " *
        "coordinates; fit the same RKBRMI"))
    vec(Statistics.mean(fit.draws; dims=2))
end
_brm_fixed_hyper_start(caller, names, point::AbstractVector{<:Real}) =
    Vector{Float64}(point)
_brm_fixed_hyper_start(caller, names, x) = throw(ArgumentError(
    "$caller: the start is a $(typeof(x)); pass a packed point or a brm_svi_fit result"))

function _brm_fixed_hyper_indices(caller, names, fixed)
    position = Dict(n => i for (i, n) in enumerate(names))
    out = Int[]
    for f in fixed
        if f isa Integer
            1 <= f <= length(names) || throw(ArgumentError(
                "$caller: fixed index $f is outside 1:$(length(names))"))
            push!(out, f)
        elseif haskey(position, Symbol(f))
            push!(out, position[Symbol(f)])
        else
            owned = findall(n -> _brm_nlme_owned(n, Symbol(f)), names)
            isempty(owned) && throw(ArgumentError(
                "$caller: `$f` in `fixed` is no coordinate of this model and owns none"))
            append!(out, owned)
        end
    end
    sort!(unique!(out))
end

"""
    brm_fixed_hyper_problem(rk, start; ad_backend, fixed=brm_hyperparameters(rk))

Prepare a [`BRMFixedHyperProblem`](@ref) at a packed point or SVI fit.
"""
function brm_fixed_hyper_problem(rk::RKBRMI, start;
        ad_backend, fixed=brm_hyperparameters(rk))
    caller = "brm_fixed_hyper_problem"
    names = Vector{Symbol}(_rk_layout_coordinate_names(rk))
    point = _brm_fixed_hyper_start(caller, names, start)
    length(point) == length(names) || throw(DimensionMismatch(
        "$caller: the point has $(length(point)) coordinates for a model with " *
        "$(length(names))"))
    all(isfinite, point) || throw(ArgumentError("$caller: the point must be finite"))
    fixed_idx = _brm_fixed_hyper_indices(caller, names, fixed)
    free = setdiff(eachindex(names), fixed_idx)
    isempty(free) && throw(ArgumentError(
        "$caller: `fixed` holds every coordinate; there is nothing to sample"))
    target = rk_logdensity_problem(rk; ad_backend, u0=copy(point))
    LogDensityProblems.dimension(target) == length(names) || error(
        "$caller: internal: the target has $(LogDensityProblems.dimension(target)) " *
        "coordinates for $(length(names)) layout names")
    value = LogDensityProblems.logdensity(target, point)
    isfinite(value) || throw(ArgumentError(
        "$caller: the log density at the point is $value"))
    BRMFixedHyperProblem(rk, target, point, collect(free), fixed_idx, names[free],
        names[fixed_idx], point[free])
end

"""
    brm_fixed_hyper_point(problem::BRMFixedHyperProblem, x)

The complete packed point for the free point `x` (`problem.point` with
`problem.free` replaced), or one complete point per column of a matrix of free
draws. The result is fresh; `x` is not mutated.
"""
function brm_fixed_hyper_point(p::BRMFixedHyperProblem, x::AbstractVector)
    length(x) == length(p.free) || throw(DimensionMismatch(
        "brm_fixed_hyper_point: $(length(x)) free coordinates, expected $(length(p.free))"))
    u = copy(p.point)
    u[p.free] = x
    u
end

function brm_fixed_hyper_point(p::BRMFixedHyperProblem, X::AbstractMatrix)
    size(X, 1) == length(p.free) || throw(DimensionMismatch(
        "brm_fixed_hyper_point: $(size(X, 1)) free coordinates per draw, expected " *
        "$(length(p.free))"))
    U = repeat(p.point, 1, size(X, 2))
    U[p.free, :] = X
    U
end

LogDensityProblems.capabilities(::Type{<:BRMFixedHyperProblem{B,P}}) where {B,P} =
    LogDensityProblems.capabilities(P)
LogDensityProblems.dimension(p::BRMFixedHyperProblem) = length(p.free)
LogDensityProblems.logdensity(p::BRMFixedHyperProblem, x) =
    LogDensityProblems.logdensity(p.target, brm_fixed_hyper_point(p, x))
function LogDensityProblems.logdensity_and_gradient(p::BRMFixedHyperProblem, x)
    value, gradient = LogDensityProblems.logdensity_and_gradient(p.target,
        brm_fixed_hyper_point(p, x))
    value, gradient[p.free]
end

"""
    brm_fixed_hyper_fit(sampler, rk::RKBRMI, start; ad_backend, device=:cpu,
        fixed = brm_hyperparameters(rk), rng = Random.default_rng())
        -> (; draws, problem, point, fixed)

MCMC over every latent of `rk` with the hyperparameters held at `start`, an
SVI fit ([`brm_svi_fit`](@ref), read as the mean of its draws) or a packed
point: the "fixed hyperparameters into MCMC" step after SVI. `sampler =
(problem, init, rng) -> draws` (free coordinates × draws) samples the
[`brm_fixed_hyper_problem`](@ref) from its `position`, as
`brm_ladder_fit`'s sampler does, for example
`(p, init, rng) -> Matrix(adaptive_warmup_mcmc(rng, p; init).posterior_position)`
with WarmupHMC.

`draws` are complete packed RK points, one per column, with the fixed
coordinates at their values, so every reader of packed RK draws takes them
(`rk_restore_draws`, the coordinate transport to SBBRMI). `fixed = ()` samples
the full model from the start instead.

`device=:cpu` calls the Julia sampler. With `device=:gpu`, pass a sampler
whose runtime implements [`BayesianRegressionModels.brm_fixed_hyper_sample`](@ref).
The GPU fit includes `execution.device`, `execution.backend` and
`execution.device_name`.
"""
function brm_fixed_hyper_fit(sampler, rk::RKBRMI, start; ad_backend, device=:cpu,
        fixed=brm_hyperparameters(rk), rng=Random.default_rng())
    device === :cpu || device === :gpu || throw(ArgumentError(
        "brm_fixed_hyper_fit: device must be :cpu or :gpu, got $(repr(device))"))
    problem = brm_fixed_hyper_problem(rk, start; ad_backend, fixed)
    execution = if device === :gpu
        result = brm_fixed_hyper_sample(Val(:gpu), sampler, problem, rng)
        result isa NamedTuple && get(result, :draws, nothing) isa AbstractMatrix || throw(ArgumentError(
            "brm_fixed_hyper_sample must return a free-coordinate draw matrix and its device information"))
        X = result.draws
        _brm_gpu_execution(Base.structdiff(result, (; draws=nothing)))
    else
        X = sampler(problem, copy(problem.position), rng)
        nothing
    end
    size(X, 1) == length(problem.free) || throw(DimensionMismatch(
        "brm_fixed_hyper_fit: the sampler returned $(size(X, 1)) rows for " *
        "$(length(problem.free)) free coordinates"))
    fit = (; draws=brm_fixed_hyper_point(problem, X), problem, point=copy(problem.point),
       fixed=copy(problem.fixed_names))
    execution === nothing ? fit : merge(fit, (; execution))
end

"""
    BayesianRegressionModels.brm_fixed_hyper_sample(::Val{:gpu}, sampler, problem, rng)

A GPU sampler implements this method for its own sampler type. Use
`problem.backend` to export the full data log density. At each evaluation insert
the free coordinates into `problem.point` at `problem.free` and differentiate
with respect to the free vector. Initialize at `problem.position`, including
when every coordinate is free. The result is a named tuple containing
`draws` (free coordinates × draws on the host), `device=:gpu`, `backend`,
and `device_name`. Additional sampler diagnostics may accompany these fields.
BRM restores complete packed draws and records the device information under
`execution`. Native Julia sampler callbacks keep their usual three arguments.
"""
brm_fixed_hyper_sample(::Val, sampler, problem, rng) = throw(ArgumentError(
    "brm_fixed_hyper_fit: device=:gpu requires a GPU sampling method for $(typeof(sampler))"))
