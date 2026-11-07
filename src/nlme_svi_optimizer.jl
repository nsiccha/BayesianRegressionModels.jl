# src/nlme_svi_optimizer.jl — checkpoints of a minibatch SVI run in progress,
# and their continuation on the same target.
#
# A checkpoint is a detached copy of the run (`_BRMSVIRun`, src/nlme_svi.jl)
# plus what it needs to be continued exactly: the controls it runs with, the
# streaming carry its target includes (for a window fit), and the identity of
# that target. It holds no compiled model, AD workspace or observations; resume
# rebuilds those from the model the caller passes again, compares its identity
# and only then continues the loop. This is not the streaming carry
# (`BRMSVIState`): that turns a fitted approximation into the prior of a new
# window, whose observations then enter once.

# Format 2 (0.2.238): a dense unit's slot of the polishing average holds its
# averaged covariance; format 1 held its averaged precision
# (`_brm_svi_upgrade_optimizer_state`).
const _BRM_SVI_OPTIMIZER_FORMAT = 2

"""
    BRMSVIOptimizerState
    brm_svi_optimizer_checkpoint(fit)

The state of a [`brm_svi_fit`](@ref) run after a completed iteration, for
continuing the same optimization with [`brm_svi_resume`](@ref), in this or
another process. For the minibatch engine it holds the raw variational factors
(also while polishing, next to the running average the result reports), the
global Adam moments and step count, the random stream, the iteration, the
ELBO trace and window statistics of the stopping rule and the polishing
state, the weighted sampler, per-unit visits, cached entropy, lazy-average
counters and structured-unit Adam moments. For `laplace=GaussianBlockLaplace()` it holds the global factor, Adam,
the ELBO so far, the averaging window of global iterates, the random stream
and a separate `draw_rng`, and the call counts and numerical records. Both
hold the optimizer controls, the streaming carry of a window fit, and the
identity of the target (the emitted program and callables, the bound data and
frozen preprocessing, the packed coordinates, which `mi(...)` completions are
non-centered, the subject groupings and coordinate ownership, and the fixed
shared values). It holds no observations,
compiled model or AD workspace. Its size is that of the variational factors
(twice while polishing; plus the averaging window for GaussianBlockLaplace)
plus one ELBO value per iteration so far; a run keeps no history of
checkpoints.

`brm_svi_optimizer_checkpoint(fit)` copies the state from the result of a
`brm_svi_fit` or `brm_svi_resume`; the `save` argument of those functions
receives such copies while they run. Persist it with Julia
`Serialization` to a trusted file; loading it needs the same Julia, package
and model code versions. Write it atomically, so that an abrupt termination
leaves the last complete checkpoint in place:

```julia
using Serialization
function save_checkpoint(path, state)
    temporary = path * ".partial"
    open(io -> serialize(io, state), temporary, "w")
    mv(temporary, path; force=true)
end
```

A checkpoint written by BRM 0.2.237 or earlier still resumes. One taken while
polishing keeps, for the iterates it had averaged, the narrower average of dense
units' precision those versions returned; later iterates are averaged in
covariance.

The streaming carry [`brm_svi_checkpoint`](@ref) is a different object: it
continues on a new, disjoint window and holds no optimizer state.
"""
struct BRMSVIOptimizerState
    format::Int
    identity::NamedTuple
    previous::Union{Nothing,BRMSVIState,BRMSVILaplaceState}
    controls::NamedTuple
    run::Union{_BRMSVIRun,_BRMSVILaplaceRun}
end

function Base.show(io::IO, s::BRMSVIOptimizerState)
    run, c = s.run, s.controls
    window = s.previous === nothing ? "" : ", window update"
    run isa _BRMSVILaplaceRun && return print(io, "BRMSVIOptimizerState(",
        "GaussianBlockLaplace, iteration ", run.iteration, " of ", c.max_iterations, window, ")")
    phase = run.converged ? "converged" :
        run.iteration >= c.max_iterations ? "reached max_iterations" :
        _brm_svi_polishing(run) ?
            "polishing $(run.iteration - run.polish_start)/$(c.polish_windows * c.window)" :
            "exploring, window $(length(run.windows))"
    print(io, "BRMSVIOptimizerState(iteration ", run.iteration, " of at most ",
        c.max_iterations, ", ", phase, window, ")")
end

const _BRM_SVI_BINDING_TAG = "brm-svi-v1"

# What the run's target and coordinates depend on besides its streaming carry.
function _brm_svi_identity(t)
    m = t.m
    estimated = Set(vcat(t.globals, reduce(vcat, t.units; init=Int[])))
    fixed = [(m.view.coordinates[i], m.unit[i]) for i in eachindex(m.unit) if !(i in estimated)]
    (; binding=_brm_rk_binding_identity(m.backend, _BRM_SVI_BINDING_TAG),
       binding_parts=_brm_rk_binding_parts(m.backend),
       completions=_rk_completion_mode(m.backend.plan),
       grouping=(; group=m.view.group, nested=Tuple(keys(m.view.nested_groups)),
           shared=Tuple(m.view.shared_groups), unit_partition=m.view.unit_partition),
       ownership=_brm_value_digest((hasproperty(t, :ownership) ? _brm_svi_unit_levels(t) : m.view.levels, t.globals, t.units)),
       fixed=_brm_value_digest(fixed),
       global_coordinates=hasproperty(t, :ownership) ?
           copy(m.view.coordinates[_brm_svi_requested_globals(t)]) : Symbol[],
       separator=hasproperty(t, :ownership) ? _brm_svi_separator_options(t) : (; separator=:none),
       field_states=get(t, :fields, nothing) === nothing ? :global : :units)
end

_brm_svi_optimizer_state(t, previous, c, run) = BRMSVIOptimizerState(
    _BRM_SVI_OPTIMIZER_FORMAT, _brm_svi_identity(t), previous, c, run)

# A copy that shares nothing with the running fit. `deepcopy` does not copy
# the task-local RNG's state; `copy` does (as a `Xoshiro` with the same stream).
function _brm_svi_detach(s::BRMSVIOptimizerState)
    d = deepcopy(s)
    d.run.rng = copy(s.run.rng)
    s.run isa _BRMSVILaplaceRun && s.run.draw_rng !== nothing &&
        (d.run.draw_rng = copy(s.run.draw_rng))
    d
end

"""
    brm_svi_optimizer_checkpoint(fit) -> BRMSVIOptimizerState

A copy, sharing nothing with `fit`, of the optimizer state of a
[`brm_svi_fit`](@ref) or [`brm_svi_resume`](@ref) result (or of a
[`BRMSVIOptimizerState`](@ref)), for continuing it with [`brm_svi_resume`](@ref).
"""
function brm_svi_optimizer_checkpoint(fit)
    hasproperty(fit, :diagnostics) && hasproperty(fit.diagnostics, :optimizer) ||
        throw(ArgumentError("brm_svi_optimizer_checkpoint: expected the result of " *
            "brm_svi_fit or brm_svi_resume"))
    _brm_svi_detach(fit.diagnostics.optimizer)
end
brm_svi_optimizer_checkpoint(state::BRMSVIOptimizerState) = _brm_svi_detach(state)
brm_svi_optimizer_checkpoint(::Union{BRMSVIState,BRMSVILaplaceState}) = throw(ArgumentError(
    "brm_svi_optimizer_checkpoint: a streaming carry (brm_svi_checkpoint) continues on a " *
    "new window with brm_svi_fit(state, new_window); it holds no optimizer state"))
brm_svi_checkpoint(::BRMSVIOptimizerState) = throw(ArgumentError(
    "brm_svi_checkpoint: an optimizer checkpoint continues the same model and data with " *
    "brm_svi_resume(model, state); a new window takes brm_svi_checkpoint of a fit result"))

# A format-1 checkpoint (before 0.2.238) averaged a dense unit's precision while
# polishing. One taken while polishing continues from the covariance that
# precision average stands for: the iterates averaged before the resume keep
# their narrower average, those after it are averaged in covariance. Before
# polishing the two formats are the same. `state` must be the caller's own copy.
function _brm_svi_upgrade_optimizer_state(state::BRMSVIOptimizerState)
    state.format == 1 || return state
    run = state.run
    run isa _BRMSVIRun && run.average !== nothing &&
        foreach(_brm_svi_unit_average_start!, run.average.Λ)
    BRMSVIOptimizerState(2, state.identity, state.previous, state.controls, run)
end

const _BRM_SVI_IDENTITY_PARTS = (
    grouping="the subject, nested or shared groupings",
    ownership="the subjects or the coordinates each subject owns",
    fixed="the values of the fixed shared coordinates",
    global_coordinates="the requested global-coordinate promotion",
    separator="the separator policy and budgets")

const _BRM_SVI_LOCATED_PROGRAM = " (the checkpoint's program digest predates " *
    "BRM's place-free program identity: it included the file and line each macro " *
    "call, such as an emitted `@kernel`, was loaded from, so the same model loaded " *
    "from another checkout, depot, host or BRM version differs there, and only that " *
    "installation can resume it)"

_brm_svi_identity_refusal(description) = ArgumentError(
    "brm_svi_resume: the model differs from the checkpoint's in $description; " *
    "resume needs the same model, data and preparation options")

function _brm_svi_check_identity(current, saved)
    for (part, description) in pairs(_BRM_SVI_IDENTITY_PARTS)
        isequal(current[part], saved[part]) || throw(_brm_svi_identity_refusal(description))
    end
    # A checkpoint older than `field_states` held every state in the global factor.
    isequal(get(current, :field_states, :global), get(saved, :field_states, :global)) ||
        throw(_brm_svi_identity_refusal("how a read-pair field's states are held"))
    nothing
end

_brm_svi_backend(x::RKBRMI) = x
_brm_svi_backend(m::BRMNLMEModel) = m.backend

# A `@brm` model lowered with its checkpoint's completion parametrization (the
# `completions` its identity records), as `reprocess` keeps a
# backend's: the lowering then reads no grouping option, so the identity is
# compared before any grouping is (preparation checks them). A source the
# checkpoint non-centred that this model does not sample is a different model.
function _brm_svi_checkpoint_lowered(x::BRMI, completions)
    completions isa Symbol && return RKBRMI(x; completions)
    rk = RKBRMI(x)
    missing_sources = setdiff(completions, _rk_completion_sources(rk.plan))
    isempty(missing_sources) || throw(_brm_svi_identity_refusal("its emitted program " *
        "(the checkpoint's model samples the missing entries of " *
        join(("`$s`" for s in sort!(collect(missing_sources))), ", ") *
        ", this one does not)"))
    RKBRMI(x; completions)
end

# Reads only the lowered backend, so a different model is refused before its
# preparation. A checkpoint older than `binding_parts` holds only the combined
# digest, for a generated callable the one of its generated name. Parts before
# format 2 (0.2.228) digest the program with the file and line of each macro
# call, which only the installation that wrote them reproduces.
function _brm_svi_check_binding(rk, saved)
    if hasproperty(saved, :binding_parts)
        parts = saved.binding_parts
        differences = _brm_rk_binding_differences(_brm_rk_binding_parts(rk), parts)
        isempty(differences) && return nothing
        get(parts, :format, 1) < 2 && replace!(differences,
            "its emitted program" => "its emitted program" * _BRM_SVI_LOCATED_PROGRAM)
        throw(_brm_svi_identity_refusal(join(differences, ", ")))
    elseif !any(stable -> _brm_rk_binding_identity(rk, _BRM_SVI_BINDING_TAG; stable) == saved.binding,
            (true, false))
        throw(_brm_svi_identity_refusal("the model's emitted program, its callables, its bound " *
            "data or frozen preprocessing, or its packed coordinates"))
    end
    nothing
end

function _brm_svi_check_run(t, s)
    _brm_svi_consistent(s.run, t, s.controls) || throw(ArgumentError(
        "brm_svi_resume: the checkpoint's optimizer state is inconsistent with its " *
        "target's layout"))
end

function _brm_svi_consistent(run::_BRMSVILaplaceRun, t, c)
    q, D = run.q, length(t.globals)
    n = length(_brm_svi_global_vector(q))
    length(q.μ) == length(q.ρ) == size(q.Β, 1) == D &&
        length(run.adam.m) == length(run.adam.v) == n && size(run.history) == (n, c.width) &&
        length(run.elbo) == run.iteration <= c.max_iterations && length(run.counts) == 5
end

function _brm_svi_consistent(run::_BRMSVIRun, t, c)
    q = run.q
    D, N = length(t.globals), length(t.units)
    factors = (q, something(run.average, q))
    all(f -> length(f.μ) == D && length(f.ρ) == D && size(f.Β, 1) == D &&
            length(f.M) == length(f.Λ) == length(f.U) == length(f.unit_opt) == N &&
            all(i -> length(f.M[i]) == size(f.Λ[i], 1) == size(f.U[i], 1) ==
                length(t.units[i]), 1:N), factors) &&
        length(run.adam.m) == length(run.adam.v) == length(_brm_svi_global_vector(q)) &&
        length(run.trace) == run.iteration && 0 <= run.polish_start <= run.iteration &&
        (run.average === nothing) == (run.polish_start == 0) &&
        length(run.windows) <= run.iteration ÷ c.window &&
        length(run.sampler.probabilities) == length(run.sampling.visits) ==
            length(run.sampling.draws) == length(run.averaged_through) ==
            length(run.unit_entropy) == N &&
        run.sampling.method == c.unit_sampling &&
        run.unit_gradient_calls == 2sum(run.sampling.visits) &&
        sum(run.sampling.draws) == c.batch_size * run.iteration
end

# The resumed run on `t`: the streaming prior from the checkpoint's carry, the
# identity compared, then a run owned by this call.
function _brm_svi_restore(t, s::BRMSVIOptimizerState)
    if s.previous !== nothing
        t, _ = _brm_svi_initialize(t, s.previous, s.previous.q.family;
            start=nothing, inits=nothing, global_scale=1.0, unit_scale=1.0)
    end
    t, _brm_svi_restored(t, s)
end

function _brm_svi_restored(t, s)
    identity = _brm_svi_identity(t)
    _brm_svi_check_identity(identity, s.identity)
    _brm_svi_check_run(t, s)
    BRMSVIOptimizerState(s.format, identity, s.previous, s.controls, s.run)
end

# A GaussianBlockLaplace run continues counting in its new target's containers.
function _brm_svi_laplace_restored(t, s)
    state = _brm_svi_restored(t, s)
    run = state.run
    t.counts .= run.counts
    merge!(empty!(t.numerical), run.numerical)
    run.counts, run.numerical = t.counts, t.numerical
    state
end

const _BRM_SVI_RESUME_MODEL_OPTIONS = (:ad_backend, :shared_point, :subject_group,
    :nested_groups, :shared_groups, :unit_partition)

"""
    brm_svi_resume([save,] x, checkpoint; stop_after=nothing,
        checkpoint_every=nothing, ndraws=1000, progress=nothing,
        snapshot=nothing, snapshot_every=nothing, snapshot_draws=0,
        threads=nothing, ad_backend=nothing, shared_point=nothing,
        subject_group=nothing, nested_groups=NamedTuple(), shared_groups=())
        -> (; draws, coordinates, diagnostics)

Continue a [`brm_svi_fit`](@ref) run on the same target from `checkpoint`, a
[`BRMSVIOptimizerState`](@ref) or a fit/resume result, exactly as the run would
have continued without stopping: the same iterates, random stream, stopping
decision and draws, in the same Julia and package versions, at any `threads`
(the fit's option; a run may continue at another). The run keeps its optimizer controls, its
`max_iterations` horizon and step-size schedule; `stop_after` bounds only this
call. A finished checkpoint does no iterations and returns draws of its
result. A `laplace=GaussianBlockLaplace()` run continues its fixed update
budget with the same method, and its carry and diagnostics (call counts,
numerical records) equal the uninterrupted run's.

`x` is the model the checkpoint was fitted on, rebuilt with the same data:
the `@brm` model ([`BRMI`](@ref)), its [`RKBRMI`](@ref) or a prepared
[`BRMNLMEModel`](@ref), with the same preparation options as the fit
(`ad_backend` may differ). For a checkpoint of a window fit,
`brm_svi_fit(previous, x_new)`, pass `x_new`; the checkpoint carries
`previous`. The emitted program and callables, the bound data and frozen
preprocessing, the packed coordinates, the groupings, the subjects and their
coordinates, and the fixed shared values must equal the checkpoint's; a
difference is refused before any optimization step, and one in the program,
callables, data or coordinates already after lowering, before the model is
prepared. The refusal names the differing part, down to the callable's binding
and the data column. A `@brm` model is lowered with the checkpoint's
parametrization of its `mi(...)` completions (which ones the fit lowered
non-centered), so that comparison reads no grouping option: a different model
is refused by name even when a grouping option is also wrong, and preparation
then checks the groupings. An older checkpoint that records no
parametrization lowers its `@brm` model from the grouping options, which are
then read (and an invalid one refused) before the comparison when the model has
a centred completion. An anonymous function or closure compares by its lowered
code and captured values, so the same source continues in another driver, at
another line or after other definitions; a named function compares by its module
and name. In the same process, pass `fit.diagnostics.model` (or the `BRMNLMEModel`
the fit was given): it prepares neither the model nor its target and worker samplers
again (`diagnostics.target.reused`, [`brm_svi_fit`](@ref)), whereas a `BRMI` or
`RKBRMI` prepares both on every call.
The program compares without the file and line of its macro calls (an emitted
`@kernel` definition), so the same model also continues from another checkout,
depot, host or BRM version. A checkpoint from an earlier version digested those
places, and only the installation that wrote it resumes a program that has one.

`save`, `stop_after`, `checkpoint_every`, `ndraws`, `progress`, `snapshot`,
`snapshot_every` and `snapshot_draws` work as in
[`brm_svi_fit`](@ref); `save` again receives copies of the state. The result
has the fit's shape: its diagnostics cover the whole run (`elbo`, `iterations`,
`unit_gradient_calls`), except `seconds`, which counts this call. A cooperative
interrupt (Treebars) passes the state of the last completed window to `save`
before it propagates; resume from the last checkpoint `save` committed.

```julia
model = builder(data)
fit = brm_svi_fit(model; rng=Xoshiro(1), stop_after=5_000, ndraws=0) do state
    save_checkpoint("svi.jls", state)
end
# later, possibly in a new process:
state = deserialize("svi.jls")
fit = brm_svi_resume(builder(data), state; stop_after=5_000) do state
    save_checkpoint("svi.jls", state)
end
fit.diagnostics.finished
```
"""
function brm_svi_resume end
brm_svi_resume(x, checkpoint; kwargs...) = _brm_svi_resume(nothing, x, checkpoint; kwargs...)
brm_svi_resume(save::Function, x, checkpoint; kwargs...) =
    _brm_svi_resume(save, x, checkpoint; kwargs...)

function _brm_svi_resume(save, x, checkpoint; ndraws=1000, stop_after=nothing,
        checkpoint_every=nothing, progress=nothing, threads=nothing, snapshot=nothing,
        snapshot_every=nothing, snapshot_draws=0, kwargs...)
    unknown = setdiff(keys(kwargs), _BRM_SVI_RESUME_MODEL_OPTIONS)
    isempty(unknown) || throw(ArgumentError("brm_svi_resume: unsupported options " *
        join(string.(unknown), ", ") * "; the checkpoint fixes the optimizer controls, " *
        "the random stream and the start"))
    x isa Union{BRMI,RKBRMI,BRMNLMEModel} || throw(ArgumentError(
        "brm_svi_resume: expected the checkpoint's model rebuilt on its data (a BRMI, " *
        "an RKBRMI or a BRMNLMEModel), got $(typeof(x))"))
    state = _brm_svi_upgrade_optimizer_state(brm_svi_optimizer_checkpoint(checkpoint))
    state.format == _BRM_SVI_OPTIMIZER_FORMAT || throw(ArgumentError(
        "brm_svi_resume: unsupported optimizer checkpoint format $(state.format)"))
    options = (; kwargs...)
    if state.previous !== nothing
        x, options = _brm_svi_window_input(state.previous, x, options; caller="brm_svi_resume")
    end
    # A `@brm` model is lowered as its checkpoint's was; an older checkpoint
    # records no parametrization and is lowered from the grouping options.
    checkpoint_completions = x isa BRMI && hasproperty(state.identity, :completions) ?
        state.identity.completions : nothing
    call = (; caller="brm_svi_resume", ndraws, stop_after, checkpoint_every, progress,
        snapshot, snapshot_every, snapshot_draws, checkpoint_completions,
        check_lowered=lowered -> _brm_svi_check_binding(_brm_svi_backend(lowered), state.identity))
    if state.run isa _BRMSVILaplaceRun
        threads === nothing || throw(ArgumentError(
            "GaussianBlockLaplace: whole-window fixed-budget optimization does not use " *
            "threads; omit this option"))
        return _brm_svi_laplace_drive(save, x, state.previous, state.controls.method;
                call..., description="BRM conditional Gaussian SVI resume", options...) do t, _
            _brm_svi_laplace_restored(t, state)
        end
    end
    if hasproperty(state.identity, :global_coordinates) &&
            state.run isa _BRMSVIRun
        options = merge(state.identity.separator,
            (; global_coordinates=state.identity.global_coordinates,
             field_states=get(state.identity, :field_states, :global)), options)
    end
    _brm_svi_drive(save, x, state.previous; call..., description="BRM SVI resume",
            threads, options...) do t
        _brm_svi_restore(t, state)
    end
end
