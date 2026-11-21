# The device runtime lives outside BRM. The package continues to own the
# partition, the window transitions, checkpoints, and the returned fit.

"""
    BayesianRegressionModels.brm_svi_device_runner(::Val{:gpu})

Provider interface for `brm_svi_fit(...; device=:gpu)` and `brm_svi_resume`.
A loaded GPU runtime implements this method and returns a callable
`runner(advance, problem, run, controls)`. It checks that the requested device
is available, puts its state on that device, calls `advance(step!)` once and
returns a named tuple `(; device=:gpu, backend, device_name)` identifying the
device used. The fit exposes it as `diagnostics.execution`.

`problem.partition` holds the built RK model, binding, observation names, key
mapping and partition declarations. `problem.bound_data(source)` provides the
inputs of the partition module's `prepare_data` in its declared order.
`global_factors` and the unit-indexed `unit_factors` are selections on
the model's original factor catalogue. `expected_layout` is that model's
packed layout. Their underlying events are in `factor_buckets[1]` for global
prior/Jacobian events; `factor_buckets[i+1]` contains unit `i`'s events.
`point` is the packed base point, `globals` and `units` are its one-based
coordinate indices, and `touched[i]` indexes the globals unit `i` reads.
`owner[j]` is the unit index for packed coordinate `j` (zero for shared).
`levels[i]` names unit `i`. `observation_units` contains the integer unit
indices with observations. Units outside this set can still own prior factors.
The remaining fields carry the native centering, field and streaming plans
and the prior projection. All problem fields are read-only.

`step!(n, phase)` advances exactly `n` numerical iterations in the given
`:exploring`, `:settling` or `:polishing` phase. It synchronizes `run` at
return, including the raw factors, optimizer moments, sampling counts, random
stream, entropy, iterate histories, polishing average, iteration and one ELBO
per iteration. Partitioning calls to `step!` differently must preserve the
random stream and the updates. BRM owns window summaries, phase transitions,
stopping, checkpoint/snapshot callbacks and the final packed draws. A phase
transition may update the host factors, moments or averaging state before
the next `step!`. A provider releases its resources even if a callback throws.
"""
brm_svi_device_runner(::Val) = throw(ArgumentError(
    "brm_svi_fit: device=:gpu needs a loaded GPU fitting runtime implementing brm_svi_device_runner"))

function _brm_svi_runner(device)
    device === :cpu && return nothing
    device === :gpu || throw(ArgumentError("brm_svi_fit: device must be :cpu or :gpu, got $(repr(device))"))
    runner = brm_svi_device_runner(Val(:gpu))
    runner === nothing && error("brm_svi_device_runner returned no GPU executor")
    runner
end

_brm_svi_laplace_device(device) = device === :cpu || throw(ArgumentError(
    "GaussianBlockLaplace supports device=:cpu; device=:gpu selects the minibatch SVI method"))

function _brm_svi_device_problem(t)
    (; owner, factors) = t.ownership
    factor_buckets = _rk_svi_factor_buckets(owner, length(t.units), factors)
    selections = _rk_svi_device_selections(factors, factor_buckets)
    partition = t.partition.context
    bound_data = source -> _rk_jax_bound_data(partition.bound, source)
    (; backend=t.m.backend, partition, bound_data, factor_buckets, selections...,
        kernels=_rk_svi_kernels(), expected_layout=t.m.backend.model.layout, owner,
        levels=_brm_svi_unit_levels(t), observation_units=t.plan.labels,
        point=t.m.unit, coordinates=t.m.view.coordinates, t.globals, t.units,
        t.touched, t.prior_mean, t.sd_prior, centering=_brm_svi_centering(t),
        t.fields, t.previous)
end

function _rk_jax_bound_data end
function _rk_svi_device_selections end
function _rk_svi_kernels end

function _brm_gpu_execution(info)
    info isa NamedTuple && get(info, :device, nothing) === :gpu &&
        get(info, :backend, nothing) isa Symbol &&
        get(info, :device_name, nothing) isa AbstractString && !isempty(info.device_name) ||
        throw(ArgumentError("GPU fitting runtime must identify its accelerator with (; device=:gpu, backend, device_name)"))
    info
end

function _brm_svi_execute!(fit!, ::Nothing, target, run, controls)
    fit!(nothing)
    (; device=:cpu, backend=:julia)
end

function _brm_svi_execute!(fit!, runner, target, run, controls)
    advanced = false
    execution = runner(_brm_svi_device_problem(target), run, controls) do step!
        advanced && error("GPU fitting runtime called advance more than once")
        step! === nothing && error("GPU fitting runtime supplied no iteration method")
        fit!(step!)
        advanced = true
        nothing
    end
    advanced || error("GPU fitting runtime returned before advancing the fit")
    _brm_gpu_execution(execution)
end
