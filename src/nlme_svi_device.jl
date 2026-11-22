# The device runtime lives outside BRM. The package continues to own the
# partition, the window transitions, checkpoints, and the returned fit.

"""
    BayesianRegressionModels.brm_svi_device_runner(::Val{:gpu})

Provider interface for `brm_svi_fit(...; device=:gpu)` and `brm_svi_resume`.
A loaded GPU runtime implements this method and returns a callable
`runner(advance, problem, run, controls)`. It checks that the requested device
is available, puts its state on that device, calls `advance(step!, synchronize!)` once and
returns a named tuple `(; device=:gpu, backend, device_name)` identifying the
device used. The fit exposes it as `diagnostics.execution`.

`problem.partition` holds the built RK model, binding, observation names, key
mapping and partition declarations. `problem.bound_data(source)` provides the
inputs of the partition module's `prepare_data` in its declared order.
The `kernels` named tuple contains RK graphs for the global draw, entropy
gradient, variance bound, antithetic pair, work weights, Adam, clipping,
dense-factor triangular solves, curvature, trust bounds and linear coupling.
Exporting these graphs to JAX requires ReactiveKernels 0.1.362 or later.
The labelled station density export requires ReactiveKernelsPPL 0.1.253 or later.
`unit_design` takes the square matrix of normal draws. `design_curvature`
takes the whitened gradient differences and the design, both in columns.
`orthogonal_gap` returns the curvature difference from identity and its
spectral radius. The coupling kernels operate on the coupled rows of a unit.
For a dense unit, `coupling_direction` and `coupling_variances` take the
lower Cholesky factor of the precision block on those rows. The control
variate uses the unit gradient evaluated at the global and unit means.
`coupling_row_update` returns the scale from previous visits, the updated
mean square and the visit count. Use the factor from before the unit step.

For a structured station factor with covariance `Diagonal(exp.(2ρ)) + B*B'`,
split its unit noise into the first `d` entries (`epsilon`) and the remaining
rank entries (`eta`). Use `structured_offset(ρ, B, epsilon, eta)` for the
antithetic offset. Pass the mean and half-difference of the two local target
gradients to `structured_gradient(ρ, B, mean_gradient, difference, epsilon, eta)`.
Pack its parameters as `x = vcat(mean, ρ, vec(B))`.
`structured_adam(ρ, B, x, gradient, moment, variance, count, beta1, beta2,
rate, rank_step, clip)` returns the proposed parameters, next moments, next
visit count and a Boolean clip indicator. Use the unit's own Adam state, the phase controls
`structured_step` and `structured_rank`, and `problem.global_clip`. Apply `structured_trust(ρ, B, x, proposed)`
before unpacking. For structured coupling, pass the selected rows of `ρ`
and `B` to `structured_direction`, `structured_variances` and
`structured_metric`. Their direction and squared-distance operands use
the same row ordering. The structured unit's entropy contribution is
`global_logdet(ρ, B) / 2`.

For the ELBO, `unit_entropy(root)` is the unit's entropy contribution and
`global_logdet(log_scale, loading) / 2` is the globals' contribution, both
up to constants. Polishing initializes a dense unit's covariance average with
`unit_covariance(precision)`. Use `weighted_average(previous, current, weight)`
for lazy unit means, covariances and coupling loadings, and
`running_average(previous, current, count)` for the global factors.

A stationarity slot contains the mean followed by marginal log standard
deviations. For globals use `marginal_logsd(log_scale, loading_row)` per row.
A dense unit starts from `dense_logsd(precision)`. For each coupled row, apply
`marginal_logsd` to its conditional log width and the matching coupling row.
Append the unit's `centering_control(x)` values and its coupling loadings in
Julia column-major order. Record a unit only on its visits.

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

`problem.iteration_draws(run, controls, phase, n; rng=run.rng)` draws `n`
iterations in native order. Each contains `batch` (indices, weights and
multiplicities), the global `epsilon` and `eta`, `field`, `units`,
`reoptimization` and the base `unit_gradient_calls`. A unit vector is
the noise for a pair. A unit matrix contains the raw normals for `unit_design`.
Extra re-optimization designs follow all ordinary unit draws.
Field draws contain reader indices and aligned noise, plus owner visits for a
field sweep. Negative reader indices carry a saved unit draw before its
conditional shift. A field sweep counts its additional evaluated neighbor terms
separately. Only the supplied RNG advances. For lookahead, pass a copy and
commit its state with the completed iterations before publishing a checkpoint.
Generate bounded chunks within a long window.

`problem.iteration_controls(run, controls, phase, it)` returns the native
`lr`, `β` and iteration options for the absolute one-based iteration `it`.
It reads the saved polishing start and legacy defaults. `sd_bound` is the
multiplier of the prior standard deviations.

`step!(n, phase)` advances exactly `n` numerical iterations in the given
`:exploring`, `:settling` or `:polishing` phase. On return it updates
`run.iteration` and appends one ELBO per iteration to `run.trace`.
`synchronize!(:histories)` transfers the iterate histories used to assess
drift and stationarity. `synchronize!(:state)` transfers the complete current
state before a phase change, save, snapshot, or return. It includes the
factors, moments, sampling counts, random stream, entropy and polishing
average. A repeated synchronization at the same iteration must preserve any
updates BRM has since made to the host state. Providers that already update
the complete host run in `step!` can call `advance(step!)`.
Partitioning calls to `step!` differently must preserve the
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
    iteration_draws(run, c, phase, n; rng=run.rng) = _brm_svi_iteration_draws(t, run, c, phase, n; rng)
    (; backend=t.m.backend, partition, bound_data, factor_buckets, selections...,
        iteration_draws, iteration_controls=_brm_svi_iteration_controls,
        kernels=_rk_svi_kernels(), global_clip=_BRM_SVI_GLOBAL_CLIP,
        unit_trust=_BRM_SVI_UNIT_TRUST, unit_mean_trust=_BRM_SVI_UNIT_MEAN_TRUST,
        coupling_trust=_BRM_SVI_COUPLING_TRUST,
        expected_layout=t.m.backend.model.layout, owner,
        levels=_brm_svi_unit_levels(t), observation_units=t.plan.labels,
        point=t.m.unit, coordinates=t.m.view.coordinates, t.globals, t.units,
        t.touched, t.prior_mean, t.sd_prior, centering=_brm_svi_centering(t),
        t.fields, field_sweep=get(t, :field_sweep, nothing), t.previous)
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
    fit!(nothing, identity)
    (; device=:cpu, backend=:julia)
end

function _brm_svi_execute!(fit!, runner, target, run, controls)
    advanced = false
    function advance(step!, synchronize! = identity)
        advanced && error("GPU fitting runtime called advance more than once")
        step! === nothing && error("GPU fitting runtime supplied no iteration method")
        fit!(step!, synchronize!)
        advanced = true
        nothing
    end
    execution = runner(advance, _brm_svi_device_problem(target), run, controls)
    advanced || error("GPU fitting runtime returned before advancing the fit")
    _brm_gpu_execution(execution)
end
