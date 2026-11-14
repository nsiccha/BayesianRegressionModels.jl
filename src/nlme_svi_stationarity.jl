# Iterate diagnostics, not diagnostics of draws from the fitted approximation.
# Dhaka et al. (2020), arxiv:2009.00666; Welandawe et al. (2024), JMLR 25(219),
# section 5.1: search recent windows for split-Rhat <= 1.1.
# Reuse MCMCDiagnosticTools' rank-normalized/folded split-Rhat. Report bulk ESS
# separately: the paper's ESS floor belongs to precision of the subsequent
# iterate average (section 5.2), not the stationarity transition itself.
#
# Each block has its own update clock. A unit that was not selected contributes
# no observation: repeated copies of an unvisited unit cannot establish stability.
# Dyadic thinning keeps at most 512 observations per moment while extending the
# time span. This is bounded-memory storage, not a change to the optimizer stream.
const _BRM_SVI_STATIONARITY_CAPACITY = 512
# At least 100 observations in each half of the split. Shorter candidate
# windows can mistake a slow autocorrelated drift for a stationary segment.
const _BRM_SVI_STATIONARITY_MIN = 200
const _BRM_SVI_STATIONARITY_RHAT = 1.1

mutable struct _BRMSVIIterates
    values::Matrix{Float64}
    count::Int
    updates::Int
    stride::Int
end
_BRMSVIIterates(d::Int) = _BRMSVIIterates(zeros(d, 0), 0, 0, 1)
Base.isequal(a::_BRMSVIIterates, b::_BRMSVIIterates) =
    all(n -> isequal(getfield(a, n), getfield(b, n)), fieldnames(_BRMSVIIterates))

# Return a writable sample slot, or nothing when this update is thinned out.
function _brm_svi_iterate_slot!(h::_BRMSVIIterates)
    h.updates += 1
    h.updates % h.stride == 0 || return nothing
    if h.count == _BRM_SVI_STATIONARITY_CAPACITY
        for j in 1:h.count÷2
            h.values[:, j] .= view(h.values, :, 2j)
        end
        h.count ÷= 2
        h.stride *= 2
        h.updates % h.stride == 0 || return nothing
    end
    if h.count == size(h.values, 2)
        n = min(_BRM_SVI_STATIONARITY_CAPACITY, max(16, 2h.count))
        # Keep unused capacity deterministic too: checkpoints compare and serialize
        # the whole buffer, including slots that have not been observed yet.
        values = zeros(size(h.values, 1), n)
        values[:, 1:h.count] .= view(h.values, :, 1:h.count)
        h.values = values
    end
    h.count += 1
    view(h.values, :, h.count)
end

_brm_svi_stationarity_enabled(c) = get(c, :stationarity, false)
_brm_svi_stationarity_state(run) = get(run.sampling, :stationarity, nothing)
_brm_svi_stationarity_empty() = (;
    passed=false, reason=:insufficient_history, rhat=nothing, ess=nothing,
    samples=0, updates=0, stride=1)

mutable struct _BRMSVIStationarity
    globals::_BRMSVIIterates
    units::Vector{_BRMSVIIterates}
    explored::Int
    settled::Bool
    next_check::Int
    checks::Int
    report::NamedTuple
    exploration::Union{Nothing,NamedTuple}
end
Base.isequal(a::_BRMSVIStationarity, b::_BRMSVIStationarity) =
    all(n -> isequal(getfield(a, n), getfield(b, n)), fieldnames(_BRMSVIStationarity))

_brm_svi_centering_dimension(q, i) = hasproperty(q, :centering) ? length(q.centering.x[i]) : 0
_brm_svi_coupling_dimension(q, i) =
    _brm_svi_coupling(q) === nothing ? 0 : length(q.coupling.V[i])
_brm_svi_unit_history_dimension(q, i) = 2length(q.M[i]) +
    _brm_svi_centering_dimension(q, i) + _brm_svi_coupling_dimension(q, i)
function _brm_svi_stationarity_state(q, c)
    _BRMSVIStationarity(_BRMSVIIterates(2length(q.μ)),
        [_BRMSVIIterates(_brm_svi_unit_history_dimension(q, i))
            for i in eachindex(q.M)],
        0, false, c.window * max(c.min_windows, c.patience + 1), 0,
        _brm_svi_stationarity_empty(), nothing)
end

function _brm_svi_record_global!(h, q)
    slot = _brm_svi_iterate_slot!(h)
    slot === nothing && return
    d = length(q.μ)
    slot[1:d] .= q.μ
    for j in 1:d
        # Marginal log SD: invariant to rotations of the low-rank loadings.
        slot[d+j] = log(hypot(exp(q.ρ[j]), norm(view(q.Β, j, :))))
    end
end

_brm_svi_marginal_logsd(Λ::AbstractMatrix) = log.(diag(_brm_svi_unit_covariance(Λ))) ./ 2
function _brm_svi_marginal_logsd(Λ::_BRMSVIUnitPrecision)
    f = Λ.factor
    [log(hypot(exp(f.ρ[j]), norm(view(f.Β, j, :)))) for j in eachindex(f.ρ)]
end
function _brm_svi_record_unit!(h, q, i)
    slot = _brm_svi_iterate_slot!(h)
    slot === nothing && return
    d = length(q.M[i])
    slot[1:d] .= q.M[i]
    slot[d+1:2d] .= _brm_svi_marginal_logsd(q.Λ[i])
    nc = _brm_svi_centering_dimension(q, i)
    if hasproperty(q, :centering)
        slot[2d+1:2d+nc] .= _brm_svi_unit_controls(q, i)
    end
    coupling = _brm_svi_coupling(q)
    if coupling !== nothing
        V = coupling.V[i]
        for (k, j) in enumerate(coupling.rows[i])
            slot[d+j] = log(hypot(exp(slot[d+j]), norm(view(V, k, :))))
        end
        # Stable marginal widths can conceal a moving global-unit dependence.
        # Keep the represented coupling coefficients too, on this unit's clock.
        slot[2d+nc+1:end] .= vec(V)
    end
end

function _brm_svi_record_iterates!(run, indices, c)
    _brm_svi_stationarity_enabled(c) || return
    h = _brm_svi_stationarity_state(run)
    _brm_svi_record_global!(h.globals, run.q)
    foreach(i -> _brm_svi_record_unit!(h.units[i], run.q, i), indices)
end

# Compute in bounded coordinate chunks: rank normalization must not allocate a
# second whole-model history. Constants are stable only after real observations.
function _brm_svi_iterate_diagnostics(values)
    n, d = size(values, 2), size(values, 1)
    rhat, ess = 1.0, Float64(n)
    for first in 1:256:d
        rows = first:min(first + 255, d)
        all(isfinite, view(values, rows, :)) || return (; rhat=Inf, ess=0.0)
        active = filter(j -> !allequal(view(values, j, :)), rows)
        isempty(active) && continue
        x = permutedims(values[active, :])
        all(isfinite, x) || return (; rhat=Inf, ess=0.0)
        stats = MCMCDiagnosticTools.ess_rhat(reshape(x, n, 1, length(active)); kind=:rank)
        all(isfinite, stats.rhat) && all(isfinite, stats.ess) || return (; rhat=Inf, ess=0.0)
        rhat = max(rhat, maximum(stats.rhat))
        ess = min(ess, minimum(stats.ess))
    end
    (; rhat, ess)
end

function _brm_svi_stationarity_check(h::_BRMSVIIterates; limit=_BRM_SVI_STATIONARITY_RHAT)
    size(h.values, 1) == 0 && return merge(_brm_svi_stationarity_empty(),
        (; passed=true, reason=:empty))
    h.count >= _BRM_SVI_STATIONARITY_MIN || return merge(_brm_svi_stationarity_empty(),
        (; samples=h.count, updates=h.updates, stride=h.stride))
    best = nothing
    for n in unique(round.(Int, range(_BRM_SVI_STATIONARITY_MIN, h.count; length=5)))
        stats = _brm_svi_iterate_diagnostics(view(h.values, :, h.count-n+1:h.count))
        passed = stats.rhat <= limit
        report = (; passed, reason=passed ? :stationary : :not_stationary,
            rhat=isfinite(stats.rhat) ? stats.rhat : nothing, ess=stats.ess,
            samples=n, updates=h.updates, stride=h.stride)
        passed && return report
        if best === nothing || (stats.rhat < something(best.rhat, Inf))
            best = report
        end
    end
    best
end

function _brm_svi_stationary!(run, c)
    _brm_svi_stationarity_enabled(c) || return true
    h = _brm_svi_stationarity_state(run)
    run.iteration >= h.next_check || return false
    # As in RABVI, space checks out as the phase length grows. Checking diagnostics
    # does not consume gradients or random numbers. A skipped check never passes.
    elapsed = run.iteration - h.explored * c.window
    h.next_check = run.iteration + max(c.window, cld(elapsed, 10))
    h.checks += 1
    global_report = _brm_svi_stationarity_check(h.globals)
    unit_reports = _brm_svi_stationarity_check.(h.units)
    reports = vcat([global_report], unit_reports)
    passed = all(r -> r.passed, reports)
    rhats = [r.rhat for r in reports if r.rhat !== nothing]
    esses = [r.ess for r in reports if r.ess !== nothing]
    h.report = (; passed, reason=passed ? :stationary : :not_stationary,
        rhat=isempty(rhats) ? nothing : maximum(rhats),
        ess=isempty(esses) ? nothing : minimum(esses),
        globals=global_report, units=unit_reports, iteration=run.iteration)
    passed
end

function _brm_svi_stationarity_explored!(run, c)
    _brm_svi_stationarity_enabled(c) || return
    h = _brm_svi_stationarity_state(run)
    h.explored = length(run.windows)
    h.exploration = h.report
    h.settled = _brm_svi_settle_windows(c) == 0
    h.globals = _BRMSVIIterates(size(h.globals.values, 1))
    h.units = [_BRMSVIIterates(size(u.values, 1)) for u in h.units]
    h.report = _brm_svi_stationarity_empty()
    h.next_check = run.iteration + c.window * (c.patience + 1)
end

# ── Exploration's drift gate ─────────────────────────────────────────────────
#
# Exploration's ELBO test cannot see a slow climb once many units share a batch,
# and settling's step, which moves each global mean in units of its own q sd,
# cannot finish one. A non-centred per-unit step `m + s z_i` that every unit's
# data pin is such a climb: `s` falls only as fast as every `z_i` grows. On a
# public toy of that shape at 256 units (batch 4) the ELBO test ended exploration
# at 23,000 iterations with log `s` at -1.40 of its plateau's -4.7, and settling
# then moved it by about 1e-6 per iteration, to a gain ratio of 0.963 against the
# generating 1.18 after 100,000 iterations (snag brm-svi-fit-unco-182711a4).
# Exploration's own step covers that descent at about the same rate per
# iteration at 32 and 256 units.
#
# So when the ELBO test would end exploration, it ends only once the globals'
# recent iterates are no longer dominated by a steady drift: some recent window
# of the global history reads split-Rhat at most `_BRM_SVI_DRIFT_RHAT`, searched
# as the averaging check searches for 1.1. A steady drift reads about 2.1 on
# every window (a line's split-Rhat), while an iterate that only wanders under
# exploration's constant step reads near 1: the exact Gaussian fixture of
# test/nlme_svi_settling.jl read 1.12 to 1.31 there. The gate reads no unit
# histories, consumes no random numbers, and keeps no state of its own, so a
# checkpoint resumes it exactly. It never holds exploration past half of
# `max_iterations`, so settling and averaging keep at least the other half.
const _BRM_SVI_DRIFT_RHAT = 1.5

_brm_svi_drift_enabled(c) = _brm_svi_stationarity_enabled(c) &&
    get(c, :exploration_drift, false) && _brm_svi_settle_windows(c) > 0
_brm_svi_drift_cap(c) = c.max_iterations ÷ 2

# Each row's rank-normalized split-Rhat (1 for a constant row, Inf for a non-finite
# one), computed in the same bounded chunks as `_brm_svi_iterate_diagnostics`.
function _brm_svi_iterate_rhats(values)
    n, d = size(values, 2), size(values, 1)
    rhat = ones(d)
    for first in 1:256:d
        rows = first:min(first + 255, d)
        bad = filter(j -> !all(isfinite, view(values, j, :)), rows)
        rhat[bad] .= Inf
        active = filter(j -> !(j in bad) && !allequal(view(values, j, :)), rows)
        isempty(active) && continue
        x = permutedims(values[active, :])
        stats = MCMCDiagnosticTools.ess_rhat(reshape(x, n, 1, length(active)); kind=:rank)
        rhat[active] .= ifelse.(isfinite.(stats.rhat), stats.rhat, Inf)
    end
    rhat
end

# The globals whose recent iterates move most: the `top` rows of the global history
# with the largest split-Rhat over its last `_BRM_SVI_STATIONARITY_MIN` samples, as
# (global index, :mean or :log_sd, Rhat). The history keeps D means, then D log sds.
function _brm_svi_moving_globals(h::_BRMSVIIterates; top=3, limit=_BRM_SVI_STATIONARITY_RHAT)
    D = size(h.values, 1) ÷ 2
    (D == 0 || h.count < _BRM_SVI_STATIONARITY_MIN) && return NamedTuple[]
    r = _brm_svi_iterate_rhats(view(h.values, :, h.count-_BRM_SVI_STATIONARITY_MIN+1:h.count))
    order = filter(k -> r[k] > limit, sortperm(r; rev=true))
    [(; index=k <= D ? k : k - D, moment=k <= D ? :mean : :log_sd, rhat=r[k])
        for k in order[1:min(top, length(order))]]
end

# Whether exploration must go on although its ELBO test passed (above), or
# `nothing` when the gate is off: old checkpoints' controls, `stationarity=false`,
# and `settle_windows=0`, where exploration already waits for the averaging check.
function _brm_svi_exploration_drift(run, c)
    _brm_svi_drift_enabled(c) || return nothing
    h = _brm_svi_stationarity_state(run)
    r = _brm_svi_stationarity_check(h.globals; limit=_BRM_SVI_DRIFT_RHAT)
    drifting = r.reason === :not_stationary
    capped = drifting && run.iteration >= _brm_svi_drift_cap(c)
    (; held=drifting && !capped, capped, rhat=r.rhat, rhat_limit=_BRM_SVI_DRIFT_RHAT,
       samples=r.samples, iteration=run.iteration, window=length(run.windows),
       moving=drifting ? _brm_svi_moving_globals(h.globals; limit=_BRM_SVI_DRIFT_RHAT) :
           NamedTuple[])
end

# Keep the gate's last reading on the current report, counting the windows it held
# exploration; `_brm_svi_stationarity_explored!` carries it into `exploration`.
function _brm_svi_note_drift!(run, drift, previous=get(_brm_svi_stationarity_state(run).report,
        :drift, nothing))
    h = _brm_svi_stationarity_state(run)
    held_windows = (previous === nothing ? 0 : previous.held_windows) + drift.held
    h.report = merge(h.report, (; drift=merge(drift, (; held_windows))))
    nothing
end

# The warning of a fit that ended without converging: its end status, the iterate
# check's reading and the globals it reads as still moving, named.
function _brm_svi_unconverged_message(run, c, names; caller)
    status = _brm_svi_end_status(run, c)
    h = _brm_svi_stationarity_state(run)
    detail = if _brm_svi_stationarity_enabled(c) && h isa _BRMSVIStationarity
        rhat = get(h.report, :rhat, nothing)
        moving = _brm_svi_moving_globals(h.globals)
        string("its parameter-iterate check did not pass",
            rhat === nothing ? "" : " (split-Rhat $(round(rhat; sigdigits=3)), limit " *
                "$(_BRM_SVI_STATIONARITY_RHAT))",
            isempty(moving) ? "" : "; globals still moving: " * join(
                ["$(names[g.index]) ($(g.moment === :mean ? "mean" : "log sd"), Rhat " *
                    "$(round(g.rhat; sigdigits=3)))" for g in moving], ", "))
    else
        "its ELBO had not stopped rising"
    end
    string("$caller: stopped ($status) after $(run.iteration) iterations without ",
        "converging: ", detail, ". The factor and its draws are those of an unconverged ",
        "fit; see `fit.diagnostics.stationarity` and `fit.diagnostics.stages`, and raise ",
        "`max_iterations`")
end

function _brm_svi_warn_unconverged(run, c, names; caller)
    _brm_svi_finished(run, c) && !run.converged || return nothing
    @warn _brm_svi_unconverged_message(run, c, names; caller)
    nothing
end

function _brm_svi_stationarity_report(run)
    h = _brm_svi_stationarity_state(run)
    h === nothing && return (; enabled=false)
    deepcopy((; enabled=true, rhat_limit=_BRM_SVI_STATIONARITY_RHAT,
       capacity=_BRM_SVI_STATIONARITY_CAPACITY,
       checks=h.checks, exploration=h.exploration, current=h.report, settled=h.settled))
end

_brm_svi_sampling_report(run) = Base.structdiff(run.sampling, NamedTuple{(:stationarity,)})

function _brm_svi_stationarity_consistent(run, c)
    _brm_svi_stationarity_enabled(c) || return true
    h = _brm_svi_stationarity_state(run)
    h isa _BRMSVIStationarity || return false
    q = run.q
    valid(x, d) = size(x.values, 1) == d &&
        0 <= x.count <= size(x.values, 2) <= _BRM_SVI_STATIONARITY_CAPACITY &&
        x.stride >= 1 && ispow2(x.stride) && x.count <= x.updates ÷ x.stride
    valid(h.globals, 2length(q.μ)) && length(h.units) == length(q.M) &&
        all(i -> valid(h.units[i], _brm_svi_unit_history_dimension(q, i)), eachindex(q.M)) &&
        0 <= h.explored <= length(run.windows)
end
