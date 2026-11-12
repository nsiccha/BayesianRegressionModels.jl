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
function _brm_svi_stationarity_state(q, c)
    _BRMSVIStationarity(_BRMSVIIterates(2length(q.μ)),
        [_BRMSVIIterates(2length(q.M[i]) + _brm_svi_centering_dimension(q, i))
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
    if hasproperty(q, :centering)
        slot[2d+1:end] .= _brm_svi_unit_controls(q, i)
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

function _brm_svi_stationarity_check(h::_BRMSVIIterates)
    size(h.values, 1) == 0 && return merge(_brm_svi_stationarity_empty(),
        (; passed=true, reason=:empty))
    h.count >= _BRM_SVI_STATIONARITY_MIN || return merge(_brm_svi_stationarity_empty(),
        (; samples=h.count, updates=h.updates, stride=h.stride))
    best = nothing
    for n in unique(round.(Int, range(_BRM_SVI_STATIONARITY_MIN, h.count; length=5)))
        stats = _brm_svi_iterate_diagnostics(view(h.values, :, h.count-n+1:h.count))
        passed = stats.rhat <= _BRM_SVI_STATIONARITY_RHAT
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
        all(i -> valid(h.units[i], 2length(q.M[i]) + _brm_svi_centering_dimension(q, i)), eachindex(q.M)) &&
        0 <= h.explored <= length(run.windows)
end
