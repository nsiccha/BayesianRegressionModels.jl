# A declining training trace nominates a pair of iterates. Only a comparison
# on fresh, paired draws can switch the run to settling at the earlier one.
# The score is the same partitioned log joint, including the change from the
# learned coordinates, plus entropy. No parameter or optimizer steps here.
function _brm_svi_score_draw(rng, t, q, sites, batch)
    (; g, η, noise, P, gP, centering, frame) =
        _brm_svi_sample_target(rng, t, q, sites, batch.indices)
    value = P
    coupling = _brm_svi_coupling(q)
    for (k, i) in enumerate(batch.indices)
        ζ = coupling === nothing ? nothing : _brm_svi_coupling_noise(coupling, i, noise, η)
        u = _brm_svi_coupled_mean(coupling, q, i, ζ) .+
            _brm_svi_unit_offset(q, i, _brm_svi_unit_noise(rng, q.U[i]))
        gu = zeros(length(u))
        # Only the value is read. The gradient buffers are scratch for the
        # target's existing joint value-and-gradient operation.
        v = if centering === nothing
            _brm_svi_unit_local!(gP, gu, t, i, g, u, t.plan, first(t.work))
        else
            first(_brm_svi_centered_unit_term!(gP, gu, centering, frame, t, q, i,
                g, u, t.plan, first(t.work)))
        end
        value += batch.weights[k] * v
    end
    value
end

function _brm_svi_compare(run, t, c, best)
    # Key validation draws by iteration, independently of the training stream.
    # Both factors get the same unit selection and Gaussian noises. Redraw
    # both on each comparison, including the earlier factor selected by the trace.
    rng = Random.Xoshiro(0x96dbd3729f6ac059 ⊻ UInt64(run.iteration))
    earlier, current = Float64[], Float64[]
    calls = 0
    for _ in 1:32
        batch = _brm_svi_batch(rng, run.sampler, min(c.batch_size, 32))
        push!(earlier, _brm_svi_score_draw(copy(rng), t, best.q, best.sites, batch))
        push!(current, _brm_svi_score_draw(rng, t, run.q, run.sites, batch))
        calls += 2length(batch.indices)
    end
    entropy(q, sites, units) = units + _brm_svi_factor_logdet(q) / 2 +
        _brm_svi_sparse_entropy(sites, q)
    earlier .+= entropy(best.q, best.sites, best.unit_entropy_sum)
    current .+= entropy(run.q, run.sites, run.entropy_sum)
    diff = earlier .- current
    loss, se = Statistics.mean(diff), Statistics.std(diff) / sqrt(length(diff))
    (; reference_iteration=best.iteration, iteration=run.iteration,
       reference_score=Statistics.mean(earlier), current_score=Statistics.mean(current),
       loss, se, draws=length(diff), unit_evaluations=calls,
       restored=isfinite(loss) && isfinite(se) && loss > 3se)
end

# Compare complete, non-overlapping groups of windows. Both halves of the
# recent group must lie below the reference, accounting for noise within and
# between its windows. A low single iterate or window cannot trigger a restore.
#
# Exploring and settling each keep their own reference: their steps differ, so
# their training windows do not compare. Exploration's is gone when settling
# starts (`_brm_svi_finish_interval!`), and a group counts only once it lies within the
# phase. Returns whether the run's factor is now the saved one.
function _brm_svi_recover!(run, t, c, synchronize!)
    get(c, :restore_best, false) && _brm_svi_drift_enabled(c) || return false
    w = length(run.windows)
    span = max(c.patience, 4)
    w % span == 0 || return false
    settling = _brm_svi_settling(run, c)
    since = settling ? _brm_svi_explored(run, c) : 0
    w - span >= since || return false
    windows = view(run.windows, w-span+1:w)
    center = Statistics.mean(v -> v.center, windows)
    se(ws) = sqrt(max(Statistics.var([v.center for v in ws]),
        Statistics.mean(v -> v.spread, ws) / c.window) / length(ws))
    best = get(run.sampling, :reference, nothing)
    if best === nothing || center > best.center
        synchronize!(:state)
        reference = (; center, se=se(windows), iteration=run.iteration,
            unit_entropy_sum=run.entropy_sum, q=deepcopy(run.q), sites=deepcopy(run.sites))
        run.sampling = merge(run.sampling, (; reference))
        return false
    end
    w - since >= (settling ? 2span : c.min_windows) || return false
    halves = (view(windows, 1:span÷2), view(windows, span÷2+1:span))
    all(ws -> best.center - Statistics.mean(v -> v.center, ws) >
        3hypot(best.se, se(ws)), halves) || return false

    synchronize!(:state)
    check = merge(_brm_svi_compare(run, t, c, best),
        (; phase=settling ? :settling : :exploring))
    history = get(run.sampling, :score_checks, nothing)
    if history === nothing
        history = typeof(check)[]
        run.sampling = merge(run.sampling, (; score_checks=history))
    end
    push!(history, check)
    check.restored || return false
    @warn string("brm_svi_fit: ", settling ? "settling" : "exploration", "'s objective fell ",
        "after iteration ", check.reference_iteration, ". On ", check.draws,
        " fresh paired draws the factor saved there scores ", round(check.loss; sigdigits=3),
        " ± ", round(check.se; sigdigits=2), " nats above the one at iteration ",
        check.iteration, settling ?
            ", so the fit averages from the saved factor and ends without converging" :
            ", so the fit continues from the saved factor",
        ". `restore_best=false` keeps the live one; see `diagnostics.stationarity.",
        settling ? "current" : "exploration", ".score_checks`")
    # The run owns the saved factor from here on: no reference may alias it.
    run.sampling = Base.structdiff(run.sampling, NamedTuple{(:reference,)})
    run.q, run.sites = best.q, best.sites
    for i in eachindex(run.unit_entropy)
        run.unit_entropy[i] = _brm_svi_unit_entropy(run.q, i)
    end
    run.entropy_sum = best.unit_entropy_sum
    true
end

# A factor restored while its centering was still deferred has just been centred
# and explores on (`_brm_svi_finish_interval!`). The globals jumped back, so their Adam
# moments and the iterate history the drift gate reads start again there.
function _brm_svi_recovered_exploring!(run, c)
    clipped = run.adam.clipped
    run.adam = _brm_svi_adam(run.q, c.lr_global)
    run.adam.clipped = clipped
    h = _brm_svi_stationarity_state(run)
    h.globals = _BRMSVIIterates(size(h.globals.values, 1))
    run
end

# A phase ends: its comparisons move to the stationarity report that describes it,
# and its reference goes.
function _brm_svi_recovery_close!(run)
    if hasproperty(run.sampling, :score_checks)
        h = _brm_svi_stationarity_state(run)
        h.report = merge(h.report, (; score_checks=run.sampling.score_checks))
    end
    run.sampling = Base.structdiff(run.sampling, NamedTuple{(:reference, :score_checks)})
    run
end
