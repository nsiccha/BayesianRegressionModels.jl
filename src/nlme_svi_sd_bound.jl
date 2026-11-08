# ── How wide the global factor may get: a bound from each global's own prior ──
#
# A STOPGAP (user, decision 1vm7om5: "mark this as a stopgap"). It holds q's width
# where the optimizer cannot; it does not make the optimizer see the prior through
# the noise, which is the cause, and it can hold q narrower than a posterior wider
# than twice its prior's Gaussian width (a heavy-tailed prior under prior-data
# conflict, a periodic or multimodal likelihood). Every fit that ends with a global
# at its bound warns and names it, and `sd_bound=nothing` turns the bound off. A
# noise-limited Adam step that would remove the cause stopped the runaway on ODL
# but held strongly informed globals that start too wide, and its fits ended
# millions of nats below the bound's (snag lowrank-svi-scal-4c45adff, decision
# 1vm7om5); the cause stays open as BayesianRegressionModels todo 1w79j69.
#
# Adam's step has the same size whatever its coordinate's gradient noise, so a
# coordinate whose per-draw gradient is mostly noise moves by about a step size
# per iteration whichever way the noise points, and its prior's pull, which the
# noise dwarfs, barely shows in the step. Its mean, its scale ρ and its rank row
# Β then drift until the noise's own size stops them, and the iterate's spread
# around the optimum grows with that size (as lr σ_g / 2 in units of the
# curvature, for a gradient rms σ_g). Where the target's gradient grows with the
# draw itself, as a log length's does through a random-feature phase
# frequency·x·exp(-log ℓ), a wider factor reaches draws with larger gradients
# and the noise grows with the factor: the loop has no bound. On ODL's
# Niedersachsen slice (82 stations, `LowRankGlobal(8)`, 40,000 iterations of
# exploration) one field log length under a N(-1.386, 0.5) prior went from sd
# 0.1 to marginal sd 378 (exp(ρ) 246, row |Β| 287) with a gradient rms of 1e28
# on its scale, against 1 to 40 on every other global, until a draw reached
# log ℓ < -709 and the density overflowed (snag lowrank-svi-scal-4c45adff). Its
# prior's pull at that width is about 1e5 nats of ELBO.
#
# Only the prior gives that coordinate a scale the noise cannot move. A global
# whose prior reads no other coordinate therefore has its marginal sd under q,
# √(exp(2ρ_k) + |Β_k|²), held at most `sd_bound` (a control of the fit, by default
# `_BRM_SVI_SD_BOUND`) times the sd of the Gaussian closest to its prior
# (`_brm_svi_prior_projection`, the factor a fit on no data would reach). A posterior wider than its prior needs a
# likelihood whose expected curvature is negative there, so an optimum seldom
# reaches the bound; a factor that does has been carried there by its noise. A
# global whose prior reads others (a centered field state, which reads its
# neighbours and its field's lengths, and those lengths) has no fixed scale and
# no bound. Neither has a flat prior's. Every global starts at `global_scale`, so
# one whose prior sd is below half of it is at its bound from the first step,
# where Adam took hundreds of iterations to shrink it: on the slice above a
# global with prior sd 1e-6 still had sd 0.008, and the largest gradient rms of
# all, after 2,000 iterations.
const _BRM_SVI_SD_BOUND = 2.0

# The fit's `sd_bound` control, `nothing` when the bound is off. Controls written
# before it existed (an older checkpoint) hold the default: a resumed run's factor
# that already drifted is held from its first step, as the reporter's
# Schleswig-Holstein checkpoint needed.
_brm_svi_sd_bound_factor(c) = get(c, :sd_bound, _BRM_SVI_SD_BOUND)
function _brm_svi_check_sd_bound(sd_bound; caller="brm_svi_fit")
    sd_bound === nothing || sd_bound isa Real && isfinite(sd_bound) && sd_bound > 0 ||
        throw(ArgumentError("$caller: sd_bound must be a positive finite factor of each " *
            "global's prior sd, or nothing to turn the bound off; got $(repr(sd_bound))"))
    sd_bound === nothing ? nothing : Float64(sd_bound)
end

# The sd of the Gaussian projection of each of `t`'s globals' own prior, `Inf`
# where it has none: the bound is `sd_bound` times it. It depends on the model's
# prior only and uses its own random stream, so a fit's draws and a resumed fit
# are unchanged by it.
function _brm_svi_sd_prior(t; rng=Random.Xoshiro(0x5370726561640001), pairs=16)
    D = length(t.globals)
    D == 0 && return Float64[]
    bound = _brm_svi_with_bound_term(t) do term
        projection = _brm_svi_prior_projection(t; rng, nonfinite=:flat, term)
        τ = [p > 0 ? 1 / sqrt(p) : Inf for p in projection.precision]
        own = _brm_svi_own_prior(t, projection.mean, τ; rng, pairs, term)
        [own[k] && isfinite(τ[k]) ? τ[k] : Inf for k in 1:D]
    end
    # Unit-owned states' conditionals sit in the units' terms, not in the global
    # term, so there a field's lengths and correlation factor would look as if
    # their prior were their own. They are the states' scales either way.
    state = get(t, :fields, nothing)
    if state !== nothing
        for field in state.fields
            field.spatial.place > 0 && (bound[field.spatial.place] = Inf)
            field.temporal.place > 0 && (bound[field.temporal.place] = Inf)
            for p in field.factor.places
                p > 0 && (bound[p] = Inf)
            end
        end
    end
    bound
end

# `f(term)` with `term(t, g)` the global term at `g`. With unit-owned field states
# (`field_states=:units`) the global states' conditionals read the units' states,
# which a fit draws each iteration: here they read the target's own point, as an
# evaluation of the whole density there does (`_brm_svi_field_joint!`), and the
# fields' reading is put back afterwards, so the target is left as it was built.
function _brm_svi_with_bound_term(f, t)
    state = get(t, :fields, nothing)
    state === nothing && return f(_brm_svi_global_term)
    others = copy(state.others)
    u = copy(t.prior_point)
    try
        f() do t, g
            u[t.globals] .= g
            _brm_svi_field_joint!(t, u)
            _brm_svi_global_term(t, g)
        end
    finally
        copyto!(state.others, others)
        foreach(field -> field.law[] = nothing, state.fields)
    end
end

# Whether each global's prior gradient reads that global only. Antithetic draws
# around `m` move every global by ±h; one whose prior is its own takes a single
# gradient at +h and a single one at -h, whatever the others do. Sixteen pairs
# move another global it reads to both of its sides with chance 1 - 2⁻¹⁵.
function _brm_svi_own_prior(t, m, τ; rng, pairs, term=_brm_svi_global_term)
    D = length(m)
    h = [isfinite(s) ? min(s, 1.0) / 2 : 0.5 for s in τ]
    above, below = fill(NaN, D), fill(NaN, D)
    own = trues(D)
    for _ in 1:pairs
        u = rand(rng, (-1.0, 1.0), D)
        _, a = term(t, m .+ h .* u)
        _, b = term(t, m .- h .* u)
        for k in 1:D
            own[k] || continue
            plus, minus = u[k] > 0 ? (a[k], b[k]) : (b[k], a[k])
            if !(isfinite(plus) && isfinite(minus))
                own[k] = false
            elseif isnan(above[k])
                above[k], below[k] = plus, minus
            else
                own[k] = _brm_svi_same_value(above[k], plus) &&
                    _brm_svi_same_value(below[k], minus)
            end
        end
    end
    own
end
_brm_svi_same_value(a, b) = abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))

# The target's `sd_prior`, `nothing` for a target without one: the analytic test
# targets are structs, which have neither the field nor `get`.
_brm_svi_sd_prior_of(t) = hasproperty(t, :sd_prior) ? getproperty(t, :sd_prior) : nothing

# q's global factor with every bounded global's marginal sd within `factor` times
# its `sd` (`_brm_svi_sd_prior`): exp(ρ_k) and the row Β_k shrink by one factor, so
# its correlations stay. With `sd` or `factor` nothing, q is untouched.
_brm_svi_bound_sd!(q, bound) = _brm_svi_bound_sd!(q, bound, 1.0)
function _brm_svi_bound_sd!(q, sd, factor)
    (sd === nothing || factor === nothing) && return q
    r = size(q.Β, 2)
    for k in eachindex(sd)
        b = factor * sd[k]
        isfinite(b) || continue
        v = exp(2q.ρ[k])
        for j in 1:r
            v += q.Β[k, j]^2
        end
        v > b^2 || continue
        f = b / sqrt(v)
        q.ρ[k] += log(f)
        for j in 1:r
            q.Β[k, j] *= f
        end
    end
    q
end

# What a fit's diagnostics report of the bound: its `factor` (`nothing` when off),
# each global's `bound` (`Inf` where it has none, or when off) and the globals
# whose marginal sd in `q` is at it.
function _brm_svi_sd_bound_report(t, q, factor)
    sd = _brm_svi_sd_prior_of(t)
    sd === nothing && return nothing
    names = t.m.view.coordinates[t.globals]
    bound = factor === nothing ? fill(Inf, length(sd)) : factor .* sd
    marginal = sqrt.(exp.(2 .* q.ρ) .+ vec(sum(abs2, q.Β; dims=2)))
    (; factor, bound, coordinates=names, stopgap=true,
       at_bound=names[[isfinite(bound[k]) && marginal[k] >= bound[k] * (1 - 1e-9)
           for k in eachindex(bound)]])
end

# The fit's warning when it ends with globals at their bound: q's sd there is the
# bound's, not the posterior's, and the bound is a stopgap.
function _brm_svi_warn_sd_bound(report; caller)
    (report === nothing || isempty(report.at_bound)) && return nothing
    n = length(report.at_bound)
    @warn string(caller, ": ", n, n == 1 ? " global ends" : " globals end",
        " at the SVI sd bound (a stopgap, ", report.factor, " times the sd of its prior's ",
        "Gaussian projection): its sd under q is the bound's, not a posterior sd. ",
        "diagnostics.sd_bound names each global and its bound; sd_bound=nothing turns ",
        "the bound off.") at_bound = report.at_bound
    nothing
end
