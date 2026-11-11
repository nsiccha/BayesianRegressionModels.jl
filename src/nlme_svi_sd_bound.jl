# Bound each global factor's marginal SD by a multiple of its own prior SD.
# This optimizer safeguard is a stopgap, not a posterior-accuracy guarantee:
# prior-data conflict or a multimodal likelihood can require a wider posterior.
# Fits that reach the bound warn and name the coordinate; `sd_bound=nothing`
# disables it. The marginal factor SD is sqrt(exp(2rho) + sum(abs2, rank_row)).
# Globals whose priors depend on other coordinates, and flat priors, have no bound.
const _BRM_SVI_SD_BOUND = 2.0

# Older checkpoints retain the default bound; `nothing` disables it.
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
