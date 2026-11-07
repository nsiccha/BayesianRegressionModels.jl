# ── Globals that few units read: reader sites ────────────────────────────────
#
# A global coordinate read by k of N units (a time window's boundary state, a
# field state, a separator) has a nonzero likelihood gradient only when one of
# its readers is in the batch, about k·B/N of the iterations, and then N/B times
# the reader's own. Adam normalizes each such hit to a step of about ten learning
# rates and sees only the prior and the entropy in between, so the coordinate's
# mean and scale random-walk instead of settling. Restarting Adam for polishing
# makes it worse: the fresh moments know only the entropy's push, so the scale
# grows and the mean slides towards the prior's for the whole stretch until a
# reader is drawn. On a toy of window units with boundary states read by two
# units each, at N/B = 128, the shipped optimizer ended 229,317 nats below the
# family's optimum, the boundaries' sds 4.1 times the optimal (median) and their
# means 15 optimal sds off (rms).
#
# For such a coordinate p, each reader i keeps a Gaussian site
# exp(ν_ip x - λ_ip x²/2) standing for E_q[unit i's term] as a function of x_p,
# and every iteration steps the coordinate on its prior's fresh gradient plus
# the sum of all its readers' sites, instead of N/B times the batch readers'
# gradients. A site changes only when its unit is visited, as the unit's own
# factor does, and it answers for the reader in between: its gradient at the
# current mean is ν - λ μ, its scale gradient -λ s², each reacting to q as that
# reader's term would to second order. Nothing is scaled by N/B, so a reader
# drawn once in a hundred iterations no longer kicks the coordinate. The
# objective, its estimate in the trace and the units' steps are unchanged.
#
# On a visit, with G the reader's gradient in x_p at the iteration's draw
# x = μ + s ε (+ Β η), averaged over the unit's antithetic pair:
# - curvature: -(G - g) ε / s, Stein's estimate of E[-∂²], with the site's own
#   gradient at the mean g as a baseline (it removes the part of G that does not
#   depend on ε and would only add noise);
# - gradient at the mean: G + λ (x - μ), with the precision from before this
#   draw, so that the correction is uncorrelated with ε.
# The precision moves by the unit step β; the gradient at the mean by
# `_BRM_SVI_SITE_PULL_STEP`, which decays in polishing as β does. A first visit
# takes both whole. Like a unit's own step, one visit changes the coordinate's
# precision at most 2.5-fold up or by half. Without that bound a site's Stein
# estimate, whose noise grows as 1/s, drove a few precisions up without end (1e15
# and beyond within 2,000 iterations on the toy above). The pull has no such
# bound: one in units of the current sd froze a coordinate whose sd had come out
# too narrow, and on 2,048 window units with learned hyperparameters six
# boundaries then stayed 20 to 95 optimal sds off and held the OU length at 5.2,
# against 17.7 without it (family optimum 40.9).
#
# Updating the gradient at the mean, not the natural parameter ν, keeps a site
# independent of where the coordinate's origin is: damping ν towards a value
# computed with another λ adds (λ_old - λ_new) μ to the pull, which on that toy
# moved the boundary means steadily away, 21 optimal sds off by the end.
#
# On the same toy (N/B = 128) the fit ends 941 nats below the family's optimum,
# 718 of them the units' own widths; the boundaries' sds are 1.01 times the
# optimal (5-95%: 0.84-1.24) and their means 1.2 optimal sds off (rms), 0.53
# after a later stop (`min_windows=120`). Exact coordinate ascent in this family
# needs 20 to 40 sweeps for those means; the damped steps need more visits.
#
# A coordinate is "sparse" when the expected number of reader draws in a batch is
# below one, B·Σ_{i reads p} π_i < 1 for sampling probabilities π. Above that the
# minibatch estimate hits it in most iterations, and every unit reads the
# population coordinates and hyperparameters. Under `LowRankGlobal` a site
# carries the coordinate's own curvature, not its coupling to other globals, so
# a sparse coordinate's rows of Β follow its prior and the sites' diagonal.

const _BRM_SVI_SITE_PULL_STEP = 0.5
# The largest relative change of a sparse coordinate's precision in one visit:
# down to ½, up to 1 + 1.5 = 2.5 times, as for a unit's factor.
const _BRM_SVI_SITE_SHRINK = 0.5
const _BRM_SVI_SITE_GROW = 1.5

# The run's reader sites. `globals` are positions in `t.globals`, `slot[p]` the
# position of global p among them (0 for a global stepped by the minibatch
# estimate), `entries[i]` the indices into `_brm_svi_unit_touched(t, i)` of unit
# i's sparse globals, and `precision[i]` / `pull[i]` its sites' (λ, ν) for them.
# The sums over readers are kept per sparse global.
struct _BRMSVISites
    globals::Vector{Int}
    readers::Vector{Int}
    expected::Vector{Float64}
    slot::Vector{Int}
    entries::Vector{Vector{Int}}
    precision::Vector{Vector{Float64}}
    pull::Vector{Vector{Float64}}
    seen::BitVector
    precision_sum::Vector{Float64}
    pull_sum::Vector{Float64}
end

# Sites for target `t` under batch size `B` and the units' sampling
# `probabilities`, or `nothing` when no global is sparse.
function _brm_svi_sites(t, B, probabilities)
    D, N = length(t.globals), length(t.units)
    readers, expected = zeros(Int, D), zeros(D)
    for i in 1:N, p in _brm_svi_unit_touched(t, i)
        readers[p] += 1
        expected[p] += B * probabilities[i]
    end
    globals = [p for p in 1:D if readers[p] > 0 && expected[p] < 1]
    isempty(globals) && return nothing
    slot = zeros(Int, D)
    slot[globals] .= eachindex(globals)
    entries = [[k for (k, p) in enumerate(_brm_svi_unit_touched(t, i)) if slot[p] > 0]
        for i in 1:N]
    _BRMSVISites(globals, readers[globals], expected[globals], slot, entries,
        [zeros(length(e)) for e in entries], [zeros(length(e)) for e in entries],
        falses(N), zeros(length(globals)), zeros(length(globals)))
end

_brm_svi_minibatch_global(::Nothing, p) = true
_brm_svi_minibatch_global(sites::_BRMSVISites, p) = sites.slot[p] == 0

# Refreshes the sites of the batch's units from this iteration's evaluations, in
# batch order: `terms[k].gg` holds unit `batch[k]`'s gradient in its touched
# globals, summed over the antithetic pair, at the global draw `g` from `ε`.
_brm_svi_update_sites!(::Nothing, t, q, g, ε, batch, terms, β, scale) = nothing
function _brm_svi_update_sites!(sites::_BRMSVISites, t, q, g, ε, batch, terms, β, scale)
    for (k, term) in enumerate(terms)
        i = batch[k]
        entries = sites.entries[i]
        isempty(entries) && continue
        touched = _brm_svi_unit_touched(t, i)
        seen = sites.seen[i]
        bλ, bν = seen ? (β, _BRM_SVI_SITE_PULL_STEP * scale) : (1.0, 1.0)
        λs, νs = sites.precision[i], sites.pull[i]
        for (e, entry) in enumerate(entries)
            p = touched[entry]
            G = term.gg[entry] / 2
            μ, s = q.μ[p], exp(q.ρ[p])
            Λ = 1 / s^2
            λ, ν = λs[e], νs[e]
            pull = seen ? ν - λ * μ : 0.0
            curvature = -(G - pull) * ε[p] / s
            λ′ = clamp((1 - bλ) * λ + bλ * curvature,
                max(λ - _BRM_SVI_SITE_SHRINK * Λ, 0.0), λ + _BRM_SVI_SITE_GROW * Λ)
            at_mean = G + (seen ? λ : λ′) * (g[p] - μ)
            pull′ = (1 - bν) * pull + bν * at_mean
            ν′ = pull′ + λ′ * μ
            slot = sites.slot[p]
            sites.precision_sum[slot] += λ′ - λ
            sites.pull_sum[slot] += ν′ - ν
            λs[e], νs[e] = λ′, ν′
        end
        sites.seen[i] = true
    end
    nothing
end

# Adds the readers' sites to `grad`, the gradient of the global vector [μ; ρ; Β],
# whose sparse coordinates hold only the global term's part.
_brm_svi_add_site_gradient!(grad, ::Nothing, q) = grad
function _brm_svi_add_site_gradient!(grad, sites::_BRMSVISites, q)
    D, r = length(q.μ), size(q.Β, 2)
    for (slot, p) in enumerate(sites.globals)
        λ = sites.precision_sum[slot]
        grad[p] += sites.pull_sum[slot] - λ * q.μ[p]
        grad[D+p] -= λ * exp(2q.ρ[p])
        for c in 1:r
            grad[2D+(c-1)*D+p] -= λ * q.Β[p, c]
        end
    end
    grad
end

# The fit's account of its sparse globals: packed positions, reader counts and
# the expected reader draws per batch that made them sparse.
_brm_svi_sites_diagnostics(::Nothing, t) =
    (; coordinates=Int[], readers=Int[], expected_readers=Float64[])
_brm_svi_sites_diagnostics(sites::_BRMSVISites, t) = (; coordinates=t.globals[sites.globals],
    readers=copy(sites.readers), expected_readers=copy(sites.expected))
