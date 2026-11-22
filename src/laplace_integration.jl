# src/laplace_integration.jl — deterministic integration over the population
# coordinates of an embedded-Laplace target: rung 5c of `brm_ladder_fit`
# (design `0r49iwd`, todo `1sc1jw3`), the counterpart of R-INLA's and
# Latte.jl's integration step (Rue, Martino and Chopin 2009, §3.1 and §6.5).
#
# Adopted from R-INLA and Latte (read at Latte 0.2.0, `src/inference/inla/`):
#   - z-coordinates from the curvature at the marginal's mode,
#     θ(z) = θ* + V Λ^(-1/2) z, with -H = V Λ Vᵀ;
#   - a grid walked along each z-axis in steps of `step` until the log density
#     drops `max_log_drop` below the mode, then filled inside that region
#     (R-INLA's defaults dz = 0.75 and diff.logdens = 6);
#   - a central composite design of 1 + 2d + 2^(d-k) points on the sphere of
#     radius f0·√d, weighted Δ_k·π̃(θ_k | y) with Rue et al.'s analytic Δ_k;
#   - grid for d ≤ 2 and CCD for d ≥ 3;
#   - draws in the way of R-INLA's posterior sampler: design points resampled
#     by weight, the integrated effects drawn from each point's conditional
#     Gaussian (here the sparse factor, `brm_laplace_draws`).
# Done differently: the CCD fraction above d = 4 is a regular resolution-V
# fraction found by a deterministic search (Latte uses a 2^(d-1) design); the
# hyperparameter marginals are moment-preserving kernel mixtures over the
# weighted design, not R-INLA's skewed Gaussian in z (measured below); and
# every result is BRM's packed draws so every BRM reader accepts it
# (`scaling-interop`). The curvature at the mode only places and scales the
# design; the result is the weighted Laplace marginal, never that Gaussian
# (`no-full-laplace`, design `1lxkp7y`).

"""
    brm_laplace_integration(target; point = nothing, design = :auto,
        draws = 1000, conditional = 0, rng = Random.default_rng(),
        max_dimension = 10, step = 0.75, max_log_drop = 6.0, f0 = 1.1,
        marginal_points = 401, curvature = nothing, latent_marginals = :none) -> NamedTuple

Deterministic integration of the embedded-Laplace marginal of `target` (a
problem from [`brm_laplace_problem`](@ref)) over its population coordinates,
for models with few of them: R-INLA's integration step. It evaluates the
Laplace marginal at the points of a design around its mode, weights them, and
returns the weighted points, each population coordinate's marginal density
and packed draws. Set `draws=0` to skip sampling.

- `point`: the marginal's mode over the population coordinates (in
  `target.names` order). By default [`brm_laplace_mode`](@ref) finds it (needs
  Optim, which `using WarmupHMC` loads).
- `design`: `:grid` walks each standardized axis in steps of `step` until the
  log density drops more than `max_log_drop` below the mode, then keeps every
  grid point within that drop. `:ccd` evaluates a central composite design of
  `1 + 2d + 2^(d-k)` points on the sphere of radius `f0 * sqrt(d)` (`f0 > 1`).
  `:auto` (the default) uses the grid for `d ≤ 2` and the CCD for `d ≥ 3`.
- `max_dimension`: above this many population coordinates the call refuses.
  A design does not scale to many coordinates; sample the marginal instead
  (`brm_ladder_fit`'s rung 5b).
- `curvature`: a precomputed curvature at `point`, a NamedTuple with
  `hessian` and `positive_definite` such as rung 5a's. By default it is
  computed by central differences of the marginal's gradient (2d gradients).
  It only places and scales the design.

The standardization is z ↦ θ(z) = θ* + V Λ^(-1/2) z with -H = V Λ Vᵀ.
Each grid point has weight proportional to its Laplace marginal. A CCD point
has weight proportional to Δ_k times its marginal, with Rue, Martino and
Chopin's (2009, §6.5) Δ_k, which make the rule exact for the first two
moments of a Gaussian.

Returns `(; draws, point, design, z, points, log_density, weights, ess, mean,
sd, marginals, log_evidence, standardization, diagnostics, latent_marginals)`:
- `draws`: `draws` packed draws (coordinates × draws, the shape of
  [`brm_laplace_draws`](@ref)). The population coordinates are design points
  resampled by weight, as R-INLA's posterior sampler does. The integrated
  effects are drawn from each point's conditional Gaussian through the
  factor, so they are joint draws. `conditional` importance-resamples them as
  in `brm_laplace_draws`.
- `z`, `points`, `log_density`, `weights`: the design in standardized and
  population coordinates (one column per point), the Laplace marginal there,
  and the normalized integration weights. A point whose marginal is not finite
  has weight 0. `ess` is the weights' effective sample size.
- `mean`, `sd`: the population coordinates' weighted moments over the design.
- `marginals`: one `(; name, x, density, mean, sd)` per population coordinate,
  a density table over `marginal_points` values. With one coordinate on a grid
  it interpolates the log marginal between the grid points. Otherwise it is a
  Gaussian kernel mixture over the weighted design points, shrunk towards
  their mean so that its mean and variance are the design's `mean` and `sd`
  (on a grid each kernel spans its cell; on a CCD it carries half the
  variance).
- `latent_marginals`: `nothing` by default. With `:gaussian` or
  `:mean_corrected`, `names`, `mean`, `sd` and `distributions` each have one
  entry per integrated coordinate. Each distribution is a Normal mixture and
  supports `pdf`, `cdf` and `quantile`. Column `k` of `conditional_mean` and
  `conditional_sd` goes with `points[:, point_indices[k]]` and `weights[k]`
  in the latent result. Transform both conditional parameters at each point
  before mixing when the physical effect depends on a population scale.
  Integrated sum-to-zero second-order walks report the standardized field
  `<declaration>.phi.<step index>` in fitted step order.
- `:mean_corrected` shifts the conditional centre by the derivative of the
  Laplace log normalizer under a linear tilt. For exact conditional precision
  H this gives H⁻¹ (D³ log p : H⁻¹) / 2. Gaussian conditionals have zero
  shift. Joint Gaussian draws use the selected centres. Positive `conditional`
  counts use the existing importance-resampled draws targeting the conditional.
- Matrix-free components above the exact size limit estimate variances with
  the solver's fixed probes applied through its covariance square root. The
  mean correction differentiates the solver's fixed-probe log determinant.
- `log_evidence`: on a grid, the log of the marginal's integral (the log
  marginal likelihood under the Laplace approximation); `nothing` for a CCD,
  whose weights are relative.
- `standardization`: `(; hessian, transform)`, with `transform` = V Λ^(-1/2).
- `diagnostics`: `evaluations`, `nonfinite`, `max_gain` (the largest log
  marginal of a design point above the centre's; a warning names a gain
  above 1, since `point` is then not the mode), `truncated` (axes whose walk
  stopped at its step limit) and `newton_steps` of the draws.
"""
function brm_laplace_integration(p::Union{BRMSparseLaplaceProblem,_BRMSampledLaplaceProblem};
        point=nothing, design::Symbol=:auto, draws::Integer=1000, conditional::Integer=0,
        rng=Random.default_rng(), max_dimension::Integer=10, step::Real=0.75,
        max_log_drop::Real=6.0, f0::Real=1.1, marginal_points::Integer=401,
        curvature=nothing, latent_marginals::Symbol=:none)
    latent_marginals in (:none, :gaussian, :mean_corrected) || throw(ArgumentError(
        "latent_marginals must be :none, :gaussian or :mean_corrected"))
    d = length(p.population)
    d >= 1 || throw(ArgumentError(
        "brm_laplace_integration: the target has no population coordinates to integrate"))
    d <= max_dimension || throw(ArgumentError(
        "brm_laplace_integration: the target has $d population coordinates, above " *
        "max_dimension = $max_dimension; a design does not scale to that many. Sample " *
        "the Laplace marginal instead (brm_ladder_fit rung 5b), or integrate more terms " *
        "so fewer population coordinates remain"))
    design in (:auto, :grid, :ccd) || throw(ArgumentError(
        "brm_laplace_integration: design = $(repr(design)); pass :auto, :grid or :ccd"))
    design = design === :auto ? (d <= 2 ? :grid : :ccd) : design
    step > 0 || throw(ArgumentError("brm_laplace_integration: step = $step must be positive"))
    max_log_drop > 0 || throw(ArgumentError(
        "brm_laplace_integration: max_log_drop = $max_log_drop must be positive"))
    f0 > 1 || throw(ArgumentError(
        "brm_laplace_integration: f0 = $f0; the CCD needs f0 > 1 for a positive centre weight"))
    draws >= 0 || throw(ArgumentError("brm_laplace_integration: draws = $draws is negative"))
    marginal_points >= 3 || throw(ArgumentError(
        "brm_laplace_integration: marginal_points = $marginal_points; pass at least 3"))
    θstar = point === nothing ? brm_laplace_mode(p).draws[p.population, 1] :
        collect(Float64, point)
    length(θstar) == d || throw(DimensionMismatch(
        "brm_laplace_integration: point has length $(length(θstar)), the target has $d " *
        "population coordinates"))
    curvature = curvature === nothing ? _brm_ladder_curvature(p, θstar) : curvature
    curvature.positive_definite || _brm_nlme_error(
        "brm_laplace_integration: the marginal's curvature at the point is not negative " *
        "definite, so it cannot place a design; the point is not a strict mode")
    H = Matrix{Float64}(curvature.hessian)
    eig = eigen(Symmetric(-H))
    A = eig.vectors * Diagonal(1 ./ sqrt.(eig.values))
    θ(z) = θstar .+ A * z
    v0 = LogDensityProblems.logdensity(p, θstar)
    isfinite(v0) || _brm_nlme_error(
        "brm_laplace_integration: the Laplace marginal is not finite at the point")
    calls = Ref(1)
    # Record each retained point while its conditional factor is still live.
    states = Dict{Vector{Float64},Any}()
    summaries = Dict{Vector{Float64},Tuple{Vector{Float64},Vector{Float64}}}()
    function record(φ)
        draws > 0 && (states[φ] = _brm_lint_draw_state(p))
        latent_marginals === :none || (summaries[φ] = _brm_lint_conditional(p, latent_marginals))
        nothing
    end
    record(θ(zeros(d)))
    start = _brm_lint_start(p)
    function value(z)
        calls[] += 1
        φ = θ(z)
        v = _brm_lint_logdensity(p, φ, start)
        isfinite(v) && (design === :ccd || v0 - v <= max_log_drop) && record(φ)
        isfinite(v) ? Float64(v) : -Inf
    end
    evaluated = design === :grid ?
        _brm_lint_grid(value, v0, d, Float64(step), Float64(max_log_drop)) :
        _brm_lint_ccd(value, v0, d, Float64(f0))
    Z, v, logΔ = evaluated.z, evaluated.log_density, evaluated.log_weight
    finite = isfinite.(v)
    lw = ifelse.(finite, logΔ .+ v, -Inf)
    w = exp.(lw .- maximum(lw))
    w ./= sum(w)
    Θ = reduce(hcat, (θ(view(Z, :, k)) for k in axes(Z, 2)); init=zeros(d, 0))
    μ = Θ * w
    σ = sqrt.(max.((Θ .- μ) .^ 2 * w, 0.0))
    gain = maximum(v[finite]) - v0
    gain > 1 && @warn("brm_laplace_integration: a design point's log marginal is " *
        "$(round(gain; sigdigits=3)) above the centre's, so `point` is not the marginal's " *
        "mode; find the mode first (brm_laplace_mode)")
    log_evidence = design === :grid ?
        _brm_lint_logsumexp(v[finite]) + d * log(step) + sum(log, 1 ./ sqrt.(eig.values)) :
        nothing
    marginals = _brm_lint_marginals(p, design, Z, v, w, Θ, μ, σ, θstar, A,
        Float64(step), Int(marginal_points))
    latent = latent_marginals === :none ? nothing :
        _brm_lint_latent_marginals(p, Θ, w, summaries, latent_marginals)
    centers = latent_marginals === :mean_corrected && conditional == 0 ? summaries : nothing
    packed, newton_steps = _brm_lint_draws(p, Θ, w, Int(draws), rng, conditional, states, centers)
    isempty(evaluated.truncated) || @warn("brm_laplace_integration: the grid walk along " *
        "standardized axes $(evaluated.truncated) stopped at its step limit before the log " *
        "marginal dropped by max_log_drop; the marginal may be improper or very flat there")
    diagnostics = (; evaluations=calls[], nonfinite=count(!, finite),
        max_gain=gain, truncated=evaluated.truncated, newton_steps)
    (; draws=packed, point=θstar, design, z=Z, points=Θ, log_density=v, weights=w,
        ess=1 / sum(abs2, w), mean=μ, sd=σ, marginals, log_evidence,
        standardization=(; hessian=H, transform=A), diagnostics, latent_marginals=latent)
end

_brm_lint_logsumexp(x) = isempty(x) ? -Inf : (m = maximum(x); m + log(sum(exp.(x .- m))))

# The grid: walk each half-axis, then fill the product of the walked ranges
# with every point within `max_log_drop` of the centre.
function _brm_lint_grid(value, v0, d, step, max_log_drop; max_steps=50, max_points=10_000)
    known = Dict{NTuple{d,Int},Float64}(ntuple(_ -> 0, d) => v0)
    lo = zeros(Int, d); hi = zeros(Int, d)
    truncated = Int[]
    for j in 1:d, dir in (-1, 1)
        s = 1
        while true
            if s > max_steps
                push!(truncated, j)
                break
            end
            key = ntuple(i -> i == j ? dir * s : 0, d)
            v = value(step .* collect(key))
            (isfinite(v) && v0 - v <= max_log_drop) || break
            known[key] = v
            dir < 0 ? (lo[j] = -s) : (hi[j] = s)
            s += 1
        end
    end
    total = prod(hi .- lo .+ 1)
    total <= max_points || throw(ArgumentError(
        "brm_laplace_integration: the grid spans $total points, above $max_points; use " *
        "design = :ccd, a larger step or a smaller max_log_drop"))
    keys = NTuple{d,Int}[]
    values = Float64[]
    for idx in Iterators.product((lo[j]:hi[j] for j in 1:d)...)
        key = NTuple{d,Int}(idx)
        v = get(known, key) do
            value(step .* collect(key))
        end
        (isfinite(v) && v0 - v <= max_log_drop) || continue
        push!(keys, key); push!(values, v)
    end
    Z = Float64[step * k[j] for j in 1:d, k in keys]
    (; z=Z, log_density=values, log_weight=zeros(length(values)), truncated=unique(truncated))
end

# The CCD of Rue, Martino and Chopin (2009, §6.5): the centre, 2d axial points
# and the factorial points, all non-centre points at radius f0·√d. Weighted
# with log Δ_k, which makes the self-normalized rule exact for ∫1 and ∫zᵀz
# under a standard Gaussian.
function _brm_lint_ccd(value, v0, d, f0)
    # in one dimension the factorial points are the axial points
    F = d == 1 ? zeros(1, 0) : _brm_ccd_fraction(d)
    r = f0 * sqrt(d)
    Z = hcat(zeros(d), (s * r * I(d)[:, j] for j in 1:d for s in (1, -1))..., f0 .* F)
    n = size(Z, 2)
    ws = 1 / ((n - 1) * (1 + exp(-r^2 / 2) * (r^2 / d - 1)))
    wc = 1 - (n - 1) * ws
    v = [k == 1 ? v0 : value(Z[:, k]) for k in 1:n]
    logΔ = [k == 1 ? log(wc) : log(ws) for k in 1:n]
    (; z=Z, log_density=v, log_weight=logΔ, truncated=Int[])
end

# The factorial part of the CCD as columns of ±1: the full factorial for
# d ≤ 4, otherwise the smallest regular 2^(d-k) fraction of resolution V or
# higher (every word of the defining relation has length ≥ 5, so main effects
# and two-factor interactions are unaliased), found by a deterministic search
# over generators that are products of four or more base factors.
function _brm_ccd_fraction(d::Integer)
    d <= 4 && return _brm_ccd_factorial(d, Int[])
    for m in 4:d
        k = d - m
        k == 0 && return _brm_ccd_factorial(d, Int[])
        candidates = [s for s in 1:(2^m - 1) if count_ones(s) >= 4]
        gens = _brm_ccd_generators(candidates, k, Int[])
        gens === nothing || return _brm_ccd_factorial(m, gens)
    end
end

function _brm_ccd_generators(candidates, k, chosen)
    length(chosen) == k && return chosen
    start = isempty(chosen) ? 1 : findfirst(==(last(chosen)), candidates) + 1
    for c in candidates[start:end]
        ok = true
        # every product of the new word with a subset of the chosen words
        for subset in 0:(2^length(chosen) - 1)
            mask = c
            for (i, g) in enumerate(chosen)
                (subset >> (i - 1)) & 1 == 1 && (mask ⊻= g)
            end
            if count_ones(mask) + count_ones(subset) + 1 < 5
                ok = false
                break
            end
        end
        ok || continue
        found = _brm_ccd_generators(candidates, k, [chosen; c])
        found === nothing || return found
    end
    nothing
end

# Columns: all 2^m sign patterns of the m base factors, then one row per
# generated factor, the product of the base factors its mask names.
function _brm_ccd_factorial(m::Integer, gens::Vector{Int})
    F = Matrix{Float64}(undef, m + length(gens), 2^m)
    for col in 0:(2^m - 1)
        for j in 1:m
            F[j, col + 1] = (col >> (j - 1)) & 1 == 0 ? -1.0 : 1.0
        end
        for (g, mask) in enumerate(gens)
            F[m + g, col + 1] = prod(F[j, col + 1] for j in 1:m if (mask >> (j - 1)) & 1 == 1)
        end
    end
    F
end

# One density table per population coordinate. A single coordinate on a grid
# interpolates the log marginal between its grid points. Otherwise each table
# is a Gaussian kernel mixture over the weighted design points, shrunk towards
# their mean so that its mean and variance are the design's (West 1993): on a
# grid each kernel has the variance of its cell projected on the coordinate,
# on a CCD half the design variance. R-INLA's and Latte's skewed Gaussian in z
# was measured instead and rejected: it sees asymmetry only along the axes, so
# on the d = 3 test (test/laplace_integration.jl) it put the log scale's
# marginal mean at its mode, 0.19 sd from the exact mean, where the design's
# own weighted mean was within 0.07 sd.
function _brm_lint_marginals(p, design, Z, v, w, Θ, μ, σ, θstar, A, step, M)
    d = length(θstar)
    if d == 1 && design === :grid
        order = sortperm(vec(Z))
        x = θstar[1] .+ A[1, 1] .* vec(Z)[order]
        lv = v[order]
        keep = isfinite.(lv)
        x, lv = x[keep], lv[keep]
        A[1, 1] < 0 && (x = reverse(x); lv = reverse(lv))
        grid = collect(range(first(x), last(x); length=M))
        logd = [_brm_lint_interpolate(x, lv, t) for t in grid]
        return [_brm_lint_table(p.names[1], grid, exp.(logd .- maximum(logd)))]
    end
    map(1:d) do i
        cell = step^2 * sum(abs2, view(A, i, :)) / 12
        s2 = σ[i]^2
        fraction = s2 > 0 ? (design === :grid ? min(cell / s2, 0.5) : 0.5) : 1.0
        a = sqrt(1 - fraction)
        bandwidth = s2 > 0 ? sqrt(fraction * s2) : sqrt(max(cell, eps()))
        centers = a .* view(Θ, i, :) .+ (1 - a) * μ[i]
        used = findall(>(0), w)
        lo = minimum(centers[used]) - 6bandwidth
        hi = maximum(centers[used]) + 6bandwidth
        grid = collect(range(lo, hi; length=M))
        density = [sum(w[k] * exp(-((t - centers[k]) / bandwidth)^2 / 2) for k in used)
            for t in grid]
        _brm_lint_table(p.names[i], grid, density)
    end
end

# Quadratic interpolation through the three grid points nearest `t`, exact
# for a Gaussian's log density (a linear one was measured 4% off in density
# midway between points 0.75 standard deviations apart).
function _brm_lint_interpolate(x, y, t)
    n = length(x)
    n == 1 && return first(y)
    n == 2 && return y[1] + (y[2] - y[1]) * (t - x[1]) / (x[2] - x[1])
    k = clamp(searchsortedlast(x, t), 1, n - 1)
    j = clamp(abs(t - x[k]) <= abs(x[k + 1] - t) ? k - 1 : k, 1, n - 2)
    a, b, c = x[j], x[j + 1], x[j + 2]
    y[j] * (t - b) * (t - c) / ((a - b) * (a - c)) +
        y[j + 1] * (t - a) * (t - c) / ((b - a) * (b - c)) +
        y[j + 2] * (t - a) * (t - b) / ((c - a) * (c - b))
end

# Normalize by the trapezoid rule; moments from the table.
function _brm_lint_table(name, x, unnormalized)
    h = length(x) > 1 ? x[2] - x[1] : 1.0
    mass = h * (sum(unnormalized) - (first(unnormalized) + last(unnormalized)) / 2)
    density = unnormalized ./ mass
    w = density .* h
    w[1] /= 2; w[end] /= 2
    m = sum(w .* x)
    (; name, x, density, mean=m, sd=sqrt(max(sum(w .* (x .- m) .^ 2), 0.0)))
end

# R-INLA's sampler: draw design points by weight, then the integrated effects
# at each from its conditional Gaussian. The design already ran each point's
# mode search, so the draws reuse its state (`states`, by point) where it was
# kept, and otherwise search the point again: equal points are drawn together,
# so that search runs once per point. The columns are then shuffled. Either
# way the draws are the same numbers: a point's mode search does not depend on
# earlier ones (`_brm_spl_mode!`).
function _brm_lint_draws(p, Θ, w, n, rng, conditional, states=Dict{Vector{Float64},Any}(), centers=nothing)
    n == 0 && return Matrix{Float64}(undef, length(p.u), 0), Int[]
    c = cumsum(w)
    picks = sort!([min(searchsortedfirst(c, rand(rng) * c[end]), length(w)) for _ in 1:n])
    out = _brm_lint_draws_at(p, Θ, picks, rng, conditional, states, centers)
    order = Random.randperm(rng, n)
    out.draws[:, order], out.diagnostics.newton_steps[order]
end

_brm_lint_draws_at(p, Θ, picks, rng, conditional, states, centers) =
    brm_laplace_draws(p, Θ[:, picks]; rng, conditional)
function _brm_lint_draws_at(p::BRMSparseLaplaceProblem, Θ, picks, rng, conditional, states, centers)
    P = Θ[:, picks]
    function state_at(k)
        state = get(states, P[:, k], nothing)
        state === nothing ? _brm_spl_mode!(p, view(P, :, k)) : state
    end
    mean_at = centers === nothing ? nothing : k -> first(centers[P[:, k]])
    _brm_spl_draws(p, length(picks), state_at; rng, conditional, mean_at)
end

# The Laplace marginal at a design point. A sparse problem's mode search there
# starts from the integrated effects' modes at the design's centre (`start`,
# `_brm_spl_search!`): neighbouring points' modes are close, so it takes about
# half the Newton steps of a search from the prior's mode (epil, 301 effects:
# 4.1 instead of 8.1 per point, the marginal within 4e-12; snag
# `brm-laplace-inte-e0991a0f`). The start is fixed for the whole design, so
# each point's value does not depend on the order the design visits them.
_brm_lint_start(p) = nothing
_brm_lint_start(p::BRMSparseLaplaceProblem) =
    p.last === nothing || !p.last.state.finite ? nothing : copy(p.last.state.z)
_brm_lint_logdensity(p, φ, start) = LogDensityProblems.logdensity(p, φ)
_brm_lint_logdensity(p::BRMSparseLaplaceProblem, φ, start) =
    _brm_spl_value(p, _brm_spl_mode!(p, φ; start))

# The last mode search's state, kept for drawing at its point
# (`_brm_spl_draw_state`); `nothing` where the problem keeps none.
_brm_lint_draw_state(p) = nothing
_brm_lint_draw_state(p::BRMSparseLaplaceProblem) =
    p.last === nothing ? nothing : _brm_spl_draw_state(p, p.last.state)

# Conditional variances on the operator's large components are estimated
# with its fixed white probes. Squaring the covariance-root actions estimates
# diag(H⁻¹), including when the logdet uses a signed control variate.
function _brm_lint_variance(p, state, selection)
    F = state.F
    if F isa _BRMOperatorFactor
        variances = copy(_brm_operator_logdet!(F).sensitivity[p.diagonal])
        for (part, ids) in zip(F.parts, F.components)
            part.factor === nothing || continue
            v = view(variances, ids)
            fill!(v, 0.0)
            root = similar(v)
            for probe in eachcol(part.probes)
                _brm_operator_root!(root, F, part, probe)
                v .+= abs2.(root)
            end
            v ./= size(part.probes, 2)
        end
        return variances
    end
    selection === nothing || return selection[p.diagonal]
    if F isa _BRMAugmentedFactor
        return diag(_brm_spl_selected(F))[1:length(p.integrated)]
    end
    variance = _brm_selinv_diag(F)
    if state.W !== nothing
        X = state.W.X
        Y = state.W.G \ X'
        for j in axes(X, 1)
            variance[j] -= dot(view(X, j, :), view(Y, :, j))
        end
    end
    variance
end

_brm_lint_conditional(p::_BRMSampledLaplaceProblem, method) = (Float64[], Float64[])
function _brm_lint_conditional(p::BRMSparseLaplaceProblem, method)
    state = p.last.state
    selection = method === :mean_corrected ? _brm_spl_conditional_selection(p, state) : nothing
    mu = copy(state.z)
    if method === :mean_corrected
        shift = _brm_spl_mean_shift(p, state, selection, Returns(true))
        shift.bad === nothing || _brm_nlme_error("latent mean correction failed at " * shift.bad)
        mu .+= shift.v
    end
    variance = _brm_lint_variance(p, state, selection)
    all(v -> isfinite(v) && v > 0, variance) || _brm_nlme_error(
        "brm_laplace_integration: nonpositive or nonfinite conditional variance")
    mu, sqrt.(variance)
end

function _brm_lint_latent_marginals(p, points, w, summaries, method)
    point_indices = findall(>(0), w)
    weights = w[point_indices]
    mixing = Categorical(weights)
    n = length(p.integrated)
    conditional_mean = zeros(n, length(weights))
    conditional_sd = similar(conditional_mean)
    for (k, j) in enumerate(point_indices)
        mu, sd = summaries[points[:, j]]
        conditional_mean[:, k] .= mu
        conditional_sd[:, k] .= sd
    end
    names = _rk_layout_coordinate_names(p.backend)
    if p isa BRMSparseLaplaceProblem && p.walk_frame !== nothing
        names = copy(names)
        for block in p.walk_frame.blocks, (j, i) in enumerate(block.ids)
            names[i] = Symbol(block.declaration, ".phi.", j)
        end
    end
    distributions = [MixtureModel(Normal.(view(conditional_mean, i, :),
        view(conditional_sd, i, :)), mixing) for i in 1:n]
    (; method, names=names[p.integrated], point_indices, weights, conditional_mean, conditional_sd,
        distributions, mean=mean.(distributions), sd=std.(distributions))
end
