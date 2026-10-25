# src/nlme_svi.jl — minibatch stochastic variational inference over an
# RK-lowered population model's own packed coordinates.
#
# The target is the RK sampler density p(u) over the packed unconstrained point
# u = (g, z_1, …, z_N): g holds every coordinate that is not a subject draw
# (population coefficients, estimated shared latents, scalars, random-effect
# scales and correlation factors) and z_i is subject i's standardized draw
# block. Given g the subjects are conditionally independent:
#
#     log p(u) = P(g) + Σ_i [log N(z_i; 0, I) + ℓ_i(g, z_i)],
#
# with P the authored priors and coordinate Jacobians at z = 0 (RK's `:prior`
# and `:log_jacobian` queries minus the subject draws' N(0, 1) constants) and
# ℓ_i the subject's observation cells. A batch of B subjects, its terms scaled
# by N/B, is an unbiased estimate of that sum, and of its gradient.
#
# The variational family is Gaussian and factorizes into a global factor and
# one factor per subject:
#
#     g   = μ + exp.(ρ) .* ε + Β η,         ε ~ N(0, I), η ~ N(0, I_r)
#     z_i ~ N(m_i, Λ_i⁻¹)
#
# The global covariance is diagonal (`MeanFieldGlobal`) or diagonal plus rank r
# (`LowRankGlobal(r)`, Β is D × r); Adam moves it every iteration along the
# minibatch ELBO gradient. A subject's factor moves only when the subject is in
# the batch, by a natural-gradient step: the improved Bayesian learning rule of
# Lin, Schmidt and Khan (2020), whose curvature estimate comes from the same
# reparameterized draw (Stein's lemma) and whose precision update stays
# positive definite. That step is Newton-like, so a subject's factor settles in
# a few visits, where plain gradient steps on its mean and scale need hundreds.

"""
    MeanFieldGlobal()
    LowRankGlobal(rank)

The global variational factor of [`brm_svi`](@ref): a diagonal Gaussian, or a
diagonal plus a rank-`rank` factor.
"""
abstract type BRMSVIFamily end
struct MeanFieldGlobal <: BRMSVIFamily end
struct LowRankGlobal <: BRMSVIFamily
    rank::Int
    LowRankGlobal(rank::Integer) = rank >= 1 ? new(rank) :
        throw(ArgumentError("LowRankGlobal: rank must be at least 1, got $rank"))
end
_brm_svi_rank(::MeanFieldGlobal) = 0
_brm_svi_rank(f::LowRankGlobal) = f.rank

# ── The minibatch target ─────────────────────────────────────────────────────

# Packed positions of the global coordinates: everything the NLME model
# estimates as θ or σ, and the covariance coordinates the NLME reading leaves to
# its estimators. Fixed shared latents (`shared_point`) stay at `m.unit`.
_brm_svi_globals(m::BRMNLMEModel) = sort!(vcat(m.theta, m.sigma,
    findall(r -> r === :subject_scale || r === :subject_correlation, m.view.role)))

function _brm_svi_target(m::BRMNLMEModel)
    m.cell_query === nothing && _brm_nlme_error(
        "brm_svi: minibatch SVI evaluates one subject at a time, which needs a " *
        "subject-indexed observation cell (`@plate for` over the subjects); " *
        "flat-row grouped observations have none")
    _brm_nlme_ragged(m.view) && _brm_nlme_error(
        "brm_svi: nested (ragged) subject blocks are not covered yet")
    globals = _brm_svi_globals(m)
    units = m.view.subject_coordinates
    d = length(first(units))
    all(u -> length(u) == d, units) || error(
        "brm_svi: internal: flat subject blocks differ in length")
    # One prepared compact cell gradient per cell group, selecting the globals
    # and one subject's draws; the subject's positions change per call.
    selected = sort!(vcat(globals, first(units)))
    samplers = map(m.cell_query) do cell
        _rk_nlme_compact_cell_sampler(m.backend, cell.names, m.unit, selected)
    end
    # For subject i, `order[i]` sorts [globals; units[i]] into packed order.
    order = [sortperm(vcat(globals, block)) for block in units]
    indices = [vcat(globals, block)[o] for (block, o) in zip(units, order)]
    nz = count(==(:subject_effect), m.view.role)
    (; m, globals, units, d, samplers, order, indices,
       z_constant=-nz * log(2π) / 2, prior_point=copy(m.unit),
       active=zeros(length(selected)), cell_gradient=zeros(length(selected)),
       gradient=zeros(length(selected)))
end

# P(g) and its gradient: the RK prior and log-Jacobian at the packed point with
# every subject draw at zero, minus those draws' N(0, 1) constants.
function _brm_svi_global_term(t, g)
    u = t.prior_point
    u[t.globals] .= g
    value, gradient = _rk_nlme_population_prior_value_gradient(t.m.population_prior, u)
    value - t.z_constant, gradient[t.globals]
end

# ℓ_i(g, z) + log N(z; 0, I) for subject i, with its gradients in g and z.
function _brm_svi_unit_term!(gg, gz, t, i, g, z)
    D = length(t.globals)
    order = t.order[i]
    joint = vcat(g, z)
    t.active .= joint[order]
    value = 0.0
    fill!(t.gradient, 0.0)
    for (cell, sampler) in zip(t.m.cell_query, t.samplers), p in cell.positions[i]
        value += first(_rk_nlme_compact_cell_value_and_gradient!(sampler,
            t.cell_gradient, t.m.unit, t.active, t.indices[i], p))
        t.gradient .+= t.cell_gradient
    end
    joint_gradient = similar(joint)
    joint_gradient[order] .= t.gradient
    gg .= view(joint_gradient, 1:D)
    gz .= view(joint_gradient, D+1:length(joint)) .- z
    value - sum(abs2, z) / 2 - length(z) * log(2π) / 2
end

# ── Variational parameters ───────────────────────────────────────────────────

function _brm_svi_params(t, μ, family::BRMSVIFamily; global_scale, unit_scale)
    D, d, N = length(t.globals), t.d, length(t.units)
    Λ = repeat(Matrix{Float64}(I, d, d) ./ unit_scale^2, 1, 1, N)
    (; family, μ=Vector{Float64}(μ), ρ=fill(log(global_scale), D),
       Β=zeros(D, _brm_svi_rank(family)), M=zeros(d, N), Λ,
       U=repeat(Matrix{Float64}(I, d, d) ./ unit_scale, 1, 1, N))
end

_brm_svi_global_covariance(q) = Diagonal(exp.(2 .* q.ρ)) + q.Β * q.Β'

# A draw of the globals from its two standard normal parts.
_brm_svi_draw_global(q, ε, η) = q.μ .+ exp.(q.ρ) .* ε .+ q.Β * η
# A draw of subject i from ε_i: m_i + U_i⁻ᵀ ε_i, with U_i the lower Cholesky
# factor of the precision Λ_i.
_brm_svi_draw_unit(q, i, εi) =
    view(q.M, :, i) .+ LowerTriangular(view(q.U, :, :, i))' \ εi

# The entropy of q, up to the constant (D + N d)(1 + log 2π)/2.
_brm_svi_entropy(q) = logdet(_brm_svi_global_covariance(q)) / 2 -
    sum(log, (q.U[k, k, i] for k in axes(q.U, 1), i in axes(q.U, 3)))

# ── The global factor: Adam ──────────────────────────────────────────────────

mutable struct _BRMSVIAdam
    lr::Float64
    β1::Float64
    β2::Float64
    m::Vector{Float64}
    v::Vector{Float64}
    t::Int
end
_BRMSVIAdam(n, lr) = _BRMSVIAdam(lr, 0.9, 0.999, zeros(n), zeros(n), 0)

# One ascent step of `x` along `g`.
function _brm_svi_step!(opt::_BRMSVIAdam, x, g, lr=opt.lr)
    opt.t += 1
    @. opt.m = opt.β1 * opt.m + (1 - opt.β1) * g
    @. opt.v = opt.β2 * opt.v + (1 - opt.β2) * g^2
    b1, b2 = 1 - opt.β1^opt.t, 1 - opt.β2^opt.t
    @. x += lr * (opt.m / b1) / (sqrt(opt.v / b2) + 1e-8)
    x
end

# The global parameters as one vector [μ; ρ; vec(Β)].
_brm_svi_global_vector(q) = vcat(q.μ, q.ρ, vec(q.Β))
function _brm_svi_set_global!(q, x)
    D = length(q.μ)
    q.μ .= view(x, 1:D)
    q.ρ .= view(x, D+1:2D)
    vec(q.Β) .= view(x, 2D+1:length(x))
    q
end

# The gradient of the global vector from w, the ELBO's gradient in the global
# coordinates, the draw's (ε, η), and the entropy's own gradient.
function _brm_svi_global_gradient(q, w, ε, η)
    Σinv = inv(Symmetric(Matrix(_brm_svi_global_covariance(q))))
    s = exp.(q.ρ)
    gρ = w .* ε .* s .+ diag(Σinv) .* s .^ 2
    gΒ = w * η' .+ Σinv * q.Β
    vcat(w, gρ, vec(gΒ))
end

# A scale-preconditioned ("whitened") global step for polishing: μ moves in
# units of the global covariance's Cholesky factor and each Β row in units of
# its scale, so Adam's step size means a fraction of the current posterior
# scale, which stays meaningful when a global's posterior sd shrinks like 1/√N.
function _brm_svi_whitened_global_step!(opt, q, x, grad, lr)
    D = length(q.μ)
    C = Matrix(cholesky(Symmetric(Matrix(_brm_svi_global_covariance(q)))).L)
    scale = vcat(ones(D), vec(repeat(exp.(q.ρ), 1, size(q.Β, 2))))
    grad = copy(grad)
    grad[1:D] .= C' * view(grad, 1:D)
    rest = D+1:length(x)
    grad[rest] .*= scale
    step = zeros(length(x))
    _brm_svi_step!(opt, step, grad, lr)
    x[1:D] .+= C * view(step, 1:D)
    x[rest] .+= scale .* view(step, rest)
    x
end

# ── A subject's factor: one natural-gradient step ───────────────────────────

# The improved Bayesian learning rule step of size β for subject i, from the
# gradient `gz` of log p at its draw z = m + U⁻ᵀ ε. Stein's lemma gives
# E[∇ log p εᵀ] = E[∇² log p] U⁻ᵀ, so -sym(gz εᵀ Uᵀ) estimates the negative
# Hessian; with G = that estimate - Λ the precision moves to
# Λ + β G + β²/2 G Λ⁻¹ G = ½(Λ + (Λ + βG) Λ⁻¹ (Λ + βG)), positive definite for
# any G, and the mean to m + β Λ_new⁻¹ gz.
function _brm_svi_unit_step!(q, i, gz, εi, β)
    Λ = Symmetric(q.Λ[:, :, i])
    U = LowerTriangular(view(q.U, :, :, i))
    H = gz * (U * εi)'
    G = -(H + H') ./ 2 .- Λ
    Λnew = Symmetric(Λ .+ β .* G .+ (β^2 / 2) .* (G * (Λ \ G)))
    F = cholesky(Λnew)
    q.Λ[:, :, i] .= Λnew
    q.U[:, :, i] .= F.L
    q.M[:, i] .+= β .* (F \ gz)
    q
end

# ── One minibatch iteration ──────────────────────────────────────────────────

# One draw of the globals, one draw per batch subject, the minibatch ELBO
# gradient, an Adam step of the globals and a natural-gradient step of each
# batch subject. Returns the minibatch estimate of E_q log p (without the
# entropy).
function _brm_svi_iteration!(rng, t, q, global_opt, batch, lr_global, β;
        whiten=false)
    D, d, N = length(t.globals), t.d, length(t.units)
    s = N / length(batch)
    ε, η = randn(rng, D), randn(rng, size(q.Β, 2))
    g = _brm_svi_draw_global(q, ε, η)
    P, gP = _brm_svi_global_term(t, g)
    w = copy(gP)
    gg, gz = zeros(D), zeros(d)
    batch_value = 0.0
    steps = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, length(batch))
    for (k, i) in enumerate(batch)
        εi = randn(rng, d)
        batch_value += _brm_svi_unit_term!(gg, gz, t, i, g, _brm_svi_draw_unit(q, i, εi))
        w .+= s .* gg
        steps[k] = (copy(gz), εi)
    end
    x = _brm_svi_global_vector(q)
    grad = _brm_svi_global_gradient(q, w, ε, η)
    whiten ? _brm_svi_whitened_global_step!(global_opt, q, x, grad, lr_global) :
        _brm_svi_step!(global_opt, x, grad, lr_global)
    _brm_svi_set_global!(q, x)
    for (k, i) in enumerate(batch)
        _brm_svi_unit_step!(q, i, steps[k]..., β)
    end
    P + s * batch_value
end

# `B` distinct subjects drawn uniformly from `1:N` in O(B) (Floyd's algorithm),
# so choosing a batch does not scale with the number of subjects.
function _brm_svi_batch(rng, N, B)
    chosen = Set{Int}()
    sizehint!(chosen, B)
    for j in N-B+1:N
        k = rand(rng, 1:j)
        push!(chosen, k in chosen ? j : k)
    end
    collect(chosen)
end

const _BRM_SVI_PHASES = (lower="Lower to ReactiveKernels", prepare="Prepare NLME model",
    target="Minibatch target", fit="Stochastic optimization", draws="Draws from q")

# Stochastic optimization in two phases. Exploration runs at constant step
# sizes (`lr_global` for Adam, `unit_step` for the subjects' natural-gradient
# steps) until the ELBO stops improving detectably: the mean over a window (one
# pass over the subjects, at least 100 iterations) has risen, over the last
# `patience` windows, by less than twice the standard error of that rise, after
# at least `min_windows`. Comparing across several windows rather than
# neighbouring ones lets a slow drift show above the minibatch noise. Polishing then restarts Adam on scale-preconditioned global
# steps (`polish_lr`, a fraction of the current scale), decays every step size
# as 1/√k over `polish_windows` windows, and returns the average of q over
# those iterations (Polyak–Ruppert), which damps the noise a stochastic step
# leaves.
function _brm_svi_fit!(rng, t, q; batch_size, max_iterations, lr_global, unit_step,
        polish_lr, window, patience, min_windows, polish_windows, node=nothing)
    N = length(t.units)
    adam(lr) = _BRMSVIAdam(length(_brm_svi_global_vector(q)), lr)
    global_opt = adam(lr_global)
    B = min(batch_size, N)
    W = window === nothing ? max(cld(N, B), 100) : window
    trace = Float64[]
    windows = NamedTuple{(:mean, :var),Tuple{Float64,Float64}}[]
    polish_start = 0
    average = nothing
    for it in 1:max_iterations
        polishing = polish_start > 0
        scale = polishing ? 1 / sqrt(1 + (it - polish_start) / W) : 1.0
        value = _brm_svi_iteration!(rng, t, q, global_opt, _brm_svi_batch(rng, N, B),
            scale * (polishing ? polish_lr : lr_global), scale * unit_step;
            whiten=polishing)
        push!(trace, value + _brm_svi_entropy(q))
        if polishing
            k = it - polish_start
            for name in (:μ, :ρ, :Β, :M, :Λ)
                a, x = getfield(average, name), getfield(q, name)
                @. a += (x - a) / k
            end
            k >= polish_windows * W && return (; q=_brm_svi_factor!(average),
                elbo=trace, elbo_windows=first.(windows), converged=true,
                iterations=it)
        end
        it % W == 0 || continue
        w = view(trace, it-W+1:it)
        push!(windows, (; mean=Statistics.mean(w), var=Statistics.var(w)))
        _brm_throw_if_interrupted(node)
        _brm_update_progress!(node, string(polishing ?
            "polishing $(it - polish_start)/$(polish_windows * W)" :
            "exploring, window $(length(windows))", " · iteration $it · ELBO ≈ ",
            round(last(windows).mean; sigdigits=6)))
        if !polishing && length(windows) >= max(min_windows, patience + 1)
            a, b = windows[end-patience], windows[end]
            stalled = b.mean - a.mean < 2 * sqrt((a.var + b.var) / W)
            if stalled
                polish_start = it
                average = deepcopy(q)
                global_opt = adam(polish_lr)
            end
        end
    end
    (; q=average === nothing ? q : _brm_svi_factor!(average), elbo=trace,
       elbo_windows=first.(windows), converged=false, iterations=max_iterations)
end

# The subjects' Cholesky factors of an averaged precision.
function _brm_svi_factor!(q)
    for i in axes(q.Λ, 3)
        q.U[:, :, i] .= cholesky(Symmetric(q.Λ[:, :, i])).L
    end
    q
end

# `n` draws from q as packed RK points, one per column.
function _brm_svi_draws(rng, t, q, n)
    U = repeat(t.m.unit, 1, n)
    for j in 1:n
        U[t.globals, j] .= _brm_svi_draw_global(q, randn(rng, length(q.μ)),
            randn(rng, size(q.Β, 2)))
        for (i, block) in enumerate(t.units)
            U[block, j] .= _brm_svi_draw_unit(q, i, randn(rng, t.d))
        end
    end
    U
end

# The model through its preparation phases, as for `brm_nlme_fit`: lowering to
# RK and `brm_nlme_model`. A stage `x` has already passed is reused.
_brm_svi_lowered(x::BRMI, phase) = _brm_run_phase(_ -> RKBRMI(x), phase)
_brm_svi_lowered(x::Union{RKBRMI,BRMNLMEModel}, phase) = (_brm_reuse_phase!(phase); x)
_brm_svi_lowered(x, phase) = throw(ArgumentError(
    "brm_svi_fit: expected a @brm model (BRMI), an RKBRMI or a BRMNLMEModel, " *
    "got $(typeof(x))"))

_brm_svi_model(x::RKBRMI, phase; ad_backend, shared_point) =
    _brm_run_phase(phase) do node
        _brm_nlme_model(x, node; ad_backend=ad_backend === nothing ?
            _brm_svi_default_ad(x) : ad_backend, observation_ad_backend=nothing,
            shared_point, subject_group=nothing, nested_groups=NamedTuple())
    end
function _brm_svi_model(m::BRMNLMEModel, phase; ad_backend, shared_point)
    ad_backend === nothing && shared_point === nothing || throw(ArgumentError(
        "brm_svi_fit: a prepared BRMNLMEModel already fixes its ad_backend and " *
        "shared_point; pass them to brm_nlme_model instead"))
    _brm_reuse_phase!(phase)
    m
end

# Implemented by `BayesianRegressionModelsReactiveKernelsEnzymeExt`: reverse-mode
# `AutoEnzyme`, the backend `brm_nlme_model` requires.
_brm_svi_default_ad(x) = throw(ArgumentError(
    "brm_svi_fit: pass `ad_backend` (reverse-mode AutoEnzyme), or load Enzyme " *
    "and DifferentiationInterface for that default"))

# The packed point the globals' means start from: the prior medians, a full
# packed point `start`, or address-keyed `inits` over the prior medians.
function _brm_svi_start(m::BRMNLMEModel, start, inits)
    start === nothing || inits === nothing || throw(ArgumentError(
        "brm_svi_fit: give either `start` or `inits`, not both"))
    median = first(_brm_nlme_prior_median_point(m))
    start === nothing || return (length(start) == length(median) ||
        throw(DimensionMismatch("brm_svi_fit: `start` has $(length(start)) " *
            "coordinates, the RK layout has $(length(median))")); Vector{Float64}(start))
    inits === nothing && return median
    brm_start(m.backend, inits; base=median)
end

"""
    brm_svi_fit(x; batch_size=32, rng=Random.default_rng(),
        family=MeanFieldGlobal(), ndraws=1000, start=nothing, inits=nothing,
        ad_backend=nothing, shared_point=nothing, progress=nothing, kwargs...)
        -> (; draws, coordinates, diagnostics)

Minibatch stochastic variational inference for a model whose subjects are
conditionally independent given the population (the subjects of
[`brm_nlme_view`](@ref)), over the RK backend's packed unconstrained
coordinates. Each iteration evaluates `batch_size` subjects, through their own
observation cells, and scales them to all subjects, so its cost follows the
batch size, not the number of subjects.

- `x` — a `@brm` model ([`BRMI`](@ref)), its [`RKBRMI`](@ref), or a prepared
  [`BRMNLMEModel`](@ref). For a BRMI or RKBRMI the model is prepared with
  `ad_backend` (default reverse-mode `AutoEnzyme`, which needs Enzyme loaded)
  and `shared_point`, as [`brm_nlme_model`](@ref) takes them. The observations
  must lie in subject-indexed cells (`@plate for` over the subjects) with flat
  (not nested) subject blocks.
- `family` — the global factor: `MeanFieldGlobal()` (diagonal) or
  `LowRankGlobal(r)` (diagonal plus rank `r`). Each subject has its own
  full-covariance Gaussian factor.
- `start` / `inits` — where the globals' means start: the prior medians by
  default, a full RK packed point, or initial values keyed by prior address as
  [`brm_start`](@ref) takes them (the rest at the prior medians).
- `progress` — a Treebars node or backend symbol: one `BRM SVI fit` node with
  the phases Lower to ReactiveKernels, Prepare NLME model, Minibatch target
  (the subjects' cell gradients and the start), Stochastic optimization (its
  message names the window, iteration and ELBO) and Draws from q. An interrupt raises `Treebars.ProgressInterrupted` at the
  next window.
- Optimizer controls: `max_iterations`, `lr_global` (Adam on the globals),
  `unit_step` (the subjects' natural-gradient step), `polish_lr`, `window`,
  `patience`, `min_windows`, `polish_windows`, `global_scale` and `unit_scale`
  (initial factor scales).

Returns `draws`, `ndraws` draws from the fitted approximation as RK packed
unconstrained points (coordinates × draws, one column per draw, the order of
`coordinate_names(rk.model.layout)`), their `coordinates`, and `diagnostics`:
the ELBO trace (`elbo`, `elbo_windows`, up to a constant), `converged`,
`iterations`, the variational parameters `q`, the packed positions of the
global coordinates (`globals`), `seconds` of optimization, and the prepared
`model`. The approximation is variational: posterior means are close, and
the spread of the population, scale and correlation coordinates is too narrow.
"""
function brm_svi_fit(x; rng=Random.default_rng(), batch_size=32,
        max_iterations=100_000, family::BRMSVIFamily=MeanFieldGlobal(),
        lr_global=0.01, unit_step=0.1, start=nothing, inits=nothing,
        ad_backend=nothing, shared_point=nothing, global_scale=0.1, unit_scale=1.0,
        ndraws=1000, polish_lr=0.05, window=nothing, patience=10, min_windows=40,
        polish_windows=20, progress=nothing)
    _brm_with_progress(progress; description="BRM SVI fit") do node
        _brm_with_phases(node, _BRM_SVI_PHASES) do phases
            m = _brm_svi_model(_brm_svi_lowered(x, phases.lower), phases.prepare;
                ad_backend, shared_point)
            t, μ = _brm_run_phase(phases.target) do _
                t = _brm_svi_target(m)
                t, _brm_svi_start(m, start, inits)[t.globals]
            end
            q = _brm_svi_params(t, μ, family; global_scale, unit_scale)
            seconds = @elapsed fit = _brm_enter_phase(phases.fit) do phase
                _brm_svi_fit!(rng, t, q; batch_size, max_iterations, lr_global,
                    unit_step, polish_lr, window, patience, min_windows,
                    polish_windows, node=phase)
            end
            draws = _brm_run_phase(_ -> _brm_svi_draws(rng, t, fit.q, ndraws),
                phases.draws)
            (; draws, coordinates=copy(m.view.coordinates), diagnostics=(;
                fit.elbo, fit.elbo_windows, fit.converged, fit.iterations, fit.q,
                globals=t.globals, seconds, model=m))
        end
    end
end
