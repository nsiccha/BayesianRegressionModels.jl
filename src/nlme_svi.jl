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
# antithetic pair of reparameterized draws (Stein's lemma) and whose precision
# update stays positive definite, inside a per-subject trust region on the
# precision's relative change and the mean's move (as Theis and Hoffman (2015)
# bound SVI's natural-gradient steps). That step is Newton-like, so a subject's
# factor settles in tens of visits, where plain gradient steps on its mean and
# scale need hundreds.

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
        "brm_svi: minibatch SVI evaluates one subject at a time, on the " *
        "subject's own observation cells, and this model's observations have " *
        "no cells attributable to single subjects")
    globals = _brm_svi_globals(m)
    units = m.view.subject_coordinates
    # Compact cell gradients selecting the globals and one subject's draws; the
    # subject's positions change per call. Subjects with blocks of one length
    # (all subjects of a flat model) share a sampler set and its workspace.
    (; samplers, of) = _brm_nlme_cell_samplers(m.backend, m.cell_query, m.unit,
        globals, units)
    work = map(eachindex(samplers)) do k
        n = length(globals) + length(units[findfirst(==(k), of)])
        (; active=zeros(n), cell_gradient=zeros(n), gradient=zeros(n))
    end
    # For subject i, `order[i]` sorts [globals; units[i]] into packed order.
    order = [sortperm(vcat(globals, block)) for block in units]
    indices = [vcat(globals, block)[o] for (block, o) in zip(units, order)]
    nz = count(==(:subject_effect), m.view.role)
    (; m, globals, units, samplers, of, work, order, indices,
       z_constant=-nz * log(2π) / 2, prior_point=copy(m.unit))
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
    w = t.work[t.of[i]]
    joint = vcat(g, z)
    w.active .= joint[order]
    value = 0.0
    fill!(w.gradient, 0.0)
    for (cell, sampler) in zip(t.m.cell_query, t.samplers[t.of[i]]),
            p in cell.positions[i]
        value += first(_rk_nlme_compact_cell_value_and_gradient!(sampler,
            w.cell_gradient, t.m.unit, w.active, t.indices[i], p))
        w.gradient .+= w.cell_gradient
    end
    joint_gradient = similar(joint)
    joint_gradient[order] .= w.gradient
    gg .= view(joint_gradient, 1:D)
    gz .= view(joint_gradient, D+1:length(joint)) .- z
    value - sum(abs2, z) / 2 - length(z) * log(2π) / 2
end

# ── Variational parameters ───────────────────────────────────────────────────

# Subject i's factor is `M[i]`, `Λ[i]` and `U[i]`, sized by its own block.
function _brm_svi_params(t, μ, family::BRMSVIFamily; global_scale, unit_scale)
    D = length(t.globals)
    eye(block) = Matrix{Float64}(I, length(block), length(block))
    (; family, μ=Vector{Float64}(μ), ρ=fill(log(global_scale), D),
       Β=zeros(D, _brm_svi_rank(family)), M=[zeros(length(b)) for b in t.units],
       Λ=[eye(b) ./ unit_scale^2 for b in t.units],
       U=[eye(b) ./ unit_scale for b in t.units])
end

_brm_svi_global_covariance(q) = Diagonal(exp.(2 .* q.ρ)) + q.Β * q.Β'

# A draw of the globals from its two standard normal parts.
_brm_svi_draw_global(q, ε, η) = q.μ .+ exp.(q.ρ) .* ε .+ q.Β * η
# A draw of subject i from ε_i: m_i + U_i⁻ᵀ ε_i, with U_i the lower Cholesky
# factor of the precision Λ_i.
_brm_svi_unit_offset(q, i, εi) = LowerTriangular(q.U[i])' \ εi
_brm_svi_draw_unit(q, i, εi) = q.M[i] .+ _brm_svi_unit_offset(q, i, εi)

# The entropy of q, up to the constant (D + Σ d_i)(1 + log 2π)/2.
_brm_svi_entropy(q) = logdet(_brm_svi_global_covariance(q)) / 2 -
    sum(U -> sum(log, diag(U)), q.U; init=0.0)

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

# The largest relative change of a subject's precision in one step: no
# eigenvalue of U⁻¹ β G U⁻ᵀ exceeds it in magnitude, so along every direction
# the precision changes by a factor between ½ and 1 + τ + τ²/2 (2.5).
const _BRM_SVI_UNIT_TRUST = 1.0
# The longest step of a subject's mean, in sds of its factor before the step.
const _BRM_SVI_UNIT_MEAN_TRUST = 1.0

# The improved Bayesian learning rule step of size β for subject i, from the
# gradients of log p at the antithetic draws z± = m ± U⁻ᵀ ε: `gz` their mean,
# which estimates E_q ∇ log p, and `gd` half their difference. Stein's lemma
# gives E[∇ log p εᵀ] = E[∇² log p] U⁻ᵀ, so -sym(gd εᵀ Uᵀ) estimates the
# negative Hessian; the difference cancels the gradient at the mean, which
# otherwise dominates a single draw's estimate whenever m is many posterior sds
# from the subject's mode. With G = that estimate - Λ the precision moves to
# Λ + β G + β²/2 G Λ⁻¹ G = ½(Λ + (Λ + βG) Λ⁻¹ (Λ + βG)), positive definite for
# any G, and the mean to m + β Λ_new⁻¹ gz.
#
# A subject whose observations pin it far more tightly than its factor's
# start (hundreds of rows against the unit start) has a gradient of 1e4 and
# more, and an unbounded step multiplied its precision by up to 1e8 in one
# visit (β²/2 G Λ⁻¹ G dominates) and then lost positive definiteness in
# floating point. Besides the antithetic draws, two changes keep the step
# sound there:
# - a trust region: β shrinks for this step until ‖U⁻¹ β G U⁻ᵀ‖ ≤ τ
#   (`_BRM_SVI_UNIT_TRUST`), so the precision grows by at most 2.5× per visit
#   along the drawn direction, and the mean moves at most one sd of the
#   factor before the step (`_BRM_SVI_UNIT_MEAN_TRUST`): while the precision
#   is still far below the subject's, Λ_new⁻¹ gz is a Newton step with too
#   little curvature, which overshot by hundreds of sds where a log-scale
#   random effect makes the density steep. A subject's step never shortens
#   another subject's.
# - the square-root form: with Λ = U Uᵀ and A = Λ + βG,
#   Λ_new = ½ (Uᵀ; U⁻¹A)ᵀ (Uᵀ; U⁻¹A), so the R of that stacked matrix's QR
#   factorization is √2 times the new Cholesky factor, which is positive
#   definite in floating point too.
function _brm_svi_unit_step!(q, i, gz, gd, εi, β)
    U = LowerTriangular(q.U[i])
    H = gd * (U * εi)'
    G = Symmetric(-(H + H') ./ 2 .- q.Λ[i])
    r = maximum(abs, eigvals(Symmetric(U \ (U \ G)')))
    β = min(β, _BRM_SVI_UNIT_TRUST / r)
    R = qr(vcat(Matrix(U'), U \ (q.Λ[i] .+ β .* G))).R
    L = LowerTriangular(R' * Diagonal(sign.(diag(R))) ./ sqrt(2))
    Δ = β .* (L' \ (L \ gz))
    Δ .*= min(1.0, _BRM_SVI_UNIT_MEAN_TRUST / norm(U' * Δ))
    q.U[i] .= L
    q.Λ[i] .= L * L'
    q.M[i] .+= Δ
    q
end

# ── One minibatch iteration ──────────────────────────────────────────────────

# One draw of the globals, an antithetic pair of draws per batch subject, the
# minibatch ELBO gradient, an Adam step of the globals and a natural-gradient
# step of each batch subject. Returns the minibatch estimate of E_q log p
# (without the entropy).
function _brm_svi_iteration!(rng, t, q, global_opt, batch, lr_global, β;
        whiten=false)
    D, N = length(t.globals), length(t.units)
    s = N / length(batch)
    ε, η = randn(rng, D), randn(rng, size(q.Β, 2))
    g = _brm_svi_draw_global(q, ε, η)
    P, gP = _brm_svi_global_term(t, g)
    w = copy(gP)
    gg₊, gg₋ = zeros(D), zeros(D)
    batch_value = 0.0
    steps = Vector{NTuple{3,Vector{Float64}}}(undef, length(batch))
    for (k, i) in enumerate(batch)
        εi = randn(rng, length(t.units[i]))
        gz₊, gz₋ = zeros(length(εi)), zeros(length(εi))
        m, δ = q.M[i], _brm_svi_unit_offset(q, i, εi)
        batch_value += (_brm_svi_unit_term!(gg₊, gz₊, t, i, g, m .+ δ) +
            _brm_svi_unit_term!(gg₋, gz₋, t, i, g, m .- δ)) / 2
        w .+= (s / 2) .* (gg₊ .+ gg₋)
        steps[k] = ((gz₊ .+ gz₋) ./ 2, (gz₊ .- gz₋) ./ 2, εi)
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
                _brm_svi_average!(getfield(average, name), getfield(q, name), k)
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

# The running mean `a` of iterates `x`, at the k-th: an array, or one array
# per subject.
_brm_svi_average!(a::AbstractArray{<:Real}, x, k) = (@. a += (x - a) / k; a)
_brm_svi_average!(a::AbstractVector{<:AbstractArray}, x, k) =
    (foreach((ai, xi) -> _brm_svi_average!(ai, xi, k), a, x); a)

# The subjects' Cholesky factors of an averaged precision.
function _brm_svi_factor!(q)
    for i in eachindex(q.Λ)
        q.U[i] .= cholesky(Symmetric(q.Λ[i])).L
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
            U[block, j] .= _brm_svi_draw_unit(q, i, randn(rng, length(block)))
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

_brm_svi_model(x::RKBRMI, phase; ad_backend, shared_point, subject_group,
        nested_groups, shared_groups) =
    _brm_run_phase(phase) do node
        _brm_nlme_model(x, node; ad_backend=ad_backend === nothing ?
            _brm_svi_default_ad(x) : ad_backend, observation_ad_backend=nothing,
            shared_point, subject_group, nested_groups, shared_groups)
    end
function _brm_svi_model(m::BRMNLMEModel, phase; ad_backend, shared_point,
        subject_group, nested_groups, shared_groups)
    ad_backend === nothing && shared_point === nothing && subject_group === nothing &&
        isempty(nested_groups) && isempty(shared_groups) || throw(ArgumentError(
        "brm_svi_fit: a prepared BRMNLMEModel already fixes its ad_backend, " *
        "shared_point and subject groupings; pass them to brm_nlme_model instead"))
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
        ad_backend=nothing, shared_point=nothing, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), progress=nothing, kwargs...)
        -> (; draws, coordinates, diagnostics)

Minibatch stochastic variational inference for a model whose subjects are
conditionally independent given the population (the subjects of
[`brm_nlme_view`](@ref)), over the RK backend's packed unconstrained
coordinates. Each iteration evaluates `batch_size` subjects, each at an
antithetic pair of draws through their own observation cells, and scales them
to all subjects, so its cost follows the batch size, not the number of
subjects.

- `x` — a `@brm` model ([`BRMI`](@ref)), its [`RKBRMI`](@ref), or a prepared
  [`BRMNLMEModel`](@ref). For a BRMI or RKBRMI the model is prepared with
  `ad_backend` (default reverse-mode `AutoEnzyme`, which needs Enzyme loaded),
  `shared_point`, `subject_group`, `nested_groups` and `shared_groups`, as
  [`brm_nlme_model`](@ref) takes them; a nested model passes `subject_group`
  and `nested_groups`, a model with a grouping crossed with the subjects
  `shared_groups`. Every observation must belong to one subject's cells: a
  cell of a `@plate for` over the subjects or over their rows, or a row of a
  flat-row response grouped by the subjects.
- `family` — the global factor: `MeanFieldGlobal()` (diagonal) or
  `LowRankGlobal(r)` (diagonal plus rank `r`). Each subject has its own
  full-covariance Gaussian factor over its block, whose length can differ
  between the subjects of a nested model.
- `start` / `inits` — where the globals' means start: the prior medians by
  default, a full RK packed point, or initial values keyed by prior address as
  [`brm_start`](@ref) takes them (the rest at the prior medians).
- `progress` — a Treebars node or backend symbol: one `BRM SVI fit` node with
  the phases Lower to ReactiveKernels, Prepare NLME model, Minibatch target
  (the subjects' cell gradients and the start), Stochastic optimization (its
  message names the window, iteration and ELBO) and Draws from q. An interrupt raises `Treebars.ProgressInterrupted` at the
  next window.
- Optimizer controls: `max_iterations`, `lr_global` (Adam on the globals),
  `unit_step` (the subjects' natural-gradient step; a step that would change a
  subject's precision more than 2.5-fold along some direction, or move its
  mean more than one sd of its factor, is shortened for that subject),
  `polish_lr`, `window`,
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
        ad_backend=nothing, shared_point=nothing, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), global_scale=0.1, unit_scale=1.0,
        ndraws=1000, polish_lr=0.05, window=nothing, patience=10, min_windows=40,
        polish_windows=20, progress=nothing)
    _brm_with_progress(progress; description="BRM SVI fit") do node
        _brm_with_phases(node, _BRM_SVI_PHASES) do phases
            m = _brm_svi_model(_brm_svi_lowered(x, phases.lower), phases.prepare;
                ad_backend, shared_point, subject_group, nested_groups, shared_groups)
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
