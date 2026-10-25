# src/nlme_svi.jl — minibatch stochastic variational inference over an
# RK-lowered population model's own packed coordinates, with an
# importance-sampling correction of the fitted approximation.
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
# The variational family is Gaussian over u, with a global factor and one
# factor per subject anchored at a reference global point g_ref:
#
#     g   = μ + L ε,                               ε ~ N(0, I)
#     z_i = m_i + A_i (g - g_ref) + S_i ε_i,       ε_i ~ N(0, I)
#
# L and S_i are lower triangular with positive diagonals (`MeanFieldGlobal`
# keeps L diagonal). A_i moves a subject's draws with the globals (without it,
# `coupling=false`, the factors are independent); because the subjects follow
# g through A_i, moving μ moves them too, which keeps the location of the
# globals and the subjects' means from fighting each other. The map
# (ε, ε_1, …) → u is block lower triangular, so log q(u) needs only diag(L) and
# diag(S_i). Stochastic steps move a subject's parameters only when the subject
# is in the batch; Laplace refreshes (one per window of iterations, a handful of
# full-model passes) set every subject's factor to its conditional mode,
# curvature and linear response at the current global mean.
#
# The Gaussian optimum of the ELBO is too narrow for hierarchical scales, so the
# fit is followed by importance sampling of the posterior: a Student-t proposal
# for g, adapted to the weighted draws over a few pilot rounds, and for each
# drawn g a Laplace proposal for every subject's draws. PSIS smooths the weights;
# where k̂ < 0.7 the draws are resampled in proportion to them.

"""
    BRMSVIFamily

The global variational factor of [`brm_svi`](@ref): `FullRankGlobal()` (a dense
lower-triangular Cholesky factor) or `MeanFieldGlobal()` (a diagonal one).
"""
abstract type BRMSVIFamily end
struct FullRankGlobal <: BRMSVIFamily end
struct MeanFieldGlobal <: BRMSVIFamily end

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

# ── Laplace refresh of the subject factors ───────────────────────────────────
#
# Every subject's conditional p(z_i | g, y_i) at once, from full-model passes:
# the sampler gradient, and one Hessian-vector product per block position whose
# direction moves that position in every subject (conditional independence
# makes the result each subject's Hessian column, as in
# `_brm_nlme_hessian_blocks`). Newton steps move each z_i to its mode at the
# global mean; the factor's covariance is the inverse negative Hessian, and its
# coupling the mode's linear response to the globals, A_i = -H_zz⁻¹ H_zg, from
# one more product per global coordinate.

function _brm_svi_hessian_blocks(t, u)
    m = t.m
    H = [zeros(t.d, t.d) for _ in t.units]
    for k in 1:t.d
        direction = zeros(length(u))
        foreach(block -> direction[block[k]] = 1.0, t.units)
        hv = _rk_eta_hvp(m.eta_hvp, direction, u)
        for (i, block) in enumerate(t.units)
            H[i][:, k] .= view(hv, block)
        end
    end
    map(h -> Symmetric((h + h') / 2), H)
end

# The negative Hessian with its eigenvalues floored, so that a Newton step
# always ascends and the factor is a proper Gaussian.
function _brm_svi_precision(H; floor=1e-3)
    e = eigen(Symmetric(-Matrix(H)))
    Symmetric(e.vectors * Diagonal(max.(e.values, floor)) * e.vectors')
end

# Newton steps moving every subject's draws in the packed point `u` (globals
# held) to their conditional modes; returns the precision blocks there.
function _brm_svi_unit_modes!(t, u; iterations=20, max_step=3.0, gtol=1e-6)
    gradient = zeros(length(u))
    for _ in 1:iterations
        _rk_value_gradient_and_pointwise!(t.m.sampler, gradient, u)
        maximum(i -> maximum(abs, view(gradient, t.units[i])), eachindex(t.units)) < gtol && break
        H = _brm_svi_hessian_blocks(t, u)
        for (i, block) in enumerate(t.units)
            step = _brm_svi_precision(H[i]) \ gradient[block]
            n = norm(step)
            n > max_step && (step .*= max_step / n)
            u[block] .+= step
        end
    end
    map(_brm_svi_precision, _brm_svi_hessian_blocks(t, u))
end

function _brm_svi_laplace_refresh!(t, q; kwargs...)
    m = t.m
    u = copy(m.unit)
    u[t.globals] .= q.μ
    for (i, block) in enumerate(t.units)
        u[block] .= view(q.M, :, i) .+ view(q.A, :, :, i) * (q.μ .- q.ref)
    end
    precisions = _brm_svi_unit_modes!(t, u; kwargs...)
    cross = [zeros(t.d, length(t.globals)) for _ in t.units]
    if size(q.A, 2) > 0
        for (k, j) in enumerate(t.globals)
            direction = zeros(length(u))
            direction[j] = 1.0
            hv = _rk_eta_hvp(m.eta_hvp, direction, u)
            for (i, block) in enumerate(t.units)
                cross[i][:, k] .= view(hv, block)
            end
        end
    end
    for (i, block) in enumerate(t.units)
        Λ = precisions[i]
        q.M[:, i] .= view(u, block)
        S = cholesky(inv(Λ)).L
        q.R[:, i] .= log.(diag(S))
        q.So[:, :, i] .= Matrix(S) .- Diagonal(diag(S))
        size(q.A, 2) > 0 && (q.A[:, :, i] .= Λ \ cross[i])
    end
    q.ref .= q.μ
    q
end

# ── Variational parameters and Adam ──────────────────────────────────────────

function _brm_svi_params(t, μ, family::BRMSVIFamily, coupling; global_scale,
        unit_scale)
    D, d, N = length(t.globals), t.d, length(t.units)
    (; family, coupling,
       μ=Vector{Float64}(μ), ref=Vector{Float64}(μ), ρ=fill(log(global_scale), D),
       Lo=zeros(D, D),
       M=zeros(d, N), R=fill(log(unit_scale), d, N), So=zeros(d, d, N),
       A=zeros(d, coupling ? D : 0, N))
end

_brm_svi_L(q) = LowerTriangular(q.Lo) - Diagonal(diag(q.Lo)) + Diagonal(exp.(q.ρ))
_brm_svi_S(q, i) = LowerTriangular(view(q.So, :, :, i)) -
    Diagonal(diag(view(q.So, :, :, i))) + Diagonal(exp.(view(q.R, :, i)))

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

# The global parameters as one vector, and the per-subject parameters as one
# column each, so that Adam keeps one state for the globals and one per subject.
_brm_svi_global_vector(q) = vcat(q.μ, q.ρ, vec(q.Lo))
function _brm_svi_set_global!(q, x)
    D = length(q.μ)
    q.μ .= view(x, 1:D)
    q.ρ .= view(x, D+1:2D)
    vec(q.Lo) .= view(x, 2D+1:length(x))
    q
end
_brm_svi_unit_vector(q, i) = vcat(view(q.M, :, i), view(q.R, :, i),
    vec(view(q.So, :, :, i)), vec(view(q.A, :, :, i)))
function _brm_svi_set_unit!(q, i, x)
    d = size(q.M, 1)
    q.M[:, i] .= view(x, 1:d)
    q.R[:, i] .= view(x, d+1:2d)
    vec(view(q.So, :, :, i)) .= view(x, 2d+1:2d+d^2)
    vec(view(q.A, :, :, i)) .= view(x, 2d+d^2+1:length(x))
    q
end

_brm_svi_strict_lower!(X) = (for j in axes(X, 2), k in 1:min(j, size(X, 1)); X[k, j] = 0.0; end; X)

# Scale-preconditioned ("whitened") steps. A location moves in units of its own
# factor (μ = μ₀ + L μ̃, m_i = m₀ + S_i m̃), and the off-diagonal and coupling
# entries of a row in units of that row's scale. Adam's step size then means a
# fraction of the current posterior scale, which stays meaningful when a
# global's posterior sd shrinks like 1/√N.
function _brm_svi_whitened_step!(opt, x, grad, lr, location_factor, row_scale, D)
    grad = copy(grad)
    grad[1:D] .= location_factor' * view(grad, 1:D)
    rest = D+1:length(x)
    grad[rest] .*= row_scale
    step = zeros(length(x))
    _brm_svi_step!(opt, step, grad, lr)
    x[1:D] .+= location_factor * view(step, 1:D)
    x[rest] .+= row_scale .* view(step, rest)
    x
end

# Row scales of the global vector [μ; ρ; vec(Lo)] past μ: 1 for ρ, the row's
# diagonal for each Lo entry.
_brm_svi_global_row_scale(q) = vcat(ones(length(q.ρ)),
    vec(repeat(exp.(q.ρ), 1, length(q.ρ))))
# … and of the unit vector [m; R; vec(So); vec(A)] past m.
function _brm_svi_unit_row_scale(q, i)
    s = exp.(view(q.R, :, i))
    vcat(ones(length(s)), vec(repeat(s, 1, length(s))),
        vec(repeat(s, 1, size(q.A, 2))))
end

# The gradient of a global parameter vector from the global-coordinate
# gradient w (already including the batch's coupling term) and ε.
function _brm_svi_global_gradient(q, w, ε, gμ)
    D = length(q.μ)
    gρ = w .* ε .* exp.(q.ρ) .+ 1.0
    gLo = q.family isa FullRankGlobal ? _brm_svi_strict_lower!(w * ε') : zeros(D, D)
    vcat(gμ, gρ, vec(gLo))
end

# ── One minibatch iteration ──────────────────────────────────────────────────

function _brm_svi_iteration!(rng, t, q, global_opt, unit_opts, batch, lr_global,
        lr_unit; local_steps=1, whiten=false)
    D, d, N = length(t.globals), t.d, length(t.units)
    s = N / length(batch)
    L = _brm_svi_L(q)
    gg, gz = zeros(D), zeros(d)
    # Extra local steps refine the batch's factors against fresh draws of the
    # current global factor, before the step every parameter takes together.
    for _ in 2:local_steps, i in batch
        g = q.μ .+ L * randn(rng, D)
        v = g .- q.ref
        εi = randn(rng, d)
        z = view(q.M, :, i) .+ view(q.A, :, :, i) * v .+ _brm_svi_S(q, i) * εi
        _brm_svi_unit_term!(gg, gz, t, i, g, z)
        _brm_svi_unit_update!(q, i, unit_opts[i],
            _brm_svi_unit_gradient(q, i, gz, v, εi), lr_unit, whiten)
    end
    ε = randn(rng, D)
    g = q.μ .+ L * ε
    v = g .- q.ref
    P, gP = _brm_svi_global_term(t, g)
    total = copy(gP)
    coupling = zeros(D)
    batch_value = 0.0
    unit_gradients = Vector{Vector{Float64}}(undef, length(batch))
    for (k, i) in enumerate(batch)
        εi = randn(rng, d)
        z = view(q.M, :, i) .+ view(q.A, :, :, i) * v .+ _brm_svi_S(q, i) * εi
        batch_value += _brm_svi_unit_term!(gg, gz, t, i, g, z)
        total .+= s .* gg
        size(q.A, 2) > 0 && (coupling .+= s .* (view(q.A, :, :, i)' * gz))
        unit_gradients[k] = _brm_svi_unit_gradient(q, i, gz, v, εi)
    end
    x = _brm_svi_global_vector(q)
    # Subject draws follow g through A_i, so the coupling term reaches μ as well
    # as L: μ's gradient is the one along which the subjects' means follow.
    grad = _brm_svi_global_gradient(q, total .+ coupling, ε, total .+ coupling)
    whiten ? _brm_svi_whitened_step!(global_opt, x, grad, lr_global, Matrix(L),
            _brm_svi_global_row_scale(q), D) :
        _brm_svi_step!(global_opt, x, grad, lr_global)
    _brm_svi_set_global!(q, x)
    for (k, i) in enumerate(batch)
        _brm_svi_unit_update!(q, i, unit_opts[i], unit_gradients[k], lr_unit, whiten)
    end
    P + s * batch_value
end

function _brm_svi_unit_update!(q, i, opt, grad, lr, whiten)
    xi = _brm_svi_unit_vector(q, i)
    whiten ? _brm_svi_whitened_step!(opt, xi, grad, lr, Matrix(_brm_svi_S(q, i)),
            _brm_svi_unit_row_scale(q, i), size(q.M, 1)) :
        _brm_svi_step!(opt, xi, grad, lr)
    _brm_svi_set_unit!(q, i, xi)
end

function _brm_svi_unit_gradient(q, i, gz, v, εi)
    d = length(gz)
    gR = gz .* εi .* exp.(view(q.R, :, i)) .+ 1.0
    gSo = _brm_svi_strict_lower!(gz * εi')
    gA = size(q.A, 2) > 0 ? gz * v' : zeros(d, 0)
    vcat(gz, gR, vec(gSo), vec(gA))
end

# The entropy of q, up to the constant (D + N d)(1 + log 2π)/2.
_brm_svi_entropy(q) = sum(q.ρ) + sum(q.R)

# ── Draws from q and the PSIS check ──────────────────────────────────────────

# The global proposal is a multivariate Student-t with `df` degrees of freedom
# (`Inf`: Gaussian) around `center` with scale factor `C`; its heavier tails
# keep the importance weights of the variational fit's too narrow global
# marginal bounded.
function _brm_svi_global_proposal(rng, center, C, df)
    D = length(center)
    ε = randn(rng, D)
    r = isinf(df) ? 1.0 : sqrt(df / rand(rng, Chisq(df)))
    g = center .+ C * (r .* ε)
    δ2 = sum(abs2, ε) * r^2
    logq = isinf(df) ? -δ2 / 2 - D * log(2π) / 2 :
        SpecialFunctions.loggamma((df + D) / 2) - SpecialFunctions.loggamma(df / 2) -
        D * log(df * π) / 2 - (df + D) / 2 * log1p(δ2 / df)
    g, logq - logdet(C)
end

# Importance weights on the global draws alone. Given g the posterior factors
# over subjects, so each subject's marginal p(y_i | g) is estimated from K draws
# of its conditional proposal, and one of them is kept in proportion to its
# weight. The weighted pairs (g, kept z) target the posterior exactly for every
# K (the extended-space argument of pseudo-marginal importance sampling); K = 1
# is the plain joint importance weight. Returns the kept packed draws (one per
# column), the global log weights, and each draw's summed per-subject noise of
# the marginal estimates.
function _brm_svi_unit_importance(rng, t, q, n, K; center=q.μ,
        C=_brm_svi_L(q), df=Inf, conditional=:laplace)
    D, d, N = length(t.globals), t.d, length(t.units)
    m = t.m
    U = repeat(m.unit, 1, n)
    logw = zeros(n)
    noise = zeros(n)
    candidates = [zeros(d, K) for _ in 1:N]
    unit_logw = zeros(N, K)
    for j in 1:n
        g, logq = _brm_svi_global_proposal(rng, center, C, df)
        v = g .- q.ref
        U[t.globals, j] .= g
        P, _ = _brm_svi_global_term(t, g)
        logw[j] = P - logq
        # Each subject's conditional proposal at this g: the fitted factor, or
        # (`:laplace`) the Gaussian at the conditional mode with the inverse
        # negative Hessian there, from Newton steps started at the factor's
        # mean. The Laplace conditional is exact when p(z_i | g, y_i) is
        # Gaussian, so its weights then carry no per-subject noise.
        factors = map(eachindex(t.units)) do i
            (view(q.M, :, i) .+ view(q.A, :, :, i) * v, _brm_svi_S(q, i))
        end
        if conditional === :laplace
            u = U[:, j]
            for (i, block) in enumerate(t.units)
                u[block] .= first(factors[i])
            end
            precisions = _brm_svi_unit_modes!(t, u)
            factors = map(eachindex(t.units)) do i
                (u[t.units[i]], cholesky(inv(precisions[i])).L)
            end
        end
        for k in 1:K
            u = U[:, j]
            for (i, block) in enumerate(t.units)
                εi = randn(rng, d)
                center_i, S = factors[i]
                z = center_i .+ S * εi
                candidates[i][:, k] .= z
                u[block] .= z
                # log N(z; 0, I) − log q_i(z | g); the likelihood is added below.
                unit_logw[i, k] = -sum(abs2, z) / 2 + sum(abs2, εi) / 2 +
                    sum(log, diag(S))
            end
            unit_logw[:, k] .+= _brm_nlme_sum_by_subject(m,
                _rk_pointwise_values(m.pointwise, u))
        end
        noise_j = 0.0
        for (i, block) in enumerate(t.units)
            row = view(unit_logw, i, :)
            logw[j] += LogExpFunctions.logsumexp(row) - log(K)
            p = exp.(row .- maximum(row))
            # The relative variance of this subject's marginal estimate over
            # its K draws (delta method: the variance of its log).
            K > 1 && (noise_j += Statistics.var(p) / (Statistics.mean(p)^2 * K))
            k = K == 1 ? 1 : _brm_svi_resample(rng, p ./ sum(p), 1)[1]
            U[block, j] .= view(candidates[i], :, k)
        end
        noise[j] = noise_j
    end
    U, logw, noise
end

"""
    brm_svi(m::BRMNLMEModel; batch_size, rng, ...) -> NamedTuple

Minibatch stochastic variational inference for a population model with
conditionally independent subjects, over the RK backend's packed unconstrained
coordinates (internal prototype; its public spelling is not settled).
"""
function brm_svi(m::BRMNLMEModel; rng=Random.default_rng(), batch_size=32,
        max_iterations=100_000, family::BRMSVIFamily=FullRankGlobal(), coupling=true,
        lr_global=0.01, lr_unit=0.05, local_steps=1, start=nothing,
        global_scale=0.1, unit_scale=1.0, ndraws=1000, unit_draws=:auto,
        proposal_df=4.0, proposal_inflation=1.2, adapt_rounds=6,
        polish_lr=0.05, window=nothing, patience=3, min_windows=5,
        polish_windows=20, laplace_refresh=true, refresh_windows=1,
        conditional=:laplace, khat_threshold=0.7)
    t = _brm_svi_target(m)
    N = length(t.units)
    μ = start === nothing ? first(_brm_nlme_prior_median_point(m))[t.globals] :
        Vector{Float64}(start)[t.globals]
    q = _brm_svi_params(t, μ, family, coupling; global_scale, unit_scale)
    fit_seconds = @elapsed fit = _brm_svi_fit!(rng, t, q; batch_size, max_iterations,
        lr_global, lr_unit, polish_lr, local_steps, window, patience, min_windows,
        polish_windows, laplace_refresh, refresh_windows)
    check_seconds = @elapsed check = _brm_svi_check(rng, t, fit.q; ndraws,
        unit_draws, proposal_df, proposal_inflation, khat_threshold, adapt_rounds,
        conditional)
    (; draws=check.draws, diagnostics=(; check.diagnostics..., fit.elbo,
        fit.elbo_windows, fit.converged, fit.iterations, fit.q, globals=t.globals,
        fit_seconds, check_seconds))
end

# Stochastic optimization in two phases. Exploration runs Adam at constant
# step sizes until the ELBO stops improving detectably: the mean over a window
# (one pass over the subjects, at least 100 iterations) has risen by less than
# twice its standard error in each of the last `patience` windows, after at
# least `min_windows`. Polishing then restarts Adam on scale-preconditioned
# steps (`polish_lr`, a fraction of the current factor scale), decays them as
# 1/√k over `polish_windows` windows, and returns the average of q over those
# iterations (Polyak–Ruppert), which damps the noise a stochastic step leaves.
function _brm_svi_fit!(rng, t, q; batch_size, max_iterations, lr_global, lr_unit,
        polish_lr, local_steps, window, patience, min_windows, polish_windows,
        laplace_refresh, refresh_windows)
    N = length(t.units)
    adam(lr) = (_BRMSVIAdam(length(_brm_svi_global_vector(q)), lr),
        [_BRMSVIAdam(length(_brm_svi_unit_vector(q, i)), lr) for i in 1:N])
    global_opt, unit_opts = adam(lr_global)
    B = min(batch_size, N)
    W = window === nothing ? max(cld(N, B), 100) : window
    trace = Float64[]
    windows = NamedTuple{(:mean, :var),Tuple{Float64,Float64}}[]
    polish_start = 0
    average = nothing
    laplace_refresh && _brm_svi_laplace_refresh!(t, q)
    for it in 1:max_iterations
        polishing = polish_start > 0
        scale = polishing ? 1 / sqrt(1 + (it - polish_start) / W) : 1.0
        batch = _brm_svi_batch(rng, N, B)
        value = _brm_svi_iteration!(rng, t, q, global_opt, unit_opts, batch,
            scale * (polishing ? polish_lr : lr_global),
            scale * (polishing ? polish_lr : lr_unit); local_steps, whiten=polishing)
        push!(trace, value + _brm_svi_entropy(q))
        if polishing
            k = it - polish_start
            for name in (:μ, :ref, :ρ, :Lo, :M, :R, :So, :A)
                a, x = getfield(average, name), getfield(q, name)
                @. a += (x - a) / k
            end
            k >= polish_windows * W && return (; q=average, elbo=trace,
                elbo_windows=first.(windows), converged=true, iterations=it)
        end
        it % W == 0 || continue
        w = view(trace, it-W+1:it)
        push!(windows, (; mean=Statistics.mean(w), var=Statistics.var(w)))
        !polishing && laplace_refresh && length(windows) % refresh_windows == 0 &&
            _brm_svi_laplace_refresh!(t, q)
        if !polishing && length(windows) >= max(min_windows, patience + 1)
            stalled = all(length(windows)-patience+1:length(windows)) do k
                a, b = windows[k-1], windows[k]
                b.mean - a.mean < 2 * sqrt((a.var + b.var) / W)
            end
            if stalled
                polish_start = it
                average = deepcopy(q)
                global_opt, unit_opts = adam(polish_lr)
            end
        end
    end
    (; q=average === nothing ? q : average, elbo=trace,
       elbo_windows=first.(windows), converged=false, iterations=max_iterations)
end

# The PSIS check of a fitted q, and its importance-resampled draws.
function _brm_svi_check(rng, t, q; ndraws, unit_draws, proposal_df,
        proposal_inflation, khat_threshold, adapt_rounds=0, conditional=:laplace,
        adapt_draws=min(ndraws, max(250, 25length(t.globals))), unit_noise=0.25,
        max_unit_draws=64, adapt_khat=0.5)
    D = length(t.globals)
    # `unit_draws = :auto` picks K per round so that the summed per-subject
    # noise of the marginal estimates, measured in the previous round, falls
    # to `unit_noise` (pilot K = 4, at most `max_unit_draws`).
    K = unit_draws === :auto ? 4 : unit_draws
    next_K(noise) = unit_draws === :auto ?
        clamp(ceil(Int, K * Statistics.median(noise) / unit_noise), 1, max_unit_draws) : K
    sample(n, center, C) = _brm_svi_unit_importance(rng, t, q, n, K;
        center, C, df=proposal_df, conditional)
    record(psis, noise) = (; pareto_k=psis.pareto_k, ess=psis.ess, unit_draws=K,
        unit_noise=Statistics.median(noise))
    # Adaptive importance sampling: each pilot round moves the global proposal
    # to its weighted draws' mean, and its covariance toward theirs (inflated
    # again), blended with the previous covariance in proportion to the
    # round's effective sample size but at least halfway, so that a round
    # dominated by a few draws cannot collapse it. Adaptation stops once a
    # pilot reaches k̂ < `adapt_khat` or after `adapt_rounds` pilots; the
    # final draws come from the proposal with the smallest pilot k̂.
    proposal = (; center=q.μ, C=proposal_inflation * Matrix(_brm_svi_L(q)))
    best = (; proposal, pareto_k=Inf)
    rounds = Any[]
    for round in 1:adapt_rounds
        U, logw, noise = sample(adapt_draws, proposal.center, proposal.C)
        psis = brm_psis_weights(logw)
        push!(rounds, record(psis, noise))
        psis.pareto_k < best.pareto_k && (best = (; proposal, psis.pareto_k))
        K = next_K(noise)
        psis.pareto_k < adapt_khat && break
        G = view(U, t.globals, :)
        center = G * psis.weights
        Σ = (G .- center) * Diagonal(psis.weights) * (G .- center)'
        α = clamp(psis.ess / (10D), 0.5, 1.0)
        previous = proposal.C * proposal.C' / proposal_inflation^2
        Σ = α .* Σ .+ (1 - α) .* previous
        proposal = (; center, C=proposal_inflation * Matrix(cholesky(Symmetric(Σ + 1e-12I)).L))
    end
    final = adapt_rounds > 0 ? best.proposal : proposal
    U, logw, noise = sample(ndraws, final.center, final.C)
    psis = brm_psis_weights(logw)
    push!(rounds, record(psis, noise))
    corrected = psis.pareto_k < khat_threshold
    # Draws × packed coordinates, the shape every BRM draw reader takes.
    proposal_draws = permutedims(U)
    draws = corrected ?
        proposal_draws[_brm_svi_resample(rng, psis.weights, ndraws), :] : proposal_draws
    (; draws, diagnostics=(; corrected, pareto_k=psis.pareto_k, ess=psis.ess,
        proposal=proposal_draws, log_weights=psis.log_weights, raw_log_weights=logw,
        rounds))
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

# Systematic resampling: `n` indices drawn in proportion to `weights`.
function _brm_svi_resample(rng, weights, n)
    c = cumsum(weights)
    u0 = rand(rng) / n
    out = Vector{Int}(undef, n)
    j = 1
    for k in 1:n
        u = u0 + (k - 1) / n
        while j < length(c) && c[j] < u
            j += 1
        end
        out[k] = j
    end
    out
end
