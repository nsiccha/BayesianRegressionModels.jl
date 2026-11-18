# Linear coupling of every unit's SVI factor to the globals (LinearCoupling, the
# default), from snag brm-svi-fit-glob-139fc492. Public synthetic toys only:
# - the reporter's generic toy, a population step mean whose per-unit steps are
#   `@plate` cell latents written non-centred (`z ~ Normal(zero_loc[i], 1)`,
#   `steps = step_mean .+ step_spread .* z`), against independently sampled NUTS;
# - a Gaussian random intercept with known scales, where the coupled family holds
#   the exact posterior, against its closed-form population sd;
# - the coupling step against exact quadratic and nonlinear conditional optima.
# RUN: julia --project=test test/nlme_svi_linear_coupling.jl [filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using Random, Statistics, LinearAlgebra, WarmupHMC, TreeArrays, MCMCDiagnosticTools

function coupling_summary(x)
    tree = TreeData(x, :draw)
    (; mean=parent(mean(tree; dims=:draw)), sd=parent(std(tree; dims=:draw)))
end

@stestset "linear coupling: the step's fixed point on a quadratic target" begin
    # One unit of d coordinates, k coupled globals, an exact Gaussian conditional:
    # ∇ log p at z = M + V ζ is -Λ (V - V*) ζ when M and Λ match the target. Each
    # unclipped step moves V a fraction β/(2k) of the way in expectation.
    rng = Xoshiro(7)
    d, k = 3, 4
    A = randn(rng, d, d)
    Λ = A * A' + d * I
    Vstar = randn(rng, d, k)
    U = cholesky(Symmetric(Λ)).L
    q = (; M=[zeros(d)], Λ=[Matrix(Λ)], U=[Matrix(U)])
    coupling = (; step=0.5, columns=[collect(1:k)], rows=[collect(1:d)], V=[zeros(d, k)])
    for _ in 1:4000
        ζ = randn(rng, k)
        ∇̄ = -Λ * ((coupling.V[1] - Vstar) * ζ)
        BRM._brm_svi_coupling_step!(coupling, q, 1, ∇̄, ζ, coupling.step)
    end
    @test norm(coupling.V[1] - Vstar) / norm(Vstar) < 1e-6
    # The trust region: a gradient far from the fixed point moves the mean at a
    # typical ζ by at most one sd of the factor.
    V0 = zeros(d, k)
    c = (; step=1.0, columns=[collect(1:k)], rows=[collect(1:d)], V=[copy(V0)])
    ζ = randn(rng, k)
    BRM._brm_svi_coupling_step!(c, q, 1, 1e6 .* ones(d), ζ, 1.0)
    @test sqrt(tr(c.V[1]' * Λ * c.V[1])) <= 1 + 1e-9
end

# A nonlinear conditional whose best linear variational mean is known exactly:
# g ~ Normal(0, 1), p(z|g) = Normal(a*g^3, 1), q(z|g) = Normal(V*g, 1).
# Its ELBO is constant - E[(a*g^3 - V*g)^2]/2, so V* = 3a. Averaging the
# conditional +/- unit draws leaves the exact mean gradient a*g^3 - V*g.
# Integrate the actual optimizer step to check its fixed point, with a small
# step so the trust bound is inactive. Normalizing by this same draw's norm
# biases the optimum even though it preserves every quadratic target's optimum.
@stestset "linear coupling: nonlinear conditional preserves the ELBO optimum" begin
    q = (; M=[zeros(1)], Λ=[ones(1, 1)], U=[ones(1, 1)])
    c = (; step=0.5, columns=[[1]], rows=[[1]], V=[zeros(1, 1)])
    a, β, h = 0.01, 0.001, 0.004
    function expected_step(slope)
        total = 0.0
        for g in -8:h:8
            c.V[1] .= slope
            BRM._brm_svi_coupling_step!(c, q, 1, [a*g^3 - slope*g], [g], β)
            total += (only(c.V[1]) - slope) / β * pdf(Normal(), g) * h
        end
        total
    end
    @test abs(expected_step(3a)) < 1e-9
    @test expected_step(0.0) > 0
    @test expected_step(6a) < 0
end

# Snag default-linearco-49897cce: a fresh coupling (with `noise` and `updates`) divides
# each row's step by the mean square of its updates so far in conditional sds where
# that exceeds one, reading the scale before the visit's own update enters it, and
# bounds a step's move of the conditional mean by a quarter of the factor's sd.
scaled_coupling(d, k; step=0.5) = (; step, columns=[collect(1:k)], rows=[collect(1:d)],
    V=[zeros(d, k)], noise=[zeros(d)], updates=[0])

@stestset "linear coupling: a quiet row's scaled step equals the unscaled one" begin
    # Updates within one conditional sd, steps within the quarter-sd bound: the same
    # arithmetic, bit for bit, as a coupling without the noise state (an older
    # optimizer checkpoint's).
    rng = Xoshiro(3)
    d, k = 3, 5
    A = randn(rng, d, d)
    Λ = A * A' + d * I
    q = (; M=[zeros(d)], Λ=[Matrix(Λ)], U=[Matrix(cholesky(Symmetric(Λ)).L)])
    scaled = scaled_coupling(d, k)
    plain = (; step=0.5, columns=[collect(1:k)], rows=[collect(1:d)], V=[zeros(d, k)])
    σ = sqrt.(diag(inv(Λ)))
    for _ in 1:200
        ζ = randn(rng, k)
        ∇̄ = Λ * (0.3 .* σ .* randn(rng, d))      # a = Λ⁻¹ ∇̄ well within one sd
        BRM._brm_svi_coupling_step!(scaled, q, 1, ∇̄, ζ, 0.5)
        BRM._brm_svi_coupling_step!(plain, q, 1, ∇̄, ζ, 0.5)
    end
    @test all(<=(1), scaled.noise[1])
    @test scaled.V[1] == plain.V[1]
    @test scaled.updates[1] == 200
    # A spike (the unscaled coupling would move the mean by one sd) moves a fresh
    # coupling's conditional mean at a typical ζ by at most a quarter sd.
    spike = scaled_coupling(d, k)
    ζ = randn(rng, k)
    BRM._brm_svi_coupling_step!(spike, q, 1, 1e6 .* ones(d), ζ, 1.0)
    @test sqrt(tr(spike.V[1]' * Λ * spike.V[1])) <= 0.25 + 1e-9
    @test sqrt(tr(spike.V[1]' * Λ * spike.V[1])) > 0.2
end

@stestset "linear coupling: a noisy row's coupling stays near its fixed point" begin
    # Two rows whose ELBO-optimal coupling is zero; the first row's updates have rms
    # 5 conditional sds (pure noise, as on a one-member funnel), the second's 0.3.
    # Unscaled, the noisy row wanders by about sqrt(β/4) * 5 of its sd; scaled, by
    # about sqrt(β/4) (0.35), while the quiet row is unchanged.
    rng = Xoshiro(5)
    d, k = 2, 20
    Λ = [4.0 0.5; 0.5 1.0]
    q = (; M=[zeros(d)], Λ=[Λ], U=[Matrix(cholesky(Symmetric(Λ)).L)])
    σ = sqrt.(diag(inv(Λ)))
    scaled = scaled_coupling(d, k)
    plain = (; step=0.5, columns=[collect(1:k)], rows=[collect(1:d)], V=[zeros(d, k)])
    ratio(c, r) = norm(c.V[1][r, :]) / σ[r]
    peak_scaled, peak_plain = zeros(d), zeros(d)
    for it in 1:20_000
        ζ = randn(rng, k)
        for c in (scaled, plain)
            # the pull of the coupled draw back to V* = 0, plus the noise
            a = -(c.V[1] * ζ) .+ [5.0, 0.3] .* σ .* randn(rng, d)
            BRM._brm_svi_coupling_step!(c, q, 1, Λ * a, ζ, 0.5)
        end
        if it > 2_000
            peak_scaled .= max.(peak_scaled, [ratio(scaled, r) for r in 1:d])
            peak_plain .= max.(peak_plain, [ratio(plain, r) for r in 1:d])
        end
    end
    @test peak_plain[1] > 2.0                     # unscaled: well above one sd
    @test peak_scaled[1] < 1.0                    # scaled: below one sd throughout
    @test sqrt(scaled.noise[1][1]) > 3            # the row's noise is still measured
    @test peak_scaled[2] < 0.5 && peak_plain[2] < 0.5
end

@stestset "linear coupling: the noise scale keeps the ELBO optimum" begin
    # The cubic conditional of the testset above, with a row scale of 1/2 (running
    # mean square 2) held fixed: the expected step is halved, its zero stays at 3a.
    q = (; M=[zeros(1)], Λ=[ones(1, 1)], U=[ones(1, 1)])
    c = (; step=0.5, columns=[[1]], rows=[[1]], V=[zeros(1, 1)], noise=[[2.0]], updates=[100])
    a, β, h = 0.01, 0.001, 0.004
    function expected_step(slope)
        total = 0.0
        for g in -8:h:8
            c.V[1] .= slope
            c.noise[1] .= 2.0
            BRM._brm_svi_coupling_step!(c, q, 1, [a*g^3 - slope*g], [g], β)
            total += (only(c.V[1]) - slope) / β * pdf(Normal(), g) * h
        end
        total
    end
    @test abs(expected_step(3a)) < 1e-9
    @test expected_step(0.0) > 0
    @test expected_step(6a) < 0
    # The scale halves the step: compare with the unscaled coupling at slope 0.
    c0 = (; step=0.5, columns=[[1]], rows=[[1]], V=[zeros(1, 1)])
    total0 = 0.0
    for g in -8:h:8
        c0.V[1] .= 0.0
        BRM._brm_svi_coupling_step!(c0, q, 1, [a*g^3], [g], β)
        total0 += only(c0.V[1]) / β * pdf(Normal(), g) * h
    end
    @test expected_step(0.0) ≈ total0 / 2 rtol=1e-12
end

# A Gaussian random intercept with known group sd and noise sd: the posterior is
# Gaussian, and a unit factor conditional on the intercept with a linear mean holds
# it exactly, so the coupled fit's intercept sd is the closed-form marginal one,
# while independent factors give its conditional sd given the groups.
@stestset "linear coupling: exact population width with known scales" begin
    G, m, τ, σ, s0 = 32, 50, 1.0, 1.0, 5.0
    rng = Xoshiro(11)
    g = repeat(1:G, inner=m)
    data = (; g, y=0.5 .+ τ .* randn(rng, G)[g] .+ σ .* randn(rng, G * m),
        known_sd=τ, known_sigma=σ, prior_sd=s0)
    model = @brm data begin
        mu ~ 1 + (1 | g)
        sd(mu, g) = known_sd
        effect(mu, Intercept) ~ Normal(0, prior_sd)
        y ~ Normal(mu, known_sigma)
    end
    rk = RKBRMI(model)
    exact = sqrt(1 / (1 / s0^2 + G / (τ^2 + σ^2 / m)))
    conditional = sqrt(1 / (1 / s0^2 + G * m / σ^2))
    coupled = brm_svi_fit(rk; subject_group=:g, batch_size=8, rng=Xoshiro(2), ndraws=4000,
        centering=nothing)
    plain = brm_svi_fit(rk; subject_group=:g, batch_size=8, rng=Xoshiro(2), ndraws=4000,
        centering=nothing, coupling=nothing)
    idx = only(findall(==(:mu_Intercept), coupled.coordinates))
    cs, ps = coupling_summary(coupled.draws[idx, :]), coupling_summary(plain.draws[idx, :])
    println("LINEAR_COUPLING known_scales exact_sd=$exact conditional_sd=$conditional " *
        "coupled_sd=$(cs.sd) plain_sd=$(ps.sd) coupled_iterations=" *
        "$(coupled.diagnostics.iterations) plain_iterations=$(plain.diagnostics.iterations)")
    @test coupled.diagnostics.coupling !== nothing
    @test plain.diagnostics.coupling === nothing
    @test 0.85 < cs.sd / exact < 1.15
    @test ps.sd / exact < 0.3
    report = coupled.diagnostics.coupling
    @test length(report.units) == G
    @test all(u -> all(u.marginal_sd .>= u.conditional_sd), report.units)
end

# The reporter's generic toy (snag brm-svi-fit-glob-139fc492): one step per unit,
# a `@plate` cell latent the data pin.
function coupling_cell_data(G, m; seed=1)
    rng = Xoshiro(seed)
    lev = 3 .+ randn(rng, G)
    stp = 0.5 .+ 0.3 .* randn(rng, G)
    row_unit = repeat(1:G, inner=m)
    after = Float64.(repeat(1:m, G) .> m ÷ 2)
    y = lev[row_unit] .+ stp[row_unit] .* after .+ 0.5 .* randn(rng, G * m)
    (; unit=collect(1:G), row_unit, after, y, zero_loc=[zeros(1) for _ in 1:G])
end

noncentred_cell_model(data) = @brm data begin
    sigma ~ Exponential(1)
    level ~ 1 + (1 | unit)
    effect(level, Intercept) ~ Normal(0, 5)
    step_mean ~ Normal(0, 5)
    step_spread ~ Exponential(1)
    rows = groups(row_unit => unit)
    @plate for i in eachindex(level)
        z ~ Normal(zero_loc[i], 1)
        steps = step_mean .+ step_spread .* z
        y[rows[i]] ~ Normal(level[i] .+ sum(steps) .* after[rows[i]], sigma)
    end
end

# The same model with the step's location where the cell reads it (the rewrite
# `brm_nlme_view`'s refusal of a computed location names).
deviation_cell_model(data) = @brm data begin
    sigma ~ Exponential(1)
    level ~ 1 + (1 | unit)
    effect(level, Intercept) ~ Normal(0, 5)
    step_mean ~ Normal(0, 5)
    step_spread ~ Exponential(1)
    rows = groups(row_unit => unit)
    @plate for i in eachindex(level)
        d ~ Normal(zero_loc[i], step_spread)
        steps = step_mean .+ d
        y[rows[i]] ~ Normal(level[i] .+ sum(steps) .* after[rows[i]], sigma)
    end
end

function check_cell_population(label, model; plain=false)
    data = coupling_cell_data(32, 50)
    rk = RKBRMI(model(data))
    fit = brm_svi_fit(rk; subject_group=:unit, batch_size=8, rng=Xoshiro(2), ndraws=4000)
    @test fit.diagnostics.coupling !== nothing
    # Cell latents are listed as not centred, with a reason.
    @test any(r -> r.kind === :cell_latent, fit.diagnostics.centering.uncentered)
    idx = only(findall(==(:step_mean), fit.coordinates))
    problems = [rk_logdensity_problem(rk; ad_backend=NLME_AD) for _ in 1:4]
    init = brm_start(rk; chains=4, rng=Xoshiro(3))
    reference = completion_warmup_mcmc([Xoshiro(10 + c) for c in 1:4], problems;
        min_completed=4, n_draws=1000, blas_threads=1, init)
    @test reference.completion.n_completed == 4
    chains = hcat([Matrix{Float64}(r.posterior_position)[idx, :]
        for r in reference.results]...)
    diagnostic = ess_rhat(reshape(chains, size(chains)..., 1))
    rhat, ess = only(diagnostic.rhat), only(diagnostic.ess)
    divergent = sum(r.n_divergent_samples for r in reference.results)
    @test rhat < 1.05
    @test ess > 100
    nuts, svi = coupling_summary(vec(chains)), coupling_summary(fit.draws[idx, :])
    ratio, error = svi.sd / nuts.sd, abs(svi.mean - nuts.mean) / nuts.sd
    println("LINEAR_COUPLING case=$label mean_error=$error sd_ratio=$ratio " *
        "svi_sd=$(svi.sd) nuts_sd=$(nuts.sd) iterations=$(fit.diagnostics.iterations) " *
        "converged=$(fit.diagnostics.converged) nuts_rhat=$rhat nuts_ess=$ess " *
        "nuts_divergent=$divergent")
    @test error < 0.6
    @test 0.6 < ratio < 1.4
    if plain
        independent = brm_svi_fit(rk; subject_group=:unit, batch_size=8, rng=Xoshiro(2),
            ndraws=4000, coupling=nothing)
        ps = coupling_summary(independent.draws[idx, :])
        println("LINEAR_COUPLING case=$label independent_sd_ratio=$(ps.sd / nuts.sd) " *
            "independent_sd=$(ps.sd) converged=$(independent.diagnostics.converged)")
        @test independent.diagnostics.coupling === nothing
        @test ps.sd / nuts.sd < 0.5
        @test svi.sd > 1.5ps.sd
    end
end

@stestset "linear coupling: non-centred @plate cell latent steps against NUTS" begin
    check_cell_population("noncentred_cell", noncentred_cell_model; plain=true)
end
@stestset "linear coupling: cell latent deviations against NUTS" begin
    check_cell_population("deviation_cell", deviation_cell_model)
end

@stestset "linear coupling: explicit policies and refusals" begin
    @test_throws ArgumentError LinearCoupling(; step=0)
    @test_throws ArgumentError LinearCoupling(; step=1.5)
    @test BRM._brm_svi_resolve_coupling(missing) isa LinearCoupling
    @test BRM._brm_svi_resolve_coupling(nothing) === nothing
    @test_throws ArgumentError BRM._brm_svi_resolve_coupling(:linear)
end
