# `brm_svi_fit(...; unit_reoptimize = k)` (todo 122p9l3, design decision
# `093fulx`; repaired in todo 12maq21): each dense batch unit is re-optimized at the
# iteration's draw of the globals, a copy of its factor taking `k` Newton steps to
# its conditional optimum given that draw, and the globals step on the copy's terms,
# so a ridge between a global and every unit (an intercept beside per-unit effects
# the data pin only in `a + u[g]`) no longer holds the global back. Checked on an
# exactly Gaussian toy whose posterior means and sds are known in closed form:
# the stock step is the positive control (its intercept's sd is far too small and
# its iterates have not settled), the re-optimized step recovers both. Also: on a
# unit far from Gaussian a copy's step that would lower its unit's objective stands
# down, the default changes nothing, the extra evaluations are counted, the options
# that cannot honour it refuse it, and checkpoints resume exactly.
#
#     julia --project=test test/nlme_svi_reoptimize.jl [testset filter...]
using Test, BayesianRegressionModels, Random, Statistics, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "nlme_svi_ridge_fixture.jl"))

function ridge_data(; G=200, n=20, P=4, seed=1)
    rng = Xoshiro(seed)
    u = 0.5 .* randn(rng, G)
    X = randn(rng, G * n, P)
    beta = randn(rng, P)
    g = repeat(1:G; inner=n)
    y = 1.0 .+ u[g] .+ X * beta .+ 0.1 .* randn(rng, G * n)
    (; g, y, x1=X[:, 1], x2=X[:, 2], x3=X[:, 3], x4=X[:, 4], known_sd=0.5, noise=0.1,
       prior_sd=10.0)
end

ridge_model(data) = @brm data begin
    mu ~ 1 + x1 + x2 + x3 + x4 + (1 | g)
    effect(mu, :) ~ Normal(0, prior_sd)
    sd(mu, g) = known_sd
    y ~ Normal(mu, noise)
end

# The exact posterior means and sds of the intercept and slopes.
function exact_globals(data)
    G = maximum(data.g)
    Z = hcat(ones(length(data.y)), data.x1, data.x2, data.x3, data.x4,
             [data.g[r] == j ? 1.0 : 0.0 for r in eachindex(data.y), j in 1:G])
    prior = vcat(fill(1 / data.prior_sd^2, 5), fill(1 / data.known_sd^2, G))
    H = Symmetric(Z' * Z ./ data.noise^2 + Diagonal(prior))
    m = H \ (Z' * data.y ./ data.noise^2)
    m[1:5], sqrt.(diag(inv(H))[1:5])
end

global_rows(rk) = (names = String.(BRM._rk_layout_coordinate_names(rk));
    [findfirst(==("pop_mu.beta_pop.$j"), names) for j in 1:5])

@stestset "options: the default changes nothing; unsupported settings refuse" begin
    data = ridge_data(; G=20, n=5)
    rk = RKBRMI(ridge_model(data))
    base = brm_svi_fit(rk; subject_group=:g, rng=Xoshiro(3), max_iterations=50, ndraws=20)
    zero = brm_svi_fit(rk; subject_group=:g, rng=Xoshiro(3), max_iterations=50, ndraws=20,
                       unit_reoptimize=0)
    @test zero.draws == base.draws
    @test zero.diagnostics.unit_gradient_calls == base.diagnostics.unit_gradient_calls
    two = brm_svi_fit(rk; subject_group=:g, rng=Xoshiro(3), max_iterations=50, ndraws=20,
                      unit_reoptimize=2)
    @test all(isfinite, two.draws)
    # Each re-optimization step costs the unit's 2d evaluations again: d = 1, a
    # batch of all 20 units, 2 steps, 50 iterations.
    @test two.diagnostics.unit_gradient_calls ==
          base.diagnostics.unit_gradient_calls + 50 * 20 * (2 * 1) * 2
    @test_throws r"nonnegative integer" brm_svi_fit(rk; subject_group=:g,
        max_iterations=1, ndraws=0, unit_reoptimize=-1)
    @test_throws r"orthogonal" brm_svi_fit(rk; subject_group=:g, max_iterations=1,
        ndraws=0, unit_reoptimize=1, unit_curvature=:pair)
end

@stestset "intercept ridge: re-optimized units let the globals reach the exact posterior" begin
    data = ridge_data()
    rk = RKBRMI(ridge_model(data))
    rows = global_rows(rk)
    m, s = exact_globals(data)
    fit(k) = brm_svi_fit(rk; subject_group=:g, rng=Xoshiro(11), max_iterations=20_000,
                         ndraws=2000, unit_reoptimize=k)
    stock, reopt = fit(0), fit(1)
    error(f) = abs.(vec(mean(f.draws[rows, :]; dims=2)) .- m) ./ s
    ratio(f) = vec(std(f.draws[rows, :]; dims=2)) ./ s
    # The positive control: with the stock step the intercept's sd is less than
    # 0.6 of the exact one (measured 0.42 at this seed and budget) and the
    # iterate check has not passed.
    @test ratio(stock)[1] < 0.6
    @test !stock.diagnostics.converged
    # Re-optimized units: every global's mean within 0.15 exact sds and its sd
    # within 20% (measured: errors at most 0.07, ratios 0.95-1.02).
    @test all(<(0.15), error(reopt))
    @test all(r -> 0.8 < r < 1.2, ratio(reopt))
    @test reopt.diagnostics.unit_gradient_calls > stock.diagnostics.unit_gradient_calls
end

# The re-optimization step lands each unit on its estimated curvature (todo 12maq21);
# the controls record it, so a stopped run resumes exactly, and a checkpoint written
# before the control (from BRM 0.2.326 on) resumes on the square-root step it ran.
@stestset "checkpoints: a re-optimized run resumes exactly; an older one keeps its step" begin
    data = ridge_data(; G=20, n=5)
    rk = RKBRMI(ridge_model(data))
    # A fresh stream per fit: a shared one would start the second fit elsewhere.
    fit(; kwargs...) = brm_svi_fit(rk; subject_group=:g, rng=Xoshiro(5),
        max_iterations=300, ndraws=20, unit_reoptimize=1, kwargs...)
    full = fit()
    @test full.diagnostics.optimizer.controls.reoptimize_newton === true
    stopped = fit(; stop_after=120)
    @test stopped.diagnostics.iterations == 120
    resumed = brm_svi_resume(rk, stopped; ndraws=20)
    @test resumed.draws == full.draws
    @test resumed.diagnostics.iterations == full.diagnostics.iterations
    @test resumed.diagnostics.elbo == full.diagnostics.elbo
    s = brm_svi_optimizer_checkpoint(stopped)
    c = s.controls
    old = BRMSVIOptimizerState(s.format, s.identity, s.previous,
        NamedTuple{filter(!=(:reoptimize_newton), keys(c))}(c), s.run)
    @test BRM._brm_svi_reoptimize_newton(old.controls) === false
    kept = brm_svi_resume(rk, old; ndraws=20)
    @test kept.diagnostics.iterations == full.diagnostics.iterations
    @test all(isfinite, kept.draws)
    @test kept.diagnostics.elbo[1:120] == full.diagnostics.elbo[1:120]
    @test kept.diagnostics.elbo[121:end] != full.diagnostics.elbo[121:end]
end

# A unit far from Gaussian: test/nlme_svi_trend.jl's ridge, whose units' rows pin
# `exp(m + s z_i)`. Off its optimum one Newton step on such a unit is a guess. When
# every copy's step stood, half the copies in the first 900 iterations read a lower
# objective than the unit's own factor, their residuals pulled the noise sd from 0.83
# to 2.3 (the data's is 0.1), and this fit stopped by its ELBO at exp(m) 0.32, ELBO
# -495 (seeds 1-6: 3 ended more than 0.1 off; research/svi_reoptimize/
# trend_ridge_coupling.jl). With the step standing only where it raises the unit's
# objective, seeds 1-6 read 1.163-1.170 against the 400-window k = 0 fit's 1.166.
@stestset "a non-Gaussian unit: a copy's step stands only where it raises the unit's objective" begin
    rk = RKBRMI(svi_ridge_model(svi_ridge_data(Xoshiro(101), 8)))
    opt = (; subject_group=:unit, batch_size=4, ndraws=200, stationarity=false,
        centering=nothing)
    reopt = brm_svi_fit(rk; rng=Xoshiro(1), opt..., unit_reoptimize=1)
    long = brm_svi_fit(rk; rng=Xoshiro(1), opt..., min_windows=400)
    gm(f) = median(exp.(rk_restore_draws(rk, f.draws).gain_mean))
    @test reopt.diagnostics.converged
    @test abs(gm(reopt) - gm(long)) < 0.02
end
