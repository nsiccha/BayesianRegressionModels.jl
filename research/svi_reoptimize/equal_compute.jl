# brm_svi_fit's unit_reoptimize = 0 against unit_reoptimize = 1 on BRM's own fixtures,
# at equal unit-gradient compute and at the default stop: the driver of snag
# brm-svi-fit-defa-cfc1bb5e's measurements (decision 04z8ti2, upload 98cf6a9e), kept
# here to re-measure k = 1 after its re-optimization step changed (todo 12maq21).
# EQ_KS (default "0,1") selects the k whose rows run; k = 0 rows do not change with
# the re-optimization step.
#
#   ridge      the intercept ridge of todo 122p9l3 (research/svi_reoptimize/intercept_ridge.jl:
#              G = 1,000 units of 20 rows, intercept + 4 slopes, unit SD 0.5 and noise sd 0.1
#              known; exactly Gaussian, so the globals' posterior is closed form), data seeds
#              1-3: k = 1 at 20,000 iterations against k = 0 at 20,000 and 40,000, then both
#              at the default stop (max_iterations 100,000).
#   noridge    the same toy without the intercept (no global-unit ridge), default stop, k = 0, 1.
#   gauss      test/nlme_svi.jl "Gaussian random effects: SVI means match NUTS" (40 subjects,
#              correlated intercept and slope, learned sigma), against its 4-chain NUTS
#              reference, k = 0, 1, fit seeds 3, 4, 5.
#   pois       test/nlme_svi.jl "Poisson random intercepts: SVI means match NUTS" (60 subjects),
#              against its NUTS reference, k = 0, 1, fit seeds 5, 6, 7.
#
#     [EQ_KS=1] julia --project=test research/svi_reoptimize/equal_compute.jl OUTDIR [phase...]
using BayesianRegressionModels, Random, Statistics, LinearAlgebra, Printf
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, WarmupHMC, Distributions
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
const NLME_AD = AutoEnzyme(; mode=Enzyme.Reverse)
const TESTDIR = joinpath(dirname(dirname(@__DIR__)), "test")
const KS = parse.(Int, split(get(ENV, "EQ_KS", "0,1"), ","))
include(joinpath(TESTDIR, "nlme_svi_fixture.jl"))

flushln(xs...) = (println(xs...); flush(stdout))
fmt(v; d=2) = "[" * join([@sprintf("%.*f", d, x) for x in v], ", ") * "]"

# ── The intercept ridge (research/svi_reoptimize/intercept_ridge.jl) ──────────
function ridge_data(; G=1000, n=20, P=4, seed=1)
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

noridge_model(data) = @brm data begin
    mu ~ 0 + x1 + x2 + x3 + x4 + (1 | g)
    effect(mu, :) ~ Normal(0, prior_sd)
    sd(mu, g) = known_sd
    y ~ Normal(mu, noise)
end

# Exact posterior means and sds of the population coefficients (intercept first when present).
function exact_globals(data; intercept=true)
    G = maximum(data.g)
    Xp = hcat(data.x1, data.x2, data.x3, data.x4)
    intercept && (Xp = hcat(ones(length(data.y)), Xp))
    p = size(Xp, 2)
    Z = hcat(Xp, [data.g[r] == j ? 1.0 : 0.0 for r in eachindex(data.y), j in 1:G])
    prior = vcat(fill(1 / data.prior_sd^2, p), fill(1 / data.known_sd^2, G))
    H = Symmetric(Z' * Z ./ data.noise^2 + Diagonal(prior))
    m = H \ (Z' * data.y ./ data.noise^2)
    S = inv(H)
    m[1:p], sqrt.(diag(S)[1:p]), m[p+1:end], sqrt.(diag(S)[p+1:end])
end

function population_rows(rk, p)
    names = String.(BRM._rk_layout_coordinate_names(rk))
    rows = [findfirst(==("pop_mu.beta_pop.$j"), names) for j in 1:p]
    any(isnothing, rows) && error("population names: ", names[1:min(end, 12)])
    rows
end

function ridge_row(label, rk, rows, m, s, k, fitseed; kw...)
    t = @elapsed fit = brm_svi_fit(rk; subject_group=:g, rng=Xoshiro(fitseed), ndraws=2000,
        unit_reoptimize=k, kw...)
    d = fit.diagnostics
    mu = vec(mean(fit.draws[rows, :]; dims=2))
    err = abs.(mu .- m) ./ s
    sdr = vec(std(fit.draws[rows, :]; dims=2)) ./ s
    @printf("| %s | %d | %d | %s | %.3g M | %.1f | %s | %s | %.2f | %.1f |\n", label, k,
        d.iterations, d.converged ? "yes" : "no", d.unit_gradient_calls / 1e6, t,
        fmt(err), fmt(sdr), maximum(err), last(d.elbo_windows))
    flush(stdout)
    (; label, k, d.iterations, d.converged, calls=d.unit_gradient_calls, seconds=t, err, sdr,
       elbo=last(d.elbo_windows))
end

const RHDR = "| Run | k | Iterations | Converged | Unit calls | Seconds | Mean errors, exact sds | sd ratios | Max error | Last-window ELBO |\n|---|---|---|---|---|---|---|---|---|---|"

function phase_ridge(; seeds=1:3)
    out = []
    for seed in seeds
        data = ridge_data(; seed)
        rk = RKBRMI(ridge_model(data))
        rows = population_rows(rk, 5)
        m, s = exact_globals(data)
        flushln("\n### Intercept ridge, data seed $seed (G = 1,000, n = 20): exact means ", fmt(m; d=4),
            ", sds ", fmt(s; d=5), "\n", RHDR)
        fs = 10 + seed
        # The first fit compiles; it is repeated below, so its row is a warm-up only.
        seed == first(seeds) && ridge_row("warm-up (compiles; not a timing)", rk, rows, m, s, 1, fs;
            max_iterations=200)
        for (label, k, iterations) in (("20,000 iterations", 0, 20_000), ("20,000 iterations", 1, 20_000),
                ("40,000 iterations (equal calls)", 0, 40_000), ("default stop", 0, 100_000),
                ("default stop", 1, 100_000))
            k in KS || continue
            push!(out, (; seed, r=ridge_row(label, rk, rows, m, s, k, fs; max_iterations=iterations)))
        end
    end
    out
end

function phase_noridge(; seeds=1:3)
    out = []
    for seed in seeds
        data = ridge_data(; seed)
        rk = RKBRMI(noridge_model(data))
        rows = population_rows(rk, 4)
        m, s = exact_globals(data; intercept=false)
        flushln("\n### No intercept (no ridge), data seed $seed: exact means ", fmt(m; d=4),
            ", sds ", fmt(s; d=5), "\n", RHDR)
        fs = 10 + seed
        for k in KS
            push!(out, (; seed, r=ridge_row("default stop", rk, rows, m, s, k, fs)))
        end
    end
    out
end

# ── BRM's NUTS-referenced fixtures (test/nlme_svi.jl) ─────────────────────────
function svi_pois_data(rng, N; nobs=4, a=0.5, sa=0.6, b=0.3)
    u = sa .* randn(rng, N)
    x = [randn(rng, nobs) for _ in 1:N]
    y = [[rand(rng, Poisson(exp(a + u[i] + b * xj))) for xj in x[i]] for i in 1:N]
    (; subject=["s$i" for i in 1:N], x, y)
end

svi_pois_model(data) = @brm data begin
    a ~ 1 + (1 | subject)
    b ~ Normal(0, 1)
    @plate for i in eachindex(a)
        y[i] ~ Poisson(exp.(a[i] .+ b .* x[i]))
    end
end

function svi_nuts_reference(rk, m; seed, n_draws=1000, chains=4)
    problem = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    u0 = first(BRM._brm_nlme_prior_median_point(m))
    reduce(hcat, map(1:chains) do c
        run = adaptive_warmup_mcmc(Xoshiro(seed + c), problem;
            init=(; position=copy(u0), squared_scale=ones(length(u0))), n_draws)
        convert(Matrix{Float64}, run.posterior_position)
    end)
end

function svi_moments(draws, reference, idx)
    sd = vec(std(reference[idx, :]; dims=2))
    (; mean_error=abs.(vec(mean(draws[idx, :]; dims=2)) .-
        vec(mean(reference[idx, :]; dims=2))) ./ sd,
       sd_ratio=vec(std(draws[idx, :]; dims=2)) ./ sd)
end

const NHDR = "| Fixture | k | Fit seed | Iterations | Converged | Unit calls | Seconds | Global mean errors, NUTS sds | Global sd ratios | Subject mean error (mean) | Subject sd ratio (mean) | Last-window ELBO |\n|---|---|---|---|---|---|---|---|---|---|---|---|"

function nuts_rows(label, rk, m, reference, fitseeds; kw...)
    out = []
    for fs in fitseeds, k in KS
        t = @elapsed fit = brm_svi_fit(m; rng=Xoshiro(fs), ndraws=2000, unit_reoptimize=k, kw...)
        d = fit.diagnostics
        g = svi_moments(fit.draws, reference, d.globals)
        u = svi_moments(fit.draws, reference, setdiff(axes(reference, 1), d.globals))
        @printf("| %s | %d | %d | %d | %s | %.3g M | %.1f | %s | %s | %.3f | %.3f | %.2f |\n", label, k, fs,
            d.iterations, d.converged ? "yes" : "no", d.unit_gradient_calls / 1e6, t,
            fmt(g.mean_error), fmt(g.sd_ratio), mean(u.mean_error), mean(u.sd_ratio),
            last(d.elbo_windows))
        flush(stdout)
        push!(out, (; label, k, fs, d.iterations, d.converged, calls=d.unit_gradient_calls,
            seconds=t, g, umean=mean(u.mean_error), usd=mean(u.sd_ratio), elbo=last(d.elbo_windows)))
    end
    out
end

function phase_gauss()
    rk = RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(2), 40)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    t = @elapsed reference = svi_nuts_reference(rk, m; seed=10)
    flushln("\n### Gaussian random effects (test/nlme_svi.jl fixture, 40 subjects, batch 8); NUTS reference ",
        size(reference, 2), " draws in ", round(t; digits=1), " s\n", NHDR)
    nuts_rows("Gaussian random effects", rk, m, reference, (3, 4, 5); batch_size=8)
end

function phase_pois()
    rk = RKBRMI(svi_pois_model(svi_pois_data(Xoshiro(4), 60)))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    t = @elapsed reference = svi_nuts_reference(rk, m; seed=20)
    flushln("\n### Poisson random intercepts (test/nlme_svi.jl fixture, 60 subjects, batch 10); NUTS reference ",
        size(reference, 2), " draws in ", round(t; digits=1), " s\n", NHDR)
    nuts_rows("Poisson random intercepts", rk, m, reference, (5, 6, 7); batch_size=10)
end

# Every code path above on small inputs and short fits (not a measurement).
function phase_smoke()
    data = ridge_data(; G=50, n=5)
    for (mdl, p, icpt) in ((ridge_model, 5, true), (noridge_model, 4, false))
        rk = RKBRMI(mdl(data)); rows = population_rows(rk, p)
        m, s = exact_globals(data; intercept=icpt)
        flushln(RHDR)
        for k in KS
            ridge_row("smoke", rk, rows, m, s, k, 1; max_iterations=300)
        end
    end
    for (label, rk) in (("smoke gauss", RKBRMI(svi_gauss_model(svi_gauss_data(Xoshiro(2), 10)))),
                        ("smoke pois", RKBRMI(svi_pois_model(svi_pois_data(Xoshiro(4), 10)))))
        m = brm_nlme_model(rk; ad_backend=NLME_AD)
        reference = svi_nuts_reference(rk, m; seed=10, n_draws=100, chains=2)
        flushln(NHDR)
        nuts_rows(label, rk, m, reference, (1,); batch_size=4, max_iterations=300)
    end
    nothing
end

using Serialization
const OUTDIR = ARGS[1]; mkpath(OUTDIR)
phases = length(ARGS) == 1 ? ["ridge", "noridge", "gauss", "pois"] : ARGS[2:end]
flushln("## unit_reoptimize default measurements on BRM ", pkgversion(BRM), " (",
    get(ENV, "BRM_SHA", "?"), "), Julia ", VERSION, ", threads ", Threads.nthreads(),
    ", BLAS threads ", BLAS.get_num_threads(), ", host ", gethostname())
for p in phases
    t = @elapsed r = p == "smoke" ? phase_smoke() : p == "ridge" ? phase_ridge() : p == "noridge" ? phase_noridge() :
        p == "gauss" ? phase_gauss() : p == "pois" ? phase_pois() : error("unknown phase $p")
    serialize(joinpath(OUTDIR, "$p.jls"), r)
    flushln("\nPhase $p done in ", round(t; digits=1), " s")
end
@printf("\nPeak RSS %.1f GB\n", Sys.maxrss() / 2^30)
