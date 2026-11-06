# A per-site read-pair gmrf field through brm_svi_fit: does the real path accept it, what
# does a step cost, and does it recover the field? Synthetic sites and rows only; the
# truth is a Matérn ν = 1 field at the sites with known hyperparameters.
#
#     julia --project=test research/svi_field_units/readpair_path.jl sites rows iterations [more groups of three]
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, LinearAlgebra, Random, Statistics, Printf
using SpecialFunctions: besselk
const BRM = BayesianRegressionModels

matern1(d, ℓ) = d == 0 ? 1.0 : (x = sqrt(2) * d / ℓ; x * besselk(1, x))

function site_data(sites, rows; σ=1.0, ℓ=0.3, τ=0.3, seed=1)
    rng = Xoshiro(seed)
    px, py = rand(rng, sites), rand(rng, sites)
    K = [matern1(hypot(px[i] - px[j], py[i] - py[j]), ℓ) for i in 1:sites, j in 1:sites]
    f = σ .* (cholesky(Symmetric(K + 1e-8I)).L * randn(rng, sites))
    slope = 0.3 .* randn(rng, sites)
    site = repeat(1:sites; inner=rows)
    x = randn(rng, length(site))
    y = f[site] .+ slope[site] .* x .+ τ .* randn(rng, length(site))
    # The exact posterior mean of f at the true hyperparameters, given the slopes.
    means = [mean(y[site .== s] .- slope[s] .* x[site .== s]) for s in 1:sites]
    exact = σ^2 .* K * (cholesky(Symmetric(σ^2 .* K + (τ^2 / rows) * I)) \ means)
    (; east=px[site], north=py[site], time=zeros(length(site)), x, y, site=string.("s", site),
       state=collect(eachindex(site)), nb=min(30, sites - 1)), (; f, exact, first_row=1 .+ rows .* (0:sites-1))
end

function site_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; neighbors=nb)
        sd(field, gmrf((east, north), time)) ~ Exponential(1)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.3), 0.7)
        length_scale(field, gmrf((east, north), time), time) = 10.0
        mu ~ 1 + (0 + x | site)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
        end
    end
end

step(f, name) = begin
    timed = @timed f()
    @printf("READPAIR step %-22s %9.1f s %10.1f MB, gc %.1f s\n", name, timed.time, timed.bytes / 2^20, timed.gctime)
    flush(stdout)
    timed.value
end

function main(sites, rows, iterations)
    println("READPAIR_TOY sites=$sites rows=$rows iterations=$iterations threads=$(Threads.nthreads())")
    flush(stdout)
    data, truth = site_data(sites, rows)
    rk = step(() -> RKBRMI(site_model(data)), "lower to RK")
    m = step("brm_nlme_model") do
        brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), subject_group=:site,
            unit_partition=:connected)
    end
    names = String.(m.view.coordinates)
    states = findall(n -> occursin(".field.", n), names)
    println("READPAIR coordinates=", length(names), " field_states=", length(states))
    fit = step("brm_svi_fit") do
        brm_svi_fit(m; batch_size=min(8, sites), max_iterations=iterations, ndraws=200, threads=1, rng=Xoshiro(3))
    end
    d = fit.diagnostics
    @printf("READPAIR %d iterations, %.4f s per step, finite ELBO %s, globals %d\n", d.iterations,
        d.seconds / max(d.iterations, 1), all(isfinite, d.elbo), length(d.globals))
    # The term's own output, on the scale the model reads it, at each site's first row.
    picks = round.(Int, range(1, size(fit.draws, 2); length=min(100, size(fit.draws, 2))))
    outputs = rk_draw_values(rk, fit.draws[:, picks], (:field,)).field
    field = reduce(hcat, [o[truth.first_row] for o in outputs])
    estimate, spread = vec(mean(field; dims=2)), vec(std(field; dims=2))
    constrained = [constrain(rk.model.layout, fit.draws[:, j]) for j in picks]
    term = first(k for k in keys(constrained[1]) if startswith(String(k), "gmrf"))
    sd_draws = [getproperty(c[term], :sigma) for c in constrained]
    ℓ_draws = [getproperty(c[term], :spatial_length) for c in constrained]
    # The intercept and the field trade off, so compare centred.
    centred(x) = x .- mean(x)
    rmse(a, b) = sqrt(mean(abs2, centred(a) .- centred(b)))
    @printf("READPAIR recovery rmse against truth %.3f, exact posterior mean against truth %.3f, against exact %.3f (field SD %.3f)\n",
        rmse(estimate, truth.f), rmse(truth.exact, truth.f), rmse(estimate, truth.exact), std(truth.f))
    @printf("READPAIR median posterior SD %.3f, sd draw mean %.3f (truth 1), ℓ draw mean %.3f (truth 0.3)\n",
        median(spread), mean(sd_draws), mean(ℓ_draws))
    @printf("READPAIR peak_rss %.2f GB\n", parse(Int, match(r"VmHWM:\s+(\d+)", read("/proc/self/status", String)).captures[1]) / 2^20)
    flush(stdout)
    nothing
end

BLAS.set_num_threads(1)
for group in Iterators.partition(parse.(Int, ARGS), 3)
    main(group...)
end
