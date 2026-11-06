# A per-site read-pair gmrf field through brm_svi_fit: does the real path accept it, what
# does a step cost, and does it recover the field? Synthetic sites and rows only; the
# truth is a Matérn ν = 1 field at the sites with known hyperparameters.
#
#     julia --project=test research/svi_field_units/readpair_path.jl sites rows iterations [more groups of three]
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme, Bessels
using DifferentiationInterface, LinearAlgebra, Random, Statistics, Printf
const BRM = BayesianRegressionModels

matern1(d, ℓ) = d == 0 ? 1.0 : (x = sqrt(2) * d / ℓ; x * besselk1(x))

function site_data(sites, rows; σ=1.0, ℓ=0.3, τ=0.3, seed=1)
    rng = Xoshiro(seed)
    px, py = rand(rng, sites), rand(rng, sites)
    K = [matern1(hypot(px[i] - px[j], py[i] - py[j]), ℓ) for i in 1:sites, j in 1:sites]
    f = σ .* (cholesky(Symmetric(K + 1e-8I)).L * randn(rng, sites))
    slope = 0.3 .* randn(rng, sites)
    site = repeat(1:sites; inner=rows)
    x = randn(rng, length(site))
    y = f[site] .+ slope[site] .* x .+ τ .* randn(rng, length(site))
    (; east=px[site], north=py[site], time=zeros(length(site)), x, y, site=string.("s", site),
       state=collect(eachindex(site)), nb=min(30, sites - 1)), f
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
        brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), subject_group=:site)
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
    # The field's coordinates are standardized; its read is the SD times the state.
    hyper(key) = findfirst(n -> endswith(n, key), names)
    constrained = [BRM.constrain(rk.model.layout, fit.draws[:, j]) for j in axes(fit.draws, 2)]
    term = first(k for k in keys(constrained[1]) if startswith(String(k), "gmrf"))
    sd_draws = [getproperty(c[term], :sigma) for c in constrained]
    ℓ_draws = [getproperty(c[term], :spatial_length) for c in constrained]
    field = reduce(hcat, [getproperty(c[term], :field) for c in constrained]) .* sd_draws'
    position = unique(i -> (data.east[i], data.north[i]), eachindex(data.east))
    order = sortperm([findfirst(==(string("s", s)), data.site) for s in 1:sites])
    estimate = vec(mean(field; dims=2))
    spread = vec(std(field; dims=2))
    @printf("READPAIR field states %d, median posterior SD %.3f, field SD truth %.3f, sd draw mean %.3f, ℓ draw mean %.3f (truth 0.3)\n",
        size(field, 1), median(spread), std(truth), mean(sd_draws), mean(ℓ_draws))
    @printf("READPAIR peak_rss %.2f GB\n", parse(Int, match(r"VmHWM:\s+(\d+)", read("/proc/self/status", String)).captures[1]) / 2^20)
    flush(stdout)
    nothing
end

BLAS.set_num_threads(1)
for group in Iterators.partition(parse.(Int, ARGS), 3)
    main(group...)
end
