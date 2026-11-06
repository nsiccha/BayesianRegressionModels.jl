# A read-pair field with its read states in units' own factors (`field_states=:units`)
# against the same field with one global factor per state (`:global`) and the exact
# Gaussian posterior, through brm_svi_fit. Synthetic stations and rows only: the truth is a
# Matérn ν = 1 × exponential field at stations reading a shared set of times.
#
#     julia --project=test research/svi_field_units/units_path.jl regime...
#
# Regimes:
#   moderate      several rows per state, blocks within the dense limit
#   weak          one noisy row per state: does a learned length stay off its limit?
#   lowrank       stations with more reads than the dense limit, so low-rank unit factors
#   lowrank_dense the same data with `unit_dense_limit=128`: dense unit factors under `:units`
#   scale         step cost and memory at growing state counts (no exact reference)
#   multires      the moderate data through `multiresolution=3` under `:global`: do the learned
#                 lengths stay near the truth, against the read-pair fit of the same data?
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, LinearAlgebra, Random, Statistics, Printf
using SpecialFunctions: besselk
const BRM = BayesianRegressionModels

matern1(d, ℓ) = d == 0 ? 1.0 : (x = sqrt(2) * d / ℓ; x * besselk(1, x))

function station_data(; stations, times, reads, rows, σ=1.0, ℓs=0.3, ℓt=2.0, τ=0.3, seed=1, exact=true)
    rng = Xoshiro(seed)
    grid = collect(float.(1:times))
    px, py = rand(rng, stations), rand(rng, stations)
    # One (station, time) per state, in row order.
    states = [(s, t) for s in 1:stations for t in sort!(randperm(rng, times)[1:reads])]
    n = length(states)
    slope = 0.3 .* randn(rng, stations)
    if exact
        K = [matern1(hypot(px[a[1]] - px[b[1]], py[a[1]] - py[b[1]]), ℓs) *
             exp(-abs(grid[a[2]] - grid[b[2]]) / ℓt) for a in states, b in states]
        f = σ .* (cholesky(Symmetric(K + 1e-8I)).L * randn(rng, n))
    else
        K = nothing
        f = σ .* randn(rng, n)
    end
    state_of_row = repeat(1:n; inner=rows)
    x = randn(rng, length(state_of_row))
    station = [states[k][1] for k in state_of_row]
    y = f[state_of_row] .+ slope[station] .* x .+ τ .* randn(rng, length(state_of_row))
    data = (; east=px[station], north=py[station], time=[grid[states[k][2]] for k in state_of_row],
        x, y, site=string.("s", station), state=collect(eachindex(state_of_row)), nb=min(20, n - 1))
    first_row = 1 .+ rows .* (0:(n - 1))
    K === nothing && return data, (; f, first_row, posterior=nothing)
    # The exact posterior of the states at the true hyperparameters, given the slopes.
    means = [mean((y .- slope[station] .* x)[state_of_row .== k]) for k in 1:n]
    prior = σ^2 .* K
    F = cholesky(Symmetric(prior + (τ^2 / rows) * I))
    centre = I - fill(1 / n, n, n)
    mean_exact = prior * (F \ means)
    covariance = prior - prior * (F \ prior)
    sd_exact = sqrt.(max.(diag(centre * covariance * centre), 0.0))
    data, (; f, first_row, posterior=(; mean=mean_exact, sd=sd_exact))
end

function station_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; neighbors=nb)
        sd(field, gmrf((east, north), time)) ~ Exponential(1)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.3), 0.7)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(3.0), 0.7)
        mu ~ 1 + (0 + x | site)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
        end
    end
end

# The same data through a `multiresolution=` field, which `field_states=:units` refuses;
# `:global` holds its shared innovations as global coordinates.
function multiresolution_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; multiresolution=3)
        sd(field, gmrf((east, north), time)) ~ Exponential(1)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.3), 0.7)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(3.0), 0.7)
        mu ~ 1 + (0 + x | site)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[s], sigma)
        end
    end
end

peak_rss() = parse(Int, match(r"VmHWM:\s+(\d+)", read("/proc/self/status", String)).captures[1]) / 2^20

centred(x) = x .- mean(x)

function score(rk, fit, truth)
    picks = round.(Int, range(1, size(fit.draws, 2); length=min(200, size(fit.draws, 2))))
    outputs = rk_draw_values(rk, fit.draws[:, picks], (:field,)).field
    field = reduce(hcat, [centred(o[truth.first_row]) for o in outputs])
    estimate, spread = vec(mean(field; dims=2)), vec(std(field; dims=2))
    constrained = [constrain(rk.model.layout, fit.draws[:, j]) for j in picks]
    term = first(k for k in keys(constrained[1]) if startswith(String(k), "gmrf"))
    value(name) = mean(getproperty(c[term], name) for c in constrained)
    noise = mean(c.sigma for c in constrained)
    target = centred(truth.f)
    inside = mean(abs.(estimate .- target) .<= 1.645 .* spread)
    exact = truth.posterior
    (; from_exact=exact === nothing ? NaN : sqrt(mean(abs2, estimate .- centred(exact.mean))),
       from_truth=sqrt(mean(abs2, estimate .- target)),
       sd_ratio=exact === nothing ? NaN : median(spread ./ max.(exact.sd, 1e-12)), inside,
       sd=value(:sigma), spatial=value(:spatial_length), temporal=value(:temporal_length), noise)
end

function recover(label; iterations, batch_size=8, modes=(:global, :units), build=station_model,
        fit_options=(;), shape...)
    data, truth = station_data(; shape...)
    rk = RKBRMI(build(data))
    m = brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), subject_group=:site,
        unit_partition=:connected)
    states = length(truth.f)
    println("UNITS == $label: $states states, $(length(data.y)) rows, $iterations iterations; ",
        "truth σ=", get(shape, :σ, 1.0), " ℓs=", get(shape, :ℓs, 0.3), " ℓt=", get(shape, :ℓt, 2.0), " τ=", get(shape, :τ, 0.3))
    if truth.posterior !== nothing
        @printf("UNITS   exact: from truth %.3f, median centred SD %.3f, field SD %.3f\n",
            sqrt(mean(abs2, centred(truth.posterior.mean) .- centred(truth.f))),
            median(truth.posterior.sd), std(truth.f))
    end
    flush(stdout)
    for mode in modes
        timed = @timed brm_svi_fit(m; batch_size, max_iterations=iterations, ndraws=200, threads=1,
            rng=Xoshiro(3), separator=:none, field_states=mode, fit_options...)
        fit = timed.value
        s = score(rk, fit, truth)
        d = fit.diagnostics
        @printf("UNITS   %-7s from exact %.3f, from truth %.3f, SD/exact %.2f, 90%% cover %.2f, σ %.2f, ℓs %.3f, ℓt %.2f, τ %.3f; %.4f s/step, %.1f s in all, globals %d, final ELBO %.1f\n",
            mode, s.from_exact, s.from_truth, s.sd_ratio, s.inside, s.sd, s.spatial, s.temporal, s.noise,
            d.seconds / max(d.iterations, 1), timed.time, length(d.globals),
            mean(d.elbo[max(1, end - 49):end]))
        flush(stdout)
    end
    @printf("UNITS   peak_rss %.2f GB\n", peak_rss())
    flush(stdout)
end

function scale(sizes; iterations=200, reads=4, times=8, rows=2)
    for stations in sizes
        data, truth = station_data(; stations, times, reads, rows, exact=false)
        rk = RKBRMI(station_model(data))
        m = brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), subject_group=:site,
            unit_partition=:connected)
        for mode in (:global, :units)
            GC.gc()
            before = Base.gc_live_bytes()
            target = @timed BRM._brm_svi_target(m; field_states=mode, separator=:none)
            fit = brm_svi_fit(m; batch_size=8, max_iterations=iterations, ndraws=0, threads=1,
                rng=Xoshiro(3), separator=:none, field_states=mode)
            d = fit.diagnostics
            @printf("UNITS scale %6d states, %-7s target %.1f s, target size %.1f MB, q size %.1f MB, %.4f s/step, peak_rss %.2f GB\n",
                length(truth.f), mode, target.time, Base.summarysize(target.value) / 2^20,
                Base.summarysize(d.q) / 2^20, d.seconds / max(d.iterations, 1), peak_rss())
            flush(stdout)
        end
    end
end

const REGIMES = Dict(
    "moderate" => () -> recover("several rows per state, blocks within the dense limit";
        iterations=4000, stations=64, times=8, reads=4, rows=4, τ=0.3),
    "weak" => () -> recover("one noisy row per state";
        iterations=4000, stations=64, times=8, reads=4, rows=1, τ=1.0, ℓs=0.5),
    "lowrank" => () -> recover("more reads per station than the dense limit";
        iterations=4000, stations=8, times=100, reads=80, rows=2, τ=0.3, ℓt=10.0),
    "lowrank_dense" => () -> recover("lowrank's data with dense unit factors";
        iterations=4000, stations=8, times=100, reads=80, rows=2, τ=0.3, ℓt=10.0,
        modes=(:units,), fit_options=(; unit_dense_limit=128)),
    "scale" => () -> scale((64, 256, 1024, 4096)),
    "multires" => () -> recover("a multiresolution field on the moderate data";
        iterations=2000, modes=(:global,), build=multiresolution_model,
        stations=64, times=8, reads=4, rows=4, τ=0.3),
)

BLAS.set_num_threads(1)
for name in ARGS
    REGIMES[name]()
end
