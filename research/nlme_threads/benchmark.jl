# Threaded SAEM on a BRM model: what `WarmupHMCNUTS(threads = true)` gains from
# the model's evaluator pool (src/nlme_view.jl, "Concurrent one-unit calls").
#
# A synthetic plate PK model: `units` subjects with `rows` observations each, a
# one-compartment curve per subject. Two measurements, both in this process:
#
# 1. A round of one-unit calls: every subject's value and η gradient,
#    `repeats` times, serially and then on every thread
#    (`Threads.@threads :dynamic`). Results must agree to the last bit.
# 2. SAEM fits with `WarmupHMCNUTS(threads = true)` and with lockstep
#    (`threads = false`), same seed and options, after a warm-up fit of each.
#
# Run it at several thread counts in separate processes (the first is the
# serial baseline) on as many CPUs as threads:
#
# RUN: julia --project=<NLMEEstimation scratch env> --threads=<T> \
#          research/nlme_threads/benchmark.jl <units> <rows> <repeats> <iterations>
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
using NLMEEstimation, LinearAlgebra, Random, Statistics
const BRM = BayesianRegressionModels
const NE = NLMEEstimation

units, rows, repeats, iterations = length(ARGS) >= 4 ? parse.(Int, ARGS[1:4]) : (80, 8, 20, 40)
BLAS.set_num_threads(1)

function pk_data(rng, n, r)
    times = collect(range(0.5, 12.0; length=r))
    weight = 0.3 .* randn(rng, n)
    dose = rand(rng, [80.0, 100.0, 120.0], n)
    dv = map(1:n) do i
        CL = exp(log(2.0) + 0.5 * weight[i] + 0.25 * randn(rng))
        V = exp(log(10.0) + 0.2 * randn(rng))
        dose[i] / V .* exp.(-(CL / V) .* times) .+ 0.4 .* randn(rng, r)
    end
    (; subject=["s$i" for i in 1:n], dose, weight, t=fill(times, n), dv)
end

pk_model(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | pk | subject)
    log_V ~ 1 + (1 | pk | subject)
    @plate for i in eachindex(log_CL)
        CL = exp(log_CL[i])
        Vc = exp(log_V[i])
        pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
        dv[i] ~ normal(pred[i], sigma)
    end
end

m = brm_nlme_model(RKBRMI(pk_model(pk_data(Xoshiro(1), units, rows)));
    ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
s = brm_nlme_start(m)
θ, σ, H = s.theta, s.sigma, s.eta
calls = [(i, H[:, i]) for i in 1:units for _ in 1:repeats]

serial_round() = [NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η) for (i, η) in calls]
function threaded_round()
    out = Vector{Any}(undef, length(calls))
    Threads.@threads :dynamic for k in eachindex(calls)
        i, η = calls[k]
        out[k] = NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, η)
    end
    out
end
best(f; n=7) = (f(); minimum(@elapsed(f()) for _ in 1:n))

reference = serial_round()
isequal(threaded_round(), reference) || error("threaded calls differ from serial calls")
t_serial, t_threaded = best(serial_round), best(threaded_round)
println("ROUND threads=$(Threads.nthreads()) units=$units rows=$rows calls=$(length(calls)) ",
    "serial=$(round(1e3t_serial; digits=2)) ms threaded=$(round(1e3t_threaded; digits=2)) ms ",
    "speedup=$(round(t_serial / t_threaded; digits=2)) evaluators=$(m.cell_gradient.evaluators.count[])")

options = SAEMOptions(; iterations, burnin=iterations ÷ 2, mcmc_burnin=5)
fit(threads) = saem(m, θ, σ, s.omega; rng=Xoshiro(2), options,
    kernel=WarmupHMCNUTS(; threads))
for threads in (true, false)
    fit(threads)  # compile
    GC.gc()
    stats = @timed fit(threads)
    r = stats.value
    println("SAEM threads=$(Threads.nthreads()) kernel_threads=$threads iterations=$iterations ",
        "seconds=$(round(stats.time; digits=2)) gc_fraction=$(round(stats.gctime / stats.time; digits=3)) ",
        "theta=$(round.(r.theta; digits=4)) sigma=$(round.(r.sigma; digits=4))")
end
