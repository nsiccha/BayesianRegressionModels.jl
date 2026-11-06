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
#    (`threads = false`), `pairs` pairs after a warm-up fit of each. The two
#    routes of a pair share a seed and run one right after the other, taking
#    turns going first, so a change in the host's load between pairs reaches
#    both routes. The ratios are lockstep's wall time over threads', so above 1
#    means threads are faster. One fit per route says little: the same code gave
#    0.99 s against 0.82 s and 0.74 s against 0.81 s on a loaded host.
#
# Run it at several thread counts in separate processes on as many CPUs as
# threads (more threads than CPUs measure something else: brm-use, "Subjects on
# threads"):
#
# RUN: julia --project=<NLMEEstimation scratch env> --threads=<T> \
#          research/nlme_threads/benchmark.jl <units> <rows> <repeats> <iterations> [<pairs>]
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
using NLMEEstimation, LinearAlgebra, Printf, Random, Statistics
const BRM = BayesianRegressionModels
const NE = NLMEEstimation

units, rows, repeats, iterations = length(ARGS) >= 4 ? parse.(Int, ARGS[1:4]) : (80, 8, 20, 40)
pairs = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 12
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
load() = join((@sprintf("%.1f", x) for x in Sys.loadavg()), "/")

println("threads=$(Threads.nthreads()) gc_threads=$(Threads.ngcthreads()) ",
    "cpu_grant=$(get(ENV, "KB_COMPUTE_CPUS", "?")) load=$(load())")

reference = serial_round()
isequal(threaded_round(), reference) || error("threaded calls differ from serial calls")
t_serial, t_threaded = best(serial_round), best(threaded_round)
println("ROUND threads=$(Threads.nthreads()) units=$units rows=$rows calls=$(length(calls)) ",
    "serial=$(round(1e3t_serial; digits=2)) ms threaded=$(round(1e3t_threaded; digits=2)) ms ",
    "speedup=$(round(t_serial / t_threaded; digits=2)) evaluators=$(m.cell_gradient.evaluators.count[])")

options = SAEMOptions(; iterations, burnin=iterations ÷ 2, mcmc_burnin=5)
fit(threads, seed) = saem(m, θ, σ, s.omega; rng=Xoshiro(seed), options,
    kernel=WarmupHMCNUTS(; threads))
timed_fit(threads, seed) = (GC.gc(); @timed fit(threads, seed))

foreach(threads -> fit(threads, 1), (true, false))  # compile
samples = map(1:pairs) do k
    order = isodd(k) ? (false, true) : (true, false)
    results = Dict(threads => timed_fit(threads, k + 100) for threads in order)
    a, b = results[false], results[true]
    println(@sprintf("PAIR %2d lockstep=%7.1f ms (GC %5.1f) threads=%7.1f ms (GC %5.1f) largest |Δθ|=%.2e |Δσ|=%.2e",
        k, 1e3a.time, 1e3a.gctime, 1e3b.time, 1e3b.gctime,
        maximum(abs, a.value.theta .- b.value.theta), maximum(abs, a.value.sigma .- b.value.sigma)))
    (; lockstep=a.time, threads=b.time)
end
lockstep, threaded = [x.lockstep for x in samples], [x.threads for x in samples]
ratios = lockstep ./ threaded
println(@sprintf("SAEM threads=%d iterations=%d pairs=%d lockstep=%.3f s threads=%.3f s (medians) lockstep/threads=%.2f (quartiles %.2f–%.2f) threads no slower in %d of %d pairs evaluators=%d load=%s",
    Threads.nthreads(), iterations, pairs, median(lockstep), median(threaded), median(ratios),
    quantile(ratios, 0.25), quantile(ratios, 0.75), count(>=(1), ratios), length(ratios),
    m.cell_gradient.evaluators.count[], load()))
