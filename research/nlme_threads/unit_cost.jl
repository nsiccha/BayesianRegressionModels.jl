# What one subject's one-unit NLME call costs against its share of the
# whole-cohort sweep, and which layer holds the difference.
#
# `WarmupHMCNUTS(threads = true)` replaces lockstep's batched sweeps by one-unit
# calls (`conditional_loglikelihood[_and_gradient](m, i, …)`), so a threaded round
# costs Σᵢ unitᵢ where lockstep's costs one sweep per leapfrog step. The ratio
# Σᵢ unitᵢ / sweep is what threads must first win back (brm-use nlme.md,
# "Subjects on threads"). Cheap subject cells make it large.
#
# Two public synthetic models, `units` subjects each:
#   pk     one-compartment curve, `pk_rows` observations; CL on four covariates
#   joint  oral one-compartment curve plus a second response linear in the
#          concentration at `qt_rows` times, covariates on all five predictors
#
# For each, warm, in this process (BLAS on one thread):
#   sweep     one `conditional_loglikelihoods[_and_gradients]` over all subjects
#   protocol  every subject's own one-unit call through the NLMEEstimation protocol
#   rk        the same subjects through RK's partition query and compact local
#             sampler directly, at a point already holding their values: BRM's
#             evaluator pool, point writes and unit checks bypassed
# Per-call times are a loop's time over its calls, best of `trials` loops of
# `reps` repetitions alternating between two η sets; allocations are per call.
#
# RUN: julia --project=<NLMEEstimation scratch env, test/README.md> \
#          research/nlme_threads/unit_cost.jl [units=24] [pk_rows=10] [qt_rows=8] [pk|joint|both]
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
using NLMEEstimation, LinearAlgebra, Printf, Random
const BRM = BayesianRegressionModels
const NE = NLMEEstimation
BLAS.set_num_threads(1)

units = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 24
pk_rows = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
qt_rows = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8
variants = length(ARGS) >= 4 && ARGS[4] != "both" ? [ARGS[4]] : ["pk", "joint"]

function cohort(rng, n, r, q)
    t = collect(range(0.5, 24.0; length=r))
    tq = collect(range(1.0, 23.0; length=q))
    male = Float64.(rand(rng, Bool, n))
    age = 40 .+ 10 .* randn(rng, n)
    weight = 70 .+ 12 .* randn(rng, n)
    ind = rand(rng, ["a", "b", "c"], n)
    dose = rand(rng, [80.0, 100.0, 120.0], n)
    dv, resp = Vector{Vector{Float64}}(undef, n), Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        CL = exp(log(2.0) + 0.2 * male[i] + 0.25 * randn(rng))
        V = exp(log(20.0) + 0.2 * randn(rng))
        k = CL / V
        ka = k + exp(0.2 * randn(rng))
        c(tt) = dose[i] * ka / (V * (ka - k)) * (exp(-k * tt) - exp(-ka * tt))
        dv[i] = c.(t) .+ 0.2 .* randn(rng, r)
        resp[i] = 400 .+ 5 .* randn(rng) .+ (2.0 + 0.3 * randn(rng)) .* c.(tq) .+ 3 .* randn(rng, q)
    end
    (; subject=["s$i" for i in 1:n], male, age, weight, ind, dose,
        t=fill(t, n), tq=fill(tq, n), dv, resp)
end

pk_model(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + male + standardize(age) + standardize(weight) + ind + (1 | pk | subject)
    log_V ~ 1 + (1 | pk | subject)
    @plate for i in eachindex(log_CL)
        CL = exp(log_CL[i])
        Vc = exp(log_V[i])
        pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
        dv[i] ~ normal(pred[i], sigma)
    end
end

joint_model(data) = @brm data begin
    sigma ~ Exponential(1)
    sigma_resp ~ Exponential(5)
    log_CL ~ 1 + male + standardize(age) + standardize(weight) + ind + (1 | pk | subject)
    log_V ~ 1 + male + standardize(age) + standardize(weight) + ind + (1 | pk | subject)
    log_dka ~ 1 + ind + (1 | pk | subject)
    base ~ 1 + male + standardize(age) + ind + (1 | rb | subject)
    slope ~ 1 + ind + (1 | rs | subject)
    @plate for i in eachindex(log_CL)
        CL = exp(log_CL[i])
        Vc = exp(log_V[i])
        k = CL / Vc
        ka = k + exp(log_dka[i])
        scale = dose[i] * ka / (Vc * (ka - k))
        pred[i] = scale .* (exp.(-k .* t[i]) .- exp.(-ka .* t[i]))
        conc = scale .* (exp.(-k .* tq[i]) .- exp.(-ka .* tq[i]))
        dv[i] ~ normal(pred[i], sigma)
        resp[i] ~ normal(base[i] .+ slope[i] .* conc, sigma_resp)
    end
end

# Each loop is a function of its arguments, so no call reads a global.
sweep_value(m, s, θ, σ, H) = NE.conditional_loglikelihoods(m, s, θ, σ, H)
sweep_gradient(m, s, θ, σ, H) = NE.conditional_loglikelihoods_and_gradients(m, s, θ, σ, H)
function protocol_value(m, θ, σ, etas)
    acc = 0.0
    for i in eachindex(etas)
        acc += NE.conditional_loglikelihood(m, i, θ, σ, etas[i])
    end
    acc
end
function protocol_gradient(m, θ, σ, etas)
    acc = 0.0
    for i in eachindex(etas)
        acc += first(NE.conditional_loglikelihood_and_gradient(m, i, θ, σ, etas[i]))
    end
    acc
end
function rk_value(query, point, n)
    acc = 0.0
    for i in 1:n
        acc += BRM._rk_nlme_partition_value(query, point, i)
    end
    acc
end
function rk_gradient(plan, point, n)
    copyto!(plan.point, point)
    acc = 0.0
    for i in 1:n
        acc += first(BRM._rk_nlme_partition_value_and_local_gradient!(plan.sampler,
            plan.local_gradients[i], plan.point, i))
    end
    acc
end

# Repetitions alternate between two argument sets whose η differ, so no call can
# be answered from a cache of the previous call at the same point.
function best_time(f, a, b; reps=300, trials=9)
    f(a...); f(b...)
    best = Inf
    for _ in 1:trials
        t = @elapsed for r in 1:reps
            isodd(r) ? f(a...) : f(b...)
        end
        best = min(best, t / reps)
    end
    best
end
allocated(f, args...) = (f(args...); @allocated f(args...))

println("julia=$(VERSION) BRM=$(pkgversion(BRM)) ReactiveKernels=$(pkgversion(ReactiveKernels)) ",
    "ReactiveKernelsPPL=$(pkgversion(ReactiveKernelsPPL)) NLMEEstimation=$(pkgversion(NE)) ",
    "Enzyme=$(pkgversion(Enzyme)) load=$(join((@sprintf("%.1f", x) for x in Sys.loadavg()), "/"))")
for variant in variants
    data = cohort(Xoshiro(1), units, pk_rows, qt_rows)
    m = brm_nlme_model(RKBRMI(variant == "pk" ? pk_model(data) : joint_model(data));
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    start = brm_nlme_start(m)
    θ, σ = start.theta, start.sigma
    n = NE.nsubjects(m)
    eta(H, i) = H isa AbstractMatrix ? H[:, i] : H[i]
    shifted(H) = H isa AbstractMatrix ? H .+ 0.01 : [h .+ 0.01 for h in H]
    Ha, Hb = start.eta, shifted(start.eta)
    etas_a, etas_b = [eta(Ha, i) for i in 1:n], [eta(Hb, i) for i in 1:n]
    subjects = collect(1:n)
    plan = BRM._brm_nlme_cell_state(m).plan
    query = m.cell_gradient.query
    pack(H) = BRM._brm_nlme_pack_point(m.unit, m.theta, m.sigma,
        m.view.subject_coordinates, θ, σ, H)
    point_a, point_b = pack(Ha), pack(Hb)

    t_sweep_v = best_time(sweep_value, (m, subjects, θ, σ, Ha), (m, subjects, θ, σ, Hb))
    t_sweep_g = best_time(sweep_gradient, (m, subjects, θ, σ, Ha), (m, subjects, θ, σ, Hb))
    t_unit_v = best_time(protocol_value, (m, θ, σ, etas_a), (m, θ, σ, etas_b))
    t_unit_g = best_time(protocol_gradient, (m, θ, σ, etas_a), (m, θ, σ, etas_b))
    a_unit_v = allocated(protocol_value, m, θ, σ, etas_a) ÷ n
    a_unit_g = allocated(protocol_gradient, m, θ, σ, etas_a) ÷ n
    # The direct gradient writes the cohort's point into the model's own plan,
    # which holds `m.unit` outside a call; put `m.unit` back afterwards.
    t_rk_v = best_time(rk_value, (query, point_a, n), (query, point_b, n))
    t_rk_g = best_time(rk_gradient, (plan, point_a, n), (plan, point_b, n))
    a_rk_v = allocated(rk_value, query, point_a, n) ÷ n
    a_rk_g = allocated(rk_gradient, plan, point_a, n) ÷ n
    copyto!(plan.point, m.unit)
    θ_len, eta_len = length(θ), length(etas_a[1])

    us(x) = 1e6 * x / n
    @printf("%s units=%d pk_rows=%d qt_rows=%d theta=%d sigma=%d eta=%d\n",
        variant, n, pk_rows, qt_rows, θ_len, length(σ), eta_len)
    @printf("  value     sweep share %.3f µs | protocol %.3f µs (%d B) | rk %.3f µs (%d B) | Σ unit / sweep %.2f (rk %.2f)\n",
        us(t_sweep_v), us(t_unit_v), a_unit_v, us(t_rk_v), a_rk_v, t_unit_v / t_sweep_v, t_rk_v / t_sweep_v)
    @printf("  gradient  sweep share %.3f µs | protocol %.3f µs (%d B) | rk %.3f µs (%d B) | Σ unit / sweep %.2f (rk %.2f)\n",
        us(t_sweep_g), us(t_unit_g), a_unit_g, us(t_rk_g), a_rk_g, t_unit_g / t_sweep_g, t_rk_g / t_sweep_g)
end
