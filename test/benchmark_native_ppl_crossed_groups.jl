using BayesianRegressionModels
using StanBlocks
using BridgeStan
using Distributions: Exponential, LKJCholesky, Normal
import DifferentiationInterface as DI
using Enzyme
using Statistics: median
using SHA: sha256

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL
const SUBJECTS = 8
const ITEMS = 10
const N = SUBJECTS * ITEMS
const WARMUP = parse(Int, get(ENV, "CROSSED_BENCH_WARMUP", "10000"))
const SAMPLES = parse(Int, get(ENV, "CROSSED_BENCH_SAMPLES", "21"))
const ITERATIONS = parse(
    Int, get(ENV, "CROSSED_BENCH_ITERATIONS", "50000"))

const subject = repeat(collect(1:SUBJECTS), outer=ITEMS)
const item = repeat(collect(1:ITEMS), inner=SUBJECTS)
const x = [sin(0.31 * row) + 0.6 * cos(0.13 * row) for row in 1:N]
const subject_intercepts = collect(range(-0.45, 0.45; length=SUBJECTS))
const subject_slopes = collect(range(0.24, -0.24; length=SUBJECTS))
const item_intercepts = [0.18 * sin(0.7 * index) for index in 1:ITEMS]
const truth = [
    0.42 * x[row] + subject_intercepts[subject[row]] +
        subject_slopes[subject[row]] * x[row] + item_intercepts[item[row]]
    for row in 1:N
]
const y = [truth[row] + 0.31 * sin(0.73 * row) for row in 1:N]
const data = (; x, subject, item, y)

const brmi = @brm data begin
    sigma ~ Exponential(2)
    mu ~ 0 + x + (1 + x | p | subject) + (1 | q | item)
    sd(:, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky(2, 2)
    sd(:, q) ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

NP.@model function natural_crossed_group_regression(x, subject, item)
    tau_p_subject[(:Intercept, :x)] ~ Exponential(1)
    L_p_subject[(:Intercept, :x)] ~ LKJCholesky(2, 2)
    b_p_subject[subject, (:Intercept, :x)] ~
        MvNormalCholesky(tau_p_subject, L_p_subject)
    tau_q_item ~ Exponential(1)
    b_q_item[item] ~ Normal(0.0, tau_q_item)
    beta_mu[(:x,)] ~ StandardNormal()
    sigma ~ Exponential(2)
    mu = dot(beta_mu, (x,)) +
        dot(b_p_subject[subject], (1, x)) + b_q_item[item]
    @. y ~ Normal(mu, sigma)
end

const natural = NP.condition(
    natural_crossed_group_regression(x, subject, item); y)
const model = NP.lower(brmi)
const plan = NP.compile(brmi)
const prepared = NP.prepare(plan)
const work = NP.workspace(prepared, Float64, DI.AutoEnzyme())

const sb = SBBRMI(brmi; mod=@__MODULE__)
const sb_problem = StanBlocks.stan_instantiate(
    sb.model; path="/tmp/native_ppl_crossed_groups_sb.stan")
const sb_model = sb_problem.model

array_json(values) = "[" * join(values, ",") * "]"
const pure_data = "{" * join((
    "\"N\":" * string(N),
    "\"S\":" * string(SUBJECTS),
    "\"I\":" * string(ITEMS),
    "\"x\":" * array_json(x),
    "\"subject\":" * array_json(subject),
    "\"item\":" * array_json(item),
    "\"y\":" * array_json(y)), ",") * "}"
const pure_stan_path = joinpath(
    @__DIR__, "benchmark_native_ppl_crossed_groups.stan")
const pure_stan_compile_path = "/tmp/native_ppl_crossed_groups_pure.stan"
cp(pure_stan_path, pure_stan_compile_path; force=true)
const pure_model = BridgeStan.StanModel(pure_stan_compile_path, pure_data)

const log_tau_p = [log(0.7), log(0.45)]
const raw_correlation = 0.3
const z_p = [0.24 * sin(0.8 * i) - 0.11 * cos(0.35 * i)
             for i in 1:(2 * SUBJECTS)]
const log_tau_q = log(0.55)
const z_q = [0.21 * sin(0.51 * i) + 0.07 * cos(0.33 * i)
             for i in 1:ITEMS]
const b_q = exp(log_tau_q) .* z_q
const beta = 0.37
const log_sigma = log(0.6)
const native_position = [
    log_tau_p..., raw_correlation, z_p..., log_tau_q, b_q...,
    beta, log_sigma]
const stan_position = [
    raw_correlation, log_tau_p..., z_p..., log_tau_q, z_q...,
    log_sigma, beta]
const sb_gradient = zeros(length(stan_position))
const pure_gradient = zeros(length(stan_position))

function native_gradient_to_stan(native_gradient)
    z_p_end = 3 + 2 * SUBJECTS
    tau_q_index = z_p_end + 1
    b_q_range = (tau_q_index + 1):(tau_q_index + ITEMS)
    beta_index = last(b_q_range) + 1
    sigma_index = beta_index + 1
    [native_gradient[3], native_gradient[1:2]...,
     native_gradient[4:z_p_end]...,
     native_gradient[tau_q_index] +
        sum(native_gradient[b_q_range] .* b_q) + ITEMS,
     (exp(log_tau_q) .* native_gradient[b_q_range])...,
     native_gradient[sigma_index], native_gradient[beta_index]]
end

function warmup!()
    for _ in 1:WARMUP
        NP.logdensity!(work, prepared, native_position)
        NP.logdensity_and_gradient!(work, prepared, native_position)
        BridgeStan.log_density(
            sb_model, stan_position; propto=false, jacobian=true)
        BridgeStan.log_density_gradient!(
            sb_model, stan_position, sb_gradient;
            propto=false, jacobian=true)
        BridgeStan.log_density(
            pure_model, stan_position; propto=false, jacobian=true)
        BridgeStan.log_density_gradient!(
            pure_model, stan_position, pure_gradient;
            propto=false, jacobian=true)
    end
end

function allocation_receipt()
    warmup!()
    (;
        native=(
            density=@allocated(NP.logdensity!(
                work, prepared, native_position)),
            gradient=@allocated(NP.logdensity_and_gradient!(
                work, prepared, native_position))),
        sb=(
            density=@allocated(BridgeStan.log_density(
                sb_model, stan_position; propto=false, jacobian=true)),
            gradient=@allocated(BridgeStan.log_density_gradient!(
                sb_model, stan_position, sb_gradient;
                propto=false, jacobian=true))),
        pure=(
            density=@allocated(BridgeStan.log_density(
                pure_model, stan_position; propto=false, jacobian=true)),
            gradient=@allocated(BridgeStan.log_density_gradient!(
                pure_model, stan_position, pure_gradient;
                propto=false, jacobian=true))))
end

function measure!(f)
    samples = zeros(SAMPLES)
    for sample in eachindex(samples)
        start = time_ns()
        for _ in 1:ITERATIONS
            f()
        end
        samples[sample] = (time_ns() - start) / ITERATIONS
    end
    (; median=median(samples), minimum=minimum(samples))
end

function timing_receipt()
    warmup!()
    gc_was_enabled = GC.enable(false)
    try
        (;
            native=(
                density=measure!(() -> NP.logdensity!(
                    work, prepared, native_position)),
                gradient=measure!(() -> NP.logdensity_and_gradient!(
                    work, prepared, native_position))),
            sb=(
                density=measure!(() -> BridgeStan.log_density(
                    sb_model, stan_position;
                    propto=false, jacobian=true)),
                gradient=measure!(() -> BridgeStan.log_density_gradient!(
                    sb_model, stan_position, sb_gradient;
                    propto=false, jacobian=true))),
            pure=(
                density=measure!(() -> BridgeStan.log_density(
                    pure_model, stan_position;
                    propto=false, jacobian=true)),
                gradient=measure!(() -> BridgeStan.log_density_gradient!(
                    pure_model, stan_position, pure_gradient;
                    propto=false, jacobian=true))))
    finally
        GC.enable(gc_was_enabled)
    end
end

function equivalence_receipt()
    native_density, native_gradient_view =
        NP.logdensity_and_gradient!(work, prepared, native_position)
    native_gradient = copy(native_gradient_view)
    native_density_stan_measure = native_density + ITEMS * log_tau_q
    native_gradient_stan = native_gradient_to_stan(native_gradient)
    sb_density, _ = BridgeStan.log_density_gradient!(
        sb_model, stan_position, sb_gradient;
        propto=false, jacobian=true)
    pure_density, _ = BridgeStan.log_density_gradient!(
        pure_model, stan_position, pure_gradient;
        propto=false, jacobian=true)
    (;
        density=(native=native_density,
                 native_stan_measure=native_density_stan_measure,
                 sb=sb_density, pure=pure_density,
                 native_minus_sb=native_density_stan_measure - sb_density,
                 native_minus_pure=native_density_stan_measure - pure_density,
                 sb_minus_pure=sb_density - pure_density),
        gradient_max_abs_difference=(
            native_sb=maximum(abs.(native_gradient_stan .- sb_gradient)),
            native_pure=maximum(abs.(native_gradient_stan .- pure_gradient)),
            sb_pure=maximum(abs.(sb_gradient .- pure_gradient))))
end

function file_sha(path)
    open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

println("context=", (;
    julia_version=VERSION, cpu=Sys.CPU_NAME, threads=Threads.nthreads(),
    rows=N, subjects=SUBJECTS, items=ITEMS,
    subject_coefficients=2, coordinates=length(native_position),
    protocol=(warmup=WARMUP, samples=SAMPLES,
              iterations_per_sample=ITERATIONS),
    group_eltypes=(subject=eltype(subject), item=eltype(item))))
println("natural_equals_brm=", natural.declaration == model)
println("brm_names=", (;
    popcoefnames=popcoefnames(brmi, :mu),
    subject_ranefcoefnames=ranefcoefnames(brmi, :p),
    item_ranefcoefnames=ranefcoefnames(brmi, :q)))
println("native_show=", sprint(show, model))
println("native_plan=", (;
    schedule=plan.graph.schedule,
    dimension=plan.graph.dimension,
    coordinate_names=keys(plan.graph.coordinates),
    coordinate_keys=map(coordinate -> coordinate.keys,
                        plan.graph.coordinates)))
println("parameterization=", (;
    native_item=:centered_b_q,
    stan_item=:noncentered_z_q,
    mapping="b_q = exp(log_tau_q) * z_q",
    log_abs_det_jacobian="ITEMS * log_tau_q"))
println("sb_unc_names=", BridgeStan.param_unc_names(sb_model))
println("pure_unc_names=", BridgeStan.param_unc_names(pure_model))
println("generated_surfaces=", (;
    sb=(path="/tmp/native_ppl_crossed_groups_sb.stan",
        lines=count(==(UInt8('\n')), codeunits(read(
            "/tmp/native_ppl_crossed_groups_sb.stan", String))) + 1,
        sha256=file_sha("/tmp/native_ppl_crossed_groups_sb.stan")),
    pure=(path=pure_stan_path,
          lines=count(==(UInt8('\n')), codeunits(read(
              pure_stan_path, String))) + 1,
          sha256=file_sha(pure_stan_path))))
println("equivalence=", equivalence_receipt())
println("linear_head=", NP.evaluate(
    work, prepared, native_position, NP.LinearPredictor())[1:8])
println("allocations=", allocation_receipt())
println("nanoseconds=", timing_receipt())
