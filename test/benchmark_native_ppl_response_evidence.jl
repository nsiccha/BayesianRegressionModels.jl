using BayesianRegressionModels
using StanBlocks
using BridgeStan
using Distributions: Exponential, Normal
import DifferentiationInterface as DI
using Enzyme
using LinearAlgebra: dot
using Statistics: median
using SHA: sha256

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL
const GROUPS = 32
const REPLICATES = 8
const N = GROUPS * REPLICATES
const LOWER = -0.4
const UPPER = 0.7
const WARMUP = parse(Int, get(
    ENV, "EVIDENCE_BENCH_WARMUP", "10000"))
const SAMPLES = parse(Int, get(
    ENV, "EVIDENCE_BENCH_SAMPLES", "21"))
const ITERATIONS = parse(Int, get(
    ENV, "EVIDENCE_BENCH_ITERATIONS", "20000"))

const group = repeat(collect(1:GROUPS), inner=REPLICATES)
const x = [
    1.3 * sin(0.071 * row) + 0.45 * cos(0.037 * row) + 0.002 * row
    for row in 1:N
]
const group_truth = [0.42 * sin(0.31 * index) for index in 1:GROUPS]
const mu_truth = [
    0.18 - 0.33 * x[row] + group_truth[group[row]] for row in 1:N
]
const latent_y = [
    mu_truth[row] + 0.62 * sin(0.53 * row) for row in 1:N
]
const y = clamp.(latent_y, LOWER, UPPER)
const data = (; x, group, y)

const brmi = @brm data begin
    sigma ~ Exponential(2)
    mu ~ 1 + x + (1 | g | group)
    sd(:, g) ~ Exponential(1)
    y ~ censored(Normal(mu, sigma); lower=-0.4, upper=0.7)
end

NP.@model function natural_censored_varying_intercept_benchmark(x, group)
    tau_g_group ~ Exponential(1)
    b_g_group[group] ~ Normal(0.0, tau_g_group)
    beta_mu[(:Intercept, :x)] ~ StandardNormal()
    sigma ~ Exponential(2)
    mu = dot(beta_mu, (1, x)) + b_g_group[group]
    @. y ~ censored(Normal(mu, sigma); lower=-0.4, upper=0.7)
end

const natural = NP.condition(
    natural_censored_varying_intercept_benchmark(x, group); y)
const model = NP.lower(brmi)
const plan = NP.compile(brmi)
const prepared = NP.prepare(plan)
const work = NP.workspace(prepared, Float64, DI.AutoEnzyme())

const sb = SBBRMI(brmi; mod=@__MODULE__)
const sb_path = "/tmp/native_ppl_response_evidence_sb.stan"
const sb_problem = StanBlocks.stan_instantiate(sb.model; path=sb_path)
const sb_model = sb_problem.model

array_json(values) = "[" * join(values, ",") * "]"
const pure_data = "{" * join((
    "\"N\":" * string(N),
    "\"G\":" * string(GROUPS),
    "\"x\":" * array_json(x),
    "\"group\":" * array_json(group),
    "\"y\":" * array_json(y),
    "\"lower_bound\":" * string(LOWER),
    "\"upper_bound\":" * string(UPPER)), ",") * "}"
const pure_stan_path = joinpath(
    @__DIR__, "benchmark_native_ppl_response_evidence.stan")
const pure_stan_compile_path =
    "/tmp/native_ppl_response_evidence_pure.stan"
cp(pure_stan_path, pure_stan_compile_path; force=true)
const pure_model = BridgeStan.StanModel(
    pure_stan_compile_path, pure_data)

const log_tau = log(0.65)
const b = [
    0.19 * sin(0.37 * index) - 0.08 * cos(0.17 * index)
    for index in 1:GROUPS
]
const beta = [0.24, -0.36]
const log_sigma = log(0.58)
const native_position = [log_tau, b..., beta..., log_sigma]
const tau = exp(log_tau)
const z = b ./ tau
const stan_position = [log_tau, z..., log_sigma, beta...]
const sb_gradient = zeros(length(stan_position))
const pure_gradient = zeros(length(stan_position))

function stan_gradient_to_native(gradient)
    gradient_z = @view gradient[2:(GROUPS + 1)]
    gradient_beta = @view gradient[(GROUPS + 3):(GROUPS + 4)]
    [gradient[1] - dot(gradient_z, z) - GROUPS,
     (gradient_z ./ tau)..., gradient_beta..., gradient[GROUPS + 2]]
end

normalized_stan_density(density, local_log_tau=log_tau) =
    density - GROUPS * local_log_tau

function native_to_stan(position)
    local_tau = exp(position[1])
    local_b = @view position[2:(GROUPS + 1)]
    local_beta = @view position[(GROUPS + 2):(GROUPS + 3)]
    local_log_sigma = position[GROUPS + 4]
    [position[1], (local_b ./ local_tau)...,
     local_log_sigma, local_beta...]
end

function mapped_finite_difference(model; step=1e-5)
    result = similar(native_position)
    plus = copy(native_position)
    minus = copy(native_position)
    for coordinate in eachindex(result)
        plus[coordinate] += step
        minus[coordinate] -= step
        plus_density = normalized_stan_density(BridgeStan.log_density(
            model, native_to_stan(plus); propto=false, jacobian=true),
            plus[1])
        minus_density = normalized_stan_density(BridgeStan.log_density(
            model, native_to_stan(minus); propto=false, jacobian=true),
            minus[1])
        result[coordinate] = (plus_density - minus_density) / (2step)
        plus[coordinate] = native_position[coordinate]
        minus[coordinate] = native_position[coordinate]
    end
    result
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
    sb_density_raw, _ = BridgeStan.log_density_gradient!(
        sb_model, stan_position, sb_gradient;
        propto=false, jacobian=true)
    pure_density_raw, _ = BridgeStan.log_density_gradient!(
        pure_model, stan_position, pure_gradient;
        propto=false, jacobian=true)
    sb_density = normalized_stan_density(sb_density_raw)
    pure_density = normalized_stan_density(pure_density_raw)
    sb_gradient_native = stan_gradient_to_native(sb_gradient)
    pure_gradient_native = stan_gradient_to_native(pure_gradient)
    sb_finite_difference = mapped_finite_difference(sb_model)
    pure_finite_difference = mapped_finite_difference(pure_model)
    native_sb_delta = native_gradient .- sb_gradient_native
    native_pure_delta = native_gradient .- pure_gradient_native
    sb_pure_delta = sb_gradient_native .- pure_gradient_native
    native_sb_index = argmax(abs.(native_sb_delta))
    native_pure_index = argmax(abs.(native_pure_delta))
    sb_pure_index = argmax(abs.(sb_pure_delta))
    (;
        density=(native=native_density, sb=sb_density, pure=pure_density,
                 native_minus_sb=native_density - sb_density,
                 native_minus_pure=native_density - pure_density,
                 sb_minus_pure=sb_density - pure_density),
        gradient_max_abs_difference=(
            native_sb=maximum(abs.(
                native_gradient .- sb_gradient_native)),
            native_pure=maximum(abs.(
                native_gradient .- pure_gradient_native)),
            sb_pure=maximum(abs.(
                sb_gradient_native .- pure_gradient_native))),
        mapped_finite_difference_max_abs_difference=(
            native_sb=maximum(abs.(
                native_gradient .- sb_finite_difference)),
            native_pure=maximum(abs.(
                native_gradient .- pure_finite_difference)),
            sb_reverse=maximum(abs.(
                sb_gradient_native .- sb_finite_difference)),
            pure_reverse=maximum(abs.(
                pure_gradient_native .- pure_finite_difference))),
        gradient_max_coordinate=(
            native_sb=(index=native_sb_index,
                       native=native_gradient[native_sb_index],
                       comparator=sb_gradient_native[native_sb_index],
                       delta=native_sb_delta[native_sb_index]),
            native_pure=(index=native_pure_index,
                         native=native_gradient[native_pure_index],
                         comparator=pure_gradient_native[native_pure_index],
                         delta=native_pure_delta[native_pure_index]),
            sb_pure=(index=sb_pure_index,
                     sb=sb_gradient_native[sb_pure_index],
                     pure=pure_gradient_native[sb_pure_index],
                     delta=sb_pure_delta[sb_pure_index])))
end

function file_sha(path)
    open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

const equivalence = equivalence_receipt()
@assert natural.declaration == model
@assert maximum(abs, (
    equivalence.density.native_minus_sb,
    equivalence.density.native_minus_pure,
    equivalence.density.sb_minus_pure)) < 1e-10
@assert equivalence.gradient_max_abs_difference.native_pure < 1e-10
@assert equivalence.gradient_max_abs_difference.native_sb < 1e-5
@assert equivalence.gradient_max_abs_difference.sb_pure < 1e-5
@assert equivalence.mapped_finite_difference_max_abs_difference.native_sb <
    1e-7
@assert equivalence.mapped_finite_difference_max_abs_difference.native_pure <
    1e-7

println("context=", (;
    julia_version=VERSION, cpu=Sys.CPU_NAME, threads=Threads.nthreads(),
    rows=N, groups=GROUPS, group_coefficients=1,
    coordinates=length(native_position),
    censor_counts=(lower=count(==(LOWER), y),
                   interior=count(value -> LOWER < value < UPPER, y),
                   upper=count(==(UPPER), y)),
    protocol=(warmup=WARMUP, samples=SAMPLES,
              iterations_per_sample=ITERATIONS)))
println("natural_equals_brm=", natural.declaration == model)
println("brm_names=", (;
    popcoefnames=popcoefnames(brmi, :mu),
    ranefcoefnames=ranefcoefnames(brmi, :g)))
println("native_show=", sprint(show, model))
println("native_plan=", (;
    schedule=plan.graph.schedule,
    dimension=plan.graph.dimension,
    coordinate_names=keys(plan.graph.coordinates),
    coordinate_keys=map(coordinate -> coordinate.keys,
                        plan.graph.coordinates)))
println("parameterization=", (;
    native=(log_tau=:log_tau_g_group, group=:b_g_group,
            beta=:beta_mu, log_sigma=:log_sigma),
    stan=(log_tau=:b_g_group_tau, group=:b_g_group_z_flat,
          log_sigma=:sigma, beta=:pop_mu_beta_pop),
    mapping="b=tau*z; normalized_stan=stan-G*log(tau)",
    gradient_mapping=
        "g_b=g_z/tau; g_u_native=g_u_stan-dot(g_z,z)-G"))
println("sb_unc_names=", BridgeStan.param_unc_names(sb_model))
println("pure_unc_names=", BridgeStan.param_unc_names(pure_model))
println("generated_surfaces=", (;
    sb=(path=sb_path,
        lines=count(==(UInt8('\n')), codeunits(read(sb_path, String))) + 1,
        sha256=file_sha(sb_path)),
    pure=(path=pure_stan_path,
          lines=count(==(UInt8('\n')), codeunits(read(
              pure_stan_path, String))) + 1,
          sha256=file_sha(pure_stan_path))))
println("equivalence=", equivalence)
println("gradient_caveat=", (;
    mapped_finite_difference_step=1e-5,
    sb_reverse_uses="Stan normal_lcdf/normal_lccdf",
    pure_reverse_uses="equivalent erfc log-tail kernel",
    interpretation=
        "SBBRMI density finite differences agree; its Stan reverse rule is the outlier"))
println("linear_head=", NP.evaluate(
    work, prepared, native_position, NP.LinearPredictor())[1:8])
println("allocations=", allocation_receipt())
println("nanoseconds=", timing_receipt())
