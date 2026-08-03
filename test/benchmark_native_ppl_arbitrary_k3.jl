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
const GROUPS = 8
const REPLICATES = 8
const N = GROUPS * REPLICATES
const WARMUP = parse(Int, get(ENV, "K3_BENCH_WARMUP", "10000"))
const SAMPLES = parse(Int, get(ENV, "K3_BENCH_SAMPLES", "21"))
const ITERATIONS = parse(Int, get(ENV, "K3_BENCH_ITERATIONS", "50000"))

const group = repeat(collect(1:GROUPS), outer=REPLICATES)
const x = repeat(collect(range(-1.75, 1.75; length=REPLICATES)), inner=GROUPS)
const w = [sin(0.31 * row) + 0.15 * cos(0.17 * row) for row in 1:N]
const group_intercepts = collect(range(-0.45, 0.45; length=GROUPS))
const group_x_slopes = collect(range(0.22, -0.22; length=GROUPS))
const group_w_slopes = [0.16 * sin(0.7 * index) for index in 1:GROUPS]
const truth = [
    0.42 * x[row] - 0.27 * w[row] + group_intercepts[group[row]] +
        group_x_slopes[group[row]] * x[row] +
        group_w_slopes[group[row]] * w[row]
    for row in 1:N
]
const y = [truth[row] + 0.31 * sin(0.73 * row) for row in 1:N]
const data = (; x, w, group, y)

const brmi = @brm data begin
    sigma ~ Exponential(2)
    mu ~ 0 + x + w + (1 + x + w | p | group)
    sd(:, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky(3, 2)
    y ~ Normal(mu, sigma)
end

NP.@model function natural_correlated_varying_three(x, w, group)
    tau_p_group[(:Intercept, :x, :w)] ~ Exponential(1)
    L_p_group[(:Intercept, :x, :w)] ~ LKJCholesky(3, 2)
    b_p_group[group, (:Intercept, :x, :w)] ~
        MvNormalCholesky(tau_p_group, L_p_group)
    beta_mu[(:x, :w)] ~ StandardNormal()
    sigma ~ Exponential(2)
    mu = dot(beta_mu, (x, w)) +
        dot(b_p_group[group], (1, x, w))
    @. y ~ Normal(mu, sigma)
end

const natural = NP.condition(
    natural_correlated_varying_three(x, w, group); y)
const model = NP.lower(brmi)
const plan = NP.compile(brmi)
const prepared = NP.prepare(plan)
const work = NP.workspace(prepared, Float64, DI.AutoEnzyme())

const sb = SBBRMI(brmi; mod=@__MODULE__)
const sb_problem = StanBlocks.stan_instantiate(
    sb.model; path="/tmp/native_ppl_arbitrary_k3_sb.stan")
const sb_model = sb_problem.model

array_json(values) = "[" * join(values, ",") * "]"
const pure_data = "{" * join((
    "\"N\":" * string(N),
    "\"G\":" * string(GROUPS),
    "\"x\":" * array_json(x),
    "\"w\":" * array_json(w),
    "\"group\":" * array_json(group),
    "\"y\":" * array_json(y)), ",") * "}"
const pure_stan_path = joinpath(
    @__DIR__, "benchmark_native_ppl_arbitrary_k3.stan")
const pure_stan_compile_path = "/tmp/native_ppl_arbitrary_k3_pure.stan"
cp(pure_stan_path, pure_stan_compile_path; force=true)
const pure_model = BridgeStan.StanModel(pure_stan_compile_path, pure_data)

const log_tau = [log(0.7), log(0.45), log(0.55)]
const raw_correlation = [0.3, -0.2, 0.25]
const z = [0.24 * sin(0.8 * i) - 0.11 * cos(0.35 * i)
           for i in 1:(3 * GROUPS)]
const beta = [0.37, -0.21]
const log_sigma = log(0.6)
const native_position = [
    log_tau..., raw_correlation..., z..., beta..., log_sigma]
const stan_position = [
    raw_correlation..., log_tau..., z..., log_sigma, beta...]
const sb_gradient = zeros(length(stan_position))
const pure_gradient = zeros(length(stan_position))

function stan_gradient_to_native(gradient)
    z_end = 6 + 3 * GROUPS
    [gradient[4:6]...,
     gradient[1:3]...,
     gradient[7:z_end]...,
     gradient[(z_end + 2):(z_end + 3)]...,
     gradient[z_end + 1]]
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
    sb_density, _ = BridgeStan.log_density_gradient!(
        sb_model, stan_position, sb_gradient;
        propto=false, jacobian=true)
    pure_density, _ = BridgeStan.log_density_gradient!(
        pure_model, stan_position, pure_gradient;
        propto=false, jacobian=true)
    (;
        density=(native=native_density, sb=sb_density, pure=pure_density,
                 native_minus_sb=native_density - sb_density,
                 native_minus_pure=native_density - pure_density,
                 sb_minus_pure=sb_density - pure_density),
        gradient_max_abs_difference=(
            native_sb=maximum(abs.(native_gradient .-
                stan_gradient_to_native(sb_gradient))),
            native_pure=maximum(abs.(native_gradient .-
                stan_gradient_to_native(pure_gradient))),
            sb_pure=maximum(abs.(sb_gradient .- pure_gradient))))
end

function file_sha(path)
    open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

println("context=", (;
    julia_version=VERSION, cpu=Sys.CPU_NAME, threads=Threads.nthreads(),
    rows=N, groups=GROUPS, group_coefficients=3,
    coordinates=length(native_position),
    protocol=(warmup=WARMUP, samples=SAMPLES,
              iterations_per_sample=ITERATIONS),
    group_eltype=eltype(group)))
println("natural_equals_brm=", natural.declaration == model)
println("brm_names=", (;
    popcoefnames=popcoefnames(brmi, :mu),
    ranefcoefnames=ranefcoefnames(brmi, :p)))
println("native_show=", sprint(show, model))
println("native_plan=", (;
    schedule=plan.graph.schedule,
    dimension=plan.graph.dimension,
    coordinate_names=keys(plan.graph.coordinates),
    coordinate_keys=map(coordinate -> coordinate.keys,
                        plan.graph.coordinates)))
println("sb_unc_names=", BridgeStan.param_unc_names(sb_model))
println("pure_unc_names=", BridgeStan.param_unc_names(pure_model))
println("generated_surfaces=", (;
    sb=(path="/tmp/native_ppl_arbitrary_k3_sb.stan",
        lines=count(==(UInt8('\n')), codeunits(read(
            "/tmp/native_ppl_arbitrary_k3_sb.stan", String))) + 1,
        sha256=file_sha("/tmp/native_ppl_arbitrary_k3_sb.stan")),
    pure=(path=pure_stan_path,
          lines=count(==(UInt8('\n')), codeunits(read(pure_stan_path, String))) + 1,
          sha256=file_sha(pure_stan_path))))
println("equivalence=", equivalence_receipt())
println("linear_head=", NP.evaluate(
    work, prepared, native_position, NP.LinearPredictor())[1:8])
println("allocations=", allocation_receipt())
println("nanoseconds=", timing_receipt())
