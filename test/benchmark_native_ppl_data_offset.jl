using BayesianRegressionModels
using StanBlocks
using BridgeStan
using Distributions: Poisson
import DifferentiationInterface as DI
using Enzyme
using Statistics: median
using SHA: sha256

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL
const N = 256
const WARMUP = parse(Int, get(ENV, "OFFSET_BENCH_WARMUP", "10000"))
const SAMPLES = parse(Int, get(ENV, "OFFSET_BENCH_SAMPLES", "21"))
const ITERATIONS = parse(
    Int, get(ENV, "OFFSET_BENCH_ITERATIONS", "50000"))

const x = [sin(0.071 * row) + 0.45 * cos(0.037 * row) for row in 1:N]
const exposure = [0.35 + 0.025 * mod(37 * row, 91) for row in 1:N]
const log_rate_truth = @. 0.18 - 0.31 * x + log(exposure)
const y = [max(0, round(Int, exp(log_rate_truth[row]) +
                         0.42 * sin(0.53 * row))) for row in 1:N]
const data = (; x, exposure, y)

const brmi = @brm data begin
    log_rate ~ 1 + x + offset(log(exposure))
    y ~ Poisson(exp(log_rate))
end

NP.@model function natural_exposure_poisson_benchmark(x, exposure)
    beta_log_rate[(:Intercept, :x)] ~ StandardNormal()
    log_rate = dot(beta_log_rate, (1, x)) + offset(log(exposure))
    @. y ~ Poisson(exp(log_rate))
end

const natural = NP.condition(
    natural_exposure_poisson_benchmark(x, exposure); y)
const model = NP.lower(brmi)
const plan = NP.compile(brmi)
const prepared = NP.prepare(plan)
const work = NP.workspace(prepared, Float64, DI.AutoEnzyme())

const sb = SBBRMI(brmi; mod=@__MODULE__)
const sb_path = "/tmp/native_ppl_data_offset_sb.stan"
const sb_problem = StanBlocks.stan_instantiate(sb.model; path=sb_path)
const sb_model = sb_problem.model

array_json(values) = "[" * join(values, ",") * "]"
const pure_data = "{" * join((
    "\"N\":" * string(N),
    "\"x\":" * array_json(x),
    "\"exposure\":" * array_json(exposure),
    "\"y\":" * array_json(y)), ",") * "}"
const pure_stan_path = joinpath(
    @__DIR__, "benchmark_native_ppl_data_offset.stan")
const pure_stan_compile_path = "/tmp/native_ppl_data_offset_pure.stan"
cp(pure_stan_path, pure_stan_compile_path; force=true)
const pure_model = BridgeStan.StanModel(pure_stan_compile_path, pure_data)

const position = [0.23, -0.34]
const sb_gradient = zeros(length(position))
const pure_gradient = zeros(length(position))

function warmup!()
    for _ in 1:WARMUP
        NP.logdensity!(work, prepared, position)
        NP.logdensity_and_gradient!(work, prepared, position)
        BridgeStan.log_density(
            sb_model, position; propto=false, jacobian=true)
        BridgeStan.log_density_gradient!(
            sb_model, position, sb_gradient;
            propto=false, jacobian=true)
        BridgeStan.log_density(
            pure_model, position; propto=false, jacobian=true)
        BridgeStan.log_density_gradient!(
            pure_model, position, pure_gradient;
            propto=false, jacobian=true)
    end
end

function allocation_receipt()
    warmup!()
    (;
        native=(
            density=@allocated(NP.logdensity!(work, prepared, position)),
            gradient=@allocated(NP.logdensity_and_gradient!(
                work, prepared, position))),
        sb=(
            density=@allocated(BridgeStan.log_density(
                sb_model, position; propto=false, jacobian=true)),
            gradient=@allocated(BridgeStan.log_density_gradient!(
                sb_model, position, sb_gradient;
                propto=false, jacobian=true))),
        pure=(
            density=@allocated(BridgeStan.log_density(
                pure_model, position; propto=false, jacobian=true)),
            gradient=@allocated(BridgeStan.log_density_gradient!(
                pure_model, position, pure_gradient;
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
                    work, prepared, position)),
                gradient=measure!(() -> NP.logdensity_and_gradient!(
                    work, prepared, position))),
            sb=(
                density=measure!(() -> BridgeStan.log_density(
                    sb_model, position; propto=false, jacobian=true)),
                gradient=measure!(() -> BridgeStan.log_density_gradient!(
                    sb_model, position, sb_gradient;
                    propto=false, jacobian=true))),
            pure=(
                density=measure!(() -> BridgeStan.log_density(
                    pure_model, position; propto=false, jacobian=true)),
                gradient=measure!(() -> BridgeStan.log_density_gradient!(
                    pure_model, position, pure_gradient;
                    propto=false, jacobian=true))))
    finally
        GC.enable(gc_was_enabled)
    end
end

function equivalence_receipt()
    native_density, native_gradient_view =
        NP.logdensity_and_gradient!(work, prepared, position)
    native_gradient = copy(native_gradient_view)
    sb_density, _ = BridgeStan.log_density_gradient!(
        sb_model, position, sb_gradient;
        propto=false, jacobian=true)
    pure_density, _ = BridgeStan.log_density_gradient!(
        pure_model, position, pure_gradient;
        propto=false, jacobian=true)
    (;
        density=(native=native_density, sb=sb_density, pure=pure_density,
                 native_minus_sb=native_density - sb_density,
                 native_minus_pure=native_density - pure_density,
                 sb_minus_pure=sb_density - pure_density),
        gradient_max_abs_difference=(
            native_sb=maximum(abs.(native_gradient .- sb_gradient)),
            native_pure=maximum(abs.(native_gradient .- pure_gradient)),
            sb_pure=maximum(abs.(sb_gradient .- pure_gradient))))
end

function file_sha(path)
    open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

println("context=", (;
    julia_version=VERSION, cpu=Sys.CPU_NAME, threads=Threads.nthreads(),
    rows=N, coordinates=length(position),
    protocol=(warmup=WARMUP, samples=SAMPLES,
              iterations_per_sample=ITERATIONS),
    input_eltypes=(x=eltype(x), exposure=eltype(exposure), y=eltype(y))))
println("natural_equals_brm=", natural.declaration == model)
println("brm_names=", (; popcoefnames=popcoefnames(brmi, :log_rate)))
println("native_show=", sprint(show, model))
println("native_plan=", (;
    schedule=plan.graph.schedule,
    dimension=plan.graph.dimension,
    coordinate_names=keys(plan.graph.coordinates),
    coordinate_keys=map(coordinate -> coordinate.keys,
                        plan.graph.coordinates)))
println("parameterization=", (;
    native=:beta_log_rate_intercept_x,
    stan=:pop_log_rate_beta_pop_intercept_x,
    mapping=:identity,
    log_abs_det_jacobian=0))
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
println("equivalence=", equivalence_receipt())
println("linear_head=", NP.evaluate(
    work, prepared, position, NP.LinearPredictor())[1:8])
println("allocations=", allocation_receipt())
println("nanoseconds=", timing_receipt())
