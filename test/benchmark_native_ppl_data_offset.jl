using BayesianRegressionModels
using StanBlocks
using BridgeStan
using Distributions: Normal, Poisson
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

# Julianic run-the-body surface — the SAME exposure-offset Poisson model authored
# as `@jmodel`, pinned DIRECTLY against Stan here. The declarative `offset(...)`
# helper has no run-the-body counterpart because a Julia body just ADDS the term:
# `log_rate = … .+ log.(exposure)`. θ layout is declaration order = the declarative
# `native` order = `position` (identity map, like the other flat scenarios), so it
# is shared unmapped. Flat/elementwise — no group gather, differentiates under
# plain AutoEnzyme.
NP.@jmodel function julianic_exposure_poisson_benchmark(x, exposure)
    beta_log_rate_intercept ~ Normal()
    beta_log_rate_x ~ Normal()
    log_rate = beta_log_rate_intercept .+ beta_log_rate_x .* x .+ log.(exposure)
    @. y ~ Poisson(exp(log_rate))
end

const jprepared = NP.jprepare(
    NP.jcondition(julianic_exposure_poisson_benchmark(x, exposure); y))
const jwork = NP.jworkspace(jprepared, Float64, DI.AutoEnzyme())

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
        NP.jlogdensity(jprepared, position)
        NP.jlogdensity_and_gradient!(jwork, jprepared, position)
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
                propto=false, jacobian=true))),
        jl=(
            density=@allocated(NP.jlogdensity(jprepared, position)),
            gradient=@allocated(NP.jlogdensity_and_gradient!(
                jwork, jprepared, position))))
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
                    propto=false, jacobian=true))),
            jl=(
                density=measure!(() -> NP.jlogdensity(
                    jprepared, position)),
                gradient=measure!(() -> NP.jlogdensity_and_gradient!(
                    jwork, jprepared, position))))
    finally
        GC.enable(gc_was_enabled)
    end
end

function equivalence_receipt()
    native_density, native_gradient_view =
        NP.logdensity_and_gradient!(work, prepared, position)
    native_gradient = copy(native_gradient_view)
    jl_density, jl_gradient_view =
        NP.jlogdensity_and_gradient!(jwork, jprepared, position)
    jl_gradient = copy(jl_gradient_view)
    sb_density, _ = BridgeStan.log_density_gradient!(
        sb_model, position, sb_gradient;
        propto=false, jacobian=true)
    pure_density, _ = BridgeStan.log_density_gradient!(
        pure_model, position, pure_gradient;
        propto=false, jacobian=true)
    (;
        density=(native=native_density, julianic=jl_density,
                 sb=sb_density, pure=pure_density,
                 native_minus_sb=native_density - sb_density,
                 native_minus_pure=native_density - pure_density,
                 julianic_minus_sb=jl_density - sb_density,
                 julianic_minus_pure=jl_density - pure_density,
                 julianic_minus_native=jl_density - native_density,
                 sb_minus_pure=sb_density - pure_density),
        gradient_max_abs_difference=(
            native_sb=maximum(abs.(native_gradient .- sb_gradient)),
            native_pure=maximum(abs.(native_gradient .- pure_gradient)),
            julianic_sb=maximum(abs.(jl_gradient .- sb_gradient)),
            julianic_pure=maximum(abs.(jl_gradient .- pure_gradient)),
            julianic_native=maximum(abs.(jl_gradient .- native_gradient)),
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
println("julianic=", (;
    dimension=NP.dimension(jprepared),
    matches_native_dimension=NP.dimension(jprepared) == plan.graph.dimension,
    coordinate_order=:declaration_order_identity_to_native))
println("equivalence=", equivalence_receipt())
println("linear_head=", NP.evaluate(
    work, prepared, position, NP.LinearPredictor())[1:8])
# Four-way wall-clock, hand-optimized Stan (`pure`) as the CEILING to beat:
# julianic_over_hand_stan < 1.0 ⇒ julianic is FASTER than hand-optimized Stan,
# > 1.0 ⇒ that many times slower. nppl(declarative) = in-house speed reference,
# sbbrmi = backend reference. Primal (density) and gradient (HMC path) both.
# CAVEAT: julianic ALLOCATES (workspace-free primal ~10 KB; grouped gradient
# 5-14 KB) where nppl/Stan are 0-alloc, and `timing_receipt` disables GC, so
# julianic's high-iteration wall-clock is inflated by accumulated garbage and
# is protocol-sensitive. `allocations=` is the DETERMINISTIC companion metric
# (isolation-confirmed intrinsic); read the ratios as directional, alloc as exact.
function speed_comparison(timing)
    over(arm) = (
        primal=timing.jl.density.median / arm.density.median,
        gradient=timing.jl.gradient.median / arm.gradient.median)
    (;
        wall_clock_median_ns=(
            hand_stan=(primal=timing.pure.density.median,
                       gradient=timing.pure.gradient.median),
            nppl_declarative=(primal=timing.native.density.median,
                              gradient=timing.native.gradient.median),
            sbbrmi=(primal=timing.sb.density.median,
                    gradient=timing.sb.gradient.median),
            julianic=(primal=timing.jl.density.median,
                      gradient=timing.jl.gradient.median)),
        julianic_over_hand_stan=over(timing.pure),
        julianic_over_nppl_declarative=over(timing.native),
        julianic_over_sbbrmi=over(timing.sb))
end

println("allocations=", allocation_receipt())
const timing_result = timing_receipt()
println("nanoseconds=", timing_result)
println("speed_comparison=", speed_comparison(timing_result))
