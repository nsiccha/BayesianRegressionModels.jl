using BayesianRegressionModels
using StanBlocks
using BridgeStan
using Distributions: Exponential, LKJCholesky, Poisson
import DifferentiationInterface as DI
using Enzyme
using Statistics: median

const NP = BayesianRegressionModels.NativePPL
const GROUPS = 8
const REPLICATES = 8
const N = GROUPS * REPLICATES
const WARMUP = 10_000
const SAMPLES = 21
const ITERATIONS = 50_000

const group = repeat(collect(1:GROUPS), outer=REPLICATES)
const x = repeat(collect(range(-1.75, 1.75; length=REPLICATES)), inner=GROUPS)
const true_intercepts = collect(range(-0.45, 0.45; length=GROUPS))
const true_slopes = collect(range(0.22, -0.22; length=GROUPS))
const truth = [
    0.42 * x[row] + true_intercepts[group[row]] +
        true_slopes[group[row]] * x[row]
    for row in 1:N
]
const bernoulli_y = Int[
    mod(37 * row + 11, 101) / 101 <
        BayesianRegressionModels._native_ppl_logistic(truth[row])
    for row in 1:N
]
const poisson_y = Int[
    max(0, round(Int,
        exp(truth[row]) + 0.55 * sin(0.63 * row) + 0.15))
    for row in 1:N
]

const bernoulli_data = (; x, group, y=bernoulli_y)
const poisson_data = (; x, group, y=poisson_y)

const bernoulli_brmi = @brm bernoulli_data begin
    mu ~ 0 + x + (1 + x | p | group)
    sd(:, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ BernoulliLogit(mu)
end

const poisson_brmi = @brm poisson_data begin
    log_rate ~ 0 + x + (1 + x | p | group)
    sd(:, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Poisson(exp(log_rate))
end

function native_case(brmi)
    model = NP.lower(brmi)
    plan = NP.compile(brmi)
    prepared = NP.prepare(plan)
    work = NP.workspace(prepared, Float64, DI.AutoEnzyme())
    (; model, plan, prepared, work)
end

function sb_case(brmi, path)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model; path)
    (; sb, problem, model=problem.model)
end

const bernoulli_native = native_case(bernoulli_brmi)
const poisson_native = native_case(poisson_brmi)
const bernoulli_sb = sb_case(
    bernoulli_brmi, "/tmp/grouped_glmm_bernoulli.stan")
const poisson_sb = sb_case(
    poisson_brmi, "/tmp/grouped_glmm_poisson.stan")

const log_tau = [log(0.7), log(0.45)]
const raw_correlation = 0.3
const z = [0.24 * sin(0.8 * i) - 0.11 * cos(0.35 * i)
           for i in 1:(2 * GROUPS)]
const beta = 0.37
const native_position = [log_tau..., raw_correlation, z..., beta]
const sb_position = [raw_correlation, log_tau..., z..., beta]
const bernoulli_sb_gradient = zeros(length(sb_position))
const poisson_sb_gradient = zeros(length(sb_position))

sb_gradient_to_native(gradient) =
    [gradient[2], gradient[3], gradient[1], gradient[4:end]...]

function warmup!(native, sb, sb_gradient)
    for _ in 1:WARMUP
        NP.logdensity!(native.work, native.prepared, native_position)
        NP.logdensity_and_gradient!(
            native.work, native.prepared, native_position)
        BridgeStan.log_density(
            sb.model, sb_position; propto=false, jacobian=true)
        BridgeStan.log_density_gradient!(
            sb.model, sb_position, sb_gradient;
            propto=false, jacobian=true)
    end
end

function allocation_receipt(native, sb, sb_gradient)
    warmup!(native, sb, sb_gradient)
    (;
        native=(
            density=@allocated(NP.logdensity!(
                native.work, native.prepared, native_position)),
            gradient=@allocated(NP.logdensity_and_gradient!(
                native.work, native.prepared, native_position))),
        sb=(
            density=@allocated(BridgeStan.log_density(
                sb.model, sb_position; propto=false, jacobian=true)),
            gradient=@allocated(BridgeStan.log_density_gradient!(
                sb.model, sb_position, sb_gradient;
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

function timing_receipt(native, sb, sb_gradient)
    warmup!(native, sb, sb_gradient)
    gc_was_enabled = GC.enable(false)
    try
        (;
            native=(
                density=measure!(() -> NP.logdensity!(
                    native.work, native.prepared, native_position)),
                gradient=measure!(() -> NP.logdensity_and_gradient!(
                    native.work, native.prepared, native_position))),
            sb=(
                density=measure!(() -> BridgeStan.log_density(
                    sb.model, sb_position;
                    propto=false, jacobian=true)),
                gradient=measure!(() -> BridgeStan.log_density_gradient!(
                    sb.model, sb_position, sb_gradient;
                    propto=false, jacobian=true))))
    finally
        GC.enable(gc_was_enabled)
    end
end

function equivalence_receipt(native, sb, sb_gradient)
    native_density, native_gradient_view =
        NP.logdensity_and_gradient!(
            native.work, native.prepared, native_position)
    native_gradient = copy(native_gradient_view)
    sb_density, _ = BridgeStan.log_density_gradient!(
        sb.model, sb_position, sb_gradient;
        propto=false, jacobian=true)
    (;
        density=(native=native_density, sb=sb_density,
                 difference=native_density - sb_density),
        gradient_max_abs_difference=maximum(abs.(
            native_gradient .- sb_gradient_to_native(sb_gradient))))
end

function report(label, brmi, native, sb, sb_gradient)
    println(label, "_brm_names=", (;
        popcoefnames=popcoefnames(
            brmi, label === :bernoulli ? :mu : :log_rate),
        ranefcoefnames=ranefcoefnames(brmi, :p)))
    println(label, "_native_show=", sprint(show, native.model))
    println(label, "_native_plan=", (;
        schedule=native.plan.graph.schedule,
        dimension=native.plan.graph.dimension,
        coordinate_names=keys(native.plan.graph.coordinates),
        coordinate_keys=map(
            coordinate -> coordinate.keys,
            native.plan.graph.coordinates)))
    println(label, "_unc_names=",
            BridgeStan.param_unc_names(sb.model))
    println(label, "_equivalence=",
            equivalence_receipt(native, sb, sb_gradient))
    println(label, "_linear_head=", NP.evaluate(
        native.work, native.prepared, native_position,
        NP.LinearPredictor())[1:8])
    println(label, "_allocations=",
            allocation_receipt(native, sb, sb_gradient))
    println(label, "_nanoseconds=",
            timing_receipt(native, sb, sb_gradient))
end

println("context=", (;
    julia_version=VERSION, cpu=Sys.CPU_NAME, threads=Threads.nthreads(),
    rows=N, groups=GROUPS, coordinates=length(native_position),
    protocol=(warmup=WARMUP, samples=SAMPLES,
              iterations_per_sample=ITERATIONS),
    group_eltype=eltype(group)))
report(:bernoulli, bernoulli_brmi, bernoulli_native,
       bernoulli_sb, bernoulli_sb_gradient)
report(:poisson, poisson_brmi, poisson_native,
       poisson_sb, poisson_sb_gradient)
