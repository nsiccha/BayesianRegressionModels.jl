# Minibatch SVI per-iteration time and worker memory against `threads`, on
# synthetic unit-partitioned models. It compares thread counts inside one
# process, so every count runs the same compiled gradients. Every count must give
# the same ELBO trace to the last bit, or the script fails.
#
# Shapes:
# - `flat`: `rows` elementwise rows per unit, one compact cell call per row, so a
#   unit's gradient makes many calls and allocates in proportion.
# - `plate`: one `@plate` cell of `rows` rows per unit and an AR(1) series.
# - `tiny`: 5 rows per unit, the case where a task costs about what a unit does.
#
# RUN: julia --project=test --threads=<max T> research/svi_threads/benchmark.jl \
#          <shape> <units> <rows> <batch> <iterations> [T ...]
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
using LinearAlgebra, Random, Statistics
const BRM = BayesianRegressionModels

function flat_data(rng, N, n)
    subject = repeat(["s$i" for i in 1:N]; inner=n)
    x = randn(rng, N * n)
    a, b = randn(rng, N), randn(rng, N)
    g = repeat(1:N; inner=n)
    (; subject, x, y=0.5 .+ 0.3 .* a[g] .+ (0.2 .+ 0.1 .* b[g]) .* x .+ 0.5 .* randn(rng, N * n))
end
flat_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | g | subject)
    y ~ Normal(mu, sigma)
end

function plate_data(rng, N, n)
    (; subject=["s$i" for i in 1:N], t=[collect(1.0:n) for _ in 1:N],
       y=[0.5 .+ cumsum(0.2 .* randn(rng, n)) .+ 0.3 .* randn(rng, n) for _ in 1:N])
end
plate_model(data) = @brm data begin
    sigma ~ Exponential(1)
    a ~ 1 + (1 | g | subject)
    b ~ 1 + (1 | g | subject)
    @plate for i in eachindex(a)
        y[i] ~ Normal(a[i] .+ b[i] .* log.(t[i]), sigma)
    end
end

function main(shape, N, n, B, iterations, counts)
    BLAS.set_num_threads(1)
    rng = Xoshiro(1)
    builder = shape == "plate" ? plate_model(plate_data(rng, N, n)) : flat_model(flat_data(rng, N, n))
    prepare = @elapsed m = brm_nlme_model(RKBRMI(builder); ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    t = BRM._brm_svi_target(m)
    println("# julia=", VERSION, " nthreads=", Threads.nthreads(), " blas=", BLAS.get_num_threads(),
        " gc_threads=", Threads.ngcthreads(), " cpus_granted=", get(ENV, "KB_COMPUTE_CPUS", "n/a"),
        " shape=", shape, " units=", N, " rows=", n, " batch=", B, " iterations=", iterations,
        " globals=", length(t.globals), " unit_dim=", maximum(length, t.units),
        " prepare_s=", round(prepare; digits=1))
    GC.gc(); live0 = Base.gc_live_bytes()
    t0 = @elapsed w = BRM._brm_svi_with_workers(t, 2)
    GC.gc(); worker_bytes = Base.gc_live_bytes() - live0
    samplers = sum(Base.summarysize, filter(!isnothing, collect(w.plans[2].samplers)); init=0)
    println("# one worker's samplers: prepared_s=", round(t0; digits=2), " retained_live_bytes=",
        worker_bytes, " summarysize_with_shared=", samplers, " rss_mb=", Sys.maxrss() ÷ 2^20)
    # The alternative to preparing a worker's samplers: copying the prepared ones.
    GC.gc(); live1 = Base.gc_live_bytes()
    try
        t1 = @elapsed copies = [s === nothing ? nothing : deepcopy(s) for s in t.plan.samplers]
        GC.gc(); copy_bytes = Base.gc_live_bytes() - live1
        println("# the same ", count(!isnothing, copies), " samplers by deepcopy: copied_s=",
            round(t1; digits=2), " retained_live_bytes=", copy_bytes)
    catch e
        println("# deepcopy of the samplers failed: ", sprint(showerror, e))
    end
    # A fit advances its `rng`, so every fit gets its own, at the same seed.
    controls = (; batch_size=B, window=max(iterations, 1), min_windows=10^6,
        max_iterations=10^7, stop_after=iterations, ndraws=0)
    # Compiles what the first iteration needs, so no count pays for it.
    brm_svi_fit(m; rng=Xoshiro(2), controls..., stop_after=2, threads=1)
    reference = nothing
    base = nothing
    println("T\tseconds\tper_iter_s\tspeedup\tgc_fraction\talloc_mb_per_iter\trss_mb\tsame_elbo")
    for T in counts
        before = Base.gc_num()
        fit = brm_svi_fit(m; rng=Xoshiro(2), controls..., threads=T)
        d = Base.GC_Diff(Base.gc_num(), before)
        seconds = fit.diagnostics.seconds
        if reference === nothing
            reference, base = fit.diagnostics.elbo, seconds
        end
        same = fit.diagnostics.elbo == reference
        println(T, "\t", round(seconds; digits=3), "\t", round(seconds / iterations; sigdigits=4), "\t",
            round(base / seconds; digits=2), "\t", round(d.total_time / 1e9 / seconds; digits=3), "\t",
            round(Base.gc_alloc_count(d) == 0 ? 0.0 : d.allocd / 2^20 / iterations; digits=2), "\t",
            Sys.maxrss() ÷ 2^20, "\t", same)
        flush(stdout)
        same || error("threads = $T changed the ELBO trace")
    end
end

let a = ARGS
    shape = a[1]
    counts = length(a) > 5 ? parse.(Int, a[6:end]) : [1, 2, 4, 8]
    main(shape, parse(Int, a[2]), parse(Int, a[3]), parse(Int, a[4]), parse(Int, a[5]), counts)
end
