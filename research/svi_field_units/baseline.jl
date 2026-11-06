# The cost of a model-scope gmrf field under today's minibatch SVI, where its
# coordinates are one shared global block. Synthetic stations read a synthetic
# field; a unit's footprint stays the same while the field grows, so everything
# that grows with `states` below is one of the three scaling problems:
#
#   1. unit structures as long as the globals (joint vector, split, write, restore,
#      and the per-unit gradient buffers),
#   2. the whole prior on every step,
#   3. the dense global update (noise, Adam and the covariance operators over 2D).
#
#     julia --project=test research/svi_field_units/baseline.jl side times stations rows batch [more groups of five]
#
# Runs on the tree of the commit that added this file: the next commit removes the
# unit structures it times. Several sizes in one process separate first-use
# compilation from what repeats.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface, Random, Statistics, Printf, LinearAlgebra
const BRM = BayesianRegressionModels

function field_toy_data(; side, times, stations, rows, seed=1)
    rng = Xoshiro(seed)
    id(i, j) = i + side * (j - 1)
    xy = zeros(side^2, 2)
    for j in 1:side, i in 1:side
        xy[id(i, j), 1], xy[id(i, j), 2] = (i - 1) / (side - 1), (j - 1) / (side - 1)
    end
    triangles = Matrix{Int}(undef, 2(side - 1)^2, 3)
    c = 0
    for j in 1:side-1, i in 1:side-1
        triangles[c+=1, :] = [id(i, j), id(i + 1, j), id(i + 1, j + 1)]
        triangles[c+=1, :] = [id(i, j), id(i + 1, j + 1), id(i, j + 1)]
    end
    # Every time appears, so the field's state count is side² × times.
    grid = collect(0.0:times-1)
    px, py = rand(rng, stations) .* 0.9 .+ 0.05, rand(rng, stations) .* 0.9 .+ 0.05
    east, north, time, station = Float64[], Float64[], Float64[], String[]
    for s in 1:stations, t in sort!(randperm(rng, times)[1:min(rows, times)])
        push!(east, px[s]); push!(north, py[s]); push!(time, grid[t]); push!(station, "s$s")
    end
    missing_times = setdiff(grid, time)
    for t in missing_times
        s = rand(rng, 1:stations)
        push!(east, px[s]); push!(north, py[s]); push!(time, t); push!(station, "s$s")
    end
    n = length(east)
    y = sin.(2π .* east) .* cos.(2π .* north) .+ 0.1 .* cos.(time ./ 5) .+ 0.2 .* randn(rng, n)
    (; east, north, time, station, y, state=collect(1:n), xy, triangles)
end

function field_toy_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        sd(field, gmrf((east, north), time)) ~ Exponential(1)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(0.5), 0.5)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(10.0), 0.5)
        mu ~ 1 + (1 | station)
        @plate for s in eachindex(state)
            y[s] ~ Normal(mu[s] .+ field[state[s]], sigma)
        end
    end
end

# Warm cost: discard two calls, then the mean of `n`.
function cost(f, n=5)
    f(); f()
    GC.gc()
    bytes = @allocated for _ in 1:n
        f()
    end
    seconds = @elapsed for _ in 1:n
        f()
    end
    (; seconds=seconds / n, bytes=bytes / n)
end
rss_gb() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 2^30
peak_gb() = parse(Int, match(r"VmHWM:\s+(\d+)", read("/proc/self/status", String)).captures[1]) / 2^20
report(name, x) = (@printf("BASELINE %-26s %12.6f s %14.1f KB\n", name, x.seconds, x.bytes / 1024); flush(stdout))
step(f, name) = begin
    timed = @timed f()
    @printf("BASELINE step %-24s %9.1f s %10.1f MB allocated, rss %.2f GB\n", name, timed.time, timed.bytes / 2^20, rss_gb())
    flush(stdout)
    timed.value
end

function main(side, times, stations, rows, batch)
    println("BASELINE_TOY side=$side times=$times stations=$stations rows=$rows batch=$batch threads=$(Threads.nthreads())")
    data = field_toy_data(; side, times, stations, rows)
    flush(stdout)
    rk = step(() -> RKBRMI(field_toy_model(data)), "lower to RK")
    m = step("brm_nlme_model") do
        brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), subject_group=:station,
            unit_partition=:connected)
    end
    # The target's parts, in its own order, then the target itself.
    ownership = step(() -> BRM._brm_svi_connected_ownership(m; separator=:none), "ownership")
    partition, view = step(() -> BRM._brm_svi_partition(m, ownership), "partition queries")
    step(() -> BRM._brm_nlme_partition_plan(partition, m.unit, ownership.globals, view; units=ownership.units), "partition sampler")
    step(() -> BRM._brm_svi_authored_prior(m, ownership.globals, ownership.units; factors=ownership.factors), "authored prior")
    t = step(() -> BRM._brm_svi_target(m; separator=:none), "whole target")
    D = length(t.globals)
    field = count(n -> occursin(".field.", String(n)), m.view.coordinates)
    footprints = length.(t.plan.footprints)
    println("BASELINE coordinates=", length(m.unit), " field_states=", field, " globals=", D,
        " units=", length(t.units), " unit_block=", maximum(length, t.units),
        " footprint_median=", median(footprints), " footprint_max=", maximum(footprints))
    @printf("BASELINE retained target %.1f MB: partition_plan %.1f MB, per-unit selections %.3f MB\n",
        Base.summarysize(t) / 2^20, Base.summarysize(t.plan) / 2^20, Base.summarysize(t.plan.indices) / 2^20)

    g = copy(m.unit[t.globals])
    i = argmax(footprints)
    z = zeros(length(t.units[i]))
    gg, gz = zeros(D), zeros(length(z))
    report("unit_term (whole)", cost(() -> BRM._brm_svi_unit_term!(gg, gz, t, i, g, z)))
    # Problem 1: what a unit term spends outside the footprint's own evaluation.
    values, gradient = view(t.work[1].values, 1:D+length(z)), view(t.work[1].gradient, 1:D+length(z))
    report("  joint vector", cost(() -> BRM._brm_svi_joint!(values, t.layouts[i], g, z)))
    report("  split gradient", cost(() -> BRM._brm_svi_split_gradient!(gg, gz, t.layouts[i], gradient)))
    report("  write and restore point", cost(() -> begin
        BRM._brm_nlme_write!(t.plan.point, t.plan.indices[i], values)
        BRM._brm_nlme_restore!(t.plan.point, m.unit, t.plan.indices[i])
    end))
    report("  per-unit gradient buffers", cost(() -> (zeros(D), zeros(D), zeros(length(z)), zeros(length(z)))))
    report("  RK partition (footprint)", cost(() -> BRM._brm_nlme_partition_evaluate!(gradient, t.plan, i)))
    # Problem 2: the whole prior, one evaluation per step.
    report("global prior term", cost(() -> BRM._brm_svi_global_term(t, g)))
    # Problem 3: the dense global update.
    family = MeanFieldGlobal()
    q = BRM._brm_svi_params(t, g, family; global_scale=0.1, unit_scale=1.0)
    x = BRM._brm_svi_global_vector(q)
    opt = BRM._BRMSVIAdam(length(x), 0.0)
    rng = Xoshiro(2)
    ε, η = randn(rng, D), zeros(0)
    report("global noise", cost(() -> randn(rng, D)))
    report("global gradient", cost(() -> BRM._brm_svi_global_gradient(q, g, ε, η)))
    grad = BRM._brm_svi_global_gradient(q, g, ε, η)
    report("global Adam and write-back", cost(() -> begin
        y = BRM._brm_svi_global_vector(q)
        BRM._brm_svi_step!(opt, y, grad)
        BRM._brm_svi_set_global!(q, y)
    end))
    @printf("BASELINE variational state %.1f MB (global factor and units), Adam %.1f MB\n",
        Base.summarysize(q) / 2^20, Base.summarysize(opt) / 2^20)
    # One whole iteration, at a batch of `batch` units, and the public entry.
    batch_units = collect(1:min(batch, length(t.units)))
    adam = BRM._BRMSVIAdam(length(x), 1e-6)
    report("whole iteration (batch=$batch)", cost(() -> BRM._brm_svi_iteration!(rng, t, q, adam, batch_units, 1e-6, 0.0), 3))
    fit = brm_svi_fit(m; batch_size=batch, max_iterations=3, ndraws=2, separator=:none, rng=Xoshiro(3))
    println("BASELINE public_fit iterations=", fit.diagnostics.iterations, " elbo_finite=", all(isfinite, fit.diagnostics.elbo),
        " draws=", size(fit.draws), " seconds=", round(fit.diagnostics.seconds; digits=2))
    @printf("BASELINE peak_rss %.2f GB\n", peak_gb())
end

BLAS.set_num_threads(1)
for group in Iterators.partition(parse.(Int, ARGS), 5)
    main(group...)
end
