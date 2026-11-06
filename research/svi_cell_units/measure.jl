# Measured effect of connected SVI's `cell_units` on the toy in `toy.jl`: unit
# layout and sampling law, per-unit gradient time against rows and reads, exact
# subsampling variance of the global gradient, warm time per update, and the
# ELBO each layout reaches. Both layouts run in one process, so they share
# compiled code. The ELBO is one estimator for both: E_q[log p] + H(q), with
# log p the full density at draws from q. The trace `brm_svi_fit` reports is not
# comparable across layouts because its minibatch noise differs.
#
# RUN: julia --project=test research/svi_cell_units/measure.jl \
#          <nsite> <nsensor> <batch> <updates> <long updates> <window> <first seed> <last seed>
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
using LinearAlgebra, LogDensityProblems, Printf, Random, Statistics
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "toy.jl"))

const AD = AutoEnzyme(; mode=Enzyme.Reverse)
const LAYOUTS = (:first_unit, :components)

summarize(x) = @sprintf("min %.3g  median %.3g  max %.3g", minimum(x), median(x), maximum(x))

# The largest unit's share and draw probability under the sampler's own law.
function report_layout(label, t)
    work = t.ownership.work
    p = BRM._brm_svi_unit_sampler(:work, work).probabilities
    dims = length.(t.units)
    biggest = argmax(work)
    @printf("%-11s units %d  globals %d  largest unit: index %d dim %d work %.0f share %.3f p %.3f\n",
        label, length(t.units), length(t.globals), biggest, dims[biggest], work[biggest],
        work[biggest] / sum(work), p[biggest])
    println("            work ", summarize(work), "  dims ", summarize(dims))
    p
end

# Value-and-gradient of every unit at packed point `u`: minimum of `repeats` warm
# calls, in seconds, and the global gradient each unit contributes.
function unit_gradients(t, u; repeats=5)
    g = u[t.globals]
    D = length(g)
    map(eachindex(t.units)) do i
        z = u[t.units[i]]
        gg, gz = zeros(D), zeros(length(z))
        BRM._brm_svi_unit_term!(gg, gz, t, i, g, z)
        seconds = minimum(1:repeats) do _
            fill!(gg, 0.0); fill!(gz, 0.0)
            @elapsed BRM._brm_svi_unit_term!(gg, gz, t, i, g, z)
        end
        (; seconds, gradient=copy(gg))
    end
end

# Exact variance of the with-replacement global gradient estimator, summed over
# coordinates: (Σ ‖gᵢ‖²/pᵢ − ‖Σ gᵢ‖²)/B. Global coordinates are the same in both
# layouts here, so the figures are comparable at a common point.
function subsampling_variance(terms, p, B)
    total = sum(term.gradient for term in terms)
    (sum(sum(abs2, term.gradient) / p[i] for (i, term) in enumerate(terms)) -
        sum(abs2, total)) / B, total
end

# The same variance estimated from sampled batches, as a check on the formula.
function empirical_variance(rng, terms, sampler, B, total; draws=4000)
    mean(1:draws) do _
        batch = BRM._brm_svi_batch(rng, sampler, B)
        estimate = zero(total)
        for (k, i) in enumerate(batch.indices)
            estimate .+= batch.weights[k] .* terms[i].gradient
        end
        sum(abs2, estimate .- total)
    end
end

# Expected time of one update: each distinct unit in a batch is evaluated twice.
expected_update_seconds(terms, p, B) =
    sum(2 * term.seconds * (1 - (1 - p[i])^B) for (i, term) in enumerate(terms))

function common_elbo(rng, t, q, full; draws=128)
    points = BRM._brm_svi_draws(rng, t, q, draws)
    logp = mean(LogDensityProblems.logdensity(full, points[:, k]) for k in 1:draws)
    logp + BRM._brm_svi_entropy(q)
end

# One optimization, with the common ELBO at every window. Evaluating it is left
# out of `seconds`, which therefore counts only the optimizer's own time.
function fit_trace(m, t, layout, globals, full; batch_size, updates, window, seed)
    rows = NamedTuple[]
    excluded, t0 = Ref(0.0), Ref(0.0)
    save = state -> begin
        before = time()
        run = state.run
        elbo = common_elbo(Xoshiro(99), t, run.q, full)
        push!(rows, (; iteration=run.iteration, seconds=before - t0[] - excluded[], elbo))
        excluded[] += time() - before
        nothing
    end
    t0[] = time()
    fit = brm_svi_fit(save, m; rng=Xoshiro(seed), batch_size, max_iterations=updates, ndraws=4,
        separator=:none, global_coordinates=globals, cell_units=layout, threads=1,
        checkpoint_every=window, window, unit_sampling=:work)
    (; fit, rows)
end

packed_means(m, t, q) = let u = copy(m.unit)
    u[t.globals] .= q.μ
    for (i, block) in enumerate(t.units)
        u[block] .= q.M[i]
    end
    u
end

function report_variance(label, u, targets, probabilities, batch)
    for layout in LAYOUTS
        terms = unit_gradients(targets[layout], u)
        p = probabilities[layout]
        V, total = subsampling_variance(terms, p, batch)
        sampler = BRM._brm_svi_unit_sampler(:work, targets[layout].ownership.work)
        E = empirical_variance(Xoshiro(3), terms, sampler, batch, total)
        update = expected_update_seconds(terms, p, batch)
        @printf("variance %-8s %-11s exact %.5g  sampled %.5g  ‖g‖² %.5g  expected update %.3f ms  variance×time %.5g\n",
            label, layout, V, E, sum(abs2, total), 1000update, V * update)
    end
end

# Rows, rows beyond a cell's first, and reads, per unit. Units are in level order,
# which is not data order, so everything goes through the attribution.
function unit_structure(m, t)
    inventory = BRM._brm_svi_factor_inventory(m.view, length(m.unit);
        rows=BRM._brm_svi_observed_rows(m.backend, m.view))
    owners = Dict(t.ownership.attribution)
    N = length(t.units)
    rows, extra, reads = zeros(N), zeros(N), zeros(N)
    for ((name, cell), support, r) in zip(inventory.sources, inventory.supports, inventory.rows)
        unit = owners[name][cell]
        rows[unit] += r
        extra[unit] += max(r - 1, 0)
        reads[unit] += length(support)
    end
    (; rows, extra, reads)
end

# Per-unit gradient time against rows and reads. This toy's vector-valued rows are
# one BLAS product, far cheaper per row than a row that computes per-row
# quantities, so it says little about what a row costs in general.
function report_cost_model(m, t)
    terms = unit_gradients(t, copy(m.unit); repeats=15)
    s = unit_structure(m, t)
    seconds = [term.seconds for term in terms]
    sites = findall(>(0), s.extra)
    sensors = setdiff(eachindex(t.units), sites)
    println("unit, rows, reads, work, measured gradient (ms):")
    for i in vcat(sites, first(sensors, 3))
        @printf("  %4d %5.0f %7.0f %9.0f %9.4f\n", i, s.rows[i], s.reads[i], t.ownership.work[i], 1000seconds[i])
    end
    per_row = cov(s.rows[sites], seconds[sites]) / var(s.rows[sites])
    per_read = median(seconds[i] / s.reads[i] for i in sensors)
    @printf("cost model: %.4f µs per site row, %.4f µs per sensor-row read\n", 1e6per_row, 1e6per_read)
    @printf("correlation of work with measured time: all units %.3f, sites %.3f\n",
        cor(t.ownership.work, seconds), cor(t.ownership.work[sites], seconds[sites]))
end

# What the row term does to the estimator: the variance and expected time the
# sampling law gives under each weight, at the same gradients.
function report_row_weight(label, m, t, u, batch)
    terms = unit_gradients(t, u)
    s = unit_structure(m, t)
    for weight in (0, 1, 4, 16, 64, 256)
        work = t.ownership.work .+ (weight - BRM._BRM_SVI_ROW_WORK) .* s.extra
        p = BRM._brm_svi_unit_sampler(:work, work).probabilities
        V, _ = subsampling_variance(terms, p, batch)
        update = expected_update_seconds(terms, p, batch)
        @printf("row weight %-7s %4d  exact variance %.5g  expected update %.3f ms  variance×time %.5g  largest p %.3f\n",
            label, weight, V, 1000update, V * update, maximum(p))
    end
end

function main(nsite, nsensor, batch, updates, long_updates, window, seeds)
    rng = Xoshiro(20261106)
    data = cell_toy_data(rng; nsite, nsensor)
    m = brm_nlme_model(RKBRMI(cell_toy_model(data)); ad_backend=AD,
        subject_group=:site, unit_partition=:connected)
    globals = cell_toy_globals(m)
    full = rk_logdensity_problem(m.backend; ad_backend=AD)
    println("toy nsite=$nsite nsensor=$nsensor sensor_rows=$(length(data.row_y)) site_rows=",
        extrema(length.(data.site_y)), " coordinates=", length(m.unit), " promoted=", length(globals),
        " batch=$batch updates=$updates threads=1")
    targets = Dict(layout => BRM._brm_svi_target(m; separator=:none,
        global_coordinates=globals, cell_units=layout) for layout in LAYOUTS)
    probabilities = Dict(layout => report_layout(string(layout), targets[layout]) for layout in LAYOUTS)

    # Exactness: each layout's unit terms sum to the full density and gradient.
    for layout in LAYOUTS, offset in (-0.31, 0.19)
        t = targets[layout]
        u = [offset + 0.12sin(j) for j in eachindex(m.unit)]
        value, gg = BRM._brm_svi_global_term(t, u[t.globals])
        gradient = zeros(length(u))
        gradient[t.globals] .= gg
        for (i, block) in enumerate(t.units)
            gu, gz = zeros(length(t.globals)), zeros(length(block))
            value += BRM._brm_svi_unit_term!(gu, gz, t, i, u[t.globals], u[block])
            gradient[t.globals] .+= gu
            gradient[block] .+= gz
        end
        expected, native = LogDensityProblems.logdensity_and_gradient(full, u)
        @printf("exactness %-11s offset %+.2f  value error %.2e  gradient error %.2e\n",
            layout, offset, abs(value - expected), norm(gradient - native, Inf))
    end

    report_cost_model(m, targets[:components])
    report_variance("start", copy(m.unit), targets, probabilities, batch)

    # Warm optimization under each layout: one short fit compiles, then each seed
    # is measured. The ELBO trace is the common estimator above.
    finals = Dict{Symbol,Any}()
    for layout in LAYOUTS
        t = targets[layout]
        fit_trace(m, t, layout, globals, full; batch_size=batch, updates=window, window, seed=1)
        for seed in seeds
            run = fit_trace(m, t, layout, globals, full; batch_size=batch, updates, window, seed)
            d = run.fit.diagnostics
            @printf("fit %-11s seed %d updates %d  optimizer seconds %.2f  per update %.2f ms  gradient calls %d\n",
                layout, seed, d.iterations, d.seconds, 1000d.seconds / d.iterations, d.unit_gradient_calls)
            println("  iteration seconds common-ELBO")
            for row in run.rows
                @printf("  %6d %8.2f %12.3f\n", row.iteration, row.seconds, row.elbo)
            end
            println("  visits ", summarize(d.sampling.visits), "  draws ", summarize(d.sampling.draws))
            finals[layout] = packed_means(m, t, d.q)
        end
    end
    # One point for both layouts: the same global gradient, so the figures compare.
    report_variance("fitted", finals[:components], targets, probabilities, batch)
    report_row_weight("start", m, targets[:components], copy(m.unit), batch)
    report_row_weight("fitted", m, targets[:components], finals[:components], batch)
    # Where each layout ends up, given far more updates: the factorization's cost
    # shows here, once slower convergence no longer can. Fewer windows, so the ELBO
    # costs little beside the updates.
    for layout in LAYOUTS, seed in seeds
        t = targets[layout]
        run = fit_trace(m, t, layout, globals, full; batch_size=batch, updates=long_updates,
            window=long_updates ÷ 10, seed)
        d = run.fit.diagnostics
        @printf("long %-11s seed %d updates %d  optimizer seconds %.2f  final ELBO %.3f  second half mean %.3f\n",
            layout, seed, d.iterations, d.seconds, run.rows[end].elbo,
            mean(row.elbo for row in run.rows[end-4:end]))
    end
end

args = parse.(Int, ARGS)
main(args[1], args[2], args[3], args[4], args[5], args[6], args[7]:args[8])
