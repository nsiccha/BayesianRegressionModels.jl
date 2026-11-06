# Measured effect of connected SVI's `cell_units` on the toy in `toy.jl`: unit
# layout and sampling law, per-unit gradient time against rows and reads, exact
# subsampling variance of the global gradient, warm time per update, and the
# ELBO each layout reaches. Both layouts run in one process, so they share
# compiled code. The ELBO is one estimator for both: E_q[log p] + H(q), with
# log p the full density at draws from q. The trace `brm_svi_fit` reports is not
# comparable across layouts because its minibatch noise differs.
#
# RUN: julia --project=test research/svi_cell_units/measure.jl \
#          <nsite> <nsensor> <batch> <updates> [window]
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
    logp = mean(LogDensityProblems.logdensity(full, view(points, :, k)) for k in 1:draws)
    logp + BRM._brm_svi_entropy(q)
end

function fit_trace(m, cell_units, globals, full; batch_size, updates, window, seed=1)
    rows = NamedTuple[]
    excluded = Ref(0.0)
    rng = Xoshiro(seed)
    target_ref = Ref{Any}(nothing)
    t0 = Ref(0.0)
    save = state -> begin
        before = time()
        target_ref[] === nothing && return nothing
        run = state.run
        elbo = common_elbo(Xoshiro(99), target_ref[], run.q, full)
        push!(rows, (; iteration=run.iteration, seconds=before - t0[] - excluded[], elbo))
        excluded[] += time() - before
        nothing
    end
    target_ref[] = BRM._brm_svi_target(m; separator=:none, global_coordinates=globals, cell_units)
    t0[] = time()
    fit = brm_svi_fit(save, m; rng, batch_size, max_iterations=updates, ndraws=4,
        separator=:none, global_coordinates=globals, cell_units, threads=1,
        checkpoint_every=window, window=window, unit_sampling=:work)
    (; fit, rows)
end

function main(nsite, nsensor, batch, updates, window)
    rng = Xoshiro(20261106)
    data = cell_toy_data(rng; nsite, nsensor)
    m = brm_nlme_model(RKBRMI(cell_toy_model(data)); ad_backend=AD,
        subject_group=:site, unit_partition=:connected)
    globals = cell_toy_globals(m)
    full = rk_logdensity_problem(m.backend; ad_backend=AD)
    println("toy nsite=$nsite nsensor=$nsensor rows=$(length(data.row_y)) site_rows=",
        extrema(length.(data.site_y)), " coordinates=", length(m.unit), " promoted=", length(globals))
    targets = Dict(layout => BRM._brm_svi_target(m; separator=:none,
        global_coordinates=globals, cell_units=layout) for layout in LAYOUTS)
    probabilities = Dict(layout => report_layout(string(layout), targets[layout]) for layout in LAYOUTS)

    # Exactness: each layout's unit terms sum to the full density and gradient.
    for layout in LAYOUTS
        t = targets[layout]
        for offset in (-0.31, 0.19)
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
    end

    # Per-unit gradient time, then the variance and expected cost it implies.
    points = Dict(:start => copy(m.unit))
    terms = Dict(layout => unit_gradients(targets[layout], points[:start]) for layout in LAYOUTS)
    for layout in LAYOUTS
        p, t = probabilities[layout], targets[layout]
        V, total = subsampling_variance(terms[layout], p, batch)
        E = empirical_variance(Xoshiro(3), terms[layout], t === nothing ? nothing :
            BRM._brm_svi_unit_sampler(:work, t.ownership.work), batch, total)
        update = expected_update_seconds(terms[layout], p, batch)
        @printf("start  %-11s exact variance %.4g  sampled %.4g  relative to ‖g‖² %.3f  expected update %.3f ms  variance×time %.4g\n",
            layout, V, E, V / sum(abs2, total), 1000update, V * update)
    end
    println("per-unit gradient time (ms) by unit, rows and reads:")
    t = targets[:components]
    scientific = length(m.view.levels)
    for i in eachindex(t.units)
        i <= scientific || i <= scientific + 3 || continue
        rows = i <= scientific ? length(data.site_y[i]) : count(==(i), Dict(t.ownership.attribution)[:row_y])
        @printf("  unit %3d rows %4d work %8.0f time %8.3f ms\n", i, rows, t.ownership.work[i],
            1000terms[:components][i].seconds)
    end

    # Warm optimization under each layout: compile first, then measure.
    for layout in LAYOUTS
        fit_trace(m, layout, globals, full; batch_size=batch, updates=window, window)
        run = fit_trace(m, layout, globals, full; batch_size=batch, updates, window)
        d = run.fit.diagnostics
        @printf("fit %-11s updates %d  optimizer seconds %.2f  per update %.2f ms  gradient calls %d\n",
            layout, d.iterations, d.seconds, 1000d.seconds / d.iterations, d.unit_gradient_calls)
        println("  iteration seconds common-ELBO")
        for row in run.rows
            @printf("  %6d %8.2f %12.2f\n", row.iteration, row.seconds, row.elbo)
        end
        V, total = subsampling_variance(unit_gradients(targets[layout],
            let u = copy(m.unit), q = BRM._brm_svi_result_q(d.optimizer.run)
                u[targets[layout].globals] .= q.μ
                for (i, block) in enumerate(targets[layout].units)
                    u[block] .= q.M[i]
                end
                u
            end; repeats=1), probabilities[layout], batch)
        @printf("  at the fit's means: exact variance %.4g  relative to ‖g‖² %.3f\n",
            V, V / max(sum(abs2, total), eps()))
        println("  visits ", summarize(d.sampling.visits), "  draws ", summarize(d.sampling.draws))
    end
end

args = parse.(Int, ARGS)
main(args[1], args[2], args[3], args[4], length(args) >= 5 ? args[5] : 100)
