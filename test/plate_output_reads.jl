# test/plate_output_reads.jl — the model reads every indexed output of an
# `@plate for` cell by its name, not only the cell's last (snag
# `model-scope-read-3649ae8a`).
#
# A cell collects its last indexed output under its own name; the others are
# cell values (SBBRMI stores `tot` as the carrier `<collected>_tot`, RKBRMI
# collects it as the graph value `tot`). A model-scope statement or another
# cell reading `tot` reads that value: the same model, with the same density
# and gradient, as the cell with `tot[s] = …` moved last. Oracles: the
# reordered spelling, through BridgeStan and through RK. Public synthetic
# models only.
#
# Run: julia --startup-file=no --project=test test/plate_output_reads.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PlateOutputReadModels
using BayesianRegressionModels, Distributions
# `tot` collected last: the spelling that always worked.
reordered(d) = @brm d begin
    sigma ~ Exponential(1.0)
    @plate for s in eachindex(land)
        x ~ LogNormal(log.(A[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        other[s] = x
        tot[s] = sum(x)
    end
    obs ~ Normal(tot[idx], sigma)
end
# `tot` before the collected `other`.
reported(d) = @brm d begin
    sigma ~ Exponential(1.0)
    @plate for s in eachindex(land)
        x ~ LogNormal(log.(A[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        tot[s] = sum(x)
        other[s] = x
    end
    obs ~ Normal(tot[idx], sigma)
end
# The reader written above the cell.
reader_first(d) = @brm d begin
    sigma ~ Exponential(1.0)
    obs ~ Normal(tot[idx], sigma)
    @plate for s in eachindex(land)
        x ~ LogNormal(log.(A[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        tot[s] = sum(x)
        other[s] = x
    end
end
# Both outputs of one cell read by the model.
both(d) = @brm d begin
    sigma ~ Exponential(1.0)
    @plate for s in eachindex(A)
        x ~ LogNormal(log.(A[s]), 0.5)
        tot[s] = sum(x)
        mx[s] = maximum(x)
    end
    land ~ Normal(mx, 1.0)
    obs ~ Normal(tot[idx], sigma)
end
# Another cell reads the output: collected, then not.
cross_collected(d) = @brm d begin
    @plate for s in eachindex(A)
        x ~ LogNormal(log.(A[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        tot[s] = sum(x)
    end
    @plate for s in eachindex(y2)
        y2[s] ~ Normal(0.1 * tot[s], 1.0)
    end
end
cross_side(d) = @brm d begin
    @plate for s in eachindex(A)
        x ~ LogNormal(log.(A[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        tot[s] = sum(x)
        other[s] = x
    end
    @plate for s in eachindex(y2)
        y2[s] ~ Normal(0.1 * tot[s], 1.0)
    end
end
# Per-cell reads (`sum(tot)`, `tot[2]`; snag `sum-a-vector-pla-812e04d2`) of a
# non-last output, and of the same output collected last: scalar cells, then
# vector cells.
scalar_side(d) = @brm d begin
    @plate for s in eachindex(B)
        x ~ LogNormal(log.(B[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        tot[s] = sum(x)
        other[s] = 2.0 .* x
    end
    total ~ Normal(sum(tot), 1.0)
    second ~ Normal(tot[2], 1.0)
end
scalar_last(d) = @brm d begin
    @plate for s in eachindex(B)
        x ~ LogNormal(log.(B[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        other[s] = 2.0 .* x
        tot[s] = sum(x)
    end
    total ~ Normal(sum(tot), 1.0)
    second ~ Normal(tot[2], 1.0)
end
# One cell read alone: what RKBRMI builds for scalar cells today.
cell_side(d) = @brm d begin
    @plate for s in eachindex(B)
        x ~ LogNormal(log.(B[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        tot[s] = sum(x)
        other[s] = 2.0 .* x
    end
    second ~ Normal(tot[2], 1.0)
end
cell_last(d) = @brm d begin
    @plate for s in eachindex(B)
        x ~ LogNormal(log.(B[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        other[s] = 2.0 .* x
        tot[s] = sum(x)
    end
    second ~ Normal(tot[2], 1.0)
end
vector_side(d) = @brm d begin
    @plate for s in eachindex(B)
        x ~ LogNormal(log.(B[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        tot[s] = x
        other[s] = sum(x)
    end
    totals ~ Normal(sum(tot), 1.0)
    seconds ~ Normal(tot[2], 1.0)
end
vector_last(d) = @brm d begin
    @plate for s in eachindex(B)
        x ~ LogNormal(log.(B[s]), 0.5)
        land[s] ~ Normal(sum(x), 1.0)
        other[s] = sum(x)
        tot[s] = x
    end
    totals ~ Normal(sum(tot), 1.0)
    seconds ~ Normal(tot[2], 1.0)
end
# Two cells write `tot`, and the model reads it.
two_writers(d) = @brm d begin
    sigma ~ Exponential(1.0)
    @plate for s in eachindex(land)
        x ~ LogNormal(log.(A[s]), 0.5)
        tot[s] = sum(x)
        a[s] = x
    end
    @plate for s in eachindex(land)
        tot[s] = 2.0
        b[s] = 1.0
        land[s] ~ Normal(b[s], 1.0)
    end
    obs ~ Normal(tot[idx], sigma)
end
end

const M = PlateOutputReadModels
const DATA = (; land = [10.0, 20.0, 15.0], A = [[1.0, 2.0], [3.0, 4.0], [2.0, 2.5, 3.0]],
    idx = [1, 2, 2, 3], obs = [3.0, 7.0, 6.5, 7.2], y2 = [1.0, 2.0, 3.0])
const CELLS = (; land = [3.0, 7.0, 4.5], B = [[1.0, 2.0], [3.0, 4.0], [2.0, 2.5]],
    total = 14.0, second = 7.0, totals = [6.0, 8.5], seconds = [3.0, 4.0])

# Equal log density and gradient at random unconstrained points of the same
# dimension (the coordinates correspond one to one: only the cell's name,
# hence the latents' prefix, differs).
function same_density(p, q; points=3)
    n = LogDensityProblems.dimension(p)
    @test LogDensityProblems.dimension(q) == n
    for _ in 1:points
        theta = 0.3 .* randn(n)
        lp, gp = LogDensityProblems.logdensity_and_gradient(p, theta)
        lq, gq = LogDensityProblems.logdensity_and_gradient(q, theta)
        @test isfinite(lp)
        @test lq ≈ lp rtol=1e-12
        @test gq ≈ gp rtol=1e-10
    end
end

rk_problem(brmi) = last(consumer_problem(brmi))

@stestset "SBBRMI: a model-scope read of a non-last output reads its carrier" begin
    sb = SBBRMI(M.reported(DATA); mod=@__MODULE__, total_groups=())
    code = stan_code(sb)
    @test occursin("obs ~ normal(other_tot[idx], sigma);", code)
    # No second carrier: `tot` itself is never declared.
    @test !occursin(r"\] tot\b", code) && !occursin(r"\btot =", code)
    reference = consumer_stan(M.reordered(DATA), "plate_output_reads_reordered")
    same_density(reference, consumer_stan(M.reported(DATA), "plate_output_reads_reported"))
    same_density(reference, consumer_stan(M.reader_first(DATA), "plate_output_reads_first"))
    d = brm_descriptor(sb; name=:reported)
    o = brm_output(d, :tot)
    @test o.name === :other_tot
    @test o.kind === :transformed_parameter
    @test !isempty(sprint(show, MIME("text/plain"), brm_description(d)))
end

@stestset "RKBRMI: a model-scope read of a non-last output reads its graph value" begin
    reference = rk_problem(M.reordered(DATA))
    same_density(reference, rk_problem(M.reported(DATA)))
    same_density(reference, rk_problem(M.reader_first(DATA)))
end

@stestset "both outputs of one cell read by the model" begin
    code = stan_code(SBBRMI(M.both(DATA); mod=@__MODULE__, total_groups=()))
    @test occursin("land ~ normal(mx, 1.0);", code)
    @test occursin("obs ~ normal(mx_tot[idx], sigma);", code)
    sb = consumer_stan(M.both(DATA), "plate_output_reads_both")
    rk = rk_problem(M.both(DATA))
    @test LogDensityProblems.dimension(rk) == LogDensityProblems.dimension(sb)
    theta = 0.2 .* randn(LogDensityProblems.dimension(sb))
    @test isfinite(LogDensityProblems.logdensity(sb, theta))
    @test isfinite(LogDensityProblems.logdensity(rk, theta))
end

@stestset "another cell reads a non-last output like a collected one" begin
    code = stan_code(SBBRMI(M.cross_side(DATA); mod=@__MODULE__, total_groups=()))
    @test occursin("other_tot[plate_i__pl_2]", code)
    same_density(consumer_stan(M.cross_collected(DATA), "plate_output_reads_cross_collected"),
        consumer_stan(M.cross_side(DATA), "plate_output_reads_cross_side"))
    same_density(rk_problem(M.cross_collected(DATA)), rk_problem(M.cross_side(DATA)))
end

@stestset "per-cell reads of a non-last output (sum, one cell)" begin
    # Scalar cells: both backends.
    code = stan_code(SBBRMI(M.scalar_side(CELLS); mod=@__MODULE__, total_groups=()))
    @test occursin("sum(other_tot)", code) && occursin("other_tot[2]", code)
    same_density(consumer_stan(M.scalar_last(CELLS), "plate_output_reads_scalar_last"),
        consumer_stan(M.scalar_side(CELLS), "plate_output_reads_scalar_side"))
    same_density(rk_problem(M.cell_last(CELLS)), rk_problem(M.cell_side(CELLS)))
    same_density(rk_problem(M.scalar_last(CELLS)), rk_problem(M.scalar_side(CELLS)))
    # Vector cells: both backends (RKBRMI reads them per cell from BRM 0.2.14).
    code = stan_code(SBBRMI(M.vector_side(CELLS); mod=@__MODULE__, total_groups=()))
    @test occursin("other_tot", code) && !occursin(r"\btot =", code)
    same_density(consumer_stan(M.vector_last(CELLS), "plate_output_reads_vector_last"),
        consumer_stan(M.vector_side(CELLS), "plate_output_reads_vector_side"))
    # RKBRMI: equal log densities. Its Enzyme gradient of these reads fails for
    # a collected output too (todo `1svm4l1`).
    last_rk, side_rk = rk_problem(M.vector_last(CELLS)), rk_problem(M.vector_side(CELLS))
    @test LogDensityProblems.dimension(side_rk) == LogDensityProblems.dimension(last_rk)
    for _ in 1:3
        theta = 0.3 .* randn(LogDensityProblems.dimension(last_rk))
        @test LogDensityProblems.logdensity(side_rk, theta) ≈
            LogDensityProblems.logdensity(last_rk, theta) rtol=1e-12
    end
    @test_broken try
        LogDensityProblems.logdensity_and_gradient(side_rk,
            zeros(LogDensityProblems.dimension(side_rk)))
        true
    catch
        false
    end
end

@stestset "omitted response and replay" begin
    omitted = (; land = DATA.land, A = DATA.A, idx = DATA.idx)
    @test occursin("other_tot[idx]",
        stan_code(SBBRMI(M.reported(omitted); mod=@__MODULE__, total_groups=())))
    prior = consumer_stan(M.reported(omitted), "plate_output_reads_omitted")
    @test isfinite(LogDensityProblems.logdensity(prior,
        zeros(LogDensityProblems.dimension(prior))))
    fit = SBBRMI(M.reported(DATA); mod=@__MODULE__, total_groups=())
    more = (; land = [10.0, 20.0, 15.0, 12.0], A = [[1.0, 2.0], [3.0, 4.0], [2.0, 2.5, 3.0], [1.5]],
        idx = [1, 4], obs = [3.0, 1.4], y2 = DATA.y2)
    replayed = reprocess(fit, more)
    @test stan_code(replayed) == stan_code(fit)
end

@stestset "the model reading an output two cells write is refused" begin
    # refused: `tot` names two cells' values, and the model cannot tell which
    # one it reads (an ambiguous name, as for the SBBRMI descriptor).
    for backend in (b -> SBBRMI(b; mod=@__MODULE__), RKBRMI)
        err = try backend(M.two_writers(DATA)); nothing catch e; e end
        @test err !== nothing
        @test occursin("several `@plate for` cells", sprint(showerror, err))
    end
end
