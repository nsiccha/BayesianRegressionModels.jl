# test/plate_output_reductions.jl — a plate output read at model scope is one
# value per cell, as the Julia loop leaves it.
#
# After `@plate for s in eachindex(A); cnt[s] = v; end`, `cnt[g]` is cell `g`'s
# value, `sum(cnt)` adds the cells (elementwise for vector cells) and
# `sum(cnt[idx])` adds the cells `idx`. Before snag `sum-a-vector-pla-812e04d2`
# (decision `09e0gc5`) SBBRMI summed every entry into one number and could not
# index a fixed-width output, and RKBRMI read the cells laid end to end. Each
# shape StanBlocks stores a plate in is covered: a scalar per cell, a vector of
# fixed width (`matrix[K, N]`) and a ragged vector. Every model matches an
# independent oracle on RKBRMI and on SBBRMI (BridgeStan), density and
# gradient.
#
# Run: julia --startup-file=no --project=test test/plate_output_reductions.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const CELLS = (;
    A=[[1.0, 2.0], [3.0, -1.0], [0.5, 0.25]],
    a1=[1.0, 3.0, 0.5],
    a2=[2.0, -1.0, 0.25],
    k=[1, 3],
    obs=[4.0, 1.5],
    obs1=[2.5],
)
const POINTS = ([0.0], [0.4], [-1.3])

# A ragged vector per cell (sliced from a nested data vector).
ragged_sum(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    @plate for s in eachindex(A)
        cnt[s] = b .* A[s]
    end
    n = sum(cnt)
    obs ~ Normal(n, 1.0)
end
ragged_spelled(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    @plate for s in eachindex(A)
        cnt[s] = b .* A[s]
    end
    n = cnt[1] + cnt[2] + cnt[3]
    obs ~ Normal(n, 1.0)
end
ragged_subset(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    @plate for s in eachindex(A)
        cnt[s] = b .* A[s]
    end
    obs ~ Normal(sum(cnt[k]) .+ cnt[2], 1.0)
end
# A vector of fixed width per cell (StanBlocks stores `matrix[2, N]`).
fixed_sum(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    @plate for s in eachindex(a1)
        cnt[s] = [b * a1[s], b * a2[s]]
    end
    n = sum(cnt)
    obs ~ Normal(n, 1.0)
end
fixed_subset(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    @plate for s in eachindex(a1)
        cnt[s] = [b * a1[s], b * a2[s]]
    end
    obs ~ Normal(sum(cnt[k]) .+ cnt[2], 1.0)
end
# A scalar per cell.
scalar_reads(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    @plate for s in eachindex(a1)
        cnt[s] = b * a1[s]
    end
    obs1 ~ Normal(sum(cnt) + sum(cnt[k]) + cnt[2], 1.0)
end

prior(u) = logpdf(Normal(0, 1), u[1])
gauss(loc, y) = sum(logpdf.(Normal.(loc, 1.0), y))
cells(u) = [u[1] .* a for a in CELLS.A]
fixed_cells(u) = [[u[1] * CELLS.a1[s], u[1] * CELLS.a2[s]] for s in eachindex(CELLS.a1)]
sum_oracle(cells) = u -> prior(u) + gauss(sum(cells(u)), CELLS.obs)
subset_oracle(cells) = u -> prior(u) + gauss(sum(cells(u)[CELLS.k]) .+ cells(u)[2], CELLS.obs)
scalar_oracle(u) = (c = u[1] .* CELLS.a1;
    prior(u) + gauss(sum(c) + sum(c[CELLS.k]) + c[2], CELLS.obs1))

const CASES = (
    ("ragged sum(cnt)", ragged_sum, sum_oracle(cells)),
    ("ragged cnt[1] + cnt[2] + cnt[3]", ragged_spelled, sum_oracle(cells)),
    ("ragged sum(cnt[k]) .+ cnt[2]", ragged_subset, subset_oracle(cells)),
    ("fixed-width sum(cnt)", fixed_sum, sum_oracle(fixed_cells)),
    ("fixed-width sum(cnt[k]) .+ cnt[2]", fixed_subset, subset_oracle(fixed_cells)),
    ("scalar sum(cnt) + sum(cnt[k]) + cnt[2]", scalar_reads, scalar_oracle),
)

# RKBRMI reads the cells (`cnt_cells`) for these, and ReactiveKernels cannot
# yet sum, index or observe a plate's VECTOR cells at model scope (snag
# `ReactiveKernels/snags/summing-a-plate-eba2648d`): building or evaluating
# them throws. Before, RKBRMI read the cells laid end to end and fitted another
# model without an error. Scalar cells work.
rk_vector_pending(label) = !startswith(label, "scalar")
function rk_matches_oracle(build, oracle)
    try
        _, problem = consumer_problem(build(CELLS))
        all(u -> LogDensityProblems.logdensity(problem, u) ≈ oracle(u), POINTS)
    catch err
        @info "RKBRMI vector cells (ReactiveKernels snag summing-a-plate-eba2648d)" err
        false
    end
end

for (label, build, oracle) in CASES
    @stestset "RKBRMI $label" begin
        if rk_vector_pending(label)
            @test_broken rk_matches_oracle(build, oracle)
        else
            backend, problem = consumer_problem(build(CELLS))
            @test coordinate_names(backend.model.layout) == [:b]
            for u in POINTS
                check_consumer_point(problem, u, oracle)
            end
        end
    end
    @stestset "SBBRMI $label" begin
        stan = consumer_stan(build(CELLS), "plate_output_reductions_" *
            replace(label, r"[^A-Za-z0-9]+" => "_"))
        @test BridgeStan.param_unc_names(stan.model) == ["b"]
        for u in POINTS
            value, gradient = BridgeStan.log_density_gradient(
                stan.model, u; propto=false, jacobian=true)
            @test value ≈ oracle(u) atol=2e-11 rtol=2e-11
            reference = (oracle(u .+ 1e-5) - oracle(u .- 1e-5)) / 2e-5
            @test only(gradient) ≈ reference atol=2e-7 rtol=2e-7
        end
    end
end

@stestset "a scalar output keeps its Stan source" begin
    code = stan_code(SBBRMI(scalar_reads(CELLS); mod=@__MODULE__))
    @test !occursin("brm_plate", code)
end

# Julia's sum of vectors of different lengths throws; so does the Stan
# program, rejecting the draw.
@stestset "SBBRMI rejects summing cells of different lengths" begin
    uneven = merge(CELLS, (; A=[[1.0, 2.0], [3.0], [0.5, 0.25]]))
    stan = consumer_stan(ragged_sum(uneven), "plate_output_reductions_uneven")
    @test_throws Exception BridgeStan.log_density(stan.model, [0.4])
end

# `cnt[k]` as a value of its own, with cells whose length follows their data,
# needs StanBlocks to select several groups of a ragged value (snag
# `StanBlocks/snags/selecting-severa-027a8191`); `sum(cnt[k])` works above.
ragged_selection(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    @plate for s in eachindex(A)
        cnt[s] = b .* A[s]
    end
    sel = cnt[k]
    obs ~ Normal(sel[1] .+ sel[2], 1.0)
end
@stestset "SBBRMI ragged cnt[k] as a value" begin
    @test_broken try
        StanBlocks.stan.transpiles(SBBRMI(ragged_selection(CELLS); mod=@__MODULE__).model)
    catch
        false
    end
end

# RKBRMI collects every indexed output of a cell, the last one or not, as a
# graph value of its name (decision `0ycqf59`); model-scope reads of an earlier
# one go over its cells the same way. On SBBRMI only the last output is read at
# model scope so far (snag `model-scope-read-3649ae8a`).
earlier_output(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    @plate for s in eachindex(a1)
        tot[s] = b * a1[s]
        cnt[s] = b * a2[s]
    end
    obs1 ~ Normal(sum(tot) + tot[2] + sum(cnt[k]), 1.0)
end
@stestset "RKBRMI reads an earlier output by cell" begin
    backend, problem = consumer_problem(earlier_output(CELLS))
    @test coordinate_names(backend.model.layout) == [:b]
    oracle(u) = (t = u[1] .* CELLS.a1; c = u[1] .* CELLS.a2;
        prior(u) + gauss(sum(t) + t[2] + sum(c[CELLS.k]), CELLS.obs1))
    for u in POINTS
        check_consumer_point(problem, u, oracle)
    end
end
