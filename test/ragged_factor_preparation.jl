# A wide predictor's ragged partition is part of the factor graph's data-only
# prefix. Its helper must retain concrete axis types through vararg forwarding.
# RUN: julia --project=test test/ragged_factor_preparation.jl
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Logging
const BRM = BayesianRegressionModels

const WIDE_RAGGED_COLUMNS = [Symbol("x", i) for i in 1:60]
const WIDE_RAGGED_DATA = merge(
    (; g=["a", "b", "a"], unit=["a", "b"], y=[0.3, 0.1]),
    NamedTuple{Tuple(WIDE_RAGGED_COLUMNS)}(Tuple(
        i % 3 == 0 ? ["r", "s", "r"] :
        i % 3 == 1 ? [0.1, 0.3, 0.5] : [1, 2, 3] for i in 1:60)))

@testset "wide ragged axes prepare stable observation factors" begin
    body = "@brm WIDE_RAGGED_DATA begin\neta ~ 1 + " *
        join(string.(WIDE_RAGGED_COLUMNS), " + ") * " + (1 | g)\n" *
        "rows = groups(g => unit)\n@plate for k in eachindex(unit)\n" *
        "y[k] ~ Normal(sum(eta[rows[k]]), 1)\nend\nend"
    rk = RKBRMI(Core.eval(@__MODULE__, Meta.parse(body)))
    # Fails on either the default warning policy or a strict error policy
    # when the data-prefix audit finds a dynamically dispatched helper call.
    f = @test_logs min_level=Logging.Warn BRM._rk_spl_factor_queries(rk, :y)
    u = fill(0.1, length(BRM._rk_layout_coordinate_names(rk)))
    expected = BRM._rk_pointwise_values(BRM._rk_pointwise_query(rk), u).y
    for i in eachindex(expected)
        sums = [sum(Base.invokelatest(q.value, u, i, t)
            for t in eachindex(BRM._rk_spl_factor_rows(f, j, i)))
            for (j, q) in enumerate(f.terms)]
        @test Base.invokelatest(f.outer.value, u, sums, i) ≈ expected[i]
    end
end
