# test/plate_matrix_data.jl — per-cell matrix data in a `@plate for` cell.
#
# A pre-grouped `Vector{Matrix}` data value is read `X[i]` in a cell, the i-th
# matrix, exactly as a `Vector{Vector}` value is read as the i-th vector.
# RKBRMI evaluates the cell as written. SBBRMI does not ingest it yet:
# StanBlocks has no data method minting its `RaggedMatrix` carrier, and the
# plate's slice of such an argument is missing too (StanBlocks snag
# `ragged-matrix-da-163255e9`). The flattened twin with Julia's `reshape` is
# the same model on both backends and pins that translation.
#
# Run: julia --startup-file=no --project=test test/plate_matrix_data.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra

const MATRIX_PANEL = (;
    X=[[0.4 -1.2; 1.5 0.3], [-0.7 0.9; 0.2 -0.4; 1.1 0.6]],
    y=[[0.8, -0.5], [0.3, 1.4, -0.9]],
)
const FLAT_PANEL = (;
    Xf=[vec(x) for x in MATRIX_PANEL.X],
    n=[size(x, 1) for x in MATRIX_PANEL.X],
    y=MATRIX_PANEL.y,
)

matrix_cells(data) = @brm data begin
    b1 ~ Normal(0.0, 1.0)
    b2 ~ Normal(0.0, 1.0)
    @plate for i in eachindex(y)
        y[i] ~ Normal(X[i] * [b1, b2], 1.0)
    end
end

reshaped_cells(data) = @brm data begin
    b1 ~ Normal(0.0, 1.0)
    b2 ~ Normal(0.0, 1.0)
    @plate for i in eachindex(y)
        y[i] ~ Normal(reshape(Xf[i], n[i], 2) * [b1, b2], 1.0)
    end
end

matrix_oracle(u) = logpdf(Normal(0, 1), u[1]) + logpdf(Normal(0, 1), u[2]) +
    sum(logpdf(MvNormal(MATRIX_PANEL.X[i] * u, I), MATRIX_PANEL.y[i])
        for i in eachindex(MATRIX_PANEL.y))

const MATRIX_POINTS = ([0.0, 0.0], [0.3, -0.7], [-1.1, 0.4])

@stestset "RKBRMI reads X[i] as the i-th matrix" begin
    backend, problem = consumer_problem(matrix_cells(MATRIX_PANEL))
    @test coordinate_names(backend.model.layout) == [:b1, :b2]
    for u in MATRIX_POINTS
        check_consumer_point(problem, u, matrix_oracle)
    end
end

@stestset "the reshape twin is the same model on both backends" begin
    _, matrix_problem = consumer_problem(matrix_cells(MATRIX_PANEL))
    _, reshaped_problem = consumer_problem(reshaped_cells(FLAT_PANEL))
    stan = consumer_stan(reshaped_cells(FLAT_PANEL), "plate_matrix_data_reshape")
    @test BridgeStan.param_unc_names(stan.model) == ["b1", "b2"]
    for u in MATRIX_POINTS
        value, gradient = LogDensityProblems.logdensity_and_gradient(reshaped_problem, u)
        @test value ≈ matrix_oracle(u) atol=2e-11 rtol=2e-11
        matrix_value, matrix_gradient = LogDensityProblems.logdensity_and_gradient(matrix_problem, u)
        @test value ≈ matrix_value atol=2e-12 rtol=2e-12
        @test gradient ≈ matrix_gradient atol=2e-12 rtol=2e-12
        stan_value, stan_gradient = BridgeStan.log_density_gradient(
            stan.model, u; propto=false, jacobian=true)
        @test stan_value ≈ value atol=2e-11 rtol=2e-11
        @test stan_gradient ≈ gradient atol=2e-10 rtol=2e-10
    end
end

@stestset "SBBRMI per-cell matrix data" begin
    # Not built yet: StanBlocks cannot ingest a `Vector{Matrix}` data value
    # (snag `ragged-matrix-da-163255e9`); emission fails with
    # "Do not know how to handle `stan_type(X, …)`".
    @test_broken (BRM.stan_code(SBBRMI(matrix_cells(MATRIX_PANEL); mod=@__MODULE__)); true)
end
