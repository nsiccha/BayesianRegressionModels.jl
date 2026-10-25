# test/plate_matrix_data.jl — per-cell matrix data in a `@plate for` cell.
#
# A pre-grouped `Vector{Matrix}` data value is read `X[i]` in a cell, the i-th
# matrix, exactly as a `Vector{Vector}` value is read as the i-th vector.
# RKBRMI evaluates the cell as written. SBBRMI passes it to StanBlocks, which
# ingests it as one `RaggedMatrix` data input and slices the i-th matrix per
# cell (StanBlocks c9504e99, snag `ragged-matrix-da-163255e9`). Both backends
# match an independent oracle, and the flattened twin with Julia's `reshape` is
# the same model on both backends.
#
# A cell may also bind a per-cell matrix to a local, here a covariance whose
# size follows its cell (StanBlocks a57951eb, snag `binding-a-per-ce-e16efa8d`):
# SBBRMI gives the same density as the inline spelling, and the descriptor
# addresses the local by its authored name, with or without an empty cell.
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

@stestset "SBBRMI reads X[i] as the i-th matrix" begin
    brmi = matrix_cells(MATRIX_PANEL)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    @test StanBlocks.stan.compiles(sb.model)
    # The whole vector of matrices is one data input the caller supplies.
    d = brm_descriptor(sb)
    x = only(input for input in d.inputs if input.name === :X)
    @test !x.derived && !x.inlined
    @test :X in required_brm_inputs(d)
    _, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "plate_matrix_data")
    @test BridgeStan.param_unc_names(stan.model) == ["b1", "b2"]
    for u in MATRIX_POINTS
        stan_value, stan_gradient = BridgeStan.log_density_gradient(
            stan.model, u; propto=false, jacobian=true)
        @test stan_value ≈ matrix_oracle(u) atol=2e-11 rtol=2e-11
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test stan_value ≈ value atol=2e-11 rtol=2e-11
        @test stan_gradient ≈ gradient atol=2e-10 rtol=2e-10
    end
end

const COVARIANCE_PANEL = (;
    x=[[0.4, -1.2], [-0.7, 0.9, 0.2]],
    y=[[0.8, -0.5], [0.3, 1.4, -0.9]],
)
# The same cells plus an empty one. Stan's `multi_normal` refuses a 0 x 0
# covariance when the density is evaluated, so this panel only checks the
# descriptor's segments.
const EMPTY_CELL_PANEL = (;
    x=[COVARIANCE_PANEL.x; [Float64[]]],
    y=[COVARIANCE_PANEL.y; [Float64[]]],
)

inline_covariance(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    tau ~ Exponential(1.0)
    @plate for i in eachindex(y)
        y[i] ~ multi_normal(b .* x[i],
            tau^2 .* (x[i] * x[i]') .+ diag_matrix(rep_vector(1.0, length(x[i]))))
    end
end

local_covariance(data) = @brm data begin
    b ~ Normal(0.0, 1.0)
    tau ~ Exponential(1.0)
    @plate for i in eachindex(y)
        ccov = tau^2 .* (x[i] * x[i]') .+ diag_matrix(rep_vector(1.0, length(x[i])))
        y[i] ~ multi_normal(b .* x[i], ccov)
    end
end

# Unconstrained (b, log tau), with the log Jacobian of tau = exp(u[2]).
function covariance_oracle(u)
    b, tau = u[1], exp(u[2])
    logpdf(Normal(0, 1), b) + logpdf(Exponential(1), tau) + u[2] +
        sum(logpdf(MvNormal(b .* x, tau^2 .* (x * x') + I), y)
            for (x, y) in zip(COVARIANCE_PANEL.x, COVARIANCE_PANEL.y))
end

@stestset "SBBRMI: a cell binds a per-cell covariance matrix to a local" begin
    # Column-major cells of 2 x 2 and 3 x 3 values, then an empty 0 x 0 one.
    for (data, segments) in ((COVARIANCE_PANEL, [4, 13]), (EMPTY_CELL_PANEL, [4, 13, 13]))
        sb = SBBRMI(local_covariance(data); mod=@__MODULE__)
        @test StanBlocks.stan.compiles(sb.model)
        ccov = brm_output(brm_descriptor(sb), :ccov)
        @test ccov.role === :group_block
        @test ccov.kind === :transformed_parameter
        @test ccov.segments == segments
    end
    inline = consumer_stan(inline_covariance(COVARIANCE_PANEL), "plate_matrix_inline_covariance")
    bound = consumer_stan(local_covariance(COVARIANCE_PANEL), "plate_matrix_local_covariance")
    @test BridgeStan.param_unc_names(bound.model) == ["b", "tau"]
    for u in ([0.0, 0.0], [0.3, -0.7], [-1.1, 0.4])
        value, gradient = BridgeStan.log_density_gradient(
            bound.model, u; propto=false, jacobian=true)
        inline_value, inline_gradient = BridgeStan.log_density_gradient(
            inline.model, u; propto=false, jacobian=true)
        @test value ≈ covariance_oracle(u) atol=2e-11 rtol=2e-11
        @test value ≈ inline_value atol=2e-12 rtol=2e-12
        @test gradient ≈ inline_gradient atol=2e-12 rtol=2e-12
    end
end
