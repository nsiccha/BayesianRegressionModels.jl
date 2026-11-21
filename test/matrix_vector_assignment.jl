using Test, Distributions, BayesianRegressionModels, StanBlocks
import BridgeStan, ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme

const matvec_builder = @brm begin
    b[1:K] .~ Normal.(0, 1)
    eta = X * b
    y ~ Normal(eta, 1)
end

const broadcast_builder = @brm begin
    b[1:K] .~ Normal.(0, 1)
    scaled = X .* b
    y ~ Normal(sum(scaled), 1)
end

const legacy_builder = @brm begin
    b[1:K] .~ Normal.(0, 1)
    scaled = X * b
    y ~ Normal(sum(scaled), 1)
end

@brm function assignment_linear_map(design, coef)
    value = design * coef
    return value
end

function check_product(sb, rk, b, lp, gradient, path)
    stan = stan_instantiate(sb; path).model
    names = ["b.$j" for j in eachindex(b)]
    @test BridgeStan.param_unc_names(stan) == names
    @test ReactiveKernelsPPL.coordinate_names(rk.model.layout) == Symbol.(names)
    value, grad = BridgeStan.log_density_gradient(stan, b; propto=false)
    @test value ≈ lp atol=1e-10
    @test grad ≈ gradient atol=1e-10
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, grad = LogDensityProblems.logdensity_and_gradient(problem, b)
    @test value ≈ lp atol=1e-10
    @test grad ≈ gradient atol=1e-10
end

@testset "matrix-vector assignments retain their operator" begin
    X = [1. 2.; 3. 4.; 5. 6.]
    data = (; X, K=2, y=[-1., 0.5, 2.])
    newdata = (; X=[2. -1.; 0.5 3.], K=2, y=[-0.2, 1.])
    b = [0.25, -0.4]
    oracle(residual, b) = sum(logpdf.(Normal(), residual)) + sum(logpdf.(Normal(), b))
    mktempdir() do dir
        for builder in (matvec_builder, brm(matvec_builder.source; mod=@__MODULE__))
            model = builder(data)
            @test occursin("eta = X * b", string(model))
            sb, rk = SBBRMI(model; mod=@__MODULE__), RKBRMI(model)
            @test occursin("X * b", stan_code(sb))
            @test occursin("X * b", string(BayesianRegressionModels._rk_emit_ast(rk.plan).main))
            path = joinpath(dir, "matvec.stan")
            for (s, r, rows) in ((sb, rk, data),
                    (reprocess(sb, newdata), reprocess(rk, newdata), newdata))
                residual = rows.y - rows.X * b
                check_product(s, r, b, oracle(residual, b), rows.X' * residual - b, path)
            end
        end
        # Integer data keeps its original type outside arithmetic. A matrix
        # also passes through a component's arguments with its row axis.
        integer_data = merge(data, (; X=Int.(X)))
        component = @brm data begin
            b[1:K] .~ Normal.(0, 1)
            mu ~ assignment_linear_map(X, b)
            y ~ Normal(mu, 1)
        end
        residual = data.y - X * b
        for (model, file) in ((matvec_builder(integer_data), "integer"),
                (component, "component"))
            sb, rk = SBBRMI(model; mod=@__MODULE__), RKBRMI(model)
            check_product(sb, rk, b, oracle(residual, b), X' * residual - b,
                joinpath(dir, file * ".stan"))
        end

        # The same spelling with a dot scales rows. Unequal values and a
        # non-square matrix expose accidental contraction or column scaling.
        for (matrix, v) in ((X, [0.2, -0.1, 0.6]), (X, [0.2]), (X[1:1, :], b),
                (newdata.X, b))
            rows = (; X=matrix, K=length(v), y=data.y)
            model = broadcast_builder(rows)
            @test occursin("X .* b", string(model))
            sb, rk = SBBRMI(model; mod=@__MODULE__), RKBRMI(model)
            @test occursin("X .* b", string(BayesianRegressionModels._rk_emit_ast(rk.plan).main))
            residual = rows.y .- sum(matrix .* v)
            # The mean is linear in v. Its j-th derivative is the mean at
            # the j-th unit vector, including for either singleton axis.
            weights = [sum(matrix .* (eachindex(v) .== j)) for j in eachindex(v)]
            check_product(sb, rk, v, oracle(residual, v), weights * sum(residual) - v,
                joinpath(dir, "broadcast.stan"))
            legacy = legacy_builder(rows)
            @test occursin("scaled = X * b", string(legacy))
            @test stan_code(SBBRMI(legacy; mod=@__MODULE__)) == stan_code(sb)
            @test BayesianRegressionModels._rk_emit_ast(RKBRMI(legacy).plan).main ==
                BayesianRegressionModels._rk_emit_ast(rk.plan).main
        end
    end
end

@testset "data coefficients and forward declarations" begin
    data = (; X=[1. 2.; 3. 4.; 5. 6.], v=[0.1,0.2], y=[-1.,0.5,2.], K=2)
    fixed = @brm data begin
        eta = X * v
        sigma ~ Exponential(1)
        y ~ Normal(eta, sigma)
    end
    sb, rk = SBBRMI(fixed; mod=@__MODULE__), RKBRMI(fixed)
    @test sb.data[:eta] ≈ data.X * data.v
    @test occursin("X * v", string(BayesianRegressionModels._rk_emit_ast(rk.plan).main))
    p = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    r = data.y - data.X * data.v
    lp, g = LogDensityProblems.logdensity_and_gradient(p, [0.])
    @test lp ≈ -1 + sum(logpdf.(Normal(), r))
    @test only(g) ≈ sum(abs2, r) - length(r)

    later = @brm data begin
        eta = X * b
        b[1:K] .~ Normal.(0, 1)
        y ~ Normal(eta, 1)
    end
    @test stan_code(SBBRMI(later; mod=@__MODULE__)) ==
        stan_code(SBBRMI(matvec_builder(data); mod=@__MODULE__))
    predictor = @brm data begin
        eta ~ 1 + X
        y ~ Normal(eta, 1)
    end
    @test popcoefnames(predictor, :eta) == [:Intercept, :X_col_1, :X_col_2]
end
