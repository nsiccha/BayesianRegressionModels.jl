using Test, Distributions, BayesianRegressionModels, StanBlocks
import BridgeStan, ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
using Statistics: mean

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

@testset "matrix products with derived coefficients" begin
    training = (; X=[1. 2.; 3. 4.; 5. 6.], K=2, y=[-1., 0.5, 2.], tau=1.3)
    square = merge(training, (; X=[2. -1.; 0.5 3.], y=[0.2, -0.7]))
    model = @brm training begin
        b[1:K] .~ Normal.(0, 1)
        scale = sqrt(1 .+ b .^ 2)
        coef = tau .* b ./ scale
        eta = X * coef
        y ~ Normal(eta, 1)
    end
    b = [0.25, -0.4]
    sb, rk = SBBRMI(model; mod=@__MODULE__), RKBRMI(model)
    @test occursin("X * coef", stan_code(sb))
    @test occursin("X * coef", string(BayesianRegressionModels._rk_emit_ast(rk.plan).main))
    mktempdir() do dir
        for (s, r, rows) in ((sb, rk, training),
                (reprocess(sb, square), reprocess(rk, square), square))
            scale = sqrt.(1 .+ b.^2)
            residual = rows.y - rows.X * (rows.tau .* b ./ scale)
            lp = sum(logpdf.(Normal(), residual)) + sum(logpdf.(Normal(), b))
            gradient = (rows.X' * residual) .* rows.tau ./ scale.^3 - b
            check_product(s, r, b, lp, gradient, joinpath(dir, "derived.stan"))
        end
    end

    # A forward read inside a derived expression is resolved after its
    # declaration. A fresh square design still uses the legacy row broadcast.
    source = """
        coef = b .* tau
        b[1:K] .~ Normal.(0, 1)
        y ~ Normal(sum(X * coef), 1)
    """
    builder = brm(source; mod=@__MODULE__)
    @test occursin("X * coef", stan_code(SBBRMI(builder(training); mod=@__MODULE__)))
    explicit_rows = brm(square, replace(source, "X * coef" => "X .* coef"); mod=@__MODULE__)
    @test stan_code(SBBRMI(builder(square); mod=@__MODULE__)) ==
        stan_code(SBBRMI(explicit_rows; mod=@__MODULE__))

    # A matrix-valued broadcast is not a coefficient vector, even when one
    # of its inputs is exactly as long as the number of design columns.
    data = merge(training, (; M=[1. 3.; 2. -4.]))
    matrices = @brm data begin
        b[1:K] .~ Normal.(0, 1)
        c = M .* b
        value = X * c
    end
    @test getf(last(getargs(parent(matrices.operations.value)))) === (*)
end

const inline_matvec = @brm begin
    b[1:K] .~ Normal.(0, 1)
    y ~ Normal(X * b, 1)
end
const explicit_matvec = @brm begin
    b[1:K] .~ Normal.(0, 1)
    y .~ Normal.(X * b, 1)
end

@brm function inline_normal_map(design, coef, response)
    response ~ Normal(design * coef, 1)
    return coef
end

@brm function observed_linear_map(design, coef, response)
    response .~ Normal.(design * coef, 1)
    return coef
end

@testset "matrix-vector products in likelihood arguments" begin
    data = (; X=[1. 2.; 3. 4.; 5. 6.], K=2, y=[-1., 0.5, 2.])
    square = (; X=[2. -1.; 0.5 3.], K=2, y=[-0.2, 1.])
    b = [0.25, -0.4]
    function verify(sb, rk, rows, path)
        residual = rows.y - rows.X * b
        lp = sum(logpdf.(Normal(), residual)) + sum(logpdf.(Normal(), b))
        check_product(sb, rk, b, lp, rows.X' * residual - b, path)
    end
    mktempdir() do dir
        for (builder, spelling) in ((inline_matvec, "y ~ Normal(X * b, 1)"),
                (explicit_matvec, "y .~ Normal.(X * b, 1)"))
            for make in (builder, brm(builder.source; mod=@__MODULE__))
                model = make(data)
                @test occursin(spelling, string(model))
                sb, rk = SBBRMI(model; mod=@__MODULE__), RKBRMI(model)
                @test occursin("X * b", stan_code(sb))
                @test occursin("X * b", string(BayesianRegressionModels._rk_emit_ast(rk.plan).main))
                path = joinpath(dir, "inline.stan")
                verify(sb, rk, data, path)
                verify(reprocess(sb, square), reprocess(rk, square), square, path)
            end
        end
        # The new observation spelling gives bare * Julia's product even
        # when the row and coefficient counts happen to agree at training.
        model = explicit_matvec(square)
        verify(SBBRMI(model; mod=@__MODULE__), RKBRMI(model), square,
            joinpath(dir, "inline.stan"))

        direct_component = @brm data begin
            b[1:K] .~ Normal.(0, 1)
            result ~ inline_normal_map(X, b, y)
        end
        dotted_component = @brm data begin
            b[1:K] .~ Normal.(0, 1)
            result ~ observed_linear_map(X, b, y)
        end
        for component in (direct_component, dotted_component)
            verify(SBBRMI(component; mod=@__MODULE__), RKBRMI(component), data,
                joinpath(dir, "component.stan"))
        end

        # The two spellings use the existing omitted-response route. They
        # keep the response separate from the declared coefficient vector.
        unobserved = (; data.X, data.K)
        for builder in (inline_matvec, explicit_matvec)
            omitted = builder(unobserved)
            sb = SBBRMI(omitted; mod=@__MODULE__)
            @test any(d -> d.target === :y && d.role === :observation,
                generative_plan(sb).declarations)
            @test StanBlocks.stanc_check(stan_code(sb); warn_pedantic=false).ok

            replay = reprocess(RKBRMI(builder(data)), (; square.X, square.K))
            predicted = ReactiveKernelsPPL.prepare_predictive(replay.model, rk_bound_plan(replay))
            @test mean.(ReactiveKernelsPPL.predictive_laws(predicted, b).y) ≈ square.X * b
        end
    end
    # A product nested under a reduction still distinguishes row scaling.
    rowdata = merge(data, (; K=3))
    scale = @brm rowdata begin
        b[1:K] .~ Normal.(0, 1)
        y .~ Normal.(sum(X .* b), 1)
    end
    @test occursin("sum(X .* b)", string(scale))
    @test occursin("brm_scale_matrix_rows", stan_code(SBBRMI(scale; mod=@__MODULE__)))
    legacy = @brm rowdata begin
        b[1:K] .~ Normal.(0, 1)
        y ~ Normal(sum(X * b), 1)
    end
    @test stan_code(SBBRMI(legacy; mod=@__MODULE__)) == stan_code(SBBRMI(scale; mod=@__MODULE__))
    bare = @brm(y .~ Normal.(m, 1))
    @test string(bare((; y=data.y, m=data.y))) == "y .~ Normal.(m, 1)"
    # A joint law remains one event. A dotted destination needs a scalar law.
    @test_throws ArgumentError brm(data, "y .~ MvNormal.(y, 1)"; mod=@__MODULE__)
    # Categorical consumes a whole probability vector even though its event
    # is scalar. The explicit form must never reinterpret it as a shared law.
    @test_throws ArgumentError brm((; y=[1,2], p=[0.4,0.6]), "y .~ Categorical.(p)"; mod=@__MODULE__)
end
