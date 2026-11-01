# Declared vectors carry their own axis into direct and grouped plate reads.
# Run: julia --project=test test/plate_vector_parameters.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const vector_plate_data = (; y=[0.4, -0.2, 0.7], n=3,
    scale=[0.5, 0.7, 0.9], axis=zeros(6),
    covariance=[1.0 0.2 0.0; 0.2 1.0 0.1; 0.0 0.1 1.0])

const vector_plate_literal = @brm begin
    x[1:3] .~ Normal.(0.0, 0.8)
    @plate for i in eachindex(y)
        y[i] ~ Normal(x[i], 0.6)
    end
end

@stestset "a downstream plate reads complete vector cells" begin
    data = (; indices=[[1, 2, 3], [3, 1]], y=[[0.1, 0.2, 0.3], [0.4, 0.5]])
    readers = (
        ("direct", quote
            @plate for g in eachindex(y)
                y[g] ~ Normal(mu[g], 0.2)
            end
        end, 1.0),
        ("expression", quote
            @plate for g in eachindex(y)
                y[g] ~ Normal(2.0 .* mu[g], 0.2)
            end
        end, 2.0),
        ("deterministic", quote
            @plate for g in eachindex(y)
                twice[g] = 2.0 .* mu[g]
            end
            @plate for g in eachindex(y)
                y[g] ~ Normal(twice[g], 0.2)
            end
        end, 2.0),
    )
    for (label, reader, multiplier) in readers
        builder = brm(quote
            x[1:3] .~ Normal.(0.0, 0.8)
            theta[1:2] .~ Normal.(0.0, 0.5)
            @plate for g in eachindex(indices)
                mu[g] = x[indices[g]] .+ theta[g]
                total[g] = sum(mu[g])
            end
            $reader
        end; mod=@__MODULE__)
        brmi = builder(data)
        rk, problem = consumer_problem(brmi)
        stan = consumer_stan(brmi, "plate-vector-output-" * label)
        names = coordinate_names(rk.model.layout)
        xi = [coordinate(names, Symbol("x.", j)) for j in 1:3]
        ti = [coordinate(names, Symbol("theta.", j)) for j in 1:2]
        mapping = [name => string(name) for name in names]
        oracle(u) = sum(logpdf.(Normal(0.0, 0.8), u[xi])) +
            sum(logpdf.(Normal(0.0, 0.5), u[ti])) +
            sum(sum(logpdf.(Normal.(multiplier .* (u[xi[data.indices[g]]] .+ u[ti[g]]),
                0.2), data.y[g])) for g in eachindex(data.y))
        for u in (zeros(5), [-0.3, 0.2, 0.5, 0.1, -0.2])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, rk, u)
        end
    end
end

const vector_plate_positive = @brm begin
    x[1:n] .~ LogNormal.(-0.4, 0.7)
    @plate for i in eachindex(y)
        y[i] ~ Normal(x[i], 0.6)
    end
end

@stestset "prior hyperparameters do not define plate rows" begin
    laws = (
        :(x[1:length(axis) / 2] .~ Normal.(0.0, 0.8)),
        :(x[1:n] .~ Normal.(0.0, scale)),
        :(x ~ MvNormal(zeros(n), covariance)),
        :(x ~ MvNormal(n, 0.8)),
        :(x ~ Dirichlet(3, 1.2)),
        :(x ~ Dirichlet([1.0, 2.0, 3.0])),
    )
    for law in laws
        builder = brm(quote
            $law
            @plate for i in eachindex(y)
                y[i] ~ Normal(x[i], 0.6)
            end
        end; mod=@__MODULE__)
        brmi = builder(vector_plate_data)
        sb = SBBRMI(brmi; mod=@__MODULE__)
        @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
        rk = RKBRMI(brmi)
        @test length(coordinate_names(rk.model.layout)) == (occursin("Dirichlet", string(law)) ? 2 : 3)
    end
end

@stestset "declared vectors beside sampled scalar hyperparameters and formula predictors" begin
    builder = @brm begin
        tau ~ Exponential(0.8)
        x[1:3] .~ Normal.(0.0, tau)
        mu ~ 1 + (1 | subject)
        @plate for i in eachindex(y)
            y[i] ~ Normal(mu[i] + x[i], 0.6)
        end
    end
    data = merge(vector_plate_data, (; subject=["c", "a", "b"]))
    brmi = builder(data)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    rk = RKBRMI(brmi)
    @test any(d -> d.target === :x && d.role === :prior, generative_plan(sb).declarations)
    @test count(n -> startswith(string(n), "x."), coordinate_names(rk.model.layout)) == 3
end

@stestset "vector declarations supply a plate axis without data rows" begin
    builder = @brm begin
        x[1:3] .~ Normal.(0.0, 0.8)
        @plate for i in eachindex(x)
            z[i] = 2.0 * x[i]
        end
        y ~ Normal(sum(z), 0.6)
    end
    data = (; y=0.4)
    brmi = builder(data)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    replay = reprocess(sb, (; y=-0.2))
    @test BRM.stan_code(replay) == BRM.stan_code(sb)
    rk, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "plate-vector-own-axis")
    names = coordinate_names(rk.model.layout)
    mapping = [name => string(name) for name in names]
    oracle(u) = sum(logpdf.(Normal(0.0, 0.8), u)) + logpdf(Normal(2sum(u), 0.6), data.y)
    for u in (zeros(3), [-0.3, 0.2, 0.5])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, rk, u)
    end
end

@stestset "direct vector extent checks and the scalar predictor guard" begin
    for constructor in (m -> SBBRMI(m; mod=@__MODULE__), RKBRMI)
        for n in (2, 4)
            @test_throws r"vector parameter `x` has .* entries for 3 plate cells" constructor(
                vector_plate_positive(merge(vector_plate_data, (; n))))
        end
        scalar = @brm begin
            x ~ 1
            @plate for i in eachindex(y)
                y[i] ~ Normal(x[i], 0.6)
            end
        end
        @test_throws r"no per-subject rows" constructor(scalar(vector_plate_data))
    end
end

@stestset "ragged observations and grouped vector-parameter reads" begin
    data = (; n=3, y=[[0.1, 0.4], [-0.2], [0.8, 0.3, 0.5]])
    brmi = vector_plate_positive(data)
    rk, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "plate-vector-ragged-observation")
    names = coordinate_names(rk.model.layout)
    mapping = [name => string(name) for name in names]
    oracle(u) = sum(logpdf.(LogNormal(-0.4, 0.7), exp.(u))) + sum(u) +
        sum(sum(logpdf.(Normal(exp(u[i]), 0.6), data.y[i])) for i in eachindex(u))
    for u in (zeros(3), [-0.3, 0.2, 0.5])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, rk, u)
    end

    grouped = @brm begin
        x[1:3] .~ Normal.(0.0, 0.8)
        rows = groups(group => cells)
        @plate for i in eachindex(cells)
            y[i] ~ Normal(sum(x[rows[i]]), 0.6)
        end
    end
    grouped_data = (; group=[2, 1, 2], cells=[2, 1], y=[0.4, -0.2])
    model = grouped(grouped_data)
    rk, problem = consumer_problem(model)
    stan = consumer_stan(model, "plate-vector-grouped")
    names = coordinate_names(rk.model.layout)
    mapping = [name => string(name) for name in names]
    grouped_oracle(u) = sum(logpdf.(Normal(0.0, 0.8), u)) +
        logpdf(Normal(u[1] + u[3], 0.6), 0.4) + logpdf(Normal(u[2], 0.6), -0.2)
    for u in (zeros(3), [-0.3, 0.2, 0.5])
        check_consumer_point(problem, u, grouped_oracle)
        check_consumer_stan(problem, stan, mapping, rk, u)
    end
    for constructor in (m -> SBBRMI(m; mod=@__MODULE__), RKBRMI)
        @test_throws r"has 3 entries.*has 2" constructor(grouped(merge(grouped_data, (; group=[2, 1]))))
    end
end
@stestset "literal and scalar-data vector extents in plate cells" begin
    for (label, builder, law, transform, jacobian) in (
            ("normal", vector_plate_literal, Normal(0.0, 0.8), identity, u -> 0.0),
            ("lognormal", vector_plate_positive, LogNormal(-0.4, 0.7), exp, sum))
        before = deepcopy(vector_plate_data)
        brmi = builder(vector_plate_data)
        sb = SBBRMI(brmi; mod=@__MODULE__)
        declaration = only(d for d in generative_plan(sb).declarations if d.target === :x)
        @test declaration.role === :prior
        @test declaration.dimension == (:x_n,)
        @test !occursin("x_gen", BRM.stan_code(sb))
        rk, problem = consumer_problem(brmi)
        stan = consumer_stan(brmi, "plate-vector-" * label)
        names = coordinate_names(rk.model.layout)
        @test length(names) == 3
        mapping = [name => string(name) for name in names]
        oracle(u) = sum(logpdf.(law, transform.(u))) + jacobian(u) +
            sum(logpdf.(Normal.(transform.(u), 0.6), vector_plate_data.y))
        for u in (zeros(3), [-0.3, 0.2, 0.5], [0.1, -0.4, 0.7])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, rk, u)
        end
        @test isequal(vector_plate_data, before)
    end
end
