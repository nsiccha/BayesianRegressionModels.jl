using Test, Random, LinearAlgebra, Distributions, BayesianRegressionModels, StanBlocks
using BayesianRegressionModels: BernoulliLogit
import BridgeStan, ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme

const habitat = @brm begin
    eta ~ 1 + offset(log_area) + X
    effect(eta, Intercept) ~ Normal(-3., 2.)
    effect(eta, X) .~ Laplace.(0., scales)
    intensity = exp(eta)
    log_odds = intensity + log(-expm1(-intensity))
    y ~ BernoulliLogit(log_odds)
end

function fixture(rows, width)
    rng = Xoshiro(473)
    (; X=randn(rng, rows, width), log_area=0.2randn(rng, rows),
       scales=0.1 .+ rand(rng, width), y=rand(rng, 0:1, rows))
end

# Different scales and nonzero slopes distinguish every column's prior and
# expose a transposed or reordered design. The Bernoulli event probability
# is 1 - exp(-exp(eta)).
function habitat_oracle(data, u)
    alpha, beta = first(u), u[2:end]
    lambda = exp.(alpha .+ data.log_area .+ data.X * beta)
    lp = logpdf(Normal(-3, 2), alpha) +
        sum(logpdf.(Laplace.(0., data.scales), beta)) +
        sum(ifelse.(data.y .== 1, log.(-expm1.(-lambda)), -lambda))
    score = ifelse.(data.y .== 1, lambda ./ expm1.(lambda), -lambda)
    lp, vcat(-(alpha + 3)/4 + sum(score), data.X' * score - sign.(beta) ./ data.scales)
end

function compare_backends(sb, rk, data, u, path)
    stan = stan_instantiate(sb; path).model
    names = BridgeStan.param_unc_names(stan)
    @test length(names) == size(data.X, 2) + 1
    axis = BridgeStan.param_names(stan; include_tp=false, include_gq=false)
    descriptor = brm_descriptor(sb)
    @test brm_population_effect_coordinates(descriptor, :eta, axis; coefficient=:X).coordinates ==
        collect(2:length(names))
    @test brm_population_effect_coordinates(descriptor, :eta, axis; coefficient=:X_col_1).coordinates == [2]
    value, grad = BridgeStan.log_density_gradient(stan, u; propto=false, jacobian=true)
    lp, score = habitat_oracle(data, u)
    @test value ≈ lp atol=1e-9
    @test grad ≈ score atol=1e-9
    transport = brm_coordinate_transport(rk, sb, names)
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    rv, rg = LogDensityProblems.logdensity_and_gradient(problem, brm_stan_to_rk(transport, u))
    @test rv ≈ lp atol=1e-9
    @test rg ≈ brm_stan_to_rk(transport, score) atol=1e-9
end

@testset "numeric matrix population term" begin
    mktempdir() do dir
        programs = String[]
        for (n, k) in ((7, 3), (492, 192))
            d = fixture(n, k)
            m = habitat(d)
            labels = [Symbol("X_col_", j) for j in 1:k]
            @test popcoefnames(m, :eta) == [:Intercept; labels]
            sb = SBBRMI(m; mod=@__MODULE__)
            push!(programs, stan_code(sb))
            rk = RKBRMI(m)
            @test count(v -> v isa AbstractMatrix, values(sb.data)) == 1
            @test rk_bound_data(rk).X_eta == hcat(ones(n), d.X)
            @test isempty(intersect(labels, keys(rk_bound_data(rk))))
            u = [-1.3; [0.015cos(j) for j in 1:k]]
            path = joinpath(dir, "habitat_$k.stan")
            compare_backends(sb, rk, d, u, path)
            k == 3 || continue
            new = merge(fixture(2, k), (; scales=d.scales))
            snew, rnew = reprocess(sb, new), reprocess(rk, new)
            @test rk_bound_data(rnew).X_eta == hcat(ones(2), new.X)
            compare_backends(snew, rnew, new, u, path)
            state = brm_fitted_state(rk)
            @test rk_bound_data(RKBRMI(habitat(new), state)).X_eta == rk_bound_data(rnew).X_eta
            wrong = merge(new, (; X=zeros(2, k + 1)))
            @test_throws "expects a numeric matrix with 3 columns" reprocess(sb, wrong)
            @test_throws "expects a numeric matrix with 3 columns" reprocess(rk, wrong)
            @test_throws "expects a numeric matrix with 3 columns" RKBRMI(habitat(wrong), state)
        end
        @test allequal(programs)
    end
end

@testset "one integer feature, individual priors and width errors" begin
    model = @brm begin
        mu ~ 0 + X
        effect(mu, X) ~ Normal(0, 2)
        effect(mu, X_col_1) ~ Normal(0, 3)
        y ~ Normal(mu, 1)
    end
    d = (; X=reshape([0, 2, 4], 3, 1), y=[1., 2., -1.])
    m = model(d)
    @test popcoefnames(m, :mu) == [:X_col_1]
    sb = SBBRMI(m; mod=@__MODULE__)
    rk = RKBRMI(m)
    b = [0.35]
    expected = logpdf(Normal(0, 3), only(b)) + sum(logpdf.(Normal.(d.X*b, 1), d.y))
    mktempdir() do dir
        stan = stan_instantiate(sb; path=joinpath(dir, "integer.stan")).model
        @test BridgeStan.log_density(stan, b; propto=false) ≈ expected
    end
    @test LogDensityProblems.logdensity(rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), b) ≈ expected
    one = fixture(4, 1)
    m1 = habitat(one)
    mktempdir() do dir
        compare_backends(SBBRMI(m1; mod=@__MODULE__), RKBRMI(m1), one,
            [-1.3, 0.02], joinpath(dir, "one_feature.stan"))
    end
    @test_throws "has no columns" popcoefnames(model(merge(d, (; X=zeros(3, 0)))), :mu)
    data = merge(fixture(4, 2), (; scales=[0.1]))
    @test_throws "has 1 entries" SBBRMI(habitat(data); mod=@__MODULE__)
    @test_throws "has 1 entries" RKBRMI(habitat(data))
end

@testset "matrix blocks retain their place among ordinary columns" begin
    builder = @brm begin
        mu ~ 1 + X + x + Z
        y ~ Normal(mu, 1)
    end
    data = (; X=[1 2; 3 4; 5 6], x=[0.1, 0.2, 0.3], Z=reshape([7., 8., 9.], :, 1),
        y=[0., 0.1, 0.2])
    model = builder(data)
    @test popcoefnames(model, :mu) == [:Intercept, :X_col_1, :X_col_2, :x, :Z_col_1]
    rk = RKBRMI(model)
    @test rk_bound_data(rk).X_mu == hcat(ones(3), data.X, data.x, data.Z)
    sb = SBBRMI(model; mod=@__MODULE__)
    syntax = stan_code(sb)
    @test occursin("append_col(", syntax)
    @test count(v -> v isa AbstractMatrix, values(sb.data)) == 2
    for backend in (sb, rk)
        @test_throws "expects a numeric matrix with 2 columns" reprocess(backend,
            merge(data, (; X=ones(3, 1))))
    end
end
