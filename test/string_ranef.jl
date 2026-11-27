# Plain String random slopes use the same level coding as explicit factors.
# Run: julia --startup-file=no --project=test test/string_ranef.jl
using Test, BayesianRegressionModels, StanBlocks, LogDensityProblems, Random
import CategoricalArrays as CA

const BRM = BayesianRegressionModels
const STRING_RANEF_DATA = (;
    c=repeat(["c", "a", "b"], 4),
    g=repeat(1:4; inner=3),
    x=collect(range(-1.0, 1.0; length=12)),
    y=collect(range(-0.8, 0.7; length=12)))
string_ranef_sb(builder, data=STRING_RANEF_DATA) =
    SBBRMI(builder(data); mod=@__MODULE__)

@testset "String random slopes: cells, contrasts and block composition" begin
    builders = (
        @brm(begin mu ~ 1 + (0 + c | g); y ~ Normal(mu, 1.0) end),
        @brm(begin mu ~ 1 + (1 + c | g); y ~ Normal(mu, 1.0) end),
        @brm(begin mu ~ 1 + (1 | g) + (0 + c | g); y ~ Normal(mu, 1.0) end),
        @brm(begin mu ~ 1 + (0 + x + c | g); y ~ Normal(mu, 1.0) end),
        @brm(begin mu ~ 1 + (0 + c || g); y ~ Normal(mu, 1.0) end),
        @brm(begin mu ~ 1 + (1 + c || g); y ~ Normal(mu, 1.0) end),
        @brm(begin mu ~ 1 + (0 + factor(c; cmc=false) | g); y ~ Normal(mu, 1.0) end),
        @brm(begin mu ~ 1 + (0 + factor(c; ref="c") | g); y ~ Normal(mu, 1.0) end),
        @brm(begin mu ~ 1 + (1 + factor(c; ref="c") | g); y ~ Normal(mu, 1.0) end),
    )
    data = STRING_RANEF_DATA
    declared = merge(data, (; c=CA.categorical(data.c; levels=["a", "b", "c"])))
    for builder in builders
        plain, categorical = string_ranef_sb(builder), string_ranef_sb(builder, declared)
        @test BRM.stan_code(plain) == BRM.stan_code(categorical)
        @test BRM.stan_data(plain) == BRM.stan_data(categorical)
        @test StanBlocks.stanc_check(BRM.stan_code(plain); warn_pedantic=false).ok
    end
end

@testset "String random slopes: reported grouped AR1 model" begin
    rng = Xoshiro(1)
    n = 60
    g = repeat(["a", "b"]; inner=n)
    t = repeat(collect(1:n); outer=2)
    h = repeat(mod1.(1:n, 6); outer=2)
    x = randn(rng, 2n)
    y = 0.5 .* x .+ 0.3 .* randn(rng, 2n)
    builder = @brm begin
        mu ~ 0 + x + (0 + g | h)
        atanh(phi) ~ 0 + g
        sigma ~ Exponential(1.0)
        @plate for p in groups(g; order=t)
            y[p] ~ AR1Normal(mu[p], sigma, phi[p]; time=t[p])
        end
    end
    twin = @brm begin
        mu ~ 0 + x + (0 + g_dummy_1 + g_dummy_2 | h)
        atanh(phi) ~ 0 + g
        sigma ~ Exponential(1.0)
        @plate for p in groups(g; order=t)
            y[p] ~ AR1Normal(mu[p], sigma, phi[p]; time=t[p])
        end
    end
    data = (; g, t, h, x, y)
    indicators = (; g_dummy_1=Float64.(g .== "a"), g_dummy_2=Float64.(g .== "b"))
    sb = string_ranef_sb(builder, data)
    numeric = string_ranef_sb(twin, merge(data, indicators))
    @test BRM.stan_code(sb) == BRM.stan_code(numeric)
    @test BRM.stan_data(sb) == BRM.stan_data(numeric)
    problem = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "string-ranef", "ar1.stan"))
    nparams = LogDensityProblems.dimension(problem)
    @test nparams == 1 + 2 + 1 + 2 + 1 + 2 * 6
    for q in (zeros(nparams), collect(range(-0.3, 0.4; length=nparams)))
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test all(isfinite, gradient)
    end
end

@testset "String random slopes: one level and string subtypes" begin
    builder = @brm begin
        mu ~ 1 + (0 + c | g)
        y ~ Normal(mu, 1.0)
    end
    for c in (fill("only", 12), SubString.(STRING_RANEF_DATA.c, 1, 1))
        data = merge(STRING_RANEF_DATA, (; c))
        sb = string_ranef_sb(builder, data)
        levels = sort(unique(c))
        @test only(ranef_blocks(sb)).n_terms == length(levels)
        for (i, level) in enumerate(levels)
            @test sb.data[Symbol(:c_dummy_, i)] == Float64.(c .== level)
        end
        @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    end
end

@testset "String random slopes: shared margins and frozen replay" begin
    builder = @brm begin
        mu ~ 1 + (0 + c | p | g)
        log(sigma) ~ 1 + (1 + c | p | g)
        sd(mu, p, c_dummy_1) ~ Exponential(0.5)
        y ~ Normal(mu, sigma)
    end
    brmi = builder(STRING_RANEF_DATA)
    @test ranefcoefnames(brmi, :p) == [
        (; predictor=:mu, coefficient=:c_dummy_1),
        (; predictor=:mu, coefficient=:c_dummy_2),
        (; predictor=:mu, coefficient=:c_dummy_3),
        (; predictor=:sigma, coefficient=:Intercept),
        (; predictor=:sigma, coefficient=:c_dummy_2),
        (; predictor=:sigma, coefficient=:c_dummy_3)]
    sb = string_ranef_sb(builder)
    @test sb.data[:n_terms_p_g] == 6
    @test sb.data[:c_dummy_1] == Float64.(STRING_RANEF_DATA.c .== "a")
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    subset = (; c=["c", "a"], g=[2, 4], x=[0.2, -0.4], y=[0.0, 0.0])
    for replay in (reprocess(sb, subset), reprocess(sb, subset; resample_groups=[:g]))
        @test replay.data[:c_dummy_1] == [0.0, 1.0]
        @test replay.data[:c_dummy_2] == [0.0, 0.0]
        @test replay.data[:c_dummy_3] == [1.0, 0.0]
        @test only(ranef_blocks(replay)).n_terms == 6
        @test StanBlocks.stanc_check(BRM.stan_code(replay); warn_pedantic=false).ok
    end
    @test_throws "not a training level" reprocess(sb, merge(subset, (; c=["a", "new"])))
end

@testset "String random slopes: explicit indicator twin and runtime" begin
    builder = @brm begin
        mu ~ 1 + x + (0 + c | g)
        y ~ Normal(mu, 1.0)
    end
    twin = @brm begin
        mu ~ 1 + x + (0 + c_dummy_1 + c_dummy_2 + c_dummy_3 | g)
        y ~ Normal(mu, 1.0)
    end
    data = STRING_RANEF_DATA
    indicators = (; c_dummy_1=Float64.(data.c .== "a"),
        c_dummy_2=Float64.(data.c .== "b"), c_dummy_3=Float64.(data.c .== "c"))
    sb, numeric = string_ranef_sb(builder), string_ranef_sb(twin, merge(data, indicators))
    @test BRM.stan_code(sb) == BRM.stan_code(numeric)
    @test BRM.stan_data(sb) == BRM.stan_data(numeric)
    problem = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "string-ranef", "model.stan"))
    n = LogDensityProblems.dimension(problem)
    @test n == 2 + 3 + 3 + 3 * 4
    for q in (zeros(n), collect(range(-0.3, 0.4; length=n)))
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test all(isfinite, gradient)
    end
end
