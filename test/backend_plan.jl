using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal

const BRM = BayesianRegressionModels

@testset "backend-neutral BRMI context and simple population design" begin
    df = (; x=Float32[0, 1, 2], y=[1.0, 1.5, 2.2])
    brmi = (@brm begin
        sigma ~ Exponential(2)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end)(df)

    context = BRM._brm_backend_context(brmi)
    @test context.parent === brmi
    @test context.target_obs == Dict(:mu => :y, :sigma => :y, :y => :y)
    @test context.data[:x] == df.x
    @test context.data[:y] == df.y

    _, rhs = getargs(linear_predictor_op(brmi, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)
    @test Tuple(c.label for c in design.columns) == (:Intercept, :x)
    @test Tuple(c.source for c in design.columns) == (nothing, :x)
    @test design.row_source === :x
    @test design.matrix == [1.0 0.0; 1.0 1.0; 1.0 2.0]

    # SBBRMI consumes the same shared column plan, but retains its existing
    # backend-specific optimized program byte-for-byte at the AST boundary.
    emitted = deepcopy(SBBRMI(brmi).model.model)
    expected = quote
        sigma ~ exponential(1.0 ./ 2)
        X_mu = hcat(rep_vector(1.0, num_elements(x)), x)
        pop_mu ~ _popefs_coefs(; X=X_mu)
        y ~ normal_id_glm(X_mu, 0.0, pop_mu, sigma)
        mu = X_mu * pop_mu
    end
    @test Base.remove_linenums!(emitted) == Base.remove_linenums!(expected)
end

@testset "shared population design is narrow and loud" begin
    intercept = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1
        y ~ Normal(mu, sigma)
    end)((; y=[0.0, 1.0, 2.0]))
    context = BRM._brm_backend_context(intercept)
    _, rhs = getargs(linear_predictor_op(intercept, :mu), 2)
    design = BRM._brm_simple_population_design(
        :mu, rhs, context.data, context.target_obs[:mu]; required=true)
    @test design.row_source === :y
    @test design.matrix == ones(3, 1)

    categorical = (@brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + g
        y ~ Normal(mu, sigma)
    end)((; g=[1, 2, 1], y=zeros(3)))
    cat_context = BRM._brm_backend_context(categorical)
    _, cat_rhs = getargs(linear_predictor_op(categorical, :mu), 2)
    @test isnothing(BRM._brm_simple_population_design(
        :mu, cat_rhs, cat_context.data, cat_context.target_obs[:mu]))
    @test_throws "supports only `1` and continuous raw-data columns" begin
        BRM._brm_simple_population_design(
            :mu, cat_rhs, cat_context.data, cat_context.target_obs[:mu];
            required=true)
    end

    missing_x = Union{Missing,Float64}[1.0, missing]
    @test_throws "never silently drops rows" begin
        broken = (@brm begin
            sigma ~ Exponential(1)
            mu ~ 1 + x
            y ~ Normal(mu, sigma)
        end)((; x=missing_x, y=zeros(2)))
        BRM._brm_backend_context(broken)
    end
end
