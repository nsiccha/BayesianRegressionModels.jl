# RUN: julia --project=test test/matrix_start_values.jl [--stan]
# Native initialization needs no compiled Stan model. The optional Stan leg
# independently reads the physical matrices and compares transported points.
using Test, BayesianRegressionModels, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL
const BRM = BayesianRegressionModels

@brm function matrix_start_component(K, eta)
    L ~ LKJCholesky(K, eta)
    return L
end

const MATRIX_START_BUILDER = @brm begin
    L ~ LKJCholesky(K, eta)
    scoped ~ matrix_start_component(K, eta)
    center ~ Normal(0.2, 0.7)
    noise ~ Exponential(0.8)
    z[1:2] .~ Normal.(0, 1)
    copy_L = L
    y ~ Normal(L[2, 1] + scoped[2, 1] + center + z[1], noise)
end

matrix_start_factor(K) = K == 2 ? [1.0 0; -0.3 sqrt(0.91)] :
    [1.0 0 0; 0.3 sqrt(0.91) 0; -0.2 0.4 sqrt(0.8)]

function matrix_start_bad_values(K)
    L = matrix_start_factor(K)
    upper = copy(L); upper[1, 2] = 0.1
    negative = copy(L); negative[2, 2] *= -1
    unnormalized = copy(L); unnormalized[2, 2] *= 0.5
    nonfinite = copy(L); nonfinite[2, 1] = NaN
    (ones(K, K), upper, negative, unnormalized, nonfinite,
        ones(K + 1, K), reshape(copy(L), 1, :))
end

@testset "top-level matrix starts use native constrained values" begin
    for K in (2, 3)
        data = (; K, eta=2.0, y=[0.1, -0.2])
        rk = RKBRMI(MATRIX_START_BUILDER(data))
        base = collect(range(-0.25, 0.35; length=rk.model.layout.total))
        saved_base = copy(base)
        before = BRM._rk_constrained_values(rk, base)
        for L in (Matrix{Int}(I, K, K), matrix_start_factor(K),
                LowerTriangular(matrix_start_factor(K)))
            saved = copy(L)
            point = brm_start(rk, (; L); base)
            values = BRM._rk_constrained_values(rk, point)
            @test values.L ≈ L atol=1e-12 rtol=1e-12
            @test values.center == before.center
            @test values.noise ≈ before.noise
            @test values.z == before.z
            @test values.scoped.L ≈ before.scoped.L
            @test all(isfinite, point)
            @test base == saved_base
            @test L == saved
            neutral = brm_start(rk, (; L))
            @test BRM._rk_constrained_values(rk, neutral).L ≈ L

            # Same physical input at top level and within a component, with
            # explicit scalar/vector values and prior medians in the same call.
            full = brm_start(rk, (; L, center=0.4, z=[-0.1, 0.6]);
                component_inits=[:scoped => (; L)], base=:prior_median)
            v = BRM._rk_constrained_values(rk, full)
            @test v.L ≈ L atol=1e-12 rtol=1e-12
            @test v.scoped.L ≈ L atol=1e-12 rtol=1e-12
            @test v.center == 0.4
            @test v.z == [-0.1, 0.6]
            @test v.noise ≈ 0.8 * log(2)
        end
        for bad in matrix_start_bad_values(K)
            @test_throws Exception brm_start(rk, (; L=bad))
        end
        @test_throws DimensionMismatch brm_start(rk, (; z=ones(1, 2)))
        @test_throws Exception brm_start(rk, (; copy_L=matrix_start_factor(K)))
        @test_throws Exception brm_start(rk, (; missing=matrix_start_factor(K)))
    end
end

if "--stan" in ARGS
    using BridgeStan
    @testset "top-level matrix starts agree with compiled Stan" begin
        for K in (2, 3)
            model = MATRIX_START_BUILDER((; K, eta=2.0, y=[0.1, -0.2]))
            rk = RKBRMI(model)
            sb = SBBRMI(model; mod=@__MODULE__)
            code = BRM.stan_code(sb)
            stan = BRM.stan_instantiate(sb; path=joinpath(mktempdir(), "matrix-start.stan")).model
            descriptor = brm_descriptor(sb)
            names = BridgeStan.param_names(stan)
            positions = brm_output_coordinates(descriptor, :L, names)
            transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan))
            base = collect(range(-0.25, 0.35; length=rk.model.layout.total))
            stan_base = brm_rk_point_to_stan(transport, base)
            saved_base = copy(stan_base)
            for L in (Matrix{Int}(I, K, K), matrix_start_factor(K),
                    LowerTriangular(matrix_start_factor(K)))
                saved = copy(L)
                r = brm_start(rk, (; L); base)
                s = brm_start(sb, stan, (; L); base=stan_base)
                @test reshape(BridgeStan.param_constrain(stan, s)[positions], K, K) ≈ L
                @test s ≈ brm_rk_point_to_stan(transport, r) atol=1e-10 rtol=1e-10
                @test brm_check_coordinate_transport(transport, rk, stan, r).max_error < 1e-10
                @test isfinite(BridgeStan.log_density(stan, s))
                @test stan_base == saved_base
                @test L == saved
                neutral = brm_start(sb, stan, (; L))
                @test reshape(BridgeStan.param_constrain(stan, neutral)[positions], K, K) ≈ L

                inits = (; L, center=0.4, z=[-0.1, 0.6])
                component_inits = [:scoped => (; L)]
                r = brm_start(rk, inits; component_inits, base=:prior_median)
                s = brm_start(sb, stan, inits; component_inits, base=:prior_median)
                @test s ≈ brm_rk_point_to_stan(transport, r) atol=1e-10 rtol=1e-10
            end
            for bad in matrix_start_bad_values(K)
                @test_throws Exception brm_start(sb, stan, (; L=bad))
            end
            @test_throws DimensionMismatch brm_start(sb, stan, (; z=ones(1, 2)))
            @test_throws Exception brm_start(sb, stan, (; copy_L=matrix_start_factor(K)))
            @test_throws Exception brm_start(sb, stan, (; missing=matrix_start_factor(K)))
            @test BRM.stan_code(sb) == code
        end
    end
end
