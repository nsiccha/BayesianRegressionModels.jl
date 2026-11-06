# brm_gmrf_predict on RKBRMI: the fitted reads return the term's own values, a new point
# follows the core's conditional at the draw's lengths, and channels and refusals behave.
using Test, BayesianRegressionModels, LinearAlgebra, Random, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
const BRM = BayesianRegressionModels

function predict_data(; seed=11)
    rng = MersenneTwister(seed)
    npositions, times = 9, [0.0, 1.0, 2.0]
    x, y = 10 .* rand(rng, npositions), 10 .* rand(rng, npositions)
    east, north = repeat(x; outer=length(times)), repeat(y; outer=length(times))
    time = repeat(times; inner=npositions)
    (; east, north, time, y=0.3 .* randn(rng, length(time)))
end

function term_output(rk, u, name)
    only(rk_draw_values(rk, reshape(u, :, 1), (name,))[name])
end

@testset "a model-scope read-pair field" begin
    data = predict_data()
    model = @brm data begin
        field ~ 0 + gmrf((east, north), time; neighbors=4)
        sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(3.0), 0.4)
        length_scale(field, gmrf((east, north), time), time) = 1.7
        y ~ Normal(field, 0.2)
    end
    rk = RKBRMI(model)
    names = BRM._rk_layout_coordinate_names(rk)
    rng = MersenneTwister(1)
    U = 0.2 .* randn(rng, length(names), 3)
    @testset "at the fitted reads it is the term's own output" begin
        out = brm_gmrf_predict(rk, U, :field; data.east, data.north, data.time)
        @test size(out) == (length(data.y), 3)
        for j in axes(U, 2)
            @test out[:, j] ≈ term_output(rk, U[:, j], :field) rtol=1e-10
        end
    end
    @testset "a new point follows the core at the draw's law" begin
        record = BRM._brm_gmrf_footprint_record(rk, :field)
        f = only(rk.plan.columns[record.fitted]).geometry
        law = BRM._brm_gmrf_draw_law(rk, record, f, U[:, 2])
        @test law.temporal_length == 1.7   # a known length has no coordinate
        @test law.sigma > 0 && law.spatial_length > 0
        p = BRM._brm_vecchia_predictor(f, [5.0], [5.0], [0.5])
        core = zeros(1)
        BRM._brm_vecchia_predict!(core, p, law.field, law.spatial_length, law.temporal_length, law.L, f)
        out = brm_gmrf_predict(rk, U, :field; east=[5.0], north=[5.0], time=[0.5])
        @test out[1, 2] ≈ law.sigma * core[1] rtol=1e-12
        # With an rng, the conditional draw is reproducible and differs from the mean.
        a = brm_gmrf_predict(rk, U, :field; east=[5.0], north=[5.0], time=[0.5], rng=Xoshiro(3))
        b = brm_gmrf_predict(rk, U, :field; east=[5.0], north=[5.0], time=[0.5], rng=Xoshiro(3))
        @test a == b && a != out
    end
    @testset "refusals name the problem" begin
        @test_throws ArgumentError brm_gmrf_predict(rk, U, :nothing_here; east=[0.0], north=[0.0], time=[0.0])
        @test_throws DimensionMismatch brm_gmrf_predict(rk, U[1:2, :], :field; east=[0.0], north=[0.0], time=[0.0])
        @test_throws ArgumentError brm_gmrf_predict(rk, U, :field; east=[NaN], north=[0.0], time=[0.0])
    end
end

@testset "channels are read through their factor" begin
    base = predict_data()
    channel = repeat(["a", "b"]; outer=length(base.y))
    data = (; east=repeat(base.east; inner=2), north=repeat(base.north; inner=2),
        time=repeat(base.time; inner=2), channel, y=repeat(base.y; inner=2))
    model = @brm data begin
        Omega ~ LKJCholesky(2, 2.0)
        field ~ 0 + gmrf((east, north), time, channel; neighbors=3, cor=Omega)
        length_scale(field, gmrf((east, north), time, channel), space) = 3.0
        y ~ Normal(field, 0.2)
    end
    rk = RKBRMI(model)
    names = BRM._rk_layout_coordinate_names(rk)
    U = 0.2 .* randn(MersenneTwister(2), length(names), 2)
    out = brm_gmrf_predict(rk, U, :field; data.east, data.north, data.time, data.channel)
    for j in axes(U, 2)
        @test out[:, j] ≈ term_output(rk, U[:, j], :field) rtol=1e-10
    end
    @test_throws ArgumentError brm_gmrf_predict(rk, U, :field; east=[1.0], north=[1.0], time=[0.0])
    @test_throws ArgumentError brm_gmrf_predict(rk, U, :field; east=[1.0], north=[1.0], time=[0.0], channel=["z"])
end

@testset "the mesh form is pointed to its own route" begin
    xy = [0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5]
    triangles = [1 2 5; 2 3 5; 3 4 5; 4 1 5]
    input = (; east=[0.2, 0.7], north=[0.3, 0.6], time=[0.0, 1.0], y=[0.1, -0.1], xy, triangles)
    model = @brm input begin
        field ~ 0 + gmrf((east, north), time; mesh=(vertices=xy, cells=triangles))
        y ~ Normal(field, 0.2)
    end
    rk = RKBRMI(model)
    U = zeros(length(BRM._rk_layout_coordinate_names(rk)), 1)
    @test_throws ArgumentError brm_gmrf_predict(rk, U, :field; east=[0.5], north=[0.5], time=[0.0])
end
