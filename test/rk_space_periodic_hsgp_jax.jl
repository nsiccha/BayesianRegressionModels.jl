using Test, Random
using BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
include("jax_module_driver.jl")

const spatial_season = @brm begin
    mu ~ 0 + hsgp(east, north, month; periodic=:month, period=12.0,
                  k=(2, 3, 2), domain=((-2.0, 2.0), (-2.0, 2.0)),
                  orthogonal_to=:intercept, at=cell_pos)
    y ~ Normal(mu, 1.0)
end

@testset "spatial × periodic HSGP JAX parity" begin
    data = (; east=[-0.8, 0.1, 0.9], north=[0.2, -0.6, 0.7],
        month=[1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        cell_pos=[1, 2, 3, 1, 2, 3, 2], y=zeros(7))
    rk = RKBRMI(spatial_season(data))
    source = jax_source(rk.model, rk_bound_plan(rk); values=(:mu,))
    @test source isa JAXSource
    points = [zeros(rk.model.layout.total),
        0.1 .* randn(Xoshiro(17), rk.model.layout.total)]
    problem = rk_logdensity_problem(rk;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    mu_query = prepare_query(rk.model, rk_bound_plan(rk), (:mu,))
    native = map(points) do u
        lp, grad = LogDensityProblems.logdensity_and_gradient(problem, u)
        (; lp, grad, mu=mu_query(u))
    end
    if brm_jax_require_python()
        run = brm_jax_run(source, [brm_jax_inputs(rk, source)], [points])
        @test isempty(run.module_constants["UNTRANSLATED"])
        for (observed, expected) in zip(only(run.datasets)["points"], native)
            @test observed["log_density"]["jit"] ≈ expected.lp rtol=1e-12 atol=1e-12
            @test vec(observed["grad"]) ≈ expected.grad rtol=1e-10 atol=1e-10
            @test vec(observed["constrain_jit"]["mu"]) ≈ expected.mu rtol=1e-12 atol=1e-12
        end
    end
end
