# BRM_JAX_REQUIRE=1 RK_JAX_PYTHON=<python with jax> julia --project=test test/rk_multigroup_hsgp_graph.jl
using Test, ADTypes, Enzyme, LogDensityProblems
using BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
include("jax_module_driver.jl")

const isotropic = @brm begin
    Omega ~ LKJCholesky(G, 2.0)
    tau ~ LogNormal(0.0, 1.0)
    mu ~ 1 + hsgp(t; k=K, by=g, cor=multigroup(Omega, tau))
    y ~ Normal(mu, 1.0)
end

const anisotropic = @brm begin
    Omega ~ LKJCholesky(G, 2.0)
    tau ~ LogNormal(0.0, 1.0)
    mu ~ 1 + hsgp(t, x; k=(2, 3), by=g, iso=false, cor=multigroup(Omega, tau))
    y ~ Normal(mu, 1.0)
end

function multigroup_data(G, K, n)
    t = collect(range(-1.0, 1.0; length=n))
    (; G, K, t, x=cos.(2 .* t), g=mod1.(1:n, G), y=sin.(3 .* t))
end

function check_multigroup_jax(builder, datasets)
    backends = [RKBRMI(builder(data)) for data in datasets]
    rk = first(backends)
    src = jax_source(rk.model, rk_bound_plan(rk); values=(:mu,))
    @test src isa JAXSource
    brm_jax_require_python() || return
    us = [[zeros(rk.model.layout.total), [0.37sin(i) for i in 1:rk.model.layout.total]]
          for rk in backends]
    inputs = [brm_jax_inputs(rk, src) for rk in backends]
    run = brm_jax_run(src, inputs, us)
    @test isempty(run.module_constants["UNTRANSLATED"])
    for (rk, dataset, locations) in zip(backends, run.datasets, us)
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        mu = prepare_query(rk.model, rk_bound_plan(rk), (:mu,))
        for (u, p) in zip(locations, dataset["points"])
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            @test p["constrain_jit"]["mu"] ≈ Base.invokelatest(mu, u) atol=1e-11
            @test p["log_density"]["jit"] ≈ value atol=1e-11
            @test p["grad_eager"] ≈ gradient atol=1e-10 rtol=1e-10
            @test p["grad_args"]["ok"]
            @test p["grad_args"]["value"] ≈ gradient atol=1e-10 rtol=1e-10
        end
    end
end

@testset "multigroup HSGP JAX density and reverse gradient" begin
    check_multigroup_jax(isotropic, [multigroup_data(2, 3, 8), multigroup_data(3, 5, 12)])
    check_multigroup_jax(anisotropic, [multigroup_data(3, 2, 15)])
end
