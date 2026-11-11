# Exercise the published OU declaration through its retained artifact, without fitting.
# Run: BRM_JAX_REQUIRE=1 julia --project=test test/docs_ou_backends.jl
using Test, Random
using BayesianRegressionModels, StanBlocks, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, ADTypes, LogDensityProblems

include("jax_module_driver.jl")

module DocsOU
include(joinpath(@__DIR__, "..", "research", "ema_ctsem", "docs_model.jl"))
end

# Independent exact-transition oracle: stationary initialization, then the
# conditional mean and variance over each observed interval.
function ou_reference(dt, z, mu, rate, stationary_sd)
    state = [mu + stationary_sd * first(z)]
    for j in 2:length(z)
        a = exp(-rate * dt[j])
        push!(state, mu * (1 - a) + a * last(state) +
            stationary_sd * sqrt(1 - a^2) * z[j])
    end
    state
end

function ou_density(layout, data, u)
    c = ReactiveKernelsPPL.constrain(layout, u)
    state = ou_reference(data.dt, c.z, c.mu, c.rate, c.stationary_sd)
    logpdf(Normal(0.0, 1.0), c.mu) +
        logpdf(LogNormal(log(0.8), 0.5), c.rate) +
        logpdf(LogNormal(log(0.6), 0.5), c.stationary_sd) +
        logpdf(LogNormal(log(0.15), 0.4), c.sigma) +
        sum(logpdf.(Normal(0.0, 1.0), c.z)) +
        sum(logpdf.(Normal.(state, c.sigma), data.y)) +
        log(c.rate) + log(c.stationary_sd) + log(c.sigma)
end

function ou_finite_difference(f, u)
    map(eachindex(u)) do i
        h = 1e-5 * max(1, abs(u[i]))
        plus, minus = copy(u), copy(u)
        plus[i] += h
        minus[i] -= h
        (f(plus) - f(minus)) / (2h)
    end
end

contexts = []

@testset "docs OU: exact transition on Stan and native" begin
    @test Base.get_extension(BayesianRegressionModels,
        :BayesianRegressionModelsReactiveKernelsExt) !== nothing
    @test DocsOU.ou_path(zeros(4), ones(4), 0.0, 1.0, 1.0) == ones(4)
    @test DocsOU.ou_path([0.0], [-0.5], 0.3, 0.7, 0.4) ≈ [0.1]

    for n in (6, 40)
        data = DocsOU.state_space_data(; n)
        saved = deepcopy(data)
        brmi = DocsOU.state_space_model(data)
        artifact = BayesianRegressionModels.emit_rk_artifact(brmi; case_id="docs-ou-$n")
        plan = BayesianRegressionModels.rk_translate_artifact(artifact)
        built = build_kernel(plan)
        rk = RKBRMI(brmi, artifact.plan, built)
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        stan = stan_code(SBBRMI(brmi; mod=DocsOU))
        @test StanBlocks.stanc_check(stan; warn_pedantic=false).ok
        us = [zeros(built.layout.total),
            0.2 .* randn(Xoshiro(144), built.layout.total),
            0.4 .* randn(Xoshiro(145), built.layout.total)]
        native = map(us) do u
            c = ReactiveKernelsPPL.constrain(built.layout, u)
            state = ou_reference(data.dt, c.z, c.mu, c.rate, c.stationary_sd)
            @test DocsOU.ou_path(data.dt, c.z, c.mu, c.rate, c.stationary_sd) ≈ state
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            oracle = u -> ou_density(built.layout, data, u)
            @test value ≈ oracle(u) rtol=1e-12 atol=1e-12
            @test gradient ≈ ou_finite_difference(oracle, u) rtol=1e-6 atol=1e-6
            (; value, gradient, state)
        end
        @test isequal(data, saved)
        push!(contexts, (; n, data, saved, rk, built, plan, us, native))
    end
end

@testset "docs OU: retained-artifact JAX parity" begin
    for (; data, saved, rk, built, plan, us, native) in contexts
        src = jax_source(built, plan; values=(:state,))
        if brm_jax_require_python()
            run = brm_jax_run(src, [brm_jax_inputs(rk, src)], [us])
            @test run.x64_after_import
            @test isempty(run.module_constants["UNTRANSLATED"])
            for (p, expected) in zip(run.datasets[1]["points"], native)
                @test p["log_density"]["eager"] ≈ expected.value rtol=1e-12 atol=1e-12
                @test p["log_density"]["jit"] ≈ expected.value rtol=1e-12 atol=1e-12
                @test vec(p["grad"]) ≈ expected.gradient rtol=1e-10 atol=1e-10
                @test vec(p["constrain_jit"]["state"]) ≈ expected.state rtol=1e-12 atol=1e-12
            end
        end
        @test isequal(data, saved)
    end
end
