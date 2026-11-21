using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Distributions
include("jax_module_driver.jl")

symmetric_cell(data) = @brm data begin
    @plate for i in eachindex(y)
        theta ~ Dirichlet(n, a)
        y[i] ~ Normal(theta[1], 0.2)
    end
end
asymmetric_cell(data) = @brm data begin
    @plate for i in eachindex(y)
        theta ~ Dirichlet(alpha)
        y[i] ~ Normal(theta[1], 0.2)
    end
end

@testset "JAX Dirichlet cell simplexes in native coordinates" begin
    for (model, data, n) in (
        (symmetric_cell, (; y=[0.2, 0.3], n=3, a=1.5), 3),
        (asymmetric_cell, (; y=[0.2, 0.3], alpha=[0.8, 1.5, 2.0]), 3),
        (symmetric_cell, (; y=[0.2, 0.3], n=1, a=1.5), 1))
        rk = RKBRMI(model(data))
        plan = rk_bound_plan(rk)
        src = jax_source(rk.model, plan)
        d = length(data.y) * (n - 1)
        @test rk.model.layout.total == d
        u = [0.3 * (-1)^j * j for j in 1:d]
        queries = [prepare_query(rk.model, plan, preset) for preset in
            (:sampler, :prior, :likelihood, :log_jacobian)]
        thetas = ReactiveKernelsPPL.constrain(rk.model.layout, u).theta
        @test all(w -> length(w) == n && sum(w) ≈ 1, thetas)
        alpha = hasproperty(data, :alpha) ? data.alpha : fill(data.a, n)
        lp = sum(w -> logpdf(Dirichlet(alpha), w) + sum(log, w), thetas)
        lp += sum(logpdf(Normal(w[1], 0.2), y) for (w, y) in zip(thetas, data.y))
        @test Base.invokelatest(first(queries), u) ≈ lp atol=2e-12
        brm_jax_require_python() || continue
        run = brm_jax_run(src, [rk_bound_data(rk, src)], [[u]])
        @test isempty(run.module_constants["UNTRANSLATED"])
        p = only(only(run.datasets)["points"])
        @test p["constrain_jit"]["theta"] ≈ permutedims(reduce(hcat, thetas))
        for (f, q) in zip(("log_density", "log_prior", "log_likelihood", "log_jacobian"), queries)
            @test p[f]["jit"] ≈ Base.invokelatest(q, u) atol=2e-12
        end
        # Compare the compiled reverse pass against differences of the native target.
        gradient = similar(u)
        for j in eachindex(u)
            hi, lo = copy(u), copy(u)
            hi[j] += 1e-5
            lo[j] -= 1e-5
            gradient[j] = (Base.invokelatest(first(queries), hi) - Base.invokelatest(first(queries), lo)) / 2e-5
        end
        @test vec(p["grad"]) ≈ gradient atol=2e-8
    end
end
