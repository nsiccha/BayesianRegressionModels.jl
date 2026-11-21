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
per_cell_concentration(data) = @brm data begin
    @plate for i in eachindex(y)
        theta ~ Dirichlet(alpha[i])
        y[i] ~ Normal(theta[1], 0.2)
    end
end

@testset "JAX Dirichlet cell simplexes in native coordinates" begin
    y = [0.2, 0.3]
    a = 1.5
    alpha = [0.8, 1.5, 2.0]
    mixed_y = [0.2, 0.3, 0.4]
    mixed_alpha = [[1.2], alpha, [0.9, 1.2]]
    for (model, observations, concentrations) in (
        (symmetric_cell((; y, n=3, a)), y, fill(fill(a, 3), length(y))),
        (asymmetric_cell((; y, alpha)), y, fill(alpha, length(y))),
        (symmetric_cell((; y, n=1, a)), y, fill([a], length(y))),
        (per_cell_concentration((; y=mixed_y, alpha=mixed_alpha)), mixed_y, mixed_alpha))
        rk = RKBRMI(model)
        plan = rk_bound_plan(rk)
        src = jax_source(rk.model, plan)
        sizes = length.(concentrations)
        d = sum(sizes .- 1)
        @test rk.model.layout.total == d
        u = [0.3 * (-1)^j * j for j in 1:d]
        queries = [prepare_query(rk.model, plan, preset) for preset in
            (:sampler, :prior, :likelihood, :log_jacobian)]
        thetas = ReactiveKernelsPPL.constrain(rk.model.layout, u).theta
        @test length.(thetas) == sizes && all(w -> sum(w) ≈ 1, thetas)
        lp = sum(logpdf(Dirichlet(α), w) + sum(log, w) for (α, w) in zip(concentrations, thetas))
        lp += sum(logpdf(Normal(w[1], 0.2), obs) for (w, obs) in zip(thetas, observations))
        @test Base.invokelatest(first(queries), u) ≈ lp atol=2e-12
        brm_jax_require_python() || continue
        run = brm_jax_run(src, [rk_bound_data(rk, src)], [[u]])
        @test isempty(run.module_constants["UNTRANSLATED"])
        p = only(only(run.datasets)["points"])
        constrained = p["constrain_jit"]
        @test constrained["theta_lengths"] == sizes
        @test all(constrained["theta"][i, 1:sizes[i]] ≈ thetas[i] for i in eachindex(sizes))
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
