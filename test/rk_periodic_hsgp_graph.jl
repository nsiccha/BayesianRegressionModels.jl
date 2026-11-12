# Periodic HSGP preprocessing and spectra are visible RK graphs. Run with
# RK_JAX_PYTHON=<python with jax> BRM_JAX_REQUIRE=1 julia --project=test test/rk_periodic_hsgp_graph.jl
using Test, Random
using BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, LogDensityProblems
include("testset_filter.jl")
include("jax_module_driver.jl")

const BRM = BayesianRegressionModels
pkgversion(ReactiveKernels) >= v"0.1.198" ||
    error("periodic HSGP JAX preparation needs ReactiveKernels 0.1.198 host helpers; run test/setup_env.jl")
periodic_model = @brm begin
    mu ~ 0 + hsgp(x; k=K, cov=:periodic, period=P)
    y ~ Normal(mu, 1.0)
end
anchored_periodic_model = @brm begin
    mu ~ 0 + hsgp(x; k=K, cov=:periodic, period=P, anchor=A)
    y ~ Normal(mu, 1.0)
end
masked_periodic_model = @brm begin
    mu ~ 0 + hsgp(x; k=K, cov=:periodic, period=P, anchor=A, rows=carried)
    y ~ Normal(mu, 1.0)
end

function periodic_data(K, P, n)
    x = collect(range(-0.7, 1.9; length=n))
    (; x, y=0.2 .* sin.(x), K, P, A=0.1P, carried=isodd.(1:n))
end

function check_periodic_graph(builder, anchor, mask, datasets)
    # One module evaluates new row counts, harmonic counts, periods and
    # anchors; these are inputs, not constants baked into the source.
    @info "periodic HSGP graph case" anchor mask modes=getproperty.(datasets, :K)
    originals = deepcopy(datasets)
    backends = RKBRMI.(builder.(datasets))
    src = jax_source(first(backends).model, rk_bound_plan(first(backends)); values=(:mu,))
    @test src isa JAXSource
    texts = [BRM._rk_program_statements(BRM._rk_emit_ast(rk.plan)) for rk in backends]
    @test allequal(texts)
    us = [[zeros(rk.model.layout.total), 0.3 .* randn(Xoshiro(5), rk.model.layout.total)]
        for rk in backends]
    references = map(backends, datasets, us) do rk, data, points
        plan = rk_bound_plan(rk)
        phi_query = prepare_query(rk.model, plan, (:hsgp_x_PHI,))
        floor_query = data.K == 1 ? nothing : prepare_query(rk.model, plan, (:hsgp_x_rho_floor,))
        value_query = prepare_query(rk.model, plan, (:mu,))
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        # The retained helper is the old native path. The basis is exact;
        # compiled scan/spectral arithmetic can differ in the last bits.
        old = BRM._brm_hsgp_basis_state((data.x,), (data.K,), :periodic, true, data.P;
            anchor=anchor ? (data.A,) : nothing, mask=mask ? data.carried : nothing)
        @test Base.invokelatest(phi_query, first(points)) == old.PHI
        floor_query === nothing || @test Base.invokelatest(floor_query, first(points)) ≈ old.rho_lower rtol=4eps(Float64)
        result = map(points) do u
            c = ReactiveKernelsPPL.constrain(rk.model.layout, u).hsgp_x
            scales = BRM.brm_hsgp_periodic_sqrt_spd(old.harmonics, c.sigma, c.rho_iso)
            expected_mu = old.PHI * (scales .* c.beta_raw)
            @test Base.invokelatest(value_query, u) ≈ expected_mu rtol=8eps(Float64) atol=8eps(Float64)
            lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            @test isfinite(lp) && all(isfinite, gradient)
            (; lp, gradient, mu=expected_mu)
        end
        result
    end
    @test isequal(datasets, originals)
    if brm_jax_require_python()
        inputs = [brm_jax_inputs(rk, src) for rk in backends]
        run = brm_jax_run(src, inputs, us)
        @info "periodic HSGP JAX parity" anchor mask versions=run.versions artifacts=run.dir
        @test isempty(run.module_constants["UNTRANSLATED"])
        for (dataset, expected) in zip(run.datasets, references),
                (p, n) in zip(dataset["points"], expected)
            @test p["log_density"]["eager"] ≈ n.lp rtol=1e-12 atol=1e-12
            @test p["log_density"]["jit"] ≈ n.lp rtol=1e-12 atol=1e-12
            @test vec(p["grad"]) ≈ n.gradient rtol=1e-10 atol=1e-10
            @test vec(p["grad_eager"]) ≈ n.gradient rtol=1e-10 atol=1e-10
            @test vec(p["constrain_jit"]["mu"]) ≈ n.mu rtol=1e-12 atol=1e-12
            @test p["log_density"]["jit_args"]["ok"]
            @test p["grad_args"]["ok"]
            @test p["log_density"]["jit_args"]["value"] ≈ n.lp rtol=1e-12 atol=1e-12
            @test vec(p["grad_args"]["value"]) ≈ n.gradient rtol=1e-10 atol=1e-10
        end
    end
end

const PERIODIC_DATASETS = [periodic_data(4, 2.0, 5), periodic_data(2, 3.7, 7), periodic_data(8, 0.9, 4)]
@stestset "periodic HSGP graph plain" begin
    check_periodic_graph(periodic_model, false, false, PERIODIC_DATASETS)
end
@stestset "periodic HSGP graph anchored" begin
    check_periodic_graph(anchored_periodic_model, true, false, PERIODIC_DATASETS)
end
@stestset "periodic HSGP graph masked" begin
    check_periodic_graph(masked_periodic_model, true, true, PERIODIC_DATASETS)
end
@stestset "periodic HSGP graph single harmonic" begin
    check_periodic_graph(periodic_model, false, false, [periodic_data(1, 3.7, 7)])
end
