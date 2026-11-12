# Public documentation model, unchanged data/priors, no fit.
# Run with BRM_JAX_REQUIRE=1 and RK_JAX_PYTHON naming a JAX interpreter.
using Test, Random, LinearAlgebra
using BayesianRegressionModels, StanBlocks, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, ADTypes, LogDensityProblems

include("jax_module_driver.jl")

module DocsWastewater
include(joinpath(@__DIR__, "..", "research", "wastewater", "cdc_ww_inference.jl"))
include(joinpath(@__DIR__, "..", "research", "wastewater", "data.jl"))
end

module StanOnlyWastewater
const source_path = joinpath(@__DIR__, "..", "research", "wastewater", "cdc_ww_inference.jl")
include_string(@__MODULE__, replace(read(source_path, String),
    "StanBlocks.@deffun @juliacompat" => "StanBlocks.@deffun"), source_path)
include(joinpath(@__DIR__, "..", "research", "wastewater", "data.jl"))
end

@testset "docs wastewater: Julia helper semantics" begin
    ww = DocsWastewater
    x, idx = [0.1, -0.3, 0.7], [3, 1, 3, 2]
    saved = deepcopy((x, idx))
    @test ww.gather_exp(x, idx) ≈ exp.(x[idx])
    @test isequal((x, idx), saved)
    @test_throws MethodError StanOnlyWastewater.gather_exp(x, idx)
    @test ww.dar_logru(0.2, [1.0, -2.0, 0.5], 0.5, 0.1, 4) ≈
        [0.2, 0.3, 0.15, 0.125]
    @test ww.ar1_dev([0.4, -0.5, 1.0], 0.6, 0.2) ≈ [0.1, -0.04, 0.176]
    @test ww.viral_shedding_trajectory(2.0, 4.0, 6.0, 6) ≈
        [100.0, 10000.0, 1000.0, 100.0, 10.0, 1.0] ./ 11211.0
    # The load floor makes viral_peak=0 a kink. For t_peak=1, duration=2
    # and six days, the first normalized weight has these one-sided slopes.
    h = 1e-7
    first_weight(v) = first(ww.viral_shedding_trajectory(1.0, v, 2.0, 6))
    @test (first_weight(h) - first_weight(0.0)) / h ≈ log(10.0) * 5 / 36 rtol=1e-6
    @test (first_weight(0.0) - first_weight(-h)) / h ≈ log(10.0) * 5 / 12 rtol=1e-6
    # One-day generation interval and zero feedback: each day doubles.
    @test ww.renewal_feedback(fill(log(2.0), 5), [1.0], 0.0, 0.01, 0.0, 1) ≈
        [0.01, 0.02, 0.04, 0.08, 0.16]
    @test ww.renewal_from_first_observed(fill(log(2.0), 5), [1.0],
        0.0, 0.01, 0.0, 1) ≈ [0.01, 0.02, 0.04, 0.08, 0.16]
    incidence, weights = [1.0, 2.0, 4.0], [0.75, 0.25]
    @test ww.shed_convolve(incidence, weights) ≈ [0.75, 1.75, 3.5]
    @test ww.delay_convolve(incidence, weights) ≈ [0.75, 1.75, 3.5]
    matrix = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    @test ww.wsum(matrix, weights) ≈ [1.25, 3.25, 5.25]
    @test ww.weekly_expand(x, idx) == x[idx]
    @test ww.weekly_expand_columns(matrix, idx) == matrix[idx, :]
    @test ww.take_window(incidence, 2, 2) == [2.0, 4.0]
    @test ww.scale_simplex(weights, 4.0) == [3.0, 1.0]
    @test ww.ww_expected_log(matrix, [1.0], [2, 3], [1, 2], [1, 1],
        [0.0], 2.0, 10.0) ≈ log.(10.0 .* ([3.0, 6.0] .+ 1e-8))
    # One population, one-day lag, 50% event probability and unit weekday effect.
    @test ww.count_interval_mean(reshape(incidence, :, 1), ones(1, 1), ones(1, 1),
        zeros(3, 1), [1, 2], [1, 3], [1, 1], [1, 2, 3], ones(7), [10.0]) ≈
        [5.0, 30.0] .+ 1e-8
    @test ww.hospital_daily_mean(incidence, zeros(3), [1, 2, 3], ones(7), 10.0) ≈
        [5.0, 10.0, 20.0]
    @test ww.ihr_scale(incidence, zeros(3), zeros(3), 10.0) ≈ [5.0, 10.0, 20.0]
    # The same helpers consume Julia's per-cell collection as Stan's columns.
    cells = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
    cells_saved = deepcopy(cells)
    @test ww.wsum(cells, weights) ≈ [1.25, 3.25, 5.25]
    @test ww.weekly_expand_columns(cells, idx) == matrix[idx, :]
    @test ww.ww_expected_log(cells, [1.0], [2, 3], [1, 2], [1, 1],
        [0.0], 2.0, 10.0) ≈ log.(10.0 .* ([3.0, 6.0] .+ 1e-8))
    @test ww.count_interval_mean([incidence], ones(1, 1), ones(1, 1),
        zeros(3, 1), [1, 2], [1, 3], [1, 1], [1, 2, 3], ones(7), [10.0]) ≈
        [5.0, 30.0] .+ 1e-8
    @test isequal(cells, cells_saved)
end

data = DocsWastewater.wastewater_public_data()
saved = deepcopy(data)
brmi = DocsWastewater.wastewater_public_model(data)
artifact = BayesianRegressionModels.emit_rk_artifact(brmi; case_id="docs-wastewater")
plan = BayesianRegressionModels.rk_translate_artifact(artifact)
built = build_kernel(plan)
rk = RKBRMI(brmi, artifact.plan, built)
# Zero sets viral_peak=0, where the clamped shedding curve has no derivative.
# Central differences must stay on smooth points of the unchanged model.
us = [fill(0.1, built.layout.total), 0.05 .* randn(Xoshiro(146), built.layout.total),
    0.1 .* randn(Xoshiro(147), built.layout.total)]
native = []

@testset "docs wastewater: Stan preservation and native gradients" begin
    stan = stan_code(SBBRMI(brmi; mod=DocsWastewater))
    @test StanBlocks.stanc_check(stan; warn_pedantic=false).ok
    @test stan == stan_code(SBBRMI(StanOnlyWastewater.wastewater_public_model(data);
        mod=StanOnlyWastewater))
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for (point, u) in enumerate(us)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @info "Wastewater native point" point value finite_gradient=all(isfinite, gradient)
        @test isfinite(value)
        @test all(isfinite, gradient)
        for seed in (148, 149)
            direction = normalize(randn(Xoshiro(seed), length(u)))
            h = 1e-5
            finite_difference = (LogDensityProblems.logdensity(problem, u + h * direction) -
                LogDensityProblems.logdensity(problem, u - h * direction)) / (2h)
            @test dot(gradient, direction) ≈ finite_difference rtol=1e-5 atol=1e-5
        end
        push!(native, (; value, gradient))
    end
    @test isequal(data, saved)
end

@testset "docs wastewater: retained-artifact JAX parity" begin
    src = jax_source(built, plan)
    if brm_jax_require_python()
        run = brm_jax_run(src, [brm_jax_inputs(rk, src)], [us])
        @info "Wastewater JAX acceptance" versions=JSON.json(run.versions) artifacts=run.dir
        @test run.x64_after_import
        @test isempty(run.module_constants["UNTRANSLATED"])
        for (p, expected) in zip(run.datasets[1]["points"], native)
            for mode in ("eager", "jit")
                @test p["log_density"][mode] ≈ expected.value rtol=1e-10 atol=1e-8
            end
            @test p["log_density"]["jit_args"]["ok"]
            @test p["log_density"]["jit_args"]["value"] ≈ expected.value rtol=1e-10 atol=1e-8
            for mode in ("grad", "grad_eager")
                @test vec(p[mode]) ≈ expected.gradient rtol=1e-8 atol=1e-6
            end
            @test p["grad_args"]["ok"]
            @test vec(p["grad_args"]["value"]) ≈ expected.gradient rtol=1e-8 atol=1e-6
        end
    end
    @test isequal(data, saved)
end
