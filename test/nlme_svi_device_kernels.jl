# RK_JAX_PYTHON selects a provisioned interpreter. On an allocated GPU
# lane, BRM_SVI_TEST_GPU=1 also checks the placement of every result.
using Test, LinearAlgebra, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
include("jax_module_driver.jl")
const BRM = BayesianRegressionModels

@testset "SVI kernels agree in Julia and compiled JAX" begin
    kernels = BRM._rk_svi_kernels()
    μ, ρ, ε = [0.1, -0.4], [-0.7, 0.2], [1.2, -0.5]
    B = reshape([0.4, -0.2], 2, 1)
    for loading in (zeros(2, 0), B, [1.0 0.4 -0.3; -0.2 0.7 0.1])
        covariance = Diagonal(exp.(2 .* ρ)) + loading * loading'
        diagonal, gradient = prepare(kernels.global_entropy)(ρ, loading)
        @test diagonal ≈ exp.(2 .* ρ) .* diag(inv(covariance))
        @test gradient ≈ covariance \ loading
        @test prepare(kernels.global_logdet)(ρ, loading) ≈ logdet(covariance)
    end
    precision = [2.0 0.4; 0.4 1.6]
    reference = inv(Symmetric(precision))
    root = Matrix(cholesky(Symmetric(precision)).L)
    @test prepare(kernels.unit_covariance)(precision) ≈ reference
    @test prepare(kernels.dense_logsd)(precision) ≈ log.(diag(reference)) / 2
    @test prepare(kernels.unit_entropy)(root) ≈ logdet(reference) / 2
    @test prepare(kernels.marginal_logsd)(ρ[1], ε) ≈ log(exp(2ρ[1]) + sum(abs2, ε))/2
    @test prepare(kernels.weighted_average)(μ, ε, 0.6) ≈ 0.4μ + 0.6ε
    @test prepare(kernels.running_average)(μ, ε, 4) ≈ (3μ + ε)/4
    @test prepare(kernels.centering_control)(-2.0) ≈ 1/(1+exp(2.0))
    lambda, axis = prepare(kernels.curvature_axis)([1.0, 0.0], [-1.0, 1e-8])
    @test lambda ≈ 2.5e-17
    @test norm(axis) ≈ 1.0
    cases = [
        :global_draw => (μ, ρ, B, ε, [0.3]),
        :global_draw => (μ, ρ, zeros(2, 0), ε, Float64[]),
        :antithetic_pair => (μ, ε),
        :unit_offset => ([2.0 9.0; 0.5 1.0], ε),
        :whitened_difference => ([2.0 9.0; 0.5 1.0], ε),
        :work_weight => ([1, 2], 3, [0.2, 0.3]),
        :global_gradient => (μ, ε, [0.3], exp.(ρ), [0.4, 0.9], B),
        :narrow_geometry => (ρ, B),
        :narrow_geometry => (ρ, zeros(2, 0)),
        :narrow_geometry => ([0.0, log(1e-10)], B),
        :global_entropy => (ρ, B),
        :global_entropy => (ρ, zeros(2, 0)),
        :global_logdet => (ρ, B),
        :global_logdet => (ρ, zeros(2, 0)),
        :global_logdet => (Float64[], zeros(0, 0)),
        :unit_entropy => (root,),
        :unit_entropy => (zeros(0, 0),),
        :unit_covariance => (precision,),
        :unit_covariance => (zeros(0, 0),),
        :weighted_average => (precision, reference, 0.3),
        :running_average => (μ, ε, 4),
        :dense_logsd => (precision,),
        :dense_logsd => (zeros(0, 0),),
        :marginal_logsd => (ρ[1], ε),
        :marginal_logsd => (ρ[2], Float64[]),
        :centering_control => (0.25,),
        :centering_control => (-900.0,),
        :global_step_scale => (ρ, B, exp.(ρ), 0.5),
        :batch_clip => ([4.0, 9.0], 10.0),
        :clip_gradient => ([-1e4, 5.0], [4.0, 0.0], 0.9, 2.0),
        :sd_shrink => ([9.0, 4.0, 1.0], [2.0, Inf, 0.0]),
        :bound_sd => (ρ, B, [0.3, Inf]),
        :bound_sd => (ρ, zeros(2, 0), [0.3, Inf]),
        :precision_root => ([2.0 -0.2; 0.0 -3.0], sqrt(2.0)),
        :mean_trust => ([1.0 0.0; -0.2 3.0], [1.0, 2.0], 1.0),
        :curvature_axis => (ε, μ),
        :curvature_axis => (ε, -ε),
        :curvature_axis => ([1.0, 0.0], [-1.0, 1e-8]),
        :curvature_axis => (zero(ε), ε),
        :unit_design => ([0.2 -0.8; -0.7 1.1],),
        :unit_design => (zeros(2, 2),),
        :design_curvature => ([0.6 -1.0; 0.4 2.0], [0.9 -0.4; 0.4 0.9]),
        :positive_curvature => ([1.0 -0.4; -0.4 1.0],),
        :positive_curvature => ([-1.0 0.7; 0.7 2.0],),
        :orthogonal_gap => ([-0.5 0.3; 0.1 1.0], true),
        :orthogonal_gap => ([-0.5 0.3; 0.1 1.0], false),
        :pair_stack => ([1.0 0.0; 0.2 1.1], [1.0 0.2; 0.2 1.25],
            2.0, [1.0, 0.0], 0.75, 1.0),
        :orthogonal_stack => (Matrix{Float64}(I, 2, 2), [1.0 0.2; 0.2 1.25],
            1.4, 0.75, 1.0),
        :dense_update => (μ, Matrix{Float64}(I, 2, 2),
            [1.0 0.0; 0.1 1.0; 1.1 -0.2; 0.3 0.9], ε, 0.75, 1.0),
        :coupling_noise => (ε, [2, 1], [0.25]),
        :coupled_mean => (μ, B, [0.2]),
        :coupling_baseline => (1.0, ε),
        :coupling_baseline => (1.0, [1.0, NaN]),
        :coupling_baseline => (-Inf, ε),
        :coupling_direction => ([2.0 9.0; 0.5 1.0], ε),
        :coupling_variances => ([2.0 9.0; 0.5 1.0],),
        :coupling_row_update => ([0.3, 2.5], 7, ε, [3.0, 2.0]),
        :coupling_metric => ([2.0 0.5; 0.5 1.0], ε),
        :coupling_step => (B, ε, [0.4], 1.4, 0.5, 0.25),
        :coupling_step => (B, ε, [0.0], 1.4, 0.5, 0.25),
        :coupling_step => (B, ε, [0.4], 1.4, 0.0, 0.25),
        :adam => (μ, [0.2, -0.1], [0.3, 0.9], ε, 0.9, 0.999, 0.19, 0.001999, 0.1),
    ]
    for loading in (zeros(2, 0), zeros(2, 2), B, [0.4 -0.3; 0.2 0.7])
        eta = fill(0.25, size(loading, 2))
        covariance = Diagonal(exp.(2 .* ρ)) + loading * loading'
        offset = exp.(ρ) .* ε + loading * eta
        score = μ + covariance \ offset
        x = vcat(μ, ρ, vec(loading))
        gradient = vcat(ε, score .* exp.(ρ) .* ε, vec(score * eta'))
        @test prepare(kernels.structured_offset)(ρ, loading, ε, eta) ≈ offset
        @test prepare(kernels.structured_precision)(ρ, loading, μ) ≈ covariance \ μ
        @test prepare(kernels.structured_gradient)(ρ, loading, ε, μ, ε, eta) ≈ gradient
        @test prepare(kernels.structured_direction)(ρ, loading, ε) ≈ covariance * ε
        @test prepare(kernels.structured_variances)(ρ, loading) ≈ diag(covariance)
        @test prepare(kernels.structured_metric)(ρ, loading, ε) ≈ dot(ε, covariance \ ε)
        step = 4 .* ones(length(x))
        fraction = min(1, inv(sqrt(dot(step[1:2], covariance \ step[1:2]))), 0.5/4)
        size(loading, 2) > 0 && (fraction = min(fraction, 0.5 * minimum(exp.(ρ)) / 4))
        @test prepare(kernels.structured_trust)(ρ, loading, x, x + step) ≈ x + fraction * step
        # An independent dense preconditioner for the complete Adam move.
        # Its mean block is a square root of the covariance, in native order.
        scale = exp.(ρ)
        V = loading ./ scale
        transform = Matrix(Diagonal(vcat(ones(4), 0.7 .* repeat(scale, length(eta)))))
        transform[1:2, 1:2] .= Diagonal(scale) * sqrt(Symmetric(I + V * V'))
        for count in (0, 3)
            m, v = count == 0 ? (zero(x), zero(x)) : (copy(x), fill(0.2, length(x)))
            g = transform' * (100 .* gradient)
            bound = count == 0 ? g : clamp.(g, -sqrt.(v ./ (1 - 0.999^count)), sqrt.(v ./ (1 - 0.999^count)))
            next_m = 0.9 .* m + 0.1 .* bound
            next_v = 0.999 .* v + 0.001 .* bound.^2
            t = count + 1
            step = 0.2 .* (next_m ./ (1 - 0.9^t)) ./ (sqrt.(next_v ./ (1 - 0.999^t)) .+ 1e-8)
            actual = prepare(kernels.structured_adam)(ρ, loading, x, 100 .* gradient, m, v,
                count, 0.9, 0.999, 0.2, 0.7, 1.0)
            @test actual[1] ≈ x + transform * step
            @test actual[2] ≈ next_m
            @test actual[3] ≈ next_v
            @test actual[4:5] == (t, bound != g)
        end
        append!(cases, [
            :structured_offset => (ρ, loading, ε, eta),
            :structured_precision => (ρ, loading, μ),
            :structured_gradient => (ρ, loading, ε, μ, ε, eta),
            :structured_trust => (ρ, loading, x, x),
            :structured_trust => (ρ, loading, x, x + step),
            :structured_direction => (ρ, loading, ε),
            :structured_variances => (ρ, loading),
            :structured_metric => (ρ, loading, ε),
            :structured_adam => (ρ, loading, x, gradient, zero(x), zero(x), 0, 0.9, 0.999, 0.2, 0.7, 10.0),
            :structured_adam => (ρ, loading, x, 100 .* gradient, x, fill(0.2, length(x)), 3, 0.9, 0.999, 0.2, 0.7, 1.0),
        ])
    end
    # An initially zero loading has a unique zero basis. With other loadings,
    # compare the invariant Adam move above because eigenvector signs may differ.
    push!(cases, :structured_trust => (Float64[], zeros(0, 0), Float64[], Float64[]),
        :structured_sqrt_basis => (zeros(2, 1),),
        :structured_sqrt_action => (exp.(ρ), B, [0.2], ε, true),
        :structured_sqrt_action => (exp.(ρ), B, [0.2], ε, false))
    @test Set(first.(cases)) == Set(keys(kernels))
    run_jax = brm_jax_require_python()
    mktempdir() do dir
        script = joinpath(dir, "check.py")
        open(script, "w") do io
            println(io, """
                import os, importlib, jax, numpy as np
                jax.config.update("jax_enable_x64", True)
                def check(name, args, expected):
                    kernel = importlib.import_module(name)
                    actual = jax.tree.leaves(jax.jit(kernel.evaluate)(*args))
                    for value, ref in zip(actual, jax.tree.leaves(expected), strict=True):
                        np.testing.assert_equal(value.shape, np.shape(ref))
                        np.testing.assert_allclose(value, ref, atol=1e-12, rtol=1e-12)
                        if os.environ.get("BRM_SVI_TEST_GPU") == "1":
                            assert value.device.platform == "gpu"
                """)
            @testset "$(case.first), case $i" for (i, case) in enumerate(cases)
                name, args = case
                signature = BRM._BRM_SVI_KERNEL_DEFINITIONS[name].args[1]
                native = getproperty(BRM, first(signature.args))
                expected = native(args...)
                actual = prepare(kernels[name])(args...)
                if name in (:global_entropy, :global_logdet, :unit_covariance, :dense_logsd,
                        :structured_precision, :structured_gradient, :structured_metric, :structured_trust, :structured_adam)
                    # Native Julia and RK may use different Cholesky implementations.
                    @test all(isapprox.(actual, expected; rtol=1e-12, atol=1e-12))
                else
                    @test actual == expected
                end
                if run_jax
                    ports = Tuple(signature.args[2:end])
                    source = jax_source(kernels[name]; have=ports, functions=(evaluate=:result,),
                        types=NamedTuple{ports}(typeof.(args)))
                    mod = "kernel_$i"
                    write(joinpath(dir, mod * ".py"), source)
                    println(io, "check(", repr(mod), ", ", _py(args), ", ", _py(expected), ")")
                end
            end
        end
        run_jax && @test success(pipeline(`$BRM_JAX_PYTHON $script`; stdout, stderr))
    end
end
