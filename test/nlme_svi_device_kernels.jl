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
    end
    cases = [
        :global_draw => (μ, ρ, B, ε, [0.3]),
        :global_draw => (μ, ρ, zeros(2, 0), ε, Float64[]),
        :antithetic_pair => (μ, ε),
        :unit_offset => ([2.0 9.0; 0.5 1.0], ε),
        :whitened_difference => ([2.0 9.0; 0.5 1.0], ε),
        :work_weight => ([1, 2], 3, [0.2, 0.3]),
        :global_gradient => (μ, ε, [0.3], exp.(ρ), [0.4, 0.9], B),
        :global_entropy => (ρ, B),
        :global_entropy => (ρ, zeros(2, 0)),
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
        :curvature_axis => (zero(ε), ε),
        :pair_stack => ([1.0 0.0; 0.2 1.1], [1.0 0.2; 0.2 1.25],
            2.0, [1.0, 0.0], 0.75, 1.0),
        :orthogonal_stack => (Matrix{Float64}(I, 2, 2), [1.0 0.2; 0.2 1.25],
            1.4, 0.75, 1.0),
        :dense_update => (μ, Matrix{Float64}(I, 2, 2),
            [1.0 0.0; 0.1 1.0; 1.1 -0.2; 0.3 0.9], ε, 0.75, 1.0),
        :adam => (μ, [0.2, -0.1], [0.3, 0.9], ε, 0.9, 0.999, 0.19, 0.001999, 0.1),
    ]
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
            for (i, (name, args)) in enumerate(cases)
                signature = BRM._BRM_SVI_KERNEL_DEFINITIONS[name].args[1]
                native = getproperty(BRM, first(signature.args))
                expected = native(args...)
                actual = prepare(kernels[name])(args...)
                if name === :global_entropy
                    # Native Julia and RK may use different Cholesky implementations.
                    @test all(isapprox.(actual, expected; rtol=1e-12, atol=1e-12))
                else
                    @test actual == expected
                end
                run_jax || continue
                ports = Tuple(signature.args[2:end])
                source = jax_source(kernels[name]; have=ports, functions=(evaluate=:result,),
                    types=NamedTuple{ports}(typeof.(args)))
                mod = "kernel_$i"
                write(joinpath(dir, mod * ".py"), source)
                println(io, "check(", repr(mod), ", ", _py(args), ", ", _py(expected), ")")
            end
        end
        run_jax && @test success(`$BRM_JAX_PYTHON $script`)
    end
end
