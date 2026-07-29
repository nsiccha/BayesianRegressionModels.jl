include(joinpath(@__DIR__, "adaptive_centering_warmuphmc.jl"))

using Enzyme
using DifferentiationInterface: AutoEnzyme
using LogDensityProblems
using StanBlocks

function enzyme_gradient_stress(problem, initial; n=5_000)
    x = copy(initial)
    checksum = zero(eltype(x))
    for i in 1:n
        x[1] = initial[1] + 1e-6 * (i % 11)
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, x)
        checksum += lp + gradient[1]
    end
    checksum
end

struct AdaptiveCenteringQuadraticTarget
    dimension::Int
end

LogDensityProblems.dimension(target::AdaptiveCenteringQuadraticTarget) =
    target.dimension
LogDensityProblems.capabilities(::Type{AdaptiveCenteringQuadraticTarget}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(::AdaptiveCenteringQuadraticTarget, x) =
    -sum(abs2, x) / 2
LogDensityProblems.logdensity_and_gradient(
    target::AdaptiveCenteringQuadraticTarget, x,
) = (LogDensityProblems.logdensity(target, x), -x)

function synthetic_adaptive_block(K, G)
    n_cholesky = K * (K - 1) ÷ 2
    ranef = BayesianRegressionModels.RanefBlock(
        :r_mu_subject, :ranef_correlated, :subject, nothing, nothing,
        string.(1:G), K, G, :r_mu_subject_z_flat, true,
    )
    AdaptiveCenteringBlock(
        ranef,
        0.0,
        reshape((n_cholesky + K + 1):(n_cholesky + K + K * G), K, G),
        collect(1:n_cholesky),
        collect((n_cholesky + 1):(n_cholesky + K)),
    )
end

@testset "small-block pullback is exact, bounded, and GC-stable" begin
    @test !isnothing(Base.get_extension(
        BayesianRegressionModels,
        :BayesianRegressionModelsWarmupHMCEnzymeExt,
    ))
    backend = AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    )
    for (K, G, allocation_limit) in ((1, 6, 4_096), (2, 18, 16_384))
        block = synthetic_adaptive_block(K, G)
        state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
        @test state isa AC_EXT.BRMAdaptiveCenteringState{true}
        controls = collect(range(0.17, 0.83, length=K * G))
        set_adaptive_sources!(state, ir, controls)
        reference_ir = materialized_reparametrizer(state, ir)
        x = collect(range(-0.63, 0.81, length=maximum(vec(block.effects))))
        if !isempty(block.cholesky_free)
            x[block.cholesky_free] .= -0.27
        end
        x[block.log_scales] .= K == 1 ? [-0.31] : [-0.31, 0.44]
        target = AdaptiveCenteringQuadraticTarget(length(x))
        wrapped = WarmupHMC.ReparametrizedProblem(ir, target, backend)
        reference = WarmupHMC.ReparametrizedProblem(reference_ir, target, backend)

        actual = LogDensityProblems.logdensity_and_gradient(wrapped, x)
        expected = LogDensityProblems.logdensity_and_gradient(reference, x)
        @test isequal(actual, expected)
        allocated = @allocated LogDensityProblems.logdensity_and_gradient(wrapped, x)
        reference_allocated =
            @allocated LogDensityProblems.logdensity_and_gradient(reference, x)
        @test allocated <= allocation_limit
        @test allocated < reference_allocated
        @test isfinite(enzyme_gradient_stress(wrapped, x))
    end
end

@testset "two-term BridgeStan target is bit-exact" begin
    sb = SBBRMI(slope_builder(df); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    block = only(adaptive_centering_blocks(sb, unc_names))
    @test block.ranef.n_terms == 2

    state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
    controls = collect(range(0.17, 0.83, length=length(block.effects)))
    set_adaptive_sources!(state, ir, controls)
    materialized_ir = materialized_reparametrizer(state, ir)

    x = zeros(length(unc_names))
    x[only(block.cholesky_free)] = -0.27
    x[block.log_scales] .= [-0.31, 0.44]
    x[vec(block.effects)] .=
        collect(range(-0.63, 0.81, length=length(block.effects)))
    backend = AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    )
    wrapped = WarmupHMC.ReparametrizedProblem(ir, problem, backend)
    materialized = WarmupHMC.ReparametrizedProblem(
        materialized_ir, problem, backend,
    )

    actual = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    expected = LogDensityProblems.logdensity_and_gradient(materialized, x)
    @test isequal(actual, expected)
end

@testset "BridgeStan target differentiates through the mixed correlated frame" begin
    sb = SBBRMI(builder(df); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    block = only(adaptive_centering_blocks(sb, unc_names))
    state, ir = AC_EXT._adaptive_centering_reparametrizer([block])
    controls = [0.2, 0.6, 0.9, 0.8, 0.1, 0.5]
    set_adaptive_sources!(state, ir, controls)

    x = zeros(length(unc_names))
    x[block.cholesky_free] .= [0.2, -0.4, 0.7]
    x[block.log_scales] .= log.([0.5, 1.5, 3.0])
    x[vec(block.effects)] .=
        collect(range(-0.8, 0.9, length=length(block.effects)))
    backend = AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    )
    reparametrized = WarmupHMC.ReparametrizedProblem(ir, problem, backend)
    materialized_ir = materialized_reparametrizer(state, ir)
    materialized = WarmupHMC.ReparametrizedProblem(
        materialized_ir, problem, backend,
    )

    lp, gradient = LogDensityProblems.logdensity_and_gradient(reparametrized, x)
    materialized_lp, materialized_gradient =
        LogDensityProblems.logdensity_and_gradient(materialized, x)
    @test isequal(lp, materialized_lp)
    @test isequal(gradient, materialized_gradient)
    ljac, model_position = ir(x)
    inner_lp, _ = LogDensityProblems.logdensity_and_gradient(problem, model_position)
    @test lp ≈ ljac + inner_lp atol=2e-12
    @test isfinite(lp)
    @test all(isfinite, gradient)

    step = 1e-5
    finite_difference = [begin
        plus, minus = copy(x), copy(x)
        plus[i] += step
        minus[i] -= step
        (LogDensityProblems.logdensity(reparametrized, plus) -
         LogDensityProblems.logdensity(reparametrized, minus)) / (2step)
    end for i in eachindex(x)]
    @test gradient ≈ finite_difference atol=2e-5 rtol=2e-5

    # One Enzyme gradient stayed green after bd9ca2d, but sustained calls through
    # its recursive K<=4 accessor corrupted Julia's GC after about 1,750 calls.
    # Keep this above the measured threshold so that process-level failure cannot
    # hide behind the otherwise-correct single-gradient assertion.
    @test isfinite(enzyme_gradient_stress(reparametrized, x))

    wrapped = adaptive_centering_problem(sb, problem, backend)
    wrapped_ir = WarmupHMC.reparametrizer(wrapped)
    wrapped_state = wrapped_ir.pairs[1][2].args[1].state
    plain = LogDensityProblems.logdensity_and_gradient(problem, zeros(length(x)))
    adaptive =
        LogDensityProblems.logdensity_and_gradient(wrapped, zeros(length(x)))
    @test adaptive[1] ≈ plain[1] atol=2e-12
    @test adaptive[2] ≈ plain[2] atol=2e-12
    @test WarmupHMC.candidate_scoring_plan(wrapped) isa
          WarmupHMC.CandidateScoringPlan

    restored = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    wrapped_state.sources .= -1.0
    WarmupHMC.restore_reparam_sources!(
        wrapped,
        [idx => WarmupHMC.PartiallyCentered(c)
         for ((idx, _), c) in zip(wrapped_ir.pairs, restored)],
    )
    @test [value.source.c for (_, value) in wrapped_ir.pairs] == restored
    @test wrapped_state.sources == restored

    copied = deepcopy(wrapped)
    copied_ir = WarmupHMC.reparametrizer(copied)
    copied_state = copied_ir.pairs[1][2].args[1].state
    @test copied_state !== wrapped_state
    copied_state.sources .= -1.0
    WarmupHMC.restore_reparam_sources!(
        copied,
        [idx => WarmupHMC.PartiallyCentered(0.0) for (idx, _) in copied_ir.pairs],
    )
    @test copied_state.sources == zeros(length(restored))
    @test wrapped_state.sources == restored
end

@testset "BridgeStan scalar random intercept adapts through the public wrapper" begin
    sb = SBBRMI(intercept_builder(df); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    unc_names = StanBlocks.BridgeStan.param_unc_names(problem.model)
    block = only(adaptive_centering_blocks(sb, unc_names))
    @test block.ranef.n_terms == 1
    @test isempty(block.cholesky_free)

    backend = AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    )
    wrapped = adaptive_centering_problem(sb, problem, backend)
    ir = WarmupHMC.reparametrizer(wrapped)
    state = ir.pairs[1][2].args[1].state
    controls = collect(range(0.2, 0.8, length=block.ranef.n_groups))
    set_adaptive_sources!(state, ir, controls)

    x = zeros(length(unc_names))
    x[only(block.log_scales)] = log(1.6)
    x[vec(block.effects)] .= collect(range(-0.5, 0.7, length=length(block.effects)))
    lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test isfinite(lp)
    @test all(isfinite, gradient)
    @test isfinite(enzyme_gradient_stress(wrapped, x))

    step = 1e-5
    finite_difference = [begin
        plus, minus = copy(x), copy(x)
        plus[i] += step
        minus[i] -= step
        (LogDensityProblems.logdensity(wrapped, plus) -
         LogDensityProblems.logdensity(wrapped, minus)) / (2step)
    end for i in eachindex(x)]
    @test gradient ≈ finite_difference atol=2e-5 rtol=2e-5

    ljac, model_position = ir(x)
    inverse_ljac, roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(
        ir, model_position,
    )
    @test inverse_ljac ≈ -ljac atol=2e-12
    @test roundtrip ≈ x atol=2e-12
end
