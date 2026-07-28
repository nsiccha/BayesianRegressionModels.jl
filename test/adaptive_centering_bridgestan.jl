include(joinpath(@__DIR__, "adaptive_centering_warmuphmc.jl"))

using Enzyme
using DifferentiationInterface: AutoEnzyme
using LogDensityProblems
using StanBlocks

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

    lp, gradient = LogDensityProblems.logdensity_and_gradient(reparametrized, x)
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
