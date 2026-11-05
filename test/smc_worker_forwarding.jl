using Test
include("smc_tuning_lifecycle_fixture.jl")
using .SMCTuningLifecycleFixture
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, Random
const F = SMCTuningLifecycleFixture

# BRM forwards ReactiveKernelsPPL 0.1.100's worker controls: `grid_start` on
# every brm_smc_tune route and `nthreads` on every brm_smc_resume route. The
# assertions hold at any Julia thread count; run with several threads (for
# example `julia -t 4`) to also exercise concurrent pilots and workers.

# Three grid knots, so independent pilots have concurrent work.
const GRID = (; temperatures=(0.0, 0.5, 1.0))
options(nthreads) = SMCOptions(; moves=1, max_tree_depth=3, history_limit=2, nthreads)

@testset "brm_smc_tune forwards grid_start: $kind" for kind in (:initial, :cumulative, :window, :laplace)
    args, settings = F.targets(kind)
    sequential = F.tune(args, settings; GRID...)
    @test sequential isa SMCTuningSchedule
    @test sequential.pilot.grid_start === :sequential
    one = F.tune(args, settings; GRID..., grid_start=:independent, options=options(1))
    many = F.tune(args, settings; GRID..., grid_start=:independent,
        options=options(max(2, Threads.nthreads())))
    @test one.pilot.grid_start === :independent
    # The schedule does not depend on how many pilots ran at once.
    @test F.equal(one, many)
    @test_throws ArgumentError F.tune(args, settings; GRID..., grid_start=:concurrent)
end

laplace_problem() = F.targets(:laplace)[1][1]

# Each case returns (initialized fit at stages=0, resume(checkpoint; kw...)).
function production(kind)
    rng = Xoshiro(811)
    if kind === :initial
        rk = RKBRMI(F.model())
        fit = brm_smc_fit(rk; rng, particles=16, stages=0, options=options(1))
        return fit, (cp; kw...) -> brm_smc_resume(rk, cp; kw...)
    elseif kind === :cumulative
        (completed, data), _ = F.targets(:cumulative)
        fit = brm_smc_update(completed, data; stages=0, options=options(1))
        return fit, (cp; kw...) -> brm_smc_resume(completed.template, cp; data, kw...)
    elseif kind === :window
        (carry, window), _ = F.targets(:window)
        fit = brm_smc_update(carry, window; rng, particles=16, stages=0, options=options(1))
        return fit, (cp; kw...) -> brm_smc_resume(window, cp; kw...)
    elseif kind === :laplace
        problem = laplace_problem()
        fit = brm_smc_fit(problem; rng, particles=16, stages=0, options=options(1))
        return fit, (cp; kw...) -> brm_smc_resume(problem, cp; kw...)
    end
end

@testset "brm_smc_resume forwards nthreads: $kind" for kind in (:initial, :cumulative, :window, :laplace)
    fit, resume = production(kind)
    cp = brm_smc_checkpoint(fit)
    saved = resume(cp)
    raised = resume(cp; nthreads=3)
    @test saved.state.options.nthreads == 1
    @test raised.state.options.nthreads == 3
    @test_throws ArgumentError resume(cp; nthreads=0)
    done_saved = brm_smc_fit(saved)
    done_raised = brm_smc_fit(raised)
    @test done_saved.state.temperature == done_raised.state.temperature == 1
    # The cloud does not depend on the worker cap; the recorded workers do.
    @test done_saved.state.positions == done_raised.state.positions
    @test done_saved.state.weights == done_raised.state.weights
    @test done_saved.state.log_normalizer == done_raised.state.log_normalizer
    # Every stage of this bridge ran after the resume (the fit stopped at
    # stages=0); earlier stages of an update belong to the previous bridge.
    after(fit) = filter(s -> s.update == fit.state.update, fit.state.history)
    n = size(done_raised.state.positions, 2)
    @test !isempty(after(done_saved))
    @test all(s -> s.nthreads == 1, after(done_saved))
    @test all(s -> s.nthreads == min(3, Threads.nthreads(), n), after(done_raised))
end
