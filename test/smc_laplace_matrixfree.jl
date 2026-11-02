# RUN: julia --project=test test/smc_laplace_matrixfree.jl
using Test, Treebars
include("smc_laplace_fixture.jl")
const SMCEXT = Base.get_extension(BRM, :BayesianRegressionModelsSMCExt)

@testset "Matrix-free Laplace SMC owns and identifies its finite target" begin
    data = (; a=repeat(["a","b","c"],inner=2), b=repeat(["x","y"],3),
        y=[0.3,0.9,-0.5,0.1,1.1,0.7])
    rk = RKBRMI(@brm data begin
        mu ~ 1 + (1 | a) + (1 | b)
        effect(mu, Intercept) ~ Normal(0,1)
        sd(mu,a) = 0.7
        sd(mu,b) = 0.5
        y ~ Normal(mu,0.8)
    end)
    problem(; solver=:matrixfree, kwargs...) = brm_laplace_problem(rk;
        ad_backend=LSMC_AD, integrate=(:a,:b), solver,
        matrixfree=MatrixFreeLaplace(; probes=4, kwargs...))
    p = problem()
    identity = SMCEXT.binding_identity(p)
    probes = deepcopy(p.algebra.probes)
    initial = brm_smc_fit(p; rng=Xoshiro(74), particles=16,
        options=LSMC_OPTIONS, stages=0)
    @test initial.backend.algebra.probes !== p.algebra.probes
    @test initial.backend.algebra.references[1].lower !== p.algebra.references[1].lower
    checkpoint = lsmc_roundtrip(brm_smc_checkpoint(initial))
    completed = brm_smc_fit(initial)
    replay = brm_smc_fit(brm_smc_resume(problem(),checkpoint))
    @test completed.state.temperature == 1
    @test completed.state.positions == replay.state.positions
    @test completed.state.weights == replay.state.weights
    @test completed.state.log_normalizer == replay.state.log_normalizer
    @test p.algebra.probes == probes
    @test SMCEXT.binding_identity(p) == identity
    out = brm_laplace_draws(completed; rng=Xoshiro(75))
    @test all(isfinite,out.draws) && out.weights == completed.state.weights
    @test out.draws[p.population,:] == completed.state.positions

    for settings in ((; seed=76), (; logdet_step=0.5), (; solve_tolerance=1e-9),
            (; preconditioner_ridge=0.02))
        changed = problem(; settings...)
        @test_throws ArgumentError brm_smc_resume(changed,checkpoint)
        @test_throws ArgumentError SMCEXT.smc_update_model(p,changed)
    end
    @test_throws ArgumentError brm_smc_resume(problem(; solver=:exact),checkpoint)
    changed = problem(); changed.algebra.probes[1][1,1] *= -1
    @test_throws ArgumentError brm_smc_resume(changed,checkpoint)
    changed = problem(); changed.algebra.references[1].inverse_scale[1] *= 1.01
    @test_throws ArgumentError brm_smc_resume(changed,checkpoint)
    # Mode/diagnostic updates do not change the frozen density identity.
    LogDensityProblems.logdensity_and_gradient(p,[0.2])
    @test SMCEXT.binding_identity(p) == identity
    resumed = brm_smc_resume(p,checkpoint)
    @test resumed.binding_id == initial.binding_id

    data2 = (; a=[data.a;data.a], b=[data.b;data.b], y=[data.y;data.y .+ 0.2])
    next_problem() = brm_laplace_problem(reprocess(rk,data2);
        ad_backend=LSMC_AD, integrate=(:a,:b), solver=:matrixfree,
        matrixfree=MatrixFreeLaplace(; probes=4))
    next = next_problem()
    @test next.algebra.references[1].inverse_scale != p.algebra.references[1].inverse_scale
    partial = brm_smc_update(completed,next; options=LSMC_OPTIONS, stages=0)
    restored = brm_smc_resume(problem(),lsmc_roundtrip(brm_smc_checkpoint(partial));
        current=next_problem())
    @test restored.binding_id == partial.binding_id
    @test restored.state.positions == partial.state.positions
    @test restored.state.weights == partial.state.weights
end
