using Test
include("smc_tuning_lifecycle_fixture.jl")
include("smc_window_selective_fixture.jl")
using .SMCTuningLifecycleFixture
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, Random
const F = SMCTuningLifecycleFixture
const OPTIONS = SMCOptions(; moves=1, max_tree_depth=3, history_limit=8, nthreads=1)

function checkpoint_case(kind)
    rng = Xoshiro(941)
    if kind in (:initial, :cumulative, :later_cumulative)
        rk = RKBRMI(F.model())
        fit = brm_smc_fit(rk; rng, particles=16, stages=0, options=OPTIONS)
        kwargs = (;)
        if kind !== :initial
            data = (; x=vcat(F.DATA.x, [2.0]), y=vcat(F.DATA.y, [0.6]))
            fit = brm_smc_update(brm_smc_fit(fit), data; stages=0, options=OPTIONS)
            kwargs = (; data)
            if kind === :later_cumulative
                next = (; x=vcat(data.x, [2.4]), y=vcat(data.y, [0.7]))
                fit = brm_smc_update(brm_smc_fit(fit), next; stages=0, options=OPTIONS)
                kwargs = (; data=next, previous_data=data)
            end
        end
        return fit, (cp -> brm_smc_resume(rk, cp; kwargs...)),
            (cp; kw...) -> brm_smc_rescore(rk, cp; kwargs..., kw...)
    elseif kind in (:window, :compact_window)
        if kind === :window
            (carry, window), _ = F.targets(:window)
            kwargs = (;)
        else
            build = SMCSelectiveFixture.hierarchical
            rk = RKBRMI(build((; nb=2, block=[1, 2], y=[0.0, 0.0])))
            carry = brm_smc_carry(rk; reference=SMCReference(zeros(3), ones(3)))
            window = build((; nb=3, block=[3], y=[0.8]))
            kwargs = (; coordinates=:auto)
        end
        fit = brm_smc_update(carry, window; rng, particles=16, stages=0,
            options=OPTIONS, kwargs...)
        return fit, (cp -> brm_smc_resume(window, cp)),
            (cp; kw...) -> brm_smc_rescore(window, cp; kw...)
    elseif kind in (:laplace, :laplace_update, :later_laplace)
        (problem,), _ = F.targets(:laplace)
        fit = brm_smc_fit(problem; rng, particles=16, stages=0, options=OPTIONS)
        kwargs = (;)
        if kind !== :laplace
            data = (; g=["a", "a", "b", "b", "a"], y=[0.2, 0.4, -0.1, 0.3, 0.6])
            rebuild(data) = brm_laplace_problem(reprocess(problem.backend, data);
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), integrate=(:g,))
            current = rebuild(data)
            fit = brm_smc_update(brm_smc_fit(fit), current; stages=0, options=OPTIONS)
            kwargs = (; current)
            if kind === :later_laplace
                next = rebuild((; g=vcat(data.g, ["b"]), y=vcat(data.y, [0.8])))
                fit = brm_smc_update(brm_smc_fit(fit), next; stages=0, options=OPTIONS)
                kwargs = (; current=next, previous=current)
            end
        end
        return fit, (cp -> brm_smc_resume(problem, cp; kwargs...)),
            (cp; kw...) -> brm_smc_rescore(problem, cp; kwargs..., kw...)
    end
    error("unknown checkpoint case: $kind")
end

# RKPPL's documented serialized endpoint-score fields. Only the fixture changes
# scores; production BRM passes the complete checkpoint to the producer API.
function drift(cp; large=false)
    source_scores, target_scores = copy(cp.state.source_scores), copy(cp.state.target_scores)
    source_scores[1] = nextfloat(source_scores[1])
    target_scores[2] = large ? target_scores[2] + 0.01 : nextfloat(nextfloat(target_scores[2]))
    merge(cp, (; state=merge(cp.state, (; source_scores, target_scores))))
end

@testset "Explicit checkpoint rescore: $kind" for kind in
        (:initial, :cumulative, :later_cumulative, :window, :compact_window,
         :laplace, :laplace_update, :later_laplace)
    fit, resume, rescore = checkpoint_case(kind)
    cp = F.roundtrip(brm_smc_checkpoint(fit))
    changed = drift(cp)
    before = F.roundtrip(changed)
    @test_throws ArgumentError resume(changed)
    @test_throws ArgumentError rescore(changed; rtol=0)
    @test_throws ArgumentError rescore(drift(cp; large=true))
    @test F.equal(changed, before)
    corrected = rescore(changed)
    @test F.equal(changed, before)
    @test corrected.state.source_scores == cp.state.source_scores
    @test corrected.state.target_scores == cp.state.target_scores
    @test length(corrected.state.rescored) == length(cp.state.rescored) + 1
    audit = last(corrected.state.rescored)
    @test audit.changed > 0
    @test audit.max_abs > 0
    @test 0 < audit.max_rel <= audit.rtol == 1e-10
    @test hasproperty(audit, :codegen)
    again = rescore(corrected; rtol=0)
    @test length(again.state.rescored) == length(corrected.state.rescored) + 1
    @test last(again.state.rescored).changed == 0
    @test last(again.state.rescored).rtol == 0
    # A rescore changes scores and audit history only, including through disk.
    @test F.equal(merge(corrected.state, (; rescored=cp.state.rescored)), cp.state)
    baseline = brm_smc_checkpoint(brm_smc_fit(resume(cp)))
    continued = brm_smc_checkpoint(brm_smc_fit(resume(F.roundtrip(corrected))))
    @test continued.state.temperature == 1
    @test F.equal(merge(continued.state, (; rescored=baseline.state.rescored)), baseline.state)
    @test F.equal(changed, before)
end

@testset "Rescore retains BRM model and data guards" begin
    rk = RKBRMI(F.model())
    fit = brm_smc_fit(rk; rng=Xoshiro(942), particles=16, options=OPTIONS)
    data = (; x=vcat(F.DATA.x, [2.0]), y=vcat(F.DATA.y, [0.6]))
    cp = brm_smc_checkpoint(brm_smc_update(fit, data; stages=0, options=OPTIONS))
    changed = merge(data, (; y=data.y .+ 0.1))
    @test_throws ArgumentError brm_smc_rescore(rk, cp; data=changed, rtol=1)
    @test_throws ArgumentError brm_smc_rescore(rk, cp; data,
        previous_data=merge(F.DATA, (; y=F.DATA.y .+ 0.1)), rtol=1)
    @test_throws ArgumentError brm_smc_rescore(RKBRMI(F.model(changed)), cp; data, rtol=1)
    @test_throws ArgumentError brm_smc_rescore(rk, merge(cp, (; format=99)); data)
    (carry, window), _ = F.targets(:window)
    wcp = brm_smc_checkpoint(brm_smc_update(carry, window;
        rng=Xoshiro(943), particles=16, stages=0, options=OPTIONS))
    @test_throws ArgumentError brm_smc_rescore(F.model((; x=[0.8, 1.7], y=[0.3, 1.0])), wcp; rtol=1)
    @test_throws ArgumentError brm_smc_rescore(window, cp)
    (problem,), _ = F.targets(:laplace)
    lcp = brm_smc_checkpoint(brm_smc_fit(problem;
        rng=Xoshiro(944), particles=16, stages=0, options=OPTIONS))
    altered = brm_laplace_problem(reprocess(problem.backend,
        (; g=["a", "a", "b", "b"], y=[0.2, 0.4, -0.1, 0.5]));
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), integrate=(:g,))
    @test_throws ArgumentError brm_smc_rescore(problem, lcp; current=altered, rtol=1)
    @test_throws ArgumentError brm_smc_rescore(problem, lcp; previous=problem)
end
