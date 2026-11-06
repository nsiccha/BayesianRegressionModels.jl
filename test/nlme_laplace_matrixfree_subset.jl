# RUN: julia --project=test test/nlme_laplace_matrixfree_subset.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra

@stestset "matrix-free plan subsets rebuild the reference in the retained coordinates" begin
    for sums in (false,true)
        d = (; a=repeat(["a","b","c"],inner=4), b=repeat(["x","y"],6),
            rows=[[1,2,3],[3,4,5],[6,7,8,9,10,11,12]],
            y=sums ? [3.5,4.0,6.5] : [1,0,2,1,3,1,2,0,1,2,1,0])
        rk = RKBRMI(if sums
            @brm d begin
                eta ~ 1 + (1 | a) + (1 | b)
                @plate for k in eachindex(rows)
                    y[k] ~ Normal(sum(exp.(eta[rows[k]])),0.7)
                end
            end
        else
            @brm d begin
                log(mu) ~ 1 + (1 | a) + (1 | b)
                y ~ Poisson(mu)
            end
        end)
        options = MatrixFreeLaplace(; probes=4)
        # The Poisson term b is prepared, then rejected by the impossible ESS
        # threshold. Nonlinear sum cells require explicit admission of both
        # terms; retain only a afterward to exercise the same strict subset.
        candidate = brm_laplace_plan(rk; ad_backend=NLME_AD,
            integrate=sums ? (:a, :b) : (:a,),
            draws=8, ess_floor=1.1, solver=:matrixfree, matrixfree=options)
        plan = if sums
            terms = NamedTuple[merge(t, (; admitted=t.grouping === :a))
                for t in candidate.terms]
            BRMLaplacePlan(terms, candidate.ess_floor, candidate.draws,
                candidate.max_cell_width, candidate.width_events, -1, -1,
                candidate.prepared, candidate.algebra_policy)
        else
            candidate
        end
        template = plan.prepared.problem
        original_probes = deepcopy(template.algebra.probes)
        original_lower = [copy(r.lower) for r in template.algebra.references]
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan)
        @test count(t->t.admitted,plan.terms) == 1
        @test length(p.integrated) < length(template.integrated)
        fresh_plan = BRMLaplacePlan(plan.terms,plan.ess_floor,plan.draws)
        fresh = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=fresh_plan,
            solver=:matrixfree, matrixfree=options)
        @test p.integrated == fresh.integrated && p.population == fresh.population
        @test p.components == fresh.components
        @test p.algebra.probes == fresh.algebra.probes
        @test all(zip(p.algebra.references,fresh.algebra.references)) do (a,b)
            a.lower == b.lower && a.permutation == b.permutation &&
                a.inverse_scale == b.inverse_scale
        end
        for point in (-0.05,0.1)
            x = fill(point,length(p.population))
            value, gradient = LogDensityProblems.logdensity_and_gradient(p,x)
            expected, expected_gradient = LogDensityProblems.logdensity_and_gradient(fresh,x)
            @test isfinite(value) && all(isfinite,gradient)
            @test value ≈ expected rtol=1e-9 atol=1e-9
            @test gradient ≈ expected_gradient rtol=2e-8 atol=2e-8
            fd = map(eachindex(x)) do j
                delta=zeros(length(x)); delta[j]=1e-5
                (LogDensityProblems.logdensity(p,x+delta) -
                    LogDensityProblems.logdensity(p,x-delta))/2e-5
            end
            @test gradient ≈ fd rtol=2e-4 atol=2e-4
        end
        @test template.algebra.probes == original_probes
        @test all(r.lower == l for (r,l) in zip(template.algebra.references,original_lower))
    end
end
