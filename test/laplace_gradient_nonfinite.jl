# RUN: julia --project=test test/laplace_gradient_nonfinite.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems

# Requirement (snag laplace-gradient-db992ae0): where the sparse Laplace log
# density is finite and every conditional mode converged but the gradient is
# not, the target counts it in `nonfinite_gradient` and warns with the first
# nonfinite term instead of returning that gradient silently. The gradient
# stays as computed, so SMC still stops on it and HMC still rejects the step.
# Exact and matrix-free algebra behave alike.
#
# |b|^2.5 has a finite value and first and second derivatives at b = 0, but an
# infinite third derivative, which the gradient multiplies by the mode's zero
# movement (Inf * 0 = NaN). The mode is b = 0 for every `shift`, so the
# marginal has the closed form below; its true derivative at shift = 0 is
# (1 - 0)/1^2 + (0.4 - 0)/0.2^2 = 11.
function third_singular_fixture(sumcell)
    data = (; n=2, idx=[1,2], rows=[[1],[2]], y=[1.0,0.4], noise=[1.0,0.2])
    model = if sumcell
        @brm data begin
            shift ~ Normal(0,1)
            field[1:n] .~ Normal.(0,1)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(shift + abs(sum(field[rows[k]]))^2.5, noise[k])
            end
        end
    else
        @brm data begin
            shift ~ Normal(0,1)
            field[1:n] .~ Normal.(0,1)
            @plate for k in eachindex(idx)
                y[k] ~ Normal(shift + abs(field[idx[k]])^2.5, noise[k])
            end
        end
    end
    RKBRMI(model)
end

const THIRD_SINGULAR_CASES = (
    (false, (; solver=:exact), "the third derivative of cell 1 of `y`"),
    (true, (; solver=:exact), "the third derivatives of the sum factors of `y`"),
    (false, (; solver=:matrixfree, matrixfree=MatrixFreeLaplace(; probes=2,
        preconditioner_memory=nothing)), "the third derivative of cell 1 of `y`"))

@stestset "a nonfinite gradient at a finite Laplace value is counted and attributed" begin
    for (sumcell, algebra, term) in THIRD_SINGULAR_CASES
        rk = third_singular_fixture(sumcell)
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:field,), maxiter=200,
            algebra...)
        @test (only(p.cells) isa BRM._BRMSumCellGroup) == sumcell
        @test (p.algebra === nothing) == (algebra.solver === :exact)
        @test p.nonfinite_gradient == 0
        full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
        closed = LogDensityProblems.logdensity(full, zeros(length(p.u))) + log(2pi)
        pattern = Regex("gradient is not finite in 1 of 1 population coordinates " *
            "\\(the first is `shift`\\).*The first nonfinite term is $term\\.")
        value, gradient = @test_logs((:warn, pattern), match_mode=:any,
            LogDensityProblems.logdensity_and_gradient(p, [0.0]))
        @test value ≈ closed atol=1e-8
        @test !isfinite(only(gradient))
        @test p.nonfinite_gradient == 1
        @test p.nonfinite == p.unconverged == 0
        # A value alone is not counted, and its central difference is the
        # finite derivative the gradient could not form.
        fd = (LogDensityProblems.logdensity(p, [1e-4]) -
              LogDensityProblems.logdensity(p, [-1e-4])) / 2e-4
        @test fd ≈ 11 rtol=1e-6
        @test p.nonfinite_gradient == 1
        LogDensityProblems.logdensity_and_gradient(p, [0.3])
        @test p.nonfinite_gradient == 2
    end
end

@stestset "targets materialized from a plan start with no nonfinite gradients" begin
    rk = third_singular_fixture(false)
    plan = brm_laplace_plan(rk; ad_backend=NLME_AD, integrate=(:field,))
    prepared = plan.prepared.problem
    LogDensityProblems.logdensity_and_gradient(prepared, [0.0])
    @test prepared.nonfinite_gradient == 1
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=plan, maxiter=200)
    @test p.nonfinite_gradient == 0
    LogDensityProblems.logdensity_and_gradient(p, [0.0])
    @test p.nonfinite_gradient == 1
    @test prepared.nonfinite_gradient == 1
end
