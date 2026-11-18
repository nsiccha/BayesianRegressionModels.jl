# test/rk_nonfinite_gradient_recovery.jl — after one evaluation whose gradient
# is not finite, the next gradient of the same `rk_logdensity_problem` is the one
# it gave before (BRM snag `rk-logdensity-pr-a8c1995e`).
#
# `rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))`
# differentiates a non-allocating ReactiveKernels kernel through
# DifferentiationInterface, which keeps one shadow per reused buffer across
# calls. Before Enzyme dogfood cb2b2b1e (repair `blas_overwritten_output`,
# libEnzyme 5639cd2f), the reverse rules of `gemv`, `gemm`, `symv`, `symm` and
# `syrk` at `beta == 0` scaled the shadow of the output the call overwrote by
# `beta` instead of clearing it. A NaN adjoint from one extreme point then stayed
# in that shadow and entered every later gradient while the log density stayed
# right: 22 of the 23 entries of `mixed` below, and the three `X * beta`
# coefficients of `guarded`. A sampler whose leapfrog step reaches such a point
# keeps a frozen chain.
#
# Each case evaluates a copy of the prepared problem at a good point, at a point
# with one extreme coordinate, and at the good point again. The extreme point must
# give a non-finite answer (or throw, for `guarded`'s refusal), else the case
# tests nothing; the good point's value and gradient afterwards must equal those
# before it, bit for bit, and so must a `deepcopy` taken after the extreme point.
# The test environment's Enzyme pin (test/setup_env.jl) must carry the repair.
# Public synthetic data only.
#
# Run: julia --startup-file=no --project=test test/rk_nonfinite_gradient_recovery.jl
using Test
using BayesianRegressionModels
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme

module NonfiniteRecovery
using BayesianRegressionModels
using BayesianRegressionModels.Distributions
# Refuses a rate below 2 in the forward pass, as a consumer's guard does.
function guarded_rate(k)
    k >= 2 || throw(DomainError(k, "guarded_rate: the rate must be at least 2"))
    k
end
const mixed = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x1 + x2 + (1 + x1 | p | subject)
    y ~ Normal(mu, sigma)
end
const guarded = @brm begin
    log_k ~ Normal(log(5.0), 0.2)
    sigma ~ Exponential(1)
    eta ~ 1 + x1 + x2
    mu = guarded_rate(exp(log_k)) .* eta
    y ~ Normal(mu, sigma)
end
end

const NOBS = 40
const DATA = (; subject=repeat(1:8; inner=5),
    x1=round.(sin.(1:NOBS); digits=3), x2=round.(cos.(2 .* (1:NOBS)); digits=3),
    y=round.(1 .+ 0.5 .* sin.(3 .* (1:NOBS) .+ 1); digits=3))
const AD = AutoEnzyme(; mode=Enzyme.Reverse)

@testset "the pinned Enzyme carries blas_overwritten_output" begin
    @test isdefined(Enzyme, :has_dogfood_repair) &&
        Enzyme.has_dogfood_repair(:blas_overwritten_output)
end

# `extremes`: coordinate name => values that each give a non-finite answer;
# `throwing`: coordinate name => values whose evaluation throws.
function check_recovery(label, builder; extremes, throwing=())
    rk = RKBRMI(builder(DATA))
    names = string.(coordinate_names(rk.model.layout))
    pristine = rk_logdensity_problem(rk; ad_backend=AD)
    u0 = brm_start(rk, (;); base=:prior_median)
    good = u0 .+ 0.1 .* sin.(eachindex(u0))
    lp_fresh, g_fresh = LogDensityProblems.logdensity_and_gradient(deepcopy(pristine), good)
    @test isfinite(lp_fresh) && all(isfinite, g_fresh)
    for (name, values, throws) in (map(p -> (p..., false), extremes)...,
                                   map(p -> (p..., true), throwing)...)
        i = findfirst(==(name), names)
        @test i !== nothing
        i === nothing && continue
        for v in values
            @testset "$label: $name = $v" begin
                live = deepcopy(pristine)
                lp1, g1 = LogDensityProblems.logdensity_and_gradient(live, good)
                bad = copy(good); bad[i] = v
                if throws
                    @test_throws Exception LogDensityProblems.logdensity_and_gradient(live, bad)
                else
                    lpb, gb = LogDensityProblems.logdensity_and_gradient(live, bad)
                    # The positive control: this point reaches a non-finite answer.
                    @test !(isfinite(lpb) && all(isfinite, gb))
                end
                copied = deepcopy(live)
                lp3, g3 = LogDensityProblems.logdensity_and_gradient(live, good)
                @test isequal(lp3, lp1)
                @test isequal(g3, g1)
                @test count(!isfinite, g3) == 0
                @test isequal(LogDensityProblems.logdensity_and_gradient(copied, good), (lp1, g1))
                @test isequal((lp1, g1), (lp_fresh, g_fresh))
            end
        end
    end
end

@testset "a non-finite gradient does not carry over" begin
    # `tau` far out makes the correlated block's density NaN.
    check_recovery("mixed", NonfiniteRecovery.mixed;
        extremes=["b_p_subject.tau.1" => (800.0, 1e300),
                  "b_p_subject.tau.2" => (800.0, 1e300)])
    # `log_k` far out overflows the rate; far below, `guarded_rate` throws.
    check_recovery("guarded", NonfiniteRecovery.guarded;
        extremes=["log_k" => (800.0, 1e300)],
        throwing=["log_k" => (-800.0,)])
end
