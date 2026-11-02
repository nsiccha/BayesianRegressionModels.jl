# Optional Laplace support must not prevent ordinary NLME fitting.
# Run in the NLMEEstimation scratch environment described in test/README.md.
using NLMEEstimation
import NLMEEstimation as NE
using Random
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

const NLMEEXT = Base.get_extension(BRM, :BayesianRegressionModelsNLMEEstimationExt)

# A public toy recurrence: each subject accumulates its own inputs. Computing
# a later entry needs the preceding entries, so a summand has no row-local form.
function accumulated_input(x, gain)
    carry = 0.0
    map(x) do value
        carry = 0.7carry + gain * value
    end
end
function BRM._rk_callable_source!(definitions, bindings, entry, ::typeof(accumulated_input))
    push!(definitions, :(ReactiveKernels.@kernel $entry(x, gain) = begin
        values = ReactiveKernels.scan(x; init=0.0) do carry, value
            next = 0.7carry + gain * value
            (next, next)
        end
        return values
    end))
    :done
end

@stestset "a scan's Laplace refusal leaves the ordinary NLME protocol usable" begin
    data = (; subject=["a", "b", "c"],
        x=[[0.2, 0.7, -0.1], [0.4, -0.3], [-0.2, 0.6, 0.3, 0.1]],
        rows=[[3, 1], [2, 1], [4, 1, 3]], y=[0.1, 0.3, -0.2])
    brmi = @brm data begin
        sigma ~ Exponential(1)
        gain ~ 1 + (1 | subject)
        @plate for i in eachindex(gain)
            pred[i] = accumulated_input(x[i], gain[i])
            y[i] ~ Normal(sum(pred[i][rows[i]]), sigma)
        end
    end
    m = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    @test NLMEEXT !== nothing
    key = collect(eachindex(m.view.families))
    # Exercise the real screening refusal, not a fabricated exception.
    @test_throws ReactiveKernels.PartitionLocalityError NLMEEXT._new_laplace_problem(m, key)
    @test !NE.provides_laplace_gradient(m)
    refusal = m.laplace[key]
    @test refusal isa ReactiveKernels.PartitionLocalityError
    @test !NE.provides_laplace_gradient(m)
    @test m.laplace[key] === refusal
    @test length(m.laplace) == 1
    @test_throws ArgumentError NLMEEXT._build_laplace_problem(m, Int[])
    @test_throws ArgumentError NLMEEXT._build_laplace_problem(m, [length(key) + 1])

    start = brm_nlme_start(m)
    # An explicit integration request keeps the exact cached diagnostic.
    explicit = try
        NE.laplace_conditionals_and_population_gradient(m, key, 1:NE.nsubjects(m),
            start.theta, start.sigma, start.omega, start.eta)
    catch err
        err
    end
    @test explicit === refusal
    @test NE.check_protocol(m, start.theta, start.sigma, start.omega; η=start.eta)
    fit = NE.saem(m, start.theta, start.sigma, start.omega; η0=start.eta,
        rng=Xoshiro(18), options=NE.SAEMOptions(iterations=4, burnin=2,
            mcmc_burnin=1, mh_rounds=2))
    @test all(isfinite, vcat(fit.theta, fit.sigma, fit.omega))
    @test isempty(fit.integrated)
    @test fit.kernel != :laplace_em
    @test m.laplace[key] === refusal
end

@stestset "a supported model still supplies the Laplace gradient" begin
    m = brm_nlme_model(RKBRMI(nested_model(NESTED_DATA)); ad_backend=NLME_AD,
        subject_group=:subject, nested_groups=(unit=:obs_subject,))
    @test NE.provides_laplace_gradient(m)
    key = collect(eachindex(m.view.families))
    problem = m.laplace[key]
    @test !(problem isa Exception)
    @test NE.provides_laplace_gradient(m)
    @test m.laplace[key] === problem
end
