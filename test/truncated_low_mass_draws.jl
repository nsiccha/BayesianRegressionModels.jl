# Predictive draws of a `truncated` response whose window holds almost none of
# its base law's mass.
#
#   julia --startup-file=no --project=test test/truncated_low_mass_draws.jl
#
# An omitted `truncated(NegativeBinomial2(mu, phi); lower, upper)` response is
# drawn by StanBlocks' `truncated` predictive companion, as is its bound
# twin's `_gen` draw. That companion redraws the base RNG until a draw lands in
# the window and rejects after 100000 attempts, so a window holding less than
# ~1e-5 of the base mass fails nearly every draw (snag
# truncated-negati-e5868621). The fix belongs to StanBlocks (snag
# truncated-rng-re-bd555cae); until the test pin carries it, the draws here are
# `@test_broken`, and the law checks run once they succeed.

using Test
using BayesianRegressionModels, StanBlocks
using BridgeStan, LogDensityProblems
using Distributions
using Statistics: mean

omit(data, keys::Symbol...) = Base.structdiff(data, NamedTuple{keys})

# `n` constrained draws (generated quantities included) at the zero
# unconstrained point, with their names; `nothing` when the truncated
# rejection sampler gives up (any other failure propagates).
function gq_draws(sb, seed, n)
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    rng = BridgeStan.StanRNG(problem.model, seed)
    u = zeros(LogDensityProblems.dimension(problem))
    try
        names, reduce(hcat, (BridgeStan.param_constrain(problem.model, u;
            include_tp=true, include_gq=true, rng) for _ in 1:n))
    catch err
        message = sprint(showerror, err)
        occursin("rejection sampler exceeded", message) || rethrow()
        println("gq_draws: ", first(split(message, '\n')))
        nothing
    end
end
column(names, draws, name) = draws[findfirst(==(name), names), :]
pmf_error(x, law, support) =
    maximum(abs(mean(x .== v) - pdf(law, v)) for v in support)
# A discrete law's mean over a support holding all but a negligible mass.
law_mean(law, support) = sum(v * pdf(law, v) for v in support)

# Mean 70 and precision 10, as Distributions spells NegativeBinomial2.
const base = NegativeBinomial(10.0, 10.0 / 80.0)
# Masses 2.9e-5, 3.9e-7 and 0.80: the first two fail the rejection sampler.
const windows = (; lo=[1, 250, 50], hi=[9, 10_000_000, 249], c=[3, 300, 70])

@testset "a two-sided truncated NegativeBinomial2 draws in windows of tiny mass" begin
    builder = @brm begin
        c ~ truncated(NegativeBinomial2(70.0, 10.0); lower=lo, upper=hi)
    end
    sb = @test_logs (:warn, r"unconditioned") SBBRMI(
        builder(omit(windows, :c)); mod=@__MODULE__)
    result = gq_draws(sb, 17, 4000)
    @test_broken !isnothing(result)
    if !isnothing(result)
        names, draws = result
        for i in eachindex(windows.lo)
            c = column(names, draws, "c_gen.$i")
            @test all(isinteger, c) && all(windows.lo[i] .<= c .<= windows.hi[i])
        end
        @test pmf_error(column(names, draws, "c_gen.1"),
                        truncated(base, 1, 9), 1:9) < 0.03
        upper_tail = truncated(base, 250, 10_000_000)
        @test mean(column(names, draws, "c_gen.2")) ≈ law_mean(upper_tail, 250:1000) rtol = 0.01
        @test pmf_error(column(names, draws, "c_gen.2"), upper_tail, 250:256) < 0.03
    end
end

@testset "one-sided truncated NegativeBinomial2 draws in tails of tiny mass" begin
    lower_only = @brm begin
        c ~ truncated(NegativeBinomial2(70.0, 10.0); lower=lo)
    end
    upper_only = @brm begin
        c ~ truncated(NegativeBinomial2(70.0, 10.0); upper=hi)
    end
    for (builder, data, law, window, support) in (
            (lower_only, (; lo=[250]), truncated(base; lower=250), (250, Inf), 250:256),
            (upper_only, (; hi=[5]), truncated(base; upper=5), (0, 5), 0:5))
        sb = @test_logs (:warn, r"unconditioned") SBBRMI(
            builder(data); mod=@__MODULE__)
        result = gq_draws(sb, 23, 4000)
        @test_broken !isnothing(result)
        if !isnothing(result)
            names, draws = result
            c = column(names, draws, "c_gen.1")
            @test all(isinteger, c) && all(window[1] .<= c .<= window[2])
            @test pmf_error(c, law, support) < 0.03
        end
    end
end

@testset "a truncated Normal draws in a tail of tiny mass" begin
    builder = @brm begin
        y ~ truncated(Normal(0.0, 1.0); lower=lo)
    end
    # P(Z >= 6) is 1e-9.
    sb = @test_logs (:warn, r"unconditioned") SBBRMI(
        builder((; lo=[6.0, -1.0])); mod=@__MODULE__)
    result = gq_draws(sb, 29, 4000)
    @test_broken !isnothing(result)
    if !isnothing(result)
        names, draws = result
        for (i, lo) in enumerate((6.0, -1.0))
            y = column(names, draws, "y_gen.$i")
            @test all(y .>= lo)
            @test mean(y) ≈ mean(truncated(Normal(); lower=lo)) atol = 0.05
        end
    end
end
