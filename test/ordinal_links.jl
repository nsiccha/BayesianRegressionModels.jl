# test/ordinal_links.jl — typed ordinal structure/link composition.
#
# Run on a capable host:
#   julia --startup-file=no --project=. test/ordinal_links.jl

using Test
using BayesianRegressionModels
using Distributions
using LogDensityProblems
using LogExpFunctions: logistic
using Random
using StanBlocks
const BridgeStan = StanBlocks.BridgeStan

@testset "executable ordinal distributions" begin
    cuts = [-1.0, 0.5]

    cumulative_logit = Ordinal(Cumulative(), LogitLink(), 0.2, cuts)
    legacy = OrderedLogistic(0.2, cuts)
    @test cumulative_logit isa DiscreteUnivariateDistribution
    @test params(cumulative_logit) ==
          (Cumulative(), LogitLink(), 0.2, cuts, 1.0)
    @test Ordinal(params(cumulative_logit)...) isa Ordinal
    @test probs(cumulative_logit) ≈ probs(legacy)
    @test logpdf.(Ref(cumulative_logit), 1:3) ≈
          logpdf.(Ref(legacy), 1:3)

    cumulative_probit = Ordinal(
        Cumulative(), ProbitLink(), 0.2, cuts; discrimination=1.4)
    z = 1.4 .* (cuts .- 0.2)
    expected_cumulative = [
        cdf(Normal(), z[1]),
        cdf(Normal(), z[2]) - cdf(Normal(), z[1]),
        ccdf(Normal(), z[2]),
    ]
    @test probs(cumulative_probit) ≈ expected_cumulative
    @test exp.(logpdf.(Ref(cumulative_probit), 1:3)) ≈ expected_cumulative

    stage_eta = [0.2, -0.1]
    stopping_cloglog = Ordinal(
        StoppingRatio(), CloglogLink(), stage_eta, cuts;
        discrimination=0.8)
    q = @. -expm1(-exp(0.8 * (cuts - stage_eta)))
    expected_stopping = [q[1], (1 - q[1]) * q[2],
                         (1 - q[1]) * (1 - q[2])]
    @test probs(stopping_cloglog) ≈ expected_stopping
    @test exp.(logpdf.(Ref(stopping_cloglog), 1:3)) ≈ expected_stopping

    # Every supported typed pair produces a normalized executable
    # distribution. This is the public composition contract, independent of
    # whichever private Stan UDF implements the pair.
    for structure in (Cumulative(), StoppingRatio()),
        link in (LogitLink(), ProbitLink(), CloglogLink())
        eta = structure isa Cumulative ? 0.2 : stage_eta
        d = Ordinal(structure, link, eta, cuts)
        @test sum(probs(d)) ≈ 1
        @test all(p -> 0 <= p <= 1, probs(d))
        draws = rand(MersenneTwister(20260729), d, 64)
        @test all(1 .<= draws .<= 3)
    end

    @test logpdf(cumulative_probit, 0) == -Inf
    @test logpdf(stopping_cloglog, 3.5) == -Inf
    @test_throws DomainError Ordinal(
        Cumulative(), LogitLink(), 0.0, [1.0, -1.0])
    @test_throws DomainError Ordinal(
        Cumulative(), LogitLink(), 0.0, cuts; discrimination=0.0)
    @test_throws ArgumentError Ordinal(
        Cumulative(), LogitLink(), [0.0, 0.0], cuts)
    @test_throws DimensionMismatch Ordinal(
        StoppingRatio(), LogitLink(), [0.0], cuts)
end

ordinal_df = (;
    x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    treat=[0, 1, 0, 1, 0, 1],
    y_cumulative=[1, 2, 3, 1, 2, 3],
    y_stopping=[1, 2, 3, 2, 1, 3],
)

ordinal_builder = @brm begin
    eta_cumulative ~ 0 + x
    log(discrimination) ~ 0 + x
    y_cumulative ~ Ordinal(
        Cumulative(), ProbitLink(), eta_cumulative;
        discrimination=discrimination)

    eta_stopping ~ 0 + x
    y_stopping ~ Ordinal(
        StoppingRatio(), CloglogLink(), eta_stopping;
        per_threshold=(treat,))
end

@testset "typed ordinal formula lowering" begin
    sb = SBBRMI(ordinal_builder(ordinal_df))
    code = BayesianRegressionModels.stan_code(sb)
    checked = StanBlocks.stanc_check(code; warn_pedantic=false)
    checked.ok || @error "stanc rejected ordinal model" output=checked.output
    @test checked.ok

    @test occursin("ordered[2] y_cumulative_thresholds;", code)
    @test occursin("vector[2] y_stopping_thresholds;", code)
    @test occursin("y_stopping_threshold_beta", code)
    @test occursin("multi_std_normal", code)
    @test occursin("brm_ordinal_lpmf", code)
    @test occursin("brm_ordinal_lpmfs", code)
    @test occursin("brm_ordinal_rng", code)
    @test sb.preproc[:y_cumulative].kind === :ordinal_outcome
    @test sb.preproc[:y_stopping].kind === :ordinal_outcome
    @test sb.preproc[:treat].kind === :ordinal_threshold_predictor
    @test eltype(sb.data[:treat]) === Float64

    new_df = (;
        x=[-0.75, 0.25, 1.25],
        treat=[1, 0, 1],
        y_cumulative=[1, 2, 3],
        y_stopping=[3, 2, 1],
    )
    replayed = reprocess(sb, new_df)
    @test replayed.data[:y_cumulative] == new_df.y_cumulative
    @test replayed.data[:y_stopping] == new_df.y_stopping
    @test replayed.data[:treat] == Float64.(new_df.treat)
    @test StanBlocks.stanc_check(
        BayesianRegressionModels.stan_code(replayed);
        warn_pedantic=false).ok
end

@testset "ordinal formula validation is early and explicit" begin
    intercept_builder = @brm begin
        eta ~ 1 + x
        y_cumulative ~ Ordinal(Cumulative(), LogitLink(), eta)
    end
    @test_throws ErrorException SBBRMI(intercept_builder(ordinal_df))

    cumulative_cs_builder = @brm begin
        eta ~ 0 + x
        y_cumulative ~ Ordinal(
            Cumulative(), LogitLink(), eta; per_threshold=(treat,))
    end
    @test_throws ErrorException SBBRMI(cumulative_cs_builder(ordinal_df))

    bad_discrimination_df = merge(ordinal_df, (; bad_disc=fill(-1.0, 6)))
    bad_discrimination_builder = @brm begin
        eta ~ 0 + x
        y_cumulative ~ Ordinal(
            Cumulative(), ProbitLink(), eta;
            discrimination=bad_disc)
    end
    @test_throws ErrorException SBBRMI(
        bad_discrimination_builder(bad_discrimination_df))

    bad_keyword_builder = @brm begin
        eta ~ 0 + x
        y_cumulative ~ Ordinal(
            Cumulative(), ProbitLink(), eta; category_specific=(treat,))
    end
    @test_throws ErrorException bad_keyword_builder(ordinal_df)

    bad_arity_keyword_builder = @brm begin
        y_cumulative ~ Ordinal(
            Cumulative(), LogitLink(); discrimination=1.0)
    end
    @test_throws ErrorException SBBRMI(
        bad_arity_keyword_builder(ordinal_df); mod=@__MODULE__)
end

const RUN_ORDINAL_BRIDGESTAN =
    get(ENV, "BRM_ORDINAL_RUNTIME", "1") != "0"

if RUN_ORDINAL_BRIDGESTAN
    @testset "ordinal density, pointwise likelihood, and RNG execute" begin
        descriptor = brm_descriptor(
            ordinal_builder, ordinal_df; mod=@__MODULE__)
        cache = joinpath(tempdir(), "brm-ordinal-links")
        isdir(cache) || mkpath(cache)
        problem = brm_execute(
            descriptor, :instantiate;
            path=joinpath(cache, string(descriptor.id) * ".stan"))
        dimension = LogDensityProblems.dimension(problem)
        q = fill(0.05, dimension)
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == dimension
        @test all(isfinite, gradient)

        pointwise = brm_execute(
            descriptor, :pointwise_loglik;
            problem, draws=q, seed=20260729)
        @test length(pointwise.y_cumulative_likelihood) ==
              length(ordinal_df.y_cumulative)
        @test length(pointwise.y_stopping_likelihood) ==
              length(ordinal_df.y_stopping)
        @test all(isfinite, pointwise.y_cumulative_likelihood)
        @test all(isfinite, pointwise.y_stopping_likelihood)

        predictions = brm_execute(
            descriptor, :predict; problem, draws=q, seed=20260729)
        @test all(1 .<= predictions.y_cumulative_gen .<= 3)
        @test all(1 .<= predictions.y_stopping_gen .<= 3)

        # Reconstruct the executable Julia distributions from the same
        # constrained Stan draw and compare every pointwise log-pmf. This pins
        # the sign of eta, discrimination placement, and threshold-specific
        # effect convention across the Julia and Stan contracts.
        names = BridgeStan.param_names(
            problem.model; include_tp=true, include_gq=false)
        values = BridgeStan.param_constrain(
            problem.model, q; include_tp=true, include_gq=false)
        constrained = Dict(name => value for (name, value) in zip(names, values))
        cuts_cumulative = [
            constrained["y_cumulative_thresholds.$k"] for k in 1:2]
        cuts_stopping = [
            constrained["y_stopping_thresholds.$k"] for k in 1:2]
        for i in eachindex(ordinal_df.y_cumulative)
            eta = constrained["eta_cumulative.$i"]
            disc = constrained["discrimination.$i"]
            d = Ordinal(
                Cumulative(), ProbitLink(), eta, cuts_cumulative;
                discrimination=disc)
            @test pointwise.y_cumulative_likelihood[i] ≈
                  logpdf(d, ordinal_df.y_cumulative[i]) atol=1e-9

            stage_eta = [
                constrained["eta_stopping.$i"] +
                constrained["y_stopping_threshold_effect.$i.$k"]
                for k in 1:2
            ]
            ds = Ordinal(
                StoppingRatio(), CloglogLink(), stage_eta, cuts_stopping)
            @test pointwise.y_stopping_likelihood[i] ≈
                  logpdf(ds, ordinal_df.y_stopping[i]) atol=1e-9
        end
    end
else
    @info "Skipping ordinal BridgeStan execution (BRM_ORDINAL_RUNTIME=0)"
end
