# test/interval_censored_predictor.jl — predictor-side interval evidence.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems

const ICP_CACHE = joinpath(tempdir(), "brm-interval-censored-predictor")
const ICP_RUNTIME = get(ENV, "BRM_INTERVAL_PREDICTOR_RUNTIME", "1") != "0"

const ICP_DF = (;
    y=[401.0, 408.0, 415.0, 411.0, 419.0, 423.0],
    conc=[0.7, 0.25, 1.2, 0.4, 1.8, 0.15],
    lloq=[0.25, 0.25, 0.4, 0.4, 0.15, 0.15],
    subject=[1, 1, 2, 2, 3, 3],
)

icp_model(df=ICP_DF) = @brm df begin
    y ~ Normal(mu, 1.)
    mu ~ 1 + interval_censored(conc; upper=lloq) +
          (1 + interval_censored(conc; upper=lloq) | subject)
    effect(mu, conc) ~ Normal(0., 3.)
    latent(mu, interval_censored(conc)) ~ Normal(0.5, 2.)
end

@testset "interval-censored predictor lowering and replay" begin
    brmi = icp_model()
    spec = only(term_priors(brmi))
    @test spec.class === :term_latent
    @test spec.term === Symbol("interval_censored(conc)")
    @test popcoefnames(brmi, :mu) == [:Intercept, :conc]
    @test [(p.predictor, p.coefficient) for p in effect_priors(brmi)] ==
          [(:mu, :conc)]

    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    suffix = :conc_lloq
    exact_key = Symbol(:icp_exact_, suffix)
    lower_key = Symbol(:icp_lower_, suffix)
    upper_key = Symbol(:icp_upper_, suffix)
    exact_index_key = Symbol(:Jexact_, suffix)
    interval_index_key = Symbol(:Jinterval_, suffix)
    @test sb.data[exact_key] == [0.7, 1.2, 1.8]
    @test sb.data[lower_key] == [0.0, 0.0, 0.0]
    @test sb.data[upper_key] == [0.25, 0.4, 0.15]
    @test sb.data[exact_index_key] == [1, 3, 5]
    @test sb.data[interval_index_key] == [2, 4, 6]

    # The fixed and random slopes reference one merged predictor. There is one
    # bounded parameter vector and one prior, even though the term appears twice.
    latent_name = "interval_censored_conc_lloq_x_interval"
    @test length(findall(latent_name * " ~ conditioning_normal", code)) == 1
    @test length(findall("vector<lower=" * String(lower_key), code)) == 1
    @test !occursin("normal(" * latent_name * ",", code) # no error-in-variable likelihood
    @test length(findall("conc", code)) > 3

    changed = merge(ICP_DF, (;
        conc=[0.1, 0.3, 1.4, 0.5, 2.0, 0.2],
        lloq=[0.1, 0.3, 0.4, 0.5, 0.15, 0.2],
    ))
    replay = reprocess(sb, changed)
    @test replay.data[exact_key] == [1.4, 2.0]
    @test replay.data[lower_key] == [0.0, 0.0, 0.0, 0.0]
    @test replay.data[upper_key] == [0.1, 0.3, 0.5, 0.2]
    @test replay.data[exact_index_key] == [3, 5]
    @test replay.data[interval_index_key] == [1, 2, 4, 6]
end

@testset "bounded latent predictor has finite density and gradient" begin
    if ICP_RUNTIME
        sb = SBBRMI(icp_model(); mod=@__MODULE__)
        code = StanBlocks.stan_code(sb.model)
        mkpath(ICP_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model; path=joinpath(ICP_CACHE, string(hash(code)) * ".stan"))
        n = LogDensityProblems.dimension(problem)
        q = [0.03 * ((i % 7) - 3) for i in 1:n]
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test length(gradient) == n
        @test all(isfinite, gradient)
    else
        @info "Skipping BridgeStan interval-predictor runtime gate"
        @test true
    end
end

@testset "all rows may be interval-censored" begin
    all_interval = merge(ICP_DF, (;
        conc=fill(0.25, 6),
        lloq=fill(0.25, 6),
    ))
    sb = SBBRMI(icp_model(all_interval); mod=@__MODULE__)
    @test isempty(sb.data[:icp_exact_conc_lloq])
    @test StanBlocks.stanc_check(
        StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
end

@testset "invalid interval evidence is refused" begin
    bad_order = merge(ICP_DF, (;
        conc=[0.7, 0.2, 1.2, 0.4, 1.8, 0.15],
    ))
    @test_throws "row 2 has `conc`" SBBRMI(icp_model(bad_order); mod=@__MODULE__)

    all_exact = merge(ICP_DF, (; conc=[0.7, 0.6, 1.2, 0.8, 1.8, 0.5]))
    @test_throws "has no BLOQ rows" SBBRMI(icp_model(all_exact); mod=@__MODULE__)

    missing_upper = @brm ICP_DF begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + interval_censored(conc)
    end
    @test_throws "requires `upper`" SBBRMI(missing_upper; mod=@__MODULE__)

    invalid_lower = @brm ICP_DF begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + interval_censored(conc; upper=lloq, lower=0.3)
    end
    @test_throws "requires lower < LLOQ" SBBRMI(invalid_lower; mod=@__MODULE__)
end

@testset "bounded latent model differs from LLOQ/2 substitution" begin
    half_df = merge(ICP_DF, (;
        conc_half=[0.7, 0.125, 1.2, 0.2, 1.8, 0.075],
    ))
    half = @brm half_df begin
        y ~ Normal(mu, 1.)
        mu ~ 1 + conc_half
    end
    half_code = StanBlocks.stan_code(SBBRMI(half; mod=@__MODULE__).model)
    interval_code = StanBlocks.stan_code(SBBRMI(icp_model(); mod=@__MODULE__).model)
    @test !occursin("conditioning_normal", half_code)
    @test !occursin("icp_lower_conc_lloq", half_code)
    @test occursin("conditioning_normal", interval_code)
    @test occursin("icp_lower_conc_lloq", interval_code)
end
