# Compare physical changes within a station to the full posterior. Variation in
# the differences depends on both the variances and the covariances of states.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface, Enzyme, WarmupHMC, LinearAlgebra, Random, Statistics
import MCMCDiagnosticTools
const BRM = BayesianRegressionModels
include("svi_field_fixture.jl")

@testset "within-station changes under unit-owned GMRF centering" begin
    data = station_data()
    ad = AutoEnzyme(; mode=Enzyme.Reverse)
    rk = RKBRMI(field_model(data))
    m = brm_nlme_model(rk; ad_backend=ad, subject_group=:site, unit_partition=:connected)
    problem = rk_logdensity_problem(rk; ad_backend=ad)
    u0 = first(BRM._brm_nlme_prior_median_point(m))
    chains = map(1:4) do chain
        rng = Xoshiro(100 + chain)
        init = (; position=u0 + 0.05 .* randn(rng, length(u0)), squared_scale=ones(length(u0)))
        run = adaptive_warmup_mcmc(rng, problem; init, n_draws=1000,
            target_acceptance_rate=0.90, divergence_tolerance=0.0)
        Matrix(run.posterior_position)
    end
    n = minimum(chain -> size(chain, 2), chains)
    draws = permutedims(stack([chain[:, end-n+1:end] for chain in chains]), (2, 3, 1))
    @test maximum(MCMCDiagnosticTools.rhat(draws)) < 1.05
    @test minimum(MCMCDiagnosticTools.ess(draws; kind=:bulk)) > 100
    reference = reduce(hcat, chains)

    record = BRM._brm_gmrf_footprint_record(rk, :field)
    names = m.view.coordinates
    amplitude = only(findall(==(Symbol(record.declaration, ".", record.locals.sigma)), names))
    footprint = BRM._brm_vecchia_footprint(data.east, data.north, data.time; neighbors=data.nb)
    states = [only(findall(==(Symbol(record.declaration, ".", record.locals.field, ".", k)), names))
        for k in eachindex(footprint.position)]
    pairs = Tuple{Int,Int}[]
    for site in unique(footprint.position)
        rows = findall(==(site), footprint.position)
        append!(pairs, zip(states[rows[1:end-1]], states[rows[2:end]]))
    end
    @test length(pairs) == 12
    changes(draws) = stack([exp.(draws[amplitude, :]) .* (draws[k, :] - draws[j, :])
        for (j, k) in pairs]; dims=1)
    target = cov(changes(reference); dims=2)
    sd = sqrt.(diag(target))
    function covariance_error(draws)
        delta = (cov(changes(draws); dims=2) - target) ./ (sd * sd')
        sqrt(mean(abs2(delta[i, j]) for i in eachindex(sd) for j in 1:i-1))
    end
    options = (; field_states=:units, separator=:none, batch_size=4, ndraws=2000, threads=1)
    for seed in (5, 6)
        plain = brm_svi_fit(m; options..., rng=Xoshiro(seed), centering=nothing, coupling=nothing)
        learned = brm_svi_fit(m; options..., rng=Xoshiro(seed),
            centering=LearnedCentering(; fields=true))
        @test count(cell -> cell.kind === :gmrf, learned.diagnostics.centering.cells) == length(states)
        plain_error, learned_error = covariance_error(plain.draws), covariance_error(learned.draws)
        @info "Within-station covariance error" seed plain_error learned_error
        @test learned_error < 0.75 * plain_error
    end
end
