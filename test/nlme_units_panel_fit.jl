# test/nlme_units_panel_fit.jl — NLMEEstimation on the regional panel of
# test/nlme_panel_fixture.jl: a size-stratified block estimated as one family
# per stratum, non-centered completions held at unit variance
# (`NLMEEstimation.fixed_omega`), the population prior over both, and the
# two-stage fit (SAEM for the population, every region sampled on its own).
#
# NLMEEstimation is not in test/Project.toml: run this file in the
# NLMEEstimation scratch environment described in test/README.md.
#
#     julia --project=<scratch-env> test/nlme_units_panel_fit.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_panel_fixture.jl"))
using NLMEEstimation, LogDensityProblems, LinearAlgebra, Random, Statistics
import WarmupHMC
const NE = NLMEEstimation

panel_model() = brm_nlme_model(RKBRMI(panel(PANEL); completions=:noncentered);
    ad_backend=NLME_AD, subject_group=:region, nested_groups=PANEL_NESTED)

@stestset "completion families are fixed at unit variance; strata are estimated" begin
    m = panel_model()
    fixed = NE.fixed_omega(m)
    completion = findfirst(f -> f.fixed !== nothing, m.view.families)
    @test length(fixed) == 1 && first(only(fixed)) == completion
    @test last(only(fixed)).sd == [1.0] && last(only(fixed)).correlation == fill(1.0, 1, 1)
    start = brm_nlme_start(m)
    @test start.omega_matrix[completion] == fill(1.0, 1, 1)
    # Every stratum's scale starts at the prior median of its margin's law,
    # `LogNormal(-1, 0.75)`.
    for (f, family) in enumerate(m.view.families)
        family.stratum === nothing && continue
        @test start.omega_matrix[f] ≈ fill(exp(-1)^2, 1, 1)
    end
    @test isfinite(start.population_logprior)
end

@stestset "the population prior's ω gradient skips the fixed family" begin
    m = panel_model()
    start = brm_nlme_start(m)
    θ, σ, ω = start.theta, start.sigma, copy(start.omega)
    rng = Xoshiro(3)
    # The fixed family's ω entries stay at its value; perturb the others.
    probe = NE.apply_fixed_omega(m, ω .+ 0.05 .* randn(rng, length(ω)))
    value, _, _, gω = NE.population_logprior_and_gradient(m, θ, σ, probe)
    @test value ≈ NE.population_logprior(m, θ, σ, probe)
    step = 1e-6
    for j in eachindex(probe)
        e = zeros(length(probe)); e[j] = step
        up, down = NE.apply_fixed_omega(m, probe .+ e), NE.apply_fixed_omega(m, probe .- e)
        up == down && continue # a fixed entry
        fd = (NE.population_logprior(m, θ, σ, up) - NE.population_logprior(m, θ, σ, down)) / 2step
        @test gω[j] ≈ fd atol=1e-5 rtol=1e-5
    end
end

@stestset "estimates name a stratified block by its strata" begin
    m = panel_model()
    start = brm_nlme_start(m)
    named = brm_nlme_estimates(m, start.theta, start.sigma, start.omega_matrix)
    stratified = only(v for v in values(named.point) if v isa NamedTuple && hasproperty(v, :strata))
    @test stratified.by === :sclass
    @test first.(stratified.strata) == [1, 2]
    @test all(s -> last(s).sd ≈ [exp(-1)], stratified.strata)
end

# The family of each coordinate in subject `i`'s compact block.
_instances(view, i) = reduce(vcat, (fill(f, view.family_sizes[f]) for f in view.subject_families[i]))

function warmup_sampler(unit)
    run = WarmupHMC.adaptive_warmup_mcmc(Random.Xoshiro(unit.subject), unit)
    Matrix(run.posterior_position)
end

@stestset "two-stage fit of the panel" begin
    rk = RKBRMI(panel(PANEL); completions=:noncentered)
    seconds = @elapsed fit = brm_two_stage_fit(warmup_sampler, rk; threads=1,
        subject_group=:region, nested_groups=PANEL_NESTED, rng=Random.Xoshiro(11))
    @info "two-stage fit of the panel" seconds
    m = fit.stage1.model
    @test size(fit.draws, 1) == length(m.view.coordinates)
    @test all(isfinite, fit.draws)
    # The completions are sampled with their unit: their draws vary.
    k = findfirst(f -> f.fixed !== nothing, m.view.families)
    completion = [c for (i, block) in enumerate(m.view.subject_coordinates)
        for (c, f) in zip(block, _instances(m.view, i)) if f == k]
    @test length(completion) == count(ismissing, PANEL.x1)
    @test all(c -> std(fit.draws[c, :]) > 0, completion)
end
