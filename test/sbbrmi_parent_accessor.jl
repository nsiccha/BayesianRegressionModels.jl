# test/sbbrmi_parent_accessor.jl — public `parent(::SBBRMI)` seam + the
# `structure_of` / `priors_of` SBBRMI overloads.
#
# Run: julia --startup-file=no --project=. test/sbbrmi_parent_accessor.jl
#
# A downstream consumer that already holds the emitted `SBBRMI` (e.g. a web
# preview that renders the SLIC listing) must be able to reach the wrapped
# `BRMI` and split-print it WITHOUT reaching into the `SBBRMI.parent` field.
# `parent(::SBBRMI)` is the stable seam (mirrors `parent(::TuringBRMI)`), and
# `structure_of` / `priors_of` accept the wrapper directly, unwrapping through
# it. Both accessors are also exported (`using BayesianRegressionModels`).

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

df = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        subject=[1, 1, 2, 2, 3, 3],
        y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

brmi = @brm df begin
    log_ka ~ 1 + x + (1 | pk | subject)
    sigma ~ Exponential(1)
    y ~ Normal(log_ka, sigma)
    effect(log_ka, Intercept) ~ Normal(log(1 / 8), 0.8)
    sd(:, pk) ~ Exponential(1)
end
sb = SBBRMI(brmi; mod=@__MODULE__)

@testset "structure_of / priors_of are exported" begin
    @test Base.isexported(BayesianRegressionModels, :structure_of)
    @test Base.isexported(BayesianRegressionModels, :priors_of)
end

@testset "parent(::SBBRMI) is the public accessor" begin
    @test parent(sb) === sb.parent
    @test parent(sb) isa BayesianRegressionModels.BRMI
    # It is the exact BRMI the wrapper was built from.
    @test parent(sb) === brmi
end

@testset "structure_of / priors_of accept an SBBRMI directly" begin
    @test structure_of(sb) isa BayesianRegressionModels.BRMI
    @test priors_of(sb) isa BayesianRegressionModels.BRMI
    # The SBBRMI overloads must equal the BRMI-taking path they unwrap to.
    @test sprint(show, structure_of(sb)) == sprint(show, structure_of(parent(sb)))
    @test sprint(show, priors_of(sb)) == sprint(show, priors_of(parent(sb)))
    # And the split is non-trivial: the prior fragment holds the effect()/sd()
    # statements, the structural fragment holds the LP + likelihood.
    @test collect(keys(priors_of(sb).operations)) == collect(keys(priors_of(brmi).operations))
    @test !isempty(keys(priors_of(sb).operations))
    @test !isempty(keys(structure_of(sb).operations))
end
