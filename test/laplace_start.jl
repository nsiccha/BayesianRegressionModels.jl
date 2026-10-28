# `brm_start` with no initial values: Pathfinder over the per-unit Laplace
# marginal of the population coordinates, each unit's random effects drawn
# from its Laplace Gaussian there (src/laplace_start.jl).
# RUN: julia --project=test test/laplace_start.jl
#
# Checked: loading WarmupHMC loads the Pathfinder extension; on well-separated
# groups every group starts near its own mean and the block's SD near the
# groups' spread, where the zero base starts every group at the population
# value; `chains` gives distinct starts; the Stan start is the RK start moved
# by the coordinate transport; initial values go on top of a start through
# `base`; a plate-cell model (the whole-model Laplace) starts at a finite
# point.
using WarmupHMC
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using BridgeStan, LogDensityProblems, Random, Statistics
import StanBlocks

const LAPLACE_START_CACHE = mktempdir()
compiled(sb, name) = StanBlocks.stan_instantiate(sb.model;
    path=joinpath(LAPLACE_START_CACHE, "$name.stan")).model

# 24 groups far apart (SD 2) against a residual SD of 0.3.
const SEPARATED = let rng = Xoshiro(20261028)
    levels = [string("g", lpad(i, 2, '0')) for i in 1:24]
    means = Dict(l => 5 + 2 * randn(rng) for l in levels)
    g = repeat(levels; inner=20)
    x = randn(rng, length(g))
    y = [means[gi] for gi in g] .+ 0.5 .* x .+ 0.3 .* randn(rng, length(g))
    (; data=(; g, x, y), means)
end

separated(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | lv | g)
    effect(mu, Intercept) ~ Normal(0, 10)
    y ~ Normal(mu, sigma)
end

# Each group's effect and the block's SD at a Stan point.
function block_at(sb, model, s)
    names = BridgeStan.param_names(model; include_tp=true, include_gq=true)
    θ = BridgeStan.param_constrain(model, s; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(model, 1))
    b = only(brm_ranef_draws(brm_descriptor(sb), permutedims(θ), names; id=:lv))
    (; effects=Dict(zip(b.levels, b.effects[1, :, 1])), sd=b.sd[1, 1],
        sigma=θ[findfirst(==("sigma"), names)])
end

@stestset "loading WarmupHMC loads the Pathfinder extension" begin
    @test Base.get_extension(BRM, :BayesianRegressionModelsPathfinderExt) !== nothing
end

@stestset "every group starts near its own mean, the block's SD near their spread" begin
    brmi = separated(SEPARATED.data)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    model = compiled(sb, "separated")
    @test brm_laplace_problem(rk; ad_backend=NLME_AD) isa BRM.BRMStructuredLaplaceProblem

    us = brm_start(rk; chains=3, rng=Xoshiro(1))
    @test length(us) == 3 && all(u -> u isa Vector{Float64}, us)
    @test all(u -> length(u) == length(BRM._rk_layout_coordinate_names(rk)), us)
    @test allunique(us)
    ℓ = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    @test all(u -> isfinite(LogDensityProblems.logdensity(ℓ, u)), us)
    @test brm_start(rk; rng=Xoshiro(1)) == us[1]

    # The Stan starts are the RK starts, moved by the coordinate transport.
    ss = brm_start(sb, model; chains=3, rng=Xoshiro(1), rk)
    t = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(model))
    @test all(k -> ss[k] ≈ brm_rk_point_to_stan(t, us[k]), 1:3)

    m = mean(values(SEPARATED.means))
    spread = std(collect(values(SEPARATED.means)))
    zero = brm_start(sb, model, (;))
    @test all(iszero, values(block_at(sb, model, zero).effects))
    for s in ss
        @test isfinite(BridgeStan.log_density(model, s))
        @test BridgeStan.log_density(model, s) > BridgeStan.log_density(model, zero)
        b = block_at(sb, model, s)
        levels = sort!(collect(keys(SEPARATED.means)))
        @test cor([b.effects[l] for l in levels], [SEPARATED.means[l] - m for l in levels]) > 0.99
        @test 0.5 < b.sd / spread < 2
        @test 0.15 < b.sigma < 0.6
    end
end

@stestset "initial values go on top of a start through base" begin
    brmi = separated(SEPARATED.data)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    model = compiled(sb, "separated_inits")
    start = brm_start(sb, model; rng=Xoshiro(2), rk)
    s = brm_start(sb, model, (; sigma=0.5); base=start)
    at, given = block_at(sb, model, start), block_at(sb, model, s)
    @test given.sigma ≈ 0.5 rtol=1e-12
    @test given.sd ≈ at.sd rtol=1e-12
    @test all(l -> given.effects[l] ≈ at.effects[l], keys(at.effects))
end

@stestset "a plate-cell model starts through the whole-model Laplace" begin
    rk = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    @test brm_laplace_problem(rk; ad_backend=NLME_AD) isa BRM.BRMLaplaceProblem
    u = brm_start(rk; rng=Xoshiro(3))
    ℓ = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    @test isfinite(LogDensityProblems.logdensity(ℓ, u))
end

@stestset "a model without random-effect units" begin
    flat(data) = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    rk = RKBRMI(flat(SEPARATED.data))
    # Capability gap, not a refusal: a model with no units has no Laplace
    # marginal yet; Pathfinder from its prior medians would serve it (BRM todo
    # `1owh97g`).
    @test_broken brm_start(rk; rng=Xoshiro(4)) isa Vector{Float64}
end
