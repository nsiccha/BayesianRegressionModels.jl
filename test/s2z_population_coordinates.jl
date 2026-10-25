# test/s2z_population_coordinates.jl — the population coefficients of a
# predictor whose sum-to-zero block (`s2z_groups`) absorbed them (snag
# `reading-the-popu-ad4f22fe`). The block integrates its means `μ` into the
# sampled `theta = β + B μ`, so `brm_population_effect_coordinates` resolves an
# absorbed coefficient to the recovered conventional `β = theta - B μ` in
# generated quantities (`recovered = true`). It pairs with the recovered
# effects `r + μ` that `brm_ranef_draws` reports for the block.
#
# The oracle is each row's emitted linear predictor: for every draw and row it
# must equal the resolved population coefficients plus the row's level effects
# (plus the offset), which holds only when both sides use the same `μ` draw.
# The sampled `theta` with those effects is checked to MISS it, so the oracle
# discriminates the recovered frame from the sampled one.
#
# Run: julia --project=test test/s2z_population_coordinates.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Random

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan

rng = MersenneTwister(5)
const N = 24
data = (; g = repeat(["a", "b", "c", "d"], N ÷ 4), x = randn(rng, N), y = randn(rng, N))
level_of(levels, labels) = [findfirst(==(l), levels) for l in labels]

# The reported shape: intercept-only location and log-scale predictors, each
# with an S2Z block of the same grouping, the scale with an offset.
intercept_builder = @brm begin
    mu ~ 1 + (1 | g)
    log(sigma) ~ 1 + offset(x) + (1 | g)
    y ~ Normal(mu, sigma)
end
# The block absorbs the intercept; the slope stays a sampled coefficient.
slope_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | g)
    y ~ Normal(mu, sigma)
end
# The block absorbs both coefficients.
nocor_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x || g)
    y ~ Normal(mu, sigma)
end

build(builder; kw...) = SBBRMI(builder(data); mod = @__MODULE__, kw...)
const S2Z = (; s2z_groups = [:g], s2z_rho = 1.0)

function constrained_draws(sb; n_draws = 4, seed = 1)
    sm = BRM.stan_instantiate(sb).model
    names = BS.param_names(sm; include_tp = true, include_gq = true)
    r = MersenneTwister(seed)
    rows = map(1:n_draws) do i
        BS.param_constrain(sm, 0.7 .* randn(r, BS.param_unc_num(sm));
                           include_tp = true, include_gq = true,
                           rng = BS.StanRNG(sm, seed + i))
    end
    (; sm, names, draws = permutedims(reduce(hcat, rows)),
       col = Dict(n => j for (j, n) in enumerate(names)), d = brm_descriptor(sb))
end
value(c, i, name) = c.draws[i, c.col[name]]
s2z_block(sb, lp) = only(b for b in s2z_effect_blocks(sb) if b.predictor === lp)
ranef(c, lp) = [b for b in brm_ranef_draws(c.d, c.draws, c.names; group = :g)
                if only(b.margins).predictor === lp]

@testset "intercept-only predictors resolve their recovered intercept" begin
    sb = build(intercept_builder; S2Z...)
    c = constrained_draws(sb)
    for (lp, row, link, inverse_link, offset) in
            ((:mu, "mu", identity, identity, zeros(N)),
             (:sigma, "log_sigma", log, exp, data.x))
        block = s2z_block(sb, lp)
        q = brm_population_effect_coordinates(c.d, lp, c.names; coefficient = :Intercept)
        @test q.recovered === true
        @test q.output.name === block.coefficients
        @test q.output.kind === :generated_quantity
        @test q.output.role === :population_effect
        @test q.output.labels == [:Intercept]
        @test q.output.logical === lp
        @test q.output.predictors == (lp,)
        @test q.coordinates == [c.col["$(block.coefficients).1"]]
        @test q.link === link
        @test q.inverse_link === inverse_link
        re = only(ranef(c, lp))
        lv = level_of(re.levels, data.g)
        theta = c.col["$(block.population).1"]
        pairs_recovered = true
        theta_misses = false
        for i in axes(c.draws, 1), r in 1:N
            effect = re.effects[i, lv[r], 1]
            lin = value(c, i, "$row.$r")
            pairs_recovered &= isapprox(c.draws[i, only(q.coordinates)] + effect + offset[r],
                                        lin; atol = 1e-10)
            theta_misses |= !isapprox(c.draws[i, theta] + effect + offset[r], lin; atol = 1e-6)
        end
        @test pairs_recovered
        @test theta_misses
        lp === :sigma && @test all(i -> value(c, i, "sigma.1") ≈
            q.inverse_link(c.draws[i, only(q.coordinates)] + re.effects[i, lv[1], 1] +
                           data.x[1]), axes(c.draws, 1))
    end
    # The recovered intercept is a generated quantity: names without them are
    # refused with the remedy, not resolved to an empty slice.
    tp_names = BS.param_names(c.sm; include_tp = true, include_gq = false)
    @test_throws "include_gq=true" brm_population_effect_coordinates(
        c.d, :mu, tp_names; coefficient = :Intercept)
    # An unknown coefficient names what the predictor has.
    @test_throws r"available labels are \(:Intercept,\)" brm_population_effect_coordinates(
        c.d, :mu, c.names; coefficient = :x)
    # A start value for the absorbed intercept sets the sampled `theta` with the
    # block means at zero: with zero contrasts every row starts at it.
    u = brm_start(sb, c.sm, (; mu = (; Intercept = 0.3), sigma = (; Intercept = -0.2)))
    th = BS.param_constrain(c.sm, u; include_tp = true)
    tn = BS.param_names(c.sm; include_tp = true)
    at(name) = th[findfirst(==(name), tn)]
    @test at("$(s2z_block(sb, :mu).population).1") ≈ 0.3
    @test at("$(s2z_block(sb, :sigma).population).1") ≈ -0.2
    @test all(r -> at("mu.$r") ≈ 0.3, 1:N)
    @test all(r -> at("log_sigma.$r") ≈ -0.2 + data.x[r], 1:N)
end

@testset "a sampled slope beside an absorbed intercept" begin
    sb = build(slope_builder; S2Z...)
    c = constrained_draws(sb)
    block = s2z_block(sb, :mu)
    @test block.population_columns == (:Intercept,)
    q0 = brm_population_effect_coordinates(c.d, :mu, c.names; coefficient = :Intercept)
    qx = brm_population_effect_coordinates(c.d, :mu, c.names; coefficient = :x)
    @test q0.recovered === true
    @test q0.output.name === block.coefficients
    @test qx.recovered === false
    @test qx.output.kind === :parameter
    @test qx.output.name !== block.population
    re = only(ranef(c, :mu))
    lv = level_of(re.levels, data.g)
    @test all(isapprox(c.draws[i, only(q0.coordinates)] + c.draws[i, only(qx.coordinates)] * data.x[r] +
                       re.effects[i, lv[r], 1], value(c, i, "mu.$r"); atol = 1e-10)
              for i in axes(c.draws, 1), r in 1:N)
    @test_throws r"available labels are \(:x, :Intercept\)" brm_population_effect_coordinates(
        c.d, :mu, c.names; coefficient = :z)
    u = brm_start(sb, c.sm, (; mu = (; Intercept = 0.5, x = 2.0)))
    th = BS.param_constrain(c.sm, u; include_tp = true)
    tn = BS.param_names(c.sm; include_tp = true)
    @test all(r -> th[findfirst(==("mu.$r"), tn)] ≈ 0.5 + 2.0 * data.x[r], 1:N)
end

@testset "a block absorbing both coefficients" begin
    sb = build(nocor_builder; S2Z...)
    c = constrained_draws(sb)
    block = s2z_block(sb, :mu)
    @test block.population_columns == (:Intercept, :x)
    q = [brm_population_effect_coordinates(c.d, :mu, c.names; coefficient = k)
         for k in (:Intercept, :x)]
    @test all(r -> r.recovered === true && r.output.name === block.coefficients, q)
    @test [only(r.coordinates) for r in q] ==
        [c.col["$(block.coefficients).$k"] for k in 1:2]
    re = ranef(c, :mu)
    @test [only(b.margins).coefficient for b in re] == [:Intercept, :x]
    lv = level_of(re[1].levels, data.g)
    @test all(isapprox(c.draws[i, only(q[1].coordinates)] + re[1].effects[i, lv[r], 1] +
                       (c.draws[i, only(q[2].coordinates)] + re[2].effects[i, lv[r], 1]) * data.x[r],
                       value(c, i, "mu.$r"); atol = 1e-10)
              for i in axes(c.draws, 1), r in 1:N)
end

@testset "the same address without s2z_groups is the sampled coefficient" begin
    # `total_groups = ()`: automatic exact totals would absorb the intercepts.
    sb = build(intercept_builder; total_groups = ())
    @test isempty(s2z_effect_blocks(sb))
    @test isempty(total_effect_blocks(sb))
    c = constrained_draws(sb; n_draws = 1)
    for lp in (:mu, :sigma)
        q = brm_population_effect_coordinates(c.d, lp, c.names; coefficient = :Intercept)
        @test q.recovered === false
        @test q.output.kind === :parameter
    end
end
