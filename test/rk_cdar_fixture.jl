# test/rk_cdar_fixture.jl — public synthetic data and models with a
# `cdar(step; by=g, cor=C)` walk over districts nested in regions, shared by
# test/rk_cdar.jl and test/rk_cdar_fit.jl. Include it after nlme_fixtures.jl,
# with LinearAlgebra and Random loaded.

# Three regions (the units), six districts nested in them, four years. District
# `k3` has no row in year 2, `k6` none in years 1 and 4: the walk passes
# through those cells. `C` is the identity, `C2` correlates districts within a
# region only, `C3` also across regions.
const KREIS = ["k1", "k2", "k3", "k4", "k5", "k6"]
const REGION_OF = Dict("k1" => "r1", "k2" => "r1", "k3" => "r2", "k4" => "r2",
    "k5" => "r3", "k6" => "r3")
const CELLS = [(k, t) for t in 1:4 for k in KREIS
    if !((k, t) in (("k3", 2), ("k6", 1), ("k6", 4)))]
blockcor(rho_within, rho_across) = [a == b ? 1.0 : REGION_OF[a] == REGION_OF[b] ?
    rho_within : rho_across for a in KREIS, b in KREIS]
const WALK_DF = let rng = Xoshiro(3), n = length(CELLS)
    (; kreis=first.(CELLS), year=last.(CELLS), region=[REGION_OF[k] for k in first.(CELLS)],
       x=round.(randn(rng, n); digits=3), y=round.(randn(rng, n); digits=3),
       C=Matrix{Float64}(I, 6, 6), C2=blockcor(0.4, 0.0), C3=blockcor(0.4, 0.2))
end

walk_model(df=WALK_DF) = @brm df begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | region) + cdar(year; by=kreis, cor=C)
    y ~ Normal(mu, sigma)
end
correlated_model(df=WALK_DF) = @brm df begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | region) + cdar(year; by=kreis, cor=C2)
    sd(mu, cdar(year)) ~ Exponential(2.0)
    ar(mu, cdar(year)) ~ Beta(2.0, 2.0)
    y ~ Normal(mu, sigma)
end
across_model(df=WALK_DF) = @brm df begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | region) + cdar(year; by=kreis, cor=C3)
    ar(mu, cdar(year)) ~ Normal(0.7, 0.1)
    y ~ Normal(mu, sigma)
end
known_model(df=WALK_DF) = @brm df begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | region) + cdar(year; by=kreis, cor=C)
    sd(mu, cdar(year)) = 0.4
    y ~ Normal(mu, sigma)
end
# Grouped by the units themselves.
region_model(df=WALK_DF) = @brm (; df..., C=Matrix{Float64}(I, 3, 3)) begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | region) + cdar(year; by=region, cor=C)
    y ~ Normal(mu, sigma)
end
# The walk alone, for the recurrence oracle.
bare_model(df=WALK_DF) = @brm df begin
    mu ~ 0 + cdar(year; by=kreis, cor=C2)
    y ~ Normal(mu, 1.0)
end

const NESTED = (; subject_group=:region, nested_groups=(kreis=:region,))

cdar_family(v) = findfirst(f -> hasproperty(f.block, :cdar), v.families)
