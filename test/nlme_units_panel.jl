# test/nlme_units_panel.jl — the NLME unit factorization (`brm_nlme_view`) of a
# regional panel: regions (the units) ⊃ districts ⊃ municipalities and
# district-years. One response is completed where it is missing (`mi`), its
# municipality effect is stratified by size class (`gr(muni, by=sclass)`), and a second
# quantity is observed only as district-year totals (a ragged join). Every
# region's random effects, completions and totals belong to that region alone.
#
# Covered: per-stratum covariance families, non-centered completions as unit
# draws in a family fixed at unit variance, totals cells attributed to the unit
# owning their rows, and unit problems whose densities match the full RK density.
#
# Run: julia --project=test test/nlme_units_panel.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra, Random

include(joinpath(@__DIR__, "nlme_panel_fixture.jl"))

panel_view(rk) = brm_nlme_view(rk; subject_group=:region, nested_groups=PANEL_NESTED)

@stestset "a stratified block is one covariance family per stratum" begin
    rk = RKBRMI(panel(PANEL); completions=:noncentered)
    v = panel_view(rk)
    @test !any(==(:unclaimed), v.role)
    stratified = [f for f in v.families if f.stratum !== nothing]
    @test length(stratified) == 2
    @test [f.block.stratum for f in stratified] == [1, 2]
    @test all(f -> f.size == 1 && length(f.scales) == 1 && isempty(f.factor), stratified)
    @test all(i -> v.role[i] === :subject_scale, reduce(vcat, (f.scales for f in stratified)))
    # Each municipality's stratified effect sits in its size class's family,
    # in the region that owns it.
    family_of = Dict(f.block.stratum => k for (k, f) in enumerate(v.families) if f.stratum !== nothing)
    for (i, margins) in enumerate(v.subject_margins)
        instances = [k for k in v.subject_families[i] if v.families[k].stratum !== nothing]
        munis = [b.level for b in margins if hasproperty(b.block, :stratum)]
        @test length(instances) == length(munis) == 2 * 4
        for (k, muni) in zip(instances, munis)
            row = findfirst(==(muni), PANEL.muni)
            @test k == family_of[PANEL.sclass[row]]
            @test PANEL.region[row] == i
        end
    end
end

@stestset "a non-centered completion is one unit draw per missing cell" begin
    rk = RKBRMI(panel(PANEL); completions=:noncentered)
    v = panel_view(rk)
    missing_rows = findall(ismissing, PANEL.x1)
    completion = only(f for f in v.families if f.fixed !== nothing)
    @test completion.fixed == fill(1.0, 1, 1) && completion.size == 1
    k = findfirst(f -> f.fixed !== nothing, v.families)
    for i in eachindex(v.levels)
        @test count(==(k), v.subject_families[i]) ==
            count(r -> PANEL.region[r] == i, missing_rows)
    end
    @test count(==(:shared_latent), v.role) == 0
    # The default (centered) completion stays shared by all units.
    centered = panel_view(RKBRMI(panel(PANEL)))
    @test count(==(:shared_latent), centered.role) == length(missing_rows)
    @test all(f -> f.fixed === nothing, centered.families)
end

@stestset "a non-location-scale completion law is refused when non-centered" begin
    gamma_completion(data) = @brm data begin
        mu ~ 1 + z + (1 | region)
        mi(x1) ~ Gamma(2.0, exp(mu))
        c_x = log(x1)
        log_w ~ 1 + c_x
        sigma_t ~ Exponential(0.03)
        rows_t = groups(district_year => dy_obs)
        @plate for k in eachindex(total)
            total[k] ~ Normal(sum(exp.(log_w[rows_t[k]])), sigma_t * total[k])
        end
    end
    @test_throws "location-scale law" RKBRMI(gamma_completion(PANEL); completions=:noncentered)
end

@stestset "district-year totals belong to the region owning their rows" begin
    rk = RKBRMI(panel(PANEL); completions=:noncentered)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:region,
        nested_groups=PANEL_NESTED)
    totals = only(last(a) for a in m.attribution if first(a) === :total)
    @test totals == [PANEL.region[findfirst(==(c), PANEL.district_year)]
        for c in PANEL.dy_obs]
end

@stestset "unit problems differ from the full RK density by their unit alone" begin
    rk = RKBRMI(panel(PANEL); completions=:noncentered)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:region,
        nested_groups=PANEL_NESTED)
    v = m.view
    rng = Xoshiro(7)
    θ = 0.1 .* randn(rng, length(m.theta))
    σ = 0.1 .* randn(rng, length(m.sigma))
    # At the unit point every scale is 1 and every factor the identity, so
    # Ω = I per family and a unit's coordinates are the RK subject draws.
    Ω = [Matrix{Float64}(I, K, K) for K in v.family_sizes]
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    others = [0.3 .* randn(rng, length(b)) for b in v.subject_coordinates]
    function point(i, z)
        u = copy(m.unit)
        u[m.theta] = θ
        u[m.sigma] = σ
        for (j, block) in enumerate(v.subject_coordinates)
            u[block] = j == i ? z : others[j]
        end
        u
    end
    for i in eachindex(v.levels)
        p = brm_nlme_unit_problem(m, i, θ, σ, Ω)
        d = LogDensityProblems.dimension(p)
        @test d == length(v.subject_coordinates[i])
        z1, z2 = 0.3 .* randn(rng, d), 0.3 .* randn(rng, d)
        @test LogDensityProblems.logdensity(p, z1) - LogDensityProblems.logdensity(p, z2) ≈
              LogDensityProblems.logdensity(full, point(i, z1)) -
              LogDensityProblems.logdensity(full, point(i, z2)) atol=1e-8
        value, gradient = LogDensityProblems.logdensity_and_gradient(p, z1)
        _, full_gradient = LogDensityProblems.logdensity_and_gradient(full, point(i, z1))
        @test value ≈ LogDensityProblems.logdensity(p, z1)
        @test gradient ≈ full_gradient[v.subject_coordinates[i]] atol=1e-8
    end
end
