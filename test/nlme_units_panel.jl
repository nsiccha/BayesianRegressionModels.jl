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

@stestset "each unit's η Hessian pattern covers its curvature" begin
    rk = RKBRMI(panel(PANEL); completions=:noncentered)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:region,
        nested_groups=PANEL_NESTED)
    v = m.view
    @test length(BRM._brm_nlme_eta_sparsity(m)) == length(v.levels)
    rng = Xoshiro(3)
    θ = 0.1 .* randn(rng, length(m.theta))
    σ = 0.1 .* randn(rng, length(m.sigma))
    H = [0.3 .* randn(rng, length(b)) for b in v.subject_coordinates]
    # The dense conditional blocks from one RK η-HVP per coordinate.
    blocks = BRM._brm_nlme_hessian_blocks(m, θ, σ, H)
    for (i, P) in enumerate(BRM._brm_nlme_eta_sparsity(m))
        n = length(v.subject_coordinates[i])
        @test size(P) == (n, n) && P == permutedims(P)
        @test all(I -> P[I] || I[1] == I[2], findall(x -> abs(x) > 1e-10, blocks[i]))
        # Units couple only through the effects above them: the pattern
        # leaves most pairs of a region's effects out.
        @test count(P) < n^2 / 2
    end
    # A flat model's subjects are small: it declares no pattern.
    flat = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    @test BRM._brm_nlme_eta_sparsity(flat) === nothing
end

# `k` copies of the panel's regions, each with its own districts,
# municipalities and district-years: region 1 keeps exactly its data.
function replicate_panel(d, k)
    shift(x, n) = reduce(vcat, [x .+ (j - 1) * n for j in 1:k])
    years = maximum(d.district_year)
    (; region=shift(d.region, maximum(d.region)),
       district=shift(d.district, maximum(d.district)),
       muni=shift(d.muni, maximum(d.muni)), sclass=repeat(d.sclass, k),
       district_year=shift(d.district_year, years), year=repeat(d.year, k),
       z=repeat(d.z, k), x1=repeat(d.x1, k), c0=d.c0,
       dy_obs=shift(d.dy_obs, years), total=repeat(d.total, k))
end

@stestset "one unit's work stays flat as regions are added" begin
    # A region's value and gradient are its own, however many regions there
    # are, and so is its work. The singular call takes the region's compact
    # cell samplers, and those cost its own rows: a totals cell observes its
    # one value per cell directly and RK computes the cell's reads at its rows
    # (BRM todo 0ww46pb, ReactiveKernels 0.1.6, Enzyme dogfood for the `hcat`
    # design read from views). Todo
    # BayesianRegressionModels:scaling-svi/todos/2026-10-25T22-21-35-231-182pl0l.
    measured = map((1, 3)) do k
        rk = RKBRMI(panel(replicate_panel(PANEL, k)); completions=:noncentered)
        m = brm_nlme_model(rk; ad_backend=NLME_AD, subject_group=:region,
            nested_groups=PANEL_NESTED)
        @test length(m.view.levels) == 3k
        θ = fill(0.1, length(m.theta))
        σ = fill(-0.5, length(m.sigma))
        η = 0.1 .* sin.(1:length(m.view.subject_coordinates[1]))
        gradient() = brm_nlme_loglikelihood_and_gradient(m, 1, θ, σ, η)
        gradient(); gradient()
        # Region 1's compact cell gradients, one cell group at a time.
        t = BRM._brm_svi_target(m)
        g = first(BRM._brm_nlme_prior_median_point(m))[t.globals]
        values = vcat(g, η)[t.order[1]]
        cell_gradient = zeros(length(values))
        cells = map(enumerate(m.cell_query)) do (k, cell)
            run() = for (j, p) in enumerate(cell.positions[1])
                BRM._brm_nlme_plan_cell!(cell_gradient, t.plan, m.unit, 1, k, j, p,
                    values)
            end
            run(); run()
            cell.names => minimum(@allocated(run()) for _ in 1:3)
        end
        # Each cell selects the globals and the draws it reads, not its
        # region's whole block, so a call's work does not grow with the region.
        for (k, cell) in enumerate(m.cell_query), j in eachindex(cell.positions[1])
            @test length(t.plan.selections[1][k][j]) == length(t.plan.buffers[k].active)
        end
        @test all(b -> length(b.active) < length(t.indices[1]), t.plan.buffers)
        gg, gz = zeros(length(t.globals)), zeros(length(η))
        (; result=gradient(), bytes=minimum(@allocated(gradient()) for _ in 1:3),
           cells, unit=(BRM._brm_svi_unit_term!(gg, gz, t, 1, g, η), copy(gg), copy(gz)))
    end
    @test measured[2].result[1] ≈ measured[1].result[1] rtol=1e-12
    @test measured[2].result[2] ≈ measured[1].result[2] rtol=1e-10
    @test measured[2].bytes <= 1.5measured[1].bytes
    # Every cell group, the totals included, costs the same at 3 and 9 regions.
    # With the totals' former emission (`ones(length(total[i])) .* s` cells,
    # observed through `flatten_cells`) the totals group grew 632,016 B ->
    # 1,624,032 B from 3 to 9 regions; now it is 141,888 B at both.
    @test first.(measured[2].cells) == first.(measured[1].cells) ==
        [(:total,), (:x1_obs,)]
    for ((_, b1), (_, b3)) in zip(measured[1].cells, measured[2].cells)
        @test b3 <= 1.1b1
    end
    @test measured[2].unit[1] ≈ measured[1].unit[1] rtol=1e-12
    @test measured[2].unit[3] ≈ measured[1].unit[3] rtol=1e-10
end
