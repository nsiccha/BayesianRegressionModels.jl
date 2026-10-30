# test/nlme_shares_units.jl — the NLME reading (`brm_nlme_view`) of a
# completion whose missing entries have known group sums,
# `mi(x)::shares(total, rows)`, on a regional panel: regions (the units) ⊃
# districts ⊃ district-years ⊃ municipality-years, the suppressed cells of a
# district-year summing to its known residual, and a law that reads the
# regions' and districts' random effects.
#
# Non-centered (`RKBRMI(brmi; completions = :noncentered)`), the grouped
# entries' coordinates are standard-normal draws and each group observes the
# rest of their density, so every coordinate is a unit draw of the region
# owning its group's rows and the group's term belongs to that region. Centred,
# the joint latent's density reads the law, and with it the units' draws:
# no shared reading carries it (snag nlme-view-shares-f1cfaa71).
#
# Covered: the refusal of the centred reading; no record for absent free
# entries; the families, owners and attribution of the group terms; free
# entries as unit draws beside them; a group spanning two regions refused by
# name; lockstep and unit-problem densities against the full RK density; and
# per-unit Laplace and SVI on the panel.
#
# Run: julia --project=test test/nlme_shares_units.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra, Random
include(joinpath(@__DIR__, "nlme_unit_checks.jl"))

include(joinpath(@__DIR__, "nlme_shares_fixture.jl"))

const NESTED = SHARES_NESTED
shares_view(rk) = brm_nlme_view(rk; NESTED...)
message(f) = try f(); "" catch e; sprint(showerror, e) end

@stestset "a centred shares completion has no NLME reading" begin
    rk = RKBRMI(shares_model(shares_panel()))
    # No free entry: no record (before, the view failed with "declaration
    # `x.y_free` owns no packed coordinate").
    records = [r for r in BRM._rk_coordinate_records(rk.plan) if r.kind === :missing_value]
    @test [r.declaration for r in records] == [Symbol("x.y_shares")]
    # refused: shared coordinates would carry a density that reads the units'
    # draws, which the population prior evaluates at zero draws; the
    # non-centered lowering is the same model (snag nlme-view-shares-f1cfaa71).
    @test occursin("completions = :noncentered", message(() -> shares_view(rk)))
end

@stestset "each group's coordinates are unit draws of the region owning its rows" begin
    data = shares_panel()
    rk = RKBRMI(shares_model(data); completions=:noncentered)
    v = shares_view(rk)
    @test !any(==(:unclaimed), v.role)
    k = only(f for (f, family) in enumerate(v.families) if get(family.block, :shares, false))
    @test v.families[k].fixed == fill(1.0, 1, 1) && v.families[k].size == 1
    # Groups of K suppressed cells hold K - 1 coordinates, in group order.
    groups = [findall(i -> ismissing(data.x[i]) && data.dy[i] == c, eachindex(data.x))
        for c in data.cells]
    coordinates = [Symbol("x.z_shares.", j) for j in 1:sum(g -> length(g) - 1, groups)]
    @test all(c -> v.role[findfirst(==(c), v.coordinates)] === :subject_effect, coordinates)
    owner = reduce(vcat, (fill(data.region[first(g)], length(g) - 1) for g in groups))
    @test [v.subject[findfirst(==(c), v.coordinates)] for c in coordinates] == owner
    for i in eachindex(v.levels)
        @test count(==(k), v.subject_families[i]) == count(==(i), owner)
    end
    # Each group's observed term belongs to its region.
    m = brm_nlme_model(rk; ad_backend=NLME_AD, NESTED...)
    @test only(last(a) for a in m.attribution if first(a) === :x_shares_weights) ==
        [data.region[first(g)] for g in groups]
    check_lockstep(m, Xoshiro(5))
    check_unit_problems(m, Xoshiro(6))
end

@stestset "a cell reads the coordinates of one group at most" begin
    # Each grouped entry's value and each group's term are computed per entry
    # and per group (BRM 0.2.75), so a cell reads its own group's coordinates,
    # as recorded on the point. Before, both were whole-vector functions, which
    # every cell reading the completion or a group's term computed whole.
    data = shares_panel()
    rk = RKBRMI(shares_model(data); completions=:noncentered)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, NESTED...)
    names = BRM._rk_layout_coordinate_names(rk)
    groups = [findall(i -> ismissing(data.x[i]) && data.dy[i] == c, eachindex(data.x))
        for c in data.cells]
    owned, next = Vector{Int}[], 0
    for g in groups
        push!(owned, [findfirst(==(Symbol("x.z_shares.", next + j)), names)
            for j in 1:(length(g) - 1)])
        next += length(g) - 1
    end
    shares = reduce(vcat, owned)
    @test length(shares) == length(unique(shares)) > 1
    names_read, wide = Set{Symbol}(), Tuple[]
    for cell in m.cell_query, i in eachindex(cell.positions), p in cell.positions[i]
        point = BRM._BRMNLMERecordingPoint(m.unit, Int[])
        BRM._rk_nlme_cell_value(cell.query, point, p)
        read = unique(intersect(point.seen, shares))
        isempty(read) || push!(names_read, cell.names...)
        any(o -> issubset(read, o), owned) || isempty(read) || push!(wide, (cell.names, p))
    end
    # The completion's readers and the groups' terms both read coordinates.
    @test :x_shares_weights in names_read && length(names_read) >= 2
    @test isempty(wide)
end

@stestset "free entries are unit draws beside the groups" begin
    data = shares_panel(; free=true)
    rk = RKBRMI(shares_model(data); completions=:noncentered)
    v = shares_view(rk)
    free = findfirst(==(Symbol("x.z_free.1")), v.coordinates)
    @test v.role[free] === :subject_effect
    @test v.subject[free] == data.region[findfirst(==(2), data.dy)]
    @test count(f -> f.fixed !== nothing, v.families) == 2
    m = brm_nlme_model(rk; ad_backend=NLME_AD, NESTED...)
    check_lockstep(m, Xoshiro(7))
    check_unit_problems(m, Xoshiro(8))
end

@stestset "a group with cells in two regions is refused by name" begin
    data = shares_panel()
    # District-year 1's first suppressed cell moves to region 2 (its district
    # with it: here each region is one district).
    region = copy(data.region); region[findfirst(i -> ismissing(data.x[i]), eachindex(data.x))] = 2
    rk = RKBRMI(shares_model(merge(data, (; region, district=region)));
        completions=:noncentered)
    # refused: a group's shares couple its entries, so its regions are not
    # conditionally independent (snag nlme-view-shares-f1cfaa71).
    @test occursin("group 1 of `mi(x)::shares(…)`",
        message(() -> brm_nlme_view(rk; subject_group=:region, nested_groups=(district=:region,))))
end

@stestset "per-unit Laplace and SVI take the groups" begin
    rk = RKBRMI(shares_model(shares_panel()); completions=:noncentered)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, NESTED...)
    x = zeros(LogDensityProblems.dimension(p))
    value, gradient = LogDensityProblems.logdensity_and_gradient(p, x)
    @test isfinite(value) && all(isfinite, gradient)
    fit = brm_svi_fit(rk; rng=Xoshiro(9), ad_backend=NLME_AD, batch_size=2,
        max_iterations=200, ndraws=20, min_windows=1, NESTED...)
    @test fit.coordinates == p.model.view.coordinates
    @test size(fit.draws, 2) == 20 && all(isfinite, fit.draws)
end

println("nlme_shares_units.jl: all testsets passed")
