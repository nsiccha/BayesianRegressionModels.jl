include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_connected_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
using LogDensityProblems, Random

function check_separator_density(t)
    full = rk_logdensity_problem(t.m.backend; ad_backend=NLME_AD)
    for offset in (-0.31, 0.19)
        u = [offset + 0.12sin(j) for j in eachindex(t.m.unit)]
        g = u[t.globals]
        value, global_gradient = BRM._brm_svi_global_term(t, g)
        gradient = zeros(length(u))
        gradient[t.globals] .= global_gradient
        for (i, block) in enumerate(t.units)
            gg, gz = zeros(length(g)), zeros(length(block))
            value += BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[block])
            gradient[t.globals] .+= gg
            gradient[block] .+= gz
        end
        expected, native = LogDensityProblems.logdensity_and_gradient(full, u)
        @test value ≈ expected atol=2e-9 rtol=2e-10
        @test gradient ≈ native atol=2e-8 rtol=2e-9
        @test t.plan.point == t.m.unit
    end
end

@testset "automatic hub separators rebuild exact inference partitions" begin
    n = 9
    data = connected_data(n, fill(1, n-1), collect(2:n))
    m = brm_nlme_model(RKBRMI(connected_model(data)); ad_backend=NLME_AD,
        unit_partition=:connected)
    scientific = deepcopy(m.view)
    hub = findall(==(1), m.view.coordinate_subject)
    t = BRM._brm_svi_target(m; max_unit_coordinates=2, max_promoted_coordinates=1)
    @test length(m.view.levels) == 1
    @test length(t.units) == n
    @test maximum(length, t.units) == 1
    @test all(j -> j in t.globals, hub)
    @test t.ownership.separator.status === :satisfied
    @test m.view.subject_coordinates == scientific.subject_coordinates
    @test m.view.subject_levels == scientific.subject_levels
    @test all(selection -> selection.globals === t.globals, t.plan.indices)
    @test all(i -> all(j -> t.ownership.owner[j] in (0, i), t.plan.footprints[i]),
        eachindex(t.units))
    manual = BRM._brm_svi_target(m; separator=:none,
        global_coordinates=m.view.coordinates[hub])
    @test manual.units == t.units && manual.globals == t.globals
    none = BRM._brm_svi_target(m; separator=:none)
    @test length(none.units) == 1
    check_separator_density(t)
    check_separator_density(manual)
    options = (; max_unit_coordinates=2, max_promoted_coordinates=1,
        batch_size=4, max_iterations=9, ndraws=8, unit_dense_limit=0)
    full = brm_svi_fit(m; rng=Xoshiro(518), options..., threads=1)
    @test full.diagnostics.sampling.method === :work
    @test full.diagnostics.unit_levels == t.ownership.levels
    part = brm_svi_fit(m; rng=Xoshiro(518), options..., stop_after=3, threads=1)
    resumed = brm_svi_resume(m, part; ndraws=8, threads=1)
    @test svi_same_result(full, resumed)
    carry = brm_svi_checkpoint(full)
    @test carry.format == 3
    @test Set(keys(carry.units)) == Set(full.diagnostics.unit_levels)
    next = merge(data, (; y=data.y .+ 0.1))
    fit = brm_svi_fit(carry, connected_model(next); max_iterations=2, ndraws=0,
        batch_size=4, threads=1)
    @test fit.diagnostics.unit_levels == full.diagnostics.unit_levels
    @test fit.diagnostics.streaming.updates == 1
end

@testset "chain separators retain the native target" begin
    n = 17
    m = brm_nlme_model(RKBRMI(connected_model(connected_data(n,
        collect(1:n-1), collect(2:n)))); ad_backend=NLME_AD, unit_partition=:connected)
    t = BRM._brm_svi_target(m; max_unit_coordinates=4, max_promoted_coordinates=4)
    @test maximum(length, t.units) <= 4
    @test t.ownership.separator.status === :satisfied
    @test t.ownership.separator.trials <= 128
    check_separator_density(t)
end
