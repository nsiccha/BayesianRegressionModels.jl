include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_connected_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_fixture.jl"))
using LogDensityProblems, Random
using ReactiveKernelsPPL: density_factors, select_factors, factor_coordinates

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

function separator_simplex_model(data)
    @brm data begin
        composition ~ Dirichlet([1.0, 2.0, 1.5])
        mu ~ 1 + (1 | effect | subject)
        extra ~ Normal(mu[1] + composition[1], 0.5)
        @plate for i in eachindex(y)
            y[i] ~ Normal(mu[i], 0.5)
        end
    end
end

@testset "explicit promotion closes a coupled prior and its transform" begin
    data = (; subject=["subject $i" for i in 1:4], y=[-0.2, 0.1, 0.3, 0.5], extra=0.4)
    m = brm_nlme_model(RKBRMI(separator_simplex_model(data));
        ad_backend=NLME_AD, unit_partition=:connected)
    catalogue = density_factors(m.backend.model, BRM.rk_bound_plan(m.backend))
    block = factor_coordinates(select_factors(catalogue; kind=:prior, site=:composition))
    @test length(block) == 2
    original = BRM._brm_svi_target(m; separator=:none)
    @test all(j -> original.ownership.owner[j] == 1, block)
    promoted = BRM._brm_svi_target(m; separator=:none, global_coordinates=[first(block)])
    @test all(in(promoted.globals), block)
    @test length(promoted.units[1]) == length(original.units[1]) - 2
    check_separator_density(original)
    check_separator_density(promoted)
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
